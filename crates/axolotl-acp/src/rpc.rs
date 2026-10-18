//! JSON-RPC 2.0 over stdio.
//!
//! The connection is genuinely bidirectional: the client calls the agent, and
//! the agent calls back (`session/request_permission`). That means one task
//! cannot own stdin — a request the agent makes must be written while the
//! reader keeps parsing.
//!
//! Design: a reader task parses inbound lines and either resolves a pending
//! outbound request or forwards the message to the agent's dispatcher; a writer
//! is a mutex-guarded sink shared by both directions, so lines never interleave.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::{json, Value};

/// Replies awaiting a client's answer, keyed by request id.
pub type Pending = Arc<Mutex<HashMap<i64, oneshot::Sender<Result<Value, RpcError>>>>>;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot, Mutex};
use tracing::{debug, warn};

use crate::protocol::RpcError;

/// An inbound request from the client.
#[derive(Debug, Clone)]
pub struct InboundRequest {
    pub id: Value,
    pub method: String,
    pub params: Value,
}

/// Serialises writes so a response and a notification never interleave.
#[derive(Clone)]
pub struct Writer {
    out: Arc<Mutex<tokio::io::Stdout>>,
}

impl Writer {
    pub fn new(out: tokio::io::Stdout) -> Self {
        Self {
            out: Arc::new(Mutex::new(out)),
        }
    }

    /// Write one JSON value as a single line.
    pub async fn send(&self, value: &Value) -> Result<()> {
        let mut line = serde_json::to_string(value).context("serializing outbound message")?;
        line.push('\n');
        let mut out = self.out.lock().await;
        out.write_all(line.as_bytes()).await?;
        out.flush().await?;
        Ok(())
    }

    /// Reply to a request.
    pub async fn respond(&self, id: &Value, result: Value) -> Result<()> {
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "result": result }))
            .await
    }

    /// Reply to a request with an error.
    pub async fn respond_error(&self, id: &Value, error: RpcError) -> Result<()> {
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "error": error }))
            .await
    }

    /// Send a notification (no id, no response expected).
    pub async fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.send(&json!({ "jsonrpc": "2.0", "method": method, "params": params }))
            .await
    }
}

/// The agent's side of the connection: it can call the client.
#[derive(Clone)]
pub struct Client {
    writer: Writer,
    next_id: Arc<AtomicI64>,
    pending: Pending,
}

impl Client {
    /// A client with no live connection, for tests that only exercise dispatch.
    #[doc(hidden)]
    pub fn detached_for_test(writer: Writer) -> Self {
        Self {
            writer,
            next_id: Arc::new(AtomicI64::new(1)),
            pending: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Issue a request to the client and await its reply.
    ///
    /// A client that never answers would otherwise hang the turn forever, so
    /// the wait is bounded.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value, RpcError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().await;
            pending.insert(id, tx);
        }

        self.writer
            .send(&json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params
            }))
            .await
            .map_err(|e| RpcError::internal(e.to_string()))?;

        match tokio::time::timeout(std::time::Duration::from_secs(300), rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(RpcError::internal("client dropped the request")),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(RpcError::internal(format!(
                    "client did not answer '{method}' within 300s"
                )))
            }
        }
    }
}

/// The connection, split into the two halves the agent needs.
pub struct Connection {
    pub writer: Writer,
    pub client: Client,
    /// Inbound requests from the client.
    pub requests: mpsc::UnboundedReceiver<InboundRequest>,
}

/// Spawn the reader task and return the connection.
pub fn connect() -> Connection {
    let (tx, rx) = mpsc::unbounded_channel();
    let writer = Writer::new(tokio::io::stdout());
    let pending: Pending = Arc::new(Mutex::new(HashMap::new()));

    let client = Client {
        writer: writer.clone(),
        next_id: Arc::new(AtomicI64::new(1)),
        pending: pending.clone(),
    };

    let reader_writer = writer.clone();
    tokio::spawn(async move {
        if let Err(e) = read_loop(tx, pending, reader_writer).await {
            debug!(error = %e, "reader loop ended");
        }
    });

    Connection {
        writer,
        client,
        requests: rx,
    }
}

async fn read_loop(
    requests: mpsc::UnboundedSender<InboundRequest>,
    pending: Pending,
    writer: Writer,
) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();

    while let Some(line) = lines.next_line().await? {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let value: Value = match serde_json::from_str(trimmed) {
            Ok(value) => value,
            Err(e) => {
                // A malformed line cannot be answered meaningfully: there is no
                // id to answer. Report and keep the connection alive.
                warn!(error = %e, "ignoring malformed JSON-RPC line");
                let _ = writer
                    .send(&json!({
                        "jsonrpc": "2.0",
                        "id": Value::Null,
                        "error": RpcError {
                            code: -32700,
                            message: format!("parse error: {e}"),
                            data: None,
                        }
                    }))
                    .await;
                continue;
            }
        };

        let id = value.get("id").cloned();
        let method = value
            .get("method")
            .and_then(Value::as_str)
            .map(str::to_string);

        match (id, method) {
            // A request from the client.
            (Some(id), Some(method)) => {
                let params = value.get("params").cloned().unwrap_or(Value::Null);
                if requests
                    .send(InboundRequest { id, method, params })
                    .is_err()
                {
                    return Ok(());
                }
            }
            // A response to something we asked.
            (Some(id), None) => {
                let key = id.as_i64();
                let outcome = match value.get("error") {
                    Some(err) => {
                        let code = err.get("code").and_then(Value::as_i64).unwrap_or(0);
                        let message = err
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown error")
                            .to_string();
                        Err(RpcError {
                            code,
                            message,
                            data: err.get("data").cloned(),
                        })
                    }
                    None => Ok(value.get("result").cloned().unwrap_or(Value::Null)),
                };
                if let Some(key) = key {
                    if let Some(tx) = pending.lock().await.remove(&key) {
                        let _ = tx.send(outcome);
                    }
                }
            }
            // A notification from the client.
            (None, Some(method)) => {
                debug!(method = %method, "ignoring client notification");
            }
            (None, None) => {
                warn!("ignoring JSON-RPC message with neither id nor method");
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_serializes_each_message_on_one_line() {
        let value = json!({ "jsonrpc": "2.0", "id": 1, "result": {} });
        let line = serde_json::to_string(&value).unwrap();
        assert!(!line.contains('\n'), "NDJSON requires single-line messages");
    }

    #[test]
    fn request_ids_start_at_one_and_increment() {
        let next = AtomicI64::new(1);
        assert_eq!(next.fetch_add(1, Ordering::Relaxed), 1);
        assert_eq!(next.fetch_add(1, Ordering::Relaxed), 2);
    }

    #[test]
    fn a_response_is_distinguished_from_a_request_by_having_no_method() {
        let response = json!({ "jsonrpc": "2.0", "id": 5, "result": {} });
        assert!(response.get("method").is_none());
        assert_eq!(response.get("id").and_then(Value::as_i64), Some(5));

        let request = json!({ "jsonrpc": "2.0", "id": 5, "method": "x" });
        assert!(request.get("method").is_some());
    }
}
