//! Minimal MCP (Model Context Protocol) client.
//!
//! jcode and OpenCode both speak MCP, and it is the cheapest way to give the
//! harness real ecosystem reach: any MCP server's tools become `Action`s in the
//! registry, so the agent loop does not change at all.
//!
//! This implements the subset that matters for tool use:
//! `initialize` → `tools/list` → `tools/call`, over stdio or HTTP.
//!
//! Deliberately not implemented: resources, prompts, sampling, notifications.
//! They are not needed to expose tools, and each adds surface area.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;
use tracing::{debug, warn};

/// Protocol version this client speaks.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// How long a single MCP request may take.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// How a server is reached.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Transport {
    /// Spawn a child process and talk JSON-RPC over its stdio.
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: HashMap<String, String>,
    },
    /// POST JSON-RPC to an HTTP endpoint.
    Http { url: String },
}

/// One MCP server from the config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub name: String,
    #[serde(flatten)]
    pub transport: Transport,
}

/// A tool advertised by an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// A live connection to one MCP server.
pub struct McpClient {
    name: String,
    transport: Transport,
    /// stdio: the child process and its pipes.
    child: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
    http: Option<reqwest::Client>,
    /// Monotonic request ids.
    next_id: Arc<Mutex<u64>>,
    /// Serialises stdio round-trips: one request/response at a time, because
    /// the stream carries no request framing beyond the id.
    lock: Arc<Mutex<()>>,
    /// Tools discovered during `initialize`.
    tools: Vec<McpTool>,
}

impl McpClient {
    /// A client with no transport, for tests that only inspect tool
    /// definitions. Any call through it fails cleanly.
    #[doc(hidden)]
    pub fn disconnected_for_test() -> Self {
        Self {
            name: "test".into(),
            transport: Transport::Http {
                url: "http://127.0.0.1:0/mcp".into(),
            },
            child: None,
            stdin: None,
            stdout: None,
            http: None,
            next_id: Arc::new(Mutex::new(0)),
            lock: Arc::new(Mutex::new(())),
            tools: Vec::new(),
        }
    }

    /// Connect and perform the MCP handshake, then list tools.
    pub async fn connect(config: &ServerConfig) -> Result<Self> {
        let mut client = Self {
            name: config.name.clone(),
            transport: config.transport.clone(),
            child: None,
            stdin: None,
            stdout: None,
            http: None,
            next_id: Arc::new(Mutex::new(0)),
            lock: Arc::new(Mutex::new(())),
            tools: Vec::new(),
        };

        match &config.transport {
            Transport::Stdio { command, args, env } => {
                client.spawn_stdio(command, args, env).await?;
            }
            Transport::Http { url } => {
                let http = reqwest::Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .build()
                    .context("building MCP HTTP client")?;
                client.http = Some(http);
                let _ = url;
            }
        }

        client.initialize().await?;
        client.tools = client.list_tools().await?;
        Ok(client)
    }

    /// Server name from the config.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Tools discovered at connect time.
    pub fn tools(&self) -> &[McpTool] {
        &self.tools
    }

    async fn spawn_stdio(
        &mut self,
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<()> {
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args)
            .envs(env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // Server logs must not corrupt the JSON-RPC stream.
            .stderr(Stdio::null());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("spawning MCP server '{command}'"))?;

        self.stdin = child.stdin.take();
        self.stdout = child.stdout.take().map(BufReader::new);
        self.child = Some(child);
        Ok(())
    }

    /// `initialize` handshake.
    async fn initialize(&mut self) -> Result<()> {
        let params = serde_json::json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": { "name": "axolotl", "version": env!("CARGO_PKG_VERSION") }
        });
        self.request("initialize", params).await?;

        // The spec requires a notification after initialize; servers may ignore
        // it, but skipping it breaks strict ones.
        self.notify("notifications/initialized").await?;
        Ok(())
    }

    /// `tools/list`, following pagination if the server provides a cursor.
    pub async fn list_tools(&mut self) -> Result<Vec<McpTool>> {
        let mut all = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let params = match &cursor {
                Some(c) => serde_json::json!({ "cursor": c }),
                None => serde_json::json!({}),
            };
            let result = self.request("tools/list", params).await?;

            if let Some(tools) = result.get("tools").and_then(|t| t.as_array()) {
                for tool in tools {
                    match serde_json::from_value::<McpTool>(tool.clone()) {
                        Ok(t) => all.push(t),
                        Err(e) => warn!(server = %self.name, error = %e, "Skipping malformed tool"),
                    }
                }
            }

            match result.get("nextCursor").and_then(|c| c.as_str()) {
                Some(next) if !next.is_empty() => cursor = Some(next.to_string()),
                _ => break,
            }
        }

        debug!(server = %self.name, count = all.len(), "Discovered MCP tools");
        Ok(all)
    }

    /// `tools/call`, returning the concatenated text content.
    pub async fn call_tool(&mut self, tool: &str, arguments: serde_json::Value) -> Result<String> {
        let result = self
            .request(
                "tools/call",
                serde_json::json!({ "name": tool, "arguments": arguments }),
            )
            .await?;

        // A tool-level failure is reported inside the result, not as a
        // JSON-RPC error, so surface it as an error here.
        if result.get("isError").and_then(|v| v.as_bool()) == Some(true) {
            anyhow::bail!("MCP tool '{tool}' failed: {}", render_content(&result));
        }
        Ok(render_content(&result))
    }

    /// Send a request and await its response.
    async fn request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let id = {
            let mut next = self.next_id.lock().await;
            *next += 1;
            *next
        };
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        });

        // Clone the endpoint out: `request_http` borrows `self` mutably, which
        // conflicts with borrowing `self.transport` for the match.
        match self.transport.clone() {
            Transport::Stdio { .. } => self.request_stdio(&payload, id).await,
            Transport::Http { url } => self.request_http(&url, &payload).await,
        }
    }

    /// Fire-and-forget notification (no id, no response).
    async fn notify(&mut self, method: &str) -> Result<()> {
        let payload = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": {}
        });
        if let Transport::Stdio { .. } = &self.transport {
            let _guard = self.lock.lock().await;
            if let Some(stdin) = self.stdin.as_mut() {
                let line = format!("{payload}\n");
                stdin.write_all(line.as_bytes()).await?;
                stdin.flush().await?;
            }
        }
        Ok(())
    }

    async fn request_stdio(
        &mut self,
        payload: &serde_json::Value,
        id: u64,
    ) -> Result<serde_json::Value> {
        let _guard = self.lock.lock().await;

        let stdin = self.stdin.as_mut().context("MCP server stdin closed")?;
        let line = format!("{payload}\n");
        stdin.write_all(line.as_bytes()).await?;
        stdin.flush().await?;

        let stdout = self.stdout.as_mut().context("MCP server stdout closed")?;
        let mut buf = String::new();

        let read = async {
            loop {
                buf.clear();
                let n = stdout.read_line(&mut buf).await?;
                if n == 0 {
                    anyhow::bail!("MCP server closed the stream");
                }
                let trimmed = buf.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
                    // Servers sometimes log to stdout despite the spec; skip.
                    debug!(server = %self.name, line = %trimmed, "Skipping non-JSON line");
                    continue;
                };
                // Skip notifications and responses to other ids.
                if value.get("id").and_then(|v| v.as_u64()) != Some(id) {
                    continue;
                }
                return Ok::<serde_json::Value, anyhow::Error>(value);
            }
        };

        let response = tokio::time::timeout(REQUEST_TIMEOUT, read)
            .await
            .with_context(|| format!("MCP request to '{}' timed out", self.name))??;

        unwrap_result(response)
    }

    async fn request_http(
        &mut self,
        url: &str,
        payload: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        let http = self.http.as_ref().context("MCP HTTP client missing")?;
        let response = http
            .post(url)
            .json(payload)
            .send()
            .await
            .with_context(|| format!("MCP HTTP request to '{url}'"))?;

        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            anyhow::bail!("MCP HTTP {status}: {body}");
        }
        let value: serde_json::Value = serde_json::from_str(&body)
            .with_context(|| format!("parsing MCP response: {}", &body[..body.len().min(200)]))?;
        unwrap_result(value)
    }

    /// Terminate the child process, if any.
    pub async fn shutdown(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

impl Drop for McpClient {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            let _ = child.start_kill();
        }
    }
}

/// Pull `result` out of a JSON-RPC response, mapping errors.
fn unwrap_result(response: serde_json::Value) -> Result<serde_json::Value> {
    if let Some(error) = response.get("error") {
        let code = error.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
        let message = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        anyhow::bail!("MCP error {code}: {message}");
    }
    Ok(response
        .get("result")
        .cloned()
        .unwrap_or(serde_json::Value::Null))
}

/// Flatten MCP content blocks into text.
fn render_content(result: &serde_json::Value) -> String {
    let Some(content) = result.get("content").and_then(|c| c.as_array()) else {
        return serde_json::to_string(result).unwrap_or_default();
    };

    let mut out = String::new();
    for block in content {
        match block.get("type").and_then(|t| t.as_str()) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    out.push_str(text);
                    out.push('\n');
                }
            }
            Some(other) => {
                out.push_str(&format!("[{other} content omitted]\n"));
            }
            None => {}
        }
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests;
