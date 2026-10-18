//! The ACP agent: dispatch inbound JSON-RPC methods onto the harness.
//!
//! Implements the baseline an agent must support — `initialize`,
//! `session/new`, `session/prompt`, `session/cancel` — plus `session/load`,
//! `session/resume`, `session/close` and `session/list`, which are advertised
//! in `initialize` and therefore must actually work.

use std::sync::Arc;

use anyhow::Result;
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};

use crate::protocol::{
    AgentCapabilities, AgentInfo, CancelParams, InitializeParams, InitializeResult,
    LoadSessionParams, NewSessionParams, NewSessionResult, PromptCapabilities, PromptParams,
    RpcError, SessionCapabilitiesOut, PROTOCOL_VERSION,
};
use crate::rpc::{Connection, InboundRequest};
use crate::session::{self, Sessions};
use crate::sink::AcpSink;

/// Run the agent until the client closes the stream.
pub async fn run(mut connection: Connection) -> Result<()> {
    let sessions = Sessions::new();

    while let Some(request) = connection.requests.recv().await {
        let writer = connection.writer.clone();
        let client = connection.client.clone();
        let sessions = sessions.clone();

        // Each request is handled on its own task so a long prompt does not
        // block `session/cancel` from being read.
        tokio::spawn(async move {
            let id = request.id.clone();
            let method = request.method.clone();
            match dispatch(&method, request, sessions, writer.clone(), client).await {
                Ok(result) => {
                    if let Err(e) = writer.respond(&id, result).await {
                        debug!(error = %e, method = %method, "could not write response");
                    }
                }
                Err(HandlerError::Rpc(error)) => {
                    if let Err(e) = writer.respond_error(&id, error).await {
                        debug!(error = %e, method = %method, "could not write error response");
                    }
                }
                Err(HandlerError::Fatal(e)) => {
                    error!(error = %e, method = %method, "handler failed");
                    let _ = writer
                        .respond_error(&id, RpcError::internal(e.to_string()))
                        .await;
                }
            }
        });
    }

    info!("client closed the connection");
    Ok(())
}

/// Why a handler failed.
enum HandlerError {
    /// Answer with a JSON-RPC error; the connection stays usable.
    Rpc(RpcError),
    /// Unexpected failure, reported as an internal error.
    Fatal(anyhow::Error),
}

impl From<anyhow::Error> for HandlerError {
    fn from(e: anyhow::Error) -> Self {
        HandlerError::Fatal(e)
    }
}

impl From<RpcError> for HandlerError {
    fn from(e: RpcError) -> Self {
        HandlerError::Rpc(e)
    }
}

/// Parse params, mapping a shape error onto an invalid-params reply.
fn parse<T: serde::de::DeserializeOwned>(params: Value) -> Result<T, HandlerError> {
    serde_json::from_value(params)
        .map_err(|e| HandlerError::Rpc(RpcError::invalid_params(e.to_string())))
}

async fn dispatch(
    method: &str,
    request: InboundRequest,
    sessions: Arc<Sessions>,
    writer: crate::rpc::Writer,
    _client: crate::rpc::Client,
) -> Result<Value, HandlerError> {
    match method {
        "initialize" => {
            let params: InitializeParams = parse(request.params)?;
            if params.protocol_version != PROTOCOL_VERSION {
                warn!(
                    client = params.protocol_version,
                    ours = PROTOCOL_VERSION,
                    "client requested a different protocol version"
                );
            }
            Ok(serde_json::to_value(initialize_result()).unwrap())
        }

        "session/new" => {
            let params: NewSessionParams = parse(request.params)?;
            // A bad cwd is the caller's error, not ours: report it as invalid
            // params so a client can tell it apart from a server fault.
            let session = session::create(params.cwd.clone())
                .await
                .map_err(|e| HandlerError::Rpc(RpcError::invalid_params(e.to_string())))?;
            let id = session.id.clone();
            let cwd = session.cwd.clone();
            sessions.insert(session).await;
            info!(session = %id, "session/new");

            // Advertise slash commands after the session exists: the catalogue
            // depends on the workspace's skills, which the cwd determines.
            crate::notify::emit_commands(&writer, &id, &cwd).await;

            Ok(serde_json::to_value(NewSessionResult { session_id: id }).unwrap())
        }

        "session/load" => {
            let params: LoadSessionParams = parse(request.params)?;
            match sessions.get(&params.session_id).await {
                Some(session) => {
                    // Replay the conversation before answering, as the spec
                    // requires for load.
                    crate::notify::replay_history(&writer, &session.id, &session.history).await;
                    Ok(json!({}))
                }
                None => Err(HandlerError::Rpc(RpcError::invalid_params(format!(
                    "unknown session '{}'",
                    params.session_id
                )))),
            }
        }

        "session/resume" => {
            let params: LoadSessionParams = parse(request.params)?;
            match sessions.get(&params.session_id).await {
                Some(_) => Ok(json!({})),
                None => Err(HandlerError::Rpc(RpcError::invalid_params(format!(
                    "unknown session '{}'",
                    params.session_id
                )))),
            }
        }

        "session/close" => {
            let params: crate::protocol::CloseSessionParams = parse(request.params)?;
            if sessions.remove(&params.session_id).await.is_some() {
                Ok(json!({}))
            } else {
                Err(HandlerError::Rpc(RpcError::invalid_params(format!(
                    "unknown session '{}'",
                    params.session_id
                ))))
            }
        }

        "session/prompt" => {
            let params: PromptParams = parse(request.params)?;
            let Some(session) = sessions.get(&params.session_id).await else {
                return Err(HandlerError::Rpc(RpcError::invalid_params(format!(
                    "unknown session '{}'",
                    params.session_id
                ))));
            };
            let session_id = session.id.clone();
            let window = session.config.context_window;

            let sink = Arc::new(AcpSink::new(writer.clone(), session_id.clone(), window));

            let result = session::run_prompt(session, params, sink, writer, _client).await?;
            Ok(serde_json::to_value(result).unwrap())
        }

        "session/cancel" => {
            let params: CancelParams = parse(request.params)?;
            if let Some(session) = sessions.get(&params.session_id).await {
                session
                    .cancelled
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                info!(session = %params.session_id, "session/cancel");
            }
            // A notification never has an id, but tolerate a client that sends
            // one as a request: answer with an empty result rather than an error.
            Ok(json!({}))
        }

        "session/list" => {
            let all = sessions.all().await;
            let entries: Vec<Value> = all
                .iter()
                .map(|s| {
                    json!({
                        "sessionId": s.id,
                        "cwd": s.cwd.to_string_lossy(),
                        "title": s.title,
                    })
                })
                .collect();
            Ok(json!({ "sessions": entries }))
        }

        other => Err(HandlerError::Rpc(RpcError::method_not_found(other))),
    }
}

/// The `initialize` result.
fn initialize_result() -> InitializeResult {
    InitializeResult {
        protocol_version: PROTOCOL_VERSION,
        agent_capabilities: AgentCapabilities {
            load_session: true,
            prompt_capabilities: PromptCapabilities {
                // The harness drives a text-only model interface. Advertising
                // image support it cannot honour would have the client send
                // screenshots that silently degrade.
                image: false,
                audio: false,
                embedded_context: true,
            },
            session_capabilities: SessionCapabilitiesOut {
                resume: json!({}),
                close: json!({}),
            },
        },
        agent_info: AgentInfo {
            name: "axolotl-acp".into(),
            title: "Axolotl".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        auth_methods: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_advertises_the_capabilities_that_are_implemented() {
        let result = initialize_result();
        assert_eq!(result.protocol_version, PROTOCOL_VERSION);
        // Every advertised capability has a handler above; advertising one that
        // is not implemented is how a client offers a control that does nothing.
        assert!(result.agent_capabilities.load_session);
        assert!(result
            .agent_capabilities
            .session_capabilities
            .resume
            .is_object());
        assert!(result
            .agent_capabilities
            .session_capabilities
            .close
            .is_object());
    }

    #[test]
    fn text_only_is_declared() {
        let result = initialize_result();
        assert!(!result.agent_capabilities.prompt_capabilities.image);
        assert!(!result.agent_capabilities.prompt_capabilities.audio);
        assert!(
            result
                .agent_capabilities
                .prompt_capabilities
                .embedded_context
        );
    }

    #[tokio::test]
    async fn an_unknown_method_is_method_not_found() {
        let sessions = Sessions::new();
        let writer = crate::rpc::Writer::new(tokio::io::stdout());
        let client = crate::rpc::Client::detached_for_test(writer.clone());
        let err = dispatch(
            "does/not/exist",
            InboundRequest {
                id: json!(1),
                method: "does/not/exist".into(),
                params: Value::Null,
            },
            sessions,
            writer,
            client,
        )
        .await
        .unwrap_err();
        match err {
            HandlerError::Rpc(e) => assert_eq!(e.code, -32601),
            HandlerError::Fatal(_) => panic!("expected a JSON-RPC error"),
        }
    }

    #[tokio::test]
    async fn a_malformed_prompt_is_invalid_params() {
        let sessions = Sessions::new();
        let writer = crate::rpc::Writer::new(tokio::io::stdout());
        let client = crate::rpc::Client::detached_for_test(writer.clone());
        let err = dispatch(
            "session/prompt",
            InboundRequest {
                id: json!(2),
                method: "session/prompt".into(),
                params: json!({ "wrong": "shape" }),
            },
            sessions,
            writer,
            client,
        )
        .await
        .unwrap_err();
        match err {
            HandlerError::Rpc(e) => assert_eq!(e.code, -32602),
            HandlerError::Fatal(_) => panic!("expected invalid params"),
        }
    }

    #[tokio::test]
    async fn prompt_for_an_unknown_session_is_rejected() {
        let sessions = Sessions::new();
        let writer = crate::rpc::Writer::new(tokio::io::stdout());
        let client = crate::rpc::Client::detached_for_test(writer.clone());
        let err = dispatch(
            "session/prompt",
            InboundRequest {
                id: json!(3),
                method: "session/prompt".into(),
                params: json!({
                    "sessionId": "sess_missing",
                    "prompt": [{ "type": "text", "text": "hi" }]
                }),
            },
            sessions,
            writer,
            client,
        )
        .await
        .unwrap_err();
        match err {
            HandlerError::Rpc(e) => assert!(e.message.contains("unknown session"), "{}", e.message),
            HandlerError::Fatal(_) => panic!("expected invalid params"),
        }
    }

    #[tokio::test]
    async fn cancel_for_an_unknown_session_is_accepted() {
        // A cancel racing a close must not be an error: the client has no way to
        // know the session went away first.
        let sessions = Sessions::new();
        let writer = crate::rpc::Writer::new(tokio::io::stdout());
        let client = crate::rpc::Client::detached_for_test(writer.clone());
        let result = dispatch(
            "session/cancel",
            InboundRequest {
                id: json!(4),
                method: "session/cancel".into(),
                params: json!({ "sessionId": "sess_gone" }),
            },
            sessions,
            writer,
            client,
        )
        .await;
        assert!(result.is_ok());
    }
}
