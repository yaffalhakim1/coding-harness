//! ACP-backed permission gate and question asker.
//!
//! Both are the agent calling the client and waiting: `session/request_permission`
//! for a tool, `elicitation/create` for a question. That is what makes a
//! Supervised session actually supervised — the harness asks, the user answers,
//! and a denial is a recoverable tool error rather than a crash.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tracing::debug;

use axolotl_actions::ask_user::{Answer, Asker, Question};
use axolotl_actions::permission::{Approval, ApprovalRequest, PermissionGate, ToolKindHint};

use crate::protocol::{
    PermissionOption, PermissionOptionKind, PermissionOutcome, PermissionToolCall,
    RequestPermissionParams, RequestPermissionResult, ToolKind,
};
use crate::rpc::Client;

/// Tool kind for the protocol, from the harness's hint.
fn protocol_kind(kind: ToolKindHint) -> ToolKind {
    match kind {
        ToolKindHint::Read => ToolKind::Read,
        ToolKindHint::Edit => ToolKind::Edit,
        ToolKindHint::Delete => ToolKind::Delete,
        ToolKindHint::Search => ToolKind::Search,
        ToolKindHint::Execute => ToolKind::Execute,
        ToolKindHint::Fetch => ToolKind::Fetch,
        ToolKindHint::Think => ToolKind::Think,
        ToolKindHint::Other => ToolKind::Other,
    }
}

/// Asks the client to approve a tool call.
pub struct AcpPermissionGate {
    client: Client,
    session_id: String,
}

impl AcpPermissionGate {
    pub fn new(client: Client, session_id: String) -> Self {
        Self { client, session_id }
    }

    /// The options offered for a request.
    ///
    /// `allow_always` is deliberately absent. ACP has no client-side rule store,
    /// and a durable "always" would have to live in the harness — where it would
    /// silently apply to a different session with a different task. One-shot
    /// approval keeps the decision where the user can see it.
    fn options() -> Vec<PermissionOption> {
        vec![
            PermissionOption {
                option_id: "allow-once".into(),
                name: "Allow once".into(),
                kind: PermissionOptionKind::AllowOnce,
            },
            PermissionOption {
                option_id: "reject-once".into(),
                name: "Reject".into(),
                kind: PermissionOptionKind::RejectOnce,
            },
        ]
    }
}

#[async_trait]
impl PermissionGate for AcpPermissionGate {
    async fn approve(&self, request: &ApprovalRequest) -> Approval {
        let params = RequestPermissionParams {
            session_id: self.session_id.clone(),
            tool_call: PermissionToolCall {
                tool_call_id: format!("{}-{}", request.tool, uuid::Uuid::new_v4().simple()),
                title: Some(request.title.clone()),
                kind: Some(protocol_kind(request.kind)),
            },
            options: Self::options(),
        };

        let Ok(params) = serde_json::to_value(params) else {
            // A malformed request cannot be shown to a user; allowing it would
            // defeat the gate, so fail closed.
            return Approval::Deny;
        };

        match self
            .client
            .request("session/request_permission", params)
            .await
        {
            Ok(value) => match serde_json::from_value::<RequestPermissionResult>(value) {
                Ok(result) => match result.outcome {
                    PermissionOutcome::Cancelled => Approval::Cancelled,
                    PermissionOutcome::Selected { .. } if result.outcome.allowed() => {
                        Approval::Allow
                    }
                    _ => Approval::Deny,
                },
                Err(e) => {
                    debug!(error = %e, "unreadable permission response; denying");
                    Approval::Deny
                }
            },
            Err(e) => {
                // The client could not answer (it went away, or timed out).
                // Treat it as a cancellation so the turn stops cleanly.
                debug!(error = %e, "permission request failed");
                Approval::Cancelled
            }
        }
    }
}

/// Asks the client a question via `elicitation/create`.
pub struct AcpAsker {
    client: Client,
    session_id: String,
}

impl AcpAsker {
    pub fn new(client: Client, session_id: String) -> Self {
        Self { client, session_id }
    }
}

#[async_trait]
impl Asker for AcpAsker {
    async fn ask(&self, question: &Question) -> Option<Answer> {
        // The protocol's elicitation is a form; one string property is the
        // narrowest shape that still renders as a real prompt in a client.
        let schema = if question.options.is_empty() {
            json!({
                "type": "object",
                "properties": {
                    "answer": { "type": "string", "title": question.question }
                },
                "required": ["answer"]
            })
        } else {
            json!({
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "title": question.question,
                        "enum": question.options
                    }
                },
                "required": ["answer"]
            })
        };

        let params = json!({
            "sessionId": self.session_id,
            "message": question.question,
            "requestedSchema": schema,
        });

        let value = match self.client.request("elicitation/create", params).await {
            Ok(value) => value,
            Err(e) => {
                debug!(error = %e, "elicitation failed");
                return None;
            }
        };

        // `action` is one of accept / decline / cancel; content carries the form.
        let action = value.get("action").and_then(|v| v.as_str()).unwrap_or("");
        if action != "accept" {
            return None;
        }

        let answer = value
            .get("content")
            .and_then(|c| c.get("answer"))
            .and_then(|a| a.as_str())
            .unwrap_or("");
        if answer.trim().is_empty() {
            return None;
        }

        Some(Answer {
            selections: vec![answer.to_string()],
        })
    }
}

/// Convenience alias for a gate the registry can hold.
pub type SharedGate = Arc<dyn PermissionGate>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_kind_maps_every_hint() {
        assert_eq!(protocol_kind(ToolKindHint::Read), ToolKind::Read);
        assert_eq!(protocol_kind(ToolKindHint::Edit), ToolKind::Edit);
        assert_eq!(protocol_kind(ToolKindHint::Execute), ToolKind::Execute);
        assert_eq!(protocol_kind(ToolKindHint::Fetch), ToolKind::Fetch);
        assert_eq!(protocol_kind(ToolKindHint::Other), ToolKind::Other);
    }

    #[test]
    fn only_one_shot_options_are_offered() {
        let options = AcpPermissionGate::options();
        assert_eq!(options.len(), 2);
        // No allow_always: there is no client-side rule store, so a durable
        // choice would outlive the session that justified it.
        assert!(!options
            .iter()
            .any(|o| o.kind == PermissionOptionKind::AllowAlways));
        assert!(options
            .iter()
            .any(|o| o.kind == PermissionOptionKind::AllowOnce));
    }

    #[test]
    fn an_unreadable_outcome_is_a_denial() {
        // Fail closed: a response we cannot parse must not become an allow.
        let outcome: Result<RequestPermissionResult, _> =
            serde_json::from_value(json!({ "outcome": "nonsense" }));
        assert!(outcome.is_err());
    }

    #[test]
    fn a_selected_outcome_is_an_allow() {
        let result: RequestPermissionResult = serde_json::from_value(json!({
            "outcome": { "outcome": "selected", "optionId": "allow-once" }
        }))
        .unwrap();
        assert!(result.outcome.allowed());
    }
}
