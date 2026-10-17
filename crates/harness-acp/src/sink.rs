//! Bridge from harness progress events to ACP `session/update` notifications.
//!
//! The harness loop already reports what it is doing through
//! [`Progress`](harness_core::progress::Progress); this maps that vocabulary
//! onto the protocol's. Keeping the translation here means the loop stays
//! protocol-agnostic and a second front end needs no changes.

use std::sync::Arc;

use harness_core::progress::{Progress, ProgressSink};
use tracing::debug;

use crate::protocol::{
    OutContent, SessionUpdate, SessionUpdateParams, ToolCallContent, ToolKind, ToolStatus,
};
use crate::rpc::Writer;

/// Emits ACP updates for one session.
pub struct AcpSink {
    writer: Writer,
    session_id: String,
    /// Context window, for `usage_update`.
    context_window: usize,
    /// Whether the client advertised notice support.
    supports_notices: bool,
}

impl AcpSink {
    pub fn new(writer: Writer, session_id: String, context_window: usize) -> Self {
        Self {
            writer,
            session_id,
            context_window,
            supports_notices: false,
        }
    }

    pub fn with_notices(mut self, supported: bool) -> Self {
        self.supports_notices = supported;
        self
    }

    /// Send one update, logging rather than propagating a write failure.
    ///
    /// A broken pipe means the client went away; the turn is over either way and
    /// surfacing that as a tool error would be misleading.
    fn emit(&self, update: SessionUpdate) {
        let params = SessionUpdateParams {
            session_id: self.session_id.clone(),
            update,
        };
        let writer = self.writer.clone();
        tokio::spawn(async move {
            if let Ok(value) = serde_json::to_value(params) {
                if let Err(e) = writer.notify("session/update", value).await {
                    debug!(error = %e, "dropping session/update after write failure");
                }
            }
        });
    }

    /// Announce a tool call starting.
    fn tool_started(&self, name: &str, arguments: &str, id: &str) {
        let raw = serde_json::from_str(arguments).ok();
        let title = describe_tool(name, raw.as_ref());
        let locations = locations_for(raw.as_ref());
        self.emit(SessionUpdate::ToolCall {
            tool_call_id: id.to_string(),
            name: Some(name.to_string()),
            title,
            kind: kind_for_tool(name),
            status: ToolStatus::Pending,
            content: None,
            raw_input: raw,
            locations,
        });
    }
}

impl ProgressSink for AcpSink {
    fn emit(&self, event: Progress) {
        match event {
            Progress::SessionStarted { .. } => {}
            Progress::TurnCompleted {
                text,
                usage,
                tool_calls,
                ..
            } => {
                if !text.trim().is_empty() {
                    self.emit(SessionUpdate::AgentMessageChunk {
                        message_id: None,
                        content: OutContent::text(text),
                    });
                }
                if usage.total_tokens > 0 && tool_calls == 0 {
                    self.emit(SessionUpdate::UsageUpdate {
                        used: usage.prompt_tokens,
                        size: self.context_window,
                        cost: None,
                    });
                }
            }
            Progress::ToolStarted { name, arguments } => {
                // The loop does not carry a tool-call id here, so the name is
                // used as the correlation key; ACP only requires uniqueness
                // within the session and the loop runs tools in order.
                self.tool_started(&name, &arguments, &name);
            }
            Progress::ToolFinished {
                name,
                success,
                duration_ms,
                preview,
            } => {
                self.emit(SessionUpdate::ToolCallUpdate {
                    tool_call_id: name,
                    status: Some(if success {
                        ToolStatus::Completed
                    } else {
                        ToolStatus::Failed
                    }),
                    content: Some(vec![ToolCallContent::text(format!(
                        "{preview} ({duration_ms}ms)"
                    ))]),
                    raw_output: Some(serde_json::json!({
                        "success": success,
                        "durationMs": duration_ms
                    })),
                });
            }
            Progress::ContextManaged {
                tokens_before,
                tokens_after,
            } => {
                // Compaction is a notice unless the client renders compaction
                // entities; the harness reports it as a notice to keep the
                // transcript honest about what happened.
                if self.supports_notices {
                    self.emit_notice(
                        "Context compacted",
                        format!(
                            "{tokens_before} → {tokens_after} tokens ({} freed)",
                            tokens_before.saturating_sub(tokens_after)
                        ),
                    );
                }
            }
            Progress::SessionFinished { .. } => {}
        }
    }
}

impl AcpSink {
    /// Emit a `notice` update, only when the client advertised support.
    fn emit_notice(&self, title: &str, description: String) {
        let params = serde_json::json!({
            "sessionId": self.session_id,
            "update": {
                "sessionUpdate": "notice",
                "severity": "info",
                "title": title,
                "description": description,
            }
        });
        let writer = self.writer.clone();
        tokio::spawn(async move {
            let _ = writer.notify("session/update", params).await;
        });
    }
}

/// Map a harness tool name onto an ACP tool kind.
///
/// Public so the dispatcher can build permission prompts without duplicating
/// the mapping.
pub fn kind_for_tool(name: &str) -> ToolKind {
    match name {
        "read_file" | "list_files" | "glob_files" => ToolKind::Read,
        "grep_text" => ToolKind::Search,
        "write_file" | "edit_file" => ToolKind::Edit,
        "bash" => ToolKind::Execute,
        "web_fetch" => ToolKind::Fetch,
        "update_plan" | "task" => ToolKind::Think,
        _ => ToolKind::Other,
    }
}

/// Human-readable title for a tool call.
fn describe_tool(name: &str, raw: Option<&serde_json::Value>) -> String {
    let arg = |key: &str| -> Option<String> {
        raw.and_then(|v| v.get(key))
            .and_then(|v| v.as_str())
            .map(str::to_string)
    };

    match name {
        "read_file" => {
            arg("path").map_or_else(|| "Reading a file".into(), |p| format!("Reading {p}"))
        }
        "write_file" => {
            arg("path").map_or_else(|| "Writing a file".into(), |p| format!("Writing {p}"))
        }
        "edit_file" => {
            arg("path").map_or_else(|| "Editing a file".into(), |p| format!("Editing {p}"))
        }
        "list_files" => "Listing files".into(),
        "glob_files" => {
            arg("pattern").map_or_else(|| "Finding files".into(), |p| format!("Finding {p}"))
        }
        "grep_text" => arg("query").map_or_else(
            || "Searching contents".into(),
            |q| format!("Searching for {q}"),
        ),
        "bash" => arg("command").map_or_else(
            || "Running a command".into(),
            |c| {
                let short: String = c.chars().take(60).collect();
                format!("Running {short}")
            },
        ),
        "web_fetch" => {
            arg("url").map_or_else(|| "Fetching a URL".into(), |u| format!("Fetching {u}"))
        }
        "update_plan" => "Updating the plan".into(),
        "recall_event" => "Recalling elided output".into(),
        "task" => "Delegating to a subagent".into(),
        other => other.replace('_', " "),
    }
}

/// File locations a tool call touches, for the client's follow-along view.
fn locations_for(
    raw: Option<&serde_json::Value>,
) -> Option<Vec<crate::protocol::ToolCallLocation>> {
    let path = raw?.get("path")?.as_str()?;
    Some(vec![crate::protocol::ToolCallLocation {
        path: path.to_string(),
        line: None,
    }])
}

/// Convenience alias so the sink can be handed to the loop as a trait object.
pub type SharedSink = Arc<AcpSink>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_kinds_match_the_semantics() {
        assert_eq!(kind_for_tool("read_file"), ToolKind::Read);
        assert_eq!(kind_for_tool("grep_text"), ToolKind::Search);
        assert_eq!(kind_for_tool("edit_file"), ToolKind::Edit);
        assert_eq!(kind_for_tool("bash"), ToolKind::Execute);
        assert_eq!(kind_for_tool("web_fetch"), ToolKind::Fetch);
        assert_eq!(kind_for_tool("update_plan"), ToolKind::Think);
        assert_eq!(kind_for_tool("mcp__thing"), ToolKind::Other);
    }

    #[test]
    fn titles_name_the_target_when_available() {
        let raw = serde_json::json!({ "path": "src/main.rs" });
        assert_eq!(
            describe_tool("read_file", Some(&raw)),
            "Reading src/main.rs"
        );
        assert_eq!(
            describe_tool("edit_file", Some(&raw)),
            "Editing src/main.rs"
        );
    }

    #[test]
    fn titles_fall_back_when_the_argument_is_absent() {
        assert_eq!(describe_tool("read_file", None), "Reading a file");
        assert_eq!(describe_tool("bash", None), "Running a command");
    }

    #[test]
    fn a_long_command_is_truncated_in_the_title() {
        let long = "x".repeat(200);
        let raw = serde_json::json!({ "command": long });
        let title = describe_tool("bash", Some(&raw));
        assert!(title.chars().count() < 100, "{title}");
    }

    #[test]
    fn locations_come_from_the_path_argument() {
        let raw = serde_json::json!({ "path": "a/b.rs" });
        let locations = locations_for(Some(&raw)).unwrap();
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].path, "a/b.rs");
    }

    #[test]
    fn no_path_means_no_locations() {
        let raw = serde_json::json!({ "command": "ls" });
        assert!(locations_for(Some(&raw)).is_none());
        assert!(locations_for(None).is_none());
    }

    #[test]
    fn unknown_tool_names_are_humanized() {
        assert_eq!(describe_tool("some_tool", None), "some tool");
    }
}
