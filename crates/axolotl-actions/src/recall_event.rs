//! The `recall_event` tool: recovers content elided by context management (M2).
//!
//! Paper §2.3: "Recall (M2) stores elided observations in the file system and
//! exposes a `recall_event` tool to read them back on demand, making elision
//! reversible." Paper Figure 31 defines the description.
//!
//! Only T2 and T4 expose this tool, because only they keep originals.

use async_trait::async_trait;

use axolotl_models::event_store::EventStore;
use axolotl_models::types::ToolDefinition;

use crate::error::{ActionError, ActionResult};
use crate::registry::{Action, ToolContext};

/// Read-only recovery of an elided tool observation.
pub struct RecallEventTool {
    _ctx: ToolContext,
    store: EventStore,
    /// Truncation limit, shared with other tools.
    max_output_chars: usize,
}

impl RecallEventTool {
    pub fn new(ctx: ToolContext, store: EventStore) -> Self {
        let max_output_chars = ctx.max_output_chars;
        Self {
            _ctx: ctx,
            store,
            max_output_chars,
        }
    }
}

#[async_trait]
impl Action for RecallEventTool {
    fn name(&self) -> &str {
        "recall_event"
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall_event".to_string(),
            description: "Fetch the full original content of a past turn (event) by its id. When the conversation has been compacted, old tool outputs are replaced by stubs like '[tool output elided ... Use recall_event(<id>) ...]' and the oldest turns may be folded into a summary; call this with the event id to get the verbatim messages (including the full tool output) back. Often you can instead just re-read the file or re-run the command — use recall_event when the output isn't easily reproducible (e.g. a past test log). Read-only; returns an error for an unknown id.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The event id shown in the elision stub"
                    }
                },
                "required": ["id"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let id = arguments
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ActionError::InvalidArgument("missing 'id'".into()))?;

        let event = self
            .store
            .get(id)
            .ok_or_else(|| ActionError::UnknownEvent {
                id: id.to_string(),
                known: self.store.len(),
            })?;

        Ok(crate::registry::truncate_output(
            event.content,
            self.max_output_chars,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;

    fn ctx() -> ToolContext {
        let dir = tempfile::tempdir().unwrap();
        ToolContext::new(Workspace::open(dir.path()).unwrap())
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    #[test]
    fn recalls_stored_content() {
        let store = EventStore::new();
        store.insert("evt-1", "tool output", "the full test log");
        let tool = RecallEventTool::new(ctx(), store);

        let out = rt()
            .block_on(tool.execute(serde_json::json!({ "id": "evt-1" })))
            .unwrap();
        assert_eq!(out, "the full test log");
    }

    #[test]
    fn unknown_id_is_a_tool_error() {
        let tool = RecallEventTool::new(ctx(), EventStore::new());
        let err = rt()
            .block_on(tool.execute(serde_json::json!({ "id": "nope" })))
            .unwrap_err();
        assert!(matches!(err, ActionError::UnknownEvent { .. }), "{err:?}");
    }

    #[test]
    fn missing_id_argument_is_a_tool_error() {
        let tool = RecallEventTool::new(ctx(), EventStore::new());
        let err = rt()
            .block_on(tool.execute(serde_json::json!({})))
            .unwrap_err();
        assert!(matches!(err, ActionError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn is_read_only() {
        let tool = RecallEventTool::new(ctx(), EventStore::new());
        assert!(tool.is_read_only());
    }

    #[test]
    fn output_is_truncated_to_the_budget() {
        let store = EventStore::new();
        store.insert("big", "tool output", "x".repeat(50_000));
        let tool = RecallEventTool::new(ctx(), store);
        let out = rt()
            .block_on(tool.execute(serde_json::json!({ "id": "big" })))
            .unwrap();
        assert!(out.contains("truncated"), "{}", &out[out.len() - 60..]);
    }
}
