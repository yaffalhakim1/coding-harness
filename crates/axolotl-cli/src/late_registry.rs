//! A registry handle that is filled in after the registry is built.
//!
//! The `task` tool needs to hand its child the same tool set it is part of,
//! which is a cycle: registry → task tool → registry. Resolving it with a
//! `OnceLock` avoids both an `Arc::get_mut` dance and a leaked allocation.

use std::sync::{Arc, OnceLock};

use axolotl_actions::registry::ActionRegistry;
use axolotl_actions::ActionError;
use axolotl_models::types::{ToolCall, ToolDefinition};

/// Holds the real registry, which is installed once construction finishes.
pub struct LateBoundRegistry(pub Arc<OnceLock<Arc<dyn ActionRegistry>>>);

#[async_trait::async_trait]
impl ActionRegistry for LateBoundRegistry {
    fn get_tool_definitions(&self) -> Vec<ToolDefinition> {
        self.0
            .get()
            .map(|r| r.get_tool_definitions())
            .unwrap_or_default()
    }

    async fn execute(&self, tool_call: &ToolCall) -> Result<String, ActionError> {
        match self.0.get() {
            Some(registry) => registry.execute(tool_call).await,
            None => Err(ActionError::InvalidArgument(
                "tool registry not ready".into(),
            )),
        }
    }

    fn has_tool(&self, name: &str) -> bool {
        self.0.get().is_some_and(|r| r.has_tool(name))
    }

    fn is_read_only(&self, name: &str) -> bool {
        self.0.get().is_some_and(|r| r.is_read_only(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axolotl_actions::{DefaultRegistry, ToolContext, Workspace};

    fn real_registry() -> Arc<dyn ActionRegistry> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.keep();
        Arc::new(DefaultRegistry::with_predefined_tools(ToolContext::new(
            Workspace::open(path).unwrap(),
        )))
    }

    #[test]
    fn before_the_cell_is_filled_it_exposes_nothing() {
        let late = LateBoundRegistry(Arc::new(OnceLock::new()));
        assert!(late.get_tool_definitions().is_empty());
        assert!(!late.has_tool("bash"));
    }

    #[tokio::test]
    async fn before_the_cell_is_filled_calls_fail_cleanly() {
        let late = LateBoundRegistry(Arc::new(OnceLock::new()));
        let err = late
            .execute(&ToolCall {
                id: "1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({}),
            })
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not ready"), "{err}");
    }

    #[test]
    fn after_the_cell_is_filled_it_delegates() {
        let cell = Arc::new(OnceLock::new());
        let late = LateBoundRegistry(cell.clone());
        let _ = cell.set(real_registry());

        assert!(late.has_tool("bash"));
        assert!(!late.get_tool_definitions().is_empty());
    }

    #[test]
    fn the_cell_cannot_be_replaced() {
        let cell = Arc::new(OnceLock::new());
        let _ = cell.set(real_registry());
        let first = cell.get().cloned();
        // A second set is a no-op, so the registry identity is stable.
        let _ = cell.set(real_registry());
        assert!(Arc::ptr_eq(&first.unwrap(), &cell.get().cloned().unwrap()));
    }
}
