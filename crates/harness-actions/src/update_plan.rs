//! The `update_plan` tool: lets the model revise its plan during execution.
//!
//! Paper §2.1: "The model maintains this plan through the `update_plan` tool."
//! This tool stores the plan in session state so it can be re-injected before
//! each turn (paper Figure 17: "subsequent turns append the plan to the model
//! input without storing it in conversation history").

use async_trait::async_trait;
use std::sync::{Arc, Mutex};

use harness_models::types::ToolDefinition;

use crate::error::{ActionError, ActionResult};
use crate::registry::{Action, ToolContext};

/// Shared plan state, updated by `update_plan` and read by the agent loop.
#[derive(Debug, Clone, Default)]
pub struct PlanState {
    inner: Arc<Mutex<Option<String>>>,
}

impl PlanState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a new plan, replacing the previous one.
    pub fn set(&self, plan: String) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = Some(plan);
        }
    }

    /// Read the current plan, if any.
    pub fn get(&self) -> Option<String> {
        self.inner.lock().ok().and_then(|g| g.clone())
    }

    /// Clear the plan.
    pub fn clear(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = None;
        }
    }
}

/// The `update_plan` tool: replaces the stored plan.
pub struct UpdatePlanTool {
    _ctx: ToolContext,
    state: PlanState,
}

impl UpdatePlanTool {
    pub fn new(ctx: ToolContext, state: PlanState) -> Self {
        Self { _ctx: ctx, state }
    }
}

#[async_trait]
impl Action for UpdatePlanTool {
    fn name(&self) -> &str {
        "update_plan"
    }

    fn is_read_only(&self) -> bool {
        // Not read-only: it mutates session state (the plan).
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "update_plan".to_string(),
            description: "Create or update your task plan (a todo list). The harness shows this plan back to you before every turn so you stay on track and don't lose steps.\n\nWhen: for any non-trivial task (about 3+ steps), call this FIRST to lay out the steps, then keep it updated as you work. Skip only for a single trivial step or a purely informational request.\n\nProtocol: pass the COMPLETE updated list every time — it fully replaces the previous list (no incremental edits). Each item has `content` (imperative, e.g. 'Fix the parser in foo.py'), `status` (pending | in_progress | completed), and `activeForm` (present-continuous, e.g. 'Fixing the parser'). Keep EXACTLY ONE task in_progress at a time; mark a task completed the moment it is actually done (do not batch completions); add new tasks as you discover them.\n\nSide effects: updates the in-memory plan only — it touches no files and is not written to disk.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "The complete plan text (numbered list of steps with status)"
                    }
                },
                "required": ["plan"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let plan = arguments
            .get("plan")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ActionError::InvalidArgument("missing 'plan'".into()))?;

        if plan.trim().is_empty() {
            return Err(ActionError::InvalidArgument("plan cannot be empty".into()));
        }

        self.state.set(plan.to_string());
        Ok("Plan updated.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ToolContext;
    use crate::workspace::Workspace;

    fn ctx() -> ToolContext {
        let dir = tempfile::tempdir().unwrap();
        ToolContext::new(Workspace::open(dir.path()).unwrap())
    }

    #[tokio::test]
    async fn update_plan_stores_the_plan() {
        let state = PlanState::new();
        let tool = UpdatePlanTool::new(ctx(), state.clone());

        let result = tool
            .execute(serde_json::json!({
                "plan": "1. Read file\n2. Edit\n3. Test"
            }))
            .await
            .unwrap();

        assert!(result.contains("Plan updated"));
        assert_eq!(state.get().unwrap(), "1. Read file\n2. Edit\n3. Test");
    }

    #[tokio::test]
    async fn update_plan_replaces_previous() {
        let state = PlanState::new();
        let tool = UpdatePlanTool::new(ctx(), state.clone());

        tool.execute(serde_json::json!({ "plan": "old plan" }))
            .await
            .unwrap();
        tool.execute(serde_json::json!({ "plan": "new plan" }))
            .await
            .unwrap();

        assert_eq!(state.get().unwrap(), "new plan");
    }

    #[tokio::test]
    async fn empty_plan_is_rejected() {
        let state = PlanState::new();
        let tool = UpdatePlanTool::new(ctx(), state.clone());

        let err = tool.execute(serde_json::json!({ "plan": "   " })).await;
        assert!(err.is_err());
        assert!(state.get().is_none());
    }

    #[tokio::test]
    async fn missing_plan_argument_is_rejected() {
        let state = PlanState::new();
        let tool = UpdatePlanTool::new(ctx(), state.clone());

        let err = tool.execute(serde_json::json!({})).await;
        assert!(err.is_err());
    }

    #[test]
    fn plan_state_default_is_none() {
        let state = PlanState::new();
        assert!(state.get().is_none());
    }

    #[test]
    fn plan_state_clear_works() {
        let state = PlanState::new();
        state.set("some plan".into());
        assert!(state.get().is_some());
        state.clear();
        assert!(state.get().is_none());
    }
}
