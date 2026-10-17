//! Default LLM-based planner.
//!
//! System prompt adapted from paper Appendix A.2 (Figure 15): the model is told
//! a plan is shown back before every turn, to keep exactly one step active, and
//! to skip planning only for trivial requests.

use anyhow::{Context, Result};
use async_trait::async_trait;
use tracing::debug;

use harness_models::provider::LLMProvider;
use harness_models::types::{ChatMessage, ModelConfig};

use crate::strategy::PlanningStrategy;

/// Planner system prompt, adapted from paper Figure 15.
pub const PLANNING_SYSTEM_PROMPT: &str = "\
# Task planning
You are a planning assistant for a coding agent. You have an `update_plan` tool that holds a todo list.
The harness shows the current plan back to the agent before every turn, so use it to stay on track.
- For any non-trivial task (about 3+ steps), the FIRST action MUST be to call `update_plan` to break
  the task into concrete steps.
- Keep it current: mark exactly ONE task `in_progress` while you work on it, and mark a task
  `completed` the moment it is actually done (don't batch completions).
- Add new tasks as you discover them; re-send the COMPLETE list each time.
- Skip planning only for a single trivial step or a purely informational request.

Produce the plan as a concise numbered list. Be specific about file paths and code changes.
Do NOT write code. Only describe the plan.";

/// Default planner that asks the LLM to produce a step-by-step plan.
pub struct DefaultPlanner {
    system_prompt: String,
}

impl DefaultPlanner {
    pub fn new() -> Self {
        Self {
            system_prompt: PLANNING_SYSTEM_PROMPT.to_string(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

impl Default for DefaultPlanner {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PlanningStrategy for DefaultPlanner {
    async fn generate_plan(
        &self,
        task_description: &str,
        llm: &dyn LLMProvider,
        config: &ModelConfig,
    ) -> Result<String> {
        debug!("Generating plan");

        let messages = vec![
            ChatMessage::system(&self.system_prompt),
            ChatMessage::user(format!(
                "## Task\n{task_description}\n\nProduce a step-by-step plan to solve this task."
            )),
        ];

        let response = llm
            .chat(&messages, &[], config)
            .await
            .context("Failed to generate plan")?;

        response
            .message
            .content
            .filter(|c| !c.trim().is_empty())
            .context("LLM did not return a plan")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use harness_models::mock::{MockProvider, ScriptedStep};

    use super::*;

    fn run(provider: Arc<MockProvider>) -> Result<String> {
        let planner = DefaultPlanner::new();
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(planner.generate_plan("fix the bug", &*provider, &ModelConfig::default()))
    }

    #[test]
    fn returns_the_plan_text() {
        let provider = MockProvider::new(vec![ScriptedStep::text("1. read\n2. edit")]);
        let plan = run(provider.clone()).unwrap();
        assert!(plan.contains("2. edit"));
        assert_eq!(provider.request_count(), 1);
    }

    #[test]
    fn sends_the_task_in_the_user_turn() {
        let provider = MockProvider::new(vec![ScriptedStep::text("1. x")]);
        let _ = run(provider.clone());
        let sent = provider.requests();
        assert_eq!(sent[0][1].role, harness_models::types::Role::User);
        assert!(sent[0][1].text().contains("fix the bug"));
        assert_eq!(sent[0][0].text(), PLANNING_SYSTEM_PROMPT);
    }

    #[test]
    fn empty_plan_is_an_error() {
        let provider = MockProvider::new(vec![ScriptedStep::text("   ")]);
        assert!(run(provider).is_err());
    }

    #[test]
    fn provider_failure_propagates() {
        let provider = MockProvider::new(vec![ScriptedStep::Fail("down".into())]);
        assert!(run(provider).is_err());
    }
}
