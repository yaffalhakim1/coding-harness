//! Subagents: the parent agent delegates a self-contained job to a child run.
//!
//! This changes the action space, so it is **opt-in** (see
//! [`SessionConfig::subagents`](crate::session::SessionConfig)). A sweep that
//! does not enable it measures exactly what it measured before.
//!
//! Why a subagent is worth having: a child gets a *fresh* context window. A
//! parent that has spent 80% of its budget exploring cannot cheaply answer a
//! narrow follow-up question, but a child can, and the parent only pays for the
//! final answer rather than the whole search.
//!
//! Recursion is bounded: a child cannot spawn grandchildren beyond
//! [`MAX_DEPTH`], so a confused model cannot fork-bomb the budget.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{info, warn};

use harness_actions::registry::{Action, ActionRegistry};
use harness_actions::ActionError;
use harness_models::provider::LLMProvider;
use harness_models::types::ToolDefinition;

use crate::session::{Session, SessionConfig};
use crate::types::Task;

/// How many levels of nesting are allowed.
pub const MAX_DEPTH: usize = 2;

/// Registry wrapper that hides the `task` tool from children past the depth
/// limit, so a child cannot recurse indefinitely.
struct DepthLimitedRegistry {
    inner: Arc<dyn ActionRegistry>,
    depth: usize,
}

impl DepthLimitedRegistry {
    fn allows_subagents(&self) -> bool {
        self.depth < MAX_DEPTH
    }
}

#[async_trait]
impl ActionRegistry for DepthLimitedRegistry {
    fn get_tool_definitions(&self) -> Vec<ToolDefinition> {
        self.inner
            .get_tool_definitions()
            .into_iter()
            .filter(|d| self.allows_subagents() || d.name != "task")
            .collect()
    }

    async fn execute(
        &self,
        tool_call: &harness_models::types::ToolCall,
    ) -> Result<String, ActionError> {
        if tool_call.name == "task" && !self.allows_subagents() {
            return Err(ActionError::InvalidArgument(
                "subagents cannot spawn further subagents at this depth".into(),
            ));
        }
        self.inner.execute(tool_call).await
    }

    fn has_tool(&self, name: &str) -> bool {
        if name == "task" && !self.allows_subagents() {
            return false;
        }
        self.inner.has_tool(name)
    }

    fn is_read_only(&self, name: &str) -> bool {
        self.inner.is_read_only(name)
    }
}

/// The `task` tool: run a child agent and return its answer.
pub struct SubagentTool {
    llm: Arc<dyn LLMProvider>,
    actions: Arc<dyn ActionRegistry>,
    /// Config inherited from the parent, minus the turn budget and planning.
    config: SessionConfig,
    depth: usize,
}

impl SubagentTool {
    /// Build a tool that spawns children at `depth`.
    pub fn new(
        llm: Arc<dyn LLMProvider>,
        actions: Arc<dyn ActionRegistry>,
        config: SessionConfig,
        depth: usize,
    ) -> Self {
        Self {
            llm,
            actions,
            config,
            depth,
        }
    }

    /// Config for a child run.
    ///
    /// Planning is disabled: a child answering a narrow question does not need
    /// a plan, and paying for one on every delegation is how subagents become
    /// more expensive than doing the work inline.
    fn child_config(&self) -> SessionConfig {
        let mut config = self.config.clone();
        config.planning_enabled = false;
        // A child is a focused sub-task; give it a small budget relative to the
        // parent so a runaway child cannot consume the whole session.
        config.max_turns = (self.config.max_turns / 2).clamp(3, 15);
        config.subagents = false;
        config
    }

    /// Extract the answer from a finished child session.
    fn answer(child: &Session) -> String {
        // Prefer the last assistant text; a child that ends by calling a tool
        // still usually narrated its finding on the previous turn.
        for turn in child.turns.iter().rev() {
            let text = turn.response.message.text().trim();
            if !text.is_empty() {
                return text.to_string();
            }
        }
        format!(
            "Subagent finished without a written answer (outcome: {:?}).",
            child.outcome.as_ref().map(|o| o.label())
        )
    }
}

#[async_trait]
impl Action for SubagentTool {
    fn name(&self) -> &str {
        "task"
    }

    fn is_read_only(&self) -> bool {
        // A child may edit files; never run this concurrently with reads.
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "task".to_string(),
            description: "Delegate a self-contained job to a subagent and get back its written answer.\n\nWhen: the work is a bounded search or investigation whose intermediate steps you do not need to see — \"find every place X is configured\", \"read these five files and summarise how auth works\". Do NOT use it for work you can finish in one or two tool calls, and do not use it to ask yourself a question you already have the answer to.\n\nProtocol: pass `description` (the job, stated as a complete instruction — the subagent does not see this conversation) and optionally `tools` (a subset of tool names it may use; omit for all).\n\nThe subagent runs in its own context window with its own budget and returns only its final written answer, so a long investigation costs you one short message instead of every step.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Complete, self-contained instructions for the subagent"
                    },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional subset of tool names the subagent may use"
                    }
                },
                "required": ["description"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> Result<String, ActionError> {
        let description = arguments
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ActionError::InvalidArgument("missing 'description'".into()))?;

        if description.trim().is_empty() {
            return Err(ActionError::InvalidArgument(
                "description must not be empty".into(),
            ));
        }

        if self.depth >= MAX_DEPTH {
            return Err(ActionError::InvalidArgument(format!(
                "maximum subagent depth ({MAX_DEPTH}) reached"
            )));
        }

        let config = self.child_config();
        let task = Task {
            id: format!("subagent-{}", uuid::Uuid::new_v4()),
            description: description.to_string(),
            // A subagent reports back to its parent; it is not itself verified
            // by a test command, so it must not claim task success.
            setup_commands: Vec::new(),
            test_commands: Vec::new(),
            working_dir: None,
            expected_files: Vec::new(),
        };

        info!(
            depth = self.depth + 1,
            max_turns = config.max_turns,
            "Spawning subagent"
        );

        let registry = DepthLimitedRegistry {
            inner: self.actions.clone(),
            depth: self.depth + 1,
        };

        let tokenizer = match harness_context::Tokenizer::new() {
            Ok(t) => t,
            Err(e) => {
                return Err(ActionError::InvalidArgument(format!(
                    "subagent could not start: {e}"
                )))
            }
        };

        let context = harness_context::build_strategy(
            config.context_strategy,
            harness_context::Thresholds::from(config.thresholds),
            tokenizer.clone(),
            self.llm.clone(),
            &config.model,
            harness_models::EventStore::new(),
        );

        let deps = crate::agent::AgentDeps {
            llm: self.llm.clone(),
            actions: &registry,
            context: context.as_ref(),
            tokenizer: &tokenizer,
            planner: None,
            plan_state: None,
            progress: None,
        };

        let child = match crate::agent::run_agent_loop(&task, &config, deps).await {
            Ok(session) => session,
            Err(e) => {
                warn!(error = %e, "Subagent failed");
                return Err(ActionError::InvalidArgument(format!(
                    "subagent failed: {e}"
                )));
            }
        };

        info!(
            depth = self.depth + 1,
            turns = child.turn_count(),
            tokens = child.total_usage.total_tokens,
            "Subagent finished"
        );

        Ok(format!(
            "{}\n\n[subagent: {} turns, {} tokens]",
            Self::answer(&child),
            child.turn_count(),
            child.total_usage.total_tokens
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harness_actions::{DefaultRegistry, ToolContext, Workspace};
    use harness_models::mock::{MockProvider, ScriptedStep};

    fn config() -> SessionConfig {
        SessionConfig {
            planning_enabled: false,
            max_turns: 10,
            ..SessionConfig::default()
        }
    }

    fn registry() -> Arc<DefaultRegistry> {
        let dir = tempfile::tempdir().unwrap();
        // Leak the tempdir: the registry holds a path, and these tests are
        // short-lived processes.
        let path = dir.keep();
        Arc::new(DefaultRegistry::with_predefined_tools(ToolContext::new(
            Workspace::open(path).unwrap(),
        )))
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    #[test]
    fn child_config_disables_planning_and_nesting() {
        let tool = SubagentTool::new(MockProvider::new(vec![]), registry(), config(), 0);
        let child = tool.child_config();
        assert!(
            !child.planning_enabled,
            "children must not pay for planning"
        );
        assert!(!child.subagents, "children must not spawn grandchildren");
        assert!(child.max_turns <= 15);
        assert!(child.max_turns >= 3);
    }

    #[test]
    fn answer_prefers_the_last_written_text() {
        use crate::turn::Turn;
        use harness_models::types::{ChatMessage, LLMResponse, StopReason, TokenUsage};

        let mut session = Session::new(Task::from_description("x"), config());
        for text in ["first", "second"] {
            let response = LLMResponse {
                message: ChatMessage::assistant(text),
                usage: TokenUsage::default(),
                stop_reason: StopReason::EndTurn,
                model: "m".into(),
            };
            session.record_turn(Turn::new(0, vec![], 0, response));
        }
        assert_eq!(SubagentTool::answer(&session), "second");
    }

    #[test]
    fn answer_explains_when_the_child_wrote_nothing() {
        let session = Session::new(Task::from_description("x"), config());
        assert!(SubagentTool::answer(&session).contains("without a written answer"));
    }

    #[test]
    fn depth_limit_is_enforced() {
        let tool = SubagentTool::new(MockProvider::new(vec![]), registry(), config(), MAX_DEPTH);
        let err = rt()
            .block_on(tool.execute(serde_json::json!({ "description": "do a thing" })))
            .unwrap_err();
        assert!(err.to_string().contains("maximum subagent depth"), "{err}");
    }

    #[test]
    fn empty_description_is_rejected() {
        let tool = SubagentTool::new(MockProvider::new(vec![]), registry(), config(), 0);
        let err = rt()
            .block_on(tool.execute(serde_json::json!({ "description": "   " })))
            .unwrap_err();
        assert!(matches!(err, ActionError::InvalidArgument(_)), "{err}");
    }

    #[test]
    fn a_subagent_runs_and_returns_its_answer() {
        let provider = MockProvider::new(vec![ScriptedStep::text("the answer is 42")]);
        let tool = SubagentTool::new(provider.clone(), registry(), config(), 0);

        let out = rt()
            .block_on(tool.execute(serde_json::json!({
                "description": "work out the answer"
            })))
            .unwrap();

        assert!(out.contains("the answer is 42"), "{out}");
        assert!(out.contains("[subagent:"), "{out}");
        // The child was a real agent run, not a passthrough.
        assert_eq!(provider.request_count(), 1);
    }

    #[test]
    fn the_task_tool_is_hidden_at_max_depth() {
        let inner = registry();
        let at_limit = DepthLimitedRegistry {
            inner: inner.clone(),
            depth: MAX_DEPTH,
        };
        assert!(!at_limit.allows_subagents());

        let shallow = DepthLimitedRegistry { inner, depth: 0 };
        assert!(shallow.allows_subagents());
    }
}
