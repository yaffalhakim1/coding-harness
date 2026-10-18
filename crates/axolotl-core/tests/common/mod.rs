//! Shared scaffolding for the agent-loop integration tests.

#![allow(dead_code)]

use std::sync::Arc;

use axolotl_actions::{DefaultRegistry, ToolContext, Workspace};
use axolotl_context::{build_strategy, ContextStrategy, Thresholds, Tokenizer};
use axolotl_core::agent::{run_agent_loop, AgentDeps};
use axolotl_core::{ActionSpace, Session, SessionConfig, Task};
use axolotl_models::mock::MockProvider;
use axolotl_models::types::ToolCall;

pub struct Harness {
    pub dir: tempfile::TempDir,
    pub actions: DefaultRegistry,
    pub context: Box<dyn ContextStrategy>,
    pub tokenizer: Tokenizer,
    /// Shared plan state for the `update_plan` tool. Present when planning is enabled.
    pub plan_state: Option<axolotl_actions::PlanState>,
    /// External store backing `recall_event` (M2).
    pub event_store: axolotl_models::EventStore,
}

/// Build a harness whose tool set and context tier follow `config`.
pub fn harness(provider: &Arc<MockProvider>, config: &SessionConfig) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let ctx = ToolContext::new(Workspace::open(dir.path()).unwrap());

    let plan_state = if config.planning_enabled {
        Some(axolotl_actions::PlanState::new())
    } else {
        None
    };

    let event_store = axolotl_models::EventStore::new();
    let keeps_originals = matches!(
        config.context_strategy,
        axolotl_context::ContextTier::T2 | axolotl_context::ContextTier::T4
    );

    let mut actions = match config.action_space {
        ActionSpace::BashOnly => {
            DefaultRegistry::bash_only_with_plan(ctx.clone(), plan_state.clone())
        }
        ActionSpace::Predefined | ActionSpace::Hybrid => {
            DefaultRegistry::with_predefined_tools_and_plan(ctx.clone(), plan_state.clone())
        }
    };
    if keeps_originals {
        actions = actions.with_recall(ctx, event_store.clone());
    }

    let tokenizer = Tokenizer::new().unwrap();
    let context = build_strategy(
        config.context_strategy,
        Thresholds::from(config.thresholds),
        tokenizer.clone(),
        provider.clone(),
        &config.model,
        event_store.clone(),
    );
    Harness {
        dir,
        actions,
        context,
        tokenizer,
        plan_state,
        event_store,
    }
}

/// Drive the agent loop to completion.
pub fn run(
    h: &Harness,
    provider: &Arc<MockProvider>,
    task: Task,
    config: &SessionConfig,
) -> Session {
    let deps = h.deps(provider);
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(run_agent_loop(&task, config, deps))
        .unwrap()
}

impl Harness {
    /// Borrow into the shape the agent loop expects.
    pub fn deps<'a>(&'a self, provider: &Arc<MockProvider>) -> AgentDeps<'a> {
        AgentDeps {
            llm: provider.clone(),
            actions: &self.actions,
            context: self.context.as_ref(),
            tokenizer: &self.tokenizer,
            planner: None,
            plan_state: self.plan_state.clone(),
            progress: None,
        }
    }
}

pub fn call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args,
    }
}

/// Default config with planning disabled, for tests that script exact turns.
pub fn config_without_planning() -> SessionConfig {
    SessionConfig {
        planning_enabled: false,
        ..SessionConfig::default()
    }
}
