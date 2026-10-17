//! Shared scaffolding for the agent-loop integration tests.

#![allow(dead_code)]

use std::sync::Arc;

use harness_actions::{DefaultRegistry, ToolContext, Workspace};
use harness_context::{build_strategy, ContextStrategy, Thresholds, Tokenizer};
use harness_core::agent::{run_agent_loop, AgentDeps};
use harness_core::{ActionSpace, Session, SessionConfig, Task};
use harness_models::mock::MockProvider;
use harness_models::types::ToolCall;

pub struct Harness {
    pub dir: tempfile::TempDir,
    pub actions: DefaultRegistry,
    pub context: Box<dyn ContextStrategy>,
    pub tokenizer: Tokenizer,
    /// Shared plan state for the `update_plan` tool. Present when planning is enabled.
    pub plan_state: Option<harness_actions::PlanState>,
    /// External store backing `recall_event` (M2).
    pub event_store: harness_models::EventStore,
}

/// Build a harness whose tool set and context tier follow `config`.
pub fn harness(provider: &Arc<MockProvider>, config: &SessionConfig) -> Harness {
    let dir = tempfile::tempdir().unwrap();
    let ctx = ToolContext::new(Workspace::open(dir.path()).unwrap());

    let plan_state = if config.planning_enabled {
        Some(harness_actions::PlanState::new())
    } else {
        None
    };

    let event_store = harness_models::EventStore::new();
    let keeps_originals = matches!(
        config.context_strategy,
        harness_context::ContextTier::T2 | harness_context::ContextTier::T4
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
    let deps = AgentDeps {
        llm: provider.clone(),
        actions: &h.actions,
        context: h.context.as_ref(),
        tokenizer: &h.tokenizer,
        planner: None,
        plan_state: h.plan_state.clone(),
    };
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(run_agent_loop(&task, config, deps))
        .unwrap()
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
