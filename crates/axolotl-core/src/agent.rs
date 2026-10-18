//! Agent execution loop.
//!
//! Paper §2: "The harness follows a ReAct loop, with each turn comprising a
//! reasoning step, an action, and an observation."
//!
//! Responsibilities kept here:
//! - turn budgeting, wall-clock timeout, and session token cap
//! - terminal detection (bounded narration, not "first text turn wins")
//! - tool execution and observation feedback
//! - stuck detection (paper §2.4)
//!
//! Everything else delegates: context to `axolotl-context`, planning to
//! `axolotl-planning`, tools to `axolotl-actions`, verification to
//! [`crate::evaluator`].

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{debug, error, info, warn};

use axolotl_actions::registry::ActionRegistry;
use axolotl_context::factory::begin_turn;
use axolotl_context::{ContextState, ContextStrategy, ContextTier, Tokenizer};
use axolotl_models::provider::LLMProvider;
use axolotl_models::types::ChatMessage;
use axolotl_planning::{DefaultPlanner, PlanningStrategy};

use crate::evaluator::{evaluate_task, run_setup};
use crate::prompts::{system_prompt, task_prompt};
use crate::session::{Session, SessionConfig, SessionOutcome};
use crate::stuck::{StuckDetector, StuckSignal};
use crate::tools::execute_tool_calls;
use crate::turn::Turn;
use crate::types::Task;

/// Everything the loop needs from outside.
pub struct AgentDeps<'a> {
    pub llm: Arc<dyn LLMProvider>,
    pub actions: &'a dyn ActionRegistry,
    pub context: &'a dyn ContextStrategy,
    pub tokenizer: &'a Tokenizer,
    pub planner: Option<&'a dyn PlanningStrategy>,
    /// Shared plan state for the `update_plan` tool. When `None`, planning is
    /// disabled and the tool is not registered.
    pub plan_state: Option<axolotl_actions::PlanState>,
    /// Observer for live progress. Defaults to discarding.
    pub progress: Option<Arc<dyn crate::progress::ProgressSink>>,
}

impl AgentDeps<'_> {
    /// Emit an event, if anyone is listening.
    fn report(&self, event: crate::progress::Progress) {
        if let Some(sink) = &self.progress {
            sink.emit(event);
        }
    }
}

/// Run the agent loop for a single task.
pub async fn run_agent_loop(
    task: &Task,
    config: &SessionConfig,
    deps: AgentDeps<'_>,
) -> Result<Session> {
    let mut session = Session::new(task.clone(), config.clone());
    start_session(&mut session, config, deps, None).await
}

/// Resume a previous session, replaying its history into the context window.
///
/// The prior turns are re-added as conversation history so the model continues
/// from where it stopped. The session keeps its original id so the trajectory
/// and snapshot are updated rather than forked; `resumed_from` records the fact
/// that this run continued another.
pub async fn resume_agent_loop(
    previous: &Session,
    config: &SessionConfig,
    deps: AgentDeps<'_>,
) -> Result<Session> {
    let mut session = previous.clone();
    // A resumed run is a new execution of the same task: timing and turn
    // numbering restart, and the old outcome must not survive into the new
    // verdict. The *history* is not dropped — it is replayed into the context
    // window by `start_session`, so the model continues rather than restarts.
    session.config = config.clone();
    session.outcome = None;
    session.completed_at = None;
    session.started_at = chrono::Utc::now();
    session.peak_context_tokens = 0;
    session.turns.clear();

    start_session(&mut session, config, deps, Some(previous)).await
}

/// Shared body of the loop, with optional history replay.
async fn start_session(
    session: &mut Session,
    config: &SessionConfig,
    deps: AgentDeps<'_>,
    previous: Option<&Session>,
) -> Result<Session> {
    info!(
        session_id = %session.id,
        task_id = %session.task.id,
        context_window = config.context_window,
        context_strategy = %config.context_strategy,
        planning_enabled = config.planning_enabled,
        action_space = %config.action_space,
        resumed_turns = previous.map(|p| p.turns.len()).unwrap_or(0),
        "Starting agent session"
    );

    if let Err(reason) = run_setup(&session.task, deps.actions).await {
        error!(reason = %reason, "Task setup failed");
        session.complete(SessionOutcome::Error(reason));
        return Ok(session.clone());
    }

    let task = session.task.clone();
    let mut state = build_initial_state(&task, config, &deps, session).await?;

    // Replay prior conversation so the model continues rather than restarts.
    if let Some(previous) = previous {
        crate::replay::replay_history(
            &mut state,
            previous,
            deps.tokenizer,
            deps.plan_state.as_ref(),
        );
    }

    let tools = deps.actions.get_tool_definitions();
    let start = Instant::now();
    let mut stuck = StuckDetector::default();
    let mut narration_turns = 0u32;

    deps.report(crate::progress::Progress::SessionStarted {
        session_id: session.id.clone(),
        task_id: session.task.id.clone(),
        resumed_turns: previous.map(|p| p.turns.len()).unwrap_or(0),
    });

    loop {
        if let Some(outcome) = guard(
            config,
            session,
            start.elapsed().as_secs(),
            state.total_tokens(),
        ) {
            session.complete(outcome);
            break;
        }

        session.observe_context(state.total_tokens());

        // Paper §2.1: "Subsequent turns append the plan to the model input
        // without storing it in conversation history." Re-inject the current
        // plan (if any) before building the window.
        if let Some(plan_state) = &deps.plan_state {
            if let Some(plan) = plan_state.get() {
                inject_current_plan(&mut state, &deps, &plan);
            }
        }

        let messages = state.build_window();
        let input_tokens = state.total_tokens();
        let turn_start = Instant::now();

        let response = match deps.llm.chat(&messages, &tools, &config.model).await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "LLM API call failed");
                session.complete(SessionOutcome::ApiError(e.to_string()));
                break;
            }
        };

        let mut turn = Turn::new(
            session.turn_count() as u32,
            messages,
            input_tokens,
            response,
        );
        info!(
            turn = turn.turn_number,
            prompt_tokens = turn.response.usage.prompt_tokens,
            completion_tokens = turn.response.usage.completion_tokens,
            tool_calls = turn.tool_calls.len(),
            stop_reason = ?turn.response.stop_reason,
            "Turn completed"
        );
        deps.report(crate::progress::Progress::TurnCompleted {
            turn: turn.turn_number,
            tool_calls: turn.tool_calls.len(),
            usage: turn.response.usage.clone(),
            text: turn.response.message.text().to_string(),
        });

        if turn.tool_calls.is_empty() {
            // No action means the model is done (paper §2: the loop ends when
            // there is no action). A response that is neither prose nor a tool
            // call is degenerate; tolerate a few of those, then stop.
            let has_content = !turn.response.message.text().trim().is_empty();
            narration_turns = if has_content { 0 } else { narration_turns + 1 };
            let limit = config.max_narration_turns.max(1);

            if has_content || narration_turns >= limit {
                turn.set_duration(turn_start.elapsed().as_millis() as u64);
                session.record_turn(turn);
                session.complete(finish(&task, deps.actions).await);
                break;
            }

            debug!(narration_turns, "Empty response; requesting an action");
            append_narration_nudge(&mut state, &deps, "");
            turn.set_duration(turn_start.elapsed().as_millis() as u64);
            session.record_turn(turn);
            continue;
        }

        narration_turns = 0;
        begin_turn(&mut state);
        state.add_message(
            turn.response.message.clone(),
            deps.tokenizer.count_message_tokens(&turn.response.message),
        );

        let signal = execute_tool_calls(
            &mut turn,
            deps.actions,
            &mut state,
            deps.tokenizer,
            &mut stuck,
            deps.progress.as_ref(),
        )
        .await;
        begin_turn(&mut state);

        turn.set_duration(turn_start.elapsed().as_millis() as u64);
        session.record_turn(turn);

        if let StuckSignal::Terminate { tool, repeats } = signal {
            warn!(tool = %tool, repeats, "Stuck: identical failing calls");
            session.complete(SessionOutcome::Stuck { tool, repeats });
            break;
        }

        let capacity = config.context_window;
        let tokens_before = state.total_tokens();
        if let Err(e) = deps.context.manage(&mut state, capacity).await {
            warn!(error = %e, "Context management failed");
            if config.context_strategy == ContextTier::T0 {
                session.complete(SessionOutcome::ContextOverflow);
                break;
            }
        }
        let tokens_after = state.total_tokens();
        if tokens_after < tokens_before {
            deps.report(crate::progress::Progress::ContextManaged {
                tokens_before,
                tokens_after,
            });
        }
        session.observe_context(state.total_tokens());
    }

    session.summarize_calls = summarize_calls(deps.context);
    log_completion(session);
    deps.report(crate::progress::Progress::SessionFinished {
        outcome: session
            .outcome
            .as_ref()
            .map(|o| o.label().to_string())
            .unwrap_or_else(|| "none".into()),
        turns: session.turn_count(),
        total_tokens: session.total_usage.total_tokens,
    });
    Ok(session.clone())
}

/// Guard rails checked before each model call.
fn guard(
    config: &SessionConfig,
    session: &Session,
    elapsed_secs: u64,
    context_tokens: usize,
) -> Option<SessionOutcome> {
    if session.turn_count() as u32 >= config.max_turns {
        warn!(turns = session.turn_count(), "Turn limit reached");
        return Some(SessionOutcome::TurnLimitReached);
    }
    if elapsed_secs > config.timeout_secs {
        warn!(elapsed_secs, "Timeout reached");
        return Some(SessionOutcome::Timeout);
    }
    if let Some(budget) = config.token_budget {
        if session.total_usage.total_tokens >= budget {
            warn!(
                used = session.total_usage.total_tokens,
                budget, "Token budget exhausted"
            );
            return Some(SessionOutcome::TurnLimitReached);
        }
    }
    if context_tokens > config.context_window && config.context_strategy == ContextTier::T0 {
        // T0 does not manage context; sending it would fail at the API anyway.
        return Some(SessionOutcome::ContextOverflow);
    }
    None
}

/// Build the initial context and optionally generate a plan.
async fn build_initial_state(
    task: &Task,
    config: &SessionConfig,
    deps: &AgentDeps<'_>,
    session: &mut Session,
) -> Result<ContextState> {
    let system = system_prompt(config.action_space, config.planning_enabled);
    let user = task_prompt(&task.description, &task.test_commands);
    let state = axolotl_context::new_state_with_preamble(deps.tokenizer, &system, &user)?;

    if !config.planning_enabled {
        return Ok(state);
    }

    let default_planner = DefaultPlanner::new();
    let planner: &dyn PlanningStrategy = deps.planner.unwrap_or(&default_planner);
    inject_plan(planner, task, config, deps, session, state).await
}

/// Generate a plan and inject it into the protected preamble.
async fn inject_plan(
    planner: &dyn PlanningStrategy,
    task: &Task,
    config: &SessionConfig,
    deps: &AgentDeps<'_>,
    session: &mut Session,
    mut state: ContextState,
) -> Result<ContextState> {
    info!("Generating plan");
    let plan = planner
        .generate_plan(&task.description, deps.llm.as_ref(), &config.model)
        .await
        .context("planning failed")?;
    let tokens = deps.tokenizer.count_tokens(&plan) + 16;
    session.set_plan(plan.clone());
    deps.context.inject_plan(&mut state, &plan, tokens);
    debug!(plan_chars = plan.len(), "Plan generated");
    Ok(state)
}

/// Append a nudge when the model narrated instead of acting.
fn append_narration_nudge(state: &mut ContextState, deps: &AgentDeps<'_>, text: &str) {
    let msg = ChatMessage::assistant(text.to_string());
    let tokens = deps.tokenizer.count_message_tokens(&msg);
    state.add_message(msg, tokens);

    let nudge = ChatMessage::user(
        "You did not call any tool. Continue the task by calling a tool now, or state that the task is complete.",
    );
    let nudge_tokens = deps.tokenizer.count_message_tokens(&nudge);
    state.add_message(nudge, nudge_tokens);
}

/// Evaluate the task and map it onto an outcome.
async fn finish(task: &Task, actions: &dyn ActionRegistry) -> SessionOutcome {
    info!("Model stopped acting — evaluating result");
    let result = evaluate_task(task, actions).await;
    if !task.is_verifiable() {
        return SessionOutcome::Unverified(result);
    }
    if result.passed {
        SessionOutcome::Success(result)
    } else {
        SessionOutcome::Failure(result)
    }
}

/// Inject the current plan into the context before each turn.
///
/// Paper §2.1 / Figure 17: the plan is supplied anew before each model call
/// rather than appended to conversation history, so it stays current without
/// accumulating duplicate copies.
fn inject_current_plan(state: &mut ContextState, deps: &AgentDeps<'_>, plan: &str) {
    let reminder = format!(
        "<system-reminder>\nCurrent plan (update it via update_plan as you progress; keep exactly one task in_progress, and mark a task completed the moment it is actually done):\n{}\n</system-reminder>",
        plan
    );
    let msg = ChatMessage::system(reminder);
    let tokens = deps.tokenizer.count_message_tokens(&msg);
    state.add_message(msg, tokens);
}

fn summarize_calls(strategy: &dyn ContextStrategy) -> usize {
    strategy.summarize_calls()
}

fn log_completion(session: &Session) {
    info!(
        session_id = %session.id,
        turns = session.turn_count(),
        total_tokens = session.total_usage.total_tokens,
        peak_context_tokens = session.peak_context_tokens,
        cost_usd = ?session.cost_usd(),
        duration_secs = session.elapsed_secs(),
        outcome = ?session.outcome.as_ref().map(|o| o.label()),
        "Session completed"
    );
}
