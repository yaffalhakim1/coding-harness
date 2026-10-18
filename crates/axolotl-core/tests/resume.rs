//! Tests for session resume: the model must continue, not restart.
//!
//! The failure mode this guards against is subtle: a resumed run can look fine
//! (it completes, it writes a trajectory) while having silently discarded the
//! prior conversation and started over.

mod common;

use axolotl_core::{SessionConfig, SessionOutcome, Task, Turn};
use axolotl_models::mock::{MockProvider, ScriptedStep};
use axolotl_models::types::{ChatMessage, LLMResponse, StopReason, TokenUsage};
use common::{call, harness, run};

/// Build a session that already did some work.
fn previous_session(task: &Task) -> axolotl_core::Session {
    let mut session = axolotl_core::Session::new(task.clone(), SessionConfig::default());

    // One turn: the model read a file and got its contents back.
    let response = LLMResponse {
        message: ChatMessage::assistant_with_tool_calls(
            Some("Reading the config first.".into()),
            vec![call(
                "c1",
                "read_file",
                serde_json::json!({ "path": "a.txt" }),
            )],
        ),
        usage: TokenUsage::default(),
        stop_reason: StopReason::ToolUse,
        model: "mock".into(),
    };
    let mut turn = Turn::new(0, vec![], 100, response);
    turn.add_tool_result(axolotl_core::ToolResult {
        tool_call_id: "c1".into(),
        tool_name: "read_file".into(),
        output: "PRIOR_OUTPUT_MARKER".into(),
        success: true,
        error: None,
        duration_ms: 1,
    });
    session.record_turn(turn);
    session.complete(SessionOutcome::TurnLimitReached);
    session
}

#[test]
fn resume_replays_prior_history_into_the_request() {
    let provider = MockProvider::new(vec![ScriptedStep::text("continuing")]);
    let config = SessionConfig {
        planning_enabled: false,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let task = Task::from_description("finish the job");
    let previous = previous_session(&task);

    let session = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(axolotl_core::agent::resume_agent_loop(
            &previous,
            &config,
            h.deps(&provider),
        ))
        .unwrap();

    // The resumed request must carry the earlier exchange, or the model is
    // starting from nothing.
    let requests = provider.requests();
    assert_eq!(requests.len(), 1);
    let sent = &requests[0];
    assert!(
        sent.iter()
            .any(|m| m.text().contains("PRIOR_OUTPUT_MARKER")),
        "prior tool output missing from the resumed request: {sent:?}"
    );
    assert!(
        sent.iter()
            .any(|m| m.text().contains("Reading the config first")),
        "prior assistant message missing from the resumed request"
    );

    // Turn numbering restarts for the new execution.
    assert_eq!(session.turns[0].turn_number, 0);
}

#[test]
fn resume_does_not_duplicate_prior_turns() {
    let provider = MockProvider::new(vec![ScriptedStep::text("done")]);
    let config = SessionConfig {
        planning_enabled: false,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let task = Task::from_description("finish the job");
    let previous = previous_session(&task);

    let session = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(axolotl_core::agent::resume_agent_loop(
            &previous,
            &config,
            h.deps(&provider),
        ))
        .unwrap();

    // Only the new turn is recorded; the old one lives in the context window,
    // not in the new session's turn list.
    assert_eq!(
        session.turns.len(),
        1,
        "resumed session must not append to the old turn list"
    );
    assert_eq!(session.id, previous.id, "resume continues the same session");
}

#[test]
fn resume_clears_the_old_outcome() {
    let provider = MockProvider::new(vec![ScriptedStep::text("done")]);
    let config = SessionConfig {
        planning_enabled: false,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let task = Task::from_description("finish the job");
    let previous = previous_session(&task);
    assert!(matches!(
        previous.outcome,
        Some(SessionOutcome::TurnLimitReached)
    ));

    let session = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(axolotl_core::agent::resume_agent_loop(
            &previous,
            &config,
            h.deps(&provider),
        ))
        .unwrap();

    // A fresh verdict is reached, not the stale one.
    assert!(!matches!(
        session.outcome,
        Some(SessionOutcome::TurnLimitReached)
    ));
}

#[test]
fn resume_restores_the_plan_when_planning_is_enabled() {
    let provider = MockProvider::new(vec![ScriptedStep::text("continuing")]);
    let config = SessionConfig {
        planning_enabled: true,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let task = Task::from_description("finish the job");

    let mut previous = previous_session(&task);
    previous.plan = Some("1. Read the config\n2. Edit it".into());

    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(axolotl_core::agent::resume_agent_loop(
            &previous,
            &config,
            h.deps(&provider),
        ))
        .unwrap();

    let plan_state = h.plan_state.as_ref().expect("planning enabled");
    assert_eq!(
        plan_state.get().as_deref(),
        Some("1. Read the config\n2. Edit it"),
        "the prior plan must survive a resume"
    );
}

#[test]
fn a_fresh_run_has_no_replay() {
    // Control: the same harness without a previous session must not inject
    // anything from another run.
    let provider = MockProvider::new(vec![ScriptedStep::text("hello")]);
    let config = SessionConfig {
        planning_enabled: false,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let session = run(&h, &provider, Task::from_description("x"), &config);

    assert_eq!(session.turns.len(), 1);
    let sent = &provider.requests()[0];
    assert!(!sent
        .iter()
        .any(|m| m.text().contains("PRIOR_OUTPUT_MARKER")));
}
