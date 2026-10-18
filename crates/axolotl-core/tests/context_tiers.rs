//! Context-management behaviour of the agent loop, driven by a scripted provider.
//!
//! The paper's central claim is that managed tiers survive windows that kill
//! T0, so these tests compare the tiers under the same tight budget.

mod common;

use axolotl_core::{SessionConfig, SessionOutcome, Task};
use axolotl_models::mock::{MockProvider, ScriptedStep};
use common::{call, config_without_planning, harness, run};

#[test]
fn t0_reports_context_overflow() {
    // T0 has no management: once the window is exceeded the run must end with
    // ContextOverflow (paper §2.3), not silently truncate.
    let big = ScriptedStep::tool_calls(vec![call(
        "1",
        "grep_text",
        serde_json::json!({ "query": ".", "path": "." }),
    )]);
    let provider = MockProvider::repeating(big);

    let config = SessionConfig {
        planning_enabled: false,
        context_strategy: axolotl_context::ContextTier::T0,
        context_window: 200,
        max_narration_turns: 100,
        max_turns: 100,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    // Enough content that one grep observation blows a 200-token window.
    for i in 0..50 {
        std::fs::write(
            h.dir.path().join(format!("f{i}.txt")),
            "needle\n".repeat(20),
        )
        .unwrap();
    }
    let session = run(&h, &provider, Task::from_description("x"), &config);

    assert!(
        matches!(session.outcome, Some(SessionOutcome::ContextOverflow)),
        "outcome: {:?}",
        session.outcome
    );
}

#[test]
fn t4_keeps_a_small_window_running() {
    // The managed tier must survive the same window that kills T0.
    let busy = ScriptedStep::tool_calls(vec![call(
        "1",
        "grep_text",
        serde_json::json!({ "query": ".", "path": "." }),
    )]);
    let provider = MockProvider::repeating(busy);

    let config = SessionConfig {
        planning_enabled: false,
        context_strategy: axolotl_context::ContextTier::T4,
        context_window: 900,
        max_turns: 6,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    for i in 0..50 {
        std::fs::write(
            h.dir.path().join(format!("f{i}.txt")),
            "needle\n".repeat(20),
        )
        .unwrap();
    }
    let session = run(&h, &provider, Task::from_description("x"), &config);

    // It stops on the turn limit, not on overflow.
    assert!(
        matches!(session.outcome, Some(SessionOutcome::TurnLimitReached)),
        "outcome: {:?}",
        session.outcome
    );
    assert!(session.peak_context_tokens > 0);
}

#[test]
fn t4_elides_rather_than_summarizing_when_possible() {
    // Staging is the whole point: if elision reclaims enough space, M3 must not
    // be called, which is what keeps T4 cheaper than T3.
    let busy = ScriptedStep::tool_calls(vec![call(
        "1",
        "grep_text",
        serde_json::json!({ "query": ".", "path": "." }),
    )]);
    let provider = MockProvider::repeating(busy);

    let config = SessionConfig {
        planning_enabled: false,
        context_strategy: axolotl_context::ContextTier::T4,
        context_window: 4_000,
        max_turns: 8,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    for i in 0..30 {
        std::fs::write(h.dir.path().join(format!("f{i}.txt")), "needle\n".repeat(5)).unwrap();
    }
    let session = run(&h, &provider, Task::from_description("x"), &config);

    // The provider only serves tool calls, so any summarization call would be
    // recorded as an extra request beyond the loop turns.
    let turns = session.turn_count();
    let requests = provider.request_count();
    assert_eq!(
        requests, turns,
        "no summarization call expected: {requests} requests for {turns} turns"
    );
    assert_eq!(session.summarize_calls, 0);
}

#[test]
fn context_tracks_peak_usage() {
    let provider = MockProvider::new(vec![
        ScriptedStep::tool_calls(vec![call("1", "list_files", serde_json::json!({}))]),
        ScriptedStep::text("done"),
    ]);
    let config = config_without_planning();
    let h = harness(&provider, &config);
    let session = run(&h, &provider, Task::from_description("x"), &config);

    assert!(session.peak_context_tokens >= session.turns[0].input_tokens);
}
