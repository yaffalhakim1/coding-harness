//! End-to-end agent loop tests driven by a scripted provider.
//!
//! These exercise the whole stack — context management, tool execution, turn
//! bookkeeping, evaluation — with no network access, which is what makes them
//! runnable in CI.
//!
//! Context-tier behaviour lives in `context_tiers.rs`.

mod common;

use axolotl_actions::registry::ActionRegistry;
use axolotl_core::{ActionSpace, SessionConfig, SessionOutcome, Task};
use axolotl_models::mock::{MockProvider, ScriptedStep};
use common::{call, config_without_planning, harness, run};
#[test]
fn solves_a_file_edit_task() {
    let dir_task_check = if cfg!(windows) {
        "type greeting.txt"
    } else {
        "cat greeting.txt"
    };
    let mut task = Task::from_description("Create greeting.txt containing hello world");
    task.test_commands.push(format!(
        "{dir_task_check} && findstr hello greeting.txt > NUL || grep -q hello greeting.txt"
    ));

    let provider = MockProvider::new(vec![
        ScriptedStep::tool_calls(vec![call(
            "1",
            "write_file",
            serde_json::json!({ "path": "greeting.txt", "content": "hello world\n" }),
        )]),
        ScriptedStep::text("Created greeting.txt and verified its contents."),
    ]);

    let config = config_without_planning();
    let h = harness(&provider, &config);
    let session = run(&h, &provider, task, &config);

    assert!(
        session
            .outcome
            .as_ref()
            .is_some_and(SessionOutcome::is_success),
        "outcome: {:?}",
        session.outcome
    );
    assert_eq!(
        std::fs::read_to_string(h.dir.path().join("greeting.txt")).unwrap(),
        "hello world\n"
    );
    // Two model turns: one action, one final summary.
    assert_eq!(
        provider.request_count(),
        2,
        "model called unexpectedly often; turns={}",
        session.turn_count()
    );
    assert_eq!(session.turn_count(), 2);
}

#[test]
fn requires_read_before_edit() {
    let mut task = Task::from_description("change the value in config.txt");
    let check = if cfg!(windows) {
        "findstr value=2 config.txt > NUL"
    } else {
        "grep -q 'value=2' config.txt"
    };
    task.test_commands.push(check.to_string());

    let provider = MockProvider::new(vec![
        // The model tries to edit without reading: the harness must reject it.
        ScriptedStep::tool_calls(vec![call(
            "1",
            "edit_file",
            serde_json::json!({ "path": "config.txt", "old_text": "value=1", "new_text": "value=2" }),
        )]),
        // Then it reads and retries.
        ScriptedStep::tool_calls(vec![call(
            "2",
            "read_file",
            serde_json::json!({ "path": "config.txt" }),
        )]),
        ScriptedStep::tool_calls(vec![call(
            "3",
            "edit_file",
            serde_json::json!({ "path": "config.txt", "old_text": "value=1", "new_text": "value=2" }),
        )]),
        ScriptedStep::text("Updated config.txt."),
    ]);

    let config = config_without_planning();
    let h = harness(&provider, &config);
    std::fs::write(h.dir.path().join("config.txt"), "value=1\n").unwrap();

    let session = run(&h, &provider, task, &config);

    assert!(
        session
            .outcome
            .as_ref()
            .is_some_and(SessionOutcome::is_success),
        "outcome: {:?}",
        session.outcome
    );
    // The first edit failed and was reported back as an observation.
    let first = &session.turns[0];
    assert!(!first.tool_results[0].success);
    assert!(first.tool_results[0]
        .error
        .as_deref()
        .unwrap()
        .contains("has not been read"));
    assert_eq!(
        std::fs::read_to_string(h.dir.path().join("config.txt")).unwrap(),
        "value=2\n"
    );
}

#[test]
fn prose_without_action_ends_the_run() {
    let mut task = Task::from_description("create a.txt with content");
    let check = if cfg!(windows) {
        "type a.txt"
    } else {
        "cat a.txt"
    };
    task.test_commands.push(check.to_string());

    // The model declares completion without acting: the task must be judged a
    // failure, not silently retried until the budget runs out.
    let provider = MockProvider::new(vec![ScriptedStep::text("I am done.")]);

    let config = config_without_planning();
    let h = harness(&provider, &config);
    let session = run(&h, &provider, task, &config);

    assert!(
        matches!(session.outcome, Some(SessionOutcome::Failure(_))),
        "outcome: {:?}",
        session.outcome
    );
    assert_eq!(provider.request_count(), 1, "must not re-poll the model");
    assert!(!h.dir.path().join("a.txt").exists());
}

#[test]
fn action_then_prose_completes_the_run() {
    let mut task = Task::from_description("create a.txt with content");
    let check = if cfg!(windows) {
        "type a.txt"
    } else {
        "cat a.txt"
    };
    task.test_commands.push(check.to_string());

    let provider = MockProvider::new(vec![
        ScriptedStep::tool_calls(vec![call(
            "1",
            "write_file",
            serde_json::json!({ "path": "a.txt", "content": "done\n" }),
        )]),
        ScriptedStep::text("Finished."),
    ]);

    let config = config_without_planning();
    let h = harness(&provider, &config);
    let session = run(&h, &provider, task, &config);

    assert!(
        session
            .outcome
            .as_ref()
            .is_some_and(SessionOutcome::is_success),
        "outcome: {:?}",
        session.outcome
    );
    assert_eq!(provider.request_count(), 2);
}

#[test]
fn empty_responses_are_retried_then_abandoned() {
    // A model that returns nothing at all is degenerate: retry a bounded number
    // of times, then stop rather than burning the whole turn budget.
    let provider = MockProvider::new(vec![
        ScriptedStep::text(""),
        ScriptedStep::text(""),
        ScriptedStep::text(""),
    ]);
    let config = SessionConfig {
        planning_enabled: false,
        max_narration_turns: 3,
        max_turns: 50,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let session = run(&h, &provider, Task::from_description("x"), &config);

    assert!(
        matches!(session.outcome, Some(SessionOutcome::Unverified(_))),
        "outcome: {:?}",
        session.outcome
    );
    assert_eq!(session.turn_count(), 3);
    assert_eq!(provider.request_count(), 3);
}

#[test]
fn failed_verification_is_a_failure_not_a_success() {
    let provider = MockProvider::new(vec![ScriptedStep::text("All done!")]);
    let mut task = Task::from_description("fix it");
    task.test_commands.push(if cfg!(windows) {
        "exit 1".into()
    } else {
        "false".into()
    });

    let config = config_without_planning();
    let h = harness(&provider, &config);
    let session = run(&h, &provider, task, &config);

    assert!(
        matches!(session.outcome, Some(SessionOutcome::Failure(_))),
        "outcome: {:?}",
        session.outcome
    );
}

#[test]
fn setup_commands_run_before_the_agent() {
    let provider = MockProvider::new(vec![ScriptedStep::text("done")]);
    let mut task = Task::from_description("inspect seeded.txt");
    task.setup_commands.push("echo seeded > seeded.txt".into());

    let config = config_without_planning();
    let h = harness(&provider, &config);
    let session = run(&h, &provider, task, &config);

    assert!(matches!(
        session.outcome,
        Some(SessionOutcome::Unverified(_))
    ));
    assert!(h.dir.path().join("seeded.txt").exists());
}

#[test]
fn setup_failure_aborts_the_session() {
    let provider = MockProvider::new(vec![ScriptedStep::text("done")]);
    let mut task = Task::from_description("x");
    task.setup_commands.push(if cfg!(windows) {
        "exit 9".into()
    } else {
        "false".into()
    });

    let config = config_without_planning();
    let h = harness(&provider, &config);
    let session = run(&h, &provider, task, &config);

    assert!(matches!(session.outcome, Some(SessionOutcome::Error(_))));
    // No model call should have been made for a broken environment.
    assert_eq!(provider.request_count(), 0);
}

#[test]
fn planning_runs_once_before_the_loop() {
    let mut task = Task::from_description("create b.txt");
    task.test_commands.push(if cfg!(windows) {
        "type b.txt".into()
    } else {
        "cat b.txt".into()
    });

    let provider = MockProvider::new(vec![
        ScriptedStep::text("1. Write the file\n2. Verify it"),
        ScriptedStep::tool_calls(vec![call(
            "1",
            "write_file",
            serde_json::json!({ "path": "b.txt", "content": "x\n" }),
        )]),
        ScriptedStep::text("done"),
    ]);
    let config = SessionConfig::default();
    let h = harness(&provider, &config);
    let session = run(&h, &provider, task, &config);

    assert_eq!(session.turn_count(), 2, "planning must not count as a turn");
    assert_eq!(
        session.plan.as_deref(),
        Some("1. Write the file\n2. Verify it")
    );
    // 1 planning call + 2 loop calls.
    assert_eq!(provider.request_count(), 3);

    // The plan must be visible to the model in every loop request.
    let requests = provider.requests();
    assert!(requests[1][2].text().contains("Write the file"));
}

#[test]
fn stuck_detection_terminates_a_failing_loop() {
    let provider = MockProvider::repeating(ScriptedStep::tool_calls(vec![call(
        "same",
        "read_file",
        serde_json::json!({ "path": "does-not-exist.txt" }),
    )]));

    let config = SessionConfig {
        planning_enabled: false,
        max_turns: 30,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let session = run(&h, &provider, Task::from_description("x"), &config);

    match &session.outcome {
        Some(SessionOutcome::Stuck { tool, repeats }) => {
            assert_eq!(tool, "read_file");
            assert_eq!(*repeats, 8);
        }
        other => panic!("expected stuck outcome, got {other:?}"),
    }
    // 8 identical calls, then termination — not the 30-turn budget.
    assert_eq!(session.turn_count(), 8);
}

#[test]
fn bash_only_action_space_exposes_shell_and_auxiliary_tools() {
    let provider = MockProvider::new(vec![ScriptedStep::text("done")]);
    let config = SessionConfig {
        planning_enabled: false,
        action_space: ActionSpace::BashOnly,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);

    // Paper §2.2: the bash-only condition "removes the predefined workspace
    // tools while retaining bash and the auxiliary tools required by the
    // enabled planning and context-management components."
    assert!(h.actions.has_tool("bash"));
    assert!(
        h.actions.has_tool("recall_event"),
        "T4 retains recall_event"
    );
    assert!(!h.actions.has_tool("read_file"));
    assert!(!h.actions.has_tool("write_file"));
    assert!(!h.actions.has_tool("grep_text"));
    assert!(!h.actions.has_tool("web_fetch"));
    assert_eq!(h.actions.get_tool_definitions().len(), 2);
}

#[test]
fn token_budget_stops_the_run() {
    let busy = ScriptedStep::tool_calls(vec![call(
        "1",
        "list_files",
        serde_json::json!({ "path": "." }),
    )]);
    let provider = MockProvider::repeating(busy);

    let config = SessionConfig {
        planning_enabled: false,
        max_turns: 100,
        token_budget: Some(500),
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let session = run(&h, &provider, Task::from_description("x"), &config);

    assert!(matches!(
        session.outcome,
        Some(SessionOutcome::TurnLimitReached)
    ));
    assert!(session.total_usage.total_tokens >= 500);
}

#[test]
fn turn_limit_is_enforced() {
    let busy = ScriptedStep::tool_calls(vec![call(
        "1",
        "list_files",
        serde_json::json!({ "path": "." }),
    )]);
    let provider = MockProvider::repeating(busy);

    let config = SessionConfig {
        planning_enabled: false,
        max_turns: 3,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let session = run(&h, &provider, Task::from_description("x"), &config);
    assert!(matches!(
        session.outcome,
        Some(SessionOutcome::TurnLimitReached)
    ));
    assert_eq!(session.turn_count(), 3);
}

#[test]
fn api_error_is_reported_not_panicked() {
    let provider = MockProvider::new(vec![ScriptedStep::Fail("connection refused".into())]);
    let config = config_without_planning();
    let h = harness(&provider, &config);
    let session = run(&h, &provider, Task::from_description("x"), &config);

    match session.outcome {
        Some(SessionOutcome::ApiError(msg)) => assert!(msg.contains("connection refused")),
        other => panic!("expected ApiError, got {other:?}"),
    }
}
