//! Tests for trajectory analysis (paper §C.1).

use super::*;
use axolotl_core::{Session, SessionConfig, SessionOutcome, Task, TaskResult};
use axolotl_models::types::{ChatMessage, LLMResponse, StopReason, TokenUsage, ToolCall};

fn call(id: &str, name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args,
    }
}

/// Build a session with one turn per supplied call list.
fn session_with(turns: Vec<Vec<ToolCall>>, outcome: SessionOutcome) -> Session {
    let mut session = Session::new(Task::from_description("t"), SessionConfig::default());
    for (i, calls) in turns.into_iter().enumerate() {
        let response = LLMResponse {
            message: ChatMessage::assistant_with_tool_calls(None, calls),
            usage: TokenUsage::default(),
            stop_reason: StopReason::ToolUse,
            model: "m".into(),
        };
        let mut turn = axolotl_core::Turn::new(i as u32, vec![], 0, response);
        turn.set_duration(1);
        session.record_turn(turn);
    }
    session.complete(outcome);
    session
}

fn passed() -> SessionOutcome {
    SessionOutcome::Success(TaskResult {
        passed: true,
        test_output: String::new(),
        files_modified: vec![],
        error: None,
    })
}

fn failed() -> SessionOutcome {
    SessionOutcome::Failure(TaskResult {
        passed: false,
        test_output: String::new(),
        files_modified: vec![],
        error: Some("tests failed".into()),
    })
}

#[test]
fn classifies_structured_tools() {
    assert_eq!(
        classify_tool("read_file", &serde_json::json!({})),
        Phase::Localize
    );
    assert_eq!(
        classify_tool("grep_text", &serde_json::json!({})),
        Phase::Localize
    );
    assert_eq!(
        classify_tool("edit_file", &serde_json::json!({})),
        Phase::Fix
    );
    assert_eq!(
        classify_tool("update_plan", &serde_json::json!({})),
        Phase::Other
    );
}

#[test]
fn write_to_source_is_a_fix_but_a_repro_is_not() {
    assert_eq!(
        classify_tool("write_file", &serde_json::json!({ "path": "src/main.rs" })),
        Phase::Fix
    );
    assert_eq!(
        classify_tool(
            "write_file",
            &serde_json::json!({ "path": "repro_issue.py" })
        ),
        Phase::Reproduce
    );
}

#[test]
fn classifies_shell_commands() {
    let verify = serde_json::json!({ "command": "cargo test --all" });
    assert_eq!(classify_tool("bash", &verify), Phase::Verify);

    let localize = serde_json::json!({ "command": "grep -rn foo src/" });
    assert_eq!(classify_tool("bash", &localize), Phase::Localize);

    let repro = serde_json::json!({ "command": "python repro.py" });
    assert_eq!(classify_tool("bash", &repro), Phase::Reproduce);

    let other = serde_json::json!({ "command": "git commit -m x" });
    assert_eq!(classify_tool("bash", &other), Phase::Other);
}

#[test]
fn shell_redirection_counts_as_an_edit() {
    // Regression: a bash-only run edits with `echo x > f`, and the first
    // classifier scored every such command as "other", so bash-only runs
    // always reported zero fixes.
    for command in [
        "echo hello> hello.txt && type hello.txt",
        "printf 'x' > src/main.rs",
        "echo a >> log.txt",
    ] {
        assert_eq!(
            classify_tool("bash", &serde_json::json!({ "command": command })),
            Phase::Fix,
            "{command}"
        );
    }
}

#[test]
fn redirect_to_nul_is_not_an_edit() {
    for command in [
        "findstr hello hello.txt > NUL",
        "cargo test > nul",
        "echo ok 2> NUL",
    ] {
        assert_ne!(
            classify_tool("bash", &serde_json::json!({ "command": command })),
            Phase::Fix,
            "{command}"
        );
    }
}

#[test]
fn powershell_file_writes_count_as_edits() {
    let command = "powershell -NoProfile -Command \"Set-Content -Path 'a.txt' -Value 'x'\"";
    assert_eq!(
        classify_tool("bash", &serde_json::json!({ "command": command })),
        Phase::Fix
    );
}

#[test]
fn a_verification_command_with_a_redirect_is_still_verification() {
    // `cargo test > out.log` verifies; the redirect is incidental.
    assert_eq!(
        classify_tool(
            "bash",
            &serde_json::json!({ "command": "cargo test > out.log" })
        ),
        Phase::Verify
    );
}

#[test]
fn successful_run_has_no_failure_stage() {
    let session = session_with(
        vec![vec![call("1", "read_file", serde_json::json!({}))]],
        passed(),
    );
    assert_eq!(analyze(&session).failure_stage, FailureStage::None);
}

#[test]
fn never_localized_is_a_file_localization_failure() {
    let session = session_with(
        vec![vec![call("1", "update_plan", serde_json::json!({}))]],
        failed(),
    );
    assert_eq!(
        analyze(&session).failure_stage,
        FailureStage::FileLocalization
    );
}

#[test]
fn located_but_never_edited_is_line_localization() {
    let session = session_with(
        vec![vec![call("1", "read_file", serde_json::json!({}))]],
        failed(),
    );
    assert_eq!(
        analyze(&session).failure_stage,
        FailureStage::LineLocalization
    );
}

#[test]
fn edited_without_verifying_is_patch_implementation() {
    let session = session_with(
        vec![
            vec![call("1", "read_file", serde_json::json!({}))],
            vec![call("2", "edit_file", serde_json::json!({}))],
        ],
        failed(),
    );
    assert_eq!(
        analyze(&session).failure_stage,
        FailureStage::PatchImplementation
    );
}

#[test]
fn edited_and_verified_but_failed_is_a_verification_failure() {
    let session = session_with(
        vec![
            vec![call("1", "read_file", serde_json::json!({}))],
            vec![call("2", "edit_file", serde_json::json!({}))],
            vec![call(
                "3",
                "bash",
                serde_json::json!({ "command": "cargo test" }),
            )],
        ],
        failed(),
    );
    assert_eq!(analyze(&session).failure_stage, FailureStage::Verification);
}

#[test]
fn resource_exhaustion_is_reported_as_exhausted() {
    let session = session_with(
        vec![vec![call("1", "read_file", serde_json::json!({}))]],
        SessionOutcome::TurnLimitReached,
    );
    let analysis = analyze(&session);
    assert_eq!(analysis.failure_stage, FailureStage::Exhausted);
    assert_eq!(analysis.termination.as_deref(), Some("turn_limit"));
}

#[test]
fn overflow_is_recorded_as_termination() {
    let session = session_with(vec![], SessionOutcome::ContextOverflow);
    let analysis = analyze(&session);
    assert_eq!(analysis.termination.as_deref(), Some("context_overflow"));
}

#[test]
fn stuck_termination_names_the_tool() {
    let session = session_with(
        vec![],
        SessionOutcome::Stuck {
            tool: "read_file".into(),
            repeats: 8,
        },
    );
    assert_eq!(
        analyze(&session).termination.as_deref(),
        Some("stuck:read_filex8")
    );
}

#[test]
fn phase_counts_match_the_turns() {
    let session = session_with(
        vec![
            vec![call("1", "read_file", serde_json::json!({}))],
            vec![call("2", "edit_file", serde_json::json!({}))],
            vec![call(
                "3",
                "bash",
                serde_json::json!({ "command": "cargo test" }),
            )],
        ],
        passed(),
    );
    let analysis = analyze(&session);
    assert_eq!(analysis.phase_counts.localize, 1);
    assert_eq!(analysis.phase_counts.fix, 1);
    assert_eq!(analysis.phase_counts.verify, 1);
    assert_eq!(analysis.phases.len(), 3);
}

#[test]
fn summary_aggregates_stages_across_runs() {
    let a = analyze(&session_with(
        vec![vec![call("1", "update_plan", serde_json::json!({}))]],
        failed(),
    ));
    let b = analyze(&session_with(
        vec![vec![call("1", "read_file", serde_json::json!({}))]],
        failed(),
    ));
    let summary = AnalysisSummary::from_runs(&[a, b]);
    assert_eq!(summary.runs, 2);
    assert_eq!(summary.passed, 0);
    // One file-localization and one line-localization failure.
    assert_eq!(summary.by_stage.len(), 2);
}

#[test]
fn empty_summary_is_safe() {
    let summary = AnalysisSummary::from_runs(&[]);
    assert_eq!(summary.runs, 0);
    assert_eq!(summary.mean_turns, 0.0);
}
