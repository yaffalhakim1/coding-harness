//! Tests for the benchmark batch runner.

use super::*;
use axolotl_core::{SessionConfig, Task};

fn task(id: &str) -> BenchmarkTask {
    BenchmarkTask {
        id: id.into(),
        benchmark: "terminal-bench".into(),
        description: "d".into(),
        repo_url: None,
        base_commit: None,
        setup_commands: vec![],
        test_commands: vec![],
        expected_files: vec![],
        environment: None,
    }
}

fn session_with(outcome: SessionOutcome) -> Session {
    let mut s = Session::new(Task::from_description("x"), SessionConfig::default());
    s.observe_context(1234);
    s.complete(outcome);
    s
}

#[test]
fn outcome_from_successful_session() {
    let result = axolotl_core::TaskResult {
        passed: true,
        test_output: String::new(),
        files_modified: vec![],
        error: None,
    };
    let session = session_with(SessionOutcome::Success(result));
    let outcome = TaskOutcome::from_session(&task("t1"), &session);
    assert!(outcome.passed);
    assert_eq!(outcome.outcome, "success");
    assert_eq!(outcome.peak_context_tokens, 1234);
}

#[test]
fn outcome_from_failure_records_the_reason() {
    let result = axolotl_core::TaskResult {
        passed: false,
        test_output: String::new(),
        files_modified: vec![],
        error: Some("tests failed".into()),
    };
    let session = session_with(SessionOutcome::Failure(result));
    let outcome = TaskOutcome::from_session(&task("t1"), &session);
    assert!(!outcome.passed);
    assert_eq!(outcome.error.as_deref(), Some("tests failed"));
}

#[test]
fn skipped_marks_the_task_errored() {
    let outcome = TaskOutcome::skipped(&task("t1"), "setup failed");
    assert_eq!(outcome.outcome, "skipped");
    assert!(!outcome.passed);
    assert!(outcome.error.unwrap().contains("setup failed"));
}

#[test]
fn summary_aggregates_counts_and_means() {
    let config = SessionConfig::default();
    let results = vec![
        TaskOutcome::skipped(&task("a"), "x"),
        TaskOutcome::skipped(&task("b"), "y"),
    ];
    let report = SweepReport::build(&config, "terminal-bench", results, 5);
    assert_eq!(report.summary.tasks, 2);
    assert_eq!(report.summary.passed, 0);
    assert_eq!(report.summary.errored, 2);
    assert_eq!(report.summary.success_rate(), 0.0);
    assert_eq!(report.summary.mean_tokens(), 0);
}

#[test]
fn success_rate_is_fraction() {
    let mut s = SweepSummary {
        tasks: 4,
        passed: 3,
        ..SweepSummary::default()
    };
    assert!((s.success_rate() - 0.75).abs() < 1e-9);
    s.tasks = 0;
    assert_eq!(s.success_rate(), 0.0);
}

#[test]
fn cost_is_absent_unless_known() {
    let summary = SweepSummary {
        tasks: 2,
        ..SweepSummary::default()
    };
    assert_eq!(summary.mean_cost(), None);
}

#[test]
fn csv_has_a_header_and_one_row_per_task() {
    let config = SessionConfig::default();
    let results = vec![
        TaskOutcome::skipped(&task("a"), "x"),
        TaskOutcome::skipped(&task("b"), "y"),
    ];
    let report = SweepReport::build(&config, "terminal-bench", results, 1);
    let csv = report.to_csv();
    assert_eq!(csv.lines().count(), 3);
    assert!(csv.starts_with("task_id,outcome"));
}

#[tokio::test]
async fn workspace_is_cleared_between_runs() {
    let root = tempfile::tempdir().unwrap();
    let dir = prepare_workspace(root.path(), "t1").await.unwrap();
    std::fs::write(dir.join("stale.txt"), "old").unwrap();

    let again = prepare_workspace(root.path(), "t1").await.unwrap();
    assert!(
        !again.join("stale.txt").exists(),
        "stale files must be removed"
    );
}

#[test]
fn sanitize_keeps_ids_filesystem_safe() {
    assert_eq!(sanitize("a/b:c*d"), "a_b_c_d");
    assert_eq!(sanitize("swe-bench-1234"), "swe-bench-1234");
}

#[tokio::test]
async fn report_round_trips_through_json() {
    let config = SessionConfig::default();
    let report = SweepReport::build(
        &config,
        "terminal-bench",
        vec![TaskOutcome::skipped(&task("a"), "x")],
        2,
    );
    let dir = tempfile::tempdir().unwrap();
    let path = report.write(dir.path()).await.unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    let parsed: SweepReport = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed.summary.tasks, 1);
    assert_eq!(parsed.results[0].task_id, "a");
}

#[tokio::test]
async fn report_filename_distinguishes_action_space() {
    // Regression: `T4/plan/predefined` and `T4/plan/bash-only` at the same
    // window used to collide on one filename, silently overwriting results.
    let dir = tempfile::tempdir().unwrap();
    let mut config = SessionConfig {
        action_space: axolotl_core::ActionSpace::Predefined,
        ..SessionConfig::default()
    };
    let a = SweepReport::build(&config, "terminal-bench", vec![], 1);
    let path_a = a.write(dir.path()).await.unwrap();

    config.action_space = axolotl_core::ActionSpace::BashOnly;
    let b = SweepReport::build(&config, "terminal-bench", vec![], 1);
    let path_b = b.write(dir.path()).await.unwrap();

    assert_ne!(path_a, path_b, "settings must not overwrite each other");
}
