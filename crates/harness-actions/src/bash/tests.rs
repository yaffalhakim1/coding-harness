//! Tests for the bash executor.

use super::*;
use crate::error::ActionError;
use crate::registry::{Action, ToolContext};

fn ctx(dir: &tempfile::TempDir) -> ToolContext {
    ToolContext::new(crate::workspace::Workspace::open(dir.path()).unwrap())
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

#[test]
fn reports_stdout_without_exit_code_on_success() {
    let dir = tempfile::tempdir().unwrap();
    let tool = BashExecutor::new(ctx(&dir), false);
    let out = rt()
        .block_on(tool.execute(serde_json::json!({ "command": "echo hi" })))
        .unwrap();
    assert!(out.contains("hi"), "{out}");
    assert!(
        !out.contains("Exit code"),
        "success must not report a code: {out}"
    );
}

#[test]
fn nonzero_exit_is_a_tool_error_carrying_output() {
    // Paper Fig. 27: non-zero exits surface as is_error=True so the model
    // cannot mistake a failing command for a passing one.
    let dir = tempfile::tempdir().unwrap();
    let tool = BashExecutor::new(ctx(&dir), false);
    let err = rt()
        .block_on(tool.execute(serde_json::json!({ "command": "exit 3" })))
        .unwrap_err();
    match err {
        ActionError::CommandFailed { code, output } => {
            assert_eq!(code, 3);
            assert!(output.contains("Exit code: 3"), "{output}");
        }
        other => panic!("expected CommandFailed, got {other:?}"),
    }
}

#[test]
fn stderr_is_labelled() {
    let dir = tempfile::tempdir().unwrap();
    let tool = BashExecutor::new(ctx(&dir), true);
    let out = rt()
        .block_on(tool.execute(serde_json::json!({ "command": "echo oops 1>&2" })))
        .unwrap();
    assert!(out.contains("STDERR:"), "{out}");
    assert!(out.contains("oops"), "{out}");
}

#[test]
fn empty_output_is_stated_explicitly() {
    let dir = tempfile::tempdir().unwrap();
    let tool = BashExecutor::new(ctx(&dir), false);
    // `exit 0` succeeds silently; the model must be told that explicitly
    // rather than receiving an empty string.
    let out = rt()
        .block_on(tool.execute(serde_json::json!({ "command": "exit 0" })))
        .unwrap();
    assert!(out.contains("(no output)"), "{out}");
}

#[test]
fn times_out() {
    let dir = tempfile::tempdir().unwrap();
    let tool = BashExecutor::new(ctx(&dir), false);
    let sleep = if cfg!(windows) {
        "ping -n 6 127.0.0.1 > nul"
    } else {
        "sleep 5"
    };
    let err = rt()
        .block_on(tool.execute(serde_json::json!({ "command": sleep, "timeout_seconds": 1 })))
        .unwrap_err();
    assert!(
        matches!(err, ActionError::Timeout { seconds: 1, .. }),
        "{err:?}"
    );
}

#[test]
fn cwd_must_stay_inside_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let tool = BashExecutor::new(ctx(&dir), false);
    let err = rt()
        .block_on(tool.execute(serde_json::json!({ "command": "echo x", "cwd": ".." })))
        .unwrap_err();
    assert!(matches!(err, ActionError::PathEscape(_)), "{err:?}");
}

#[test]
fn runs_in_the_workspace_by_default() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
    let tool = BashExecutor::new(ctx(&dir), false);
    let list = if cfg!(windows) { "dir /b" } else { "ls" };
    let out = rt()
        .block_on(tool.execute(serde_json::json!({ "command": list })))
        .unwrap();
    assert!(out.contains("marker.txt"), "{out}");
}

#[test]
fn output_truncation_respects_char_boundaries() {
    // Regression: the old implementation sliced by byte index and panicked on
    // multi-byte characters.
    let dir = tempfile::tempdir().unwrap();
    let tool = BashExecutor::new(ctx(&dir).with_max_output_chars(20), false);
    let out = rt()
        .block_on(
            tool.execute(serde_json::json!({ "command": "echo ααααααααααααααααααααααααααααα" })),
        )
        .unwrap();
    assert!(out.contains("truncated"), "{out}");
}

#[test]
fn bash_only_description_differs_from_predefined() {
    let dir = tempfile::tempdir().unwrap();
    let predefined = BashExecutor::new(ctx(&dir), false).definition();
    let bash_only = BashExecutor::new(ctx(&dir), true).definition();
    assert!(bash_only.description.contains("ONLY tool"));
    assert!(!predefined.description.contains("ONLY tool"));
    assert!(predefined.description.contains("read_file"));
}

#[test]
fn missing_command_argument_is_a_tool_error() {
    let dir = tempfile::tempdir().unwrap();
    let err = rt()
        .block_on(BashExecutor::new(ctx(&dir), false).execute(serde_json::json!({})))
        .unwrap_err();
    assert!(matches!(err, ActionError::InvalidArgument(_)), "{err:?}");
}
