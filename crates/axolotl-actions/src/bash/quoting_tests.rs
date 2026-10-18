//! Regression tests for shell quoting on the bash tool.
//!
//! Windows cmd.exe strips one layer of `"` from a `/C` argument, so a command
//! containing quoted text (a common pattern for exact-match searches) must
//! survive the round trip intact.

use super::*;

fn ctx(dir: &tempfile::TempDir) -> ToolContext {
    ToolContext::new(crate::workspace::Workspace::open(dir.path()).unwrap())
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

#[cfg(windows)]
#[test]
fn quoted_arguments_survive_the_shell() {
    // `findstr /x` matches whole lines and requires CRLF line endings, so the
    // fixture must be written in binary with \r\n. (A file written with plain
    // \n makes findstr return exit 1 regardless of correct quoting.)
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("c.txt"), b"enabled=true\r\n").unwrap();
    let tool = BashExecutor::new(ctx(&dir), false);

    let command = "findstr /x /c:\"enabled=true\" c.txt";
    match rt().block_on(tool.execute(serde_json::json!({ "command": command }))) {
        Ok(out) => assert!(out.contains("enabled=true"), "{out}"),
        Err(e) => panic!("quoted command failed: {e}"),
    }
}

#[cfg(windows)]
#[test]
fn quoted_arguments_that_do_not_match_still_fail_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("c.txt"), b"enabled=false\r\n").unwrap();
    let tool = BashExecutor::new(ctx(&dir), false);

    let command = "findstr /x /c:\"enabled=true\" c.txt";
    let err = rt()
        .block_on(tool.execute(serde_json::json!({ "command": command })))
        .unwrap_err();
    assert!(
        matches!(err, ActionError::CommandFailed { code: 1, .. }),
        "expected findstr's exit code 1, got {err:?}"
    );
}

#[cfg(windows)]
#[test]
fn echo_with_quotes_writes_the_quoted_text() {
    let dir = tempfile::tempdir().unwrap();
    let tool = BashExecutor::new(ctx(&dir), false);
    rt().block_on(tool.execute(serde_json::json!({
        "command": "echo \"hello world\" > out.txt"
    })))
    .unwrap();
    let content = std::fs::read_to_string(dir.path().join("out.txt")).unwrap();
    assert!(content.contains("hello world"), "{content:?}");
}
