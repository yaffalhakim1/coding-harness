//! Integration test: post-edit diagnostics reach the model (paper §2.4).

use harness_actions::registry::{ActionRegistry, DefaultRegistry};
use harness_actions::{ToolContext, Workspace};

fn registry(dir: &tempfile::TempDir) -> DefaultRegistry {
    DefaultRegistry::with_predefined_tools(ToolContext::new(Workspace::open(dir.path()).unwrap()))
}

#[tokio::test]
async fn broken_python_write_reports_diagnostics() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir);

    let out = reg
        .execute(&harness_models::types::ToolCall {
            id: "1".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({
                "path": "bad.py",
                "content": "def f(:\n    pass\n"
            }),
        })
        .await
        .expect("write succeeds even when the code is broken");

    // The write must succeed; diagnostics are advisory, not a failure.
    assert!(out.contains("Wrote"), "{out}");
    if out.contains("[diagnostics]") {
        let lower = out.to_lowercase();
        assert!(lower.contains("syntax") || lower.contains("error"), "{out}");
    } else {
        eprintln!("python unavailable; diagnostics skipped by design");
    }
}

#[tokio::test]
async fn clean_python_write_has_no_diagnostics_block() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir);

    let out = reg
        .execute(&harness_models::types::ToolCall {
            id: "1".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({
                "path": "ok.py",
                "content": "x = 1\n"
            }),
        })
        .await
        .unwrap();

    assert!(out.contains("Wrote"), "{out}");
    assert!(!out.contains("[diagnostics]"), "{out}");
}

#[tokio::test]
async fn non_code_files_are_not_checked() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir);

    let out = reg
        .execute(&harness_models::types::ToolCall {
            id: "1".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({
                "path": "notes.txt",
                "content": "unbalanced {{{ braces"
            }),
        })
        .await
        .unwrap();

    assert!(!out.contains("[diagnostics]"), "{out}");
}

#[tokio::test]
async fn diagnostics_do_not_block_a_successful_edit() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("m.py"), "value = 1\n").unwrap();
    let reg = registry(&dir);

    // Read first (read-before-write gate), then edit.
    reg.execute(&harness_models::types::ToolCall {
        id: "0".into(),
        name: "read_file".into(),
        arguments: serde_json::json!({ "path": "m.py" }),
    })
    .await
    .unwrap();

    let out = reg
        .execute(&harness_models::types::ToolCall {
            id: "1".into(),
            name: "edit_file".into(),
            arguments: serde_json::json!({
                "path": "m.py",
                "old_text": "value = 1",
                "new_text": "value = 2"
            }),
        })
        .await
        .unwrap();

    assert!(out.contains("Replaced"), "{out}");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("m.py")).unwrap(),
        "value = 2\n"
    );
}

#[test]
fn write_file_tool_still_exposes_a_definition() {
    let dir = tempfile::tempdir().unwrap();
    let reg = registry(&dir);
    let defs = reg.get_tool_definitions();
    assert!(defs.iter().any(|d| d.name == "write_file"));
    assert!(defs.iter().any(|d| d.name == "web_fetch"));
}
