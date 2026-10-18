//! Tests for the directory listing tool.

use super::*;

fn ctx(dir: &tempfile::TempDir) -> ToolContext {
    ToolContext::new(crate::workspace::Workspace::open(dir.path()).unwrap())
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

#[test]
fn list_files_is_sorted_and_marks_directories() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("b.txt"), "x").unwrap();
    std::fs::write(dir.path().join("a.txt"), "y").unwrap();
    std::fs::create_dir(dir.path().join("src")).unwrap();

    let out = rt()
        .block_on(ListDirTool::new(ctx(&dir)).execute(serde_json::json!({})))
        .unwrap();
    let lines: Vec<_> = out.lines().collect();
    assert_eq!(lines[0], "a.txt (1 bytes)");
    assert_eq!(lines[1], "b.txt (1 bytes)");
    assert_eq!(lines[2], "src/");
}

#[test]
fn list_files_requires_directory() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "x").unwrap();
    let err = rt()
        .block_on(ListDirTool::new(ctx(&dir)).execute(serde_json::json!({ "path": "a.txt" })))
        .unwrap_err();
    assert!(
        matches!(err, crate::error::ActionError::NotADirectory { .. }),
        "{err:?}"
    );
}

#[test]
fn list_files_errors_for_missing_path() {
    let dir = tempfile::tempdir().unwrap();
    let err = rt()
        .block_on(ListDirTool::new(ctx(&dir)).execute(serde_json::json!({ "path": "nope" })))
        .unwrap_err();
    assert!(
        matches!(err, crate::error::ActionError::NotFound { .. }),
        "{err:?}"
    );
}

#[test]
fn list_files_recursive_nests_subdirectories() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("src/deep")).unwrap();
    std::fs::write(dir.path().join("src/deep/f.rs"), "x").unwrap();

    let out = rt()
        .block_on(ListDirTool::new(ctx(&dir)).execute(serde_json::json!({ "recursive": true })))
        .unwrap();
    assert!(out.contains("f.rs"), "{out}");
    assert!(
        out.lines().any(|l| l.starts_with("  ")),
        "expected indentation: {out}"
    );
}

#[test]
fn list_files_skips_ignored_directories() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("node_modules")).unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();

    let out = rt()
        .block_on(ListDirTool::new(ctx(&dir)).execute(serde_json::json!({ "recursive": true })))
        .unwrap();
    assert!(!out.contains("node_modules"), "{out}");
    assert!(out.contains("src/"), "{out}");
}

#[test]
fn list_files_honours_max_entries() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..20 {
        std::fs::write(dir.path().join(format!("f{i}.txt")), "x").unwrap();
    }
    let out = rt()
        .block_on(ListDirTool::new(ctx(&dir)).execute(serde_json::json!({ "max_entries": 5 })))
        .unwrap();
    assert!(out.contains("entry limit reached"), "{out}");
}

#[test]
fn empty_directory_is_reported_clearly() {
    let dir = tempfile::tempdir().unwrap();
    let out = rt()
        .block_on(ListDirTool::new(ctx(&dir)).execute(serde_json::json!({})))
        .unwrap();
    assert!(out.contains("is empty"), "{out}");
}
