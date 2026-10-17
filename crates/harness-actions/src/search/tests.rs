//! Tests for glob/grep and the bash executor.

use super::*;
use crate::registry::Action;

fn ctx(dir: &tempfile::TempDir) -> ToolContext {
    ToolContext::new(crate::workspace::Workspace::open(dir.path()).unwrap())
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

fn seed(dir: &tempfile::TempDir) {
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(
        dir.path().join("src/lib.rs"),
        "pub fn add(a: i32) -> i32 { a }\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("README.md"), "# Title\n").unwrap();
    std::fs::create_dir_all(dir.path().join("node_modules")).unwrap();
    std::fs::write(dir.path().join("node_modules/x.rs"), "ignored\n").unwrap();
}

#[test]
fn glob_finds_nested_rust_files() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir);
    let out = rt()
        .block_on(
            SearchFilesTool::new(ctx(&dir)).execute(serde_json::json!({ "pattern": "**/*.rs" })),
        )
        .unwrap();
    let normalized = out.replace('\\', "/");
    assert!(normalized.contains("src/main.rs"), "{out}");
    assert!(normalized.contains("src/lib.rs"), "{out}");
    assert!(
        !normalized.contains("node_modules"),
        "ignored dir leaked: {out}"
    );
}

#[test]
fn glob_single_star_stays_within_segment() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir);
    let out = rt()
        .block_on(SearchFilesTool::new(ctx(&dir)).execute(serde_json::json!({ "pattern": "*.md" })))
        .unwrap();
    assert!(out.contains("README.md"), "{out}");
    assert!(!out.contains("main.rs"), "{out}");
}

#[test]
fn glob_no_match_is_not_an_error() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir);
    let out = rt()
        .block_on(
            SearchFilesTool::new(ctx(&dir)).execute(serde_json::json!({ "pattern": "**/*.zig" })),
        )
        .unwrap();
    assert_eq!(out, "No files found matching pattern.");
}

#[test]
fn glob_invalid_pattern_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir);
    let err = rt()
        .block_on(
            SearchFilesTool::new(ctx(&dir)).execute(serde_json::json!({ "pattern": "[unclosed" })),
        )
        .unwrap_err();
    assert!(matches!(err, ActionError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn glob_bracket_class_is_honoured() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir);
    let out = rt()
        .block_on(
            SearchFilesTool::new(ctx(&dir))
                .execute(serde_json::json!({ "pattern": "src/*.r[as]" })),
        )
        .unwrap();
    assert!(out.contains("lib.rs"), "{out}");
    assert!(out.contains("main.rs"), "{out}");
}

#[test]
fn grep_returns_path_line_and_text() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir);
    let out = rt()
        .block_on(
            GrepTool::new(ctx(&dir))
                .execute(serde_json::json!({ "query": "pub fn", "include": "*.rs" })),
        )
        .unwrap();
    let normalized = out.replace('\\', "/");
    assert!(normalized.contains("src/lib.rs:1: pub fn add"), "{out}");
    assert_eq!(
        out.lines().count(),
        1,
        "README.md should be filtered: {out}"
    );
}

#[test]
fn grep_case_insensitive_flag() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir);
    let tool = GrepTool::new(ctx(&dir));
    let exact = rt()
        .block_on(tool.execute(serde_json::json!({ "query": "title" })))
        .unwrap();
    assert_eq!(exact, "No matches found.");

    let loose = rt()
        .block_on(tool.execute(serde_json::json!({ "query": "title", "case_sensitive": false })))
        .unwrap();
    assert!(loose.contains("# Title"), "{loose}");
}

#[test]
fn grep_invalid_regex_is_a_tool_error() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir);
    let err = rt()
        .block_on(GrepTool::new(ctx(&dir)).execute(serde_json::json!({ "query": "(" })))
        .unwrap_err();
    assert!(matches!(err, ActionError::InvalidRegex(_)), "{err:?}");
}

#[test]
fn grep_invalid_include_glob_is_a_tool_error_not_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    seed(&dir);
    let err = rt()
        .block_on(
            GrepTool::new(ctx(&dir))
                .execute(serde_json::json!({ "query": "x", "include": "[bad" })),
        )
        .unwrap_err();
    assert!(matches!(err, ActionError::InvalidArgument(_)), "{err:?}");
}

#[test]
fn grep_translates_glob_metacharacters_are_not_treated_as_path_escapes() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.rs"), "needle\n").unwrap();
    std::fs::write(dir.path().join("ab.rs"), "needle\n").unwrap();
    let out = rt()
        .block_on(
            GrepTool::new(ctx(&dir))
                .execute(serde_json::json!({ "query": "needle", "include": "a.rs" })),
        )
        .unwrap();
    // `a.rs` must match only a.rs, not ab.rs (the old glob rewrite matched both).
    assert_eq!(out.lines().count(), 1, "{out}");
}
