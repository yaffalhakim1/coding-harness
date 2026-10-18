//! Tests for the file operation tools.

use super::*;

fn ctx(dir: &tempfile::TempDir) -> ToolContext {
    ToolContext::new(crate::workspace::Workspace::open(dir.path()).unwrap())
}

fn write(dir: &tempfile::TempDir, name: &str, content: &str) {
    std::fs::write(dir.path().join(name), content).unwrap();
}

fn read_all(tool: &ReadFileTool, dir: &tempfile::TempDir, name: &str) -> String {
    let _ = dir;
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(tool.execute(serde_json::json!({ "path": name })))
        .unwrap()
}

#[test]
fn read_returns_numbered_lines() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "first\nsecond\n");
    let tool = ReadFileTool::new(ctx(&dir));
    let out = read_all(&tool, &dir, "a.txt");
    assert!(out.contains("lines 1-2 of 2"), "{out}");
    assert!(out.contains("     1\tfirst"), "{out}");
}

#[test]
fn read_respects_offset_and_limit() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "l1\nl2\nl3\nl4\n");
    let tool = ReadFileTool::new(ctx(&dir));
    let out = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(tool.execute(serde_json::json!({ "path": "a.txt", "offset": 2, "limit": 2 })))
        .unwrap();
    assert!(out.contains("lines 2-3 of 4"), "{out}");
    assert!(!out.contains("\tl1"), "{out}");
}

#[test]
fn full_read_then_edit_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "hello world\n");
    let c = ctx(&dir);
    let reader = ReadFileTool::new(c.clone());
    let editor = EditFileTool::new(c);

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(reader.execute(serde_json::json!({ "path": "a.txt" })))
        .unwrap();
    rt.block_on(
        editor.execute(
            serde_json::json!({ "path": "a.txt", "old_text": "world", "new_text": "there" }),
        ),
    )
    .unwrap();

    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "hello there\n"
    );
}

#[test]
fn edit_without_read_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "x\n");
    let editor = EditFileTool::new(ctx(&dir));
    let err = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(
            editor
                .execute(serde_json::json!({ "path": "a.txt", "old_text": "x", "new_text": "y" })),
        )
        .unwrap_err();
    assert!(
        matches!(err, ActionError::ReadBeforeWrite { .. }),
        "{err:?}"
    );
}

#[test]
fn partial_read_does_not_authorise_edit() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "1\n2\n3\n");
    let c = ctx(&dir);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(
        ReadFileTool::new(c.clone()).execute(serde_json::json!({ "path": "a.txt", "limit": 1 })),
    )
    .unwrap();
    let err = rt
        .block_on(
            EditFileTool::new(c)
                .execute(serde_json::json!({ "path": "a.txt", "old_text": "2", "new_text": "X" })),
        )
        .unwrap_err();
    assert!(matches!(err, ActionError::PartialRead { .. }), "{err:?}");
}

#[test]
fn ambiguous_match_requires_replace_all() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "x x\n");
    let c = ctx(&dir);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(ReadFileTool::new(c.clone()).execute(serde_json::json!({ "path": "a.txt" })))
        .unwrap();
    let editor = EditFileTool::new(c);

    let err = rt
        .block_on(
            editor
                .execute(serde_json::json!({ "path": "a.txt", "old_text": "x", "new_text": "y" })),
        )
        .unwrap_err();
    assert!(
        matches!(err, ActionError::AmbiguousMatch { count: 2, .. }),
        "{err:?}"
    );

    rt.block_on(editor.execute(
        serde_json::json!({ "path": "a.txt", "old_text": "x", "new_text": "y", "replace_all": true }),
    ))
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "y y\n"
    );
}

#[test]
fn identical_old_and_new_text_is_a_noop_error() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "x\n");
    let err = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(
            EditFileTool::new(ctx(&dir))
                .execute(serde_json::json!({ "path": "a.txt", "old_text": "x", "new_text": "x" })),
        )
        .unwrap_err();
    assert!(matches!(err, ActionError::NoOp), "{err:?}");
}

#[test]
fn write_creates_new_file_without_overwrite() {
    let dir = tempfile::tempdir().unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(
        WriteFileTool::new(ctx(&dir))
            .execute(serde_json::json!({ "path": "sub/new.txt", "content": "hi" })),
    )
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.path().join("sub/new.txt")).unwrap(),
        "hi"
    );
}

#[test]
fn write_refuses_overwrite_without_flag() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "old");
    let err = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(
            WriteFileTool::new(ctx(&dir))
                .execute(serde_json::json!({ "path": "a.txt", "content": "new" })),
        )
        .unwrap_err();
    assert!(
        matches!(err, ActionError::ExistsWithoutOverwrite { .. }),
        "{err:?}"
    );
}

#[test]
fn write_refuses_overwrite_of_unread_file() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "old");
    let err =
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(WriteFileTool::new(ctx(&dir)).execute(
                serde_json::json!({ "path": "a.txt", "content": "new", "overwrite": true }),
            ))
            .unwrap_err();
    assert!(
        matches!(err, ActionError::ReadBeforeWrite { .. }),
        "{err:?}"
    );
}

#[test]
fn write_with_overwrite_after_read_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "old");
    let c = ctx(&dir);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(ReadFileTool::new(c.clone()).execute(serde_json::json!({ "path": "a.txt" })))
        .unwrap();
    rt.block_on(
        WriteFileTool::new(c)
            .execute(serde_json::json!({ "path": "a.txt", "content": "new", "overwrite": true })),
    )
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
        "new"
    );
}

#[test]
fn read_rejects_directory_and_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    let tool = ReadFileTool::new(ctx(&dir));
    let rt = tokio::runtime::Runtime::new().unwrap();
    let err = rt
        .block_on(tool.execute(serde_json::json!({ "path": "." })))
        .unwrap_err();
    assert!(matches!(err, ActionError::IsDirectory { .. }), "{err:?}");

    let err = rt
        .block_on(tool.execute(serde_json::json!({ "path": "nope.txt" })))
        .unwrap_err();
    assert!(matches!(err, ActionError::NotFound { .. }), "{err:?}");
}

#[test]
fn read_rejects_binary_and_offset_past_eof() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("bin"), [0u8, 1, 2, 3]).unwrap();
    write(&dir, "a.txt", "one\n");
    let tool = ReadFileTool::new(ctx(&dir));
    let rt = tokio::runtime::Runtime::new().unwrap();

    let err = rt
        .block_on(tool.execute(serde_json::json!({ "path": "bin" })))
        .unwrap_err();
    assert!(matches!(err, ActionError::Binary { .. }), "{err:?}");

    let err = rt
        .block_on(tool.execute(serde_json::json!({ "path": "a.txt", "offset": 9 })))
        .unwrap_err();
    assert!(matches!(err, ActionError::OffsetPastEof { .. }), "{err:?}");
}

#[test]
fn external_modification_invalidates_read_record() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "original\n");
    let c = ctx(&dir);
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(ReadFileTool::new(c.clone()).execute(serde_json::json!({ "path": "a.txt" })))
        .unwrap();

    // Another process (or `bash`) changes the file.
    write(&dir, "a.txt", "changed\n");

    let err = rt
        .block_on(EditFileTool::new(c).execute(
            serde_json::json!({ "path": "a.txt", "old_text": "changed", "new_text": "x" }),
        ))
        .unwrap_err();
    assert!(
        matches!(err, ActionError::ExternallyModified { .. }),
        "{err:?}"
    );
}

#[test]
fn tools_share_file_state_across_instances() {
    let dir = tempfile::tempdir().unwrap();
    write(&dir, "a.txt", "x\n");
    let c = ctx(&dir);
    let root = c.workspace.root().to_path_buf();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(ReadFileTool::new(c.clone()).execute(serde_json::json!({ "path": "a.txt" })))
        .unwrap();
    // File state is keyed by the resolved path, so a second tool built from a
    // clone of the same context must see the recorded read.
    assert!(c.files.has_full_read(&root.join("a.txt")));
}
