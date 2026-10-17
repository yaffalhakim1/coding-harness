//! File operation tools: read_file, write_file, edit_file.
//!
//! Argument names and semantics follow paper Table 1 and Appendix B.1
//! (Figures 21-23). `list_files` lives in `crate::list_dir`.

use async_trait::async_trait;
use tracing::debug;

use harness_models::types::ToolDefinition;

use crate::args::{optional_usize, require_str};
use crate::error::{ActionError, ActionResult};
use crate::registry::{Action, ToolContext};

/// Files above this size are refused by `read_file` and redirected to `bash`.
const MAX_READ_BYTES: u64 = 2 * 1024 * 1024;
/// Default page size for `read_file` (paper Fig. 21).
const DEFAULT_READ_LIMIT: usize = 2000;

/// Read a file, with offset/limit and line numbers.
pub struct ReadFileTool {
    ctx: ToolContext,
}

impl ReadFileTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Action for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".to_string(),
            description: "Read the contents of a text file from the workspace, returned with 1-based line numbers and a header describing what range was read.\n\nWhen: call this BEFORE edit_file or write_file(overwrite=true) on any existing file — the harness enforces read-before-write and rejects edits without a recorded full read. For directory listings use list_files; for content search use grep_text.\n\nProtocol: returns lines [offset, offset+limit) using 1-based offsets. Default reads from line 1 with limit=2000. The read is recorded as full_read=true ONLY when offset=1 AND limit covers the entire file; partial reads do NOT satisfy the read-before-write gate.\n\nError modes: returns an error for non-existent paths, paths outside the workspace, directories (suggests list_files), binary files (NUL byte in first 8KB), files exceeding the read cap (suggests bash with head/sed instead), or offsets past EOF.\n\nSide effects: records the read in the session's file state.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the workspace root"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based line to start reading from (default 1)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default 2000)"
                    }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let path = require_str(&arguments, "path")?;
        let offset = optional_usize(&arguments, "offset")?.unwrap_or(1).max(1);
        let limit = optional_usize(&arguments, "limit")?
            .unwrap_or(DEFAULT_READ_LIMIT)
            .max(1);

        let full_path = self.ctx.workspace.resolve(path)?;
        let rel = self.ctx.workspace.display_path(&full_path);
        debug!(path = %rel, offset, limit, "Reading file");

        let meta = tokio::fs::metadata(&full_path)
            .await
            .map_err(|_| ActionError::NotFound { path: rel.clone() })?;

        if meta.is_dir() {
            return Err(ActionError::IsDirectory { path: rel });
        }
        if meta.len() > MAX_READ_BYTES {
            return Err(ActionError::TooLarge {
                path: rel,
                size: meta.len(),
                limit: MAX_READ_BYTES,
            });
        }

        let bytes = tokio::fs::read(&full_path)
            .await
            .map_err(|e| ActionError::io(&rel, e))?;

        if is_binary(&bytes) {
            return Err(ActionError::Binary { path: rel });
        }

        let content = String::from_utf8_lossy(&bytes).into_owned();
        let lines: Vec<&str> = content.lines().collect();

        if offset > lines.len().max(1) {
            return Err(ActionError::OffsetPastEof {
                offset,
                total: lines.len(),
            });
        }

        let start = offset - 1;
        let end = (start + limit).min(lines.len());
        self.ctx
            .files
            .record_read(&full_path, &content, offset, end - start);

        let mut out = format!("[{}] lines {}-{} of {}\n", rel, offset, end, lines.len());
        for (i, line) in lines[start..end].iter().enumerate() {
            out.push_str(&format!("{:>6}\t{}\n", start + i + 1, line));
        }
        Ok(crate::registry::truncate_output(
            out,
            self.ctx.max_output_chars,
        ))
    }
}

/// Write a file, creating it or (with `overwrite`) replacing it.
pub struct WriteFileTool {
    ctx: ToolContext,
}

impl WriteFileTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Action for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".to_string(),
            description: "Write content to a file path — creates a new file or overwrites an existing one.\n\nWhen: use this for (a) creating new files, (b) full-file rewrites of existing files. For small in-place edits to existing files use edit_file instead. New-file creation does not require overwrite=true; only overwriting an existing file does.\n\nProtocol: if the path doesn't exist, the file is created and parent directories are created as needed. If the path exists, you MUST pass overwrite=true AND you MUST have called read_file with a full read on this path in this session first.\n\nError modes: returns a recoverable error when the file exists and overwrite=false (suggests edit_file or overwrite=true), when the path is a directory, or when the existing file was never read or only partially read.\n\nSide effects: writes the file and creates parent directories as needed. Updates the session's file state with the new content.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the workspace root"
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete content to write"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Required to replace an existing file (default false)"
                    }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let path = require_str(&arguments, "path")?;
        let content = require_str(&arguments, "content")?;
        let overwrite = arguments
            .get("overwrite")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        let full_path = self.ctx.workspace.resolve(path)?;
        let rel = self.ctx.workspace.display_path(&full_path);

        if full_path.exists() {
            if full_path.is_dir() {
                return Err(ActionError::IsDirectory { path: rel });
            }
            if !overwrite {
                return Err(ActionError::ExistsWithoutOverwrite { path: rel });
            }
            let existing = tokio::fs::read_to_string(&full_path)
                .await
                .map_err(|e| ActionError::io(&rel, e))?;
            self.ctx
                .files
                .ensure_read_for_write(&full_path, &rel, &existing)?;
        }

        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| ActionError::io(&rel, e))?;
        }
        tokio::fs::write(&full_path, &content)
            .await
            .map_err(|e| ActionError::io(&rel, e))?;

        self.ctx.files.record_write(&full_path, content);
        debug!(path = %rel, bytes = content.len(), "Wrote file");

        // Paper §2.4: surface syntax problems immediately so the model can fix
        // them before spending a turn on the tests.
        let diagnostics = crate::diagnostics::check_file(&full_path)
            .await
            .unwrap_or_default();

        Ok(format!(
            "Wrote {} bytes to {rel}{diagnostics}",
            content.len()
        ))
    }
}

/// Replace an exact string in an existing file.
pub struct EditFileTool {
    ctx: ToolContext,
}

impl EditFileTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Action for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".to_string(),
            description: "Edit a text file by replacing one or more occurrences of old_text with new_text.\n\nWhen: use this for SMALL, TARGETED changes to EXISTING files. For new files use write_file; for full-file rewrites use write_file(overwrite=true).\n\nProtocol: old_text MUST appear in the file at least once and the match is byte-exact (including whitespace and line endings). If old_text appears more than once, either provide MORE surrounding context to make it unique, OR pass replace_all=true. old_text == new_text is rejected as a no-op. The file MUST have been read fully in this session BEFORE calling edit_file.\n\nError modes: non-existent path (suggests write_file), path is a directory, old_text not found, ambiguous match without replace_all, identical old_text/new_text.\n\nSide effects: writes the modified content to disk and updates the session's file state.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to the workspace root"
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace (byte-exact, including whitespace)"
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence (default false)"
                    }
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let path = require_str(&arguments, "path")?;
        let old_text = require_str(&arguments, "old_text")?;
        let new_text = require_str(&arguments, "new_text")?;
        let replace_all = arguments
            .get("replace_all")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        if old_text == new_text {
            return Err(ActionError::NoOp);
        }

        let full_path = self.ctx.workspace.resolve(path)?;
        let rel = self.ctx.workspace.display_path(&full_path);

        if !full_path.exists() {
            return Err(ActionError::NotFound { path: rel });
        }
        if full_path.is_dir() {
            return Err(ActionError::IsDirectory { path: rel });
        }

        let content = tokio::fs::read_to_string(&full_path)
            .await
            .map_err(|e| ActionError::io(&rel, e))?;

        self.ctx
            .files
            .ensure_read_for_write(&full_path, &rel, &content)?;

        let count = content.matches(old_text).count();
        match count {
            0 => return Err(ActionError::NoMatch { path: rel }),
            1 => {}
            n if !replace_all => {
                return Err(ActionError::AmbiguousMatch {
                    path: rel,
                    count: n,
                })
            }
            _ => {}
        }

        let updated = if replace_all {
            content.replace(old_text, new_text)
        } else {
            content.replacen(old_text, new_text, 1)
        };

        tokio::fs::write(&full_path, &updated)
            .await
            .map_err(|e| ActionError::io(&rel, e))?;
        self.ctx.files.record_write(&full_path, &updated);

        debug!(path = %rel, replacements = count, "Edited file");

        let diagnostics = crate::diagnostics::check_file(&full_path)
            .await
            .unwrap_or_default();

        Ok(format!(
            "Replaced {count} occurrence(s) in {rel}{diagnostics}"
        ))
    }
}

/// A NUL byte in the first 8KB marks a binary file (paper Fig. 21).
fn is_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8192).any(|b| *b == 0)
}

#[cfg(test)]
mod tests;
