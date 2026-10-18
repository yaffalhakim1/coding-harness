//! Directory listing tool (paper Figure 24).

use async_trait::async_trait;

use axolotl_models::types::ToolDefinition;

use crate::args::{optional_bool, optional_str, optional_usize};
use crate::error::{ActionError, ActionResult};
use crate::registry::{Action, ToolContext};

/// Default entry cap for `list_files` (paper Fig. 24).
const DEFAULT_LIST_LIMIT: usize = 200;
/// Ignore patterns applied by the read-only discovery tools (paper §2.4).
pub(crate) const IGNORED_DIRS: &[&str] =
    &[".git", "__pycache__", ".venv", "node_modules", "target"];

/// List directory contents.
pub struct ListDirTool {
    ctx: ToolContext,
}

impl ListDirTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Action for ListDirTool {
    fn name(&self) -> &str {
        "list_files"
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_files".to_string(),
            description: "List files and directories in the workspace, returning a tree-style view with relative paths and (for files) byte sizes.\n\nWhen: use this to explore project structure before reading specific files. For pattern-based file discovery use glob_files; for content search use grep_text.\n\nProtocol: returns up to max_entries entries (default 200), sorted alphabetically within each directory. Set recursive=true to descend into subdirectories — depth is reflected in output indentation. Workspace ignore patterns (.git, __pycache__, .venv, node_modules, target) are silently filtered.\n\nError modes: returns an error if path doesn't exist or isn't a directory.\n\nSide effects: none — read-only.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list, relative to the workspace root (default '.')"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Descend into subdirectories (default false)"
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum entries to return (default 200)"
                    }
                },
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let path = optional_str(&arguments, "path").unwrap_or(".");
        let recursive = optional_bool(&arguments, "recursive");
        let max_entries = optional_usize(&arguments, "max_entries")?
            .unwrap_or(DEFAULT_LIST_LIMIT)
            .clamp(1, 2000);

        let full_path = self.ctx.workspace.resolve(path)?;
        let display = self.ctx.workspace.display_path(&full_path);

        if !full_path.exists() {
            return Err(ActionError::NotFound { path: display });
        }
        if !full_path.is_dir() {
            return Err(ActionError::NotADirectory { path: display });
        }

        let mut lines = Vec::new();
        let depth_limit = if recursive { usize::MAX } else { 1 };
        collect_entries(&full_path, depth_limit, max_entries, &mut lines);

        if lines.is_empty() {
            return Ok(format!("{display} is empty"));
        }
        Ok(lines.join("\n"))
    }
}

/// Walk a directory synchronously; listing is cheap enough that spawning
/// per-entry tasks would cost more than it saves.
fn collect_entries(dir: &std::path::Path, depth: usize, max: usize, out: &mut Vec<String>) {
    if out.len() >= max {
        return;
    }
    let mut entries: Vec<_> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.filter_map(Result::ok).collect(),
        Err(_) => return,
    };
    entries.sort_by_key(std::fs::DirEntry::file_name);

    for entry in entries {
        if out.len() >= max {
            out.push("…[entry limit reached]".to_string());
            return;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);

        if is_dir && IGNORED_DIRS.contains(&name.as_str()) {
            continue;
        }
        if is_dir {
            out.push(format!("{name}/"));
            if depth > 1 {
                let mut nested = Vec::new();
                collect_entries(&entry.path(), depth - 1, max, &mut nested);
                out.extend(nested.into_iter().map(|l| format!("  {l}")));
            }
        } else {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            out.push(format!("{name} ({size} bytes)"));
        }
    }
}

#[cfg(test)]
mod tests;
