//! Search tools: glob_files (by path pattern), grep_text (by content).
//!
//! Paper Appendix B.2 (Figures 25–26). Both are read-only and apply the shared
//! ignore patterns.

use std::path::Path;

use async_trait::async_trait;
use regex::Regex;
use tracing::debug;

use axolotl_models::types::ToolDefinition;

use crate::error::{ActionError, ActionResult};
use crate::registry::{Action, ToolContext};

/// Ignore patterns applied to every read-only discovery tool (paper Fig. 24).
const IGNORED_SEGMENTS: &[&str] = crate::list_dir::IGNORED_DIRS;
const DEFAULT_GLOB_LIMIT: usize = 200;
const DEFAULT_GREP_LIMIT: usize = 100;
/// Files above this size are skipped by grep (paper Fig. 26: 10MB).
const MAX_GREP_FILE_BYTES: u64 = 10 * 1024 * 1024;

/// Find files matching a glob pattern.
pub struct SearchFilesTool {
    ctx: ToolContext,
}

impl SearchFilesTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Action for SearchFilesTool {
    fn name(&self) -> &str {
        "glob_files"
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "glob_files".to_string(),
            description: "Find files matching a glob pattern within the workspace.\n\nWhen: use this when you know the file pattern (**/*.py, src/components/*.tsx). For listing all entries in a single directory use list_files; for content search use grep_text.\n\nProtocol: returns up to max_matches matching file paths (default 200), workspace-relative. Pattern syntax: ** recursive, * within a segment, ? single char, [abc] character class. Only files are returned — directories matching the pattern are excluded. Workspace ignore patterns (.git, __pycache__, .venv, node_modules, target) are filtered silently.\n\nError modes: returns an empty result (not an error) when no files match. Returns an error if the path doesn't exist or isn't a directory.\n\nSide effects: none — read-only.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern matched against the workspace-relative path (e.g. '**/*.rs', 'src/**/*.ts')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default '.')"
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "Maximum matches to return (default 200)"
                    }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let pattern = crate::args::require_str(&arguments, "pattern")?;
        let search_path = arguments
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(".");
        let max = crate::args::optional_usize(&arguments, "max_matches")?
            .unwrap_or(DEFAULT_GLOB_LIMIT)
            .clamp(1, 1000);

        let root = self.ctx.workspace.resolve(search_path)?;
        if !root.is_dir() {
            return Err(ActionError::NotADirectory {
                path: self.ctx.workspace.display_path(&root),
            });
        }

        let matcher = globset::GlobBuilder::new(pattern)
            .literal_separator(true)
            .build()
            .map(|g| g.compile_matcher())
            .map_err(|e| ActionError::InvalidArgument(format!("invalid glob '{pattern}': {e}")))?;

        debug!(pattern = %pattern, "Globbing files");

        let mut matches = Vec::new();
        for entry in walk_files(&root) {
            let relative = self.ctx.workspace.display_path(&entry);
            let candidate = relative.replace('\\', "/");
            if matcher.is_match(&candidate) || matcher.is_match(&entry) {
                matches.push(relative);
                if matches.len() >= max {
                    matches.push("…[match limit reached]".to_string());
                    break;
                }
            }
        }

        matches.sort();
        if matches.is_empty() {
            return Ok("No files found matching pattern.".to_string());
        }
        Ok(matches.join("\n"))
    }
}

/// Search file contents by regex.
pub struct GrepTool {
    ctx: ToolContext,
}

impl GrepTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl Action for GrepTool {
    fn name(&self) -> &str {
        "grep_text"
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "grep_text".to_string(),
            description: "Search for a regex pattern across workspace files, returning matched lines with paths and line numbers.\n\nWhen: use this when you need to find code by content — function definitions, error messages, configuration values, TODOs. For pattern-based file discovery (no content matching) use glob_files.\n\nProtocol: query is a regex. Returns up to max_matches hits (default 100), formatted as path:line:matched_text. Use include to restrict to a filename glob (e.g. *.py). Set case_sensitive=false for case-insensitive matching.\n\nError modes: returns an empty result (not an error) when no matches. Binary files and files >10MB are silently skipped. Workspace ignore patterns are respected.\n\nSide effects: none — read-only.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Regex pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (default '.')"
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob to restrict the file name (e.g. '*.rs')"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether matching is case sensitive (default true)"
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "Maximum hits to return (default 100)"
                    }
                },
                "required": ["query"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let query = crate::args::require_str(&arguments, "query")?;
        let search_path = arguments
            .get("path")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(".");
        let case_sensitive = arguments
            .get("case_sensitive")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let max = crate::args::optional_usize(&arguments, "max_matches")?
            .unwrap_or(DEFAULT_GREP_LIMIT)
            .clamp(1, 1000);

        let pattern = if case_sensitive {
            query.to_string()
        } else {
            format!("(?i){query}")
        };
        let regex = Regex::new(&pattern).map_err(|e| ActionError::InvalidRegex(e.to_string()))?;

        let include = arguments
            .get("include")
            .and_then(serde_json::Value::as_str)
            .map(|glob| {
                globset::Glob::new(glob)
                    .map(|g| g.compile_matcher())
                    .map_err(|e| ActionError::InvalidArgument(format!("invalid include glob: {e}")))
            })
            .transpose()?;

        let root = self.ctx.workspace.resolve(search_path)?;
        debug!(pattern = %query, "Searching file contents");

        let files = if root.is_file() {
            vec![root]
        } else {
            walk_files(&root)
        };

        let mut results = Vec::new();
        for file in files {
            let display = self.ctx.workspace.display_path(&file);
            if let Some(ref filter) = include {
                let name = file.file_name().map(|n| n.to_string_lossy().to_string());
                let matches = name.as_deref().is_some_and(|n| filter.is_match(n))
                    || filter.is_match(&display);
                if !matches {
                    continue;
                }
            }

            let Ok(meta) = std::fs::metadata(&file) else {
                continue;
            };
            if meta.len() > MAX_GREP_FILE_BYTES {
                continue;
            }
            let Ok(bytes) = std::fs::read(&file) else {
                continue;
            };
            if bytes.iter().take(8192).any(|b| *b == 0) {
                continue;
            }
            let content = String::from_utf8_lossy(&bytes);

            for (line_no, line) in content.lines().enumerate() {
                if regex.is_match(line) {
                    results.push(format!("{}:{}: {}", display, line_no + 1, line.trim()));
                    if results.len() >= max {
                        results.push("[... truncated at match limit]".to_string());
                        return Ok(results.join("\n"));
                    }
                }
            }
        }

        if results.is_empty() {
            Ok("No matches found.".to_string())
        } else {
            Ok(results.join("\n"))
        }
    }
}

/// Walk files under `root`, skipping ignored directories and symlinks.
fn walk_files(root: &Path) -> Vec<std::path::PathBuf> {
    walkdir::WalkDir::new(root)
        .max_depth(20)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            !(e.file_type().is_dir() && IGNORED_SEGMENTS.contains(&name.as_ref()))
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(walkdir::DirEntry::into_path)
        .collect()
}

#[cfg(test)]
mod tests;
