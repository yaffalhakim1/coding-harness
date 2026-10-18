//! Error types for tool execution.

/// Errors a tool can return.
///
/// Every variant is *recoverable*: the agent loop turns it into an observation
/// the model can act on (paper §2.4 — "Tool errors are returned to the model as
/// observations rather than raised, so a failed action never crashes the loop").
#[derive(Debug, thiserror::Error)]
pub enum ActionError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("path '{0}' escapes the workspace root")]
    PathEscape(String),

    #[error("workspace error: {0}")]
    Workspace(String),

    #[error("file '{path}' has not been read in this session; read it with read_file first")]
    ReadBeforeWrite { path: String },

    #[error(
        "file '{path}' only had a partial read ({lines} of {total} lines); read it fully first"
    )]
    PartialRead {
        path: String,
        lines: usize,
        total: usize,
    },

    #[error("file '{path}' was modified outside the harness since it was read; re-read it first")]
    ExternallyModified { path: String },

    #[error("file '{path}' already exists; pass overwrite=true to replace it")]
    ExistsWithoutOverwrite { path: String },

    #[error("'{path}' is a directory; use list_files")]
    IsDirectory { path: String },

    #[error("'{path}' is not a directory")]
    NotADirectory { path: String },

    #[error("file '{path}' does not exist; use write_file to create it")]
    NotFound { path: String },

    #[error("text to replace was not found in '{path}'")]
    NoMatch { path: String },

    #[error("text to replace occurs {count} times in '{path}'; add surrounding context or pass replace_all=true")]
    AmbiguousMatch { path: String, count: usize },

    #[error("old_text and new_text are identical; nothing to do")]
    NoOp,

    #[error("'{path}' appears to be a binary file")]
    Binary { path: String },

    #[error("'{path}' is {size} bytes, above the {limit}-byte read limit; use bash with head/sed")]
    TooLarge { path: String, size: u64, limit: u64 },

    #[error("offset {offset} is past end of file ({total} lines)")]
    OffsetPastEof { offset: usize, total: usize },

    #[error("unknown tool '{0}'")]
    UnknownTool(String),

    #[error("tool arguments were not valid JSON: {0}")]
    MalformedArguments(String),

    #[error("I/O error on '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("command timed out after {seconds}s; partial output:\n{output}")]
    Timeout { seconds: u64, output: String },

    #[error("command exited with code {code}:\n{output}")]
    CommandFailed { code: i32, output: String },

    #[error("invalid regex: {0}")]
    InvalidRegex(String),

    #[error("unknown event id '{id}' (the store holds {known} events); it may have been evicted or never elided")]
    UnknownEvent { id: String, known: usize },

    #[error("fetching '{url}' failed: {reason}")]
    Fetch { url: String, reason: String },

    #[error("MCP tool '{tool}' failed: {reason}")]
    Mcp { tool: String, reason: String },

    #[error("the user denied '{tool}': {title}")]
    Denied { tool: String, title: String },

    #[error("'{tool}' was cancelled")]
    Cancelled { tool: String },
}

impl ActionError {
    pub fn io(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::Io {
            path: path.to_string(),
            source,
        }
    }
}

pub type ActionResult<T> = std::result::Result<T, ActionError>;
