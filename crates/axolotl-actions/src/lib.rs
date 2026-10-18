//! axolotl-actions: Tool registry and execution.
//!
//! Provides predefined tools (read_file, write_file, edit_file, list_files,
//! glob_files, grep_text, bash) and the bash-only action space.

pub mod args;
pub mod ask_user;
pub mod bash;
pub mod diagnostics;
pub mod error;
pub mod file_ops;
pub mod file_state;
pub mod list_dir;
pub mod mcp;
pub mod permission;
pub mod recall_event;
pub mod registry;
pub mod search;
pub mod skills;
pub mod slash;
pub mod todo;
pub mod update_plan;
pub mod web_fetch;
pub mod workspace;

pub use ask_user::{Answer, AskUserTool, Asker, Question};
pub use bash::BashExecutor;
pub use error::{ActionError, ActionResult};
pub use file_ops::{EditFileTool, ReadFileTool, WriteFileTool};
pub use file_state::FileState;
pub use list_dir::ListDirTool;
pub use permission::{
    AllowAll, Approval, ApprovalRequest, DenyAll, DestructiveCommandGate, PermissionGate,
};
pub use recall_event::RecallEventTool;
pub use registry::{truncate_output, Action, ActionRegistry, DefaultRegistry, ToolContext};
pub use search::{GrepTool, SearchFilesTool};
pub use skills::Skill;
pub use todo::{Todo, TodoList, TodoStatus, TodoTool};
pub use update_plan::{PlanState, UpdatePlanTool};
pub use web_fetch::WebFetchTool;
pub use workspace::Workspace;
