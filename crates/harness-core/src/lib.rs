//! harness-core: Agent orchestration and execution loop.
//!
//! This crate contains the main agent loop that coordinates:
//! - Context management (via harness-context)
//! - Planning (via harness-planning)
//! - Tool execution (via harness-actions)
//! - LLM calls (via harness-models)

pub mod agent;
pub mod evaluator;
pub mod progress;
pub mod prompts;
pub mod replay;
pub mod session;
pub mod session_archive;
pub mod stuck;
pub mod subagent;
pub mod tools;
pub mod trajectory;
pub mod turn;
pub mod types;

pub use agent::{run_agent_loop, AgentDeps};
pub use evaluator::{evaluate_task, run_setup};
pub use progress::{ChannelSink, NoopSink, Progress, ProgressSink, RecordingSink};
pub use session::{ActionSpace, Session, SessionConfig, SessionOutcome, ThresholdSettings};
pub use session_archive::{
    list_sessions, load_session, write_session, SessionInfo, SESSION_SUFFIX,
};
pub use stuck::{StuckDetector, StuckSignal};
pub use subagent::SubagentTool;
pub use trajectory::write_trajectory;
pub use turn::{ToolResult, Turn};
pub use types::{Task, TaskResult};
