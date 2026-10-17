//! harness-core: Agent orchestration and execution loop.
//!
//! This crate contains the main agent loop that coordinates:
//! - Context management (via harness-context)
//! - Planning (via harness-planning)
//! - Tool execution (via harness-actions)
//! - LLM calls (via harness-models)

pub mod agent;
pub mod evaluator;
pub mod prompts;
pub mod session;
pub mod stuck;
pub mod tools;
pub mod trajectory;
pub mod turn;
pub mod types;

pub use agent::{run_agent_loop, AgentDeps};
pub use evaluator::{evaluate_task, run_setup};
pub use session::{ActionSpace, Session, SessionConfig, SessionOutcome, ThresholdSettings};
pub use stuck::{StuckDetector, StuckSignal};
pub use trajectory::write_trajectory;
pub use turn::{ToolResult, Turn};
pub use types::{Task, TaskResult};
