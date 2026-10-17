//! harness-tui: a live terminal view of an agent run.
//!
//! The TUI is a *viewer*: it owns no agent logic and never modifies the run. It
//! consumes [`Progress`](harness_core::progress::Progress) events from a channel
//! and renders them, which keeps the harness itself headless and testable.

pub mod app;
pub mod ui;

pub use app::{App, RunStatus};
