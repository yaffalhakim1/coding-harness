//! axolotl-planning: Plan generation and tracking.
//!
//! Implements planning strategies that help the agent:
//! - Decompose complex tasks into steps
//! - Stay focused during execution
//! - Track progress

pub mod default;
pub mod strategy;

pub use default::DefaultPlanner;
pub use strategy::PlanningStrategy;
