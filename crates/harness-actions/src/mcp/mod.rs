//! MCP (Model Context Protocol) support.
//!
//! A small client for the tool-use subset of MCP, plus the adapter that exposes
//! discovered tools as harness [`Action`](crate::registry::Action)s.

pub mod adapter;
pub mod client;

pub use adapter::{connect_all, McpToolAction, NAME_SEPARATOR};
pub use client::{McpClient, McpTool, ServerConfig, Transport, PROTOCOL_VERSION};
