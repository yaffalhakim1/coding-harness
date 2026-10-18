//! axolotl-acp: run the coding harness as an Agent Client Protocol agent.
//!
//! This is the integration surface for clients like Kerenzikov: it speaks ACP
//! over stdio, so the client launches the binary and drives it with the same
//! protocol it uses for Copilot, Cursor, Grok and Kimi.
//!
//! Everything the harness already does — T0–T4 context management, the tool
//! set, planning, MCP — stays behind the protocol. The client sees only the
//! agent's updates and its stop reasons.

pub mod agent;
pub mod notify;
pub mod permission;
pub mod protocol;
pub mod rpc;
pub mod session;
pub mod sink;

pub use protocol::PROTOCOL_VERSION;
