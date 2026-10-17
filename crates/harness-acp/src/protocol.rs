//! ACP wire types.
//!
//! Written against the Agent Client Protocol v1 schema
//! (<https://agentclientprotocol.com/protocol/v1/schema>). Property keys are
//! camelCase; string values carried by discriminator fields are snake_case.
//!
//! Only the subset this agent needs is modelled. Unknown fields are ignored on
//! deserialize, which is required: the protocol adds capabilities without
//! breaking changes, and a strict shape would reject a client that sends more
//! than we know about.
//!
//! Split into [`inbound`] (client → agent), [`outbound`] (agent → client) and
//! [`rpc`] (the JSON-RPC envelope) so no single file carries the whole schema.

pub mod inbound;
pub mod outbound;
pub mod rpc;

pub use inbound::*;
pub use outbound::*;
pub use rpc::*;

/// Protocol version this agent speaks.
pub const PROTOCOL_VERSION: u32 = 1;

#[cfg(test)]
mod tests;
