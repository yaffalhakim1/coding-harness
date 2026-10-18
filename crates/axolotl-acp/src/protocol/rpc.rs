//! The JSON-RPC 2.0 envelope.

use serde::Serialize;

/// A JSON-RPC error object.
#[derive(Debug, Clone, Serialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl RpcError {
    pub fn method_not_found(method: &str) -> Self {
        Self {
            code: -32601,
            message: format!("method not found: {method}"),
            data: None,
        }
    }

    /// The caller's request was wrong. Distinct from `internal` so a client can
    /// tell "you sent something bad" from "we broke".
    pub fn invalid_params(reason: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: reason.into(),
            data: None,
        }
    }

    pub fn internal(reason: impl Into<String>) -> Self {
        Self {
            code: -32603,
            message: reason.into(),
            data: None,
        }
    }
}
