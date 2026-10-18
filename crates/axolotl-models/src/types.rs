//! Core types shared across the harness.
//!
//! These are the harness's own types. Provider adapters translate them into
//! whatever their wire format requires; they are never serialized to an API
//! directly (see `openai::wire`).

use serde::{Deserialize, Serialize};

/// Role of a chat participant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Option<String>,
    /// Tool call ID (for role=Tool messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls emitted by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: Some(content.into()),
            tool_call_id: None,
            tool_calls: None,
        }
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Some(content.into()),
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }

    pub fn assistant_with_tool_calls(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            tool_call_id: None,
            tool_calls: Some(tool_calls),
        }
    }

    /// True when the message carries no text and no tool calls.
    pub fn is_empty(&self) -> bool {
        self.content.as_deref().map_or(true, str::is_empty) && self.has_no_tool_calls()
    }

    pub fn has_no_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().map_or(true, Vec::is_empty)
    }

    /// Text content, treating `None` as empty.
    pub fn text(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
}

/// A tool call emitted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Token usage for a single LLM call.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl TokenUsage {
    pub fn merge(&mut self, other: &TokenUsage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// Response from an LLM call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    pub message: ChatMessage,
    pub usage: TokenUsage,
    pub stop_reason: StopReason,
    pub model: String,
}

impl LLMResponse {
    /// Tool calls requested by the model, empty when it asked for none.
    pub fn tool_calls(&self) -> &[ToolCall] {
        self.message.tool_calls.as_deref().unwrap_or_default()
    }
}

/// Why the LLM stopped generating.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    StopSequence,
    Other(String),
}

/// Configuration for an LLM model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub provider: String,
    pub model_id: String,
    pub api_base: Option<String>,
    pub api_key_env: Option<String>,
    pub temperature: f32,
    pub max_tokens: Option<usize>,
    pub timeout_secs: u64,
    /// Input price in USD per million tokens (for cost accounting).
    #[serde(default)]
    pub input_cost_per_mtok: Option<f64>,
    /// Output price in USD per million tokens (for cost accounting).
    #[serde(default)]
    pub output_cost_per_mtok: Option<f64>,
    /// Maximum retry attempts for retryable API failures.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

fn default_max_retries() -> u32 {
    4
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            provider: "openai".to_string(),
            model_id: "gpt-4o".to_string(),
            api_base: None,
            api_key_env: Some("OPENAI_API_KEY".to_string()),
            temperature: 0.0,
            max_tokens: Some(4096),
            timeout_secs: 120,
            input_cost_per_mtok: None,
            output_cost_per_mtok: None,
            max_retries: default_max_retries(),
        }
    }
}

impl ModelConfig {
    /// Cost of the given usage in USD, using configured prices when present.
    ///
    /// Falls back to `None` when prices are unknown — callers must not invent
    /// a price, because published results depend on it.
    pub fn cost_usd(&self, usage: &TokenUsage) -> Option<f64> {
        let input = self.input_cost_per_mtok?;
        let output = self.output_cost_per_mtok?;
        Some(
            usage.prompt_tokens as f64 / 1_000_000.0 * input
                + usage.completion_tokens as f64 / 1_000_000.0 * output,
        )
    }
}

/// A tool definition exposed to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_message_detection() {
        assert!(ChatMessage::assistant("").is_empty());
        let mut msg = ChatMessage::assistant("");
        msg.content = None;
        assert!(msg.is_empty());
    }

    #[test]
    fn message_with_tool_calls_is_not_empty() {
        let call = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({ "command": "ls" }),
        };
        let msg = ChatMessage::assistant_with_tool_calls(None, vec![call]);
        assert!(!msg.is_empty());
        assert_eq!(msg.tool_calls.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn cost_requires_both_prices() {
        let mut cfg = ModelConfig::default();
        let usage = TokenUsage {
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            total_tokens: 2_000_000,
        };
        assert_eq!(cfg.cost_usd(&usage), None);

        cfg.input_cost_per_mtok = Some(2.0);
        cfg.output_cost_per_mtok = Some(8.0);
        assert_eq!(cfg.cost_usd(&usage), Some(10.0));
    }
}
