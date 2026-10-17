//! Turn recording — tracks each LLM interaction.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use harness_models::types::{ChatMessage, LLMResponse, ToolCall};

/// A single turn in the agent loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub id: String,
    pub turn_number: u32,
    pub timestamp: DateTime<Utc>,
    /// The context window sent to the model this turn.
    pub input_messages: Vec<ChatMessage>,
    /// Tokens in the context window (cheaper to read than re-tokenizing).
    pub input_tokens: usize,
    pub response: LLMResponse,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    /// Wall-clock time for the model call plus tool execution.
    pub duration_ms: u64,
}

impl Turn {
    pub fn new(
        turn_number: u32,
        input: Vec<ChatMessage>,
        input_tokens: usize,
        response: LLMResponse,
    ) -> Self {
        let tool_calls = response.message.tool_calls.clone().unwrap_or_default();

        Self {
            id: Uuid::new_v4().to_string(),
            turn_number,
            timestamp: Utc::now(),
            input_messages: input,
            input_tokens,
            response,
            tool_calls,
            tool_results: Vec::new(),
            duration_ms: 0,
        }
    }

    pub fn add_tool_result(&mut self, result: ToolResult) {
        self.tool_results.push(result);
    }

    pub fn set_duration(&mut self, duration_ms: u64) {
        self.duration_ms = duration_ms;
    }

    /// Any tool call in this turn failed.
    pub fn had_tool_failure(&self) -> bool {
        self.tool_results.iter().any(|r| !r.success)
    }

    /// Trajectory representation: a preview of tool output rather than the full
    /// body, so trajectory files stay readable at sweep scale.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "turn_number": self.turn_number,
            "timestamp": self.timestamp.to_rfc3339(),
            "input_tokens": self.input_tokens,
            "prompt_tokens": self.response.usage.prompt_tokens,
            "output_tokens": self.response.usage.completion_tokens,
            "stop_reason": format!("{:?}", self.response.stop_reason),
            "assistant_text": preview(self.response.message.text(), 500),
            "tool_calls": self.tool_calls.iter().map(|tc| {
                serde_json::json!({ "name": tc.name, "arguments": tc.arguments })
            }).collect::<Vec<_>>(),
            "tool_results": self.tool_results.iter().map(|tr| {
                serde_json::json!({
                    "tool_name": tr.tool_name,
                    "success": tr.success,
                    "output_preview": preview(&tr.preview_text(), 600),
                    "error": tr.error,
                    "duration_ms": tr.duration_ms,
                })
            }).collect::<Vec<_>>(),
            "duration_ms": self.duration_ms,
        })
    }
}

fn preview(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect::<String>() + "…"
}

/// Result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub tool_name: String,
    pub output: String,
    pub success: bool,
    pub error: Option<String>,
    pub duration_ms: u64,
}

impl ToolResult {
    /// The text handed back to the model as a tool observation.
    ///
    /// Failures are rendered as an observation, never raised, so a bad call can
    /// be recovered from (paper §2.4).
    pub fn observation(&self) -> String {
        match (&self.error, self.success) {
            (Some(err), _) => format!("Error: {err}"),
            (None, true) => self.output.clone(),
            (None, false) => "Error: tool failed without a message".to_string(),
        }
    }

    /// Text to preview in a trajectory: the observation actually shown to the
    /// model, so failures are as inspectable as successes.
    pub fn preview_text(&self) -> String {
        if self.success {
            self.output.clone()
        } else {
            self.observation()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harness_models::types::{StopReason, TokenUsage};

    fn response(text: &str) -> LLMResponse {
        LLMResponse {
            message: ChatMessage::assistant(text),
            usage: TokenUsage::default(),
            stop_reason: StopReason::EndTurn,
            model: "mock".into(),
        }
    }

    #[test]
    fn turn_captures_tool_calls() {
        let call = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({}),
        };
        let resp = LLMResponse {
            message: ChatMessage::assistant_with_tool_calls(None, vec![call]),
            usage: TokenUsage::default(),
            stop_reason: StopReason::ToolUse,
            model: "m".into(),
        };
        let turn = Turn::new(0, vec![], 10, resp);
        assert_eq!(turn.tool_calls.len(), 1);
        assert!(!turn.had_tool_failure());
    }

    #[test]
    fn failure_observation_is_wrapped() {
        let result = ToolResult {
            tool_call_id: "1".into(),
            tool_name: "bash".into(),
            output: String::new(),
            success: false,
            error: Some("boom".into()),
            duration_ms: 1,
        };
        assert_eq!(result.observation(), "Error: boom");
        // Failures must remain inspectable in a trajectory.
        assert_eq!(result.preview_text(), "Error: boom");
    }

    #[test]
    fn success_observation_is_passed_through() {
        let result = ToolResult {
            tool_call_id: "1".into(),
            tool_name: "bash".into(),
            output: "ok".into(),
            success: true,
            error: None,
            duration_ms: 1,
        };
        assert_eq!(result.observation(), "ok");
    }

    #[test]
    fn trajectory_preview_is_char_safe() {
        let turn = Turn::new(0, vec![], 1, response(&"α".repeat(1000)));
        let json = turn.to_json();
        let text = json["assistant_text"].as_str().unwrap();
        assert!(text.ends_with('…'));
    }
}
