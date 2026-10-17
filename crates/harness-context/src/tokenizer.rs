//! Token counting.
//!
//! `Tokenizer` wraps `tiktoken-rs` and caches BPE tables, which are expensive to
//! construct (a few hundred ms) and are needed on every turn.
//!
//! Approximate counting is available via [`approx_tokens`] but is only used
//! where an exact answer is impossible (e.g. a provider that reports no usage);
//! the hot path uses the real tokenizer.

use std::sync::Arc;

use anyhow::Result;
use tiktoken_rs::{cl100k_base, CoreBPE};

use harness_models::types::ChatMessage;

/// Per-message framing overhead in the OpenAI chat format.
const MESSAGE_OVERHEAD: usize = 4;
/// Per-tool-call framing overhead.
const TOOL_CALL_OVERHEAD: usize = 3;
/// Reply priming appended to every request.
const REPLY_PRIMING: usize = 3;

/// Token counter backed by a real BPE table.
#[derive(Clone)]
pub struct Tokenizer {
    bpe: Arc<CoreBPE>,
}

impl Tokenizer {
    /// Create a tokenizer using the cl100k_base encoding (GPT-4, GPT-3.5,
    /// and a close approximation for most modern OpenAI-compatible models).
    pub fn new() -> Result<Self> {
        let bpe = cl100k_base().map_err(|e| anyhow::anyhow!("failed to load cl100k_base: {e}"))?;
        Ok(Self { bpe: Arc::new(bpe) })
    }

    /// Count tokens in a string.
    pub fn count_tokens(&self, text: &str) -> usize {
        self.bpe.encode_ordinary(text).len()
    }

    /// Estimate tokens for a chat message, including framing overhead.
    pub fn count_message_tokens(&self, message: &ChatMessage) -> usize {
        let mut tokens = MESSAGE_OVERHEAD;

        if let Some(content) = &message.content {
            tokens += self.count_tokens(content);
        }

        if let Some(tool_calls) = &message.tool_calls {
            for call in tool_calls {
                tokens += self.count_tokens(&call.name);
                tokens += self.count_tokens(&call.arguments.to_string());
                tokens += TOOL_CALL_OVERHEAD;
            }
        }

        tokens
    }

    /// Estimate tokens for a list of messages.
    pub fn count_messages(&self, messages: &[ChatMessage]) -> usize {
        messages
            .iter()
            .map(|m| self.count_message_tokens(m))
            .sum::<usize>()
            + REPLY_PRIMING
    }

    /// Truncate `text` to at most `max_tokens`, preferring a line boundary.
    ///
    /// Used when building elision previews so a stub never itself becomes bulky.
    pub fn take_tokens(&self, text: &str, max_tokens: usize) -> String {
        let mut out = String::new();
        for line in text.lines() {
            let candidate = if out.is_empty() {
                line.to_string()
            } else {
                format!("{out}\n{line}")
            };
            if self.count_tokens(&candidate) > max_tokens {
                break;
            }
            out = candidate;
        }

        if out.is_empty() && !text.is_empty() {
            // Single line longer than the budget: fall back to a character cut
            // sized by the observed chars-per-token ratio.
            let ratio = text.chars().count().max(1) as f64 / self.count_tokens(text).max(1) as f64;
            let chars = (max_tokens as f64 * ratio).max(1.0) as usize;
            out = text.chars().take(chars).collect();
        }
        out
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer").finish_non_exhaustive()
    }
}

/// Rough token estimate for a string when no tokenizer is available.
///
/// ~4 bytes per token is the usual rule of thumb for English and code. Used
/// only as a fallback; never for budgeting decisions.
pub fn approx_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use harness_models::types::{ToolCall, ToolDefinition};

    #[test]
    fn counts_monotonically() {
        let tok = Tokenizer::new().unwrap();
        let short = tok.count_tokens("hello");
        let long = tok.count_tokens("hello world, this is a longer sentence");
        assert!(long > short);
        assert!(short >= 1);
    }

    #[test]
    fn empty_string_is_zero_tokens() {
        let tok = Tokenizer::new().unwrap();
        assert_eq!(tok.count_tokens(""), 0);
    }

    #[test]
    fn message_overhead_is_included() {
        let tok = Tokenizer::new().unwrap();
        let bare = tok.count_tokens("hi");
        assert_eq!(tok.count_message_tokens(&ChatMessage::user("hi")), bare + 4);
    }

    #[test]
    fn tool_calls_are_counted() {
        let tok = Tokenizer::new().unwrap();
        let call = ToolCall {
            id: "x".into(),
            name: "bash".into(),
            arguments: serde_json::json!({ "command": "echo hello" }),
        };
        let with = ChatMessage::assistant_with_tool_calls(None, vec![call]);
        let without = ChatMessage::assistant("");
        assert!(tok.count_message_tokens(&with) > tok.count_message_tokens(&without));
    }

    #[test]
    fn messages_priming_is_included() {
        let tok = Tokenizer::new().unwrap();
        let m = ChatMessage::user("hi");
        assert_eq!(
            tok.count_messages(std::slice::from_ref(&m)),
            tok.count_message_tokens(&m) + 3
        );
    }

    #[test]
    fn counts_tool_definition_parameters() {
        // Tool schemas are part of every request; a large schema is a real
        // cost and must be measurable.
        let tok = Tokenizer::new().unwrap();
        let schema =
            serde_json::json!({ "type": "object", "properties": { "path": { "type": "string" } } });
        let def = ToolDefinition {
            name: "read_file".into(),
            description: "read".into(),
            parameters: schema,
        };
        let n = tok.count_tokens(&serde_json::to_string(&def.parameters).unwrap());
        assert!(n > 5);
    }

    #[test]
    fn take_tokens_prefers_line_boundaries() {
        let tok = Tokenizer::new().unwrap();
        let text = "line one\nline two\nline three";
        let taken = tok.take_tokens(text, 6);
        assert!(taken.starts_with("line one"), "{taken}");
        assert!(tok.count_tokens(&taken) <= 6, "{taken}");
    }

    #[test]
    fn take_tokens_handles_a_single_long_line() {
        let tok = Tokenizer::new().unwrap();
        let text = "a".repeat(4000);
        let taken = tok.take_tokens(&text, 10);
        assert!(!taken.is_empty());
        assert!(taken.len() < 4000);
    }

    #[test]
    fn approx_is_never_zero() {
        assert_eq!(approx_tokens(""), 0);
        assert_eq!(approx_tokens("abcd"), 1);
        assert_eq!(approx_tokens("abcde"), 2);
    }
}
