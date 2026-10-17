//! Summarization mechanism (M3) from paper §2.3 and Appendix A.3.
//!
//! The summary is produced by a separate, tool-free call to the same model under
//! evaluation, using the prompt in Figure 18. The previous running summary is
//! supplied alongside the newly selected events so the model updates rather than
//! re-derives it.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tracing::{debug, warn};

use harness_models::provider::LLMProvider;
use harness_models::types::{ChatMessage, ModelConfig};

use crate::strategy::{ContextEntry, ContextError};

/// Summarization prompt, reproduced verbatim from paper Figure 18.
pub const SUMMARY_SYSTEM_PROMPT: &str = "\
You are compacting a coding agent's conversation to save context. You are given the OLDEST part of the
transcript (and possibly a previous summary to update).

Respond with TEXT ONLY — do NOT call any tools. Produce a concise structured summary under these
exact headings (omit a heading only if truly empty):
## Goal
The user's task / intent (preserve it precisely).
## Files touched
Paths examined or modified, with the key changes made and why.
## Done
What has been accomplished, with concrete results (tests passing, bug located, ...).
## Pending
What still needs doing.
## Errors & fixes
Errors hit and how they were resolved (or are still open).
## Current state
Key variables / branch / test status / anything needed to continue.
## Next step
The immediate next action, in line with the most recent work.
Be specific (keep exact file paths, function names, error messages, task ids). If a previous summary is given,
UPDATE it: keep still-true facts, drop stale ones, merge in the new events. The raw events remain
retrievable, so summarize — don't transcribe.";

/// Turns a batch of history into a running summary.
#[async_trait]
pub trait Summarizer: Send + Sync {
    /// Summarize `events`, merging `previous` when present.
    async fn summarize(
        &self,
        previous: Option<&str>,
        events: &[ContextEntry],
    ) -> Result<String, ContextError>;
}

/// Render events into the transcript block sent to the summarizer.
pub fn render_events(events: &[ContextEntry]) -> String {
    let mut out = String::new();
    for entry in events {
        let role = match entry.message.role {
            harness_models::types::Role::System => "system",
            harness_models::types::Role::User => "user",
            harness_models::types::Role::Assistant => "assistant",
            harness_models::types::Role::Tool => "tool",
        };
        let mut line = format!("[{role}] {}", entry.message.text());
        if let Some(calls) = &entry.message.tool_calls {
            for call in calls {
                line.push_str(&format!("\n  -> {} {}", call.name, call.arguments));
            }
        }
        out.push_str(&line);
        out.push_str("\n\n");
    }
    out
}

/// Build the user message containing previous summary + new events.
pub fn build_user_message(previous: Option<&str>, events: &[ContextEntry]) -> String {
    let mut user = String::new();
    if let Some(prev) = previous {
        user.push_str("## Previous summary (update it)\n");
        user.push_str(prev);
        user.push_str("\n\n");
    }
    user.push_str("## Oldest transcript events to fold in\n");
    user.push_str(&render_events(events));
    user
}

/// LLM-backed summarizer: one tool-free call to the model under evaluation.
pub struct LlmSummarizer {
    llm: Arc<dyn LLMProvider>,
    config: ModelConfig,
    timeout: Duration,
}

impl LlmSummarizer {
    pub fn new(llm: Arc<dyn LLMProvider>, config: ModelConfig) -> Self {
        let mut config = config;
        // A summary is a compact rewrite of known material. Allowing the same
        // output budget as a normal turn would make M3 the dominant cost.
        config.max_tokens = Some(config.max_tokens.unwrap_or(1024).min(2048));
        config.temperature = 0.0;
        Self {
            llm,
            config,
            timeout: Duration::from_secs(90),
        }
    }

    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.config.max_tokens = Some(tokens);
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl Summarizer for LlmSummarizer {
    async fn summarize(
        &self,
        previous: Option<&str>,
        events: &[ContextEntry],
    ) -> Result<String, ContextError> {
        if events.is_empty() {
            return Err(ContextError::SummaryFailed {
                reason: "nothing to summarize".into(),
            });
        }

        let messages = vec![
            ChatMessage::system(SUMMARY_SYSTEM_PROMPT),
            ChatMessage::user(build_user_message(previous, events)),
        ];

        debug!(
            events = events.len(),
            max_tokens = self.config.max_tokens,
            "Requesting context summary"
        );

        let response =
            tokio::time::timeout(self.timeout, self.llm.chat(&messages, &[], &self.config))
                .await
                .map_err(|_| ContextError::SummaryTimeout {
                    seconds: self.timeout.as_secs(),
                })?
                .map_err(|e| ContextError::SummaryFailed {
                    reason: e.to_string(),
                })?;

        let text = response
            .message
            .content
            .filter(|c| !c.trim().is_empty())
            .ok_or_else(|| ContextError::SummaryFailed {
                reason: "model returned an empty summary".into(),
            })?;

        if !text.contains("## ") {
            warn!("context summary did not follow the requested headings");
        }
        Ok(text)
    }
}

#[cfg(test)]
mod tests;
