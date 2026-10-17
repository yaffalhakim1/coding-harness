//! Deterministic in-memory provider for tests and offline runs.
//!
//! Downstream crates (core, context, bench) drive the agent loop against this
//! provider so their tests never touch the network.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::provider::{LLMProvider, ProviderError, ProviderResult};
use crate::types::{ChatMessage, LLMResponse, ModelConfig, StopReason, TokenUsage, ToolDefinition};

/// A scripted response or a scripted failure.
#[derive(Debug, Clone)]
pub enum ScriptedStep {
    Respond(LLMResponse),
    Fail(String),
}

impl ScriptedStep {
    /// Script an assistant message that issues tool calls.
    pub fn tool_calls(calls: Vec<crate::types::ToolCall>) -> Self {
        Self::Respond(LLMResponse {
            message: ChatMessage::assistant_with_tool_calls(None, calls),
            usage: TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
            },
            stop_reason: StopReason::ToolUse,
            model: "mock".into(),
        })
    }

    /// Script a plain assistant text turn.
    pub fn text(content: impl Into<String>) -> Self {
        Self::Respond(LLMResponse {
            message: ChatMessage::assistant(content),
            usage: TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
            },
            stop_reason: StopReason::EndTurn,
            model: "mock".into(),
        })
    }
}

/// Provider that replays a fixed script and records every request it received.
pub struct MockProvider {
    script: Mutex<VecDeque<ScriptedStep>>,
    requests: Mutex<Vec<Vec<ChatMessage>>>,
    /// Emitted when the script runs out, so a runaway loop fails loudly
    /// instead of silently repeating the last step.
    fallback: Mutex<ScriptedStep>,
}

impl MockProvider {
    pub fn new(script: Vec<ScriptedStep>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into()),
            requests: Mutex::new(Vec::new()),
            fallback: Mutex::new(ScriptedStep::text("script exhausted")),
        })
    }

    /// Script that returns one tool call, then a final text turn.
    pub fn once(step: ScriptedStep) -> Arc<Self> {
        Self::new(vec![step, ScriptedStep::text("done")])
    }

    /// Replace the fallback used when the script runs out.
    ///
    /// Infallible: the provider is shared behind an `Arc`, so a consuming
    /// builder would be unusable in the places it matters.
    pub fn set_fallback(&self, step: ScriptedStep) {
        if let Ok(mut slot) = self.fallback.lock() {
            *slot = step;
        }
    }

    /// Convenience constructor: a provider that repeats `step` forever.
    pub fn repeating(step: ScriptedStep) -> Arc<Self> {
        let provider = Self::new(Vec::new());
        provider.set_fallback(step);
        provider
    }

    /// Every message list passed to `chat`, in order.
    pub fn requests(&self) -> Vec<Vec<ChatMessage>> {
        self.requests.lock().map(|g| g.clone()).unwrap_or_default()
    }

    pub fn request_count(&self) -> usize {
        self.requests.lock().map(|g| g.len()).unwrap_or(0)
    }
}

#[async_trait]
impl LLMProvider for MockProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        _tools: &[ToolDefinition],
        _config: &ModelConfig,
    ) -> ProviderResult<LLMResponse> {
        if let Ok(mut log) = self.requests.lock() {
            log.push(messages.to_vec());
        }

        let next = self
            .script
            .lock()
            .ok()
            .and_then(|mut s| s.pop_front())
            .or_else(|| self.fallback.lock().ok().map(|f| f.clone()))
            .unwrap_or_else(|| ScriptedStep::text("script exhausted"));

        match next {
            ScriptedStep::Respond(response) => Ok(response),
            ScriptedStep::Fail(message) => Err(ProviderError::Transport(message)),
        }
    }

    fn name(&self) -> &str {
        "mock"
    }

    async fn health_check(&self, _config: &ModelConfig) -> ProviderResult<bool> {
        Ok(true)
    }
}
