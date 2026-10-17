//! LLM provider trait and error types.

use async_trait::async_trait;

use crate::types::{ChatMessage, LLMResponse, ModelConfig, ToolDefinition};

/// Errors a provider can surface.
///
/// `retryable` distinguishes transient failures (rate limits, 5xx, network)
/// from permanent ones (bad request, auth) so the adapter can back off only
/// where it helps.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("missing model configuration: {0}")]
    Config(String),

    #[error("API key environment variable '{0}' is not set")]
    MissingApiKey(String),

    #[error("authentication failed ({status}): {body}")]
    Auth { status: u16, body: String },

    #[error("rate limited ({status}): {body}")]
    RateLimited { status: u16, body: String },

    #[error("server error ({status}): {body}")]
    Server { status: u16, body: String },

    #[error("invalid request ({status}): {body}")]
    BadRequest { status: u16, body: String },

    #[error("transport failure: {0}")]
    Transport(String),

    #[error("could not decode response: {0}")]
    Decode(String),

    #[error("response contained no choices")]
    EmptyResponse,

    #[error("retries exhausted after {attempts} attempts: {source}")]
    RetriesExhausted {
        attempts: u32,
        #[source]
        source: Box<ProviderError>,
    },
}

impl ProviderError {
    /// Whether repeating the identical request could plausibly succeed.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ProviderError::RateLimited { .. }
                | ProviderError::Server { .. }
                | ProviderError::Transport(_)
        )
    }
}

pub type ProviderResult<T> = std::result::Result<T, ProviderError>;

/// Unified interface for LLM providers.
///
/// Implementations handle:
/// - API authentication
/// - Request serialization (tool definitions, messages)
/// - Response parsing (message, usage, stop reason)
/// - Retry logic with exponential backoff
#[async_trait]
pub trait LLMProvider: Send + Sync {
    /// Send a chat completion request.
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        config: &ModelConfig,
    ) -> ProviderResult<LLMResponse>;

    /// Provider name (e.g., "openai", "anthropic").
    fn name(&self) -> &str;

    /// Check if the provider is configured and reachable.
    async fn health_check(&self, config: &ModelConfig) -> ProviderResult<bool>;
}

/// Backoff schedule for retryable provider failures.
///
/// Exponential with full jitter: `delay = random(0, base * 2^attempt)`, capped at
/// `max`. Full jitter (rather than a linear ramp) is what keeps a fleet of
/// batch runs from retrying in lockstep after a shared rate-limit window.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub base_delay: std::time::Duration,
    pub max_delay: std::time::Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            base_delay: std::time::Duration::from_millis(500),
            max_delay: std::time::Duration::from_secs(30),
        }
    }
}

impl RetryPolicy {
    /// Delay before retry number `attempt` (0-based).
    pub fn delay(&self, attempt: u32) -> std::time::Duration {
        let exp = self.base_delay.as_millis() as u64 * 2u64.saturating_pow(attempt.min(10));
        let capped = exp.min(self.max_delay.as_millis() as u64);
        if capped == 0 {
            return std::time::Duration::ZERO;
        }
        std::time::Duration::from_millis(rand::Rng::gen_range(&mut rand::thread_rng(), 0..=capped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_classification() {
        assert!(ProviderError::RateLimited {
            status: 429,
            body: String::new()
        }
        .is_retryable());
        assert!(ProviderError::Server {
            status: 503,
            body: String::new()
        }
        .is_retryable());
        assert!(ProviderError::Transport("reset".into()).is_retryable());

        assert!(!ProviderError::Auth {
            status: 401,
            body: String::new()
        }
        .is_retryable());
        assert!(!ProviderError::BadRequest {
            status: 400,
            body: String::new()
        }
        .is_retryable());
        assert!(!ProviderError::MissingApiKey("X".into()).is_retryable());
    }

    #[test]
    fn delay_is_capped() {
        let policy = RetryPolicy {
            base_delay: std::time::Duration::from_millis(10),
            max_delay: std::time::Duration::from_millis(50),
        };
        for attempt in 0..12 {
            assert!(policy.delay(attempt) <= std::time::Duration::from_millis(50));
        }
    }

    #[test]
    fn first_delay_never_exceeds_base() {
        let policy = RetryPolicy::default();
        for _ in 0..50 {
            assert!(policy.delay(0) <= policy.base_delay);
        }
    }
}
