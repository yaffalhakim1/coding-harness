//! axolotl-models: LLM provider adapters
//!
//! Provides a unified interface for interacting with different LLM providers
//! (OpenAI, Anthropic, local models via OpenAI-compatible APIs).

pub mod event_store;
pub mod failover;
pub mod mock;
pub mod openai;
pub mod provider;
pub mod types;

pub use event_store::{EventStore, StoredEvent};
pub use failover::{FailoverProvider, Route};
pub use openai::OpenAIProvider;
pub use provider::{LLMProvider, ProviderError, ProviderResult, RetryPolicy};
pub use types::*;
