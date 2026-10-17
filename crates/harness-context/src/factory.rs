//! Construct the strategy for a tier from configuration.

use std::sync::Arc;

use anyhow::{Context, Result};

use harness_models::event_store::EventStore;
use harness_models::provider::LLMProvider;
use harness_models::types::ModelConfig;

use crate::strategy::{ContextState, ContextStrategy, ContextTier};
use crate::summarizer::{LlmSummarizer, Summarizer};
use crate::t0::T0Strategy;
use crate::t1::T1Strategy;
use crate::t2::T2Strategy;
use crate::t3::T3Strategy;
use crate::t4::T4Strategy;
use crate::thresholds::Thresholds;
use crate::tokenizer::Tokenizer;

/// Build the strategy for `tier`.
///
/// T3 and T4 need a model to summarize with. They receive an
/// [`LlmSummarizer`] wrapping the provider under evaluation, matching paper
/// §2.3 ("a separate, tool-free call to the same model under evaluation").
///
/// `store` is the external event store (M2). T2 and T4 write elided originals
/// into it so the `recall_event` tool can serve them.
pub fn build_strategy(
    tier: ContextTier,
    thresholds: Thresholds,
    tokenizer: Tokenizer,
    llm: Arc<dyn LLMProvider>,
    model: &ModelConfig,
    store: EventStore,
) -> Box<dyn ContextStrategy> {
    let summarizer: Arc<dyn Summarizer> = Arc::new(LlmSummarizer::new(llm, model.clone()));
    match tier {
        ContextTier::T0 => Box::new(T0Strategy::new()),
        ContextTier::T1 => Box::new(T1Strategy::new(tokenizer).with_thresholds(thresholds)),
        ContextTier::T2 => Box::new(
            T2Strategy::new(tokenizer)
                .with_thresholds(thresholds)
                .with_store(store),
        ),
        ContextTier::T3 => {
            Box::new(T3Strategy::new(tokenizer, summarizer).with_thresholds(thresholds))
        }
        ContextTier::T4 => Box::new(
            T4Strategy::new(tokenizer, summarizer)
                .with_thresholds(thresholds)
                .with_store(store),
        ),
    }
}

/// Create an empty context state with the tokenizer's first two messages sealed
/// as the preamble.
pub fn new_state_with_preamble(
    tokenizer: &Tokenizer,
    system: &str,
    task: &str,
) -> Result<ContextState> {
    let mut state = ContextState::new();
    let system_msg = harness_models::types::ChatMessage::system(system);
    let task_msg = harness_models::types::ChatMessage::user(task);

    let system_tokens = tokenizer
        .count_message_tokens(&system_msg)
        .checked_add(0)
        .context("system prompt token count overflowed")?;
    let task_tokens = tokenizer.count_message_tokens(&task_msg);

    state.add_message(system_msg, system_tokens);
    state.add_message(task_msg, task_tokens);
    state.seal_preamble();
    Ok(state)
}

/// Prepare a state so the first turn appends at a valid boundary.
pub fn begin_turn(state: &mut ContextState) {
    state.start_turn();
}
