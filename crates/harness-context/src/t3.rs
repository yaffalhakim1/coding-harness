//! T3: LLM summarization only (M3).
//!
//! Paper §2.3: "Tier 3 uses summarization alone (M3), folding the middle region
//! into a running summary without elision." Single-action tiers act at the hard
//! threshold B2.

use std::sync::Arc;

use tracing::debug;

use crate::compress::summarize_oldest_middle;
use crate::strategy::{ContextState, ContextStrategy, ContextTier};
use crate::summarizer::Summarizer;
use crate::thresholds::Thresholds;
use crate::tokenizer::Tokenizer;

pub struct T3Strategy {
    tokenizer: Tokenizer,
    thresholds: Thresholds,
    summarizer: Arc<dyn Summarizer>,
    /// Count of summarization calls, for cost comparison against T4.
    summarize_calls: std::sync::atomic::AtomicUsize,
}

impl T3Strategy {
    pub fn new(tokenizer: Tokenizer, summarizer: Arc<dyn Summarizer>) -> Self {
        Self {
            tokenizer,
            thresholds: Thresholds::default(),
            summarizer,
            summarize_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn with_thresholds(mut self, thresholds: Thresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// How many times M3 has run.
    pub fn summarize_calls(&self) -> usize {
        self.summarize_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[async_trait::async_trait]
impl ContextStrategy for T3Strategy {
    fn tier(&self) -> ContextTier {
        ContextTier::T3
    }

    async fn manage(&self, state: &mut ContextState, capacity: usize) -> anyhow::Result<()> {
        let trigger = self.thresholds.hard(capacity);
        if state.total_tokens() < trigger {
            return Ok(());
        }

        let recent_start = state.recent_window_start(self.thresholds.recent_turns);
        let target = self.thresholds.soft(capacity);

        debug!(
            tokens = state.total_tokens(),
            trigger, target, "T3: summarization threshold reached"
        );

        self.summarize_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        summarize_oldest_middle(
            state,
            self.summarizer.as_ref(),
            &self.tokenizer,
            recent_start,
            target,
        )
        .await;

        Ok(())
    }

    fn summarize_calls(&self) -> usize {
        T3Strategy::summarize_calls(self)
    }
}
