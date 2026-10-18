//! T4: Elision -> Summarization (recommended).
//!
//! The paper's best-performing strategy (§3.2): "Staging elision before LLM
//! summarization (T4) provides the strongest efficiency among the
//! context-management strategies."
//!
//! Algorithm 1:
//! 1. Above soft threshold B1, elide bulky tool observations in the middle
//!    region (M1) and store the originals (M2).
//! 2. If still above hard threshold B2, summarize the oldest middle events into
//!    the running summary (M3).
//!
//! Staging matters: elision is free and reclaims most of the space, so
//! summarization — the expensive model call — only fires when elision was not
//! enough. That is the whole cost advantage over T3.

use std::sync::Arc;

use tracing::{debug, info};

use crate::compress::summarize_oldest_middle;
use axolotl_models::event_store::EventStore;

use crate::elision::Elider;
use crate::strategy::{ContextState, ContextStrategy, ContextTier};
use crate::summarizer::Summarizer;
use crate::thresholds::Thresholds;
use crate::tokenizer::Tokenizer;

pub struct T4Strategy {
    tokenizer: Tokenizer,
    thresholds: Thresholds,
    summarizer: Arc<dyn Summarizer>,
    /// Where elided originals go so `recall_event` can serve them (M2).
    store: EventStore,
    /// Count of summarization calls, for cost reporting.
    summarize_calls: std::sync::atomic::AtomicUsize,
}

impl T4Strategy {
    pub fn new(tokenizer: Tokenizer, summarizer: Arc<dyn Summarizer>) -> Self {
        Self {
            tokenizer,
            thresholds: Thresholds::default(),
            summarizer,
            store: EventStore::new(),
            summarize_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn with_thresholds(mut self, thresholds: Thresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Share the event store used by the `recall_event` tool.
    pub fn with_store(mut self, store: EventStore) -> Self {
        self.store = store;
        self
    }

    /// How many times M3 has run. Elision-first means this stays near zero
    /// relative to T3.
    pub fn summarize_calls(&self) -> usize {
        self.summarize_calls
            .load(std::sync::atomic::Ordering::Relaxed)
    }
    /// Recover an elided region's original content (M2).
    pub fn recall<'a>(&self, state: &'a ContextState, event_id: &str) -> Option<&'a str> {
        Elider::recall(state, event_id)
    }
}

#[async_trait::async_trait]
impl ContextStrategy for T4Strategy {
    fn tier(&self) -> ContextTier {
        ContextTier::T4
    }

    fn summarize_calls(&self) -> usize {
        T4Strategy::summarize_calls(self)
    }

    async fn manage(&self, state: &mut ContextState, capacity: usize) -> anyhow::Result<()> {
        let soft = self.thresholds.soft(capacity);
        let tokens = state.total_tokens();

        if tokens < soft {
            return Ok(()); // below B1, nothing to do
        }

        let recent_start = state.recent_window_start(self.thresholds.recent_turns);

        // Stage 1 (M1 + M2): cheap elision down to B1.
        let freed_by_elision = Elider::new(&self.tokenizer, true)
            .with_store(self.store.clone())
            .elide_to_target(state, soft, recent_start);

        let hard = self.thresholds.hard(capacity);
        if state.total_tokens() < hard {
            debug!(
                freed_tokens = freed_by_elision,
                remaining = state.total_tokens(),
                "T4: elision sufficient — no summarization call"
            );
            return Ok(());
        }

        // Stage 2 (M3): expensive, only now.
        self.summarize_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let freed_by_summary = summarize_oldest_middle(
            state,
            self.summarizer.as_ref(),
            &self.tokenizer,
            recent_start,
            soft,
        )
        .await;

        info!(
            freed_by_elision,
            freed_by_summary,
            remaining = state.total_tokens(),
            "T4: staged management complete"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests;
