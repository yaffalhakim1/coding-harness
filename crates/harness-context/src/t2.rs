//! T2: Elision + recoverable recall (M1 + M2).
//!
//! Paper §2.3: "Tier 2 adds recall (M2): elided observations are stored
//! externally and recoverable via `recall_event`, making elision reversible."
//!
//! The paper found this "rarely invoked and does not improve accuracy over
//! elision alone", so T2 reuses T1's machinery and only differs by retaining the
//! originals in [`crate::strategy::ElidedRegion::original`].

use tracing::debug;

use harness_models::event_store::EventStore;

use crate::elision::Elider;
use crate::strategy::{ContextState, ContextStrategy, ContextTier};
use crate::thresholds::Thresholds;
use crate::tokenizer::Tokenizer;

pub struct T2Strategy {
    tokenizer: Tokenizer,
    thresholds: Thresholds,
    /// Where elided originals go so `recall_event` can serve them (M2).
    store: EventStore,
}

impl T2Strategy {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            thresholds: Thresholds::default(),
            store: EventStore::new(),
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

    /// Recover the verbatim content of an elided region by event id.
    pub fn recall<'a>(&self, state: &'a ContextState, event_id: &str) -> Option<&'a str> {
        Elider::recall(state, event_id)
    }
}

#[async_trait::async_trait]
impl ContextStrategy for T2Strategy {
    fn tier(&self) -> ContextTier {
        ContextTier::T2
    }

    async fn manage(&self, state: &mut ContextState, capacity: usize) -> anyhow::Result<()> {
        let trigger = self.thresholds.hard(capacity);
        if state.total_tokens() < trigger {
            return Ok(());
        }

        let recent_start = state.recent_window_start(self.thresholds.recent_turns);
        let target = self.thresholds.soft(capacity);
        let freed = Elider::new(&self.tokenizer, true)
            .with_store(self.store.clone())
            .elide_to_target(state, target, recent_start);

        debug!(
            freed_tokens = freed,
            recallable = state.elided_regions.len(),
            "T2: elided with recall storage"
        );
        Ok(())
    }
}
