//! T1: Rule-based elision only (M1).
//!
//! Paper §2.3: "Tier 1 uses elision alone (M1), replacing stale tool-observation
//! bodies with short stubs and discarding the original content." Because T1 has
//! only one action, it operates at the hard threshold B2.

use tracing::{debug, info};

use crate::elision::Elider;
use crate::strategy::{ContextState, ContextStrategy, ContextTier};
use crate::thresholds::Thresholds;
use crate::tokenizer::Tokenizer;

pub struct T1Strategy {
    tokenizer: Tokenizer,
    thresholds: Thresholds,
}

impl T1Strategy {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            thresholds: Thresholds::default(),
        }
    }

    pub fn with_thresholds(mut self, thresholds: Thresholds) -> Self {
        self.thresholds = thresholds;
        self
    }
}

#[async_trait::async_trait]
impl ContextStrategy for T1Strategy {
    fn tier(&self) -> ContextTier {
        ContextTier::T1
    }

    async fn manage(&self, state: &mut ContextState, capacity: usize) -> anyhow::Result<()> {
        // Single-action tiers act at the hard threshold (paper §2.3).
        let trigger = self.thresholds.hard(capacity);
        if state.total_tokens() < trigger {
            return Ok(());
        }

        let recent_start = state.recent_window_start(self.thresholds.recent_turns);
        let target = self.thresholds.soft(capacity);

        debug!(
            tokens = state.total_tokens(),
            trigger, target, "T1: elision threshold reached"
        );

        let freed =
            Elider::new(&self.tokenizer, false).elide_to_target(state, target, recent_start);
        info!(
            freed_tokens = freed,
            remaining = state.total_tokens(),
            "T1: elided tool outputs"
        );
        Ok(())
    }
}
