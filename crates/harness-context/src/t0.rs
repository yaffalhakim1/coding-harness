//! T0: No context management (baseline).
//!
//! Paper §2.3: "Tier 0 disables context management; trajectories that outgrow
//! the window terminate with an error."
//!
//! Nothing is dropped or compacted. `manage` returns `Err` once the window is
//! exceeded, which the agent loop maps to a `ContextOverflow` outcome. This is
//! what makes the managed tiers' "zero overflow failures" measurable.

use tracing::warn;

use crate::strategy::{ContextError, ContextState, ContextStrategy, ContextTier};

pub struct T0Strategy;

impl T0Strategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for T0Strategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ContextStrategy for T0Strategy {
    fn tier(&self) -> ContextTier {
        ContextTier::T0
    }

    async fn manage(&self, state: &mut ContextState, capacity: usize) -> anyhow::Result<()> {
        let tokens = state.total_tokens();
        if tokens > capacity {
            warn!(
                tokens,
                capacity, "T0: context window exceeded — terminating (no management)"
            );
            return Err(ContextError::Overflow { tokens, capacity }.into());
        }
        Ok(())
    }
}
