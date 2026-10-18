//! axolotl-context: Context window management strategies.
//!
//! Implements the T0-T4 context management tiers from the paper:
//! - T0: No management (baseline; terminates on overflow)
//! - T1: Rule-based elision only
//! - T2: Elision + recoverable recall
//! - T3: LLM summarization only
//! - T4: Elision -> Summarization (best accuracy/cost tradeoff)

pub mod compress;
pub mod elision;
pub mod factory;
pub mod strategy;
pub mod summarizer;
pub mod t0;
pub mod t1;
pub mod t2;
pub mod t3;
pub mod t4;
pub mod thresholds;
pub mod tokenizer;

pub use factory::{begin_turn, build_strategy, new_state_with_preamble};
pub use strategy::{
    ContextEntry, ContextError, ContextState, ContextStrategy, ContextTier, ElidedRegion, Summary,
};
pub use summarizer::{LlmSummarizer, Summarizer};
pub use thresholds::Thresholds;
pub use tokenizer::{approx_tokens, Tokenizer};
