//! Context strategy trait, state types, and errors.
//!
//! Region model from paper §2.3 / Algorithm 1 line 5:
//!
//! ```text
//! [ preamble        | middle (compactable)        | recent (verbatim) ]
//!  system + task     oldest ... newest-turn-N      last >= 2 turns
//! ```
//!
//! Elision (M1/M2) and summarization (M3) only ever touch the middle region.
//! The preamble is never elided — eliding the task description is how a managed
//! run turns into a T0 failure with extra steps.
//!
//! `ContextState`'s inherent methods live in [`state`] to keep this file focused
//! on the trait.

mod state;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use harness_models::types::ChatMessage;

/// Which context management tier to use.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContextTier {
    /// T0: No management — raw append, terminate on overflow
    T0,
    /// T1: Rule-based elision only
    T1,
    /// T2: Elision + recoverable recall storage
    T2,
    /// T3: LLM summarization only
    T3,
    /// T4: Elision -> Summarization (recommended)
    T4,
}

impl ContextTier {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_uppercase().as_str() {
            "T0" => Some(Self::T0),
            "T1" => Some(Self::T1),
            "T2" => Some(Self::T2),
            "T3" => Some(Self::T3),
            "T4" => Some(Self::T4),
            _ => None,
        }
    }
}

impl std::fmt::Display for ContextTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ContextTier::T0 => "T0",
            ContextTier::T1 => "T1",
            ContextTier::T2 => "T2",
            ContextTier::T3 => "T3",
            ContextTier::T4 => "T4",
        };
        f.write_str(name)
    }
}

/// State of the context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextState {
    /// Messages currently in the context window, in conversation order.
    pub messages: Vec<ContextEntry>,
    /// Sum of `tokens` over all entries in `messages`.
    pub token_count: usize,
    /// Number of leading entries that form the preamble (system + task).
    pub preamble_len: usize,
    /// Regions that were elided (for T2 recall).
    pub elided_regions: Vec<ElidedRegion>,
    /// Running summary of old context (T3/T4). At most one entry.
    pub summaries: Vec<Summary>,
    /// Turn boundaries: index of the first entry of each turn.
    pub turn_starts: Vec<usize>,
}

/// A single entry in the context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEntry {
    pub id: String,
    pub message: ChatMessage,
    pub tokens: usize,
}

impl ContextEntry {
    pub fn is_tool_output(&self) -> bool {
        self.message.role == harness_models::types::Role::Tool
    }

    /// M1 elides *tool observations*, not assistant reasoning or the task.
    pub fn is_candidate_for_elision(&self) -> bool {
        self.is_tool_output()
    }
}

/// A region of elided content, retained for T2 recall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElidedRegion {
    pub id: String,
    pub entry_ids: Vec<String>,
    pub tokens_freed: usize,
    pub content_preview: String,
    /// Full original content, so T2 can recover it (empty for T1).
    pub original: String,
}

/// The running summary of old context (for T3/T4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    pub id: String,
    pub content: String,
    pub tokens: usize,
    pub entry_ids_absorbed: Vec<String>,
    /// Number of turns folded in, for reporting.
    pub turns_absorbed: usize,
}

impl Summary {
    /// Text inserted into the model input (paper Fig. 19, M3 wrapper).
    pub fn render(&self) -> String {
        format!(
            "<context-summary>\n{}\n(Older events were summarized. Their raw content is recallable via recall_event.)\n</context-summary>",
            self.content
        )
    }
}

/// The context management strategy trait.
///
/// `manage` is async because M3 (summarization) calls the model. Tiers that do
/// not summarize complete without awaiting anything.
#[async_trait]
pub trait ContextStrategy: Send + Sync {
    /// The tier identifier.
    fn tier(&self) -> ContextTier;

    /// Apply context management to the current state.
    ///
    /// Returns `Err` when the window cannot be brought back under capacity.
    /// T0 uses this to signal overflow; managed tiers compact instead.
    async fn manage(&self, state: &mut ContextState, capacity: usize) -> Result<()>;

    /// Inject a plan into the context.
    ///
    /// The plan lives outside conversation history and is re-supplied each turn
    /// (paper §2.1), so it is appended to the protected preamble region and is
    /// never summarized away.
    fn inject_plan(&self, state: &mut ContextState, plan_text: &str, tokens: usize) {
        let msg = ChatMessage::system(format!(
            "<system-reminder>\nCurrent plan (update it via update_plan as you progress; keep exactly one task in_progress, and mark a task completed the moment it is actually done):\n{plan_text}\n</system-reminder>"
        ));
        state.add_message(msg, tokens);
        state.preamble_len = state.messages.len();
    }

    /// Current token count.
    fn token_count(&self, state: &ContextState) -> usize {
        state.token_count
    }

    /// How many summarization calls (M3) this strategy has made.
    ///
    /// T4 stages elision before summarization, so this counter is the direct
    /// measure of its cost advantage over T3 (paper §3.2).
    fn summarize_calls(&self) -> usize {
        0
    }
}

/// Errors raised by context management.
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("context overflow: {tokens} tokens exceed the {capacity}-token budget")]
    Overflow { tokens: usize, capacity: usize },

    #[error("summarization failed: {reason}")]
    SummaryFailed { reason: String },

    #[error("summarization timed out after {seconds}s")]
    SummaryTimeout { seconds: u64 },
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Seed a state with a preamble and `turns` turns of two messages each.
    pub fn seeded_state(turns: usize, tool_tokens: usize) -> ContextState {
        let mut state = ContextState::new();
        state.add_message(ChatMessage::system("system prompt"), 10);
        state.add_message(ChatMessage::user("task"), 10);
        state.seal_preamble();

        for t in 0..turns {
            state.start_turn();
            state.add_message(ChatMessage::assistant(format!("thinking {t}")), 5);
            state.add_message(
                ChatMessage::tool_result(format!("call-{t}"), "x".repeat(tool_tokens * 4)),
                tool_tokens,
            );
        }
        state.start_turn();
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harness_models::types::Role;

    #[test]
    fn preamble_is_sealed() {
        let state = test_support::seeded_state(2, 100);
        assert_eq!(state.preamble_len, 2);
        assert!(!state.is_compactable(0));
        assert!(!state.is_compactable(1));
        assert!(state.is_compactable(2));
    }

    #[test]
    fn summary_is_injected_after_preamble() {
        let mut state = test_support::seeded_state(1, 10);
        state.summaries.push(Summary {
            id: "s".into(),
            content: "earlier stuff".into(),
            tokens: 5,
            entry_ids_absorbed: vec![],
            turns_absorbed: 1,
        });
        let window = state.build_window();
        assert_eq!(window[0].role, Role::System);
        assert_eq!(window[1].role, Role::User);
        assert_eq!(
            window[2].role,
            Role::System,
            "summary must follow the preamble"
        );
        assert!(window[2].text().contains("earlier stuff"));
        assert_eq!(window.len(), state.messages.len() + 1);
    }

    #[test]
    fn usage_ratio_of_empty_state_is_zero() {
        assert_eq!(ContextState::new().usage_ratio(1000), 0.0);
    }

    #[test]
    fn tier_parsing() {
        assert_eq!(ContextTier::parse("t4"), Some(ContextTier::T4));
        assert_eq!(ContextTier::parse("T0"), Some(ContextTier::T0));
        assert_eq!(ContextTier::parse("T9"), None);
    }

    #[test]
    fn summary_render_uses_the_figure_19_wrapper() {
        let s = Summary {
            id: "s".into(),
            content: "body".into(),
            tokens: 1,
            entry_ids_absorbed: vec![],
            turns_absorbed: 1,
        };
        let rendered = s.render();
        assert!(rendered.starts_with("<context-summary>"));
        assert!(rendered.contains("body"));
        assert!(rendered.trim_end().ends_with("</context-summary>"));
    }
}
