//! Elision mechanism (M1) and recall storage (M2).
//!
//! Paper §2.3: "Elision (M1) replaces the body of a stale tool observation with
//! a short stub. Recall (M2) stores elided observations in the file system and
//! exposes a `recall_event` tool to read them back on demand, making elision
//! reversible."
//!
//! Stub text is reproduced from paper Figure 19 so trajectories stay comparable
//! with the paper's.

use uuid::Uuid;

use harness_models::event_store::EventStore;
use harness_models::types::ChatMessage;

use crate::strategy::{ContextEntry, ContextState, ElidedRegion};
use crate::tokenizer::Tokenizer;

/// Minimum size before a tool observation counts as "bulky".
const MIN_BULKY_TOKENS: usize = 100;
/// Preview length kept in a stub for the model's benefit.
const PREVIEW_TOKENS: usize = 40;

/// Stub for T1 (no recall available).
pub fn stub_without_recall(lines: usize, chars: usize) -> String {
    format!(
        "[tool output elided: {lines} lines / {chars} chars. Re-read or re-run to get it again.]"
    )
}

/// Stub for T2/T4 (recall available), paper Figure 19.
pub fn stub_with_recall(lines: usize, chars: usize, event_id: &str) -> String {
    format!(
        "[tool output elided: {lines} lines / {chars} chars. Use recall_event({event_id}) for the full output, or re-read/re-run.]"
    )
}

/// Elides bulky tool observations in the middle region.
pub struct Elider<'a> {
    tokenizer: &'a Tokenizer,
    /// Retain originals so T2 can recall them.
    recall: bool,
    /// Where originals go so `recall_event` can serve them (M2).
    store: Option<EventStore>,
}

impl<'a> Elider<'a> {
    pub fn new(tokenizer: &'a Tokenizer, recall: bool) -> Self {
        Self {
            tokenizer,
            recall,
            store: None,
        }
    }

    /// Attach the external store that `recall_event` reads from.
    pub fn with_store(mut self, store: EventStore) -> Self {
        self.store = Some(store);
        self
    }

    /// Elide middle-region tool observations, oldest first, until the total
    /// falls to `target_tokens` or nothing compactable remains.
    ///
    /// Returns the number of tokens freed.
    pub fn elide_to_target(
        &self,
        state: &mut ContextState,
        target_tokens: usize,
        recent_start: usize,
    ) -> usize {
        let mut freed = 0usize;

        for idx in state.preamble_len..recent_start.min(state.messages.len()) {
            if state.total_tokens().saturating_sub(freed) <= target_tokens {
                break;
            }
            freed += self.elide_entry(state, idx);
        }

        freed
    }

    /// Elide one entry if eligible; returns tokens freed (0 when skipped).
    fn elide_entry(&self, state: &mut ContextState, idx: usize) -> usize {
        let entry = &state.messages[idx];
        if !entry.is_candidate_for_elision() || entry.tokens < MIN_BULKY_TOKENS {
            return 0;
        }

        let original = entry.message.text().to_string();
        let original_tokens = entry.tokens;
        let lines = original.lines().count();
        let chars = original.chars().count();

        let event_id = Uuid::new_v4().to_string();
        let stub = if self.recall {
            stub_with_recall(lines, chars, &event_id)
        } else {
            stub_without_recall(lines, chars)
        };
        let stub_tokens = self.tokenizer.count_tokens(&stub) + 4;

        if stub_tokens >= original_tokens {
            return 0; // nothing to gain
        }

        let entry_id = state.messages[idx].id.clone();
        let call_id = state.messages[idx]
            .message
            .tool_call_id
            .clone()
            .unwrap_or_default();

        state.rewrite_entry(idx, ChatMessage::tool_result(call_id, stub), stub_tokens);

        let preview = self.tokenizer.take_tokens(&original, PREVIEW_TOKENS);
        let freed = original_tokens - stub_tokens;

        // M2: keep the verbatim content recoverable. Written to the shared
        // store so the `recall_event` tool can serve it, not just the state.
        if self.recall {
            if let Some(store) = &self.store {
                store.insert(event_id.clone(), "tool output", original.clone());
            }
        }

        state.elided_regions.push(ElidedRegion {
            id: event_id,
            entry_ids: vec![entry_id],
            tokens_freed: freed,
            content_preview: preview,
            original: if self.recall { original } else { String::new() },
        });

        freed
    }

    /// Recover an elided region's original content by event id (M2).
    pub fn recall<'s>(state: &'s ContextState, event_id: &str) -> Option<&'s str> {
        state
            .elided_regions
            .iter()
            .find(|r| r.id == event_id && !r.original.is_empty())
            .map(|r| r.original.as_str())
    }
}

/// Total characters across the middle region, used for stub statistics.
pub fn middle_chars(state: &ContextState, recent_start: usize) -> usize {
    state
        .messages
        .iter()
        .take(recent_start.min(state.messages.len()))
        .skip(state.preamble_len)
        .map(|e: &ContextEntry| e.message.text().chars().count())
        .sum()
}
