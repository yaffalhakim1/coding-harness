//! Inherent methods for [`ContextState`].
//!
//! Kept out of `strategy.rs` so the trait definition and the state machine read
//! as separate concerns.

use uuid::Uuid;

use axolotl_models::types::ChatMessage;

use super::{ContextEntry, ContextState};

impl ContextState {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            token_count: 0,
            preamble_len: 0,
            elided_regions: Vec::new(),
            summaries: Vec::new(),
            turn_starts: Vec::new(),
        }
    }

    /// Add a message, tracking its token cost.
    pub fn add_message(&mut self, message: ChatMessage, tokens: usize) {
        self.messages.push(ContextEntry {
            id: Uuid::new_v4().to_string(),
            message,
            tokens,
        });
        self.token_count += tokens;
    }

    /// Freeze the current entries as the preamble (system prompt + task).
    ///
    /// Later additions are middle/recent candidates.
    pub fn seal_preamble(&mut self) {
        self.preamble_len = self.messages.len();
    }

    /// Mark a new turn boundary at the current end of the log.
    pub fn start_turn(&mut self) {
        if self.turn_starts.last() == Some(&self.messages.len()) {
            return;
        }
        self.turn_starts.push(self.messages.len());
    }

    /// Index of the first entry belonging to the last `n` completed turns.
    ///
    /// A boundary equal to `messages.len()` is the open, not-yet-populated
    /// boundary for the next turn. Counting it would make "last 2 turns" mean
    /// "last 1 turn" on every call, so it is ignored here.
    ///
    /// Floored at the preamble so the recent window can never overlap it.
    pub fn recent_window_start(&self, n: usize) -> usize {
        let completed: Vec<usize> = self
            .turn_starts
            .iter()
            .copied()
            .filter(|&s| s < self.messages.len())
            .collect();

        let raw = if completed.is_empty() {
            self.messages.len()
        } else {
            let take = n.max(1).min(completed.len());
            completed[completed.len() - take]
        };
        raw.max(self.preamble_len)
    }

    /// Token count including the running summary, which is injected but is not
    /// a `messages` entry.
    pub fn total_tokens(&self) -> usize {
        self.token_count + self.summaries.iter().map(|s| s.tokens).sum::<usize>()
    }

    /// Fraction of `capacity` currently in use.
    pub fn usage_ratio(&self, capacity: usize) -> f32 {
        if capacity == 0 {
            return 1.0;
        }
        self.total_tokens() as f32 / capacity as f32
    }

    /// Build the message list to send to the LLM.
    ///
    /// Layout: the running summary (if any) is inserted immediately after the
    /// preamble, per paper §2.3.
    pub fn build_window(&self) -> Vec<ChatMessage> {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        let split = self.preamble_len.min(self.messages.len());

        messages.extend(self.messages[..split].iter().map(|e| e.message.clone()));
        if let Some(summary) = self.summaries.last() {
            messages.push(ChatMessage::system(summary.render()));
        }
        messages.extend(self.messages[split..].iter().map(|e| e.message.clone()));

        messages
    }

    /// True when the entry is in the compactable middle region.
    pub fn is_compactable(&self, idx: usize) -> bool {
        idx >= self.preamble_len && idx < self.messages.len()
    }

    /// Apply a token delta to the running total without recomputing.
    pub fn adjust_tokens(&mut self, delta: isize) {
        self.token_count = (self.token_count as isize + delta).max(0) as usize;
    }

    /// Replace an entry's message and update the token total.
    pub fn rewrite_entry(&mut self, idx: usize, message: ChatMessage, tokens: usize) {
        self.adjust_tokens(tokens as isize - self.messages[idx].tokens as isize);
        self.messages[idx].message = message;
        self.messages[idx].tokens = tokens;
    }

    /// Remove entries by index, keeping turn boundaries valid.
    pub fn remove_entries(&mut self, indices: &[usize]) {
        let mut sorted: Vec<usize> = indices.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        let mut freed = 0usize;
        for idx in sorted.iter().rev() {
            if *idx < self.messages.len() {
                freed += self.messages.remove(*idx).tokens;
            }
        }
        self.adjust_tokens(-(freed as isize));
        self.reindex_turns(&sorted);
    }

    /// Shift turn boundaries after removals, dropping boundaries for turns
    /// that no longer have any entries.
    fn reindex_turns(&mut self, removed: &[usize]) {
        let original_len = self.messages.len() + removed.len();
        let old_starts = self.turn_starts.clone();
        let mut new_starts = Vec::with_capacity(old_starts.len());

        for (turn_idx, start) in old_starts.iter().enumerate() {
            let end = old_starts
                .get(turn_idx + 1)
                .copied()
                .unwrap_or(original_len);
            let removed_in_turn = removed
                .iter()
                .filter(|i| **i >= *start && **i < end)
                .count();
            if removed_in_turn == end - start {
                continue; // whole turn removed
            }
            let shift = removed.iter().filter(|i| **i < *start).count();
            new_starts.push(start.saturating_sub(shift));
        }
        self.turn_starts = new_starts;
    }
}

impl Default for ContextState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::test_support::seeded_state;

    #[test]
    fn recent_window_respects_turn_boundaries() {
        let state = seeded_state(3, 10);
        // Boundaries [2, 4, 6, 8]; 8 is the open boundary for the next turn.
        // The last 2 *completed* turns are turns 1 and 2, so the window starts
        // at 4 and covers 4 entries.
        let start = state.recent_window_start(2);
        assert_eq!(start, 4);
        assert_eq!(state.messages.len() - start, 4);
    }

    #[test]
    fn recent_window_never_overlaps_preamble() {
        let state = seeded_state(1, 10);
        assert_eq!(state.recent_window_start(5), state.preamble_len);
    }

    #[test]
    fn summary_included_in_total_tokens() {
        let mut state = seeded_state(1, 10);
        let before = state.total_tokens();
        state.summaries.push(super::super::Summary {
            id: "s".into(),
            content: "x".into(),
            tokens: 7,
            entry_ids_absorbed: vec![],
            turns_absorbed: 1,
        });
        assert_eq!(state.total_tokens(), before + 7);
    }

    #[test]
    fn removing_entries_keeps_turn_starts_valid() {
        let mut state = seeded_state(3, 10);
        // Turn starts: [2, 4, 6, 8]. Remove turn 0 (indices 2 and 3).
        state.remove_entries(&[2, 3]);
        assert_eq!(state.messages.len(), 6);
        // Old turn 1 starts at 2, old turn 2 at 4. The open boundary that was
        // at 8 is now at 6 and refers to a turn with no entries, so it is
        // dropped rather than left pointing at a phantom turn.
        assert_eq!(state.turn_starts, vec![2, 4]);
        assert!(state.turn_starts.iter().all(|&s| s < state.messages.len()));
    }

    #[test]
    fn removing_an_entire_history_leaves_no_boundaries() {
        let mut state = seeded_state(2, 10);
        state.remove_entries(&[2, 3, 4, 5]);
        assert!(state.messages.len() <= 2);
        assert!(state.turn_starts.is_empty());
    }

    #[test]
    fn rewrite_entry_updates_totals() {
        let mut state = seeded_state(1, 100);
        let before = state.token_count;
        state.rewrite_entry(2, ChatMessage::assistant("short"), 1);
        assert_eq!(state.token_count, before - 4);
    }

    #[test]
    fn adjust_tokens_never_underflows() {
        let mut state = ContextState::new();
        state.adjust_tokens(-100);
        assert_eq!(state.token_count, 0);
    }

    #[test]
    fn start_turn_is_idempotent_at_a_boundary() {
        let mut state = seeded_state(1, 10);
        let before = state.turn_starts.len();
        state.start_turn();
        state.start_turn();
        assert_eq!(state.turn_starts.len(), before);
    }
}
