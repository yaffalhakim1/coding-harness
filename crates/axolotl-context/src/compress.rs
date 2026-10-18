//! Summarization-to-running-summary helper shared by T3 and T4.

use tracing::{info, warn};

use crate::strategy::{ContextState, Summary};
use crate::summarizer::Summarizer;
use crate::tokenizer::Tokenizer;

/// Fold the oldest middle-region entry groups into the running summary.
///
/// Returns the number of tokens freed. The caller decides when to call this;
/// this function only performs one summarization round.
///
/// A round absorbs whole turns at a time, never a fragment of a turn, so the
/// transcript sent to the summarizer is always coherent.
pub async fn summarize_oldest_middle(
    state: &mut ContextState,
    summarizer: &dyn Summarizer,
    tokenizer: &Tokenizer,
    recent_start: usize,
    target_tokens: usize,
) -> usize {
    let end = recent_start.min(state.messages.len());
    if state.preamble_len >= end {
        return 0; // nothing in the middle
    }

    // Absorb whole turns from the oldest middle turn until enough is gone.
    let mut absorb_end = state.preamble_len;
    for turn_start in state.turn_starts.iter().copied() {
        if turn_start <= state.preamble_len || turn_start >= end {
            continue;
        }
        absorb_end = turn_start;
        let absorbed: usize = state.messages[state.preamble_len..absorb_end]
            .iter()
            .map(|e| e.tokens)
            .sum();
        if state.total_tokens().saturating_sub(absorbed) <= target_tokens {
            break;
        }
    }

    if absorb_end <= state.preamble_len {
        return 0;
    }

    let events: Vec<_> = state.messages[state.preamble_len..absorb_end].to_vec();
    let absorbed_tokens: usize = events.iter().map(|e| e.tokens).sum();
    if absorbed_tokens == 0 {
        return 0;
    }

    let previous = state.summaries.last().map(|s| s.content.clone());
    let turns_absorbed = state
        .turn_starts
        .iter()
        .filter(|s| **s > state.preamble_len && **s < absorb_end)
        .count()
        .max(1);

    let summary_text = match summarizer.summarize(previous.as_deref(), &events).await {
        Ok(text) => text,
        Err(e) => {
            // A failed summarization must not corrupt the transcript: leave the
            // events in place and let the next turn retry.
            warn!(error = %e, "Summarization failed — keeping history verbatim");
            return 0;
        }
    };

    let summary_tokens = tokenizer.count_tokens(&summary_text) + 8;
    let absorbed_ids: Vec<String> = events.iter().map(|e| e.id.clone()).collect();

    // Absorbed entries now live only inside the summary.
    let indices: Vec<usize> = (state.preamble_len..absorb_end).collect();
    state.remove_entries(&indices);

    let summary = Summary {
        id: uuid::Uuid::new_v4().to_string(),
        content: summary_text,
        tokens: summary_tokens,
        entry_ids_absorbed: absorbed_ids,
        turns_absorbed,
    };

    // Exactly one running summary; older ones were already folded into it.
    state.summaries.clear();
    state.summaries.push(summary);

    let freed = absorbed_tokens.saturating_sub(summary_tokens);
    info!(
        events = events.len(),
        turns = turns_absorbed,
        freed_tokens = freed,
        "Summarized oldest middle events"
    );
    freed
}
