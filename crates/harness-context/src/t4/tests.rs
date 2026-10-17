//! Tier tests driven by a deterministic summarizer.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use harness_models::types::ChatMessage;

use crate::strategy::{test_support::seeded_state, ContextEntry, ContextError, ContextState};
use crate::strategy::{ContextStrategy, ContextTier};
use crate::summarizer::Summarizer;
use crate::t0::T0Strategy;
use crate::t1::T1Strategy;
use crate::t2::T2Strategy;
use crate::t3::T3Strategy;
use crate::t4::T4Strategy;
use crate::tokenizer::Tokenizer;

/// Records how often it was called and returns fixed-length text.
struct FakeSummarizer {
    calls: Mutex<Vec<String>>,
    output: String,
}

impl FakeSummarizer {
    fn new(output: &str) -> Arc<Self> {
        Arc::new(Self {
            calls: Mutex::new(Vec::new()),
            output: output.to_string(),
        })
    }

    fn calls(&self) -> usize {
        self.calls.lock().map(|c| c.len()).unwrap_or(0)
    }
}

#[async_trait]
impl Summarizer for FakeSummarizer {
    async fn summarize(
        &self,
        previous: Option<&str>,
        events: &[ContextEntry],
    ) -> Result<String, ContextError> {
        if let Ok(mut calls) = self.calls.lock() {
            calls.push(format!(
                "previous={} events={}",
                previous.is_some(),
                events.len()
            ));
        }
        Ok(self.output.clone())
    }
}

/// A summarizer that always fails, to prove failure is non-destructive.
struct FailingSummarizer;

#[async_trait]
impl Summarizer for FailingSummarizer {
    async fn summarize(
        &self,
        _previous: Option<&str>,
        _events: &[ContextEntry],
    ) -> Result<String, ContextError> {
        Err(ContextError::SummaryFailed {
            reason: "boom".into(),
        })
    }
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

#[test]
fn t0_terminates_on_overflow() {
    let strategy = T0Strategy::new();
    let mut state = seeded_state(5, 1000);
    let capacity = 3000;
    assert!(state.total_tokens() > capacity);

    let err = rt()
        .block_on(strategy.manage(&mut state, capacity))
        .unwrap_err();
    assert!(err.to_string().contains("context overflow"), "{err}");
    // Nothing is dropped: T0 is the unmanaged baseline.
    assert_eq!(state.messages.len(), 12);
}

#[test]
fn t0_does_nothing_below_capacity() {
    let strategy = T0Strategy::new();
    let mut state = seeded_state(2, 10);
    let before = state.total_tokens();
    rt().block_on(strategy.manage(&mut state, 100_000)).unwrap();
    assert_eq!(state.total_tokens(), before);
    assert_eq!(state.messages.len(), 6);
}

#[test]
fn t1_elides_tool_outputs_only() {
    let tok = Tokenizer::new().unwrap();
    let strategy = T1Strategy::new(tok);
    let mut state = seeded_state(6, 800);
    let capacity = 4000;

    rt().block_on(strategy.manage(&mut state, capacity))
        .unwrap();

    for entry in &state.messages {
        let is_tool = entry.message.role == harness_models::types::Role::Tool;
        if !is_tool {
            assert!(
                !entry.message.text().contains("tool output elided"),
                "non-tool entry was elided: {}",
                entry.message.text()
            );
        }
    }
    assert!(!state.elided_regions.is_empty(), "expected elisions");
    assert!(state.total_tokens() < 6 * 800 + 20);
}

#[test]
fn t1_never_elides_the_preamble() {
    let tok = Tokenizer::new().unwrap();
    let strategy = T1Strategy::new(tok);
    let mut state = seeded_state(8, 900);
    rt().block_on(strategy.manage(&mut state, 4000)).unwrap();

    assert_eq!(state.messages[0].message.text(), "system prompt");
    assert_eq!(state.messages[1].message.text(), "task");
}

#[test]
fn t1_keeps_the_recent_window_verbatim() {
    let tok = Tokenizer::new().unwrap();
    let strategy = T1Strategy::new(tok);
    let mut state = seeded_state(8, 900);
    let recent_start = state.recent_window_start(2);
    let recent_before: Vec<String> = state.messages[recent_start..]
        .iter()
        .map(|e| e.message.text().to_string())
        .collect();

    rt().block_on(strategy.manage(&mut state, 4000)).unwrap();

    let recent_after: Vec<String> = state.messages[recent_start..]
        .iter()
        .map(|e| e.message.text().to_string())
        .collect();
    assert_eq!(
        recent_before, recent_after,
        "recent turns must stay verbatim"
    );
}

#[test]
fn t1_stub_text_matches_the_paper() {
    let long = "x".repeat(4000);
    let mut state = ContextState::new();
    state.add_message(ChatMessage::system("s"), 1);
    state.seal_preamble();
    state.start_turn();
    state.add_message(ChatMessage::tool_result("c1", long), 1000);
    // Two later turns, so the bulky observation falls out of the 2-turn
    // verbatim window and becomes elidable.
    state.start_turn();
    state.add_message(ChatMessage::assistant("next"), 5);
    state.start_turn();
    state.add_message(ChatMessage::assistant("done"), 5);

    let tok = Tokenizer::new().unwrap();
    let strategy = T1Strategy::new(tok);
    rt().block_on(strategy.manage(&mut state, 200)).unwrap();

    let stub = state.messages[1].message.text();
    assert!(
        stub.starts_with("[tool output elided:"),
        "unexpected stub: {stub}"
    );
    assert!(stub.contains("Re-read or re-run"), "{stub}");
    assert!(!stub.contains("recall_event"), "T1 has no recall: {stub}");
}

#[test]
fn recent_turns_are_never_elided() {
    // Regression: a single-turn history must not be compacted even when far
    // over budget, because the recent window protects it.
    let long = "z".repeat(4000);
    let mut state = ContextState::new();
    state.add_message(ChatMessage::system("s"), 1);
    state.seal_preamble();
    state.start_turn();
    state.add_message(ChatMessage::tool_result("c1", long.clone()), 1000);

    let tok = Tokenizer::new().unwrap();
    let strategy = T1Strategy::new(tok);
    rt().block_on(strategy.manage(&mut state, 200)).unwrap();

    assert_eq!(
        state.messages[1].message.text(),
        long,
        "recent tool output was elided"
    );
    assert!(state.elided_regions.is_empty());
}

#[test]
fn t2_stores_originals_for_recall() {
    let long = "y".repeat(4000);
    let mut state = ContextState::new();
    state.add_message(ChatMessage::system("s"), 1);
    state.seal_preamble();
    state.start_turn();
    state.add_message(ChatMessage::tool_result("c1", long.clone()), 1000);
    state.start_turn();
    state.add_message(ChatMessage::assistant("next"), 5);
    state.start_turn();
    state.add_message(ChatMessage::assistant("done"), 5);

    let tok = Tokenizer::new().unwrap();
    let strategy = T2Strategy::new(tok);
    rt().block_on(strategy.manage(&mut state, 200)).unwrap();

    let stub = state.messages[1].message.text();
    assert!(
        stub.contains("recall_event"),
        "T2 stub must offer recall: {stub}"
    );

    let event = state.elided_regions.first().expect("region recorded");
    let recovered = strategy.recall(&state, &event.id).expect("recall works");
    assert_eq!(recovered, long);
}

#[test]
fn t3_summarizes_with_the_model() {
    let tok = Tokenizer::new().unwrap();
    let fake = FakeSummarizer::new("## Goal\nfix it\n## Next step\nedit file");
    let strategy = T3Strategy::new(tok, fake.clone());
    let mut state = seeded_state(6, 800);

    rt().block_on(strategy.manage(&mut state, 4000)).unwrap();

    assert_eq!(fake.calls(), 1);
    assert_eq!(state.summaries.len(), 1);
    assert!(state.summaries[0].content.contains("## Goal"));
    // The counter must match the model calls, or the T3/T4 cost comparison is
    // comparing a real number against a fabricated zero.
    assert_eq!(strategy.summarize_calls(), 1);
    assert_eq!(strategy.summarize_calls(), fake.calls());
}

#[test]
fn t3_summary_is_injected_after_preamble() {
    let tok = Tokenizer::new().unwrap();
    let fake = FakeSummarizer::new("SUMMARY BODY");
    let strategy = T3Strategy::new(tok, fake);
    let mut state = seeded_state(6, 800);
    rt().block_on(strategy.manage(&mut state, 4000)).unwrap();

    let window = state.build_window();
    assert_eq!(window[0].role, harness_models::types::Role::System);
    assert_eq!(window[1].role, harness_models::types::Role::User);
    assert!(window[2].text().contains("SUMMARY BODY"), "{:?}", window[2]);
}

#[test]
fn summary_failure_leaves_history_intact() {
    let tok = Tokenizer::new().unwrap();
    let strategy = T3Strategy::new(tok, Arc::new(FailingSummarizer));
    let mut state = seeded_state(6, 800);
    let before = state.messages.len();

    // Must not error out or drop anything.
    rt().block_on(strategy.manage(&mut state, 4000)).unwrap();
    assert_eq!(state.messages.len(), before);
    assert!(state.summaries.is_empty());
}

#[test]
fn t4_elides_before_summarizing() {
    let tok = Tokenizer::new().unwrap();
    let fake = FakeSummarizer::new("## Goal\nx");
    let strategy = T4Strategy::new(tok, fake.clone());
    // Enough middle tool output that elision alone gets under B2.
    let mut state = seeded_state(4, 900);
    let capacity = 5000;

    rt().block_on(strategy.manage(&mut state, capacity))
        .unwrap();

    assert!(
        !state.elided_regions.is_empty(),
        "T4 must elide before considering summarization"
    );
    assert_eq!(fake.calls(), 0, "elision alone should have sufficed");
    assert_eq!(strategy.summarize_calls(), 0);
    assert!(state.total_tokens() < capacity);
}

#[test]
fn t4_summarizes_when_elision_is_not_enough() {
    let tok = Tokenizer::new().unwrap();
    let fake = FakeSummarizer::new("## Goal\nx");
    let strategy = T4Strategy::new(tok, fake.clone());
    // No tool observations at all: elision has nothing to work with.
    let mut state = ContextState::new();
    state.add_message(ChatMessage::system("s"), 10);
    state.add_message(ChatMessage::user("task"), 10);
    state.seal_preamble();
    for t in 0..10 {
        state.start_turn();
        state.add_message(ChatMessage::assistant(format!("reasoning {t}")), 800);
    }
    state.start_turn();
    state.add_message(ChatMessage::assistant("now"), 10);

    rt().block_on(strategy.manage(&mut state, 3000)).unwrap();

    assert_eq!(fake.calls(), 1, "summarization must fire");
    assert_eq!(strategy.summarize_calls(), 1);
    assert_eq!(state.summaries.len(), 1);
    assert!(state.total_tokens() < 3000, "{}", state.total_tokens());
}

#[test]
fn t4_does_nothing_below_b1() {
    let tok = Tokenizer::new().unwrap();
    let fake = FakeSummarizer::new("x");
    let strategy = T4Strategy::new(tok, fake.clone());
    let mut state = seeded_state(2, 10);
    let before = state.total_tokens();

    rt().block_on(strategy.manage(&mut state, 100_000)).unwrap();
    assert_eq!(state.total_tokens(), before);
    assert_eq!(fake.calls(), 0);
}

#[test]
fn t4_forwards_previous_summary_to_next_call() {
    let tok = Tokenizer::new().unwrap();
    let fake = FakeSummarizer::new("## Goal\nx");
    let strategy = T4Strategy::new(tok, fake.clone());
    let mut state = ContextState::new();
    state.add_message(ChatMessage::system("s"), 10);
    state.seal_preamble();
    for t in 0..8 {
        state.start_turn();
        state.add_message(ChatMessage::assistant(format!("reasoning {t}")), 800);
    }
    state.start_turn();
    state.add_message(ChatMessage::assistant("now"), 10);

    rt().block_on(strategy.manage(&mut state, 3000)).unwrap();
    // More reasoning pushes it over again.
    for t in 8..16 {
        state.start_turn();
        state.add_message(ChatMessage::assistant(format!("reasoning {t}")), 800);
    }
    state.start_turn();
    state.add_message(ChatMessage::assistant("now2"), 10);
    rt().block_on(strategy.manage(&mut state, 3000)).unwrap();

    let calls = fake.calls.lock().unwrap().clone();
    assert!(calls.len() >= 2, "{calls:?}");
    assert!(
        calls[1].contains("previous=true"),
        "second call must merge: {calls:?}"
    );
}

#[test]
fn tier_identifiers_are_reported() {
    let tok = Tokenizer::new().unwrap();
    assert_eq!(T0Strategy::new().tier(), ContextTier::T0);
    assert_eq!(T1Strategy::new(tok.clone()).tier(), ContextTier::T1);
    assert_eq!(T2Strategy::new(tok.clone()).tier(), ContextTier::T2);
    assert_eq!(
        T3Strategy::new(tok.clone(), FakeSummarizer::new("x")).tier(),
        ContextTier::T3
    );
    assert_eq!(
        T4Strategy::new(tok, FakeSummarizer::new("x")).tier(),
        ContextTier::T4
    );
}
