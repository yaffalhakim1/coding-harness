//! Cross-crate integration tests for the individual components.
//!
//! The full agent loop is covered in `crates/axolotl-core/tests/agent_loop.rs`;
//! this file checks that the pieces compose as documented.

use axolotl_context::strategy::{ContextState, ContextStrategy, ContextTier};
use axolotl_context::t0::T0Strategy;
use axolotl_context::t4::T4Strategy;
use axolotl_models::types::ChatMessage;

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Runtime::new().unwrap()
}

#[test]
fn context_state_basic() {
    let mut state = ContextState::new();
    assert_eq!(state.token_count, 0);
    assert!(state.messages.is_empty());

    state.add_message(ChatMessage::user("hello"), 5);
    assert_eq!(state.token_count, 5);
    assert_eq!(state.messages.len(), 1);
}

#[test]
fn context_state_build_window() {
    let mut state = ContextState::new();
    state.add_message(ChatMessage::system("You are a helper"), 10);
    state.add_message(ChatMessage::user("Help me"), 5);
    state.add_message(ChatMessage::assistant("Sure!"), 3);

    let window = state.build_window();
    assert_eq!(window.len(), 3);
}

#[test]
fn t0_performs_no_management_below_capacity() {
    let strategy = T0Strategy::new();
    let mut state = ContextState::new();

    for i in 0..10 {
        state.add_message(ChatMessage::user(format!("message {i}")), 100);
    }
    assert_eq!(state.token_count, 1000);

    rt().block_on(strategy.manage(&mut state, 2000)).unwrap();
    assert_eq!(state.token_count, 1000);
    assert_eq!(state.messages.len(), 10);
}

#[test]
fn t0_terminates_on_overflow() {
    // Paper §2.3: T0 is unmanaged and terminates when the window is exceeded.
    // It must not silently drop history, or the paper's baseline comparison
    // measures the wrong thing.
    let strategy = T0Strategy::new();
    let mut state = ContextState::new();
    state.add_message(ChatMessage::system("system"), 50);
    for i in 0..20 {
        state.add_message(ChatMessage::user(format!("message {i}")), 100);
    }
    assert_eq!(state.token_count, 2050);

    let err = rt().block_on(strategy.manage(&mut state, 1000)).unwrap_err();
    assert!(err.to_string().contains("context overflow"), "{err}");
    assert_eq!(state.messages.len(), 21, "T0 must not drop messages");
}

#[test]
fn context_tier_display() {
    assert_eq!(ContextTier::T0.to_string(), "T0");
    assert_eq!(ContextTier::T4.to_string(), "T4");
    assert_eq!(ContextTier::parse("t4"), Some(ContextTier::T4));
}

#[test]
fn t4_strategy_creation_reports_its_tier() {
    use axolotl_context::summarizer::{Summarizer, SUMMARY_SYSTEM_PROMPT};
    use axolotl_context::{ContextEntry, Thresholds};
    use axolotl_context::tokenizer::Tokenizer;
    use std::sync::Arc;

    struct Noop;
    #[async_trait::async_trait]
    impl Summarizer for Noop {
        async fn summarize(
            &self,
            _previous: Option<&str>,
            _events: &[ContextEntry],
        ) -> Result<String, axolotl_context::ContextError> {
            Ok(String::new())
        }
    }

    let strategy = T4Strategy::new(Tokenizer::new().unwrap(), Arc::new(Noop))
        .with_thresholds(Thresholds {
            soft_ratio: 0.6,
            hard_ratio: 0.85,
            recent_ratio: 0.3,
            recent_turns: 2,
        });

    assert_eq!(strategy.tier(), ContextTier::T4);
    assert_eq!(strategy.summarize_calls(), 0);
    assert!(SUMMARY_SYSTEM_PROMPT.contains("## Goal"));
}

#[test]
fn thresholds_match_the_paper() {
    let t = axolotl_context::Thresholds::default();
    assert!((t.soft_ratio - 0.6).abs() < f32::EPSILON);
    assert!((t.hard_ratio - 0.85).abs() < f32::EPSILON);
    assert!((t.recent_ratio - 0.3).abs() < f32::EPSILON);
    assert_eq!(t.recent_turns, 2);
}

#[test]
fn default_config_file_is_valid() {
    let cfg = axolotl_core::SessionConfig::default();
    assert_eq!(cfg.context_strategy, ContextTier::T4);
    assert_eq!(cfg.context_window, 128_000);
}
