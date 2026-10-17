//! Tests for the summarization mechanism.

use std::sync::Arc;

use harness_models::mock::{MockProvider, ScriptedStep};
use harness_models::types::{ChatMessage, ModelConfig};

use super::*;
use crate::strategy::ContextEntry;

fn entry(role_text: &str) -> ContextEntry {
    ContextEntry {
        id: "e".into(),
        message: ChatMessage::user(role_text),
        tokens: 10,
    }
}

#[test]
fn prompt_matches_the_paper_headings() {
    // Paper Figure 18 requires these exact headings; drift here silently
    // changes what the model produces compared to the paper.
    for heading in [
        "## Goal",
        "## Files touched",
        "## Done",
        "## Pending",
        "## Errors & fixes",
        "## Current state",
        "## Next step",
    ] {
        assert!(SUMMARY_SYSTEM_PROMPT.contains(heading), "missing {heading}");
    }
    assert!(SUMMARY_SYSTEM_PROMPT.contains("Respond with TEXT ONLY"));
}

#[test]
fn user_message_includes_previous_summary_when_present() {
    let events = vec![entry("did stuff")];
    let msg = build_user_message(Some("OLD SUMMARY"), &events);
    assert!(msg.contains("OLD SUMMARY"));
    assert!(msg.contains("did stuff"));
}

#[test]
fn user_message_omits_previous_section_when_absent() {
    let msg = build_user_message(None, &[entry("x")]);
    assert!(!msg.contains("Previous summary"));
}

#[test]
fn render_events_includes_tool_call_arguments() {
    let entry = ContextEntry {
        id: "e".into(),
        message: ChatMessage::assistant_with_tool_calls(
            Some("looking".into()),
            vec![harness_models::types::ToolCall {
                id: "1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({ "path": "src/main.rs" }),
            }],
        ),
        tokens: 5,
    };
    let rendered = render_events(&[entry]);
    assert!(rendered.contains("[assistant] looking"));
    assert!(rendered.contains("read_file"));
    assert!(rendered.contains("src/main.rs"));
}

#[test]
fn llm_summarizer_calls_the_model_and_returns_text() {
    let provider = MockProvider::new(vec![ScriptedStep::text("## Goal\nfixed")]);
    let summarizer = LlmSummarizer::new(provider.clone(), ModelConfig::default());
    let events = vec![entry("old work")];

    let out = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(summarizer.summarize(None, &events))
        .unwrap();

    assert!(out.contains("## Goal"));
    assert_eq!(provider.request_count(), 1);

    // The request must be tool-free and carry the paper's system prompt.
    let sent = provider.requests();
    assert_eq!(sent[0][0].text(), SUMMARY_SYSTEM_PROMPT);
}

#[test]
fn empty_events_is_an_error_without_a_model_call() {
    let provider = MockProvider::new(vec![]);
    let summarizer = LlmSummarizer::new(provider.clone(), ModelConfig::default());
    let err = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(summarizer.summarize(None, &[]))
        .unwrap_err();
    assert!(matches!(err, ContextError::SummaryFailed { .. }), "{err:?}");
    assert_eq!(provider.request_count(), 0);
}

#[test]
fn provider_failure_is_reported() {
    let provider = MockProvider::new(vec![ScriptedStep::Fail("network down".into())]);
    let summarizer: Arc<dyn Summarizer> =
        Arc::new(LlmSummarizer::new(provider, ModelConfig::default()));
    let err = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(summarizer.summarize(None, &[entry("x")]))
        .unwrap_err();
    assert!(matches!(err, ContextError::SummaryFailed { .. }), "{err:?}");
}
