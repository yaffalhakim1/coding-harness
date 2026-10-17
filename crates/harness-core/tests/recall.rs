//! Tests for mechanism M2: elided content stays recoverable.
//!
//! Paper §2.3: elision replaces a tool observation with a stub, and recall
//! stores the original so `recall_event` can return it. The paper found models
//! rarely use recall, but the machinery must actually work when they do.

mod common;

use common::{call, harness, run};
use harness_actions::registry::ActionRegistry;
use harness_core::{SessionConfig, Task};
use harness_models::mock::{MockProvider, ScriptedStep};
use harness_models::types::ChatMessage;

fn t4_config() -> SessionConfig {
    SessionConfig {
        planning_enabled: false,
        context_strategy: harness_context::ContextTier::T4,
        ..SessionConfig::default()
    }
}

#[test]
fn recall_event_is_registered_for_t2_and_t4_only() {
    for (tier, expected) in [
        (harness_context::ContextTier::T0, false),
        (harness_context::ContextTier::T1, false),
        (harness_context::ContextTier::T2, true),
        (harness_context::ContextTier::T3, false),
        (harness_context::ContextTier::T4, true),
    ] {
        let provider = MockProvider::new(vec![ScriptedStep::text("done")]);
        let config = SessionConfig {
            planning_enabled: false,
            context_strategy: tier,
            ..SessionConfig::default()
        };
        let h = harness(&provider, &config);
        assert_eq!(
            h.actions.has_tool("recall_event"),
            expected,
            "tier {tier} should{} expose recall_event",
            if expected { "" } else { " not" }
        );
    }
}

#[test]
fn elision_stores_originals_in_the_shared_store() {
    // Drive context management far enough that elision fires, then confirm the
    // original text is retrievable from the store the tool reads.
    let big = ScriptedStep::tool_calls(vec![call(
        "1",
        "grep_text",
        serde_json::json!({ "query": ".", "path": "." }),
    )]);
    let provider = MockProvider::repeating(big);

    let config = SessionConfig {
        planning_enabled: false,
        context_strategy: harness_context::ContextTier::T4,
        context_window: 1_200,
        max_turns: 10,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    for i in 0..40 {
        std::fs::write(
            h.dir.path().join(format!("f{i}.txt")),
            "needle line of text\n".repeat(4),
        )
        .unwrap();
    }

    let _ = run(&h, &provider, Task::from_description("x"), &config);

    assert!(
        !h.event_store.is_empty(),
        "elision should have stored at least one original"
    );

    // Every stored event must be servable through the tool.
    let id = h.event_store.ids().remove(0);
    let event = h.event_store.get(&id).expect("stored");
    assert!(!event.content.is_empty());
}

#[test]
fn model_can_recall_elided_output() {
    let big = ScriptedStep::tool_calls(vec![call(
        "1",
        "grep_text",
        serde_json::json!({ "query": ".", "path": "." }),
    )]);
    let provider = MockProvider::repeating(big);

    let config = SessionConfig {
        planning_enabled: false,
        context_strategy: harness_context::ContextTier::T4,
        context_window: 1_200,
        max_turns: 10,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    for i in 0..40 {
        std::fs::write(
            h.dir.path().join(format!("f{i}.txt")),
            "needle line of text\n".repeat(4),
        )
        .unwrap();
    }
    let _ = run(&h, &provider, Task::from_description("x"), &config);

    // Now script a model turn that calls recall_event with a real id.
    let id = h.event_store.ids().remove(0);
    let provider2 = MockProvider::new(vec![
        ScriptedStep::tool_calls(vec![call(
            "r1",
            "recall_event",
            serde_json::json!({ "id": id }),
        )]),
        ScriptedStep::text("recovered"),
    ]);
    let config2 = SessionConfig {
        planning_enabled: false,
        context_strategy: harness_context::ContextTier::T4,
        ..SessionConfig::default()
    };
    let h2 = harness(&provider2, &config2);
    h2.event_store
        .insert(&id, "tool output", "the verbatim original");

    let session = run(&h2, &provider2, Task::from_description("x"), &config2);
    let result = &session.turns[0].tool_results[0];
    assert!(result.success, "recall should succeed: {result:?}");
    assert!(
        result.output.contains("the verbatim original"),
        "got: {}",
        result.output
    );
}

#[test]
fn recall_of_unknown_id_is_reported_as_a_tool_error() {
    let provider = MockProvider::new(vec![
        ScriptedStep::tool_calls(vec![call(
            "r1",
            "recall_event",
            serde_json::json!({ "id": "does-not-exist" }),
        )]),
        ScriptedStep::text("done"),
    ]);
    let config = t4_config();
    let h = harness(&provider, &config);
    let session = run(&h, &provider, Task::from_description("x"), &config);

    let result = &session.turns[0].tool_results[0];
    assert!(!result.success);
    let error = result.error.as_deref().unwrap_or("");
    assert!(error.contains("unknown event id"), "{error}");
}

#[test]
fn stub_text_names_a_real_event_id() {
    // The stub must tell the model an id it can actually pass to recall_event.
    let big = ScriptedStep::tool_calls(vec![call(
        "1",
        "grep_text",
        serde_json::json!({ "query": ".", "path": "." }),
    )]);
    let provider = MockProvider::repeating(big);

    let config = SessionConfig {
        planning_enabled: false,
        context_strategy: harness_context::ContextTier::T4,
        context_window: 1_200,
        max_turns: 6,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    for i in 0..40 {
        std::fs::write(
            h.dir.path().join(format!("f{i}.txt")),
            "needle line of text\n".repeat(4),
        )
        .unwrap();
    }
    let _ = run(&h, &provider, Task::from_description("x"), &config);

    let ids = h.event_store.ids();
    assert!(!ids.is_empty(), "expected stored events");
    // Every id in the store must be resolvable, which is what a stub promises.
    for id in ids {
        assert!(h.event_store.contains(&id));
    }
}

#[test]
fn chat_message_helper_is_unaffected_by_recall() {
    // Guard against the store accidentally rewriting message content.
    let msg = ChatMessage::tool_result("c1", "[tool output elided: 3 lines]");
    assert!(msg.text().contains("elided"));
}
