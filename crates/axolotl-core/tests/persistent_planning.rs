//! Tests for the persistent planning scaffold (paper §2.1).
//!
//! The paper's planning scaffold has four pieces:
//! 1. System instruction defining the protocol (Figure 15)
//! 2. First-turn reminder requesting an initial plan (Figure 16)
//! 3. `update_plan` tool for the model to revise the plan (Figure 30)
//! 4. Plan injection before each turn (Figure 17)

mod common;

use axolotl_actions::registry::ActionRegistry;
use axolotl_core::{SessionConfig, Task};
use axolotl_models::mock::{MockProvider, ScriptedStep};
use common::{call, harness, run};

#[test]
fn update_plan_tool_is_registered_when_planning_enabled() {
    let provider = MockProvider::new(vec![ScriptedStep::text("done")]);
    let config = SessionConfig {
        planning_enabled: true,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    assert!(h.actions.has_tool("update_plan"));
}

#[test]
fn update_plan_tool_is_not_registered_when_planning_disabled() {
    let provider = MockProvider::new(vec![ScriptedStep::text("done")]);
    let config = SessionConfig {
        planning_enabled: false,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    assert!(!h.actions.has_tool("update_plan"));
}

#[test]
fn model_can_call_update_plan_during_execution() {
    let provider = MockProvider::new(vec![
        // Call 1 is the planner: with planning on, the loop generates an
        // initial plan before the first agent turn.
        ScriptedStep::text("1. Read file\n2. Edit\n3. Test"),
        // Turn 1: the model revises the plan through the tool.
        ScriptedStep::tool_calls(vec![call(
            "1",
            "update_plan",
            serde_json::json!({ "plan": "1. Read file\n2. Edit\n3. Test" }),
        )]),
        // Turn 2: model signals completion.
        ScriptedStep::text("Plan created, task complete."),
    ]);

    let config = SessionConfig {
        planning_enabled: true,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let session = run(&h, &provider, Task::from_description("x"), &config);

    let plan_state = h
        .plan_state
        .as_ref()
        .expect("plan_state exists when planning is enabled");
    let plan = plan_state.get().expect("the tool stored a plan");
    assert!(plan.contains("2. Edit"), "{plan}");

    assert_eq!(session.turn_count(), 2);
}

#[test]
fn plan_is_reinjected_before_each_turn() {
    // The plan must be re-supplied before every model call (paper Figure 17),
    // not only once at the start, so revisions take effect immediately.
    let provider = MockProvider::new(vec![
        // Call 1: planner.
        ScriptedStep::text("initial plan"),
        // Turn 1: revise the plan.
        ScriptedStep::tool_calls(vec![call(
            "1",
            "update_plan",
            serde_json::json!({ "plan": "initial plan" }),
        )]),
        // Turn 2: do some work.
        ScriptedStep::tool_calls(vec![call(
            "2",
            "list_files",
            serde_json::json!({ "path": "." }),
        )]),
        // Turn 3: revise again.
        ScriptedStep::tool_calls(vec![call(
            "3",
            "update_plan",
            serde_json::json!({ "plan": "updated plan" }),
        )]),
        // Turn 4: finish.
        ScriptedStep::text("done"),
    ]);

    let config = SessionConfig {
        planning_enabled: true,
        ..SessionConfig::default()
    };
    let h = harness(&provider, &config);
    let session = run(&h, &provider, Task::from_description("x"), &config);

    let plan = h
        .plan_state
        .as_ref()
        .and_then(|s| s.get())
        .expect("a plan was stored");
    assert_eq!(plan, "updated plan");

    assert_eq!(session.turn_count(), 4);

    // The revised plan must appear in the request made *after* the update,
    // which is what makes the scaffold persistent rather than one-shot.
    let requests = provider.requests();
    assert!(
        requests[2]
            .iter()
            .any(|m| m.text().contains("initial plan")),
        "turn after the first update must see the plan"
    );
    assert!(
        requests[4]
            .iter()
            .any(|m| m.text().contains("updated plan")),
        "turn after the second update must see the revised plan"
    );
}
