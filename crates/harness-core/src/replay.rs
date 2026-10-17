//! Replaying a previous session's history into a fresh context window.

use tracing::debug;

use harness_actions::PlanState;
use harness_context::factory::begin_turn;
use harness_context::{ContextState, Tokenizer};
use harness_models::types::ChatMessage;

use crate::session::Session;

/// Replay a previous session's turns into the context window.
///
/// Each prior turn contributes the assistant message it produced plus the tool
/// observations the model saw, which is exactly what the model needs to continue
/// rather than restart. Turn boundaries are preserved so the recent-window
/// protection still knows which turns are recent.
pub fn replay_history(
    state: &mut ContextState,
    previous: &Session,
    tokenizer: &Tokenizer,
    plan_state: Option<&PlanState>,
) {
    for turn in &previous.turns {
        begin_turn(state);
        let assistant = turn.response.message.clone();
        let tokens = tokenizer.count_message_tokens(&assistant);
        state.add_message(assistant, tokens);

        for result in &turn.tool_results {
            let observation =
                ChatMessage::tool_result(result.tool_call_id.clone(), result.observation());
            let tokens = tokenizer.count_message_tokens(&observation);
            state.add_message(observation, tokens);
        }
        begin_turn(state);
    }

    // Carry the plan forward, or the resumed run loses its scaffold.
    if let Some(plan_state) = plan_state {
        if plan_state.get().is_none() {
            if let Some(plan) = &previous.plan {
                plan_state.set(plan.clone());
            }
        }
    }

    debug!(
        replayed_turns = previous.turns.len(),
        tokens = state.total_tokens(),
        "Replayed prior session history"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionConfig;
    use crate::turn::Turn;
    use crate::types::Task;
    use harness_models::types::{LLMResponse, StopReason, TokenUsage, ToolCall};

    fn session_with_turn() -> Session {
        let mut session = Session::new(Task::from_description("t"), SessionConfig::default());
        let response = LLMResponse {
            message: ChatMessage::assistant_with_tool_calls(
                Some("thinking".into()),
                vec![ToolCall {
                    id: "c1".into(),
                    name: "read_file".into(),
                    arguments: serde_json::json!({}),
                }],
            ),
            usage: TokenUsage::default(),
            stop_reason: StopReason::ToolUse,
            model: "m".into(),
        };
        let mut turn = Turn::new(0, vec![], 0, response);
        turn.add_tool_result(crate::turn::ToolResult {
            tool_call_id: "c1".into(),
            tool_name: "read_file".into(),
            output: "MARKER".into(),
            success: true,
            error: None,
            duration_ms: 1,
        });
        session.record_turn(turn);
        session
    }

    #[test]
    fn replays_assistant_and_tool_messages() {
        let tok = Tokenizer::new().unwrap();
        let mut state = ContextState::new();
        replay_history(&mut state, &session_with_turn(), &tok, None);

        let text: String = state.messages.iter().map(|e| e.message.text()).collect();
        assert!(text.contains("thinking"), "{text}");
        assert!(text.contains("MARKER"), "{text}");
    }

    #[test]
    fn replay_preserves_turn_boundaries() {
        let tok = Tokenizer::new().unwrap();
        let mut state = ContextState::new();
        replay_history(&mut state, &session_with_turn(), &tok, None);
        // One turn replayed means at least one boundary is recorded.
        assert!(!state.turn_starts.is_empty());
    }

    #[test]
    fn replay_restores_the_plan_when_the_slot_is_empty() {
        let tok = Tokenizer::new().unwrap();
        let mut state = ContextState::new();
        let plan_state = PlanState::new();
        let mut session = session_with_turn();
        session.plan = Some("1. do it".into());

        replay_history(&mut state, &session, &tok, Some(&plan_state));
        assert_eq!(plan_state.get().as_deref(), Some("1. do it"));
    }

    #[test]
    fn replay_does_not_overwrite_an_existing_plan() {
        let tok = Tokenizer::new().unwrap();
        let mut state = ContextState::new();
        let plan_state = PlanState::new();
        plan_state.set("current".into());
        let mut session = session_with_turn();
        session.plan = Some("stale".into());

        replay_history(&mut state, &session, &tok, Some(&plan_state));
        assert_eq!(plan_state.get().as_deref(), Some("current"));
    }

    #[test]
    fn an_empty_session_replays_nothing() {
        let tok = Tokenizer::new().unwrap();
        let mut state = ContextState::new();
        let session = Session::new(Task::from_description("t"), SessionConfig::default());
        replay_history(&mut state, &session, &tok, None);
        assert!(state.messages.is_empty());
    }
}
