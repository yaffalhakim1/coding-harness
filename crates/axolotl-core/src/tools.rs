//! Tool execution within a turn.
//!
//! Paper §2.2 / Table 1: read-only tools "may execute concurrently within a
//! model turn", and §3.1 caps that at eight per step. Mutating tools run
//! sequentially in emission order, because a write followed by an edit of the
//! same file is order-sensitive.
//!
//! Concurrency is applied to *runs* of adjacent read-only calls, not to the
//! whole turn: a `write_file` between two `read_file`s must still observe the
//! first read's effect on file state, so order has to be preserved across it.

use std::sync::Arc;
use std::time::Instant;

use futures::future::join_all;
use tracing::{info, warn};

use axolotl_actions::registry::ActionRegistry;
use axolotl_context::{ContextState, Tokenizer};
use axolotl_models::types::{ChatMessage, ToolCall};

use crate::stuck::{StuckDetector, StuckSignal};
use crate::turn::{ToolResult, Turn};

/// Paper §3.1: "up to eight read-only tools may run in parallel per step."
pub const MAX_PARALLEL_READS: usize = 8;

/// Execute every tool call in `turn`, feeding observations back into `state`.
///
/// Returns the strongest [`StuckSignal`] observed. An error in one call never
/// aborts the loop: it becomes an observation the model can act on.
pub async fn execute_tool_calls(
    turn: &mut Turn,
    actions: &dyn ActionRegistry,
    state: &mut ContextState,
    tokenizer: &Tokenizer,
    stuck: &mut StuckDetector,
    progress: Option<&Arc<dyn crate::progress::ProgressSink>>,
) -> StuckSignal {
    let mut strongest = StuckSignal::Continue { reminder: None };
    let calls = turn.tool_calls.clone();

    for batch in group_into_batches(&calls, actions) {
        let results = if batch.len() == 1 {
            vec![run_one(batch[0], actions, progress).await]
        } else {
            // Adjacent read-only calls are independent, so fan them out.
            join_all(batch.iter().map(|call| run_one(call, actions, progress))).await
        };

        for (call, result) in batch.iter().zip(results) {
            let observation = ChatMessage::tool_result(call.id.clone(), result.observation());
            let tokens = tokenizer.count_message_tokens(&observation);
            state.add_message(observation, tokens);

            let signal = stuck.observe(&call.name, &call.arguments, !result.success);
            strongest = merge(strongest, signal);
            turn.add_tool_result(result);
        }
    }

    if let StuckSignal::Continue {
        reminder: Some(text),
    } = &strongest
    {
        let msg = ChatMessage::user(text.clone());
        let tokens = tokenizer.count_message_tokens(&msg);
        state.add_message(msg, tokens);
    }

    strongest
}

/// Split calls into execution batches.
///
/// Each batch is either one mutating call (run alone, in order) or a run of up
/// to [`MAX_PARALLEL_READS`] adjacent read-only calls (run concurrently).
fn group_into_batches<'a>(
    calls: &'a [ToolCall],
    actions: &dyn ActionRegistry,
) -> Vec<Vec<&'a ToolCall>> {
    let mut batches: Vec<Vec<&ToolCall>> = Vec::new();
    let mut reads: Vec<&ToolCall> = Vec::new();

    for call in calls {
        if actions.is_read_only(&call.name) {
            reads.push(call);
            if reads.len() == MAX_PARALLEL_READS {
                batches.push(std::mem::take(&mut reads));
            }
        } else {
            // A mutation breaks the run: flush pending reads first so they are
            // observed before it runs.
            if !reads.is_empty() {
                batches.push(std::mem::take(&mut reads));
            }
            batches.push(vec![call]);
        }
    }
    if !reads.is_empty() {
        batches.push(reads);
    }
    batches
}

/// Execute one call and normalise the outcome into a [`ToolResult`].
async fn run_one(
    call: &ToolCall,
    actions: &dyn ActionRegistry,
    progress: Option<&Arc<dyn crate::progress::ProgressSink>>,
) -> ToolResult {
    let started = Instant::now();
    info!(tool = %call.name, args = %call.arguments, "Executing tool call");
    if let Some(sink) = progress {
        sink.emit(crate::progress::Progress::ToolStarted {
            name: call.name.clone(),
            arguments: call.arguments.to_string(),
        });
    }

    let result = match actions.execute(call).await {
        Ok(output) => ToolResult {
            tool_call_id: call.id.clone(),
            tool_name: call.name.clone(),
            output,
            success: true,
            error: None,
            duration_ms: started.elapsed().as_millis() as u64,
        },
        Err(e) => {
            warn!(error = %e, tool = %call.name, "Tool execution failed");
            ToolResult {
                tool_call_id: call.id.clone(),
                tool_name: call.name.clone(),
                output: String::new(),
                success: false,
                error: Some(e.to_string()),
                duration_ms: started.elapsed().as_millis() as u64,
            }
        }
    };

    if let Some(sink) = progress {
        let preview = result
            .preview_text()
            .lines()
            .next()
            .unwrap_or("")
            .to_string();
        sink.emit(crate::progress::Progress::ToolFinished {
            name: call.name.clone(),
            success: result.success,
            duration_ms: result.duration_ms,
            preview,
        });
    }

    result
}

/// A termination outranks a reminder, which outranks silence.
fn merge(current: StuckSignal, next: StuckSignal) -> StuckSignal {
    match (&current, &next) {
        (StuckSignal::Terminate { .. }, _) => current,
        (_, StuckSignal::Terminate { .. }) => next,
        (StuckSignal::Continue { reminder: Some(_) }, _) => current,
        (_, StuckSignal::Continue { reminder: Some(_) }) => next,
        _ => current,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axolotl_actions::registry::DefaultRegistry;
    use axolotl_actions::ToolContext;
    use axolotl_models::types::ToolCall;

    fn setup() -> (tempfile::TempDir, DefaultRegistry, ContextState, Tokenizer) {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(axolotl_actions::Workspace::open(dir.path()).unwrap());
        let registry = DefaultRegistry::with_predefined_tools(ctx);
        let state = ContextState::new();
        (dir, registry, state, Tokenizer::new().unwrap())
    }

    fn turn_with(calls: Vec<ToolCall>) -> Turn {
        let response = axolotl_models::types::LLMResponse {
            message: ChatMessage::assistant_with_tool_calls(None, calls),
            usage: Default::default(),
            stop_reason: axolotl_models::types::StopReason::ToolUse,
            model: "mock".into(),
        };
        Turn::new(0, vec![], 0, response)
    }

    #[tokio::test]
    async fn writes_observation_for_each_call() {
        let (dir, registry, mut state, tok) = setup();
        let call = ToolCall {
            id: "c1".into(),
            name: "write_file".into(),
            arguments: serde_json::json!({ "path": "out.txt", "content": "hi" }),
        };
        let mut turn = turn_with(vec![call]);
        let mut stuck = StuckDetector::default();

        let signal =
            execute_tool_calls(&mut turn, &registry, &mut state, &tok, &mut stuck, None).await;

        assert!(matches!(signal, StuckSignal::Continue { reminder: None }));
        assert_eq!(state.messages.len(), 1);
        assert_eq!(
            state.messages[0].message.role,
            axolotl_models::types::Role::Tool
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("out.txt")).unwrap(),
            "hi"
        );
        assert!(turn.tool_results[0].success);
    }

    #[tokio::test]
    async fn tool_error_becomes_an_observation() {
        let (_dir, registry, mut state, tok) = setup();
        let call = ToolCall {
            id: "c1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({ "path": "missing.txt" }),
        };
        let mut turn = turn_with(vec![call]);
        let mut stuck = StuckDetector::default();

        execute_tool_calls(&mut turn, &registry, &mut state, &tok, &mut stuck, None).await;

        assert!(!turn.tool_results[0].success);
        let observation = state.messages[0].message.text();
        assert!(observation.starts_with("Error:"), "{observation}");
        assert_eq!(
            state.messages[0].message.tool_call_id.as_deref(),
            Some("c1")
        );
    }

    #[tokio::test]
    async fn repeated_failing_call_terminates() {
        let (_dir, registry, mut state, tok) = setup();
        let mut stuck = StuckDetector::new(2, 3);
        let mut last = StuckSignal::Continue { reminder: None };

        for _ in 0..3 {
            let call = ToolCall {
                id: "c1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({ "path": "missing.txt" }),
            };
            let mut turn = turn_with(vec![call]);
            last =
                execute_tool_calls(&mut turn, &registry, &mut state, &tok, &mut stuck, None).await;
        }

        assert!(
            matches!(last, StuckSignal::Terminate { repeats: 3, .. }),
            "{last:?}"
        );
    }

    #[tokio::test]
    async fn reminder_is_appended_to_context() {
        let (_dir, registry, mut state, tok) = setup();
        let mut stuck = StuckDetector::new(2, 99);
        let call = ToolCall {
            id: "c1".into(),
            name: "list_files".into(),
            arguments: serde_json::json!({}),
        };
        for _ in 0..2 {
            let mut turn = turn_with(vec![call.clone()]);
            execute_tool_calls(&mut turn, &registry, &mut state, &tok, &mut stuck, None).await;
        }
        let has_reminder = state
            .messages
            .iter()
            .any(|e| e.message.text().contains("SAME arguments"));
        assert!(has_reminder);
    }

    #[test]
    fn merge_prefers_termination() {
        let reminder = StuckSignal::Continue {
            reminder: Some("r".into()),
        };
        let terminate = StuckSignal::Terminate {
            tool: "t".into(),
            repeats: 8,
        };
        assert_eq!(merge(reminder.clone(), terminate.clone()), terminate);
        assert_eq!(merge(terminate.clone(), reminder.clone()), terminate);
        assert_eq!(
            merge(StuckSignal::Continue { reminder: None }, reminder.clone()),
            reminder
        );
    }

    fn tc(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: serde_json::json!({}),
        }
    }

    #[test]
    fn adjacent_reads_share_one_batch() {
        let (_dir, registry, _, _) = setup();
        let calls = vec![
            tc("1", "read_file"),
            tc("2", "grep_text"),
            tc("3", "list_files"),
        ];
        let batches = group_into_batches(&calls, &registry);
        assert_eq!(batches.len(), 1, "adjacent reads must fan out together");
        assert_eq!(batches[0].len(), 3);
    }

    #[test]
    fn a_mutation_splits_the_batches() {
        let (_dir, registry, _, _) = setup();
        let calls = vec![
            tc("1", "read_file"),
            tc("2", "write_file"),
            tc("3", "read_file"),
        ];
        let batches = group_into_batches(&calls, &registry);
        // read | write | read — order must be preserved across the write.
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0][0].id, "1");
        assert_eq!(batches[1][0].id, "2");
        assert_eq!(batches[2][0].id, "3");
    }

    #[test]
    fn mutations_never_share_a_batch() {
        let (_dir, registry, _, _) = setup();
        let calls = vec![tc("1", "write_file"), tc("2", "edit_file")];
        let batches = group_into_batches(&calls, &registry);
        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(|b| b.len() == 1));
    }

    #[test]
    fn reads_are_capped_at_eight_per_batch() {
        let (_dir, registry, _, _) = setup();
        let calls: Vec<ToolCall> = (0..10).map(|i| tc(&i.to_string(), "read_file")).collect();
        let batches = group_into_batches(&calls, &registry);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), MAX_PARALLEL_READS);
        assert_eq!(batches[1].len(), 2);
    }

    #[test]
    fn empty_turn_produces_no_batches() {
        let (_dir, registry, _, _) = setup();
        assert!(group_into_batches(&[], &registry).is_empty());
    }

    #[tokio::test]
    async fn parallel_reads_all_report_results() {
        let (dir, registry, mut state, tok) = setup();
        std::fs::write(dir.path().join("a.txt"), "alpha").unwrap();
        std::fs::write(dir.path().join("b.txt"), "beta").unwrap();

        let mut turn = turn_with(vec![
            ToolCall {
                id: "1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({ "path": "a.txt" }),
            },
            ToolCall {
                id: "2".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({ "path": "b.txt" }),
            },
        ]);
        let mut stuck = StuckDetector::default();

        execute_tool_calls(&mut turn, &registry, &mut state, &tok, &mut stuck, None).await;

        assert_eq!(turn.tool_results.len(), 2);
        assert!(turn.tool_results.iter().all(|r| r.success));
        // Observations keep emission order even though execution fanned out.
        assert_eq!(state.messages[0].message.tool_call_id.as_deref(), Some("1"));
        assert_eq!(state.messages[1].message.tool_call_id.as_deref(), Some("2"));
        assert!(state.messages[0].message.text().contains("alpha"));
        assert!(state.messages[1].message.text().contains("beta"));
    }

    #[tokio::test]
    async fn a_write_between_reads_sees_the_first_read() {
        // The read-then-write ordering matters: `write_file` checks
        // read-before-write, so if the write ran before the read it would fail.
        let (dir, registry, mut state, tok) = setup();
        std::fs::write(dir.path().join("c.txt"), "old").unwrap();

        let mut turn = turn_with(vec![
            ToolCall {
                id: "1".into(),
                name: "read_file".into(),
                arguments: serde_json::json!({ "path": "c.txt" }),
            },
            ToolCall {
                id: "2".into(),
                name: "write_file".into(),
                arguments: serde_json::json!({
                    "path": "c.txt", "content": "new", "overwrite": true
                }),
            },
        ]);
        let mut stuck = StuckDetector::default();

        execute_tool_calls(&mut turn, &registry, &mut state, &tok, &mut stuck, None).await;

        assert!(turn.tool_results[0].success);
        assert!(
            turn.tool_results[1].success,
            "write must run after the read: {:?}",
            turn.tool_results[1].error
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("c.txt")).unwrap(),
            "new"
        );
    }
}
