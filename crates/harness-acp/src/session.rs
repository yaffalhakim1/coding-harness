//! Session state and prompt handling for the ACP agent.
//!
//! One process serves many sessions, matching how the protocol models a client
//! that may open several. Each session keeps its own conversation, config and
//! cancellation flag.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use harness_actions::{DefaultRegistry, ToolContext, Workspace};
use harness_context::{build_strategy, ContextTier, Thresholds, Tokenizer};
use harness_core::progress::ProgressSink;
use harness_core::{AgentDeps, SessionConfig, Task};
use harness_models::types::ModelConfig;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::protocol::{ContentBlock, PromptParams, PromptResult, StopReason, ToolKind, ToolStatus};
use crate::sink::AcpSink;

/// Default context window when the model config does not set one.
const DEFAULT_CONTEXT_WINDOW: usize = 128_000;

/// One live session.
#[derive(Debug)]
pub struct Session {
    pub id: String,
    pub cwd: PathBuf,
    pub config: SessionConfig,
    /// Set by `session/cancel`; the loop checks it between turns.
    pub cancelled: Arc<AtomicBool>,
    /// The conversation so far, replayed into context on each prompt.
    pub history: Vec<harness_models::types::ChatMessage>,
    pub title: Option<String>,
}

impl Session {
    fn new(id: String, cwd: PathBuf, config: SessionConfig) -> Self {
        Self {
            id,
            cwd,
            config,
            cancelled: Arc::new(AtomicBool::new(false)),
            history: Vec::new(),
            title: None,
        }
    }
}

/// All sessions in this process.
#[derive(Default)]
pub struct Sessions {
    inner: Mutex<HashMap<String, Session>>,
}

impl Sessions {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub async fn insert(&self, session: Session) {
        self.inner.lock().await.insert(session.id.clone(), session);
    }

    pub async fn get(&self, id: &str) -> Option<Session> {
        self.inner.lock().await.get(id).cloned()
    }

    pub async fn remove(&self, id: &str) -> Option<Session> {
        self.inner.lock().await.remove(id)
    }

    /// Snapshot every session, for `session/list`.
    pub async fn all(&self) -> Vec<Session> {
        self.inner.lock().await.values().cloned().collect()
    }

    pub async fn is_empty(&self) -> bool {
        self.inner.lock().await.is_empty()
    }

    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
    }
}

impl Clone for Session {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            cwd: self.cwd.clone(),
            config: self.config.clone(),
            cancelled: self.cancelled.clone(),
            history: self.history.clone(),
            title: self.title.clone(),
        }
    }
}

/// Build the harness configuration for a session.
///
/// Model and provider come from the environment so one binary can be launched
/// by a client that knows nothing about the harness's own config format.
pub fn session_config() -> SessionConfig {
    let model_id =
        std::env::var("HARNESS_MODEL").unwrap_or_else(|_| "deepseek-v4-1-flash".to_string());
    let api_base = std::env::var("HARNESS_API_BASE").ok();
    let api_key_env =
        std::env::var("HARNESS_API_KEY_ENV").unwrap_or_else(|_| "KENARI_API_KEY".to_string());

    let strategy = std::env::var("HARNESS_STRATEGY")
        .ok()
        .and_then(|s| ContextTier::parse(&s))
        .unwrap_or(ContextTier::T4);
    let window = std::env::var("HARNESS_CONTEXT_WINDOW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CONTEXT_WINDOW);
    let planning = std::env::var("HARNESS_PLANNING")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let max_turns = std::env::var("HARNESS_MAX_TURNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);

    SessionConfig {
        context_window: window,
        context_strategy: strategy,
        planning_enabled: planning,
        max_turns,
        model: ModelConfig {
            provider: "openai".into(),
            model_id,
            api_base,
            api_key_env: Some(api_key_env),
            temperature: 0.0,
            max_tokens: Some(4096),
            timeout_secs: 180,
            max_retries: 3,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Create a session and answer `session/new`.
pub async fn create(cwd: String) -> Result<Session> {
    let cwd = PathBuf::from(cwd);
    if !cwd.is_absolute() {
        anyhow::bail!("cwd must be absolute, got '{}'", cwd.display());
    }
    if !cwd.is_dir() {
        anyhow::bail!("cwd '{}' is not a directory", cwd.display());
    }
    let id = format!("sess_{}", uuid::Uuid::new_v4().simple());
    let session = Session::new(id, cwd, session_config());
    info!(session = %session.id, "created session");
    Ok(session)
}

/// Run one prompt turn and return its stop reason.
///
/// The turn is driven by the harness loop; this function owns the mapping from
/// protocol input to a task and from loop outcome to a stop reason.
pub async fn run_prompt(
    session: Session,
    params: PromptParams,
    sink: Arc<AcpSink>,
    writer: crate::rpc::Writer,
) -> Result<PromptResult> {
    let text = prompt_text(&params.prompt);
    if text.trim().is_empty() {
        anyhow::bail!("prompt contained no text");
    }

    // Echo the user message so clients that render only deltas still show it.
    emit_user_chunk(&writer, &session.id, &text).await;

    let task = Task::from_description(text.clone());
    let mut config = session.config.clone();
    config.planning_enabled = false;

    let ws = Workspace::open(&session.cwd)
        .with_context(|| format!("opening workspace '{}'", session.cwd.display()))?;
    let ctx = ToolContext::new(ws);

    let plan_state = None;
    let event_store = harness_models::EventStore::new();
    let keeps_originals = matches!(config.context_strategy, ContextTier::T2 | ContextTier::T4);
    let mut actions = DefaultRegistry::with_predefined_tools_and_plan(ctx.clone(), plan_state);
    if keeps_originals {
        actions = actions.with_recall(ctx, event_store.clone());
    }

    let tokenizer = Tokenizer::new()?;
    let provider = match &config.model.api_base {
        Some(base) => harness_models::OpenAIProvider::with_base_url(base.clone())?,
        None => harness_models::OpenAIProvider::new()?,
    };
    let llm: Arc<dyn harness_models::LLMProvider> = Arc::new(provider);

    let context = build_strategy(
        config.context_strategy,
        Thresholds::from(config.thresholds),
        tokenizer.clone(),
        llm.clone(),
        &config.model,
        event_store,
    );

    let deps = AgentDeps {
        llm,
        actions: &actions,
        context: context.as_ref(),
        tokenizer: &tokenizer,
        planner: None,
        plan_state: None,
        progress: Some(sink.clone() as Arc<dyn ProgressSink>),
    };

    let outcome = harness_core::agent::run_agent_loop(&task, &config, deps).await;

    let stop = match outcome {
        Ok(result) => {
            if session.cancelled.load(Ordering::Relaxed) {
                StopReason::Cancelled
            } else if result
                .outcome
                .as_ref()
                .is_some_and(|o| matches!(o, harness_core::SessionOutcome::TurnLimitReached))
            {
                StopReason::MaxTurnRequests
            } else {
                StopReason::EndTurn
            }
        }
        Err(e) => {
            warn!(error = %e, "prompt turn failed");
            emit_error_notice(&writer, &session.id, &e.to_string()).await;
            StopReason::EndTurn
        }
    };

    debug!(session = %session.id, ?stop, "turn finished");
    Ok(PromptResult { stop_reason: stop })
}

/// Flatten a prompt's content blocks into one text instruction.
fn prompt_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .map(ContentBlock::as_text)
        .collect::<Vec<_>>()
        .join("\n\n")
}

async fn emit_user_chunk(writer: &crate::rpc::Writer, session_id: &str, text: &str) {
    let params = serde_json::json!({
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "user_message_chunk",
            "content": { "type": "text", "text": text }
        }
    });
    let _ = writer.notify("session/update", params).await;
}

/// Surface a turn failure the way the protocol expects: a notice when the
/// client renders notices, otherwise an agent message so it is not lost.
async fn emit_error_notice(writer: &crate::rpc::Writer, session_id: &str, reason: &str) {
    let params = serde_json::json!({
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": { "type": "text", "text": format!("[turn failed: {reason}]") }
        }
    });
    let _ = writer.notify("session/update", params).await;
}

/// Emit the plan as an ACP `plan` update.
pub async fn emit_plan(writer: &crate::rpc::Writer, session_id: &str, steps: &[String]) {
    let entries: Vec<serde_json::Value> = steps
        .iter()
        .map(|step| {
            serde_json::json!({
                "content": step,
                "priority": "medium",
                "status": "pending"
            })
        })
        .collect();
    let params = serde_json::json!({
        "sessionId": session_id,
        "update": { "sessionUpdate": "plan", "entries": entries }
    });
    let _ = writer.notify("session/update", params).await;
}

/// Replay history for `session/load`.
///
/// The protocol requires the full conversation as `session/update`
/// notifications before the response, so a client that reconnects rebuilds its
/// transcript.
pub async fn replay_history(
    writer: &crate::rpc::Writer,
    session_id: &str,
    history: &[harness_models::types::ChatMessage],
) {
    for message in history {
        let (kind, text) = match message.role {
            harness_models::types::Role::User => ("user_message_chunk", message.text()),
            harness_models::types::Role::Assistant => ("agent_message_chunk", message.text()),
            // System and tool messages are not part of the visible transcript.
            _ => continue,
        };
        if text.trim().is_empty() {
            continue;
        }
        let params = serde_json::json!({
            "sessionId": session_id,
            "update": {
                "sessionUpdate": kind,
                "content": { "type": "text", "text": text }
            }
        });
        let _ = writer.notify("session/update", params).await;
    }
}

/// Tool kinds and statuses re-exported for the dispatcher's permission prompts.
pub fn permission_kind(tool: &str) -> ToolKind {
    crate::sink::kind_for_tool(tool)
}

/// Status used when a tool is cancelled mid-flight.
pub const CANCELLED_STATUS: ToolStatus = ToolStatus::Failed;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_text_joins_blocks() {
        let blocks = vec![
            ContentBlock::Text {
                text: "first".into(),
            },
            ContentBlock::Text {
                text: "second".into(),
            },
        ];
        assert_eq!(prompt_text(&blocks), "first\n\nsecond");
    }

    #[test]
    fn prompt_text_includes_degraded_attachments() {
        let blocks = vec![ContentBlock::Image { mime_type: None }];
        let text = prompt_text(&blocks);
        assert!(text.contains("image omitted"), "{text}");
    }

    #[test]
    fn session_config_reads_the_environment() {
        // Defaults must be usable with nothing set.
        let config = session_config();
        assert!(!config.model.model_id.is_empty());
        assert!(config.context_window > 0);
    }

    #[tokio::test]
    async fn create_rejects_a_relative_cwd() {
        let err = create("relative/path".into()).await.unwrap_err();
        assert!(err.to_string().contains("absolute"), "{err}");
    }

    #[tokio::test]
    async fn create_rejects_a_missing_directory() {
        let missing = std::env::temp_dir().join("definitely-not-here-xyz");
        let err = create(missing.to_string_lossy().to_string())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a directory"), "{err}");
    }

    #[tokio::test]
    async fn create_accepts_an_absolute_directory() {
        let dir = tempfile::tempdir().unwrap();
        let session = create(dir.path().to_string_lossy().to_string())
            .await
            .unwrap();
        assert!(session.id.starts_with("sess_"));
        assert_eq!(session.cwd, dir.path());
    }

    #[tokio::test]
    async fn sessions_store_and_remove() {
        let store = Sessions::new();
        let dir = tempfile::tempdir().unwrap();
        let session = create(dir.path().to_string_lossy().to_string())
            .await
            .unwrap();
        let id = session.id.clone();

        store.insert(session).await;
        assert_eq!(store.len().await, 1);
        assert!(store.get(&id).await.is_some());
        assert!(store.remove(&id).await.is_some());
        assert_eq!(store.len().await, 0);
    }
}
