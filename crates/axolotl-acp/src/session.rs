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
use axolotl_actions::{DefaultRegistry, ToolContext, Workspace};
use axolotl_context::{build_strategy, ContextTier, Thresholds, Tokenizer};
use axolotl_core::progress::ProgressSink;
use axolotl_core::{AgentDeps, SessionConfig, Task};
use axolotl_models::types::ModelConfig;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::protocol::{ContentBlock, PromptParams, PromptResult, StopReason};
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
    pub history: Vec<axolotl_models::types::ChatMessage>,
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
        std::env::var("AXOLOTL_MODEL").unwrap_or_else(|_| "deepseek-v4-1-flash".to_string());
    let api_base = std::env::var("AXOLOTL_API_BASE").ok();
    let api_key_env =
        std::env::var("AXOLOTL_API_KEY_ENV").unwrap_or_else(|_| "KENARI_API_KEY".to_string());

    let strategy = std::env::var("AXOLOTL_STRATEGY")
        .ok()
        .and_then(|s| ContextTier::parse(&s))
        .unwrap_or(ContextTier::T4);
    let window = std::env::var("axolotl_context_WINDOW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CONTEXT_WINDOW);
    let planning = std::env::var("axolotl_planning")
        .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let max_turns = std::env::var("AXOLOTL_MAX_TURNS")
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
    client: crate::rpc::Client,
) -> Result<PromptResult> {
    let text = prompt_text(&params.prompt);
    if text.trim().is_empty() {
        anyhow::bail!("prompt contained no text");
    }

    // Echo the user message so clients that render only deltas still show it.
    emit_user_chunk(&writer, &session.id, &text).await;

    // Skills live in the workspace, so discovery needs the cwd before anything
    // else happens.
    let skills = axolotl_actions::skills::discover(&session.cwd);

    // A slash command may answer without the model at all, which is what makes
    // `/help` free.
    let text = match axolotl_actions::slash::resolve(&text, &skills) {
        axolotl_actions::slash::Resolution::NotACommand => text,
        axolotl_actions::slash::Resolution::Prompt(rewritten) => rewritten,
        axolotl_actions::slash::Resolution::Reply(reply) => {
            let params = serde_json::json!({
                "sessionId": session.id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": { "type": "text", "text": reply }
                }
            });
            let _ = writer.notify("session/update", params).await;
            return Ok(PromptResult {
                stop_reason: StopReason::EndTurn,
            });
        }
    };

    let task = Task::from_description(text.clone());
    let mut config = session.config.clone();
    config.planning_enabled = false;

    let ws = Workspace::open(&session.cwd)
        .with_context(|| format!("opening workspace '{}'", session.cwd.display()))?;

    // Supervised approval: every mutating tool asks the client. The destructive
    // pattern list sits inside it, so an obvious `rm -rf` is refused without
    // waking the user.
    let gate = Arc::new(
        axolotl_actions::permission::DestructiveCommandGate::with_defaults(Arc::new(
            crate::permission::AcpPermissionGate::new(client.clone(), session.id.clone()),
        )),
    );
    let ctx = ToolContext::new(ws).with_permissions(gate);

    let plan_state = None;
    let event_store = axolotl_models::EventStore::new();
    let keeps_originals = matches!(config.context_strategy, ContextTier::T2 | ContextTier::T4);
    let todos = axolotl_actions::todo::TodoList::new();
    let asker: Arc<dyn axolotl_actions::ask_user::Asker> = Arc::new(
        crate::permission::AcpAsker::new(client.clone(), session.id.clone()),
    );

    let mut actions = DefaultRegistry::with_predefined_tools_and_plan(ctx.clone(), plan_state)
        .with_conversation_tools(ctx.clone(), todos.clone(), Some(asker));
    if keeps_originals {
        actions = actions.with_recall(ctx, event_store.clone());
    }

    let tokenizer = Tokenizer::new()?;
    let provider = match &config.model.api_base {
        Some(base) => axolotl_models::OpenAIProvider::with_base_url(base.clone())?,
        None => axolotl_models::OpenAIProvider::new()?,
    };
    let llm: Arc<dyn axolotl_models::LLMProvider> = Arc::new(provider);

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

    let outcome = axolotl_core::agent::run_agent_loop(&task, &config, deps).await;

    let stop = match outcome {
        Ok(result) => {
            if session.cancelled.load(Ordering::Relaxed) {
                StopReason::Cancelled
            } else if result
                .outcome
                .as_ref()
                .is_some_and(|o| matches!(o, axolotl_core::SessionOutcome::TurnLimitReached))
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
