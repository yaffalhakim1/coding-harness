//! Session state management.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use harness_context::{ContextTier, Thresholds};
use harness_models::types::{ModelConfig, TokenUsage};

use crate::turn::Turn;
use crate::types::{Task, TaskResult};

/// Configuration for a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    pub context_window: usize,
    pub context_strategy: ContextTier,
    pub planning_enabled: bool,
    pub action_space: ActionSpace,
    pub model: ModelConfig,
    pub max_turns: u32,
    pub timeout_secs: u64,
    /// Context-management thresholds (paper §3.1).
    #[serde(default)]
    pub thresholds: ThresholdSettings,
    /// Consecutive content-only turns tolerated before the run ends.
    #[serde(default = "default_max_narration")]
    pub max_narration_turns: u32,
    /// Hard cap on total tokens per session (plan §6: cost runaway).
    #[serde(default)]
    pub token_budget: Option<usize>,
    /// Whether the `task` tool is exposed, letting the agent spawn subagents.
    ///
    /// Off by default: subagents change the action space, and a sweep must not
    /// silently measure a different harness than the one it claims to.
    #[serde(default)]
    pub subagents: bool,
}

fn default_max_narration() -> u32 {
    3
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            context_window: 128_000,
            context_strategy: ContextTier::T4,
            planning_enabled: true,
            action_space: ActionSpace::Predefined,
            model: ModelConfig::default(),
            max_turns: 50,
            timeout_secs: 600,
            thresholds: ThresholdSettings::default(),
            max_narration_turns: default_max_narration(),
            token_budget: None,
            subagents: false,
        }
    }
}

/// Serde-friendly mirror of [`Thresholds`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThresholdSettings {
    pub soft_ratio: f32,
    pub hard_ratio: f32,
    pub recent_ratio: f32,
    pub recent_turns: usize,
}

impl Default for ThresholdSettings {
    fn default() -> Self {
        let t = Thresholds::default();
        Self {
            soft_ratio: t.soft_ratio,
            hard_ratio: t.hard_ratio,
            recent_ratio: t.recent_ratio,
            recent_turns: t.recent_turns,
        }
    }
}

impl From<ThresholdSettings> for Thresholds {
    fn from(s: ThresholdSettings) -> Self {
        Self {
            soft_ratio: s.soft_ratio,
            hard_ratio: s.hard_ratio,
            recent_ratio: s.recent_ratio,
            recent_turns: s.recent_turns,
        }
    }
}

/// Which action space to expose to the model (paper §2.2).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ActionSpace {
    /// Predefined tools: read_file, write_file, edit_file, list_files, glob_files, grep_text, bash.
    Predefined,
    /// Bash-only: a single bash tool that accepts any command.
    BashOnly,
    /// Both predefined tools and bash available.
    Hybrid,
}

impl ActionSpace {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "predefined" => Some(Self::Predefined),
            "bash-only" | "bashonly" | "bash" => Some(Self::BashOnly),
            "hybrid" => Some(Self::Hybrid),
            _ => None,
        }
    }
}

impl std::fmt::Display for ActionSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActionSpace::Predefined => write!(f, "predefined"),
            ActionSpace::BashOnly => write!(f, "bash-only"),
            ActionSpace::Hybrid => write!(f, "hybrid"),
        }
    }
}

/// Outcome of a session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionOutcome {
    /// Task completed and verification passed.
    Success(TaskResult),
    /// Task completed but verification failed.
    Failure(TaskResult),
    /// The model stopped without an error, but the task has no test commands,
    /// so success cannot be claimed (paper's metric is verifier-based).
    Unverified(TaskResult),
    /// Session hit the turn limit.
    TurnLimitReached,
    /// Session timed out.
    Timeout,
    /// Context overflowed (only possible for unmanaged T0).
    ContextOverflow,
    /// Stuck detection ended the run (paper §2.4).
    Stuck { tool: String, repeats: u32 },
    /// LLM API error.
    ApiError(String),
    /// Unknown/unexpected error.
    Error(String),
}

impl SessionOutcome {
    /// Short machine-readable name for reports.
    pub fn label(&self) -> &'static str {
        match self {
            SessionOutcome::Success(_) => "success",
            SessionOutcome::Failure(_) => "failure",
            SessionOutcome::Unverified(_) => "unverified",
            SessionOutcome::TurnLimitReached => "turn_limit",
            SessionOutcome::Timeout => "timeout",
            SessionOutcome::ContextOverflow => "context_overflow",
            SessionOutcome::Stuck { .. } => "stuck",
            SessionOutcome::ApiError(_) => "api_error",
            SessionOutcome::Error(_) => "error",
        }
    }

    /// Whether the run resolved the task.
    pub fn is_success(&self) -> bool {
        matches!(self, SessionOutcome::Success(_))
    }

    /// The verifier result, when one was produced.
    pub fn task_result(&self) -> Option<&TaskResult> {
        match self {
            SessionOutcome::Success(r)
            | SessionOutcome::Failure(r)
            | SessionOutcome::Unverified(r) => Some(r),
            _ => None,
        }
    }
}

impl PartialEq for TaskResult {
    fn eq(&self, other: &Self) -> bool {
        self.passed == other.passed
            && self.test_output == other.test_output
            && self.files_modified == other.files_modified
            && self.error == other.error
    }
}

impl Eq for TaskResult {}

/// A session represents one complete task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub task: Task,
    pub config: SessionConfig,
    pub turns: Vec<Turn>,
    pub plan: Option<String>,
    pub outcome: Option<SessionOutcome>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub total_usage: TokenUsage,
    /// Peak context tokens observed, for cost/context analysis (paper Fig. 5a).
    pub peak_context_tokens: usize,
    /// Number of summarization calls made by context management.
    pub summarize_calls: usize,
}

impl Session {
    pub fn new(task: Task, config: SessionConfig) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            task,
            config,
            turns: Vec::new(),
            plan: None,
            outcome: None,
            started_at: Utc::now(),
            completed_at: None,
            total_usage: TokenUsage::default(),
            peak_context_tokens: 0,
            summarize_calls: 0,
        }
    }

    pub fn record_turn(&mut self, turn: Turn) {
        self.total_usage.merge(&turn.response.usage);
        self.turns.push(turn);
    }

    pub fn set_plan(&mut self, plan: String) {
        self.plan = Some(plan);
    }

    pub fn complete(&mut self, outcome: SessionOutcome) {
        self.completed_at = Some(Utc::now());
        self.outcome = Some(outcome);
    }

    pub fn observe_context(&mut self, tokens: usize) {
        self.peak_context_tokens = self.peak_context_tokens.max(tokens);
    }

    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    pub fn elapsed_secs(&self) -> u64 {
        let end = self.completed_at.unwrap_or_else(Utc::now);
        (end - self.started_at).num_seconds().max(0) as u64
    }

    /// Session cost in USD, or `None` when the model config has no prices.
    pub fn cost_usd(&self) -> Option<f64> {
        self.config.model.cost_usd(&self.total_usage)
    }

    /// Serialize the full session to JSON for trajectory logging.
    pub fn to_trajectory_json(&self) -> serde_json::Value {
        serde_json::json!({
            "session_id": self.id,
            "task_id": self.task.id,
            "config": {
                "context_window": self.config.context_window,
                "context_strategy": self.config.context_strategy.to_string(),
                "planning_enabled": self.config.planning_enabled,
                "action_space": self.config.action_space.to_string(),
                "model": self.config.model.model_id,
                "max_turns": self.config.max_turns,
            },
            "plan": self.plan,
            "turns": self.turns.iter().map(|t| t.to_json()).collect::<Vec<_>>(),
            "outcome": self.outcome.as_ref().map(|o| o.label()),
            "outcome_detail": self.outcome.as_ref().map(|o| format!("{o:?}")),
            "total_tokens": self.total_usage.total_tokens,
            "prompt_tokens": self.total_usage.prompt_tokens,
            "completion_tokens": self.total_usage.completion_tokens,
            "total_cost_usd": self.cost_usd(),
            "total_turns": self.turn_count(),
            "peak_context_tokens": self.peak_context_tokens,
            "summarize_calls": self.summarize_calls,
            "duration_secs": self.elapsed_secs(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_space_parsing() {
        assert_eq!(
            ActionSpace::parse("predefined"),
            Some(ActionSpace::Predefined)
        );
        assert_eq!(ActionSpace::parse("bash-only"), Some(ActionSpace::BashOnly));
        assert_eq!(ActionSpace::parse("Bash_Only"), Some(ActionSpace::BashOnly));
        assert_eq!(ActionSpace::parse("hybrid"), Some(ActionSpace::Hybrid));
        assert_eq!(ActionSpace::parse("nope"), None);
    }

    #[test]
    fn default_config_matches_the_paper_baseline() {
        let cfg = SessionConfig::default();
        // Paper §3.1: T4/128k with planning on and the predefined tool set is
        // the baseline setting for the component ablations.
        assert_eq!(cfg.context_strategy, ContextTier::T4);
        assert_eq!(cfg.context_window, 128_000);
        assert!(cfg.planning_enabled);
        assert_eq!(cfg.action_space, ActionSpace::Predefined);
        assert!((cfg.thresholds.soft_ratio - 0.6).abs() < f32::EPSILON);
        assert!((cfg.thresholds.hard_ratio - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn outcome_labels() {
        let r = TaskResult {
            passed: true,
            test_output: String::new(),
            files_modified: vec![],
            error: None,
        };
        assert_eq!(SessionOutcome::Success(r.clone()).label(), "success");
        assert!(SessionOutcome::Success(r.clone()).is_success());
        assert!(SessionOutcome::Success(r.clone()).task_result().is_some());
        assert!(!SessionOutcome::TurnLimitReached.is_success());
        assert!(SessionOutcome::TurnLimitReached.task_result().is_none());
    }

    #[test]
    fn cost_is_none_without_prices() {
        let session = Session::new(Task::from_description("x"), SessionConfig::default());
        assert_eq!(session.cost_usd(), None);
    }

    #[test]
    fn trajectory_includes_outcome_and_peak_context() {
        let mut session = Session::new(Task::from_description("x"), SessionConfig::default());
        session.observe_context(500);
        session.observe_context(400);
        session.complete(SessionOutcome::Timeout);
        let json = session.to_trajectory_json();
        assert_eq!(json["outcome"], "timeout");
        assert_eq!(json["peak_context_tokens"], 500);
        assert_eq!(json["total_turns"], 0);
    }
}
