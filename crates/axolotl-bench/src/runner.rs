//! Benchmark batch runner: isolation, retries, resumption, and result export.
//!
//! Paper §3.1 describes running each task in its own container. That is not
//! portable, so this runner gives each task its own directory and runs the
//! task's setup commands there; the isolation boundary is the working
//! directory, not a VM. That is weaker than the paper's setup and is recorded
//! as such in the results.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use axolotl_core::{Session, SessionOutcome};

use crate::task::BenchmarkTask;

/// One task's outcome, persisted so a sweep can be resumed or re-analysed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskOutcome {
    pub task_id: String,
    pub benchmark: String,
    /// `success`, `failure`, `unverified`, `context_overflow`, ...
    pub outcome: String,
    pub passed: bool,
    pub turns: usize,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub peak_context_tokens: usize,
    pub summarize_calls: usize,
    pub cost_usd: Option<f64>,
    pub duration_secs: u64,
    pub session_id: String,
    /// Failure reason when the session did not run to completion.
    pub error: Option<String>,
}

impl TaskOutcome {
    /// Build from a finished session.
    pub fn from_session(task: &BenchmarkTask, session: &Session) -> Self {
        let passed = session
            .outcome
            .as_ref()
            .is_some_and(SessionOutcome::is_success);
        let error = match session.outcome.as_ref() {
            Some(SessionOutcome::ApiError(e)) => Some(format!("api error: {e}")),
            Some(SessionOutcome::Error(e)) => Some(e.clone()),
            Some(SessionOutcome::Failure(r)) => r.error.clone(),
            _ => None,
        };
        Self {
            task_id: task.id.clone(),
            benchmark: task.benchmark.clone(),
            outcome: session
                .outcome
                .as_ref()
                .map(|o| o.label().to_string())
                .unwrap_or_else(|| "none".into()),
            passed,
            turns: session.turn_count(),
            prompt_tokens: session.total_usage.prompt_tokens,
            completion_tokens: session.total_usage.completion_tokens,
            total_tokens: session.total_usage.total_tokens,
            peak_context_tokens: session.peak_context_tokens,
            summarize_calls: session.summarize_calls,
            cost_usd: session.cost_usd(),
            duration_secs: session.elapsed_secs(),
            session_id: session.id.clone(),
            error,
        }
    }

    /// Placeholder for a task that never ran (setup failure, crash).
    pub fn skipped(task: &BenchmarkTask, reason: impl Into<String>) -> Self {
        Self {
            task_id: task.id.clone(),
            benchmark: task.benchmark.clone(),
            outcome: "skipped".into(),
            passed: false,
            turns: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            peak_context_tokens: 0,
            summarize_calls: 0,
            cost_usd: None,
            duration_secs: 0,
            session_id: String::new(),
            error: Some(reason.into()),
        }
    }
}

/// Aggregate metrics for a sweep.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SweepSummary {
    pub benchmark: String,
    pub context_strategy: String,
    pub context_window: usize,
    pub planning_enabled: bool,
    pub action_space: String,
    pub model: String,
    pub tasks: usize,
    pub passed: usize,
    pub unverified: usize,
    pub errored: usize,
    pub overflowed: usize,
    pub total_tokens: usize,
    pub total_turns: usize,
    pub peak_context_tokens: usize,
    pub summarize_calls: usize,
    pub total_cost_usd: f64,
    pub cost_known: bool,
    pub duration_secs: u64,
}

impl SweepSummary {
    pub fn success_rate(&self) -> f64 {
        if self.tasks == 0 {
            return 0.0;
        }
        self.passed as f64 / self.tasks as f64
    }

    pub fn mean_tokens(&self) -> usize {
        self.total_tokens / self.tasks.max(1)
    }

    pub fn mean_turns(&self) -> usize {
        self.total_turns / self.tasks.max(1)
    }

    pub fn mean_cost(&self) -> Option<f64> {
        self.cost_known
            .then(|| self.total_cost_usd / self.tasks.max(1) as f64)
    }
}

/// Everything a sweep produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepReport {
    pub summary: SweepSummary,
    pub results: Vec<TaskOutcome>,
}

impl SweepReport {
    pub fn build(
        config: &axolotl_core::SessionConfig,
        benchmark: &str,
        results: Vec<TaskOutcome>,
        duration_secs: u64,
    ) -> Self {
        let mut summary = SweepSummary {
            benchmark: benchmark.to_string(),
            context_strategy: config.context_strategy.to_string(),
            context_window: config.context_window,
            planning_enabled: config.planning_enabled,
            action_space: config.action_space.to_string(),
            model: config.model.model_id.clone(),
            tasks: results.len(),
            duration_secs,
            ..SweepSummary::default()
        };

        for result in &results {
            if result.passed {
                summary.passed += 1;
            }
            match result.outcome.as_str() {
                "unverified" => summary.unverified += 1,
                "context_overflow" => summary.overflowed += 1,
                "api_error" | "error" | "skipped" => summary.errored += 1,
                _ => {}
            }
            summary.total_tokens += result.total_tokens;
            summary.total_turns += result.turns;
            summary.peak_context_tokens =
                summary.peak_context_tokens.max(result.peak_context_tokens);
            summary.summarize_calls += result.summarize_calls;
            if let Some(cost) = result.cost_usd {
                summary.total_cost_usd += cost;
                summary.cost_known = true;
            }
        }

        Self { summary, results }
    }

    /// Persist as JSON, next to the trajectories it describes.
    pub async fn write(&self, dir: &Path) -> Result<PathBuf> {
        tokio::fs::create_dir_all(dir)
            .await
            .with_context(|| format!("creating '{}'", dir.display()))?;
        let name = format!(
            "{}-{}-{}k-{}-{}.json",
            self.summary.benchmark,
            self.summary.context_strategy,
            self.summary.context_window / 1000,
            if self.summary.planning_enabled {
                "plan"
            } else {
                "noplan"
            },
            self.summary.action_space
        );
        let path = dir.join(name);
        let json = serde_json::to_string_pretty(self).context("serializing sweep report")?;
        tokio::fs::write(&path, json)
            .await
            .with_context(|| format!("writing '{}'", path.display()))?;
        info!(path = %path.display(), "Sweep report written");
        Ok(path)
    }

    /// Append rows to a CSV for plotting (paper §6: export to CSV).
    pub fn to_csv(&self) -> String {
        let mut csv = String::from(
            "task_id,outcome,passed,turns,prompt_tokens,completion_tokens,total_tokens,peak_context_tokens,summarize_calls,cost_usd,duration_secs\n",
        );
        for r in &self.results {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{}\n",
                r.task_id,
                r.outcome,
                r.passed,
                r.turns,
                r.prompt_tokens,
                r.completion_tokens,
                r.total_tokens,
                r.peak_context_tokens,
                r.summarize_calls,
                r.cost_usd.map(|c| format!("{c:.6}")).unwrap_or_default(),
                r.duration_secs
            ));
        }
        csv
    }
}

/// Prepare a per-task workspace directory.
///
/// Removes a previous run's directory so a task never inherits stale files,
/// which would silently make a verifier pass or fail for the wrong reason.
pub async fn prepare_workspace(root: &Path, task_id: &str) -> Result<PathBuf> {
    let dir = root.join(sanitize(task_id));
    if dir.exists() {
        tokio::fs::remove_dir_all(&dir)
            .await
            .with_context(|| format!("clearing '{}'", dir.display()))?;
    }
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("creating '{}'", dir.display()))?;
    Ok(dir)
}

/// Make a task id safe to use as a directory name.
fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Timing helper for a sweep.
pub struct SweepTimer {
    start: Instant,
}

impl Default for SweepTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl SweepTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed_secs(&self) -> u64 {
        self.start.elapsed().as_secs()
    }
}

/// Warn once when a sweep is configured in a way that makes its numbers
/// incomparable to the paper's.
pub fn warn_about_config(config: &axolotl_core::SessionConfig) {
    if config.model.input_cost_per_mtok.is_none() || config.model.output_cost_per_mtok.is_none() {
        warn!("no token prices configured; cost columns will be empty");
    }
    if config.context_window < 8_000 {
        warn!(
            window = config.context_window,
            "very small context window; most tasks will not fit"
        );
    }
}

#[cfg(test)]
mod tests;
