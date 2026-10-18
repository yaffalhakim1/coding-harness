//! Benchmark task traits and types.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use axolotl_core::types::{Task, TaskResult};

/// A task from a benchmark suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkTask {
    pub id: String,
    pub benchmark: String,
    pub description: String,
    pub repo_url: Option<String>,
    pub base_commit: Option<String>,
    pub setup_commands: Vec<String>,
    pub test_commands: Vec<String>,
    pub expected_files: Vec<String>,
    pub environment: Option<String>,
}

impl From<BenchmarkTask> for Task {
    fn from(bt: BenchmarkTask) -> Self {
        Task {
            id: bt.id,
            description: bt.description,
            setup_commands: bt.setup_commands,
            test_commands: bt.test_commands,
            working_dir: None,
            expected_files: bt.expected_files,
        }
    }
}

/// Result of running a benchmark task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub task_id: String,
    pub benchmark: String,
    pub session_id: String,
    pub passed: bool,
    pub total_tokens: usize,
    pub cost_usd: Option<f64>,
    pub turns: usize,
    pub duration_secs: u64,
    pub outcome: String,
}

/// Benchmark loader trait.
#[async_trait]
pub trait Benchmark: Send + Sync {
    /// Benchmark name.
    fn name(&self) -> &str;

    /// Load all tasks from the benchmark.
    async fn load_tasks(&self, path: &str) -> Result<Vec<BenchmarkTask>>;

    /// Load a single task by ID.
    async fn load_task(&self, path: &str, task_id: &str) -> Result<Option<BenchmarkTask>>;

    /// Evaluate a task result (run tests, check pass/fail).
    async fn evaluate(&self, task: &BenchmarkTask, working_dir: &str) -> Result<TaskResult>;
}

/// Run a benchmark task's verification commands in `working_dir`.
///
/// Shared by every benchmark: the difference between suites is how tasks are
/// loaded and which commands they declare, not how a shell exit code is read.
pub async fn run_verification(task: &BenchmarkTask, working_dir: &str) -> Result<TaskResult> {
    let context = axolotl_actions::ToolContext::new(
        axolotl_actions::Workspace::open(working_dir).map_err(|e| anyhow::anyhow!("{e}"))?,
    );
    let registry = axolotl_actions::DefaultRegistry::with_predefined_tools(context);
    Ok(axolotl_core::evaluate_task(&task.clone().into(), &registry).await)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task() -> BenchmarkTask {
        BenchmarkTask {
            id: "t1".into(),
            benchmark: "terminal-bench".into(),
            description: "do it".into(),
            repo_url: None,
            base_commit: None,
            setup_commands: vec![],
            test_commands: vec![],
            expected_files: vec![],
            environment: None,
        }
    }

    #[test]
    fn converts_to_core_task() {
        let mut bt = task();
        bt.test_commands.push("true".into());
        bt.expected_files.push("a.txt".into());
        let core: Task = bt.into();
        assert_eq!(core.test_commands, vec!["true"]);
        assert_eq!(core.expected_files, vec!["a.txt"]);
    }

    #[tokio::test]
    async fn verification_runs_commands() {
        let dir = tempfile::tempdir().unwrap();
        let mut bt = task();
        bt.test_commands.push(if cfg!(windows) {
            "exit 0".into()
        } else {
            "true".into()
        });
        let result = run_verification(&bt, dir.path().to_str().unwrap())
            .await
            .unwrap();
        assert!(result.passed);
    }

    #[tokio::test]
    async fn verification_fails_on_nonzero_exit() {
        let dir = tempfile::tempdir().unwrap();
        let mut bt = task();
        bt.test_commands.push(if cfg!(windows) {
            "exit 1".into()
        } else {
            "false".into()
        });
        let result = run_verification(&bt, dir.path().to_str().unwrap())
            .await
            .unwrap();
        assert!(!result.passed);
    }
}
