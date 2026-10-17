//! Terminal-Bench 2.1 benchmark loader.
//!
//! Terminal-Bench tasks are command-line focused — the agent must
//! accomplish tasks using shell commands.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use tracing::info;

use harness_core::types::TaskResult;

use crate::task::{run_verification, Benchmark, BenchmarkTask};

/// Terminal-Bench task JSON format.
#[derive(Debug, Deserialize)]
struct TerminalBenchTaskJson {
    id: String,
    description: String,
    #[serde(default)]
    setup: Vec<String>,
    #[serde(default)]
    test: Vec<String>,
    #[serde(default)]
    environment: String,
}

pub struct TerminalBench;

impl TerminalBench {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TerminalBench {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for TerminalBench {
    fn name(&self) -> &str {
        "terminal-bench"
    }

    async fn load_tasks(&self, path: &str) -> Result<Vec<BenchmarkTask>> {
        info!(path = %path, "Loading Terminal-Bench tasks");

        let content = tokio::fs::read_to_string(path)
            .await
            .context("Failed to read Terminal-Bench task file")?;

        let tasks_json: Vec<TerminalBenchTaskJson> =
            serde_json::from_str(&content).context("Failed to parse Terminal-Bench JSON")?;

        let tasks: Vec<BenchmarkTask> = tasks_json
            .into_iter()
            .map(|t| BenchmarkTask {
                id: t.id,
                benchmark: "terminal-bench".to_string(),
                description: t.description,
                repo_url: None,
                base_commit: None,
                setup_commands: t.setup,
                test_commands: t.test,
                expected_files: Vec::new(),
                environment: if t.environment.is_empty() {
                    None
                } else {
                    Some(t.environment)
                },
            })
            .collect();

        info!(count = tasks.len(), "Loaded Terminal-Bench tasks");
        Ok(tasks)
    }

    async fn load_task(&self, path: &str, task_id: &str) -> Result<Option<BenchmarkTask>> {
        let tasks = self.load_tasks(path).await?;
        Ok(tasks.into_iter().find(|t| t.id == task_id))
    }

    async fn evaluate(&self, task: &BenchmarkTask, working_dir: &str) -> Result<TaskResult> {
        info!(task_id = %task.id, "Evaluating Terminal-Bench task");
        run_verification(task, working_dir).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn loads_tasks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.json");
        std::fs::write(
            &path,
            r#"[{"id":"t1","description":"make a file","setup":["echo hi > a"],"test":["test -f a"],"environment":"ubuntu"}]"#,
        )
        .unwrap();

        let tasks = TerminalBench::new()
            .load_tasks(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "make a file");
        assert_eq!(tasks[0].test_commands, vec!["test -f a"]);
        assert_eq!(tasks[0].environment.as_deref(), Some("ubuntu"));
    }

    #[tokio::test]
    async fn defaults_are_optional() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.json");
        std::fs::write(&path, r#"[{"id":"t1","description":"d"}]"#).unwrap();

        let tasks = TerminalBench::new()
            .load_tasks(path.to_str().unwrap())
            .await
            .unwrap();
        assert!(tasks[0].setup_commands.is_empty());
        assert!(tasks[0].test_commands.is_empty());
        assert!(tasks[0].environment.is_none());
    }

    #[tokio::test]
    async fn finds_task_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.json");
        std::fs::write(
            &path,
            r#"[{"id":"t1","description":"a"},{"id":"t2","description":"b"}]"#,
        )
        .unwrap();
        let found = TerminalBench::new()
            .load_task(path.to_str().unwrap(), "t2")
            .await
            .unwrap();
        assert_eq!(found.unwrap().description, "b");
    }
}
