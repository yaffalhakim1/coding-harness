//! SWE-Bench Verified benchmark loader.
//!
//! SWE-Bench tasks are GitHub issues with associated test patches.
//! Task format: JSON files with repo, commit, problem_statement, test_patch.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use tracing::{info, warn};

use harness_core::types::TaskResult;

use crate::task::{run_verification, Benchmark, BenchmarkTask};

/// SWE-Bench task JSON format.
#[derive(Debug, Deserialize)]
struct SweBenchTaskJson {
    instance_id: String,
    repo: String,
    base_commit: String,
    problem_statement: String,
    #[serde(default)]
    test_patch: String,
    #[serde(default)]
    version: String,
}

pub struct SweBench;

impl SweBench {
    pub fn new() -> Self {
        Self
    }

    /// Test command derived from the instance's `test_patch`.
    ///
    /// SWE-Bench patches touch specific test files; the conventional runner for
    /// a Python repo is `python -m pytest` on exactly those files. Without this,
    /// the whole suite runs and the verifier measures unrelated failures.
    fn test_commands(test_patch: &str) -> Vec<String> {
        let mut files: Vec<String> = test_patch
            .lines()
            .filter_map(|line| line.strip_prefix("+++ b/"))
            .map(str::trim)
            .filter(|f| f.ends_with(".py"))
            .map(str::to_string)
            .collect();
        files.sort();
        files.dedup();

        if files.is_empty() {
            warn!("test patch has no Python test files; falling back to the full suite");
            vec!["python -m pytest -x -q".to_string()]
        } else {
            vec![format!("python -m pytest -q {}", files.join(" "))]
        }
    }
}

impl Default for SweBench {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Benchmark for SweBench {
    fn name(&self) -> &str {
        "swe-bench"
    }

    async fn load_tasks(&self, path: &str) -> Result<Vec<BenchmarkTask>> {
        info!(path = %path, "Loading SWE-Bench tasks");

        let content = tokio::fs::read_to_string(path)
            .await
            .context("Failed to read SWE-Bench task file")?;

        let tasks_json: Vec<SweBenchTaskJson> =
            serde_json::from_str(&content).context("Failed to parse SWE-Bench JSON")?;

        let tasks: Vec<BenchmarkTask> = tasks_json
            .into_iter()
            .map(|t| BenchmarkTask {
                id: t.instance_id,
                benchmark: "swe-bench".to_string(),
                description: t.problem_statement,
                repo_url: Some(format!("https://github.com/{}", t.repo)),
                base_commit: Some(t.base_commit.clone()),
                setup_commands: vec![
                    format!("git clone https://github.com/{} .", t.repo),
                    format!("git checkout {}", t.base_commit),
                ],
                test_commands: Self::test_commands(&t.test_patch),
                expected_files: Vec::new(),
                environment: if t.version.is_empty() {
                    None
                } else {
                    Some(t.version)
                },
            })
            .collect();

        info!(count = tasks.len(), "Loaded SWE-Bench tasks");
        Ok(tasks)
    }

    async fn load_task(&self, path: &str, task_id: &str) -> Result<Option<BenchmarkTask>> {
        let tasks = self.load_tasks(path).await?;
        Ok(tasks.into_iter().find(|t| t.id == task_id))
    }

    async fn evaluate(&self, task: &BenchmarkTask, working_dir: &str) -> Result<TaskResult> {
        info!(task_id = %task.id, "Evaluating SWE-Bench task");
        run_verification(task, working_dir).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commands_target_patched_files() {
        let patch = "--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1 +1 @@\n-old\n+new\n";
        assert_eq!(
            SweBench::test_commands(patch),
            vec!["python -m pytest -q tests/test_x.py"]
        );
    }

    #[test]
    fn test_commands_deduplicate_files() {
        let patch = "+++ b/tests/a.py\n+++ b/tests/a.py\n+++ b/tests/b.py\n";
        assert_eq!(
            SweBench::test_commands(patch),
            vec!["python -m pytest -q tests/a.py tests/b.py"]
        );
    }

    #[test]
    fn empty_patch_falls_back_to_full_suite() {
        assert_eq!(SweBench::test_commands(""), vec!["python -m pytest -x -q"]);
    }

    #[tokio::test]
    async fn loads_and_maps_fields() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.json");
        std::fs::write(
            &path,
            r#"[{"instance_id":"i1","repo":"o/r","base_commit":"abc","problem_statement":"fix","test_patch":"+++ b/t.py\n","version":"1.0"}]"#,
        )
        .unwrap();

        let tasks = SweBench::new()
            .load_tasks(path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].id, "i1");
        assert_eq!(tasks[0].base_commit.as_deref(), Some("abc"));
        assert_eq!(tasks[0].environment.as_deref(), Some("1.0"));
        assert!(tasks[0]
            .setup_commands
            .iter()
            .any(|c| c.contains("git checkout abc")));
    }

    #[tokio::test]
    async fn finds_task_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tasks.json");
        std::fs::write(
            &path,
            r#"[{"instance_id":"i1","repo":"o/r","base_commit":"a","problem_statement":"p"},
                {"instance_id":"i2","repo":"o/r","base_commit":"b","problem_statement":"q"}]"#,
        )
        .unwrap();

        let found = SweBench::new()
            .load_task(path.to_str().unwrap(), "i2")
            .await
            .unwrap();
        assert_eq!(found.unwrap().description, "q");
    }

    #[tokio::test]
    async fn missing_file_is_an_error() {
        assert!(SweBench::new().load_tasks("nope.json").await.is_err());
    }
}
