//! Core types for the agent loop.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

/// A task to be solved by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub description: String,
    #[serde(default)]
    pub setup_commands: Vec<String>,
    #[serde(default)]
    pub test_commands: Vec<String>,
    pub working_dir: Option<String>,
    #[serde(default)]
    pub expected_files: Vec<String>,
}

impl Task {
    /// A task from a plain description, with no verification commands.
    pub fn from_description(description: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            description: description.into(),
            setup_commands: Vec::new(),
            test_commands: Vec::new(),
            working_dir: None,
            expected_files: Vec::new(),
        }
    }

    /// Whether the task defines any automated check.
    pub fn is_verifiable(&self) -> bool {
        !self.test_commands.is_empty()
    }

    /// Try to run the task in `root`.
    ///
    /// A task may carry a `working_dir`; it is honoured only when it resolves
    /// inside `root`, so a benchmark file cannot point the agent elsewhere.
    pub fn resolve_working_dir(
        &self,
        root: &std::path::Path,
    ) -> Result<std::path::PathBuf, String> {
        let Some(dir) = &self.working_dir else {
            return Ok(root.to_path_buf());
        };
        let candidate = root.join(dir);
        let canonical_root = root
            .canonicalize()
            .map_err(|e| format!("workspace root '{}': {e}", root.display()))?;
        let canonical = candidate
            .canonicalize()
            .map_err(|e| format!("working_dir '{}': {e}", candidate.display()))?;
        if !canonical.starts_with(&canonical_root) {
            return Err(format!("working_dir '{dir}' escapes the workspace"));
        }
        Ok(canonical)
    }

    /// Setup commands, de-duplicated so a retry does not re-clone a repo.
    pub fn unique_setup_commands(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        self.setup_commands
            .iter()
            .filter(|c| seen.insert((*c).clone()))
            .cloned()
            .collect()
    }
}

/// Result of evaluating a completed task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub passed: bool,
    /// Concatenated output of every verification command.
    pub test_output: String,
    pub files_modified: Vec<String>,
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_from_description_has_no_checks() {
        let task = Task::from_description("do a thing");
        assert!(!task.is_verifiable());
        assert!(!task.id.is_empty());
    }

    #[test]
    fn verifiable_when_test_commands_present() {
        let mut task = Task::from_description("x");
        task.test_commands.push("true".into());
        assert!(task.is_verifiable());
    }

    #[test]
    fn working_dir_defaults_to_root() {
        let dir = tempfile::tempdir().unwrap();
        let task = Task::from_description("x");
        let resolved = task.resolve_working_dir(dir.path()).unwrap();
        // The root is returned unchanged (not canonicalised) so callers can
        // compare it directly with the workspace they already hold.
        assert_eq!(resolved, dir.path());
    }

    #[test]
    fn working_dir_inside_root_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        let mut task = Task::from_description("x");
        task.working_dir = Some("sub".into());
        assert!(task.resolve_working_dir(dir.path()).is_ok());
    }

    #[test]
    fn working_dir_escape_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let mut task = Task::from_description("x");
        task.working_dir = Some("..".into());
        assert!(task.resolve_working_dir(dir.path()).is_err());
    }

    #[test]
    fn setup_commands_are_deduplicated() {
        let mut task = Task::from_description("x");
        task.setup_commands = vec!["a".into(), "b".into(), "a".into()];
        assert_eq!(task.unique_setup_commands(), vec!["a", "b"]);
    }
}
