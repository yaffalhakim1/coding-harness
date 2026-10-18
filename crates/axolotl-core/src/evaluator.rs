//! Task evaluation: setup + verification commands.
//!
//! The scaffold treated any `Ok(..)` from the bash tool as a pass, and the bash
//! tool returns `Ok` for exit code 1 — so every task "passed". Success here is
//! decided by the *exit status* of each test command, which is what SWE-Bench
//! and Terminal-Bench verifiers report.

use tracing::{debug, warn};

use axolotl_models::types::ToolCall;

use crate::types::{Task, TaskResult};

/// Run the task's setup commands once, before the agent starts.
///
/// Returns an error string when setup fails; the caller decides whether that is
/// fatal (a broken environment makes the run meaningless).
pub async fn run_setup(
    task: &Task,
    actions: &dyn axolotl_actions::registry::ActionRegistry,
) -> Result<(), String> {
    for command in task.unique_setup_commands() {
        let call = bash_call(&command);
        match actions.execute(&call).await {
            Ok(_) => debug!(command = %command, "Setup command succeeded"),
            Err(e) => {
                warn!(command = %command, "Setup command failed");
                return Err(format!(
                    "setup command failed: {command}\n{}",
                    failure_output(&e)
                ));
            }
        }
    }
    Ok(())
}

/// Run the task's verification commands and decide pass/fail.
pub async fn evaluate_task(
    task: &Task,
    actions: &dyn axolotl_actions::registry::ActionRegistry,
) -> TaskResult {
    if !task.is_verifiable() {
        return TaskResult {
            passed: false,
            test_output: String::new(),
            files_modified: Vec::new(),
            error: Some("task has no test commands; success cannot be verified".into()),
        };
    }

    let mut all_passed = true;
    let mut output = String::new();

    for command in &task.test_commands {
        let call = bash_call(command);
        match actions.execute(&call).await {
            Ok(result) => {
                output.push_str(&format!(
                    "$ {command}\n{}\n[exit 0 → pass]\n",
                    result.trim_end()
                ));
            }
            Err(e) => {
                // A non-zero exit arrives as a typed error carrying the
                // command's output (paper Fig. 27).
                all_passed = false;
                let body = failure_output(&e);
                let code = failure_code(&e)
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "none".to_string());
                output.push_str(&format!(
                    "$ {command}\n{}\n[exit {code} → fail]\n",
                    body.trim_end()
                ));
            }
        }
    }

    TaskResult {
        passed: all_passed,
        test_output: output,
        files_modified: Vec::new(),
        error: if all_passed {
            None
        } else {
            Some("verification commands failed".into())
        },
    }
}

fn bash_call(command: &str) -> ToolCall {
    ToolCall {
        id: uuid::Uuid::new_v4().to_string(),
        name: "bash".into(),
        arguments: serde_json::json!({ "command": command }),
    }
}

/// Exit code reported by a failed command, or `None` when it timed out or
/// never ran.
fn failure_code(error: &axolotl_actions::ActionError) -> Option<i32> {
    match error {
        axolotl_actions::ActionError::CommandFailed { code, .. } => Some(*code),
        _ => None,
    }
}

/// Command output carried by a failure, when there is any.
fn failure_output(error: &axolotl_actions::ActionError) -> String {
    match error {
        axolotl_actions::ActionError::CommandFailed { output, .. }
        | axolotl_actions::ActionError::Timeout { output, .. } => output.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axolotl_actions::registry::DefaultRegistry;
    use axolotl_actions::ToolContext;

    fn registry(dir: &tempfile::TempDir) -> DefaultRegistry {
        let ctx = ToolContext::new(axolotl_actions::Workspace::open(dir.path()).unwrap());
        DefaultRegistry::with_predefined_tools(ctx)
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    fn success_cmd() -> &'static str {
        if cfg!(windows) {
            "exit 0"
        } else {
            "true"
        }
    }

    fn failure_cmd() -> &'static str {
        if cfg!(windows) {
            "exit 1"
        } else {
            "false"
        }
    }

    #[test]
    fn failure_code_and_output_are_extracted() {
        use axolotl_actions::ActionError;

        let failed = ActionError::CommandFailed {
            code: 2,
            output: "boom".into(),
        };
        assert_eq!(failure_code(&failed), Some(2));
        assert_eq!(failure_output(&failed), "boom");

        let timed_out = ActionError::Timeout {
            seconds: 5,
            output: String::new(),
        };
        assert_eq!(failure_code(&timed_out), None);
    }

    #[tokio::test]
    async fn passing_command_passes() {
        let dir = tempfile::tempdir().unwrap();
        let mut task = Task::from_description("x");
        task.test_commands.push(success_cmd().into());
        let result = evaluate_task(&task, &registry(&dir)).await;
        assert!(result.passed, "{}", result.test_output);
        assert!(result.test_output.contains("pass"));
    }

    #[tokio::test]
    async fn failing_command_fails() {
        // Regression: the scaffold marked this as passed because bash returned
        // `Ok` for a non-zero exit code.
        let dir = tempfile::tempdir().unwrap();
        let mut task = Task::from_description("x");
        task.test_commands.push(failure_cmd().into());
        let result = evaluate_task(&task, &registry(&dir)).await;
        assert!(!result.passed, "{}", result.test_output);
        assert!(result.error.is_some());
    }

    #[tokio::test]
    async fn all_commands_must_pass() {
        let dir = tempfile::tempdir().unwrap();
        let mut task = Task::from_description("x");
        task.test_commands = vec![success_cmd().into(), failure_cmd().into()];
        assert!(!evaluate_task(&task, &registry(&dir)).await.passed);
    }

    #[tokio::test]
    async fn task_without_tests_is_unverified() {
        let dir = tempfile::tempdir().unwrap();
        let task = Task::from_description("x");
        let result = evaluate_task(&task, &registry(&dir)).await;
        assert!(!result.passed);
        assert!(result.error.unwrap().contains("no test commands"));
    }

    #[tokio::test]
    async fn setup_failure_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let mut task = Task::from_description("x");
        task.setup_commands.push(failure_cmd().into());
        assert!(run_setup(&task, &registry(&dir)).await.is_err());
    }

    #[tokio::test]
    async fn setup_success_is_silent() {
        let dir = tempfile::tempdir().unwrap();
        let mut task = Task::from_description("x");
        task.setup_commands.push(success_cmd().into());
        assert!(run_setup(&task, &registry(&dir)).await.is_ok());
    }

    #[test]
    fn runtime_constructs() {
        let _ = rt();
    }
}
