//! Shell execution tool (paper §2.4, Appendix B.3 Figures 27–28).
//!
//! Commands run through the platform shell — `/bin/sh -c` on unix, `cmd /C` on
//! Windows — so pipes and redirects work in both cases. The paper's harness ran
//! in Linux containers; this port keeps the same command strings usable on the
//! host it is actually run from.

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use tokio::io::AsyncReadExt;
use tracing::{debug, warn};

use axolotl_models::types::ToolDefinition;

use crate::error::{ActionError, ActionResult};
use crate::registry::{Action, ToolContext};

const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_TIMEOUT_SECS: u64 = 600;

/// Runs shell commands inside the workspace.
pub struct BashExecutor {
    ctx: ToolContext,
    /// Bash-only action space: the description tells the model every workspace
    /// operation must go through the shell (paper Fig. 28).
    bash_only: bool,
}

impl BashExecutor {
    pub fn new(ctx: ToolContext, bash_only: bool) -> Self {
        Self { ctx, bash_only }
    }

    fn description(&self) -> String {
        let shared = format!(
            "Execute a shell command in the workspace and return its exit code, stdout, and stderr. Useful for running tests, linters, build commands, and read-only git queries.\n\nProtocol: command is run via `{} -c` so pipes, redirects, and shell builtins work. cwd defaults to the workspace root and must be inside the workspace. timeout_seconds (default {DEFAULT_TIMEOUT_SECS}, max {MAX_TIMEOUT_SECS}) kills the process on expiry and returns whatever output it produced before the kill. Non-zero exit codes are reported in the output so you see the failure clearly.\n\nThe harness does NOT track which files the shell touched — run `git status` or `git diff` afterwards if you need to know. Files modified this way must be re-read with read_file before edit_file will accept them.",
            shell_program()
        );
        if self.bash_only {
            format!(
                "{shared}\n\nThis is your ONLY tool for touching the codebase — reading, searching and editing files all happen through commands you write. Nothing validates your edits, so after modifying a file read it back to confirm the change landed as intended."
            )
        } else {
            format!(
                "{shared}\n\nFor file reading prefer read_file (it records read state for read-before-write); for file search prefer grep_text; for file discovery prefer glob_files."
            )
        }
    }
}

/// The `-c` program for the host platform.
fn shell_program() -> &'static str {
    if cfg!(windows) {
        "cmd /C"
    } else {
        "/bin/sh"
    }
}

#[async_trait]
impl Action for BashExecutor {
    fn name(&self) -> &str {
        "bash"
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "bash".to_string(),
            description: self.description(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Kill the process after this many seconds (default 120, max 600)"
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory relative to the workspace root (default '.')"
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let command = arguments
            .get("command")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ActionError::InvalidArgument("missing 'command'".into()))?;

        let timeout_secs = arguments
            .get("timeout_seconds")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .clamp(1, MAX_TIMEOUT_SECS);

        let cwd = match arguments.get("cwd").and_then(serde_json::Value::as_str) {
            Some(dir) => {
                let resolved = self.ctx.workspace.resolve(dir)?;
                if !resolved.is_dir() {
                    return Err(ActionError::NotADirectory {
                        path: self.ctx.workspace.display_path(&resolved),
                    });
                }
                resolved
            }
            None => self.ctx.workspace.root().to_path_buf(),
        };

        debug!(command = %command, timeout_secs, "Executing command");

        let mut child = shell_command(command)
            .current_dir(&cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ActionError::io("bash", e))?;

        // Take the pipes before waiting so output cannot be lost to a full
        // OS buffer while the child is still running.
        let mut stdout_pipe = child.stdout.take();
        let mut stderr_pipe = child.stderr.take();

        let collect = async {
            let mut stdout = Vec::new();
            let mut stderr = Vec::new();
            if let Some(pipe) = stdout_pipe.as_mut() {
                let _ = pipe.read_to_end(&mut stdout).await;
            }
            if let Some(pipe) = stderr_pipe.as_mut() {
                let _ = pipe.read_to_end(&mut stderr).await;
            }
            let status = child.wait().await;
            (stdout, stderr, status)
        };

        match tokio::time::timeout(Duration::from_secs(timeout_secs), collect).await {
            Ok((stdout, stderr, status)) => {
                let status = status.map_err(|e| ActionError::io("bash", e))?;
                let rendered = self.render(command, &stdout, &stderr, status.code());

                // Paper Fig. 27: a non-zero exit is surfaced as a tool error so
                // the model cannot mistake a failed command for a passing one.
                // The output is carried in the error rather than discarded.
                match status.code() {
                    Some(0) => Ok(rendered),
                    Some(code) => Err(ActionError::CommandFailed {
                        code,
                        output: rendered,
                    }),
                    None => Err(ActionError::CommandFailed {
                        code: -1,
                        output: rendered,
                    }),
                }
            }
            Err(_) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                warn!(command = %command, timeout_secs, "Command timed out");
                Err(ActionError::Timeout {
                    seconds: timeout_secs,
                    output: String::new(),
                })
            }
        }
    }
}

/// Build the platform shell invocation.
///
/// Unix: `sh -c <command>` is exact; the command is one argv element.
///
/// Windows: `cmd /C <command>` is **not** argv-safe. When the command reaches
/// `cmd` as a quoted argv element, `cmd` strips the inner quotes, so
/// `findstr /c:"enabled=true"` arrives as `/c:enabled=true` and silently
/// matches nothing (verified: exit 1 vs exit 0 for the same command passed via
/// `raw_arg`). `raw_arg` passes the command line through verbatim, which is
/// what a user typing into `cmd` gets.
fn shell_command(command: &str) -> tokio::process::Command {
    #[cfg(windows)]
    {
        let mut cmd = tokio::process::Command::new("cmd");
        cmd.arg("/C");
        cmd.raw_arg(command);
        cmd
    }

    #[cfg(not(windows))]
    {
        let mut cmd = tokio::process::Command::new("/bin/sh");
        cmd.args(["-c", command]);
        cmd
    }
}

impl BashExecutor {
    fn render(&self, command: &str, stdout: &[u8], stderr: &[u8], code: Option<i32>) -> String {
        let mut out = String::new();
        if !stdout.is_empty() {
            out.push_str(&String::from_utf8_lossy(stdout));
        }
        if !stderr.is_empty() {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("STDERR:\n");
            out.push_str(&String::from_utf8_lossy(stderr));
        }
        if out.trim().is_empty() {
            out.push_str("(no output)");
        }
        match code {
            Some(0) => {}
            Some(c) => {
                if !out.ends_with('\n') {
                    out.push('\n');
                }
                out.push_str(&format!("Exit code: {c}"));
            }
            None => out.push_str("\nProcess terminated by signal"),
        }
        debug!(command = %command, bytes = out.len(), "Command finished");
        crate::registry::truncate_output(out, self.ctx.max_output_chars)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod quoting_tests;
