//! Tool approval.
//!
//! A harness that can run `bash` needs a way to ask before it does. This module
//! defines the gate the registry consults and the decision it returns; the
//! wiring to a human lives in the front end (ACP's
//! `session/request_permission`, or a CLI prompt).
//!
//! Policy is deliberately coarse: a tool is either always safe, always needs
//! approval, or needs approval only for some arguments (`bash` is the obvious
//! one). That is enough to make Supervised mode meaningful without inventing a
//! rule language.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

/// What the gate decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    /// Run it.
    Allow,
    /// The user said no. The tool returns a recoverable error.
    Deny,
    /// Nobody answered — the turn was cancelled, or the client went away.
    Cancelled,
}

impl Approval {
    pub fn allowed(&self) -> bool {
        matches!(self, Approval::Allow)
    }
}

/// A request to run one tool call.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub tool: String,
    /// Human-readable summary, e.g. `Running rm -rf build`.
    pub title: String,
    /// The tool category, so a UI can pick an icon.
    pub kind: ToolKindHint,
    /// The raw arguments, for a UI that wants to show them.
    pub arguments: Value,
}

/// Coarse tool category, mirroring the ACP `ToolKind` vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKindHint {
    Read,
    Edit,
    Delete,
    Search,
    Execute,
    Fetch,
    Think,
    Other,
}

impl ToolKindHint {
    /// Whether a category is inherently read-only.
    pub fn is_read_only(&self) -> bool {
        matches!(
            self,
            ToolKindHint::Read | ToolKindHint::Search | ToolKindHint::Think
        )
    }
}

/// Categorise a tool by name.
pub fn kind_of(name: &str) -> ToolKindHint {
    match name {
        "read_file" | "list_files" | "glob_files" => ToolKindHint::Read,
        "grep_text" => ToolKindHint::Search,
        "write_file" | "edit_file" => ToolKindHint::Edit,
        "bash" => ToolKindHint::Execute,
        "web_fetch" => ToolKindHint::Fetch,
        "update_plan" | "task" | "ask_user" => ToolKindHint::Think,
        _ => ToolKindHint::Other,
    }
}

/// One-line description of what a call is about to do.
///
/// Shared with the ACP sink so the approval prompt and the transcript say the
/// same thing about the same call.
pub fn describe(name: &str, arguments: &serde_json::Value) -> String {
    let arg = |key: &str| -> Option<String> {
        arguments
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    };

    match name {
        "read_file" => {
            arg("path").map_or_else(|| "Reading a file".into(), |p| format!("Reading {p}"))
        }
        "write_file" => {
            arg("path").map_or_else(|| "Writing a file".into(), |p| format!("Writing {p}"))
        }
        "edit_file" => {
            arg("path").map_or_else(|| "Editing a file".into(), |p| format!("Editing {p}"))
        }
        "list_files" => "Listing files".into(),
        "glob_files" => {
            arg("pattern").map_or_else(|| "Finding files".into(), |p| format!("Finding {p}"))
        }
        "grep_text" => arg("query").map_or_else(
            || "Searching contents".into(),
            |q| format!("Searching for {q}"),
        ),
        "bash" => arg("command").map_or_else(
            || "Running a command".into(),
            |c| format!("Running {}", c.chars().take(60).collect::<String>()),
        ),
        "web_fetch" => {
            arg("url").map_or_else(|| "Fetching a URL".into(), |u| format!("Fetching {u}"))
        }
        "update_plan" => "Updating the plan".into(),
        "recall_event" => "Recalling elided output".into(),
        "task" => "Delegating to a subagent".into(),
        "ask_user" => "Asking the user a question".into(),
        other => other.replace('_', " "),
    }
}

/// Asks a human (or a policy) whether a tool call may run.
#[async_trait]
pub trait PermissionGate: Send + Sync {
    async fn approve(&self, request: &ApprovalRequest) -> Approval;
}

/// Allows everything. The default, so an unconfigured harness behaves exactly
/// as it did before permissions existed.
pub struct AllowAll;

#[async_trait]
impl PermissionGate for AllowAll {
    async fn approve(&self, _request: &ApprovalRequest) -> Approval {
        Approval::Allow
    }
}

/// Denies everything that reaches it. Useful as a base for a policy that only
/// needs to allow a specific set.
pub struct DenyAll;

#[async_trait]
impl PermissionGate for DenyAll {
    async fn approve(&self, _request: &ApprovalRequest) -> Approval {
        Approval::Deny
    }
}

/// Denies a fixed set of shell command patterns.
///
/// This is the harness's own last line of defence, not a security boundary: a
/// determined model can write the same effect a hundred ways. It exists so the
/// obvious destructive commands fail fast without waking the user.
pub struct DestructiveCommandGate {
    inner: Arc<dyn PermissionGate>,
    patterns: Vec<String>,
}

impl DestructiveCommandGate {
    /// Wrap `inner`, refusing commands containing any of `patterns`.
    pub fn new(inner: Arc<dyn PermissionGate>, patterns: Vec<String>) -> Self {
        Self { inner, patterns }
    }

    /// The default deny list, matching the paper's §2.4 permission layer.
    pub fn with_defaults(inner: Arc<dyn PermissionGate>) -> Self {
        Self::new(
            inner,
            [
                "rm -rf",
                "rm -fr",
                "sudo ",
                "git push",
                "git reset --hard",
                "chmod -R",
                "chown -R",
                "mkfs",
                "dd if=",
                ":(){",
                "shutdown",
                "format ",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        )
    }

    /// Whether a `bash` command matches a destructive pattern.
    pub fn is_destructive(&self, command: &str) -> Option<&str> {
        let lower = command.to_ascii_lowercase();
        self.patterns
            .iter()
            .find(|p| lower.contains(&p.to_ascii_lowercase()))
            .map(String::as_str)
    }
}

#[async_trait]
impl PermissionGate for DestructiveCommandGate {
    async fn approve(&self, request: &ApprovalRequest) -> Approval {
        if request.tool == "bash" {
            if let Some(command) = request.arguments.get("command").and_then(Value::as_str) {
                if let Some(pattern) = self.is_destructive(command) {
                    tracing::warn!(
                        pattern = %pattern,
                        command = %command,
                        "refusing a destructive command without asking"
                    );
                    return Approval::Deny;
                }
            }
        }
        self.inner.approve(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(tool: &str, args: Value) -> ApprovalRequest {
        ApprovalRequest {
            tool: tool.into(),
            title: "test".into(),
            kind: ToolKindHint::Other,
            arguments: args,
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    #[test]
    fn allow_all_allows() {
        let gate = AllowAll;
        assert_eq!(
            rt().block_on(gate.approve(&request("bash", serde_json::json!({})))),
            Approval::Allow
        );
    }

    #[test]
    fn deny_all_denies() {
        let gate = DenyAll;
        assert_eq!(
            rt().block_on(gate.approve(&request("read_file", serde_json::json!({})))),
            Approval::Deny
        );
    }

    #[test]
    fn destructive_commands_are_recognised() {
        let gate = DestructiveCommandGate::with_defaults(Arc::new(AllowAll));
        assert!(gate.is_destructive("rm -rf build").is_some());
        assert!(gate.is_destructive("git reset --hard HEAD").is_some());
        assert!(gate.is_destructive("sudo apt install x").is_some());
        assert!(gate.is_destructive("cargo test").is_none());
        assert!(gate.is_destructive("ls -la").is_none());
    }

    #[test]
    fn destructive_commands_are_denied_even_when_the_inner_gate_allows() {
        let gate = DestructiveCommandGate::with_defaults(Arc::new(AllowAll));
        let approval = rt().block_on(gate.approve(&request(
            "bash",
            serde_json::json!({ "command": "rm -rf /" }),
        )));
        assert_eq!(approval, Approval::Deny);
    }

    #[test]
    fn ordinary_commands_pass_through_to_the_inner_gate() {
        let gate = DestructiveCommandGate::with_defaults(Arc::new(AllowAll));
        let approval = rt().block_on(gate.approve(&request(
            "bash",
            serde_json::json!({ "command": "cargo test" }),
        )));
        assert_eq!(approval, Approval::Allow);
    }

    #[test]
    fn non_bash_tools_are_not_pattern_matched() {
        // A file path containing "rm -rf" is not a destructive command.
        let gate = DestructiveCommandGate::with_defaults(Arc::new(AllowAll));
        let approval = rt().block_on(gate.approve(&request(
            "read_file",
            serde_json::json!({ "path": "notes/rm -rf.md" }),
        )));
        assert_eq!(approval, Approval::Allow);
    }

    #[test]
    fn read_only_kinds_are_classified() {
        assert!(ToolKindHint::Read.is_read_only());
        assert!(ToolKindHint::Search.is_read_only());
        assert!(!ToolKindHint::Execute.is_read_only());
        assert!(!ToolKindHint::Edit.is_read_only());
    }
}
