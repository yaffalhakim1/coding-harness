//! Session archive: full-history persistence, listing, and resume.
//!
//! The trajectory file (`<id>.json`) is a *summary* for analysis — previews
//! instead of full tool output, so sweeps stay readable. Resuming needs the
//! actual conversation, so it is stored separately as a full session snapshot.
//!
//! Layout under the session directory:
//!
//! ```text
//! <dir>/<id>.json        trajectory summary (analysis)
//! <dir>/<id>.session.json full session (resume)
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::session::Session;

/// Suffix distinguishing a resumable snapshot from an analysis trajectory.
pub const SESSION_SUFFIX: &str = ".session.json";

/// Write a full session snapshot that can be resumed later.
pub async fn write_session(session: &Session, dir: impl AsRef<Path>) -> Result<PathBuf> {
    let dir = dir.as_ref();
    tokio::fs::create_dir_all(dir)
        .await
        .with_context(|| format!("creating session dir '{}'", dir.display()))?;

    let path = dir.join(format!("{}{SESSION_SUFFIX}", session.id));
    let json = serde_json::to_string_pretty(session).context("serializing session")?;

    let tmp = dir.join(format!("{}.tmp", session.id));
    tokio::fs::write(&tmp, json)
        .await
        .with_context(|| format!("writing '{}'", tmp.display()))?;
    tokio::fs::rename(&tmp, &path)
        .await
        .with_context(|| format!("finalizing '{}'", path.display()))?;

    info!(path = %path.display(), "Session snapshot written");
    Ok(path)
}

/// Load a session snapshot by id or path.
pub async fn load_session(reference: &str, dir: impl AsRef<Path>) -> Result<Session> {
    let path = resolve_session(reference, dir.as_ref())?;
    let text = tokio::fs::read_to_string(&path)
        .await
        .with_context(|| format!("reading '{}'", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing '{}'", path.display()))
}

/// Resolve a reference to a session snapshot file.
///
/// Accepts an existing file, `<dir>/<id>.session.json`, or `<dir>/<id>.json`
/// when only a trajectory exists (the caller gets a clear error in that case,
/// because a summary cannot be resumed).
pub fn resolve_session(reference: &str, dir: &Path) -> Result<PathBuf> {
    let direct = Path::new(reference);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }

    let snapshot = dir.join(format!("{reference}{SESSION_SUFFIX}"));
    if snapshot.is_file() {
        return Ok(snapshot);
    }

    let summary = dir.join(format!("{reference}.json"));
    if summary.is_file() {
        anyhow::bail!(
            "'{reference}' has only an analysis trajectory at {}; \
             a resumable snapshot ({reference}{SESSION_SUFFIX}) was never written",
            summary.display()
        );
    }

    anyhow::bail!("no session found for '{reference}' in '{}'", dir.display())
}

/// A row in the session list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub task_id: String,
    /// First line of the task description.
    pub task: String,
    pub outcome: Option<String>,
    pub turns: usize,
    pub total_tokens: usize,
    pub started_at: Option<DateTime<Utc>>,
    pub duration_secs: u64,
    /// Whether a resumable snapshot exists (vs only an analysis summary).
    pub resumable: bool,
}

/// List sessions in `dir`, newest first.
///
/// Reads both snapshot and summary files; a session present in both appears
/// once, marked resumable.
pub async fn list_sessions(dir: impl AsRef<Path>) -> Result<Vec<SessionInfo>> {
    let dir = dir.as_ref();
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries = tokio::fs::read_dir(dir)
        .await
        .with_context(|| format!("reading '{}'", dir.display()))?;

    let mut by_id: std::collections::HashMap<String, SessionInfo> =
        std::collections::HashMap::new();

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.ends_with(".json") {
            continue;
        }
        let Ok(text) = tokio::fs::read_to_string(&path).await else {
            continue;
        };
        let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };

        // The snapshot serializes the `Session` struct directly (`id`), while
        // the analysis trajectory uses `session_id`.
        let Some(id) = json["session_id"]
            .as_str()
            .or_else(|| json["id"].as_str())
            .map(str::to_string)
        else {
            continue;
        };
        let is_snapshot = name.ends_with(SESSION_SUFFIX);

        // The snapshot has richer data; prefer it and mark resumable.
        if let Some(existing) = by_id.get(&id) {
            if existing.resumable && !is_snapshot {
                continue;
            }
        }

        let info = SessionInfo {
            id: id.clone(),
            task_id: json["task_id"]
                .as_str()
                .or_else(|| json["task"]["id"].as_str())
                .unwrap_or("?")
                .to_string(),
            task: describe_task(&json),
            // Summary files store a lowercase label; snapshots store the enum.
            // `outcome_label` handles both shapes, so use it for everything.
            outcome: outcome_label(&json["outcome"]),
            turns: json["total_turns"]
                .as_u64()
                .or_else(|| json["turns"].as_array().map(|a| a.len() as u64))
                .unwrap_or(0) as usize,
            total_tokens: json["total_tokens"]
                .as_u64()
                .or_else(|| json["total_usage"]["total_tokens"].as_u64())
                .unwrap_or(0) as usize,
            started_at: json["started_at"]
                .as_str()
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|d| d.with_timezone(&Utc)),
            duration_secs: json["duration_secs"].as_u64().unwrap_or(0),
            resumable: is_snapshot,
        };
        by_id.insert(id, info);
    }

    let mut sessions: Vec<SessionInfo> = by_id.into_values().collect();
    // Newest first; sessions without a timestamp sort last.
    sessions.sort_by_key(|s| std::cmp::Reverse(s.started_at));
    Ok(sessions)
}

/// Pull a human-readable task line out of either file shape.
fn describe_task(json: &serde_json::Value) -> String {
    // Summary shape: `task_id` only. Snapshot shape: full `task` object.
    if let Some(desc) = json["task"]["description"].as_str() {
        return first_line(desc);
    }
    json["task_id"].as_str().unwrap_or("?").to_string()
}

/// Label an outcome that was serialized as a tagged enum.
///
/// The snapshot writes `SessionOutcome` structurally, so the variant name is
/// the first key (or the string itself). Variant names are CamelCase, but the
/// canonical labels are snake_case — and some variants already contain an
/// acronym boundary (`TurnLimitReached` → `turn_limit`), so the mapping is
/// explicit rather than mechanical.
fn outcome_label(value: &serde_json::Value) -> Option<String> {
    let variant = value
        .as_str()
        .map(str::to_string)
        .or_else(|| value.as_object()?.keys().next().cloned())?;

    let label = match variant.as_str() {
        "Success" => "success",
        "Failure" => "failure",
        "Unverified" => "unverified",
        "TurnLimitReached" => "turn_limit",
        "Timeout" => "timeout",
        "ContextOverflow" => "context_overflow",
        "Stuck" => "stuck",
        "ApiError" => "api_error",
        "Error" => "error",
        // A label that is already snake_case (from a summary file) passes
        // through unchanged.
        other => other,
    };
    Some(label.to_string())
}

fn first_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or("").trim();
    if line.chars().count() > 80 {
        line.chars().take(80).collect::<String>() + "…"
    } else {
        line.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{SessionConfig, SessionOutcome};
    use crate::trajectory::write_trajectory;
    use crate::types::Task;

    fn session() -> Session {
        let mut s = Session::new(
            Task::from_description("fix the parser bug\nmore detail"),
            SessionConfig::default(),
        );
        s.complete(SessionOutcome::Timeout);
        s
    }

    #[tokio::test]
    async fn snapshot_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let original = session();
        write_session(&original, dir.path()).await.unwrap();

        let loaded = load_session(&original.id, dir.path()).await.unwrap();
        assert_eq!(loaded.id, original.id);
        assert_eq!(loaded.task.description, original.task.description);
        assert_eq!(loaded.outcome, original.outcome);
    }

    #[tokio::test]
    async fn snapshot_preserves_turns() {
        let dir = tempfile::tempdir().unwrap();
        let mut original = session();
        // A turn is what makes a session resumable rather than just recorded.
        let response = axolotl_models::types::LLMResponse {
            message: axolotl_models::types::ChatMessage::assistant("hello"),
            usage: Default::default(),
            stop_reason: axolotl_models::types::StopReason::EndTurn,
            model: "m".into(),
        };
        original.record_turn(crate::turn::Turn::new(0, vec![], 0, response));
        write_session(&original, dir.path()).await.unwrap();

        let loaded = load_session(&original.id, dir.path()).await.unwrap();
        assert_eq!(loaded.turns.len(), 1);
    }

    #[tokio::test]
    async fn resolving_a_summary_only_session_explains_why() {
        let dir = tempfile::tempdir().unwrap();
        let s = session();
        write_trajectory(&s, dir.path()).await.unwrap();

        let err = load_session(&s.id, dir.path()).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("resumable snapshot"), "{msg}");
    }

    #[tokio::test]
    async fn unknown_session_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load_session("nope", dir.path()).await.is_err());
    }

    #[tokio::test]
    async fn list_reports_snapshots_as_resumable() {
        let dir = tempfile::tempdir().unwrap();
        let s = session();
        write_session(&s, dir.path()).await.unwrap();

        let list = list_sessions(dir.path()).await.unwrap();
        assert_eq!(list.len(), 1);
        assert!(list[0].resumable);
        assert!(list[0].task.contains("fix the parser bug"));
    }

    #[tokio::test]
    async fn list_marks_summary_only_sessions_not_resumable() {
        let dir = tempfile::tempdir().unwrap();
        write_trajectory(&session(), dir.path()).await.unwrap();

        let list = list_sessions(dir.path()).await.unwrap();
        assert_eq!(list.len(), 1);
        assert!(!list[0].resumable);
    }

    #[tokio::test]
    async fn list_deduplicates_a_session_present_in_both_files() {
        let dir = tempfile::tempdir().unwrap();
        let s = session();
        write_trajectory(&s, dir.path()).await.unwrap();
        write_session(&s, dir.path()).await.unwrap();

        let list = list_sessions(dir.path()).await.unwrap();
        assert_eq!(list.len(), 1, "one session, not two rows");
        assert!(list[0].resumable, "the snapshot should win");
    }

    #[tokio::test]
    async fn list_of_a_missing_directory_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope");
        assert!(list_sessions(&missing).await.unwrap().is_empty());
    }

    #[test]
    fn first_line_truncates_long_descriptions() {
        let long = "x".repeat(200);
        assert!(first_line(&long).ends_with('…'));
    }

    #[test]
    fn outcome_labels_match_the_session_variants() {
        // Every variant must map to the same label `SessionOutcome::label`
        // reports, or the session list disagrees with the trajectory.
        for (json, expected) in [
            (serde_json::json!({ "Success": {} }), "success"),
            (serde_json::json!({ "Failure": {} }), "failure"),
            (serde_json::json!({ "Unverified": {} }), "unverified"),
            (
                serde_json::json!({ "TurnLimitReached": null }),
                "turn_limit",
            ),
            (serde_json::json!({ "Timeout": null }), "timeout"),
            (
                serde_json::json!({ "ContextOverflow": null }),
                "context_overflow",
            ),
            (serde_json::json!({ "Stuck": {} }), "stuck"),
            (serde_json::json!({ "ApiError": "x" }), "api_error"),
            (serde_json::json!({ "Error": "x" }), "error"),
        ] {
            assert_eq!(outcome_label(&json).as_deref(), Some(expected), "{json}");
        }
        // Summary files already hold the label.
        assert_eq!(
            outcome_label(&serde_json::json!("success")).as_deref(),
            Some("success")
        );
        assert_eq!(outcome_label(&serde_json::Value::Null), None);
    }

    #[tokio::test]
    async fn list_reads_snapshot_turn_and_token_counts() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = session();
        s.total_usage.total_tokens = 4321;
        write_session(&s, dir.path()).await.unwrap();

        let list = list_sessions(dir.path()).await.unwrap();
        assert_eq!(list[0].total_tokens, 4321);
        assert_eq!(list[0].outcome.as_deref(), Some("timeout"));
    }
}
