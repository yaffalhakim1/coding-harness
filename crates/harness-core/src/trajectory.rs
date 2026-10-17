//! Trajectory persistence (implementation plan §5.3).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::info;

use crate::session::Session;

/// Directory where trajectories are written by default.
pub const DEFAULT_TRAJECTORY_DIR: &str = "trajectories";

/// Write the session's trajectory to `dir/<session_id>.json`.
///
/// Returns the written path so callers can report it.
pub async fn write_trajectory(session: &Session, dir: impl AsRef<Path>) -> Result<PathBuf> {
    let dir = dir.as_ref();
    tokio::fs::create_dir_all(dir)
        .await
        .with_context(|| format!("creating trajectory dir '{}'", dir.display()))?;

    let path = dir.join(format!("{}.json", session.id));
    let json = serde_json::to_string_pretty(&session.to_trajectory_json())
        .context("serializing trajectory")?;

    // Write-then-rename so a crashed run never leaves a half-written file that
    // a later analysis pass would silently read.
    let tmp = dir.join(format!("{}.json.tmp", session.id));
    tokio::fs::write(&tmp, json)
        .await
        .with_context(|| format!("writing '{}'", tmp.display()))?;
    tokio::fs::rename(&tmp, &path)
        .await
        .with_context(|| format!("finalizing '{}'", path.display()))?;

    info!(path = %path.display(), "Trajectory written");
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{SessionConfig, SessionOutcome};
    use crate::types::Task;

    #[tokio::test]
    async fn writes_a_readable_trajectory() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = Session::new(Task::from_description("x"), SessionConfig::default());
        session.complete(SessionOutcome::Timeout);

        let path = write_trajectory(&session, dir.path()).await.unwrap();
        assert!(path.exists());

        let text = std::fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["session_id"], session.id);
        assert_eq!(json["outcome"], "timeout");
    }

    #[tokio::test]
    async fn creates_missing_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a/b/c");
        let session = Session::new(Task::from_description("x"), SessionConfig::default());
        let path = write_trajectory(&session, &nested).await.unwrap();
        assert!(path.exists());
    }

    #[tokio::test]
    async fn leaves_no_temp_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let session = Session::new(Task::from_description("x"), SessionConfig::default());
        write_trajectory(&session, dir.path()).await.unwrap();

        let entries: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(entries.len(), 1, "{entries:?}");
    }
}
