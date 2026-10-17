//! Session file state: read-before-write and external-modification tracking.
//!
//! Paper §2.4 (Safety): "A read-before-write check refuses to edit or overwrite
//! a file that has not been read in the current session, and detects external
//! modification through a content hash."
//!
//! The state is keyed by *resolved absolute path*, so `./a.txt`, `a.txt` and an
//! absolute path all refer to the same record.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use sha2::{Digest, Sha256};

use crate::error::{ActionError, ActionResult};

/// What the harness knows about one file.
#[derive(Debug, Clone)]
struct FileRecord {
    /// Lines the model has seen.
    lines_read: usize,
    /// Total lines in the file at the time of the read.
    total_lines: usize,
    /// SHA-256 of the content the model saw.
    digest: String,
}

impl FileRecord {
    fn is_full(&self) -> bool {
        self.lines_read >= self.total_lines
    }
}

/// Per-session record of file reads and modifications.
#[derive(Debug, Default)]
pub struct FileState {
    files: Mutex<HashMap<PathBuf, FileRecord>>,
}

impl FileState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the digest of file content.
    pub fn digest(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Record that the model has read `path` (lines `[start_line, ...)`,
    /// 1-based, `lines_read` lines long).
    ///
    /// A record becomes `full_read` only when the read started at line 1 and
    /// covered the whole file, matching the paper's gate: partial reads do not
    /// satisfy read-before-write.
    pub fn record_read(&self, path: &Path, content: &str, start_line: usize, lines_read: usize) {
        let total_lines = content.lines().count();
        let digest = Self::digest(content);
        let mut files = self.files.lock().unwrap_or_else(PoisonError::into_inner);

        // A later partial read must not erase an earlier full read of the same
        // bytes; keep the best coverage seen for this exact content.
        let covered = if start_line <= 1 {
            let end = start_line.saturating_sub(1) + lines_read;
            if end >= total_lines {
                lines_read
            } else {
                0
            }
        } else {
            files
                .get(path)
                .filter(|r| r.digest == digest)
                .map_or(0, |r| r.lines_read)
        };

        files.insert(
            path.to_path_buf(),
            FileRecord {
                lines_read: covered,
                total_lines,
                digest,
            },
        );
    }

    /// Record a successful whole-file write; the model now knows the content.
    pub fn record_write(&self, path: &Path, content: &str) {
        let total_lines = content.lines().count();
        self.files
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(
                path.to_path_buf(),
                FileRecord {
                    lines_read: total_lines,
                    total_lines,
                    digest: Self::digest(content),
                },
            );
    }

    /// Enforce read-before-write for an existing file.
    ///
    /// `current` is the content on disk right now; a digest mismatch means the
    /// file changed outside the harness (another process, a `bash` edit), so
    /// the stale record must not authorise a write.
    pub fn ensure_read_for_write(
        &self,
        path: &Path,
        display: &str,
        current: &str,
    ) -> ActionResult<()> {
        let files = self.files.lock().unwrap_or_else(PoisonError::into_inner);
        let record = files
            .get(path)
            .ok_or_else(|| ActionError::ReadBeforeWrite {
                path: display.to_string(),
            })?;

        if record.digest != Self::digest(current) {
            return Err(ActionError::ExternallyModified {
                path: display.to_string(),
            });
        }

        if !record.is_full() {
            return Err(ActionError::PartialRead {
                path: display.to_string(),
                lines: record.lines_read,
                total: record.total_lines,
            });
        }

        Ok(())
    }

    /// Whether a full read is recorded for `path`.
    pub fn has_full_read(&self, path: &Path) -> bool {
        self.files
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(path)
            .is_some_and(FileRecord::is_full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> PathBuf {
        PathBuf::from("/ws/a.txt")
    }

    #[test]
    fn write_requires_prior_read() {
        let state = FileState::new();
        let err = state.ensure_read_for_write(&p(), "a.txt", "x").unwrap_err();
        assert!(matches!(err, ActionError::ReadBeforeWrite { .. }));
    }

    #[test]
    fn full_read_authorises_write() {
        let state = FileState::new();
        let content = "a\nb\nc";
        state.record_read(&p(), content, 1, 3);
        assert!(state.has_full_read(&p()));
        assert!(state.ensure_read_for_write(&p(), "a.txt", content).is_ok());
    }

    #[test]
    fn partial_read_does_not_authorise_write() {
        let state = FileState::new();
        let content = "a\nb\nc\nd";
        state.record_read(&p(), content, 1, 2);
        assert!(!state.has_full_read(&p()));
        let err = state
            .ensure_read_for_write(&p(), "a.txt", content)
            .unwrap_err();
        assert!(matches!(err, ActionError::PartialRead { .. }));
    }

    #[test]
    fn read_starting_past_line_one_is_partial() {
        let state = FileState::new();
        let content = "a\nb\nc";
        state.record_read(&p(), content, 2, 2);
        assert!(!state.has_full_read(&p()));
    }

    #[test]
    fn external_modification_is_detected() {
        let state = FileState::new();
        state.record_read(&p(), "original", 1, 1);
        let err = state
            .ensure_read_for_write(&p(), "a.txt", "changed by bash")
            .unwrap_err();
        assert!(matches!(err, ActionError::ExternallyModified { .. }));
    }

    #[test]
    fn write_updates_state() {
        let state = FileState::new();
        state.record_write(&p(), "new\ncontent");
        assert!(state.has_full_read(&p()));
        assert!(state
            .ensure_read_for_write(&p(), "a.txt", "new\ncontent")
            .is_ok());
    }
}
