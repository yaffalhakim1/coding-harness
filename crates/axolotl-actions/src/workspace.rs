//! Workspace guard.
//!
//! Paper §2.4 (Safety): "A workspace guard resolves each path and rejects any
//! that escapes the project root, including through symlinks."
//!
//! Every tool goes through [`Workspace`] before touching the filesystem, so a
//! model emitting `../../etc/passwd` or a path through a symlink gets a
//! recoverable tool error instead of an escape.

use std::path::{Component, Path, PathBuf};

use crate::error::ActionError;

/// A resolved workspace root that tool paths are confined to.
#[derive(Debug, Clone)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    /// Canonicalise `root`, failing if it does not exist.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, ActionError> {
        let root = root.as_ref();
        let canonical = root.canonicalize().map_err(|e| {
            ActionError::Workspace(format!("workspace root '{}': {e}", root.display()))
        })?;
        Ok(Self { root: canonical })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a model-supplied path to an absolute path inside the workspace.
    ///
    /// Absolute paths are accepted only when they already point inside the
    /// root; anything else is rejected with [`ActionError::PathEscape`].
    pub fn resolve(&self, requested: &str) -> Result<PathBuf, ActionError> {
        if requested.trim().is_empty() {
            return Err(ActionError::InvalidArgument(
                "path must not be empty".into(),
            ));
        }

        let raw = Path::new(requested);
        let joined = if raw.is_absolute() {
            normalize(raw)
        } else {
            normalize(&self.root.join(raw))
        };

        if !joined.starts_with(&self.root) {
            return Err(ActionError::PathEscape(requested.to_string()));
        }

        // A symlink inside the workspace can still point outside it. Resolve
        // the deepest existing ancestor and re-check.
        self.verify_no_symlink_escape(&joined)?;

        Ok(joined)
    }

    /// Reject resolved paths whose real location escapes the root.
    fn verify_no_symlink_escape(&self, path: &Path) -> Result<(), ActionError> {
        let mut probe = path;
        loop {
            match probe.canonicalize() {
                Ok(real) if !real.starts_with(&self.root) => {
                    return Err(ActionError::PathEscape(
                        path.strip_prefix(&self.root)
                            .unwrap_or(path)
                            .display()
                            .to_string(),
                    ));
                }
                Ok(_) => return Ok(()),
                Err(_) => match probe.parent() {
                    Some(parent) if parent.starts_with(&self.root) => probe = parent,
                    _ => return Ok(()),
                },
            }
        }
    }

    /// Path relative to the root, for display in tool output.
    pub fn display_path(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .unwrap_or(path)
            .display()
            .to_string()
    }
}

/// Lexically remove `.` and fold `..` without touching the filesystem.
///
/// `Path::canonicalize` would fail for paths that do not exist yet (a
/// `write_file` to a new file), so resolution has to be lexical first and
/// verified against existing ancestors second.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_workspace() -> (tempfile::TempDir, Workspace) {
        let dir = tempfile::tempdir().unwrap();
        let ws = Workspace::open(dir.path()).unwrap();
        (dir, ws)
    }

    #[test]
    fn resolves_relative_paths_inside_root() {
        let (_dir, ws) = temp_workspace();
        let resolved = ws.resolve("src/main.rs").unwrap();
        assert!(resolved.starts_with(ws.root()));
        assert!(resolved.ends_with("src/main.rs"));
    }

    #[test]
    fn rejects_parent_traversal() {
        let (_dir, ws) = temp_workspace();
        let err = ws.resolve("../escape.txt").unwrap_err();
        assert!(matches!(err, ActionError::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn rejects_nested_traversal_that_escapes() {
        let (_dir, ws) = temp_workspace();
        let err = ws.resolve("src/../../escape.txt").unwrap_err();
        assert!(matches!(err, ActionError::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn allows_nested_traversal_that_stays_inside() {
        let (_dir, ws) = temp_workspace();
        let resolved = ws.resolve("src/../main.rs").unwrap();
        assert!(resolved.ends_with("main.rs"));
    }

    #[test]
    fn rejects_absolute_path_outside_root() {
        let (_dir, ws) = temp_workspace();
        let outside = if cfg!(windows) {
            "C:\\Windows\\win.ini"
        } else {
            "/etc/passwd"
        };
        let err = ws.resolve(outside).unwrap_err();
        assert!(matches!(err, ActionError::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn dot_is_inside_root() {
        let (_dir, ws) = temp_workspace();
        assert_eq!(ws.resolve(".").unwrap(), ws.root());
    }

    #[test]
    fn rejects_empty_path() {
        let (_dir, ws) = temp_workspace();
        assert!(matches!(
            ws.resolve("  ").unwrap_err(),
            ActionError::InvalidArgument(_)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        let (dir, ws) = temp_workspace();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();

        let err = ws.resolve("link/secret.txt").unwrap_err();
        assert!(matches!(err, ActionError::PathEscape(_)), "got {err:?}");
    }

    #[cfg(windows)]
    #[test]
    fn rejects_symlink_escape() {
        let (dir, ws) = temp_workspace();
        let outside = tempfile::tempdir().unwrap();
        if std::os::windows::fs::symlink_dir(outside.path(), dir.path().join("link")).is_err() {
            // Creating symlinks on Windows needs developer mode; skip.
            return;
        }
        let err = ws.resolve("link/secret.txt").unwrap_err();
        assert!(matches!(err, ActionError::PathEscape(_)), "got {err:?}");
    }

    #[test]
    fn display_path_is_relative() {
        let (_dir, ws) = temp_workspace();
        let p = ws.resolve("a/b.txt").unwrap();
        assert_eq!(
            ws.display_path(&p),
            Path::new("a").join("b.txt").display().to_string()
        );
    }

    #[test]
    fn open_fails_for_missing_root() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope");
        assert!(Workspace::open(missing).is_err());
    }

    #[test]
    fn rewrite_writes_inside_root() {
        let (dir, ws) = temp_workspace();
        let p = ws.resolve("x.txt").unwrap();
        fs::write(&p, "hi").unwrap();
        assert!(dir.path().join("x.txt").exists());
    }
}
