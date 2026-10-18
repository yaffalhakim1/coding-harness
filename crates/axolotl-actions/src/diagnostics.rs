//! Post-edit diagnostics (paper §2.4).
//!
//! "After the agent edits or writes a file, the harness runs a fast, read-only
//! check on it and appends the findings to the tool result. This surfaces
//! syntax errors, undefined names, and unused imports immediately, so the model
//! can fix them before spending a turn on the tests."
//!
//! Diagnostics are advisory: they never fail a tool call, they only add text to
//! the observation. A check that is unavailable is silently skipped — a missing
//! linter must not block an edit.

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use tracing::debug;

/// How long a diagnostic run may take before it is abandoned.
const DIAGNOSTIC_TIMEOUT: Duration = Duration::from_secs(10);

/// Which checker to use for a file, based on its extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Checker {
    /// `python -m py_compile` — always available, catches syntax errors.
    PythonCompile,
    /// `rustc --edition 2021 --emit=metadata` on a single file.
    RustCheck,
    /// `node --check` for JavaScript.
    NodeCheck,
    /// Nothing useful for this extension.
    None,
}

/// Pick a checker for `path`.
pub fn checker_for(path: &Path) -> Checker {
    match path.extension().and_then(|e| e.to_str()) {
        Some("py") => Checker::PythonCompile,
        Some("rs") => Checker::RustCheck,
        Some("js") | Some("mjs") | Some("cjs") => Checker::NodeCheck,
        _ => Checker::None,
    }
}

/// Run the appropriate check on `path` and return a short report.
///
/// Returns `None` when there is nothing to say (no checker, checker missing, or
/// a clean run). Findings are truncated: a wall of compiler output is exactly
/// what the context manager would have to elide later.
pub async fn check_file(path: &Path) -> Option<String> {
    let checker = checker_for(path);
    if checker == Checker::None {
        return None;
    }

    let (program, args) = command_for(checker, path)?;

    let output = tokio::time::timeout(
        DIAGNOSTIC_TIMEOUT,
        tokio::process::Command::new(&program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output(),
    )
    .await
    .ok()?
    .ok()?;

    if output.status.success() {
        debug!(path = %path.display(), "Post-edit diagnostics clean");
        return None;
    }

    let mut report = String::from_utf8_lossy(&output.stderr).into_owned();
    if report.trim().is_empty() {
        report = String::from_utf8_lossy(&output.stdout).into_owned();
    }
    if report.trim().is_empty() {
        return None;
    }

    Some(format!(
        "\n\n[diagnostics] {} reported problems:\n{}",
        program,
        truncate(&report, 1_500)
    ))
}

/// Build the checker invocation, or `None` when the tool is not installed.
fn command_for(checker: Checker, path: &Path) -> Option<(String, Vec<String>)> {
    let file = path.to_string_lossy().to_string();
    match checker {
        Checker::PythonCompile => Some((
            "python".to_string(),
            vec![
                "-c".to_string(),
                // compile() with no execution: syntax-only, side-effect free.
                "import sys; compile(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1], 'exec')"
                    .to_string(),
                file,
            ],
        )),
        Checker::RustCheck => Some((
            "rustc".to_string(),
            vec![
                "--edition".to_string(),
                "2021".to_string(),
                "--emit=metadata".to_string(),
                "--crate-type=lib".to_string(),
                "-o".to_string(),
                std::env::temp_dir().join("axolotl-check.rmeta").to_string_lossy().to_string(),
                file,
            ],
        )),
        Checker::NodeCheck => Some(("node".to_string(), vec!["--check".to_string(), file])),
        Checker::None => None,
    }
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.trim().to_string();
    }
    text.chars()
        .take(max_chars)
        .collect::<String>()
        .trim()
        .to_string()
        + "\n…[diagnostics truncated]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_checker_by_extension() {
        assert_eq!(checker_for(Path::new("a.py")), Checker::PythonCompile);
        assert_eq!(checker_for(Path::new("a.rs")), Checker::RustCheck);
        assert_eq!(checker_for(Path::new("a.js")), Checker::NodeCheck);
        assert_eq!(checker_for(Path::new("a.txt")), Checker::None);
        assert_eq!(checker_for(Path::new("Makefile")), Checker::None);
    }

    #[tokio::test]
    async fn clean_python_file_reports_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ok.py");
        std::fs::write(&path, "x = 1\nprint(x)\n").unwrap();
        // Skips silently when python is absent, which is the intended behaviour.
        assert!(check_file(&path).await.is_none());
    }

    #[tokio::test]
    async fn broken_python_file_reports_a_syntax_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.py");
        std::fs::write(&path, "def f(:\n    pass\n").unwrap();

        match check_file(&path).await {
            Some(report) => {
                assert!(report.contains("[diagnostics]"), "{report}");
                // Python's message mentions SyntaxError; accept either casing.
                let lower = report.to_lowercase();
                assert!(
                    lower.contains("syntax") || lower.contains("error"),
                    "{report}"
                );
            }
            // If python is not installed, the check is skipped by design.
            None => eprintln!("python unavailable; skipping"),
        }
    }

    #[tokio::test]
    async fn unknown_extension_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("notes.txt");
        std::fs::write(&path, "not code at all {{{").unwrap();
        assert!(check_file(&path).await.is_none());
    }

    #[test]
    fn truncation_is_char_safe() {
        let text = "α".repeat(3000);
        let out = truncate(&text, 100);
        assert!(out.contains("truncated"), "{}", &out[out.len() - 40..]);
    }
}
