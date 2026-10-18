//! Trajectory analysis (paper §C.1).
//!
//! The paper decomposes trajectories into interpretable categories rather than
//! judging only by final success:
//!
//! - **Failure stage** for unresolved SWE-Bench runs: file localization → line
//!   localization → patch implementation → verification. The *earliest* failed
//!   stage is assigned, so a downstream symptom does not mask an upstream error.
//! - **Behaviour encoding** for SWE-Bench turns: Localize / Reproduce / Fix /
//!   Verify / Other.
//! - **Action encoding** for Terminal-Bench calls: understand / write / verify /
//!   other.
//!
//! The paper uses an LLM judge for these labels. This module uses deterministic
//! rules instead: cheaper, reproducible, and good enough to answer "where do
//! runs die", which is what the harness needs to tune itself.

use serde::{Deserialize, Serialize};

use axolotl_core::Session;

/// Where a failed run broke down (paper §C.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureStage {
    /// Never found the relevant file.
    FileLocalization,
    /// Found the file but not the relevant lines.
    LineLocalization,
    /// Found the location but the edit did not apply or was wrong.
    PatchImplementation,
    /// Edited, but verification still failed.
    Verification,
    /// Ran out of turns/time/context before any of the above.
    Exhausted,
    /// Not applicable (the run succeeded).
    None,
}

impl FailureStage {
    pub fn label(&self) -> &'static str {
        match self {
            FailureStage::FileLocalization => "file_localization",
            FailureStage::LineLocalization => "line_localization",
            FailureStage::PatchImplementation => "patch_implementation",
            FailureStage::Verification => "verification",
            FailureStage::Exhausted => "exhausted",
            FailureStage::None => "none",
        }
    }
}

/// What a turn was for (paper §C.1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Locating code: read, grep, glob, list.
    Localize,
    /// Creating or running a reproduction.
    Reproduce,
    /// Editing repository source.
    Fix,
    /// Running validation.
    Verify,
    /// Setup, navigation, planning, recall.
    Other,
}

impl Phase {
    pub fn label(&self) -> &'static str {
        match self {
            Phase::Localize => "localize",
            Phase::Reproduce => "reproduce",
            Phase::Fix => "fix",
            Phase::Verify => "verify",
            Phase::Other => "other",
        }
    }
}

/// Per-run analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryAnalysis {
    pub session_id: String,
    pub task_id: String,
    pub outcome: String,
    pub passed: bool,
    pub turns: usize,
    /// Turns spent in each phase, in order.
    pub phases: Vec<Phase>,
    pub phase_counts: PhaseCounts,
    /// Earliest failed stage, for unresolved runs.
    pub failure_stage: FailureStage,
    /// Whether the run ever edited a file.
    pub edited: bool,
    /// Whether the run ever ran a verification command.
    pub verified: bool,
    pub tool_calls: usize,
    pub failed_tool_calls: usize,
    pub total_tokens: usize,
    pub peak_context_tokens: usize,
    pub summarize_calls: usize,
    /// Terminal outcome detail, when the session ended abnormally.
    pub termination: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct PhaseCounts {
    pub localize: usize,
    pub reproduce: usize,
    pub fix: usize,
    pub verify: usize,
    pub other: usize,
}

impl PhaseCounts {
    fn add(&mut self, phase: Phase) {
        match phase {
            Phase::Localize => self.localize += 1,
            Phase::Reproduce => self.reproduce += 1,
            Phase::Fix => self.fix += 1,
            Phase::Verify => self.verify += 1,
            Phase::Other => self.other += 1,
        }
    }
}

/// Classify a tool call into a phase.
///
/// Rules follow paper Table 8's "Source in trajectory" column.
pub fn classify_tool(name: &str, arguments: &serde_json::Value) -> Phase {
    let command = arguments
        .get("command")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match name {
        "read_file" | "list_files" | "glob_files" | "grep_text" => Phase::Localize,
        "edit_file" => Phase::Fix,
        "write_file" => {
            // Writing a scratch repro script is reproduction; writing source is
            // a fix. The path is the only signal available.
            let path = arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if path.contains("repro") || path.contains("scratch") || path.contains("tmp") {
                Phase::Reproduce
            } else {
                Phase::Fix
            }
        }
        "bash" => classify_command(&command),
        "update_plan" | "recall_event" | "web_fetch" => Phase::Other,
        _ => Phase::Other,
    }
}

/// Classify a shell command.
fn classify_command(command: &str) -> Phase {
    let first = command.split_whitespace().next().unwrap_or("");

    // Verification first: a test runner is unambiguous.
    if contains_any(
        command,
        &[
            "pytest",
            "cargo test",
            "cargo check",
            "cargo build",
            "npm test",
            "npm run",
            "go test",
            "unittest",
            "ruff",
            "clippy",
            "mypy",
            "tsc",
            "jest",
            "make test",
        ],
    ) {
        return Phase::Verify;
    }

    // A redirect that writes into a file is an edit, whatever produced the text.
    // Checked before the read commands because `echo x > f` also contains `echo`.
    if writes_a_file(command) {
        return Phase::Fix;
    }

    // Reproduction: running a script the agent wrote.
    if contains_any(
        command,
        &["python ", "python3 ", "node ", "./run", "bash ./", "sh ./"],
    ) {
        return Phase::Reproduce;
    }

    // Localization: inspection commands.
    if matches!(
        first,
        "ls" | "dir"
            | "cat"
            | "type"
            | "grep"
            | "rg"
            | "find"
            | "findstr"
            | "head"
            | "tail"
            | "wc"
            | "where"
            | "which"
    ) || contains_any(command, &["git status", "git diff", "git log", "git show"])
    {
        return Phase::Localize;
    }

    Phase::Other
}

/// Whether a command mutates a file on disk.
///
/// Shell writes are how a bash-only agent edits code, so without this every
/// bash-only run would report zero fixes and a bogus failure stage.
fn writes_a_file(command: &str) -> bool {
    // Redirection into a path (but not `2>` or `> NUL`, which are sinks).
    let mut rest = command;
    while let Some(pos) = rest.find('>') {
        let after = rest[pos + 1..].trim_start_matches('>').trim_start();
        if after.is_empty() {
            break;
        }
        let target = after.split_whitespace().next().unwrap_or("");
        if !target.is_empty()
            && !target.eq_ignore_ascii_case("nul")
            && !target.starts_with("&")
            && !target.starts_with('&')
        {
            return true;
        }
        rest = &rest[pos + 1..];
    }

    // Windows/POSIX file-writing verbs.
    contains_any(
        command,
        &[
            "set-content",
            "out-file",
            "add-content",
            "tee ",
            "sed -i",
            "touch ",
            "mkdir ",
            "copy ",
            "move ",
            "del ",
            "rm ",
            "ren ",
        ],
    )
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// Analyse a finished session.
pub fn analyze(session: &Session) -> TrajectoryAnalysis {
    let passed = session.outcome.as_ref().is_some_and(|o| o.is_success());

    let mut phases = Vec::new();
    let mut counts = PhaseCounts::default();
    let mut tool_calls = 0usize;
    let mut failed_tool_calls = 0usize;

    for turn in &session.turns {
        // A turn's phase is the phase of its first action; later calls in the
        // same turn are usually a continuation of it.
        if let Some(first) = turn.tool_calls.first() {
            let phase = classify_tool(&first.name, &first.arguments);
            phases.push(phase);
            counts.add(phase);
        }
        tool_calls += turn.tool_calls.len();
        failed_tool_calls += turn.tool_results.iter().filter(|r| !r.success).count();
    }

    let edited = counts.fix > 0;
    let verified = counts.verify > 0;

    let failure_stage = if passed {
        FailureStage::None
    } else {
        assign_failure_stage(counts, edited, verified, session)
    };

    TrajectoryAnalysis {
        session_id: session.id.clone(),
        task_id: session.task.id.clone(),
        outcome: session
            .outcome
            .as_ref()
            .map(|o| o.label().to_string())
            .unwrap_or_else(|| "none".into()),
        passed,
        turns: session.turn_count(),
        phases,
        phase_counts: counts,
        failure_stage,
        edited,
        verified,
        tool_calls,
        failed_tool_calls,
        total_tokens: session.total_usage.total_tokens,
        peak_context_tokens: session.peak_context_tokens,
        summarize_calls: session.summarize_calls,
        termination: session.outcome.as_ref().and_then(|o| match o {
            axolotl_core::SessionOutcome::TurnLimitReached => Some("turn_limit".into()),
            axolotl_core::SessionOutcome::Timeout => Some("timeout".into()),
            axolotl_core::SessionOutcome::ContextOverflow => Some("context_overflow".into()),
            axolotl_core::SessionOutcome::Stuck { tool, repeats } => {
                Some(format!("stuck:{tool}x{repeats}"))
            }
            _ => None,
        }),
    }
}

/// Earliest failed stage, per paper §C.1.1.
fn assign_failure_stage(
    counts: PhaseCounts,
    edited: bool,
    verified: bool,
    session: &Session,
) -> FailureStage {
    // Ran out of resources before doing anything conclusive.
    if matches!(
        session.outcome,
        Some(axolotl_core::SessionOutcome::TurnLimitReached)
            | Some(axolotl_core::SessionOutcome::Timeout)
            | Some(axolotl_core::SessionOutcome::ContextOverflow)
    ) {
        return FailureStage::Exhausted;
    }

    if counts.localize == 0 {
        return FailureStage::FileLocalization;
    }
    if !edited {
        // Found things to read but never changed anything: it never located the
        // line it needed to change.
        return FailureStage::LineLocalization;
    }
    if !verified {
        // Edited without checking: the patch is unvalidated.
        return FailureStage::PatchImplementation;
    }
    FailureStage::Verification
}

/// Aggregate over many runs, for the paper's reporting tables.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisSummary {
    pub runs: usize,
    pub passed: usize,
    pub mean_turns: f64,
    pub mean_tool_calls: f64,
    pub mean_failed_tool_calls: f64,
    /// Runs whose earliest failure was each stage.
    pub by_stage: Vec<(String, usize)>,
    /// Total turns spent in each phase, across all runs.
    pub by_phase: Vec<(String, usize)>,
}

impl AnalysisSummary {
    pub fn from_runs(runs: &[TrajectoryAnalysis]) -> Self {
        if runs.is_empty() {
            return Self::default();
        }
        let mut stage_totals: Vec<(String, usize)> = Vec::new();
        let mut phase_totals = PhaseCounts::default();

        for run in runs {
            if !run.passed {
                let label = run.failure_stage.label().to_string();
                match stage_totals.iter_mut().find(|(l, _)| *l == label) {
                    Some((_, n)) => *n += 1,
                    None => stage_totals.push((label, 1)),
                }
            }
            phase_totals.localize += run.phase_counts.localize;
            phase_totals.reproduce += run.phase_counts.reproduce;
            phase_totals.fix += run.phase_counts.fix;
            phase_totals.verify += run.phase_counts.verify;
            phase_totals.other += run.phase_counts.other;
        }
        stage_totals.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

        let n = runs.len() as f64;
        Self {
            runs: runs.len(),
            passed: runs.iter().filter(|r| r.passed).count(),
            mean_turns: runs.iter().map(|r| r.turns as f64).sum::<f64>() / n,
            mean_tool_calls: runs.iter().map(|r| r.tool_calls as f64).sum::<f64>() / n,
            mean_failed_tool_calls: runs.iter().map(|r| r.failed_tool_calls as f64).sum::<f64>()
                / n,
            by_stage: stage_totals,
            by_phase: vec![
                ("localize".into(), phase_totals.localize),
                ("reproduce".into(), phase_totals.reproduce),
                ("fix".into(), phase_totals.fix),
                ("verify".into(), phase_totals.verify),
                ("other".into(), phase_totals.other),
            ],
        }
    }
}

#[cfg(test)]
mod tests;
