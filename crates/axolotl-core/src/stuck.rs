//! Stuck detection (paper §2.4, Appendix A.4).
//!
//! A *streak* is the run of most recent tool calls sharing the same tool name
//! and byte-identical arguments. A reminder is injected once per streak at five
//! identical calls; an identical *failing* streak is terminated at eight.

/// Reminder injected when a failing call repeats.
pub fn failing_reminder(tool: &str, repeats: u32) -> String {
    format!(
        "<system-reminder>\nYou have called `{tool}` with the SAME arguments {repeats} times and it keeps failing the same way.\nRepeating it will NOT work. STOP — read the actual error, then try a DIFFERENT command, inspect\nmore context, or step back and reconsider your plan. Do NOT issue the same call again.\n</system-reminder>"
    )
}

/// Reminder injected when a call merely repeats.
pub fn repeat_reminder(tool: &str, repeats: u32) -> String {
    format!(
        "<system-reminder>\nYou have called `{tool}` with the SAME arguments {repeats} times. You're not making progress — you\nalready have this result. Move on to the next concrete step instead of repeating it.\n</system-reminder>"
    )
}

/// Outcome of recording one turn's tool calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StuckSignal {
    /// Keep going, optionally with a reminder to inject.
    Continue { reminder: Option<String> },
    /// End the run: the streak is too long.
    Terminate { tool: String, repeats: u32 },
}

/// Tracks streaks of identical calls.
#[derive(Debug)]
pub struct StuckDetector {
    last_signature: Option<String>,
    last_tool: String,
    repeats: u32,
    /// Whether a reminder was already injected for the current streak.
    reminded: bool,
    failing: bool,
    remind_threshold: u32,
    terminate_threshold: u32,
}

impl Default for StuckDetector {
    fn default() -> Self {
        Self::new(5, 8)
    }
}

impl StuckDetector {
    /// Paper §3.1: reminder after five identical calls, terminate after eight
    /// identical failing calls.
    pub fn new(remind_threshold: u32, terminate_threshold: u32) -> Self {
        Self {
            last_signature: None,
            last_tool: String::new(),
            repeats: 0,
            reminded: false,
            failing: false,
            remind_threshold: remind_threshold.max(1),
            terminate_threshold: terminate_threshold.max(1),
        }
    }

    /// Record a tool call.
    ///
    /// Read-only calls that repeat are also tracked: re-reading the same file
    /// is as much a dead end as re-running the same failing test.
    pub fn observe(
        &mut self,
        tool: &str,
        arguments: &serde_json::Value,
        failed: bool,
    ) -> StuckSignal {
        let signature = format!("{tool}\u{0}{arguments}");

        if self.last_signature.as_deref() == Some(signature.as_str()) {
            self.repeats += 1;
            self.failing = failed;
        } else {
            self.last_signature = Some(signature);
            self.last_tool = tool.to_string();
            self.repeats = 1;
            self.reminded = false;
            self.failing = failed;
        }

        if self.failing && self.repeats >= self.terminate_threshold {
            return StuckSignal::Terminate {
                tool: self.last_tool.clone(),
                repeats: self.repeats,
            };
        }

        if self.repeats >= self.remind_threshold && !self.reminded {
            self.reminded = true;
            let reminder = if self.failing {
                failing_reminder(&self.last_tool, self.repeats)
            } else {
                repeat_reminder(&self.last_tool, self.repeats)
            };
            return StuckSignal::Continue {
                reminder: Some(reminder),
            };
        }

        StuckSignal::Continue { reminder: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> serde_json::Value {
        serde_json::json!({ "command": "cargo test" })
    }

    #[test]
    fn distinct_calls_never_trigger() {
        let mut d = StuckDetector::default();
        for i in 0..20 {
            let a = serde_json::json!({ "command": format!("echo {i}") });
            assert_eq!(
                d.observe("bash", &a, false),
                StuckSignal::Continue { reminder: None }
            );
        }
    }

    #[test]
    fn reminder_fires_once_at_five() {
        let mut d = StuckDetector::default();
        for i in 1..=4 {
            assert_eq!(
                d.observe("bash", &args(), false),
                StuckSignal::Continue { reminder: None },
                "call {i}"
            );
        }
        match d.observe("bash", &args(), false) {
            StuckSignal::Continue { reminder: Some(r) } => {
                assert!(r.contains("SAME arguments 5 times"), "{r}");
                assert!(r.contains("not making progress"), "{r}");
            }
            other => panic!("expected reminder, got {other:?}"),
        }
        // Only once per streak.
        assert_eq!(
            d.observe("bash", &args(), false),
            StuckSignal::Continue { reminder: None }
        );
    }

    #[test]
    fn failing_streak_terminates_at_eight() {
        let mut d = StuckDetector::default();
        for _ in 0..7 {
            let signal = d.observe("bash", &args(), true);
            assert!(matches!(signal, StuckSignal::Continue { .. }), "{signal:?}");
        }
        match d.observe("bash", &args(), true) {
            StuckSignal::Terminate { tool, repeats } => {
                assert_eq!(tool, "bash");
                assert_eq!(repeats, 8);
            }
            other => panic!("expected termination, got {other:?}"),
        }
    }

    #[test]
    fn non_failing_streak_never_terminates() {
        let mut d = StuckDetector::default();
        for _ in 0..20 {
            assert!(matches!(
                d.observe("read_file", &args(), false),
                StuckSignal::Continue { .. }
            ));
        }
    }

    #[test]
    fn changing_arguments_resets_the_streak() {
        let mut d = StuckDetector::default();
        for _ in 0..4 {
            d.observe("bash", &args(), true);
        }
        d.observe("bash", &serde_json::json!({ "command": "other" }), true);
        // Back to 1 repeat, so no reminder yet.
        assert_eq!(
            d.observe("bash", &args(), true),
            StuckSignal::Continue { reminder: None }
        );
    }

    #[test]
    fn success_then_failure_uses_current_status() {
        let mut d = StuckDetector::new(2, 3);
        d.observe("bash", &args(), true);
        // The latest call succeeded, so this is not counted as a failing streak.
        assert!(matches!(
            d.observe("bash", &args(), false),
            StuckSignal::Continue { .. }
        ));
    }

    #[test]
    fn reminder_text_names_the_tool() {
        assert!(failing_reminder("grep_text", 5).contains("`grep_text`"));
        assert!(repeat_reminder("read_file", 5).contains("`read_file`"));
    }
}
