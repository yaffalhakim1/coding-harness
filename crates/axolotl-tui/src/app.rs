//! Application state for the TUI.
//!
//! Deliberately plain data plus small transitions: rendering reads it, the event
//! pump writes it, and neither depends on the other.

use std::collections::VecDeque;

use axolotl_core::progress::Progress;

/// How many transcript lines to retain.
///
/// A long run emits thousands of events; keeping them all would grow without
/// bound for no benefit, since the view only shows the tail.
const MAX_LINES: usize = 500;

/// Where a run is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Running,
    Finished,
}

/// One transcript entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Line {
    /// A session-level notice.
    Notice(String),
    /// The model's text.
    Assistant(String),
    /// A tool call starting.
    ToolStart { name: String },
    /// A tool call finishing.
    ToolEnd {
        name: String,
        success: bool,
        duration_ms: u64,
        preview: String,
    },
    /// Context was compacted.
    Compaction { freed: usize },
}

impl Line {
    /// Render for display, without styling.
    pub fn text(&self) -> String {
        match self {
            Line::Notice(t) => format!("· {t}"),
            Line::Assistant(t) => t.clone(),
            Line::ToolStart { name } => format!("→ {name}"),
            Line::ToolEnd {
                name,
                success,
                duration_ms,
                preview,
            } => {
                let mark = if *success { "✓" } else { "✗" };
                let preview = preview.trim();
                if preview.is_empty() {
                    format!("  {mark} {name} ({duration_ms}ms)")
                } else {
                    format!("  {mark} {name} ({duration_ms}ms) {preview}")
                }
            }
            Line::Compaction { freed } => format!("· context compacted, {freed} tokens freed"),
        }
    }

    /// Whether this line is an error, for styling.
    pub fn is_error(&self) -> bool {
        matches!(self, Line::ToolEnd { success: false, .. })
    }
}

/// TUI state.
pub struct App {
    pub task: String,
    pub model: String,
    pub strategy: String,
    pub window: usize,
    pub lines: VecDeque<Line>,
    pub status: RunStatus,
    pub turns: u32,
    pub tool_calls: u32,
    pub failed_tool_calls: u32,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub outcome: Option<String>,
    /// Whether the transcript is scrolled to the bottom.
    pub follow: bool,
    /// Scroll offset from the top of the retained window.
    pub scroll: usize,
}

impl App {
    pub fn new(
        task: impl Into<String>,
        model: impl Into<String>,
        strategy: String,
        window: usize,
    ) -> Self {
        Self {
            task: task.into(),
            model: model.into(),
            strategy,
            window,
            lines: VecDeque::new(),
            status: RunStatus::Running,
            turns: 0,
            tool_calls: 0,
            failed_tool_calls: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            outcome: None,
            follow: true,
            scroll: 0,
        }
    }

    /// Apply one progress event.
    pub fn apply(&mut self, event: Progress) {
        match event {
            Progress::SessionStarted {
                resumed_turns,
                session_id,
                ..
            } => {
                if resumed_turns > 0 {
                    self.push(Line::Notice(format!(
                        "resumed {session_id} with {resumed_turns} prior turns"
                    )));
                } else {
                    self.push(Line::Notice(format!("session {session_id}")));
                }
            }
            Progress::TurnCompleted {
                turn,
                tool_calls,
                usage,
                text,
            } => {
                self.turns = turn + 1;
                self.prompt_tokens += usage.prompt_tokens;
                self.completion_tokens += usage.completion_tokens;
                if !text.trim().is_empty() {
                    self.push(Line::Assistant(text.trim().to_string()));
                }
                if tool_calls > 0 {
                    self.tool_calls += tool_calls as u32;
                }
            }
            Progress::ToolStarted { name, .. } => {
                self.push(Line::ToolStart { name });
            }
            Progress::ToolFinished {
                name,
                success,
                duration_ms,
                preview,
            } => {
                if !success {
                    self.failed_tool_calls += 1;
                }
                self.push(Line::ToolEnd {
                    name,
                    success,
                    duration_ms,
                    preview,
                });
            }
            Progress::ContextManaged {
                tokens_before,
                tokens_after,
            } => {
                self.push(Line::Compaction {
                    freed: tokens_before.saturating_sub(tokens_after),
                });
            }
            Progress::SessionFinished {
                outcome,
                turns,
                total_tokens,
            } => {
                self.status = RunStatus::Finished;
                self.turns = turns as u32;
                self.prompt_tokens = total_tokens;
                self.outcome = Some(outcome);
            }
        }
    }

    fn push(&mut self, line: Line) {
        if self.lines.len() == MAX_LINES {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    /// Total tokens billed across the run.
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens + self.completion_tokens
    }

    /// Scroll up by `n` lines, disabling follow.
    pub fn scroll_up(&mut self, n: usize) {
        self.follow = false;
        self.scroll = self.scroll.saturating_add(n).min(self.lines.len());
    }

    /// Scroll down; reaching the bottom re-enables follow.
    pub fn scroll_down(&mut self, n: usize) {
        self.scroll = self.scroll.saturating_sub(n);
        if self.scroll == 0 {
            self.follow = true;
        }
    }

    /// Toggle whether the view follows new output.
    pub fn toggle_follow(&mut self) {
        self.follow = !self.follow;
        if self.follow {
            self.scroll = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axolotl_models::types::TokenUsage;

    fn app() -> App {
        App::new("do a thing", "model", "T4".into(), 128_000)
    }

    fn turn(turn: u32, text: &str, tools: usize) -> Progress {
        Progress::TurnCompleted {
            turn,
            tool_calls: tools,
            usage: TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
            },
            text: text.into(),
        }
    }

    #[test]
    fn assistant_text_becomes_a_line() {
        let mut app = app();
        app.apply(turn(0, "working on it", 0));
        assert_eq!(app.lines.len(), 1);
        assert_eq!(app.lines[0], Line::Assistant("working on it".into()));
    }

    #[test]
    fn empty_text_adds_no_line() {
        let mut app = app();
        app.apply(turn(0, "   ", 1));
        assert!(app.lines.is_empty());
    }

    #[test]
    fn token_usage_accumulates() {
        let mut app = app();
        app.apply(turn(0, "a", 0));
        app.apply(turn(1, "b", 0));
        assert_eq!(app.prompt_tokens, 200);
        assert_eq!(app.completion_tokens, 40);
        assert_eq!(app.total_tokens(), 240);
        assert_eq!(app.turns, 2);
    }

    #[test]
    fn tool_events_pair_up() {
        let mut app = app();
        app.apply(Progress::ToolStarted {
            name: "bash".into(),
            arguments: "{}".into(),
        });
        app.apply(Progress::ToolFinished {
            name: "bash".into(),
            success: true,
            duration_ms: 12,
            preview: "ok".into(),
        });
        assert_eq!(app.lines.len(), 2);
        assert_eq!(
            app.lines[0],
            Line::ToolStart {
                name: "bash".into()
            }
        );
        assert!(!app.lines[1].is_error());
    }

    #[test]
    fn failed_tool_calls_are_counted_and_marked() {
        let mut app = app();
        app.apply(Progress::ToolFinished {
            name: "read_file".into(),
            success: false,
            duration_ms: 3,
            preview: "Error: missing".into(),
        });
        assert_eq!(app.failed_tool_calls, 1);
        assert!(app.lines[0].is_error());
        assert!(app.lines[0].text().contains('✗'));
    }

    #[test]
    fn compaction_reports_tokens_freed() {
        let mut app = app();
        app.apply(Progress::ContextManaged {
            tokens_before: 1000,
            tokens_after: 400,
        });
        assert!(app.lines[0].text().contains("600"));
    }

    #[test]
    fn finish_sets_status_and_outcome() {
        let mut app = app();
        app.apply(Progress::SessionFinished {
            outcome: "success".into(),
            turns: 5,
            total_tokens: 900,
        });
        assert_eq!(app.status, RunStatus::Finished);
        assert_eq!(app.outcome.as_deref(), Some("success"));
        assert_eq!(app.turns, 5);
    }

    #[test]
    fn transcript_is_bounded() {
        let mut app = app();
        for i in 0..(MAX_LINES + 50) {
            app.apply(turn(i as u32, &format!("line {i}"), 0));
        }
        assert_eq!(app.lines.len(), MAX_LINES);
        // The oldest lines were dropped, the newest kept.
        assert!(app.lines.back().unwrap().text().contains("line"));
    }

    #[test]
    fn scrolling_disables_follow_and_returning_re_enables_it() {
        let mut app = app();
        for i in 0..20 {
            app.apply(turn(i, &format!("line {i}"), 0));
        }
        assert!(app.follow);
        app.scroll_up(5);
        assert!(!app.follow);
        assert_eq!(app.scroll, 5);
        app.scroll_down(5);
        assert!(app.follow);
        assert_eq!(app.scroll, 0);
    }

    #[test]
    fn scroll_is_clamped_to_the_transcript() {
        let mut app = app();
        app.apply(turn(0, "only line", 0));
        app.scroll_up(100);
        assert_eq!(app.scroll, 1);
    }

    #[test]
    fn toggle_follow_resets_scroll_when_re_enabled() {
        let mut app = app();
        app.apply(turn(0, "x", 0));
        app.scroll_up(1);
        app.toggle_follow();
        assert!(app.follow);
        assert_eq!(app.scroll, 0);
    }
}
