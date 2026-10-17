//! Rendering for the TUI.
//!
//! Pure functions over [`App`]: no terminal I/O here, so the layout can be
//! tested against a `TestBackend` without a real terminal.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line as TextLine, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Line, RunStatus};

/// Draw the whole screen.
pub fn draw(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Min(5),    // transcript
            Constraint::Length(1), // footer
        ])
        .split(frame.area());

    draw_header(frame, app, chunks[0]);
    draw_transcript(frame, app, chunks[1]);
    draw_footer(frame, app, chunks[2]);
}

fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let status = match app.status {
        RunStatus::Running => Span::styled(
            " running ",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        ),
        RunStatus::Finished => {
            let ok = app.outcome.as_deref() == Some("success");
            Span::styled(
                format!(" {} ", app.outcome.as_deref().unwrap_or("done")),
                Style::default()
                    .fg(Color::Black)
                    .bg(if ok { Color::Green } else { Color::Red }),
            )
        }
    };

    let meta = format!(
        " {}  ·  {}  ·  {}/{}k  ·  {} turns  ·  {}/{} tools",
        app.model,
        app.strategy,
        app.window / 1000,
        "",
        app.turns,
        app.tool_calls - app.failed_tool_calls,
        app.tool_calls,
    );

    let block = Block::default().borders(Borders::ALL).title(" harness ");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let line = TextLine::from(vec![
        status,
        Span::raw(meta),
        Span::raw("  "),
        Span::styled(
            format!("{} tok", app.total_tokens()),
            Style::default().add_modifier(Modifier::DIM),
        ),
    ]);
    frame.render_widget(Paragraph::new(line), inner);
}

fn draw_transcript(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" transcript ");
    let inner_height = area.height.saturating_sub(2) as usize;

    let all: Vec<TextLine> = app.lines.iter().map(render_line).collect();

    // Show the tail when following, otherwise the window above the scroll offset.
    let end = if app.follow {
        all.len()
    } else {
        all.len().saturating_sub(app.scroll)
    };
    let start = end.saturating_sub(inner_height);
    let visible: Vec<TextLine> = all[start..end].to_vec();

    let paragraph = Paragraph::new(visible)
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

/// Turn a transcript entry into a styled line.
pub fn render_line(line: &Line) -> TextLine<'static> {
    match line {
        Line::Notice(text) => TextLine::from(Span::styled(
            format!("· {text}"),
            Style::default().add_modifier(Modifier::DIM),
        )),
        Line::Assistant(text) => TextLine::from(text.clone()),
        Line::ToolStart { name } => TextLine::from(Span::styled(
            format!("→ {name}"),
            Style::default().fg(Color::Cyan),
        )),
        Line::ToolEnd {
            name,
            success,
            duration_ms,
            preview,
        } => {
            let (mark, colour) = if *success {
                ("✓", Color::Green)
            } else {
                ("✗", Color::Red)
            };
            let preview = preview.trim();
            let mut spans = vec![
                Span::styled(format!("  {mark} {name}"), Style::default().fg(colour)),
                Span::styled(
                    format!(" ({duration_ms}ms)"),
                    Style::default().add_modifier(Modifier::DIM),
                ),
            ];
            if !preview.is_empty() {
                spans.push(Span::raw(format!(" {preview}")));
            }
            TextLine::from(spans)
        }
        Line::Compaction { freed } => TextLine::from(Span::styled(
            format!("· context compacted, {freed} tokens freed"),
            Style::default().fg(Color::Magenta),
        )),
    }
}

fn draw_footer(frame: &mut Frame, app: &App, area: Rect) {
    let hint = if app.status == RunStatus::Finished {
        "q quit  ·  ↑↓ scroll  ·  f follow"
    } else {
        "q quit  ·  ↑↓ scroll  ·  f follow  ·  running…"
    };
    let text = if app.follow {
        hint.to_string()
    } else {
        format!("{hint}  ·  scrolled {} lines", app.scroll)
    };
    frame.render_widget(
        Paragraph::new(Span::styled(
            text,
            Style::default().add_modifier(Modifier::DIM),
        )),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::App;
    use harness_core::progress::Progress;
    use harness_models::types::TokenUsage;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn app_with_events() -> App {
        let mut app = App::new("test task", "gpt-test", "T4".into(), 128_000);
        app.apply(Progress::SessionStarted {
            session_id: "s1".into(),
            task_id: "t1".into(),
            resumed_turns: 0,
        });
        app.apply(Progress::TurnCompleted {
            turn: 0,
            tool_calls: 1,
            usage: TokenUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
            },
            text: "I will read the file".into(),
        });
        app.apply(Progress::ToolStarted {
            name: "read_file".into(),
            arguments: "{}".into(),
        });
        app.apply(Progress::ToolFinished {
            name: "read_file".into(),
            success: true,
            duration_ms: 4,
            preview: "contents".into(),
        });
        app
    }

    /// Render into an 80x24 test terminal and return the text.
    fn render(app: &App) -> String {
        let backend = TestBackend::new(80, 24);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut out = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn renders_without_panicking_on_an_empty_app() {
        let app = App::new("t", "m", "T4".into(), 1000);
        let text = render(&app);
        assert!(text.contains("harness"));
        assert!(text.contains("transcript"));
    }

    #[test]
    fn header_shows_model_and_strategy() {
        let text = render(&app_with_events());
        assert!(text.contains("gpt-test"), "{text}");
        assert!(text.contains("T4"), "{text}");
    }

    #[test]
    fn transcript_shows_assistant_and_tool_lines() {
        let text = render(&app_with_events());
        assert!(text.contains("I will read the file"), "{text}");
        assert!(text.contains("read_file"), "{text}");
    }

    #[test]
    fn a_failed_tool_is_visible() {
        let mut app = app_with_events();
        app.apply(Progress::ToolFinished {
            name: "bash".into(),
            success: false,
            duration_ms: 1,
            preview: "boom".into(),
        });
        let text = render(&app);
        assert!(text.contains("bash"), "{text}");
        assert!(text.contains("boom"), "{text}");
    }

    #[test]
    fn finished_run_shows_its_outcome() {
        let mut app = app_with_events();
        app.apply(Progress::SessionFinished {
            outcome: "success".into(),
            turns: 1,
            total_tokens: 120,
        });
        let text = render(&app);
        assert!(text.contains("success"), "{text}");
    }

    #[test]
    fn render_line_marks_errors() {
        let line = Line::ToolEnd {
            name: "x".into(),
            success: false,
            duration_ms: 1,
            preview: String::new(),
        };
        let rendered = render_line(&line);
        let joined: String = rendered.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(joined.contains('✗'), "{joined}");
    }

    #[test]
    fn scrolling_keeps_rendering() {
        let mut app = app_with_events();
        for i in 0..40 {
            app.apply(Progress::TurnCompleted {
                turn: i,
                tool_calls: 0,
                usage: TokenUsage::default(),
                text: format!("line {i}"),
            });
        }
        app.scroll_up(10);
        let text = render(&app);
        assert!(text.contains("scrolled 10 lines"), "{text}");
    }
}
