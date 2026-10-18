//! Rendering for the TUI.
//!
//! Layout follows OpenCode's: no boxes. A one-line header, a borderless
//! transcript with a thin gutter, a rule, and a status line. Borders on every
//! panel cost six rows and four columns to say nothing.
//!
//! Pure functions over [`App`], so the layout is testable against a
//! `TestBackend` without a terminal.

use std::collections::VecDeque;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line as TextLine, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Line, RunStatus};
use crate::theme;

/// Width of the gutter to the left of the transcript.
const GUTTER: u16 = 2;

/// Draw the whole screen.
pub fn draw(frame: &mut Frame, app: &App) {
    // The task is pinned above the transcript rather than scrolled with it: a
    // long run should never leave you unsure what it is working on.
    let has_task = !app.task.trim().is_empty();
    let mut constraints = vec![Constraint::Length(1)]; // header
    if has_task {
        constraints.push(Constraint::Length(1)); // task
    }
    constraints.push(Constraint::Min(3)); // transcript
    constraints.push(Constraint::Length(1)); // rule
    constraints.push(Constraint::Length(1)); // status

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(frame.area());

    draw_header(frame, app, chunks[0]);
    let mut next = 1;
    if has_task {
        draw_task(frame, app, chunks[next]);
        next += 1;
    }
    draw_transcript(frame, app, chunks[next]);
    draw_rule(frame, chunks[next + 1]);
    draw_status(frame, app, chunks[next + 2]);
}

/// The task, pinned.
fn draw_task(frame: &mut Frame, app: &App, area: Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(GUTTER), Constraint::Min(10)])
        .split(area);

    frame.render_widget(
        Paragraph::new(TextLine::from(Span::styled("│", theme::muted()))),
        columns[0],
    );
    frame.render_widget(
        Paragraph::new(TextLine::from(Span::styled(
            app.task.clone(),
            theme::strong(),
        ))),
        columns[1],
    );
}

/// `◆ axolotl  model · tier · 128k`
fn draw_header(frame: &mut Frame, app: &App, area: Rect) {
    let marker = Span::styled("◆ ", theme::accent());
    let name = Span::styled("axolotl", theme::heading());
    let separator = Span::styled("  ·  ", theme::muted());

    let line = TextLine::from(vec![
        marker,
        name,
        separator.clone(),
        Span::styled(app.model.clone(), theme::dim()),
        separator.clone(),
        Span::styled(app.strategy.clone(), theme::dim()),
        separator,
        Span::styled(format!("{}k", app.window / 1000), theme::dim()),
    ]);

    frame.render_widget(Paragraph::new(line), area);
}

/// The transcript, with a gutter marker per entry.
fn draw_transcript(frame: &mut Frame, app: &App, area: Rect) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(GUTTER), Constraint::Min(10)])
        .split(area);

    // A tool's start and finish are one logical entry. Showing both doubles the
    // transcript to say the same thing, so a start is dropped once its end has
    // arrived; only an in-flight call keeps a "…" line.
    let lines = visible_lines(&app.lines);

    let height = area.height as usize;
    let rendered: Vec<TextLine> = lines.iter().map(|l| render_line(l)).collect();

    let end = if app.follow {
        rendered.len()
    } else {
        rendered.len().saturating_sub(app.scroll)
    };
    let start = end.saturating_sub(height);

    let gutter: Vec<TextLine> = lines[start..end]
        .iter()
        .map(|line| TextLine::from(gutter_marker(line)))
        .collect();

    frame.render_widget(Paragraph::new(gutter), columns[0]);
    frame.render_widget(
        Paragraph::new(rendered[start..end].to_vec()).wrap(Wrap { trim: false }),
        columns[1],
    );
}

/// Collapse tool starts that are immediately followed by their end.
///
/// The app appends `ToolStarted` then `ToolFinished` for the same call, in
/// order, so an adjacent pair is the signal. A start with no end yet is an
/// in-flight call and is kept.
fn visible_lines(lines: &VecDeque<Line>) -> Vec<&Line> {
    let mut out = Vec::with_capacity(lines.len());
    let mut index = 0;

    while index < lines.len() {
        let line = &lines[index];
        let is_start = matches!(line, Line::ToolStart { .. });
        let next_is_end = matches!(lines.get(index + 1), Some(Line::ToolEnd { .. }));

        if is_start && next_is_end {
            index += 1; // skip the start; the end carries the information
            continue;
        }
        out.push(line);
        index += 1;
    }
    out
}

/// The one-character marker shown beside a transcript entry.
fn gutter_marker(line: &Line) -> Span<'static> {
    match line {
        Line::Assistant(_) => Span::styled("│", theme::muted()),
        Line::ToolStart { .. } | Line::ToolEnd { .. } => Span::styled("│", theme::accent()),
        Line::Notice(_) | Line::Compaction { .. } => Span::styled("│", theme::muted()),
    }
}

/// Turn a transcript entry into a styled line.
pub fn render_line(line: &Line) -> TextLine<'static> {
    match line {
        Line::Notice(text) => TextLine::from(Span::styled(text.clone(), theme::muted())),

        // The model's prose is the only thing at full contrast: everything else
        // is metadata about the run.
        Line::Assistant(text) => TextLine::from(Span::styled(text.clone(), theme::text())),

        Line::ToolStart { name } => TextLine::from(vec![
            Span::styled(name.clone(), theme::accent()),
            Span::styled(" …", theme::muted()),
        ]),

        Line::ToolEnd {
            name,
            success,
            duration_ms,
            preview,
        } => {
            let (mark, style) = if *success {
                ("✓", theme::ok())
            } else {
                ("✗", theme::err())
            };
            let mut spans = vec![
                Span::styled(format!("{mark} "), style),
                Span::styled(name.clone(), theme::dim()),
                Span::styled(
                    format!("  {}", format_duration(*duration_ms)),
                    theme::muted(),
                ),
            ];
            let preview = preview.trim();
            if !preview.is_empty() {
                spans.push(Span::styled("  ", theme::muted()));
                spans.push(Span::styled(preview.to_string(), theme::muted()));
            }
            TextLine::from(spans)
        }

        Line::Compaction { freed } => TextLine::from(Span::styled(
            format!("context compacted · {} tokens freed", format_count(*freed)),
            theme::muted(),
        )),
    }
}

/// `840ms`, `1.2s`, `14s` — never `1204ms`.
fn format_duration(ms: u64) -> String {
    match ms {
        0..=999 => format!("{ms}ms"),
        1_000..=59_999 => {
            let seconds = ms as f64 / 1_000.0;
            // `14.0s` reads as a measurement; `14s` reads as a duration.
            if (seconds - seconds.round()).abs() < 0.05 {
                format!("{}s", seconds.round() as u64)
            } else {
                format!("{seconds:.1}s")
            }
        }
        _ => format!("{}s", ms / 1_000),
    }
}

/// Thousands separators, so `51000` reads as `51,000`.
fn format_count(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn draw_rule(frame: &mut Frame, area: Rect) {
    let width = area.width as usize;
    frame.render_widget(
        Paragraph::new(Span::styled(theme::RULE.repeat(width), theme::muted())),
        area,
    );
}

/// `running  ·  3 turns  ·  12 tools  ·  18.4k tokens`
///
/// Degrades by dropping the least important group first, so a narrow terminal
/// shows fewer facts rather than a truncated one.
fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let width = area.width as usize;
    let dot = "  ·  ";

    let mut groups: Vec<Span<'static>> = vec![status_span(app)];

    if app.turns > 0 {
        groups.push(Span::styled(format!("{} turns", app.turns), theme::dim()));
    }
    if app.tool_calls > 0 {
        let failed = app.failed_tool_calls;
        let (text, style) = if failed == 0 {
            (format!("{} tools", app.tool_calls), theme::dim())
        } else {
            (
                format!("{} tools, {} failed", app.tool_calls, failed),
                theme::err(),
            )
        };
        groups.push(Span::styled(text, style));
    }
    if app.total_tokens() > 0 {
        groups.push(Span::styled(
            format_tokens(app.total_tokens()),
            theme::dim(),
        ));
    }
    if !app.follow {
        groups.push(Span::styled(
            format!("↑ {} lines", app.scroll),
            theme::accent(),
        ));
    }

    // Drop groups from the end until the bar fits.
    let measure = |spans: &[Span<'static>]| -> usize {
        spans
            .iter()
            .map(|s| s.content.chars().count())
            .sum::<usize>()
            + dot.len() * spans.len().saturating_sub(1)
    };
    while groups.len() > 1 && measure(&groups) > width {
        groups.pop();
    }

    let mut spans = Vec::with_capacity(groups.len() * 2);
    for (index, group) in groups.into_iter().enumerate() {
        if index > 0 {
            spans.push(Span::styled(dot, theme::muted()));
        }
        spans.push(group);
    }

    frame.render_widget(Paragraph::new(TextLine::from(spans)), area);
}

fn status_span(app: &App) -> Span<'static> {
    match app.status {
        RunStatus::Running => Span::styled("running", theme::accent()),
        RunStatus::Finished => {
            let outcome = app.outcome.as_deref().unwrap_or("done");
            let style = if outcome == "success" {
                theme::ok()
            } else {
                theme::err()
            };
            Span::styled(outcome.to_string(), style.add_modifier(Modifier::BOLD))
        }
    }
}

/// `18.4k` rather than `18432`.
fn format_tokens(tokens: usize) -> String {
    if tokens < 1_000 {
        format!("{tokens} tokens")
    } else if tokens < 1_000_000 {
        format!("{:.1}k tokens", tokens as f64 / 1_000.0)
    } else {
        format!("{:.1}M tokens", tokens as f64 / 1_000_000.0)
    }
}

/// The key hints, shown by the binary when a run is interactive.
pub fn key_hints() -> &'static str {
    "q quit  ↑↓ scroll  f follow"
}

/// A one-line banner printed before the alternate screen is entered, so the
/// hints survive a terminal that swallows the status line.
pub fn banner(app: &App) -> String {
    format!(
        "axolotl · {} · {} · {}k\n{}\n",
        app.model,
        app.strategy,
        app.window / 1000,
        key_hints()
    )
}


#[cfg(test)]
mod tests;
