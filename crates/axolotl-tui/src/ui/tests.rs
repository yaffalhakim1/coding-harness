//! Layout and formatting tests.

use super::*;
use crate::app::App;
use axolotl_core::progress::Progress;
use axolotl_models::types::TokenUsage;
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
fn renders_an_empty_app() {
    let app = App::new("t", "m", "T4".into(), 1000);
    let text = render(&app);
    assert!(text.contains("axolotl"), "{text}");
    assert!(text.contains("running"), "{text}");
}

#[test]
fn the_header_carries_model_tier_and_window() {
    let text = render(&app_with_events());
    assert!(text.contains("gpt-test"), "{text}");
    assert!(text.contains("T4"), "{text}");
    assert!(text.contains("128k"), "{text}");
}

#[test]
fn the_transcript_shows_prose_and_tools() {
    let text = render(&app_with_events());
    assert!(text.contains("I will read the file"), "{text}");
    assert!(text.contains("read_file"), "{text}");
    assert!(text.contains("4ms"), "{text}");
}

#[test]
fn a_failed_tool_is_marked() {
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
    assert!(text.contains('✗'), "{text}");
}

#[test]
fn the_status_bar_summarises_the_run() {
    let mut app = app_with_events();
    app.apply(Progress::SessionFinished {
        outcome: "success".into(),
        turns: 3,
        total_tokens: 18_432,
    });
    let text = render(&app);
    assert!(text.contains("success"), "{text}");
    assert!(text.contains("3 turns"), "{text}");
    assert!(text.contains("18.5k tokens"), "{text}");
}

#[test]
fn a_rule_separates_the_transcript_from_the_status_bar() {
    let text = render(&app_with_events());
    assert!(text.contains('─'), "expected a horizontal rule: {text}");
}

#[test]
fn no_box_borders_are_drawn() {
    // The redesign's whole point: chrome costs rows and says nothing.
    let text = render(&app_with_events());
    for border in ['┌', '┐', '└', '┘', '├', '┤'] {
        assert!(
            !text.contains(border),
            "found a box border {border}: {text}"
        );
    }
}

#[test]
fn scrolling_shows_the_offset() {
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
    assert!(text.contains("10 lines"), "{text}");
}

#[test]
fn the_task_is_pinned_above_the_transcript() {
    let text = render(&app_with_events());
    assert!(text.contains("test task"), "{text}");

    // Pinned means it survives scrolling: it is not part of the tail.
    let mut app = app_with_events();
    for i in 0..40 {
        app.apply(Progress::TurnCompleted {
            turn: i,
            tool_calls: 0,
            usage: TokenUsage::default(),
            text: format!("line {i}"),
        });
    }
    app.scroll_up(20);
    let scrolled = render(&app);
    assert!(scrolled.contains("test task"), "{scrolled}");
}

#[test]
fn a_tool_start_is_dropped_once_its_end_arrives() {
    // Regression: the transcript showed `read_file …` then `✓ read_file`,
    // doubling every tool call to say the same thing.
    let text = render(&app_with_events());
    assert!(!text.contains("read_file …"), "{text}");
    assert!(text.contains("✓ read_file"), "{text}");
}

#[test]
fn an_in_flight_tool_still_shows() {
    let mut app = app_with_events();
    app.apply(Progress::ToolStarted {
        name: "bash".into(),
        arguments: "{}".into(),
    });
    let text = render(&app);
    assert!(text.contains("bash …"), "{text}");
}

#[test]
fn durations_read_in_human_units() {
    assert_eq!(format_duration(3), "3ms");
    assert_eq!(format_duration(999), "999ms");
    assert_eq!(format_duration(1_204), "1.2s");
    assert_eq!(format_duration(14_000), "14s");
}

#[test]
fn counts_get_thousands_separators() {
    assert_eq!(format_count(0), "0");
    assert_eq!(format_count(999), "999");
    assert_eq!(format_count(1_000), "1,000");
    assert_eq!(format_count(51_000), "51,000");
    assert_eq!(format_count(1_234_567), "1,234,567");
}

#[test]
fn a_narrow_status_bar_drops_groups_instead_of_truncating() {
    let mut app = app_with_events();
    app.apply(Progress::SessionFinished {
        outcome: "success".into(),
        turns: 3,
        total_tokens: 18_432,
    });
    let backend = TestBackend::new(30, 8);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| draw(f, &app)).unwrap();
    let buffer = terminal.backend().buffer();
    let mut line = String::new();
    for x in 0..buffer.area.width {
        line.push_str(buffer[(x, buffer.area.height - 1)].symbol());
    }
    // Whatever survives must be whole words, not a cut-off token count.
    assert!(line.contains("success"), "{line}");
    assert!(!line.trim_end().ends_with('k'), "truncated: {line}");
}

#[test]
fn render_line_marks_errors() {
    let line = Line::ToolEnd {
        name: "x".into(),
        success: false,
        duration_ms: 1,
        preview: String::new(),
    };
    let joined: String = render_line(&line)
        .spans
        .iter()
        .map(|s| s.content.as_ref())
        .collect();
    assert!(joined.contains('✗'), "{joined}");
}

#[test]
fn token_formatting_is_compact() {
    assert_eq!(format_tokens(999), "999 tokens");
    assert_eq!(format_tokens(18_432), "18.4k tokens");
    assert_eq!(format_tokens(2_500_000), "2.5M tokens");
}

#[test]
fn the_gutter_marks_tools_and_prose_differently() {
    let tool = gutter_marker(&Line::ToolEnd {
        name: "x".into(),
        success: true,
        duration_ms: 1,
        preview: String::new(),
    });
    let prose = gutter_marker(&Line::Assistant("hi".into()));
    assert_eq!(tool.style.fg, Some(theme::ACCENT));
    assert_eq!(prose.style.fg, Some(theme::MUTED));
}

#[test]
fn a_long_line_does_not_overflow_the_width() {
    // Wrapping is on, so a long line must not panic or truncate the layout.
    let mut app = app_with_events();
    app.apply(Progress::TurnCompleted {
        turn: 1,
        tool_calls: 0,
        usage: TokenUsage::default(),
        text: "x".repeat(400),
    });
    let text = render(&app);
    assert!(text.contains("axolotl"));
}
