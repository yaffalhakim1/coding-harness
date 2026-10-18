//! Renders the TUI into a buffer and prints it, so the layout can be eyeballed
//! without a terminal.
//!
//! Run with `cargo test -p axolotl-tui --test preview -- --nocapture`.

use axolotl_core::progress::Progress;
use axolotl_models::types::TokenUsage;
use axolotl_tui::{ui, App};
use ratatui::backend::TestBackend;
use ratatui::Terminal;

fn render(app: &App, width: u16, height: u16) -> String {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ui::draw(f, app)).unwrap();
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

fn populated() -> App {
    let mut app = App::new(
        "Fix the panic in src/parser.rs on empty input",
        "deepseek-v4-1-flash",
        "T4".into(),
        128_000,
    );
    app.apply(Progress::SessionStarted {
        session_id: "sess_9f2a".into(),
        task_id: "t1".into(),
        resumed_turns: 0,
    });
    app.apply(Progress::TurnCompleted {
        turn: 0,
        tool_calls: 2,
        usage: TokenUsage {
            prompt_tokens: 1_200,
            completion_tokens: 140,
            total_tokens: 1_340,
        },
        text: "I'll look at the parser first, then reproduce the panic.".into(),
    });
    app.apply(Progress::ToolStarted {
        name: "read_file".into(),
        arguments: r#"{"path":"src/parser.rs"}"#.into(),
    });
    app.apply(Progress::ToolFinished {
        name: "read_file".into(),
        success: true,
        duration_ms: 3,
        preview: "[src/parser.rs] lines 1-42 of 42".into(),
    });
    app.apply(Progress::ToolStarted {
        name: "bash".into(),
        arguments: r#"{"command":"cargo test parser"}"#.into(),
    });
    app.apply(Progress::ToolFinished {
        name: "bash".into(),
        success: false,
        duration_ms: 1_204,
        preview: "Exit code: 101".into(),
    });
    app.apply(Progress::ContextManaged {
        tokens_before: 92_000,
        tokens_after: 41_000,
    });
    app.apply(Progress::TurnCompleted {
        turn: 1,
        tool_calls: 0,
        usage: TokenUsage {
            prompt_tokens: 3_100,
            completion_tokens: 260,
            total_tokens: 3_360,
        },
        text:
            "The panic is a missing empty-slice check in `parse_tokens`. I'll guard it and re-run."
                .into(),
    });
    app
}

#[test]
fn preview_running() {
    let app = populated();
    println!("\n=== running (100x20) ===");
    print!("{}", render(&app, 100, 20));
}

#[test]
fn preview_finished() {
    let mut app = populated();
    app.apply(Progress::SessionFinished {
        outcome: "success".into(),
        turns: 3,
        total_tokens: 18_432,
    });
    println!("\n=== finished (100x14) ===");
    print!("{}", render(&app, 100, 14));
}

#[test]
fn preview_narrow() {
    let app = populated();
    println!("\n=== narrow (52x16) ===");
    print!("{}", render(&app, 52, 16));
}

#[test]
fn preview_empty() {
    let app = App::new("do a thing", "gpt-5", "T4".into(), 128_000);
    println!("\n=== empty (80x10) ===");
    print!("{}", render(&app, 80, 10));
}
