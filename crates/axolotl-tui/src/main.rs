//! A live terminal view of an agent run.
//!
//! Usage:
//!
//! ```text
//! axolotl-tui --task "fix the parser" --workspace . --config config/kenari.toml
//! ```
//!
//! The run happens on a background task; the terminal loop owns the screen and
//! only reads progress events, so a slow render never stalls the agent.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use axolotl_actions::{DefaultRegistry, ToolContext, Workspace};
use axolotl_context::{build_strategy, ContextTier, Thresholds, Tokenizer};
use axolotl_core::progress::ChannelSink;
use axolotl_core::{AgentDeps, SessionConfig, Task};
use axolotl_tui::{ui, App, RunStatus};

/// Everything the TUI needs to start a run.
struct Args {
    task: String,
    workspace: PathBuf,
    model: String,
    api_base: Option<String>,
    api_key_env: String,
    strategy: ContextTier,
    window: usize,
    max_turns: u32,
    /// Print the transcript instead of taking over the terminal.
    headless: bool,
}

fn parse_args() -> Result<Args> {
    let mut task = None;
    let mut workspace = PathBuf::from(".");
    let mut model = "deepseek-v4-1-flash".to_string();
    let mut api_base = std::env::var("AXOLOTL_API_BASE").ok();
    let mut api_key_env = "KENARI_API_KEY".to_string();
    let mut strategy = ContextTier::T4;
    let mut window = 128_000usize;
    let mut max_turns = 30u32;
    let mut headless = false;

    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--task" | "-t" => task = args.next(),
            "--workspace" | "-w" => {
                workspace = PathBuf::from(args.next().context("--workspace needs a value")?)
            }
            "--model" => model = args.next().context("--model needs a value")?,
            "--api-base" => api_base = args.next(),
            "--api-key-env" => api_key_env = args.next().context("--api-key-env needs a value")?,
            "--strategy" => {
                let value = args.next().context("--strategy needs a value")?;
                strategy = ContextTier::parse(&value)
                    .with_context(|| format!("unknown strategy '{value}'"))?;
            }
            "--window" => {
                window = args
                    .next()
                    .context("--window needs a value")?
                    .parse()
                    .context("--window must be a number")?
            }
            "--max-turns" => {
                max_turns = args
                    .next()
                    .context("--max-turns needs a value")?
                    .parse()
                    .context("--max-turns must be a number")?
            }
            "--headless" => headless = true,
            "--help" | "-h" => {
                println!(
                    "axolotl-tui --task <TEXT> [--workspace DIR] [--model NAME]\n\
                     \x20           [--api-base URL] [--api-key-env VAR]\n\
                     \x20           [--strategy T0|T1|T2|T3|T4] [--window N] [--max-turns N]\n\
                     \x20           [--headless]"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown flag '{other}' (try --help)"),
        }
    }

    Ok(Args {
        task: task.context("--task is required (try --help)")?,
        workspace,
        model,
        api_base,
        api_key_env,
        strategy,
        window,
        max_turns,
        headless,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;

    let (sink, mut events) = ChannelSink::new();

    let mut app = App::new(
        args.task.clone(),
        args.model.clone(),
        args.strategy.to_string(),
        args.window,
    );

    if args.headless {
        return run_headless(args, sink, &mut app, &mut events).await;
    }

    // Spawn the run; the UI loop below is the only thing touching the terminal.
    let handle = tokio::spawn(run_agent(args, sink));

    enable_raw_mode().context("enabling raw mode")?;
    let mut stdout = std::io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    // Restore the terminal even if the loop panics. Without this a crash leaves
    // the shell in raw mode with no cursor, which looks like a hung terminal.
    let _guard = TerminalGuard;

    event_loop(&mut terminal, &mut app, &mut events, handle).await
}

/// Restores the terminal on drop.
struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        disable_raw_mode().ok();
        std::io::stdout().execute(LeaveAlternateScreen).ok();
        std::io::stdout().execute(crossterm::cursor::Show).ok();
    }
}

/// Print the transcript as it happens, without touching the terminal.
///
/// This is what makes the view testable and usable in CI: the same `App` and
/// event stream drive it, so the headless path cannot silently diverge from the
/// interactive one.
async fn run_headless(
    args: Args,
    sink: Arc<ChannelSink>,
    app: &mut App,
    events: &mut tokio::sync::mpsc::UnboundedReceiver<axolotl_core::progress::Progress>,
) -> Result<()> {
    let handle = tokio::spawn(run_agent(args, sink));

    while let Some(event) = events.recv().await {
        app.apply(event);
        // Print only the newest line, so the output reads like a log.
        if let Some(line) = app.lines.back() {
            println!("{}", line.text());
        }
    }

    let session = handle.await.context("agent task panicked")??;
    println!();
    println!("outcome    {}", app.outcome.as_deref().unwrap_or("none"));
    println!("turns      {}", app.turns);
    println!(
        "tools      {}/{} ok",
        app.tool_calls - app.failed_tool_calls,
        app.tool_calls
    );
    println!("tokens     {}", session.total_usage.total_tokens);

    Ok(())
}

/// Render and handle input until the run finishes and the user quits.
async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    events: &mut tokio::sync::mpsc::UnboundedReceiver<axolotl_core::progress::Progress>,
    handle: tokio::task::JoinHandle<Result<axolotl_core::Session>>,
) -> Result<()> {
    loop {
        terminal.draw(|frame| ui::draw(frame, app))?;

        // Drain everything pending, then wait briefly for more.
        while let Ok(event) = events.try_recv() {
            app.apply(event);
        }

        if crossterm::event::poll(Duration::from_millis(120))? {
            if let Event::Key(key) = crossterm::event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Esc => return Ok(()),
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(())
                    }
                    KeyCode::Up | KeyCode::Char('k') => app.scroll_up(1),
                    KeyCode::PageUp => app.scroll_up(10),
                    KeyCode::Down | KeyCode::Char('j') => app.scroll_down(1),
                    KeyCode::PageDown => app.scroll_down(10),
                    KeyCode::Char('f') => app.toggle_follow(),
                    _ => {}
                }
            }
        }

        // Once the run is over, keep the final screen until the user quits.
        if handle.is_finished() && app.status == RunStatus::Finished {
            terminal.draw(|frame| ui::draw(frame, app))?;
            loop {
                if crossterm::event::poll(Duration::from_millis(200))? {
                    if let Event::Key(key) = crossterm::event::read()? {
                        if key.kind == KeyEventKind::Press
                            && matches!(
                                key.code,
                                KeyCode::Char('q') | KeyCode::Esc | KeyCode::Enter
                            )
                        {
                            return Ok(());
                        }
                    }
                }
            }
        }
    }
}

/// Build the harness and run the agent, reporting progress to `sink`.
async fn run_agent(args: Args, sink: Arc<ChannelSink>) -> Result<axolotl_core::Session> {
    std::fs::create_dir_all(&args.workspace)
        .with_context(|| format!("creating '{}'", args.workspace.display()))?;

    let config = SessionConfig {
        context_window: args.window,
        context_strategy: args.strategy,
        planning_enabled: false,
        max_turns: args.max_turns,
        model: axolotl_models::types::ModelConfig {
            provider: "openai".into(),
            model_id: args.model.clone(),
            api_base: args.api_base.clone(),
            api_key_env: Some(args.api_key_env.clone()),
            temperature: 0.0,
            max_tokens: Some(2048),
            timeout_secs: 120,
            max_retries: 2,
            ..Default::default()
        },
        ..Default::default()
    };

    let ws = Workspace::open(&args.workspace)?;
    let ctx = ToolContext::new(ws);
    let actions = DefaultRegistry::with_predefined_tools(ctx);
    let tokenizer = Tokenizer::new()?;

    let provider = match &args.api_base {
        Some(base) => axolotl_models::OpenAIProvider::with_base_url(base.clone())?,
        None => axolotl_models::OpenAIProvider::new()?,
    };
    let llm: Arc<dyn axolotl_models::LLMProvider> = Arc::new(provider);

    let context = build_strategy(
        config.context_strategy,
        Thresholds::default(),
        tokenizer.clone(),
        llm.clone(),
        &config.model,
        axolotl_models::EventStore::new(),
    );

    let deps = AgentDeps {
        llm,
        actions: &actions,
        context: context.as_ref(),
        tokenizer: &tokenizer,
        planner: None,
        plan_state: None,
        progress: Some(sink),
    };

    axolotl_core::agent::run_agent_loop(&Task::from_description(args.task), &config, deps).await
}
