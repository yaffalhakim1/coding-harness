//! harness-cli: Command-line interface for the coding agent harness.
//!
//! Usage:
//!   harness run --task <task_description> [--config <config.toml>]
//!   harness bench --benchmark <swe-bench|terminal-bench> --tasks <path> [--limit N]
//!   harness analyze --trajectory <session_id>

mod ablate_cmd;
mod bench_cmd;
mod config;
mod credentials;
mod doctor_cmd;
mod run_cmd;
mod trajectory_analysis;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "harness", version, about = "A modular coding agent harness")]
struct Cli {
    /// Logging filter (e.g. `info`, `debug`, `harness_core=trace`).
    #[arg(long, global = true, default_value = "info")]
    log: String,

    /// Emit logs as JSON.
    #[arg(long, global = true)]
    log_json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the agent on a single task.
    Run(RunArgs),
    /// Run a benchmark suite.
    Bench(BenchArgs),
    /// Run the paper's component ablations and compare settings.
    Ablate(AblateArgs),
    /// Check that a configuration can actually run.
    Doctor(DoctorArgs),
    /// Print a summary of a saved trajectory.
    Analyze(AnalyzeArgs),
}

#[derive(clap::Args, Debug)]
pub struct RunArgs {
    /// Task description (natural language).
    #[arg(long, conflicts_with = "task_file")]
    task: Option<String>,

    /// Read the task from a JSON file (`{description, test_commands, ...}`).
    #[arg(long)]
    task_file: Option<PathBuf>,

    /// Config file.
    #[arg(long, default_value = "config/default.toml")]
    config: PathBuf,

    /// Workspace the agent may modify.
    #[arg(long, default_value = ".")]
    workspace: PathBuf,

    /// Where to write the trajectory.
    #[arg(long, default_value = "trajectories")]
    trajectory_dir: PathBuf,

    /// Validate configuration and credentials without calling the model.
    #[arg(long)]
    dry_run: bool,

    #[arg(long, value_enum)]
    context_strategy: Option<TierArg>,

    #[arg(long)]
    context_window: Option<usize>,

    #[arg(long)]
    planning: Option<bool>,

    #[arg(long, value_enum)]
    action_space: Option<ActionSpaceArg>,

    #[arg(long)]
    model: Option<String>,
}

#[derive(clap::Args, Debug)]
pub struct BenchArgs {
    #[arg(long, value_enum)]
    benchmark: BenchmarkArg,

    /// Path to the benchmark task JSON.
    #[arg(long)]
    tasks: PathBuf,

    /// Maximum number of tasks to run.
    #[arg(long)]
    limit: Option<usize>,

    #[arg(long, default_value = "config/default.toml")]
    config: PathBuf,

    /// Where to write per-task workspaces.
    #[arg(long, default_value = "bench-work")]
    work_dir: PathBuf,

    #[arg(long, default_value = "trajectories")]
    trajectory_dir: PathBuf,

    /// Where to write the sweep report (JSON + CSV).
    #[arg(long, default_value = "results")]
    results_dir: PathBuf,

    /// Override the context-management tier (for ablations).
    #[arg(long, value_enum)]
    context_strategy: Option<TierArg>,

    /// Override the context-window budget.
    #[arg(long)]
    context_window: Option<usize>,

    /// Override planning.
    #[arg(long)]
    planning: Option<bool>,

    /// Override the action space.
    #[arg(long, value_enum)]
    action_space: Option<ActionSpaceArg>,

    /// Validate configuration without running tasks.
    #[arg(long)]
    dry_run: bool,
}

#[derive(clap::Args, Debug)]
pub struct AnalyzeArgs {
    /// Session id or path to a trajectory JSON file.
    #[arg(long)]
    trajectory: String,

    #[arg(long, default_value = "trajectories")]
    dir: PathBuf,
}

#[derive(clap::Args, Debug)]
pub struct DoctorArgs {
    #[arg(long, default_value = "config/default.toml")]
    pub config: PathBuf,

    #[arg(long, default_value = ".")]
    pub workspace: PathBuf,

    /// Send one real request to confirm the endpoint answers.
    #[arg(long)]
    pub check_api: bool,
}

#[derive(clap::Args, Debug)]
pub struct AblateArgs {
    #[arg(long, default_value = "config/default.toml")]
    pub config: PathBuf,

    /// Path to the benchmark task JSON.
    #[arg(long)]
    pub tasks: PathBuf,

    /// Maximum number of tasks per setting.
    #[arg(long)]
    pub limit: Option<usize>,

    #[arg(long, default_value = "ablate-work")]
    pub work_dir: PathBuf,

    #[arg(long, default_value = "trajectories")]
    pub trajectory_dir: PathBuf,

    #[arg(long, default_value = "results")]
    pub results_dir: PathBuf,

    /// Context windows to sweep, in tokens (comma separated).
    #[arg(long, value_delimiter = ',', default_value = "32000,128000")]
    pub windows: Vec<usize>,

    /// Run only this tier, across the requested windows.
    #[arg(long, value_enum)]
    pub tier: Option<TierArg>,

    #[arg(long)]
    pub dry_run: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum TierArg {
    #[value(name = "T0")]
    T0,
    #[value(name = "T1")]
    T1,
    #[value(name = "T2")]
    T2,
    #[value(name = "T3")]
    T3,
    #[value(name = "T4")]
    T4,
}

impl From<TierArg> for harness_context::ContextTier {
    fn from(t: TierArg) -> Self {
        match t {
            TierArg::T0 => Self::T0,
            TierArg::T1 => Self::T1,
            TierArg::T2 => Self::T2,
            TierArg::T3 => Self::T3,
            TierArg::T4 => Self::T4,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum ActionSpaceArg {
    #[value(name = "predefined")]
    Predefined,
    #[value(name = "bash-only")]
    BashOnly,
    #[value(name = "hybrid")]
    Hybrid,
}

impl From<ActionSpaceArg> for harness_core::ActionSpace {
    fn from(a: ActionSpaceArg) -> Self {
        match a {
            ActionSpaceArg::Predefined => Self::Predefined,
            ActionSpaceArg::BashOnly => Self::BashOnly,
            ActionSpaceArg::Hybrid => Self::Hybrid,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum BenchmarkArg {
    #[value(name = "swe-bench")]
    SweBench,
    #[value(name = "terminal-bench")]
    TerminalBench,
}

fn init_tracing(level: &str, json: bool) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(level))
        .map_err(|e| anyhow::anyhow!("invalid log filter '{level}': {e}"))?;

    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false);
    if json {
        builder
            .json()
            .try_init()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
    } else {
        builder.try_init().map_err(|e| anyhow::anyhow!("{e}"))?;
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log, cli.log_json)?;

    let exit_code = match cli.command {
        Command::Run(args) => run_cmd::run(args).await?,
        Command::Bench(args) => bench_cmd::run(args).await?,
        Command::Ablate(args) => {
            ablate_cmd::run(ablate_cmd::AblateArgs {
                config: args.config,
                tasks: args.tasks,
                limit: args.limit,
                work_dir: args.work_dir,
                trajectory_dir: args.trajectory_dir,
                results_dir: args.results_dir,
                windows: args.windows,
                only_tier: args.tier.map(Into::into),
                dry_run: args.dry_run,
            })
            .await?
        }
        Command::Analyze(args) => run_cmd::analyze(args)?,
        Command::Doctor(args) => {
            doctor_cmd::run(doctor_cmd::DoctorArgs {
                config: args.config,
                workspace: args.workspace,
                check_api: args.check_api,
            })
            .await?
        }
    };

    std::process::exit(exit_code);
}
