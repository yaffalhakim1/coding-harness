//! `axolotl ablate`: run the paper's component ablations and compare settings.
//!
//! Paper §3.1 defines the comparison families this command reproduces:
//! - context-management strategy (T0–T4) across context-window budgets
//! - planning on vs off
//! - predefined tools vs bash-only
//!
//! Each setting is a full sweep. Results are written per setting and compared in
//! a summary table, which is what the paper's Tables 3 and 4 report.

use std::path::PathBuf;

use anyhow::{Context, Result};
use tracing::info;

use axolotl_bench::{Benchmark, SweepReport, SweepTimer, TaskOutcome, TerminalBench};
use axolotl_context::{ContextTier, Tokenizer};
use axolotl_core::ActionSpace;

use crate::config::FileConfig;
use crate::credentials;

/// One point in the ablation grid.
#[derive(Debug, Clone)]
pub struct Setting {
    pub label: String,
    pub tier: ContextTier,
    pub window: usize,
    pub planning: bool,
    pub action_space: ActionSpace,
}

impl Setting {
    fn new(tier: ContextTier, window: usize, planning: bool, action_space: ActionSpace) -> Self {
        let window_k = window / 1000;
        let label = format!(
            "{tier}/{window_k}k/{}/{}",
            if planning { "plan" } else { "noplan" },
            action_space
        );
        Self {
            label,
            tier,
            window,
            planning,
            action_space,
        }
    }
}

/// Build the default ablation grid for the paper's headline experiment.
///
/// Context tiers T0–T4 at each budget, then two single-component ablations at
/// the largest budget (paper §3.1: "Planning and the action space are evaluated
/// only under T4/128k").
pub fn default_grid(windows: &[usize], baseline_window: usize) -> Vec<Setting> {
    let mut settings = Vec::new();
    for &window in windows {
        for tier in [
            ContextTier::T0,
            ContextTier::T1,
            ContextTier::T2,
            ContextTier::T3,
            ContextTier::T4,
        ] {
            settings.push(Setting::new(tier, window, true, ActionSpace::Predefined));
        }
    }
    // Component ablations at the baseline window.
    settings.push(Setting::new(
        ContextTier::T4,
        baseline_window,
        false,
        ActionSpace::Predefined,
    ));
    settings.push(Setting::new(
        ContextTier::T4,
        baseline_window,
        true,
        ActionSpace::BashOnly,
    ));
    settings
}

/// Arguments for the ablate command.
pub struct AblateArgs {
    pub config: PathBuf,
    pub tasks: PathBuf,
    pub limit: Option<usize>,
    pub work_dir: PathBuf,
    pub trajectory_dir: PathBuf,
    pub results_dir: PathBuf,
    /// Context windows to sweep, in tokens.
    pub windows: Vec<usize>,
    /// Restrict to a single tier (skips the tier sweep).
    pub only_tier: Option<ContextTier>,
    pub dry_run: bool,
}

pub async fn run(args: AblateArgs) -> Result<i32> {
    let file = FileConfig::load(&args.config)?;
    let base = file.to_session_config()?;

    let tasks = TerminalBench::new()
        .load_tasks(&args.tasks.to_string_lossy())
        .await?;
    let mut tasks = tasks;
    if let Some(limit) = args.limit {
        tasks.truncate(limit);
    }
    if tasks.is_empty() {
        anyhow::bail!("no tasks loaded from '{}'", args.tasks.display());
    }

    let baseline_window = args.windows.iter().copied().max().unwrap_or(128_000);
    let grid = if let Some(tier) = args.only_tier {
        args.windows
            .iter()
            .map(|&w| Setting::new(tier, w, base.planning_enabled, base.action_space))
            .collect()
    } else {
        default_grid(&args.windows, baseline_window)
    };

    if args.dry_run {
        println!(
            "ablation grid ({} settings x {} tasks):",
            grid.len(),
            tasks.len()
        );
        for setting in &grid {
            println!("  {}", setting.label);
        }
        return Ok(0);
    }

    credentials::check_api_key(&base.model.provider, base.model.api_key_env.as_deref())?;
    let tokenizer = Tokenizer::new()?;
    let mut reports = Vec::new();

    for setting in &grid {
        let mut config = base.clone();
        config.context_strategy = setting.tier;
        config.context_window = setting.window;
        config.planning_enabled = setting.planning;
        config.action_space = setting.action_space;

        info!(setting = %setting.label, "Running setting");
        let timer = SweepTimer::new();
        let mut results = Vec::with_capacity(tasks.len());

        for task in &tasks {
            let workspace = match axolotl_bench::prepare_workspace(
                &args.work_dir,
                &format!("{}-{}", setting.label.replace(['/', '.'], "-"), task.id),
            )
            .await
            {
                Ok(dir) => dir,
                Err(e) => {
                    results.push(TaskOutcome::skipped(task, e.to_string()));
                    continue;
                }
            };

            let core_task = axolotl_core::Task {
                id: task.id.clone(),
                description: task.description.clone(),
                setup_commands: task.setup_commands.clone(),
                test_commands: task.test_commands.clone(),
                working_dir: None,
                expected_files: task.expected_files.clone(),
            };

            match crate::run_cmd::execute(&config, &core_task, &workspace, Some(&tokenizer)).await {
                Ok(session) => {
                    let outcome = TaskOutcome::from_session(task, &session);
                    let _ = axolotl_core::write_trajectory(&session, &args.trajectory_dir).await;
                    results.push(outcome);
                }
                Err(e) => results.push(TaskOutcome::skipped(task, e.to_string())),
            }
        }

        let report = SweepReport::build(&config, "terminal-bench", results, timer.elapsed_secs());
        let path = report
            .write(&args.results_dir)
            .await
            .with_context(|| format!("writing results for {}", setting.label))?;
        info!(path = %path.display(), "Setting complete");
        reports.push(report);
    }

    print_comparison(&reports);
    let all_passed = reports.iter().all(|r| r.summary.passed == r.summary.tasks);
    Ok(if all_passed { 0 } else { 1 })
}

/// Print the comparison table (mirrors paper Tables 3/4 layout).
fn print_comparison(reports: &[SweepReport]) {
    println!();
    println!(
        "{:<28} {:>6} {:>8} {:>10} {:>9} {:>8}",
        "setting", "pass", "rate", "mean tok", "mean turn", "overflow"
    );
    for report in reports {
        let s = &report.summary;
        println!(
            "{:<28} {:>3}/{:<2} {:>7.1}% {:>10} {:>9} {:>8}",
            format!(
                "{}/{}/{}k/{}",
                s.context_strategy,
                if s.planning_enabled { "plan" } else { "noplan" },
                s.context_window / 1000,
                s.action_space
            ),
            s.passed,
            s.tasks,
            s.success_rate() * 100.0,
            s.mean_tokens(),
            s.mean_turns(),
            s.overflowed
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_grid_covers_all_tiers_at_each_window() {
        let grid = default_grid(&[32_000, 128_000], 128_000);
        // 5 tiers x 2 windows + 2 component ablations.
        assert_eq!(grid.len(), 12);

        let tiers: Vec<ContextTier> = grid.iter().map(|s| s.tier).collect();
        for tier in [
            ContextTier::T0,
            ContextTier::T1,
            ContextTier::T2,
            ContextTier::T3,
            ContextTier::T4,
        ] {
            assert!(tiers.contains(&tier), "missing {tier}");
        }
    }

    #[test]
    fn component_ablations_use_the_baseline_window() {
        let grid = default_grid(&[32_000, 64_000], 64_000);
        let no_plan = grid
            .iter()
            .find(|s| !s.planning && s.tier == ContextTier::T4)
            .expect("a planning-off setting");
        assert_eq!(no_plan.window, 64_000);

        let bash_only = grid
            .iter()
            .find(|s| s.action_space == ActionSpace::BashOnly)
            .expect("a bash-only setting");
        assert_eq!(bash_only.window, 64_000);
        assert_eq!(bash_only.tier, ContextTier::T4);
    }

    #[test]
    fn labels_are_unique_and_readable() {
        let grid = default_grid(&[32_000, 128_000], 128_000);
        let mut labels: Vec<&str> = grid.iter().map(|s| s.label.as_str()).collect();
        labels.sort();
        let count = labels.len();
        labels.dedup();
        assert_eq!(labels.len(), count, "labels must be unique");

        assert!(grid.iter().any(|s| s.label == "T4/128k/plan/predefined"));
        assert!(grid.iter().any(|s| s.label == "T0/32k/plan/predefined"));
    }
}
