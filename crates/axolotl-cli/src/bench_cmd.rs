//! `axolotl bench`: run a benchmark suite and report aggregate metrics.

use anyhow::Result;
use tracing::{info, warn};

use axolotl_bench::{prepare_workspace, SweepReport, SweepTimer, TaskOutcome};
use axolotl_bench::{Benchmark, SweBench, SweepSummary, TerminalBench};
use axolotl_context::Tokenizer;

use crate::config::FileConfig;
use crate::credentials;
use crate::{BenchArgs, BenchmarkArg};

pub async fn run(args: BenchArgs) -> Result<i32> {
    let file = FileConfig::load(&args.config)?;
    let mut config = file.to_session_config()?;

    if let Some(tier) = args.context_strategy {
        config.context_strategy = tier.into();
    }
    if let Some(window) = args.context_window {
        config.context_window = window;
    }
    if let Some(planning) = args.planning {
        config.planning_enabled = planning;
    }
    if let Some(space) = args.action_space {
        config.action_space = space.into();
    }

    if args.dry_run {
        credentials::check_api_key(&config.model.provider, config.model.api_key_env.as_deref())?;
        println!(
            "configuration OK ({} benchmark)",
            benchmark_name(args.benchmark)
        );
        println!("  context_strategy {}", config.context_strategy);
        println!("  context_window   {}", config.context_window);
        println!("  planning         {}", config.planning_enabled);
        println!("  action_space     {}", config.action_space);
        return Ok(0);
    }

    credentials::check_api_key(&config.model.provider, config.model.api_key_env.as_deref())?;
    axolotl_bench::runner::warn_about_config(&config);

    let tasks = load_tasks(args.benchmark, &args.tasks, args.limit).await?;
    if tasks.is_empty() {
        anyhow::bail!("no tasks loaded from '{}'", args.tasks.display());
    }
    info!(
        count = tasks.len(),
        tier = %config.context_strategy,
        window = config.context_window,
        "Running benchmark"
    );

    let tokenizer = Tokenizer::new()?;
    let timer = SweepTimer::new();
    let mut results = Vec::with_capacity(tasks.len());

    for (index, task) in tasks.iter().enumerate() {
        info!(progress = format!("{}/{}", index + 1, tasks.len()), task = %task.id, "Task");

        let workspace = match prepare_workspace(&args.work_dir, &task.id).await {
            Ok(dir) => dir,
            Err(e) => {
                warn!(task = %task.id, error = %e, "Could not prepare workspace");
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

        let session = match crate::run_cmd::execute(
            &config,
            &core_task,
            &workspace,
            Some(&tokenizer),
        )
        .await
        {
            Ok(session) => session,
            Err(e) => {
                // A harness-level failure must not abort the sweep: record it
                // and continue, or one bad task hides every later result.
                warn!(task = %task.id, error = %e, "Task run failed");
                results.push(TaskOutcome::skipped(task, e.to_string()));
                continue;
            }
        };

        let outcome = TaskOutcome::from_session(task, &session);
        println!(
            "{}  {}  turns={}  tokens={}  peak={}",
            if outcome.passed { "PASS" } else { "FAIL" },
            outcome.task_id,
            outcome.turns,
            outcome.total_tokens,
            outcome.peak_context_tokens
        );

        if let Err(e) = axolotl_core::write_trajectory(&session, &args.trajectory_dir).await {
            warn!(task = %task.id, error = %e, "Could not write trajectory");
        }
        results.push(outcome);
    }

    let report = SweepReport::build(
        &config,
        benchmark_name(args.benchmark),
        results,
        timer.elapsed_secs(),
    );
    print_summary(&report.summary);

    let dir = args.results_dir.clone();
    let json_path = report.write(&dir).await?;
    let csv_path = dir.join(format!(
        "{}-{}-{}k-{}.csv",
        report.summary.benchmark,
        report.summary.context_strategy,
        report.summary.context_window / 1000,
        report.summary.action_space
    ));
    tokio::fs::write(&csv_path, report.to_csv()).await?;
    println!("results          {}", json_path.display());
    println!("csv              {}", csv_path.display());

    // Exit 0 only on a clean sweep, so a CI gate can tell "ran and passed" from
    // "ran and some tasks failed".
    Ok(if report.summary.passed == report.summary.tasks {
        0
    } else {
        1
    })
}

async fn load_tasks(
    which: BenchmarkArg,
    path: &std::path::Path,
    limit: Option<usize>,
) -> Result<Vec<axolotl_bench::BenchmarkTask>> {
    let path = path.to_string_lossy().to_string();
    let mut tasks = match which {
        BenchmarkArg::SweBench => SweBench::new().load_tasks(&path).await?,
        BenchmarkArg::TerminalBench => TerminalBench::new().load_tasks(&path).await?,
    };
    if let Some(limit) = limit {
        tasks.truncate(limit);
    }
    Ok(tasks)
}

fn benchmark_name(which: BenchmarkArg) -> &'static str {
    match which {
        BenchmarkArg::SweBench => "swe-bench",
        BenchmarkArg::TerminalBench => "terminal-bench",
    }
}

fn print_summary(summary: &SweepSummary) {
    println!();
    println!(
        "config           {} / {}k / planning={} / {}",
        summary.context_strategy,
        summary.context_window / 1000,
        summary.planning_enabled,
        summary.action_space
    );
    println!("model            {}", summary.model);
    println!("tasks            {}", summary.tasks);
    println!(
        "passed           {} ({:.1}%)",
        summary.passed,
        summary.success_rate() * 100.0
    );
    if summary.unverified > 0 {
        println!("unverified       {}", summary.unverified);
    }
    if summary.overflowed > 0 {
        println!("overflowed       {}", summary.overflowed);
    }
    if summary.errored > 0 {
        println!("errored          {}", summary.errored);
    }
    println!("mean tokens      {}", summary.mean_tokens());
    println!("mean turns       {}", summary.mean_turns());
    println!("peak context     {}", summary.peak_context_tokens);
    println!("summarizations   {}", summary.summarize_calls);
    match summary.mean_cost() {
        Some(cost) => println!("mean cost        ${cost:.4}"),
        None => println!("mean cost        unknown (no prices configured)"),
    }
    println!("duration         {}s", summary.duration_secs);
}
