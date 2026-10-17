//! `harness run` and `harness analyze`.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::info;

use harness_actions::{DefaultRegistry, ToolContext, Workspace};
use harness_context::{build_strategy, Thresholds, Tokenizer};
use harness_core::{run_agent_loop, write_trajectory, ActionSpace, AgentDeps, Session, Task};
use harness_models::OpenAIProvider;

use crate::config::FileConfig;
use crate::credentials;
use crate::{AnalyzeArgs, RunArgs};

/// Run one task. Returns the process exit code.
pub async fn run(args: RunArgs) -> Result<i32> {
    let config = load_config(&args)?;

    let task = match (&args.task, &args.task_file) {
        (Some(description), _) => Task::from_description(description.clone()),
        (None, Some(path)) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading task file '{}'", path.display()))?;
            serde_json::from_str(&text)
                .with_context(|| format!("parsing task file '{}'", path.display()))?
        }
        (None, None) => anyhow::bail!("provide --task or --task-file"),
    };

    if args.dry_run {
        let credential = credentials::check_api_key(
            &config.model.provider,
            config.model.api_key_env.as_deref(),
        )?;
        println!("configuration OK");
        println!("  model            {}", config.model.model_id);
        println!("  provider         {}", config.model.provider);
        println!(
            "  api key          {} ({})",
            credential.env_var, credential.source
        );
        println!(
            "  api_base         {}",
            config.model.api_base.as_deref().unwrap_or("(default)")
        );
        println!("  context_window   {}", config.context_window);
        println!("  context_strategy {}", config.context_strategy);
        println!("  planning         {}", config.planning_enabled);
        println!("  action_space     {}", config.action_space);
        println!("  max_turns        {}", config.max_turns);
        println!("  workspace        {}", args.workspace.display());
        return Ok(0);
    }

    credentials::check_api_key(&config.model.provider, config.model.api_key_env.as_deref())?;
    let chain = load_model_chain(&args.config)?;
    let session = execute_with_chain(&config, &chain, &task, &args.workspace, None).await?;
    report(&session);

    let path = write_trajectory(&session, &args.trajectory_dir).await?;
    println!("trajectory: {}", path.display());

    Ok(exit_code_for(&session))
}

/// Load the model preference chain: primary model, then any configured
/// fallbacks.
///
/// Keys are resolved for every entry up front so a fallback is usable the
/// moment the primary fails. A fallback with no available key is dropped with a
/// warning rather than kept as a route that can never succeed.
pub fn load_model_chain(path: &Path) -> Result<Vec<harness_models::types::ModelConfig>> {
    let file = FileConfig::load(path)?;
    let chain = file.model_chain();

    let mut usable = Vec::with_capacity(chain.len());
    for (index, model) in chain.into_iter().enumerate() {
        match credentials::check_api_key(&model.provider, model.api_key_env.as_deref()) {
            Ok(_) => usable.push(model),
            Err(e) if index == 0 => return Err(e),
            Err(e) => {
                tracing::warn!(
                    model = %model.model_id,
                    error = %e,
                    "Dropping fallback model: no usable credential"
                );
            }
        }
    }
    Ok(usable)
}

/// Load the config file and apply CLI overrides.
fn load_config(args: &RunArgs) -> Result<harness_core::SessionConfig> {
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
    if let Some(model) = &args.model {
        config.model.model_id = model.clone();
    }
    Ok(config)
}

/// Build the LLM provider for a config, including any configured fallbacks.
///
/// One model returns the plain OpenAI-compatible adapter; more than one wraps
/// them in a [`harness_models::FailoverProvider`], which only switches routes on
/// transport, rate-limit and 5xx errors.
pub fn build_provider(
    chain: &[harness_models::types::ModelConfig],
) -> Result<Arc<dyn harness_models::LLMProvider>> {
    if chain.is_empty() {
        anyhow::bail!("no model configured");
    }

    let mut routes = Vec::with_capacity(chain.len());
    for model in chain {
        let provider = match &model.api_base {
            Some(base) => {
                OpenAIProvider::with_base_url(base.clone()).context("building provider")?
            }
            None => OpenAIProvider::new().context("building provider")?,
        };
        routes.push(harness_models::Route::new(
            Arc::new(provider),
            model.clone(),
        ));
    }

    if routes.len() == 1 {
        return Ok(routes.remove(0).provider);
    }
    Ok(Arc::new(harness_models::FailoverProvider::new(routes)))
}

/// Execute a task in `workspace`. Shared with the benchmark runner.
pub async fn execute(
    config: &harness_core::SessionConfig,
    task: &Task,
    workspace: &Path,
    tokenizer: Option<&Tokenizer>,
) -> Result<Session> {
    let chain = vec![config.model.clone()];
    execute_with_chain(config, &chain, task, workspace, tokenizer).await
}

/// Execute with an explicit model preference chain (primary, then fallbacks).
pub async fn execute_with_chain(
    config: &harness_core::SessionConfig,
    chain: &[harness_models::types::ModelConfig],
    task: &Task,
    workspace: &Path,
    tokenizer: Option<&Tokenizer>,
) -> Result<Session> {
    std::fs::create_dir_all(workspace)
        .with_context(|| format!("creating workspace '{}'", workspace.display()))?;

    let ws = Workspace::open(workspace)?;
    let ctx = ToolContext::new(ws);

    // When planning is enabled, create a shared PlanState and pass it to the
    // registry so the `update_plan` tool is available. The agent loop will
    // re-inject the current plan before each turn.
    let plan_state = if config.planning_enabled {
        Some(harness_actions::PlanState::new())
    } else {
        None
    };

    // The external store backs `recall_event` (M2). Only T2 and T4 retain
    // elided originals, so only they expose the tool.
    let event_store = harness_models::EventStore::new();
    let keeps_originals = matches!(
        config.context_strategy,
        harness_context::ContextTier::T2 | harness_context::ContextTier::T4
    );

    let mut actions = match config.action_space {
        ActionSpace::BashOnly => {
            DefaultRegistry::bash_only_with_plan(ctx.clone(), plan_state.clone())
        }
        ActionSpace::Predefined | ActionSpace::Hybrid => {
            DefaultRegistry::with_predefined_tools_and_plan(ctx.clone(), plan_state.clone())
        }
    };
    if keeps_originals {
        actions = actions.with_recall(ctx, event_store.clone());
    }

    let tokenizer = match tokenizer {
        Some(t) => t.clone(),
        None => Tokenizer::new()?,
    };

    let llm = build_provider(chain)?;

    let context = build_strategy(
        config.context_strategy,
        Thresholds::from(config.thresholds),
        tokenizer.clone(),
        llm.clone(),
        &config.model,
        event_store.clone(),
    );

    let deps = AgentDeps {
        llm,
        actions: &actions,
        context: context.as_ref(),
        tokenizer: &tokenizer,
        planner: None,
        plan_state,
    };

    run_agent_loop(task, config, deps).await
}

/// Print a human-readable summary of a session.
pub fn report(session: &Session) {
    let outcome = session
        .outcome
        .as_ref()
        .map(harness_core::SessionOutcome::label)
        .unwrap_or("none");

    info!(outcome, "Session finished");
    println!();
    println!("outcome          {outcome}");
    println!("turns            {}", session.turn_count());
    println!(
        "tokens           {} (prompt {}, completion {})",
        session.total_usage.total_tokens,
        session.total_usage.prompt_tokens,
        session.total_usage.completion_tokens
    );
    match session.cost_usd() {
        Some(cost) => println!("cost             ${cost:.4}"),
        None => println!("cost             unknown (no prices in config)"),
    }
    println!("peak context     {}", session.peak_context_tokens);
    println!("summarizations   {}", session.summarize_calls);
    println!("duration         {}s", session.elapsed_secs());

    if let Some(result) = session.outcome.as_ref().and_then(|o| o.task_result()) {
        if !result.test_output.is_empty() {
            println!();
            println!("verification:");
            println!("{}", result.test_output.trim_end());
        }
    }
}

fn exit_code_for(session: &Session) -> i32 {
    if session
        .outcome
        .as_ref()
        .is_some_and(harness_core::SessionOutcome::is_success)
    {
        0
    } else {
        1
    }
}

/// Print a summary of a saved trajectory.
pub fn analyze(args: AnalyzeArgs) -> Result<i32> {
    let path = resolve_trajectory(&args.trajectory, &args.dir)?;
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading trajectory '{}'", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)?;

    println!(
        "session          {}",
        json["session_id"].as_str().unwrap_or("?")
    );
    println!(
        "task             {}",
        json["task_id"].as_str().unwrap_or("?")
    );
    println!(
        "outcome          {}",
        json["outcome"].as_str().unwrap_or("none")
    );
    println!("config           {}", json["config"]);
    println!("turns            {}", json["total_turns"]);
    println!("tokens           {}", json["total_tokens"]);
    println!("peak context     {}", json["peak_context_tokens"]);

    if let Some(turns) = json["turns"].as_array() {
        println!();
        println!(
            "{:<5} {:<8} {:<8} {:<14} tools",
            "turn", "in", "out", "stop"
        );
        for turn in turns {
            let tools: Vec<String> = turn["tool_calls"]
                .as_array()
                .map(|calls| {
                    calls
                        .iter()
                        .filter_map(|c| c["name"].as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            println!(
                "{:<5} {:<8} {:<8} {:<14} {}",
                turn["turn_number"],
                turn["prompt_tokens"],
                turn["output_tokens"],
                turn["stop_reason"].as_str().unwrap_or("?"),
                tools.join(",")
            );
        }
    }

    // Failure-stage attribution from the durable trajectory record.
    if let Some(analysis) = crate::trajectory_analysis::analyse_trajectory(&json) {
        crate::trajectory_analysis::print_analysis(&analysis);
    }
    Ok(0)
}

/// Resolve a trajectory reference: an existing file, the newest JSON in a
/// directory, or a session id inside `dir`.
fn resolve_trajectory(id_or_path: &str, dir: &Path) -> Result<std::path::PathBuf> {
    let direct = Path::new(id_or_path);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }

    // A directory is interpreted as "its newest trajectory", which is what a
    // user gets from passing `--trajectory-dir` by mistake.
    if direct.is_dir() {
        let mut matches: Vec<std::path::PathBuf> = std::fs::read_dir(direct)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        matches.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
        return matches
            .pop()
            .ok_or_else(|| anyhow::anyhow!("no trajectories in '{}'", direct.display()));
    }

    let candidate = dir.join(format!("{id_or_path}.json"));
    if candidate.is_file() {
        return Ok(candidate);
    }
    anyhow::bail!(
        "no trajectory found for '{}' (looked for a file and {})",
        id_or_path,
        candidate.display()
    )
}
