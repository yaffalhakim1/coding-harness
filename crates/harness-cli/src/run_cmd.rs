//! `harness run` and `harness analyze`.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};

use harness_actions::{DefaultRegistry, ToolContext, Workspace};
use harness_context::{build_strategy, Thresholds, Tokenizer};
use harness_core::{run_agent_loop, write_trajectory, ActionSpace, AgentDeps, Session, Task};
use harness_models::OpenAIProvider;

use crate::config::FileConfig;
use crate::credentials;
use crate::RunArgs;

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
    let mcp_servers = load_mcp_servers(&args.config)?;
    let session =
        execute_with_mcp(&config, &chain, &mcp_servers, &task, &args.workspace, None).await?;
    crate::report::report(&session);

    let path = write_trajectory(&session, &args.trajectory_dir).await?;
    println!("trajectory: {}", path.display());

    // A snapshot makes the run resumable; the trajectory alone is only a
    // summary and cannot be continued.
    if let Ok(snapshot) =
        harness_core::session_archive::write_session(&session, &args.session_dir).await
    {
        println!("session:    {}", snapshot.display());
    }

    Ok(crate::report::exit_code_for(&session))
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
    if let Some(turns) = args.max_turns {
        config.max_turns = turns;
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

/// Load and validate the MCP servers declared in a config file.
pub fn load_mcp_servers(path: &Path) -> Result<Vec<harness_actions::mcp::ServerConfig>> {
    let file = FileConfig::load(path)?;
    file.mcp
        .iter()
        .map(|server| server.to_server_config())
        .collect()
}

/// Execute with an explicit model preference chain (primary, then fallbacks).
pub async fn execute_with_chain(
    config: &harness_core::SessionConfig,
    chain: &[harness_models::types::ModelConfig],
    task: &Task,
    workspace: &Path,
    tokenizer: Option<&Tokenizer>,
) -> Result<Session> {
    execute_with_mcp(config, chain, &[], task, workspace, tokenizer).await
}

/// Execute with a model chain and MCP servers.
pub async fn execute_with_mcp(
    config: &harness_core::SessionConfig,
    chain: &[harness_models::types::ModelConfig],
    mcp_servers: &[harness_actions::mcp::ServerConfig],
    task: &Task,
    workspace: &Path,
    tokenizer: Option<&Tokenizer>,
) -> Result<Session> {
    let prepared = prepare_with_mcp(config, chain, workspace, tokenizer, mcp_servers).await?;
    run_agent_loop(task, config, prepared.deps()).await
}

/// Resume a previous session with an explicit model chain.
pub async fn resume_with_chain(
    previous: &Session,
    config: &harness_core::SessionConfig,
    chain: &[harness_models::types::ModelConfig],
    workspace: &Path,
    tokenizer: Option<&Tokenizer>,
) -> Result<Session> {
    let prepared = prepare(config, chain, workspace, tokenizer).await?;
    harness_core::agent::resume_agent_loop(previous, config, prepared.deps()).await
}

/// A registry handle that is filled in after the registry is built.
///
/// The `task` tool needs to hand its child the same tool set it is part of,
/// which is a cycle: registry → task tool → registry. Resolving it with a
/// A harness assembled for one run.
///
/// Holds the tool registry, context strategy and plan state together so the
/// borrow in [`AgentDeps`] stays valid for the duration of a run.
pub struct Prepared {
    pub actions: Arc<DefaultRegistry>,
    pub context: Box<dyn harness_context::ContextStrategy>,
    pub tokenizer: Tokenizer,
    pub plan_state: Option<harness_actions::PlanState>,
    pub llm: Arc<dyn harness_models::LLMProvider>,
}

impl Prepared {
    /// Borrow into the shape the agent loop expects.
    pub fn deps(&self) -> AgentDeps<'_> {
        AgentDeps {
            llm: self.llm.clone(),
            actions: self.actions.as_ref(),
            context: self.context.as_ref(),
            tokenizer: &self.tokenizer,
            planner: None,
            plan_state: self.plan_state.clone(),
            progress: None,
        }
    }
}

/// Build the harness for a config: tools, context strategy, plan state.
pub async fn prepare(
    config: &harness_core::SessionConfig,
    chain: &[harness_models::types::ModelConfig],
    workspace: &Path,
    tokenizer: Option<&Tokenizer>,
) -> Result<Prepared> {
    prepare_with_mcp(config, chain, workspace, tokenizer, &[]).await
}

/// Build the harness, connecting the given MCP servers and registering their
/// tools alongside the built-in ones.
pub async fn prepare_with_mcp(
    config: &harness_core::SessionConfig,
    chain: &[harness_models::types::ModelConfig],
    workspace: &Path,
    tokenizer: Option<&Tokenizer>,
    mcp_servers: &[harness_actions::mcp::ServerConfig],
) -> Result<Prepared> {
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

    // MCP tools are additive to whatever action space was chosen: a bash-only
    // run may still use an MCP server the user configured explicitly.
    if !mcp_servers.is_empty() {
        let (mcp_actions, failures) = harness_actions::mcp::connect_all(mcp_servers).await;
        for failure in failures {
            tracing::warn!(server = %failure, "MCP server skipped");
        }
        for action in mcp_actions {
            actions.register(std::sync::Arc::new(action));
        }
    }

    let tokenizer = match tokenizer {
        Some(t) => t.clone(),
        None => Tokenizer::new()?,
    };

    let llm = build_provider(chain)?;

    // Subagents are opt-in because they change the action space. The `task`
    // tool needs a handle to the registry it lives in, so it is given a shared
    // cell that is filled after registration.
    let actions: Arc<DefaultRegistry> = if config.subagents {
        let cell: Arc<std::sync::OnceLock<Arc<dyn harness_actions::ActionRegistry>>> =
            Arc::new(std::sync::OnceLock::new());
        let task_tool = harness_core::SubagentTool::new(
            llm.clone(),
            Arc::new(crate::late_registry::LateBoundRegistry(cell.clone())),
            config.clone(),
            0,
        );
        let mut registry = actions;
        registry.register(Arc::new(task_tool));
        let shared = Arc::new(registry);
        let _ = cell.set(shared.clone());
        shared
    } else {
        Arc::new(actions)
    };

    let context = build_strategy(
        config.context_strategy,
        Thresholds::from(config.thresholds),
        tokenizer.clone(),
        llm.clone(),
        &config.model,
        event_store,
    );

    Ok(Prepared {
        actions,
        context,
        tokenizer,
        plan_state,
        llm,
    })
}
