//! `axolotl sessions`: list saved sessions, and `axolotl resume`: continue one.

use std::path::PathBuf;

use anyhow::Result;
use tracing::info;

use axolotl_core::session_archive::{list_sessions, load_session, write_session};
use axolotl_core::{write_trajectory, SessionConfig};

use crate::config::FileConfig;
use crate::credentials;

/// Arguments for `sessions`.
pub struct SessionsArgs {
    pub dir: PathBuf,
    /// Maximum rows to print.
    pub limit: Option<usize>,
}

/// Arguments for `resume`.
pub struct ResumeArgs {
    /// Session id or path to a snapshot.
    pub session: String,
    pub dir: PathBuf,
    pub trajectory_dir: PathBuf,
    pub config: PathBuf,
    pub workspace: PathBuf,
    /// Override the context tier for the continued run.
    pub context_strategy: Option<crate::TierArg>,
    /// Extra turns granted for this run (defaults to the original config).
    pub max_turns: Option<u32>,
    pub dry_run: bool,
}

pub async fn sessions(args: SessionsArgs) -> Result<i32> {
    let mut list = list_sessions(&args.dir).await?;
    if list.is_empty() {
        println!("no sessions in '{}'", args.dir.display());
        return Ok(0);
    }
    if let Some(limit) = args.limit {
        list.truncate(limit);
    }

    println!(
        "{:<38} {:<10} {:>6} {:>10} {:<8} task",
        "session", "outcome", "turns", "tokens", "resumable"
    );
    for info in &list {
        println!(
            "{:<38} {:<10} {:>6} {:>10} {:<8} {}",
            info.id,
            info.outcome.as_deref().unwrap_or("-"),
            info.turns,
            info.total_tokens,
            if info.resumable { "yes" } else { "no" },
            info.task
        );
    }
    Ok(0)
}

pub async fn resume(args: ResumeArgs) -> Result<i32> {
    let previous = load_session(&args.session, &args.dir).await?;

    let file = FileConfig::load(&args.config)?;
    let mut config: SessionConfig = file.to_session_config()?;
    if let Some(tier) = args.context_strategy {
        config.context_strategy = tier.into();
    }
    if let Some(turns) = args.max_turns {
        config.max_turns = turns;
    }

    if args.dry_run {
        println!("would resume session {}", previous.id);
        println!("  task             {}", previous.task.description);
        println!("  prior turns      {}", previous.turns.len());
        println!("  prior outcome    {:?}", previous.outcome);
        println!("  context_strategy {}", config.context_strategy);
        println!("  max_turns        {}", config.max_turns);
        return Ok(0);
    }

    credentials::check_api_key(&config.model.provider, config.model.api_key_env.as_deref())?;
    let chain = crate::run_cmd::load_model_chain(&args.config)?;
    let tokenizer = axolotl_context::Tokenizer::new()?;

    let session = crate::run_cmd::resume_with_chain(
        &previous,
        &config,
        &chain,
        &args.workspace,
        Some(&tokenizer),
    )
    .await?;

    crate::report::report(&session);
    write_trajectory(&session, &args.trajectory_dir).await?;
    write_session(&session, &args.dir).await?;
    info!(session = %session.id, "Session resumed and saved");

    Ok(
        if session
            .outcome
            .as_ref()
            .is_some_and(axolotl_core::SessionOutcome::is_success)
        {
            0
        } else {
            1
        },
    )
}
