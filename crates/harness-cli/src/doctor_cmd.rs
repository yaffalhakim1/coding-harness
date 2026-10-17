//! `harness doctor`: check that a configuration can actually run.
//!
//! OpenCode and jcode both ship a diagnostic command. The failure modes worth
//! catching before spending tokens are: a config that does not parse, a
//! credential that is absent, a workspace that is not writable, and a provider
//! that does not answer.

use std::path::PathBuf;

use anyhow::Result;

use harness_actions::registry::ActionRegistry;
use harness_actions::{DefaultRegistry, ToolContext, Workspace};
use harness_context::Tokenizer;
use harness_models::ModelConfig;

use crate::config::FileConfig;
use crate::credentials;

/// Arguments for `doctor`.
pub struct DoctorArgs {
    pub config: PathBuf,
    pub workspace: PathBuf,
    /// Send one real request to confirm the endpoint answers.
    pub check_api: bool,
}

/// One check result.
struct Check {
    name: &'static str,
    ok: bool,
    detail: String,
}

impl Check {
    fn pass(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            ok: true,
            detail: detail.into(),
        }
    }

    fn fail(name: &'static str, detail: impl Into<String>) -> Self {
        Self {
            name,
            ok: false,
            detail: detail.into(),
        }
    }
}

pub async fn run(args: DoctorArgs) -> Result<i32> {
    let mut checks = Vec::new();

    // 1. Config parses and maps onto a session config.
    let file = match FileConfig::load(&args.config) {
        Ok(file) => {
            checks.push(Check::pass("config", args.config.display().to_string()));
            Some(file)
        }
        Err(e) => {
            checks.push(Check::fail("config", e.to_string()));
            None
        }
    };

    let session = match &file {
        Some(file) => match file.to_session_config() {
            Ok(config) => {
                checks.push(Check::pass(
                    "settings",
                    format!(
                        "{} / {}k / planning={} / {}",
                        config.context_strategy,
                        config.context_window / 1000,
                        config.planning_enabled,
                        config.action_space
                    ),
                ));
                Some(config)
            }
            Err(e) => {
                checks.push(Check::fail("settings", e.to_string()));
                None
            }
        },
        None => None,
    };

    // 2. Credentials for the primary and each fallback.
    let mut chain: Vec<ModelConfig> = Vec::new();
    if let Some(file) = &file {
        for (index, model) in file.model_chain().into_iter().enumerate() {
            let label = if index == 0 { "credential" } else { "fallback" };
            match credentials::check_api_key(&model.provider, model.api_key_env.as_deref()) {
                Ok(cred) => {
                    checks.push(Check::pass(
                        label,
                        format!("{} via {}", model.model_id, cred.source),
                    ));
                    chain.push(model);
                }
                Err(e) => checks.push(Check::fail(label, e.to_string())),
            }
        }
    }

    // 3. Workspace is usable.
    match Workspace::open(&args.workspace) {
        Ok(ws) => {
            let probe = ws.root().join(".harness-doctor");
            let writable = std::fs::write(&probe, b"ok").is_ok();
            let _ = std::fs::remove_file(&probe);
            if writable {
                checks.push(Check::pass("workspace", ws.root().display().to_string()));
            } else {
                checks.push(Check::fail(
                    "workspace",
                    format!("{} is not writable", ws.root().display()),
                ));
            }
        }
        Err(e) => checks.push(Check::fail("workspace", e.to_string())),
    }

    // 4. Tokenizer loads (it is a hard dependency of context management).
    match Tokenizer::new() {
        Ok(tok) => checks.push(Check::pass(
            "tokenizer",
            format!(
                "{} tokens for 'hello world'",
                tok.count_tokens("hello world")
            ),
        )),
        Err(e) => checks.push(Check::fail("tokenizer", e.to_string())),
    }

    // 5. The tool set resolves for the configured action space.
    if let Some(config) = &session {
        if let Ok(ws) = Workspace::open(&args.workspace) {
            let ctx = ToolContext::new(ws);
            let registry = match config.action_space {
                harness_core::ActionSpace::BashOnly => DefaultRegistry::bash_only(ctx),
                _ => DefaultRegistry::with_predefined_tools(ctx),
            };
            let names: Vec<String> = registry
                .get_tool_definitions()
                .into_iter()
                .map(|d| d.name)
                .collect();
            checks.push(Check::pass("tools", names.join(", ")));
        }
    }

    // 6. Optionally confirm the endpoint answers.
    if args.check_api {
        if let Some(config) = &session {
            match crate::run_cmd::build_provider(&chain) {
                Ok(provider) => {
                    let model = chain
                        .first()
                        .cloned()
                        .unwrap_or_else(|| config.model.clone());
                    match provider.health_check(&model).await {
                        Ok(true) => {
                            let route = if chain.len() > 1 {
                                format!(
                                    "{} (or a fallback of {} route(s))",
                                    model.model_id,
                                    chain.len() - 1
                                )
                            } else {
                                model.model_id.clone()
                            };
                            checks.push(Check::pass("api", format!("{route} answered")));
                        }
                        Ok(false) => checks.push(Check::fail(
                            "api",
                            format!(
                                "no route answered (tried {})",
                                chain
                                    .iter()
                                    .map(|m| m.model_id.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ),
                        )),
                        Err(e) => checks.push(Check::fail("api", e.to_string())),
                    }
                }
                Err(e) => checks.push(Check::fail("api", e.to_string())),
            }
        }
    }

    print_checks(&checks);
    let failed = checks.iter().filter(|c| !c.ok).count();
    Ok(if failed == 0 { 0 } else { 1 })
}

fn print_checks(checks: &[Check]) {
    println!();
    for check in checks {
        println!(
            "{}  {:<12} {}",
            if check.ok { "ok  " } else { "FAIL" },
            check.name,
            check.detail
        );
    }
    println!();
    let failed = checks.iter().filter(|c| !c.ok).count();
    if failed == 0 {
        println!("all checks passed");
    } else {
        println!("{failed} check(s) failed");
    }
}
