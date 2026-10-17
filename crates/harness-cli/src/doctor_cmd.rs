//! `harness doctor`: check that a configuration can actually run.
//!
//! OpenCode and jcode both ship a diagnostic command. The failure modes worth
//! catching before spending tokens are: a config that does not parse, a
//! credential that is absent, a workspace that is not writable, and a provider
//! that does not answer.

use std::path::PathBuf;
use std::sync::Arc;

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

    // 5. The tool set resolves for the configured action space, including the
    //    opt-in tools (subagents) and any MCP servers.
    if let Some(config) = &session {
        if let Ok(ws) = Workspace::open(&args.workspace) {
            let ctx = ToolContext::new(ws);
            let mut registry = match config.action_space {
                harness_core::ActionSpace::BashOnly => DefaultRegistry::bash_only(ctx.clone()),
                _ => DefaultRegistry::with_predefined_tools(ctx.clone()),
            };
            if config.subagents {
                // `doctor` only reports the surface, so a placeholder registry
                // handle is enough: the tool is never invoked here.
                registry.register(Arc::new(harness_core::SubagentTool::new(
                    harness_models::mock::MockProvider::new(Vec::new()),
                    Arc::new(PlaceholderRegistry),
                    config.clone(),
                    0,
                )));
            }
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

/// A registry that exposes nothing, used by `doctor` when only the tool
/// *surface* matters.
struct PlaceholderRegistry;

#[async_trait::async_trait]
impl ActionRegistry for PlaceholderRegistry {
    fn get_tool_definitions(&self) -> Vec<harness_models::types::ToolDefinition> {
        Vec::new()
    }

    async fn execute(
        &self,
        _tool_call: &harness_models::types::ToolCall,
    ) -> std::result::Result<String, harness_actions::ActionError> {
        Err(harness_actions::ActionError::InvalidArgument(
            "doctor does not execute tools".into(),
        ))
    }

    fn has_tool(&self, _name: &str) -> bool {
        false
    }
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
