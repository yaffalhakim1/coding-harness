//! Live end-to-end test against a real OpenAI-compatible endpoint.
//!
//! Skipped unless a key is available. Run explicitly with:
//!
//! ```text
//! cargo test -p harness-core --test live_e2e -- --ignored --nocapture
//! ```
//!
//! Credentials resolve in this order:
//! 1. `OPENAI_API_KEY` (OpenAI, or any compatible endpoint via `HARNESS_TEST_API_BASE`)
//! 2. `KENARI_API_KEY` (the Kenari gateway, `https://kenari.id/v1`)
//!
//! Optional env vars:
//! - `HARNESS_TEST_MODEL`: default `deepseek-v4-1-flash` on Kenari,
//!   `gpt-4o-mini` on OpenAI
//! - `HARNESS_TEST_API_BASE`: overrides the endpoint

use std::sync::Arc;

use harness_actions::{DefaultRegistry, ToolContext, Workspace};
use harness_context::{build_strategy, ContextTier, Thresholds, Tokenizer};
use harness_core::agent::{run_agent_loop, AgentDeps};
use harness_core::{SessionConfig, SessionOutcome, Task};
use harness_models::types::ModelConfig;
use harness_models::OpenAIProvider;

const KENARI_BASE: &str = "https://kenari.id/v1";

/// (api_key_env, base_url, default_model) for whichever provider has a key.
fn credentials() -> Option<(&'static str, String, String)> {
    let base_override = std::env::var("HARNESS_TEST_API_BASE").ok();

    for (env_var, default_base, default_model) in [
        ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
        ("KENARI_API_KEY", KENARI_BASE, "deepseek-v4-1-flash"),
    ] {
        let present = std::env::var(env_var).is_ok_and(|k| !k.trim().is_empty());
        if present {
            let base = base_override
                .clone()
                .unwrap_or_else(|| default_base.to_string());
            return Some((env_var, base, default_model.to_string()));
        }
    }
    None
}

fn model_config(env_var: &str, base: &str, default_model: &str) -> ModelConfig {
    ModelConfig {
        provider: "openai".into(),
        model_id: std::env::var("HARNESS_TEST_MODEL").unwrap_or_else(|_| default_model.to_string()),
        api_base: Some(base.to_string()),
        api_key_env: Some(env_var.to_string()),
        temperature: 0.0,
        max_tokens: Some(1024),
        timeout_secs: 120,
        ..ModelConfig::default()
    }
}

/// A tiny file-edit task: create a file and make two checks pass.
fn file_edit_task() -> Task {
    let mut task = Task::from_description(
        "Create a file named `answer.txt` in the workspace containing exactly the text: 42\n\
         Nothing else in the file.",
    );
    if cfg!(windows) {
        task.test_commands
            .push("findstr 42 answer.txt > NUL".into());
        task.test_commands.push("type answer.txt".into());
    } else {
        task.test_commands.push("grep -qx 42 answer.txt".into());
        task.test_commands.push("cat answer.txt".into());
    }
    task
}

#[tokio::test]
#[ignore = "requires an API key and spends real tokens"]
async fn solves_a_file_edit_task_against_the_real_api() {
    let Some((env_var, base, default_model)) = credentials() else {
        eprintln!("no API key found (OPENAI_API_KEY or KENARI_API_KEY); skipping");
        return;
    };
    eprintln!("live test: {base} / {default_model}");

    let dir = tempfile::tempdir().unwrap();
    let ctx = ToolContext::new(Workspace::open(dir.path()).unwrap());
    let event_store = harness_models::EventStore::new();
    let actions =
        DefaultRegistry::with_predefined_tools(ctx.clone()).with_recall(ctx, event_store.clone());
    let tokenizer = Tokenizer::new().unwrap();

    let llm: Arc<dyn harness_models::LLMProvider> =
        Arc::new(OpenAIProvider::with_base_url(&base).expect("provider"));

    let config = SessionConfig {
        context_strategy: ContextTier::T4,
        planning_enabled: true,
        max_turns: 20,
        timeout_secs: 300,
        model: model_config(env_var, &base, &default_model),
        ..SessionConfig::default()
    };

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
        plan_state: None,
        progress: None,
    };

    let session = run_agent_loop(&file_edit_task(), &config, deps)
        .await
        .expect("agent loop");

    let verification = session
        .outcome
        .as_ref()
        .and_then(|o| o.task_result())
        .map(|r| r.test_output.clone())
        .unwrap_or_default();

    assert!(
        session
            .outcome
            .as_ref()
            .is_some_and(SessionOutcome::is_success),
        "outcome: {:?}\nverification:\n{verification}",
        session.outcome
    );

    let content = std::fs::read_to_string(dir.path().join("answer.txt")).unwrap();
    assert_eq!(content.trim(), "42");
    assert!(session.total_usage.total_tokens > 0);

    if let Ok(path) = std::env::var("HARNESS_TEST_TRAJECTORY") {
        harness_core::write_trajectory(&session, &path)
            .await
            .expect("write trajectory");
        eprintln!("trajectory: {path}");
    }

    println!(
        "live run: {} turns, {} tokens, {}s",
        session.turn_count(),
        session.total_usage.total_tokens,
        session.elapsed_secs()
    );
}

#[tokio::test]
#[ignore = "requires an API key"]
async fn provider_health_check_against_the_real_api() {
    use harness_models::LLMProvider;

    let Some((env_var, base, default_model)) = credentials() else {
        eprintln!("no API key found; skipping");
        return;
    };

    let provider = OpenAIProvider::with_base_url(&base).expect("provider");
    let healthy = provider
        .health_check(&model_config(env_var, &base, &default_model))
        .await
        .expect("health check");
    assert!(healthy, "provider reported an unhealthy status");
}
