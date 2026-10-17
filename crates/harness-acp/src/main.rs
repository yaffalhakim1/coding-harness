//! `harness-acp`: an ACP agent over stdio.
//!
//! Launch: `harness-acp [--stdio]`
//!
//! Configuration comes from the environment so a client that knows nothing
//! about the harness can launch it:
//!
//! | Variable | Meaning |
//! |---|---|
//! | `HARNESS_MODEL` | model id (default `deepseek-v4-1-flash`) |
//! | `HARNESS_API_BASE` | OpenAI-compatible endpoint |
//! | `HARNESS_API_KEY_ENV` | env var holding the key (default `KENARI_API_KEY`) |
//! | `HARNESS_STRATEGY` | context tier, `T0`–`T4` (default `T4`) |
//! | `HARNESS_CONTEXT_WINDOW` | token budget (default 128000) |
//! | `HARNESS_PLANNING` | `1`/`0` (default `0`) |
//! | `HARNESS_MAX_TURNS` | per-prompt turn cap (default 40) |
//!
//! All logging goes to stderr: stdout carries the JSON-RPC stream and a stray
//! `println!` would corrupt it.

use anyhow::Result;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    // Log to stderr only. This is not a preference: stdout is the protocol.
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap_or_default();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(false)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    for arg in &args {
        match arg.as_str() {
            "--stdio" => {}
            "--help" | "-h" => {
                println!(
                    "harness-acp — an Agent Client Protocol agent\n\n\
                     Usage: harness-acp [--stdio]\n\n\
                     Speaks ACP over stdio. Configure via HARNESS_MODEL,\n\
                     HARNESS_API_BASE, HARNESS_API_KEY_ENV, HARNESS_STRATEGY,\n\
                     HARNESS_CONTEXT_WINDOW, HARNESS_PLANNING, HARNESS_MAX_TURNS."
                );
                return Ok(());
            }
            "--version" | "-V" => {
                println!("harness-acp {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            other => anyhow::bail!("unknown argument '{other}' (try --help)"),
        }
    }

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        protocol = harness_acp::PROTOCOL_VERSION,
        "harness-acp starting"
    );

    let connection = harness_acp::rpc::connect();
    harness_acp::agent::run(connection).await
}
