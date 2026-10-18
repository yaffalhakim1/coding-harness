//! `axolotl-acp`: an ACP agent over stdio.
//!
//! Launch: `axolotl-acp [--stdio]`
//!
//! Configuration comes from the environment so a client that knows nothing
//! about the harness can launch it:
//!
//! | Variable | Meaning |
//! |---|---|
//! | `AXOLOTL_MODEL` | model id (default `deepseek-v4-1-flash`) |
//! | `AXOLOTL_API_BASE` | OpenAI-compatible endpoint |
//! | `AXOLOTL_API_KEY_ENV` | env var holding the key (default `KENARI_API_KEY`) |
//! | `AXOLOTL_STRATEGY` | context tier, `T0`–`T4` (default `T4`) |
//! | `axolotl_context_WINDOW` | token budget (default 128000) |
//! | `axolotl_planning` | `1`/`0` (default `0`) |
//! | `AXOLOTL_MAX_TURNS` | per-prompt turn cap (default 40) |
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
                    "axolotl-acp — an Agent Client Protocol agent\n\n\
                     Usage: axolotl-acp [--stdio]\n\n\
                     Speaks ACP over stdio. Configure via AXOLOTL_MODEL,\n\
                     AXOLOTL_API_BASE, AXOLOTL_API_KEY_ENV, AXOLOTL_STRATEGY,\n\
                     axolotl_context_WINDOW, axolotl_planning, AXOLOTL_MAX_TURNS."
                );
                return Ok(());
            }
            "--version" | "-V" => {
                println!("axolotl-acp {}", env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            other => anyhow::bail!("unknown argument '{other}' (try --help)"),
        }
    }

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        protocol = axolotl_acp::PROTOCOL_VERSION,
        "axolotl-acp starting"
    );

    let connection = axolotl_acp::rpc::connect();
    axolotl_acp::agent::run(connection).await
}
