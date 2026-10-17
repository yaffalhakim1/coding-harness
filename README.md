# Coding Agent Harness

A modular coding agent harness in Rust, built to test the findings of
["An Empirical Study of Harness Design for Coding Agents"](https://arxiv.org/abs/2609.20804)
(arXiv:2609.20804) by ablation rather than assertion.

## Architecture

```
harness-cli (binary)
  └── harness-core (agent loop + state)
        ├── harness-context (T0–T4 context management)
        ├── harness-planning (plan generation)
        ├── harness-actions (tool registry + bash)
        ├── harness-models (LLM adapters)
        └── harness-bench (benchmarks, sweeps, analysis)
```

| Crate | Responsibility |
|-------|---------------|
| `harness-core` | Agent loop, session state, stuck detection, verification |
| `harness-context` | Context tiers T0–T4, tokenizer, summarizer, event store wiring |
| `harness-planning` | Initial plan generation |
| `harness-actions` | Tools, workspace guard, diagnostics, plan/recall tools |
| `harness-models` | OpenAI-compatible adapters, retry/backoff, mock provider |
| `harness-bench` | Task loaders, batch runner, sweep reports, trajectory analysis |
| `harness-cli` | `run` / `bench` / `ablate` / `analyze` |

## Quick Start

```bash
cargo build
cargo test

# Check that a config can actually run
cargo run -- doctor --config config/default.toml --check-api

# Validate configuration without spending tokens
cargo run -- run --task "explain this repo" --dry-run

# Run one task
cargo run -- run \
  --task "Fix the panic in src/parser.rs on empty input" \
  --config config/default.toml \
  --workspace . \
  --context-strategy T4 --context-window 128000 --planning true

# Run a benchmark sweep
cargo run -- bench --benchmark terminal-bench --tasks bench/tiny-tasks.json --limit 3

# Reproduce the paper's ablations
cargo run -- ablate --tasks bench/tiny-tasks.json \
  --config config/bench-tight.toml --windows 1000

# Inspect one trajectory, including failure-stage attribution
cargo run -- analyze --trajectory trajectories
```

Exit code is `0` only when the task's verifier commands all pass.

## Context Management

| Tier | Strategy | Notes |
|------|----------|-------|
| T0 | No management | Terminates with `context_overflow` when the window is exceeded |
| T1 | Rule-based elision | Replaces bulky middle-region tool output with stubs |
| T2 | Elision + recall | Keeps originals; stub offers `recall_event(id)` |
| T3 | LLM summarization | Folds the oldest middle events into a running summary |
| **T4** | **Elision → Summarization** | **Best accuracy/cost tradeoff** |

Region model (paper Algorithm 1): the **preamble** (system prompt, task, plan) and
a **recent window** of at least two turns are never compacted. Only the **middle**
is elided or summarized. Thresholds come from §3.1: elide at 60% of the window,
summarize at 85%, keep the last 30% verbatim.

## Planning

With planning enabled the harness generates an initial plan, registers
`update_plan`, and re-injects the current plan before every turn (paper Figure
17) so revisions take effect immediately without duplicating history.

## Action Space

`predefined` (read/write/edit/list/glob/grep/bash/web_fetch), `bash-only`, or
`hybrid`. `update_plan` and `recall_event` are auxiliary and stay available in
every space, per paper §2.2.

**Parallel reads.** Adjacent read-only tool calls run concurrently, up to eight
per step (paper §3.1). A mutating call flushes the pending reads first, so a
`read_file` → `write_file` sequence still observes the read — which is what the
read-before-write gate depends on.

## Reliability

**Provider failover.** A config may list fallback models:

```toml
[model]
model_id = "primary"
api_key_env = "PRIMARY_API_KEY"

[[model.fallbacks]]
model_id = "backup"
api_base = "https://other.example/v1"
api_key_env = "BACKUP_API_KEY"
```

Routes are tried in order. Only transport, rate-limit and 5xx errors fail over —
a bad request or a rejected key is returned immediately, because retrying those
against another provider just spends money on a request that is wrong either
way. The answering model is recorded in the response, so a silent failover is
visible in the trajectory.

## Diagnostics

```bash
harness doctor --config config/default.toml --check-api
```

Checks that the config parses, every credential (primary and fallbacks)
resolves, the workspace is writable, the tokenizer loads, and the tool set
resolves for the configured action space. `--check-api` sends one real request.

## Configuration

`config/default.toml` (OpenAI), `config/kenari.toml` (Kenari gateway), and
`config/bench-tight.toml` (tight-budget ablation).

### Credentials

Keys resolve in this order:

1. The environment variable named by `model.api_key_env`
2. `~/.local/share/opencode/auth.json`, when a matching `api` entry exists

`--dry-run` prints which source was used. No key needs to appear in a config file.

### Environment Variables

- `OPENAI_API_KEY`, `KENARI_API_KEY` — provider keys
- `RUST_LOG` — log filter (e.g. `info`, `debug`, `harness_core=trace`)

Cost reporting requires `input_cost_per_mtok` and `output_cost_per_mtok`. Without
both, cost is reported as unknown rather than guessed.

## Trajectory Output

Every session writes `trajectories/<session_id>.json`: config, plan, per-turn
tokens, tool calls and results, peak context, summarization count, outcome, and
cost when prices are configured.

## Validation Status

- 299 offline tests, including a scripted end-to-end agent loop.
- Live end-to-end test against a real model:
  `cargo test -p harness-core --test live_e2e -- --ignored --nocapture`
- Ablations reproduce the paper's four claims in direction — see `task.md` for
  the measured tables and `docs/replication.md` for how to re-run them.

## Development

```bash
cargo test
cargo clippy --workspace --all-targets
cargo fmt
```

## License

MIT
