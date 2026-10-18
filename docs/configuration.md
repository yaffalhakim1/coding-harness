# Configuration

Every key, with its default and what it actually does.

## File layout

```toml
context_window = 128000
context_strategy = "T4"
planning_enabled = true
action_space = "predefined"
max_turns = 50
timeout_secs = 600

[thresholds]
soft_ratio = 0.6
hard_ratio = 0.85
recent_ratio = 0.3
recent_turns = 2

[model]
provider = "openai"
model_id = "gpt-4o"
api_key_env = "OPENAI_API_KEY"
temperature = 0.0
max_tokens = 4096
timeout_secs = 120
max_retries = 4

[[mcp]]
name = "demo-tools"
type = "stdio"
command = "python"
args = ["mcp/demo_server.py"]
```

Shipped configs: `config/default.toml` (OpenAI), `config/kenari.toml` (Kenari
gateway), `config/bench-tight.toml` (tight-budget ablation),
`config/mcp-demo.toml`, `config/subagent-demo.toml`, `config/failover-demo.toml`.

## Session keys

| Key | Default | Meaning |
|---|---|---|
| `context_window` | `128000` | Token budget. The model's real limit, not a target. |
| `context_strategy` | `"T4"` | `T0`–`T4`. See [how-it-works.md](how-it-works.md#context-management). |
| `planning_enabled` | `true` | Generate a plan before acting. Costs one extra call. |
| `action_space` | `"predefined"` | `predefined`, `bash-only`, or `hybrid`. |
| `max_turns` | `50` | Model calls before the run stops. |
| `timeout_secs` | `600` | Wall clock for the whole session. |
| `token_budget` | unset | Hard cap on total tokens. Stops the run when reached. |
| `subagents` | `false` | Expose the `task` tool. Off by default because it changes the action space. |

### When to change what

**`context_window`** — set it to what the model actually has. Setting it lower
than reality is a real experiment: it makes context management matter. Setting
it higher than reality means the API rejects the request.

**`context_strategy`** — leave at `T4` unless you are running a comparison. T0 is
a baseline, not a setting: it terminates on overflow.

**`planning_enabled`** — the paper's finding is conditional. Weak models gain
accuracy; strong models mostly pay extra. If you are unsure, try both on one real
task and compare cost.

**`action_space`** — `bash-only` is cheaper for models good at shell, and worse for
models that are not. `predefined` is the safer default.

**`token_budget`** — worth setting for anything unattended. Without it, a confused
run can spend a lot before the turn limit stops it.

### Thresholds

Only meaningful for T1–T4. They are fractions of `context_window`.

| Key | Default | Meaning |
|---|---|---|
| `soft_ratio` | `0.6` | Start eliding here. |
| `hard_ratio` | `0.85` | Start summarising here. |
| `recent_ratio` | `0.3` | Keep this much verbatim as recent turns. |
| `recent_turns` | `2` | Floor: never protect fewer than this many turns. |

Raising `soft_ratio` delays compaction (more context, higher cost). Lowering
`hard_ratio` makes summarisation fire sooner (cheaper, lossier).

## Model keys

| Key | Default | Meaning |
|---|---|---|
| `provider` | `"openai"` | Label only; the adapter is OpenAI-compatible either way. |
| `model_id` | `"gpt-4o"` | Sent as `model` in the request. |
| `api_base` | unset | Endpoint. Unset means OpenAI. |
| `api_key_env` | `"OPENAI_API_KEY"` | Env var holding the key. |
| `temperature` | `0.0` | 0 is right for coding; sampling hurts reproducibility. |
| `max_tokens` | `4096` | Output cap per call. |
| `timeout_secs` | `120` | Per-request timeout. |
| `max_retries` | `4` | Retries for retryable errors only. |
| `input_cost_per_mtok` | unset | USD per million input tokens. |
| `output_cost_per_mtok` | unset | USD per million output tokens. |

**Cost reporting needs both prices.** With one or neither, cost reports as
unknown rather than guessed — a wrong number in a results table is worse than a
blank one.

### Failover

```toml
[model]
model_id = "primary"
api_key_env = "PRIMARY_KEY"

[[model.fallbacks]]
model_id = "backup"
api_base = "https://other.example/v1"
api_key_env = "BACKUP_KEY"
```

Routes are tried in order. Only transport, rate-limit and 5xx errors fail over; an
auth or bad-request error returns immediately, because it is wrong for every
provider.

The answering model is recorded in the response and in the trajectory, so a
failover is visible rather than silent.

## MCP servers

```toml
[[mcp]]
name = "demo-tools"
type = "stdio"
command = "python"
args = ["mcp/demo_server.py"]
```

```toml
[[mcp]]
name = "remote"
type = "http"
url = "https://example.test/mcp"
```

Tools appear to the model as `<server>__<tool>`, so a server cannot shadow a
built-in. A server that fails to connect is logged and skipped — one broken server
does not stop the run.

## Credentials

Resolution order:

1. the environment variable named by `model.api_key_env`
2. `~/.local/share/opencode/auth.json`, when a matching `api` entry exists

`--dry-run` prints which source was used. No key needs to live in a config file.

## ACP environment

`axolotl-acp` is launched by a client that knows nothing about TOML, so it reads
the environment instead:

| Variable | Default |
|---|---|
| `AXOLOTL_MODEL` | `deepseek-v4-1-flash` |
| `AXOLOTL_API_BASE` | provider default |
| `AXOLOTL_API_KEY_ENV` | `KENARI_API_KEY` |
| `AXOLOTL_STRATEGY` | `T4` |
| `AXOLOTL_CONTEXT_WINDOW` | `128000` |
| `AXOLOTL_PLANNING` | `0` |
| `AXOLOTL_MAX_TURNS` | `40` |

## CLI overrides

`run` accepts overrides that win over the file:

```
--context-strategy T0|T1|T2|T3|T4
--context-window <N>
--planning true|false
--action-space predefined|bash-only|hybrid
--model <ID>
--max-turns <N>
```

## Validating

```bash
axolotl doctor --config config/kenari.toml --check-api
```

Checks the file parses, every credential resolves (primary and fallbacks), the
workspace is writable, the tokenizer loads, and the tool set resolves for the
chosen action space. `--check-api` sends one real request.

`--dry-run` on `run` is the cheaper check: it validates without contacting the
model.
