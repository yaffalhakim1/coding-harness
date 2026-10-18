# Axolotl

A lightweight coding agent written in Rust, designed to be driven by
[Kerenzikov](https://github.com/yaffalhakim1/Kerenzikov-app) over the Agent
Client Protocol.

```bash
axolotl run --task "fix the parser" --workspace .
```

## Why

OpenCode is a Node process tree. Axolotl is one Rust binary that carries the
agent loop, the tools, context management and MCP. Kerenzikov launches it the
same way it launches Copilot, Cursor, Grok or Kimi — over ACP — so a session
costs one process, not a runtime.

The design is informed by ["An Empirical Study of Harness Design for Coding
Agents"](https://arxiv.org/abs/2609.20804): context management tiers T0–T4,
persistent planning, and the predefined-vs-bash action space are all
implemented and measurable (`axolotl ablate`).

## Crates

| Crate | Responsibility |
|-------|---------------|
| `axolotl-core` | Agent loop, session state, stuck detection, verification, subagents |
| `axolotl-context` | Context tiers T0–T4, tokenizer, summarizer, recall store |
| `axolotl-planning` | Initial plan generation |
| `axolotl-actions` | Tools, workspace guard, permissions, skills, slash commands, MCP |
| `axolotl-models` | OpenAI-compatible adapters, retry/backoff, failover, mock |
| `axolotl-bench` | Task loaders, batch runner, sweep reports, trajectory analysis |
| `axolotl-acp` | ACP v1 server over stdio — the Kerenzikov integration |
| `axolotl-tui` | Live terminal view |
| `axolotl-cli` | `run` / `bench` / `ablate` / `analyze` / `sessions` / `resume` / `doctor` |

## Install

```bash
cargo install --path crates/axolotl-cli
cargo install --path crates/axolotl-acp
cargo install --path crates/axolotl-tui
```

`axolotl` lands in `~/.cargo/bin`, already on PATH. See `docs/install.md`.

## Quick start

```bash
# Check the configuration before spending tokens
axolotl doctor --config config/kenari.toml --check-api

# Run a task
axolotl run --config config/kenari.toml --task "add a --verbose flag to main"

# Resume the last session
axolotl sessions
axolotl resume --session <id>
```

## Interactive features

### Permission prompts

Every mutating tool asks before it runs. In ACP that is
`session/request_permission`; the client shows the title, the tool kind, and
offers allow-once / reject. A denial comes back to the model as a recoverable
observation, so it can try another approach instead of failing the turn.

A destructive-command list (`rm -rf`, `sudo`, `git push`, `git reset --hard`,
`chmod -R`, …) is refused without waking the user.

### Questions

`ask_user` lets the model ask instead of guessing. Over ACP it routes through
`elicitation/create`. A dismissed question returns "continue with your best
judgement" rather than an error, so the turn survives.

### Todo list

`todo` is a structured checklist, distinct from `update_plan`'s free text. It
is re-injected each turn, and `TodoList::has_unfinished` lets the harness tell
whether work actually finished.

### Slash commands

Advertised to the client as `available_commands_update` after `session/new`.

| Command | Behaviour |
|---|---|
| `/help` | lists commands, no model call |
| `/skills` | lists workspace skills, no model call |
| `/plan [what]` | asks for a step-by-step plan |
| `/review [focus]` | reviews current changes |
| `/test` | runs tests and reports failures |
| `/<skill>` | injects that skill's body |

Anything else is an ordinary prompt. A skill shadows a built-in of the same
name, so a workspace can override `/test` without patching the harness.

### Skills

```
.axolotl/skills/<name>/SKILL.md
.agents/skills/<name>/SKILL.md
```

```markdown
---
name: commit
description: Write a conventional commit message
---

Read the staged diff, then write one commit message...
```

Frontmatter is optional — a body-only file uses its directory name. The body is
injected only when the skill is invoked, so a repo can carry many skills without
paying for them in every prompt.

## Context management

| Tier | Strategy |
|------|----------|
| T0 | No management — terminates on overflow |
| T1 | Rule-based elision |
| T2 | Elision + recall (`recall_event`) |
| T3 | LLM summarization |
| **T4** | **Elision → summarization** (default) |

The preamble (system prompt, task, plan) and the last two turns are never
compacted; only the middle is. Thresholds are the paper's 0.6 / 0.85 soft and
hard budgets with a 0.3 recent window.

## ACP

```bash
AXOLOTL_MODEL=deepseek-v4-1-flash \
AXOLOTL_API_BASE=https://kenari.id/v1 \
AXOLOTL_API_KEY_ENV=KENARI_API_KEY \
axolotl-acp --stdio
```

Implements `initialize`, `session/new|load|resume|close|list|prompt|cancel`,
plus `session/request_permission` and `elicitation/create` calls back to the
client. See `docs/acp.md`.

## Configuration

`config/default.toml` (OpenAI), `config/kenari.toml` (Kenari gateway),
`config/bench-tight.toml` (tight-budget ablation), `config/mcp-demo.toml`,
`config/subagent-demo.toml`.

Credentials resolve from the environment, then from
`~/.local/share/opencode/auth.json`. `--dry-run` reports which source was used.

## Validation

- 496 tests, clippy and fmt clean
- Live end-to-end against a real model: file-edit task, ACP prompt turn,
  permission-gated write
- Ablations reproduce the paper's findings in direction — see `task.md`

## Development

```bash
cargo test
cargo clippy --workspace --all-targets
cargo fmt
```

## License

MIT
