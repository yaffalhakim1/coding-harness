# Axolotl

A lightweight coding agent in Rust. One binary, millisecond startup, a few
megabytes of memory — built to be driven from the terminal or from
[Kerenzikov](https://github.com/yaffalhakim1/Kerenzikov-app) over the Agent
Client Protocol.

```bash
axolotl run --task "the parser panics on empty input — fix it"
```

**New here?** Read [docs/what-it-does.md](docs/what-it-does.md) — plain language,
no Rust, and it says what it *cannot* do as clearly as what it can.

## Why it exists

OpenCode is a Node process tree that can use hundreds of megabytes before doing
anything useful. Axolotl is one Rust binary that starts in milliseconds, so you
can run several at once. It is the agent behind Kerenzikov, a native desktop app
for coding agents.

## Install

```bash
cargo install --path crates/axolotl-cli   # the `axolotl` command
cargo install --path crates/axolotl-acp   # ACP server, for Kerenzikov
cargo install --path crates/axolotl-tui   # live terminal view
```

Then check the setup before spending tokens:

```bash
axolotl doctor --config config/kenari.toml --check-api
```

Details: [docs/install.md](docs/install.md).

## Commands

| Command | What it does |
|---|---|
| `axolotl run` | run one task |
| `axolotl sessions` | list saved sessions |
| `axolotl resume` | continue a saved session |
| `axolotl doctor` | validate config, credentials, workspace, endpoint |
| `axolotl analyze` | failure-stage attribution for one trajectory |
| `axolotl bench` | run a benchmark suite |
| `axolotl ablate` | compare context tiers and component settings |

## What it does

**Reads and edits code** with tools for reading, writing, targeted edits,
listing, filename search, and content search — plus a shell for everything else.

**Verifies its work** by running your test command and reading the exit status.
It cannot claim a test passed unless it ran it.

**Asks before acting.** Writing, editing, or running a command requires approval
in the ACP path. A rejection comes back as a recoverable observation, so the
model tries another approach rather than failing the turn.

**Asks you questions** when genuinely blocked, instead of guessing.

**Keeps a checklist** for multi-step work, and will not call the job finished
with items still open.

**Uses skills** from `.axolotl/skills/<name>/SKILL.md`, loaded on demand.

**Understands slash commands**: `/help`, `/skills`, `/plan`, `/review`, `/test`,
and one per skill.

**Delegates** a bounded investigation to a subagent with its own context window.

**Uses MCP servers** — any server's tools become available as
`<server>__<tool>`.

**Resumes** a saved session where it stopped.

## What is behind it

Context management follows ["An Empirical Study of Harness Design for Coding
Agents"](https://arxiv.org/abs/2609.20804). Five tiers, differing in which of
three mechanisms they use:

| Tier | Elision | Recall | Summarisation |
|---|---|---|---|
| T0 | — | — | — |
| T1 | ✓ | — | — |
| T2 | ✓ | ✓ | — |
| T3 | — | — | ✓ |
| **T4** | ✓ | ✓ | ✓ |

Only the *middle* of the conversation is ever compacted: the system prompt, task
and plan stay verbatim, and so do the last two turns. T4 elides stale tool output
first (cheap) and only summarises if that was not enough — which is why it
reaches the same accuracy as T3 with fewer model calls.

Measured at a 1k window, where the budget binds:

| Setting | Pass | Mean tokens | Overflow |
|---|---|---|---|
| T0 | 0/3 | 2,734 | 3 |
| T1 | 3/3 | 26,406 | 0 |
| T3 | 3/3 | 29,046 | 0 |
| **T4** | **3/3** | **27,064** | 0 |

At 4k and 32k every tier ties — the benefit disappears once the window stops
binding, which is the paper's other half. Full tables and caveats in
[task.md](task.md) and [docs/replication.md](docs/replication.md).

## Documentation

| Document | For |
|---|---|
| [what-it-does.md](docs/what-it-does.md) | non-technical overview, and what it cannot do |
| [install.md](docs/install.md) | install and first run |
| [configuration.md](docs/configuration.md) | every config key and its default |
| [how-it-works.md](docs/how-it-works.md) | the loop, context model, tool layer, failure modes |
| [acp.md](docs/acp.md) | protocol methods, approval flow, gaps |
| [replication.md](docs/replication.md) | reproducing the paper's findings |
| [task.md](task.md) | current status, measured results, known gaps |
| [implementation_plan.md](implementation_plan.md) | original design and phased plan |

## Crates

| Crate | Responsibility |
|-------|---------------|
| `axolotl-core` | Agent loop, session state, stuck detection, verification, subagents |
| `axolotl-context` | Context tiers T0–T4, tokenizer, summarizer, recall store |
| `axolotl-planning` | Initial plan generation |
| `axolotl-actions` | Tools, workspace guard, permissions, skills, slash commands, MCP |
| `axolotl-models` | OpenAI-compatible adapters, retry/backoff, failover, mock |
| `axolotl-bench` | Task loaders, batch runner, sweep reports, trajectory analysis |
| `axolotl-acp` | ACP v1 server over stdio |
| `axolotl-tui` | Live terminal view |
| `axolotl-cli` | The `axolotl` command |

`axolotl-bench` is a leaf: nothing on the run path depends on it, so it is not
compiled into the agent you actually use.

## Known gaps

Stated here rather than buried, because they affect daily use:

- **Unproven on real work.** Every task it has completed is synthetic. It has not
  yet fixed a real bug in a real repository.
- **No permission prompt in the terminal.** Only the ACP path asks. `axolotl run`
  refuses a destructive list and runs everything else immediately.
- **No rewind or branch.** Resume only.
- **ACP sessions are in-memory**, so resume does not survive a process restart.
- **No model catalog** over ACP; the model is set at launch.

## Development

```bash
cargo test
cargo clippy --workspace --all-targets
cargo fmt
```

Live tests need a real model:

```bash
cargo test -p axolotl-core --test live_e2e -- --ignored
```

TUI layout is checked by rendering it into a buffer:

```bash
cargo test -p axolotl-tui --test preview -- --nocapture
```

## License

MIT
