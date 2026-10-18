# Task: Axolotl — Phases 1–6

Status: **all six phases complete.**
Verified against a real model (Kenari / `deepseek-v4-1-flash`).
Design: `implementation_plan.md`. Behavioural source of truth:
`2609.20804v1.pdf`. Replication guide: `docs/replication.md`.

## Build & quality gates

- [x] `cargo check --all-targets` clean, no warnings
- [x] `cargo clippy --workspace --all-targets` clean
- [x] `cargo fmt --all --check` clean
- [x] `cargo test --workspace` — 299 tests + 2 live (ignored by default)
- [x] No source file > 400 LOC
- [x] No function > 60 LOC
- [x] `tracing` throughout; every shell command bounded by a timeout

## Phase 1 — Foundation

### `axolotl-models`
- [x] OpenAI wire types separate from internal types (stringified tool
      arguments, always-string `content`, `tool_call_id` on tool messages)
- [x] Typed `ProviderError` with retryable classification
- [x] Exponential backoff with full jitter
- [x] Per-request timeout and `max_tokens` from `ModelConfig`
- [x] Malformed tool arguments surfaced, not silently dropped
- [x] `MockProvider` for deterministic downstream tests
- [x] `EventStore` — shared external store for elided content (M2)

### `axolotl-actions`
- [x] Workspace guard: rejects `..`, absolute escapes, symlink escapes
- [x] Read-before-write gate + external-modification detection (SHA-256)
- [x] `read_file` offset/limit/line numbers; binary and size guards
- [x] `write_file` `overwrite` gate; `edit_file` unique-match + `replace_all`
- [x] `list_files`, `glob_files` (globset), `grep_text`
- [x] `bash`: platform shell, timeout, 24k truncation on char boundaries
- [x] Non-zero exit is a typed tool error carrying its output
- [x] Windows `cmd /C` uses `raw_arg` (plain argv mangles quoted arguments)

### `axolotl-core`
- [x] Single planning path through `axolotl-planning`
- [x] Loop ends when the model takes no action
- [x] Guard rails: turn limit, wall-clock timeout, session token budget
- [x] Stuck detection: reminder at 5 identical calls, terminate at 8 failures
- [x] Setup commands run once; failure aborts the session
- [x] Verification by process exit status; unverifiable tasks report `Unverified`
- [x] Trajectory JSON with per-turn tokens, tools, durations, peak context

### `axolotl-cli`
- [x] `run` / `bench` / `ablate` / `analyze` subcommands
- [x] `--dry-run` validates config and credentials without spending tokens
- [x] Credentials resolve from the environment, falling back to OpenCode's
      `auth.json`, with the source reported in `--dry-run`
- [x] CLI overrides for tier, window, planning, action space, model

## Phase 2 — Context management (T0–T4)

- [x] Real tokenizer (`tiktoken` cl100k_base); no `len()/4` estimates
- [x] Region model: preamble / middle / recent, per Algorithm 1
- [x] Preamble (system + task + plan) never elided or summarized
- [x] Recent window budgeted at 0.3 of the window, floored at 2 turns
- [x] Open turn boundary no longer counts as a completed turn
- [x] T0 terminates on overflow instead of silently dropping
- [x] T1 elides middle tool observations; stub text from paper Fig. 19
- [x] T2 retains originals and offers `recall_event` in the stub
- [x] T3 `Summarizer` trait + `LlmSummarizer` using the Fig. 18 prompt verbatim
- [x] T4 stages elision (B1 = 0.6) before summarization (B2 = 0.85)
- [x] Summarization failure leaves history intact and is retried next turn
- [x] Running summary merged each pass, injected after the preamble
- [x] `summarize_calls` counter on both T3 and T4, so their costs compare

## Phase 3 — Planning (persistent scaffold)

- [x] `update_plan` tool registered when planning is enabled
- [x] `PlanState` shared between the tool and the agent loop
- [x] Plan re-injected before each turn (paper Figure 17)
- [x] Tests assert the revised plan reaches the *next* request
- [x] Live E2E: model adopted the planner's plan and revised it mid-run

## Phase 4 — Action space

- [x] `recall_event` (M2) registered for T2/T4 only, reading the shared store
- [x] Elision writes originals into the store the tool reads
- [x] Stub text names a real, resolvable event id
- [x] `web_fetch` with HTML→text reduction, 2MB cap, typed fetch errors
- [x] Post-edit diagnostics: Python syntax check, `rustc` check, `node --check`,
      advisory only, never blocks an edit
- [x] Predefined / bash-only / hybrid spaces; auxiliary tools retained in
      bash-only per paper §2.2
- [x] **Parallel read-only tool calls**, up to 8 per step (paper §3.1), with
      mutations forcing a barrier so read-before-write still holds

## Phase 5 — Benchmark integration

- [x] SWE-Bench loader deriving test commands from the test patch
- [x] Terminal-Bench loader
- [x] Verification via process exit status (was a placeholder)
- [x] Per-task workspace isolation, cleared between runs
- [x] A failing task never aborts the sweep
- [x] `SweepReport` with per-task rows, aggregate metrics, JSON + CSV export
- [x] Unique result filenames per setting (action space included)

## Phase 6 — Analysis & replication

- [x] Failure-stage attribution: file → line → patch → verification (paper §C.1.1)
- [x] Behaviour encoding: localize / reproduce / fix / verify / other (§C.1.2)
- [x] Shell commands classified, including redirect-to-file as an edit
- [x] `axolotl analyze` reports stage, phases, termination for one trajectory
- [x] `axolotl ablate` runs the paper's 12-setting grid and prints a comparison
- [x] `docs/replication.md` documents the procedure and its caveats

## Validation

### Offline
299 tests, including a scripted agent loop covering read-before-write recovery,
stuck termination, overflow, token budget, plan re-injection, M2 recall round
trip, diagnostics, failure-stage attribution, and sweep aggregation.

### Live (real model, real spend)
- [x] `live_e2e`: creates a file, passes two verifier commands.
      4 turns, ~17k tokens, ~15s.
- [x] `axolotl run`: writes a trajectory, reports `unverified` honestly when a
      task has no test commands.
- [x] Persistent planning verified live (initial plan → mid-run revisions).

### Paper replication (3 Terminal-Bench-style tasks, `deepseek-v4-1-flash`)

**1k window — the budget binds:**

| Setting | Pass | Mean tokens | Mean turns | Overflow |
|---|---|---|---|---|
| T0 | **0/3** | 2,734 | 0 | **3** |
| T1 | 3/3 | 26,406 | 5 | 0 |
| T2 | 3/3 | 29,200 | 6 | 0 |
| T3 | 3/3 | 29,046 | 6 | 0 |
| **T4** | **3/3** | **27,064** | 5 | 0 |
| T4, no planning | 3/3 | **18,037** | 4 | 0 |
| T4, bash-only | 3/3 | **18,683** | 6 | 0 |

Summarization calls: **T3 = 15, T4 = 14.** T4 stages elision first, so it
reaches the same accuracy with fewer expensive model calls.

**4k and 32k — nothing binds:**

| Window | T0 | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|
| 4k | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| 32k | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

All tiers tie once the window stops binding, which is the paper's other half:
"its accuracy benefit diminishes as the context window expands."

### What these numbers do and do not show

They reproduce the *direction* of all four paper claims:

1. Context management prevents overflow, most valuable at tight budgets — T0
   fails 3/3 at 1k and passes 3/3 at 4k+; managed tiers never overflow.
2. T4 is the most efficient tier — equal accuracy, fewest summarization calls.
3. Planning costs more for a capable model (81k vs 54k tokens at 1k).
4. bash-only is cheaper than the predefined tool set (56k vs 81k at 1k).

They do **not** establish significance: 3 tasks, one model not in the paper, and
no McNemar test. Directory isolation is also weaker than the paper's containers.
See `docs/replication.md`.

## Beyond the paper (parity with OpenCode / jcode)

Added after comparing feature sets with OpenCode and jcode. These change
reliability and ergonomics, not the measured behaviour of the harness.

- [x] **Parallel read-only tools** — paper §3.1 calls for it; the scaffold ran
      everything sequentially
- [x] **Provider failover** — ordered fallback models; only transport, rate-limit
      and 5xx errors switch routes
- [x] **`axolotl doctor`** — config, credentials (including fallbacks),
      workspace, tokenizer, tool set, optional live API check
- [x] **Credential discovery** — falls back to OpenCode's `auth.json`
- [x] **Session list + resume** — snapshots persist the full conversation;
      resuming replays it into a fresh window so the model continues
- [x] **MCP client** — stdio and HTTP; server tools register as `<server>__<tool>`
- [x] **Subagents** — opt-in `task` tool, two levels deep, children get their own
      context window
- [x] **TUI** — ratatui viewer driven by progress events, plus `--headless`

Still deliberately absent:

- Swarm coordination — subagents cover delegation without the conflict machinery
- Semantic memory — the paper's context management is the subject under test
- LSP integration — post-edit diagnostics cover the immediate need

## ACP server (the Kerenzikov integration)

- [x] `axolotl-acp` binary speaking Agent Client Protocol v1 over stdio
- [x] `initialize` with `loadSession`, `session/resume`, `session/close`,
      `embeddedContext` — every advertised capability has a handler
- [x] `session/new` / `load` / `resume` / `close` / `list` / `prompt` / `cancel`
- [x] Streamed `agent_message_chunk`, `tool_call`, `tool_call_update`,
      `usage_update`, `plan`
- [x] Tool `kind` mapping (read/edit/search/execute/fetch/think)
- [x] Environment-driven config, because the client launches the binary
- [x] Logging to stderr only — stdout is the protocol stream
- [x] 8 stdio integration tests spawning the real binary, plus a live
      `#[ignore]`d prompt turn verified against a real model
- [x] `docs/acp.md` documents the method table and the gaps

Bugs this caught: `mimeType` and `optionId` were not camelCase-renamed (silently
`None`), a bad `cwd` returned `-32603` instead of `-32602`, and `session/close`
reused a params struct that required `cwd`.

## Known gaps

- [ ] Isolation is a per-task directory, not a container; tasks that install
      packages or mutate global state are not safely parallelisable.
- [ ] `write_file` emits LF on Windows, so `findstr /X` cannot verify its output.
      The prompt warns about this; normalizing to CRLF would remove the trap.
- [ ] Failure-stage attribution uses deterministic rules, not the paper's LLM
      judge, so labels will not match theirs exactly.
- [ ] Cost accounting needs prices in config; unknown costs report as `null`
      rather than being guessed.
- [ ] `web_fetch` reduces HTML with a hand-rolled stripper, not a real parser.
- [ ] **No `session/request_permission`.** A client cannot gate a `bash` call,
      which is what makes Supervised mode meaningful. This is the largest gap
      for the Kerenzikov integration and the one worth closing first.
- [ ] ACP sessions are in-memory: `session/resume` works within one process
      lifetime. The CLI already has a snapshot store, so wiring it here is small.
