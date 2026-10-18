# Implementation Plan: Axolotl in Rust

**Based on:** "An Empirical Study of Harness Design for Coding Agents" (arXiv:2609.20804)
**Target:** Modular, benchmark-aware Axolotl
**Language:** Rust (2021 edition, MSRV 1.75)

> **Phase 1 status:** see `task.md` for the live checklist. Section 0 below
> records the scaffolding review that drove the Phase 1 work.

---

## 0. Scaffolding review (Phase 1 entry)

The skeleton compiles up to `axolotl-actions` but diverges from both this plan
and the paper. Ordered by cost of leaving it alone:

### 0.1 Build
`axolotl-actions/src/{registry,bash,file_ops,search}.rs` import
`axolotl_models::types::*` while `crates/axolotl-actions/Cargo.toml` never
declares the dependency. Four `E0433`s.

### 0.2 Cost bugs (spend real money on every turn)
- `axolotl-core/src/agent.rs::generate_plan` is dead code duplicated from
  `axolotl-planning`; `axolotl-core` lists `axolotl-planning` in `Cargo.toml`
  but never uses it. Two planners, one dead.
- The planner path passes an empty tool slice to `chat.completions`, which
  several OpenAI-compatible servers reject with 400.

### 0.3 Correctness bugs (the benchmark would measure nothing)
- `evaluate_task` ignores `setup_commands`, and marks a task passed for every
  command that returned `Ok(..)` — but `BashExecutor` returns `Ok` for exit
  code 1. Every task "passes". The paper's headline metric is task success.
- `LLMResponse::is_terminal()` is `tool_calls.is_none()`, so any narration turn
  ends the run. The loop needs a bounded retry instead.
- `T0Strategy` deletes the oldest messages on overflow. Paper §2.3: "Tier 0
  disables context management; trajectories that outgrow the window terminate
  with an error." The baseline is the denominator of every context finding;
  getting it wrong inverts the paper.
- `axolotl-context` never protects the preamble and never budgets the recent
  window (Algorithm 1 line 5). It elides the system prompt as readily as a
  stale `ls`.

### 0.4 Panics and unsafe paths
- `registry.rs`: `std::env::current_dir().unwrap()`.
- `bash.rs`: `&result[..50_000]` panics on any multi-byte character; hardcodes
  `bash -c`, absent on this Windows host.
- `search.rs`: `Regex::new(..).unwrap()` in the include filter; the glob→regex
  rewrite escapes `.` before rewriting `*`, and passes `[`/`]` through.
- No workspace guard (§2.4: "resolves each path and rejects any that escapes the
  project root, including through symlinks").

### 0.5 Paper components that are absent
`read-before-write`, stuck detection, `update_plan`, `recall_event`, trajectory
output, and the CLI are named in the docs but not implemented.

### 0.6 Design decisions taken to fix them
1. **Wire types are separate from in-memory types.** The OpenAI wire format
   stores `tool_calls` as `[{id, type, function:{name, arguments: String}}]`
   and the `tool` role needs `tool_call_id`. Serializing the internal type
   directly emits an object where a JSON string is required.
2. **Terminal detection lives in the loop.** The run ends after N consecutive
   content-only turns (default 3), not on the first narration.
3. **Elision never touches the preamble or the recent window.** `ContextState`
   tracks `preamble_len`; entries in the protected region are never candidates.
4. **T0 terminates on overflow** instead of compacting, per §2.3.
5. **Summarization is a trait** (`Summarizer`) so T3/T4 are unit-testable with a
   deterministic fake and injected with the LLM-backed impl in production.
6. **Stuck detection belongs to the loop**, since §2.4 makes it a fixed
   component. Thresholds from §3.1: remind at 5 identical calls, terminate at 8
   identical failing calls.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    axolotl-cli (binary)                      │
│  CLI entrypoint: run tasks, benchmarks, analyze trajectories │
└──────────────┬──────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   axolotl-core      │  Agent loop, state machine, traits
    │  (orchestrator)     │
    └────┬─────┬─────┬────┘
         │     │     │
   ┌─────▼┐ ┌─▼───┐ ┌▼──────────┐
   │context│ │plan │ │ actions   │
   │       │ │ning│ │(tool reg.)│
   └───┬───┘ └──┬──┘ └────┬─────┘
       │        │          │
   ┌───▼────────▼──────────▼───┐
   │      axolotl-models       │  LLM provider adapters
   └─────────────┬─────────────┘
                 │
   ┌─────────────▼─────────────┐
   │     axolotl-bench         │  SWE-Bench / Terminal-Bench runners
   └───────────────────────────┘
```

### Crate Responsibilities

| Crate | Responsibility | Key Traits |
|-------|---------------|------------|
| `axolotl-core` | Execution loop, agent state, message passing | `Agent`, `Session`, `Turn` |
| `axolotl-context` | Context window management, token counting, elision, summarization | `ContextStrategy` (T0–T4), `Summarizer` |
| `axolotl-planning` | Plan generation, step decomposition | `PlanningStrategy` |
| `axolotl-actions` | Tool registry, bash executor, predefined tools | `Action`, `ActionRegistry`, `Executor` |
| `axolotl-models` | LLM API adapters (OpenAI, Anthropic, local) | `LLMProvider`, `ChatMessage` |
| `axolotl-bench` | Benchmark task loading, evaluation, scoring | `Benchmark`, `Task`, `Evaluator` |
| `axolotl-cli` | Binary entrypoint, config loading, trajectory export | — |

---

## 2. Key Design Decisions (from the paper)

### 2.1 Context Management (highest leverage)

The paper evaluates 5 tiers (T0–T4). We implement all 5:

| Tier | Strategy | Description |
|------|----------|-------------|
| **T0** | No management | Raw append, terminate on overflow (baseline) |
| **T1** | Elision only | Rule-based truncation of old tool outputs |
| **T2** | Elision + Recall | Elision + recoverable storage (rarely used per paper) |
| **T3** | Summarization only | LLM summarizes old context |
| **T4** | Elision → Summarization | Rule-based elision first, then LLM summarize remaining (BEST) |

**Implementation priority:** T4 first, then T0 as baseline. T1–T3 as variants.

Thresholds from §3.1: soft `B1 = 0.6` and hard `B2 = 0.85` of the usable window;
the verbatim recent window is budgeted at `0.3` of the window, floored at two
turns; tool results are truncated to 24k characters.

### 2.2 Planning

- **For weaker models:** Planning increases accuracy (keeps trajectory alive)
- **For stronger models:** Planning reduces cost (trims redundant verification)
- **Implementation:** Optional planning step before action selection. Configurable on/off.

### 2.3 Action Space

- **Predefined tools:** `read_file`, `write_file`, `edit_file`, `search`, `edit_file` — helps weaker models
- **Bash-only:** Raw shell execution — lower cost for bash-capable models
- **Implementation:** Trait-based registry, swappable via config.

---

## 3. Data Model

### 3.1 Core Types

```rust
// A single turn in the agent loop
struct Turn {
    id: Uuid,
    turn_number: u32,
    input: ChatMessage,       // What we send to the LLM
    response: LLMResponse,    // What the LLM returned
    tool_calls: Vec<ToolCall>,
    tool_results: Vec<ToolResult>,
    token_usage: TokenUsage,
    duration_ms: u64,
}

// Agent session (one task execution)
struct Session {
    id: Uuid,
    task: Task,
    config: SessionConfig,
    turns: Vec<Turn>,
    context: ContextState,
    plan: Option<Plan>,
    outcome: SessionOutcome,
}

// Context state tracks what's in the window
struct ContextState {
    messages: Vec<ContextEntry>,
    token_count: usize,
    elided_regions: Vec<ElidedRegion>,  // For T2 recall
    summaries: Vec<Summary>,
}
```

### 3.2 Configuration

```rust
struct SessionConfig {
    context_window: usize,          // 32k, 64k, 96k, 128k
    context_strategy: ContextTier,  // T0, T1, T2, T3, T4
    planning_enabled: bool,
    action_space: ActionSpace,      // Predefined, BashOnly, Hybrid
    model: ModelConfig,
    max_turns: u32,
    timeout_secs: u64,
}
```

---

## 4. Implementation Phases

### Phase 1: Foundation (Week 1–2)
**Goal:** Core types, LLM adapter, basic agent loop

- [ ] Define core types: `ChatMessage`, `Turn`, `Session`, `TokenUsage`
- [ ] Implement `axolotl-models`: OpenAI-compatible adapter (covers OpenAI, Anthropic via proxy, local models)
- [ ] Implement `axolotl-core`: basic agent loop (send → receive → execute tool → repeat)
- [ ] Implement `axolotl-actions`: bash executor + file read/write tools
- [ ] Implement `axolotl-cli`: run a single task from CLI
- [ ] End-to-end test: solve a simple file-edit task

### Phase 2: Context Management (Week 3–4)
**Goal:** All 5 context tiers working

- [ ] Token counting via `tiktoken-rs`
- [ ] T0: No management (baseline)
- [ ] T1: Rule-based elision (truncate old tool outputs beyond threshold)
- [ ] T2: Elision + recall storage
- [ ] T3: LLM summarization of old context
- [ ] T4: Elision → Summarization pipeline (the winner)
- [ ] Context overflow detection and graceful handling
- [ ] Unit tests for each strategy

### Phase 3: Planning (Week 5)
**Goal:** Planning module with on/off toggle

- [ ] `PlanningStrategy` trait
- [ ] Default planner: ask LLM to produce a step-by-step plan before acting
- [ ] Plan tracking: mark steps as completed during execution
- [ ] Integration with context (plan stays in context, steps get checked off)
- [ ] Ablation test: same task with planning on vs off

### Phase 4: Action Space (Week 6)
**Goal:** Predefined tools vs bash-only, configurable

- [ ] Predefined tools: `read_file`, `write_file`, `edit_file`, `search_files`, `list_dir`, `run_tests`
- [ ] Bash-only mode: single `bash` tool that accepts any command
- [ ] Hybrid mode: both available
- [ ] Tool call parsing and validation
- [ ] Sandboxed bash execution (timeout, working directory, env)

### Phase 5: Benchmark Integration (Week 7–8)
**Goal:** Run SWE-Bench Verified and Terminal-Bench tasks

- [ ] SWE-Bench task loader (parse task JSON, set up git repo)
- [ ] Terminal-Bench task loader
- [ ] Evaluator: run test suite, check pass/fail
- [ ] Trajectory logging: save every turn to JSON for analysis
- [ ] Batch runner: run N tasks, collect results
- [ ] Metrics: success rate, cost (tokens), turns, duration

### Phase 6: Analysis & Paper Replication (Week 9–10)
**Goal:** Replicate key findings from the paper

- [ ] Trajectory analysis: where do runs fail? (overflow, no edit, wrong file)
- [ ] Cost vs accuracy plots across context budgets
- [ ] Ablation studies: context strategy, planning, action space
- [ ] Export results to CSV/JSON for visualization

---

## 5. Key Implementation Details

### 5.1 Agent Loop (axolotl-core)

```rust
pub async fn run_agent_loop(
    task: &Task,
    config: &SessionConfig,
    llm: &dyn LLMProvider,
    actions: &dyn ActionRegistry,
    context: &mut dyn ContextStrategy,
) -> Result<SessionOutcome> {
    let mut session = Session::new(task, config);

    // Optional planning phase
    if config.planning_enabled {
        let plan = generate_plan(task, llm).await?;
        session.plan = Some(plan);
        context.inject_plan(&session.plan);
    }

    for turn_num in 0..config.max_turns {
        // 1. Build context window
        let messages = context.build_window(config.context_window)?;

        // 2. Call LLM
        let response = llm.chat(&messages).await?;
        session.record_turn(turn_num, &response);

        // 3. Check for completion
        if response.is_terminal() {
            break;
        }

        // 4. Execute tool calls
        for tool_call in &response.tool_calls {
            let result = actions.execute(tool_call).await?;
            context.add_tool_result(result);
        }

        // 5. Apply context management
        context.manage()?;

        // 6. Check timeout
        if session.elapsed() > config.timeout_secs {
            return Ok(SessionOutcome::Timeout);
        }
    }

    Ok(session.evaluate())
}
```

### 5.2 Context Strategy T4 (the winner)

```rust
pub struct T4Strategy {
    elision_threshold: f32,   // 0.6 — B1, start eliding
    summary_threshold: f32,   // 0.85 — B2, start summarizing
    summary_ratio: f32,        // 0.3 — summarize 30% of context
    summarizer: Box<dyn Summarizer>,
}

impl ContextStrategy for T4Strategy {
    fn manage(&self, state: &mut ContextState) -> Result<()> {
        // Step 1: Rule-based elision (cheap) — M1/M2
        self.elide_middle_outputs(state)?;

        // Step 2: If still over the hard threshold, summarize — M3
        if state.usage_ratio() > self.summary_threshold {
            self.summarize_oldest_middle(state)?;
        }

        Ok(())
    }
}
```

### 5.3 Trajectory Logging

Every session writes to `trajectories/<session_id>.json`:

```json
{
  "session_id": "uuid",
  "task_id": "swe-bench-123",
  "config": { ... },
  "turns": [
    {
      "turn_number": 0,
      "input_tokens": 1200,
      "output_tokens": 340,
      "tool_calls": [{"name": "read_file", "args": {...}}],
      "tool_results": [{"output": "file contents...", "tokens": 800}],
      "duration_ms": 2340
    }
  ],
  "outcome": "success",
  "total_tokens": 45000,
  "total_cost_usd": 0.12,
  "total_turns": 15
}
```

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Tokenizer mismatch** | Use `tiktoken-rs` for OpenAI models; for Anthropic, use their published tokenizer or approximate with cl100k_base |
| **LLM API rate limits** | Implement exponential backoff with jitter in `axolotl-models` |
| **Context overflow crashes** | Always check token count before sending; graceful degradation to T0 if management fails |
| **Benchmark setup complexity** | Use Docker containers for SWE-Bench; start with a small subset (10 tasks) |
| **Async debugging pain** | Use `tracing` crate with structured spans for every turn and tool call |
| **Cost runaway** | Hard cap on tokens per session; kill session if exceeded |

---

## 7. Testing Strategy

### Unit Tests
- Token counting accuracy
- Elision logic correctness
- Tool call parsing

### Integration Tests
- Single task end-to-end (mock LLM)
- Context overflow handling
- Planning on/off comparison

### Benchmark Tests
- Run 10 SWE-Bench tasks with T4/128k (baseline)
- Compare T0 vs T4 success rates
- Measure cost difference

---

## 8. File Structure

```
axolotl/
├── Cargo.toml                    # Workspace root
├── implementation_plan.md        # THIS FILE
├── README.md
├── config/
│   └── default.toml              # Default session config
├── crates/
│   ├── axolotl-core/             # Agent loop + state
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── agent.rs          # Agent loop
│   │       ├── session.rs        # Session state
│   │       ├── turn.rs           # Turn recording
│   │       └── types.rs          # Core types
│   ├── axolotl-context/          # Context management
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── strategy.rs       # ContextStrategy trait
│   │       ├── t0.rs             # No management
│   │       ├── t1.rs             # Elision only
│   │       ├── t2.rs             # Elision + recall
│   │       ├── t3.rs             # Summarization only
│   │       ├── t4.rs             # Elision → Summarization
│   │       └── tokenizer.rs      # Token counting
│   ├── axolotl-planning/         # Planning module
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── strategy.rs       # PlanningStrategy trait
│   │       └── default.rs        # Default LLM-based planner
│   ├── axolotl-actions/          # Tool registry
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── registry.rs       # ActionRegistry trait
│   │       ├── bash.rs           # Bash executor
│   │       ├── file_ops.rs       # read/write/edit file tools
│   │       └── search.rs         # Code search tools
│   ├── axolotl-models/           # LLM adapters
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── provider.rs       # LLMProvider trait
│   │       ├── openai.rs         # OpenAI-compatible adapter
│   │       └── types.rs          # ChatMessage, LLMResponse
│   ├── axolotl-bench/            # Benchmark runners
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── swe_bench.rs      # SWE-Bench loader/evaluator
│   │       ├── terminal_bench.rs # Terminal-Bench loader/evaluator
│   │       └── task.rs           # Task trait
│   └── axolotl-cli/              # CLI binary
│       ├── Cargo.toml
│       └── src/
│           └── main.rs           # CLI entrypoint
├── trajectories/                 # Output directory for session logs
├── tests/                        # Integration tests
│   └── integration_test.rs
└── docs/
    └── architecture.md
```

---

## 9. Dependencies Rationale

| Crate | Why |
|-------|-----|
| `tokio` | Async runtime for concurrent LLM calls and tool execution |
| `async-trait` | Trait objects with async methods |
| `serde` + `serde_json` | Serialization for trajectories, configs, API calls |
| `reqwest` | HTTP client for LLM APIs |
| `tracing` | Structured logging — critical for debugging agent loops |
| `tiktoken-rs` | Token counting for OpenAI models |
| `clap` | CLI argument parsing |
| `uuid` | Session and turn IDs |
| `chrono` | Timestamps |
| `thiserror` | Ergonomic error types |
| `anyhow` | Application-level error handling |
| `tempfile` | Sandboxed execution directories |
| `walkdir` | File system traversal for search tools |
| `regex` | Pattern matching in search tools |
