# How Axolotl works

The mechanism, for someone who will change the code.

## The loop

One turn, in order:

```
build context window
  → call the model
    → no tool calls?  ──yes──> evaluate the task, stop
         │ no
         ↓
    append the assistant message
    → execute tool calls (reads in parallel, mutations in order)
      → append observations
        → manage context (elide, then summarise)
          → check guards (turns, timeout, token budget, stuck)
            → repeat
```

`axolotl-core/src/agent.rs` owns this. Everything else is called from it:

| Concern | Crate | Entry point |
|---|---|---|
| Context window | `axolotl-context` | `ContextStrategy::manage` |
| Tool execution | `axolotl-actions` | `ActionRegistry::execute` |
| Model calls | `axolotl-models` | `LLMProvider::chat` |
| Planning | `axolotl-planning` | `PlanningStrategy::generate_plan` |
| Verification | `axolotl-core` | `evaluator::evaluate_task` |

## Termination

The loop stops on any of:

- **No action.** A response with no tool calls means the model is done. A
  response that is *neither* prose nor a tool call is degenerate; three of those
  in a row ends the run.
- **Turn limit** (`max_turns`, default 50).
- **Wall clock** (`timeout_secs`, default 600).
- **Token budget** (`token_budget`), if set.
- **Context overflow** — only reachable under T0, which does not manage context.
- **Stuck detection** — see below.

## Stuck detection

A *streak* is a run of tool calls with the same name and byte-identical
arguments. At five repeats the harness injects a reminder to change approach. At
eight repeats **that keep failing**, it ends the run. The thresholds come from
the paper (§3.1).

This exists because a confused model will re-issue the same failing command until
it exhausts the turn budget. Ending early is cheaper and more honest than
pretending it was still working.

## Context management

Five strategies, all in `axolotl-context`. They differ in *which* of three
mechanisms they enable:

| Tier | Elision (M1) | Recall (M2) | Summarisation (M3) |
|---|---|---|---|
| T0 | — | — | — |
| T1 | ✓ | — | — |
| T2 | ✓ | ✓ | — |
| T3 | — | — | ✓ |
| **T4** | ✓ | ✓ | ✓ |

**The region model** is the part that matters:

```
[ preamble              | middle (compactable)      | recent (verbatim) ]
  system prompt           oldest … newest-N turns     last ≥ 2 turns
  + task
  + current plan
```

Only the middle is ever compacted. The preamble holds the task description and
the plan — eliding those is how a managed run turns into an unmanaged failure
with extra steps. The recent window is protected so the model can see what it
just did.

**T4 stages the two operations.** Above the soft threshold (0.6 of the window) it
elides bulky tool output in the middle region, storing the originals. Only if
that was not enough — still above the hard threshold, 0.85 — does it call the
model to summarise. Elision is free; summarisation is not. That ordering is the
whole cost advantage over T3, and it is measurable: `summarize_calls` counts it.

**T0 terminates on overflow.** It does not drop messages. That is what makes
"managed tiers never overflow" a measurable claim rather than a tautology.

Elision stubs are reproduced verbatim from the paper (Figure 19), so trajectories
stay comparable:

```
[tool output elided: 412 lines / 18403 chars. Use recall_event(<id>) for the
full output, or re-read/re-run.]
```

`recall_event` is only registered for T2 and T4, because only they keep
originals.

## The tool layer

Every tool goes through `Workspace`, which canonicalises the path and rejects
anything that escapes the root — including through a symlink. This is the
boundary; a tool cannot reach outside the workspace even if the model asks it to.

**Read-before-write.** `edit_file` and `write_file(overwrite)` refuse unless the
file was read *fully* in this session, and refuse if the content hash changed
since. Partial reads do not count. This is what stops a model from overwriting a
file it never looked at.

**Parallel reads.** Adjacent read-only calls run concurrently, up to eight per
step (paper §3.1). A mutating call flushes the pending reads first, so a
`read_file` → `write_file` sequence still observes the read — which the
read-before-write gate depends on.

**Errors are observations.** A tool failure never aborts the loop; it becomes
text the model can act on. A non-zero shell exit is a typed error carrying the
command's output.

## Permissions

`PermissionGate` is consulted before any mutating tool runs. Read-only tools skip
it — prompting for every `read_file` is how a prompt becomes noise.

Three implementations:

- `AllowAll` — the default, so an unconfigured harness behaves as before.
- `DestructiveCommandGate` — refuses `rm -rf`, `sudo`, `git reset --hard` and
  friends without asking. A fast path, **not** a security boundary: a determined
  model can write the same effect many ways.
- `AcpPermissionGate` — asks the connected client.

Only the ACP path has a real gate. **`axolotl run` still allows everything that
is not on the destructive list.** That is a known gap, not a design choice.

## The model layer

`LLMProvider` is one method. `OpenAIProvider` implements it against any
OpenAI-compatible endpoint, which covers OpenAI, the Kenari gateway, and local
models.

Two details that are easy to get wrong:

**Wire types are separate from internal types.** The OpenAI wire format carries
tool arguments as a JSON *string* nested inside a function object, and the `tool`
role requires `tool_call_id`. Serialising the internal type directly emits an
object where a string is required, and every call fails.

**Retries only for retryable errors.** Rate limits, 5xx and transport failures get
exponential backoff with full jitter. Auth and bad-request errors are returned
immediately — retrying those just spends money on a request that is wrong either
way.

`FailoverProvider` wraps an ordered list of routes and applies the same
classification: only transport, rate-limit and 5xx errors move to the next route.
The answering model is recorded in the response, so a silent failover is visible
in the trajectory.

## Verification

A task's `test_commands` are run after the model stops. **Success is the process
exit status**, nothing else. A task with no test commands reports `unverified`
rather than success — the paper's metric is verifier-based, and claiming
otherwise would make the benchmark meaningless.

## Trajectories

Every session writes two files:

| File | Contents | Purpose |
|---|---|---|
| `trajectories/<id>.json` | per-turn summary, output *previews* | analysis; stays readable at sweep scale |
| `sessions/<id>.session.json` | full session | resume |

They are separate because a trajectory full of complete tool output is
unreadable, and a summary cannot be resumed.

## The ACP server

`axolotl-acp` speaks Agent Client Protocol v1 over stdio. See
[acp.md](acp.md) for the method table and the gaps.

Two design points:

**stdout is the protocol.** All logging goes to stderr. A stray `println!` would
corrupt the JSON-RPC stream.

**The connection is bidirectional.** The agent calls the client back —
`session/request_permission` for a tool, `elicitation/create` for a question. That
means one task cannot own stdin: a request must be written while the reader keeps
parsing. `rpc.rs` splits them into a reader task and a mutex-guarded writer.

## Failure modes worth knowing

| Symptom | Cause |
|---|---|
| `LLM API returned 400` mentioning `messages[2].content` | A tool-call-only assistant message serialised `content: null`. The wire type always emits a string. |
| Every task "passes" | The verifier is reading `Ok` instead of the exit status. |
| A command with quotes silently matches nothing on Windows | `cmd /C` strips inner quotes when given an argv element; `raw_arg` passes the line verbatim. |
| `mimeType` / `optionId` parse as `None` | `rename_all` on an enum renames variants, not fields. Each camelCase field needs its own `rename`. |
| Session list shows `turnlimitreached` | The enum variant name leaked instead of the label. |

## Where to change what

| Task | File |
|---|---|
| Add a tool | `axolotl-actions/src/<name>.rs`, register in `registry.rs` |
| Change the system prompt | `axolotl-core/src/prompts.rs` |
| Change the loop | `axolotl-core/src/agent.rs` |
| Change context policy | `axolotl-context/src/{t1,t3,t4}.rs`, thresholds in `thresholds.rs` |
| Add a provider | `axolotl-models/src/provider.rs` + an adapter |
| Add a slash command | `axolotl-actions/src/slash.rs` |
| Add an ACP method | `axolotl-acp/src/agent.rs` |
| Change TUI layout | `axolotl-tui/src/ui.rs`, colours in `theme.rs` |

## Testing

- Unit tests live beside the code, in `mod tests`.
- Integration tests that drive the real binary: `axolotl-acp/tests/stdio.rs`,
  `axolotl-acp/tests/interactive.rs`.
- `#[ignore]`d live tests need a real model:
  `cargo test -p axolotl-core --test live_e2e -- --ignored`
- `axolotl-tui/tests/preview.rs` renders the TUI into a buffer and prints it,
  because layout problems are visible, not assertable.

The rule that has caught the most bugs: **drive the real binary, not just the
unit**. Three protocol bugs (camelCase fields, a wrong error code, a params struct
that required an extra field) were invisible to unit tests and obvious the moment
a real client spoke the protocol.
