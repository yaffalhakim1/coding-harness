# Launching this harness from Kerenzikov
#
# Kerenzikov drives external agent CLIs over a protocol. `axolotl-acp` speaks
# ACP v1 over stdio, which is the same transport it already uses for Copilot,
# Cursor, Grok and Kimi — so it slots in as a provider without touching
# Kerenzikov's driver layer.
#
# Two things make that possible, and both matter:
#
#   1. ACP over stdio, not a bespoke protocol. Adding a provider in Kerenzikov
#      means "route the transport in driver::start"; ACP is already one of the
#      routes (crates/waku-core/src/driver/acp.rs).
#
#   2. Configuration through the environment, because the client launches the
#      binary and knows nothing about this project's config files.

## Launch command

```
axolotl-acp --stdio
```

## Environment

| Variable | Meaning | Default |
|---|---|---|
| `AXOLOTL_MODEL` | model id | `deepseek-v4-1-flash` |
| `AXOLOTL_API_BASE` | OpenAI-compatible endpoint | provider default |
| `AXOLOTL_API_KEY_ENV` | env var holding the API key | `KENARI_API_KEY` |
| `AXOLOTL_STRATEGY` | context tier `T0`–`T4` | `T4` |
| `axolotl_context_WINDOW` | token budget | `128000` |
| `axolotl_planning` | `1` / `0` | `0` |
| `AXOLOTL_MAX_TURNS` | turns per prompt | `40` |

`AXOLOTL_STRATEGY` is the one setting worth exposing in a UI: T4 is the
paper's best accuracy/cost tier, and a tight `axolotl_context_WINDOW` makes the
difference measurable rather than theoretical.

## What the agent advertises

`initialize` reports:

- `protocolVersion: 1`
- `loadSession: true` — `session/load` replays the conversation
- `sessionCapabilities.resume` — reconnect without replay
- `sessionCapabilities.close`
- `promptCapabilities.embeddedContext: true`, `image: false`, `audio: false`

Image and audio are deliberately false. The harness drives a text-only model
interface; advertising support it cannot honour would have a client send a
screenshot that silently degrades to a placeholder. Embedded text resources are
supported, so a client may attach file contents.

## Methods implemented

| Method | Behaviour |
|---|---|
| `initialize` | version + capabilities, as above |
| `session/new` | creates a session rooted at an absolute `cwd` |
| `session/load` | replays history as `session/update`, then responds |
| `session/resume` | restores without replay |
| `session/close` | cancels and frees the session |
| `session/list` | the sessions this process holds |
| `session/prompt` | runs a turn; streams updates; returns a `stopReason` |
| `session/cancel` | sets the cancellation flag for that session |

## Streamed updates

| `sessionUpdate` | Source in the harness |
|---|---|
| `user_message_chunk` | the prompt, echoed so delta-only clients show it |
| `agent_message_chunk` | the model's text for the turn |
| `tool_call` / `tool_call_update` | each tool execution, with `kind`, `title`, `rawInput`, `rawOutput` |
| `usage_update` | prompt tokens against the context window |
| `plan` | the plan, when planning is enabled |

Tool `kind` is mapped from the harness's own tool names: `read_file` → `read`,
`edit_file` → `edit`, `grep_text` → `search`, `bash` → `execute`, `web_fetch` →
`fetch`, `update_plan` → `think`.

## Adding it as a Kerenzikov provider

Following Kerenzikov's own "Adding a provider" checklist:

1. `ProviderKind` variant with `id: "axolotl"`, display name, and command
   `axolotl-acp`.
2. `ProviderResumeCursor` variant carrying `session_id`. The harness holds
   sessions in memory for the process lifetime, so a resume after the process
   exits starts a new session — the cursor is honest about that rather than
   pretending to restore.
3. Transport: ACP (`driver/acp.rs`), already implemented on their side.
4. Map the stream: `agent_message_chunk` → `TextDelta`, `agent_thought_chunk` →
   `ReasoningDelta`, `tool_call*` → `RichActivity`, `plan` → `TodoUpdated`,
   `usage_update` → `UsageUpdated`, `session/prompt` response → `TurnFinished`
   keyed off `stopReason`.
5. Access modes: the harness has no permission channel yet, so it maps like Amp
   and Pi — Full access only, with the client deciding the posture at launch.
6. `#[ignore]`d integration test driving a real turn: `crates/axolotl-acp/tests/stdio.rs`.

## Known gaps against Kerenzikov's checklist

Stated plainly, because a control that silently does nothing is worse than an
absent one:

- **No `session/request_permission`.** The harness's tool layer has no approval
  hook, so a client cannot gate a `bash` call. This is the largest gap and the
  one worth closing first: it is what makes Supervised mode mean something.
- **No `fs/*` or `terminal/*` client capabilities are requested.** The harness
  uses its own tools, which is the correct choice — advertising a client
  capability it does not use is what strands an agent mid-tool-call.
- **No rewind or branch.** `session/load` replays; nothing truncates. Kerenzikov's
  `supports_conversation_fork` / `supports_conversation_rollback` should be
  false, which hides those affordances.
- **No model catalog.** `AXOLOTL_MODEL` is set at launch; there is no
  `session/set_model`.
- **Sessions are in-memory.** `session/resume` works within one process
  lifetime. Persisting them would need a snapshot store — the CLI already has
  one (`axolotl sessions`), so wiring it here is a small change.
