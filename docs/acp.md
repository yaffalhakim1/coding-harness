# Launching Axolotl from Kerenzikov
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
| `AXOLOTL_CONTEXT_WINDOW` | token budget | `128000` |
| `AXOLOTL_PLANNING` | `1` / `0` | `0` |
| `AXOLOTL_MAX_TURNS` | turns per prompt | `40` |

`AXOLOTL_STRATEGY` is the one setting worth exposing in a UI: T4 is the
paper's best accuracy/cost tier, and a tight `AXOLOTL_CONTEXT_WINDOW` makes the
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

## Approval and questions

Two calls go the other way: the agent asks the client.

**`session/request_permission`** — sent before any mutating tool runs. The client
receives the tool's title, kind, and two options:

| `optionId` | `kind` | Meaning |
|---|---|---|
| `allow-once` | `allow_once` | run it |
| `reject-once` | `reject_once` | refuse |

`allow_always` is deliberately not offered. ACP has no client-side rule store, so
a durable "always" would have to live in the agent — where it would silently
apply to a different session with a different task. One-shot keeps the decision
where the user can see it.

A rejection comes back to the model as a **recoverable observation**, not a
failed turn: it reads "the user denied X" and tries another approach. A
cancellation (client went away, or the turn was cancelled) stops the turn.

Read-only tools skip the request entirely. Prompting for every `read_file` is how
a prompt becomes noise the user clicks through.

A destructive-command list sits inside the gate: `rm -rf`, `sudo`, `git push`,
`git reset --hard`, `chmod -R` and similar are refused without asking. It is a
fast path, not a security boundary.

**`elicitation/create`** — sent by the `ask_user` tool. The schema is a single
string property carrying the question as its title, with an `enum` of the
options when the model offered any. A `decline` or `cancel` action returns "the
user dismissed the question — continue with your best judgement" rather than an
error, so a dismissal cannot dead-end a turn.

## Slash commands

Advertised as `available_commands_update` immediately after `session/new`,
because the catalogue depends on the workspace's skills.

| Command | Behaviour |
|---|---|
| `/help` | lists commands — no model call |
| `/skills` | lists workspace skills — no model call |
| `/plan [what]` | asks for a step-by-step plan |
| `/review [focus]` | reviews the current changes |
| `/test` | runs tests and reports failures |
| `/<skill>` | injects that skill's body |

A skill shadows a built-in of the same name, so a workspace can override `/test`
without patching the agent.

## Skills

Discovered from `.axolotl/skills/<name>/SKILL.md` and
`.agents/skills/<name>/SKILL.md`. Frontmatter (`name`, `description`) is
optional; a body-only file uses its directory name. The body is injected only
when the skill is invoked, so a repo can carry many skills without paying for
them in every prompt.

## Known gaps
 against Kerenzikov's checklist

Stated plainly, because a control that silently does nothing is worse than an
absent one:

- **No rewind or branch.** `session/load` replays; nothing truncates. A client
  should report `supports_conversation_fork` / `supports_conversation_rollback`
  as false, which hides those affordances.
- **Sessions are in-memory.** `session/resume` works within one process lifetime.
  Persisting them would need a snapshot store — the CLI already has one
  (`axolotl sessions`), so wiring it here is a small change.
- **No model catalog.** `AXOLOTL_MODEL` is set at launch; there is no
  `session/set_model`.

