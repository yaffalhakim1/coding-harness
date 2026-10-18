# What Axolotl does

A short, plain-language guide. No Rust, no configuration files, no protocol.

## The one-sentence version

Axolotl is a coding agent you run in a terminal. You give it a job in English; it
reads your code, edits files, runs your tests, and tells you what it changed.

```bash
axolotl run --task "the parser panics on empty input — fix it"
```

## Why it exists

Most coding agents are large. OpenCode is a Node.js process tree that can use
hundreds of megabytes before it does anything useful. Axolotl is one Rust binary
that starts in milliseconds and uses a few megabytes, so you can run several at
once on a laptop without noticing.

It is also the agent behind
[Kerenzikov](https://github.com/yaffalhakim1/Kerenzikov-app), a native desktop
app for coding agents. Axolotl is what Kerenzikov talks to.

## What it can do

**Read and change code.** It has tools to read files, write files, make targeted
edits, list directories, search by filename, and search file contents. It also
has a shell, so it can run anything you could type.

**Run your tests.** It will run your test command and read the output. It cannot
claim a test passed unless it actually ran it.

**Ask before doing something dangerous.** Writing a file, editing a file, or
running a command requires your approval. You see what it wants to do and can say
yes or no. If you say no, it tries a different approach rather than giving up.

**Ask you questions.** When it genuinely cannot decide — two reasonable designs,
an unclear requirement — it asks instead of guessing.

**Keep a to-do list.** For anything more than a couple of steps it writes out a
plan and checks items off as it goes, so you can see progress.

**Use skills.** If your repo has a `.axolotl/skills/` folder, the agent can load
those instructions on demand. Useful for project-specific rules like "we always
write conventional commits."

**Understand slash commands.** `/help`, `/plan`, `/review`, `/test` and one per
skill. `/help` and `/skills` answer instantly without calling the model.

**Delegate.** It can hand a self-contained investigation to a helper agent with
its own fresh context, and get back just the answer.

**Use MCP servers.** If you already have tools exposed over the Model Context
Protocol, the agent can use them.

**Resume.** Every run is saved. You can list past sessions and continue one where
it stopped, rather than starting over.

## What it cannot do

Being straight with you, because these matter more than the feature list:

- **It has not been proven on real work yet.** Every task it has completed so far
  is a small synthetic one — create a file, count lines. It has never fixed a
  real bug in a real repository. Until it has, treat it as promising rather than
  proven.
- **It has no permission prompt in the terminal.** The desktop app asks before
  running a command; `axolotl run` does not. A destructive command list is
  refused, but everything else runs immediately.
- **It cannot rewind or branch a conversation.** You can resume, not undo.
- **Sessions do not survive a restart** when used from the desktop app. The
  terminal version saves them; the protocol server does not yet.
- **It has no editor, no file tree, no diff viewer.** It is a command, not an app.
  For a graphical experience use Kerenzikov.

## How it decides what to do

Each turn it looks at the conversation, decides on one action, does it, and looks
at the result. That repeats until it believes the job is done or hits a limit.

Three things shape that loop, and you can configure all three:

**How much it remembers.** Long jobs produce a lot of output — test logs, file
contents. Axolotl has five strategies for managing that, from "keep everything
and stop if it overflows" to "summarise the old parts." The default is the one
that a published study found best: shrink stale tool output first (cheap), and
only pay for summarisation if that was not enough.

**Whether it plans first.** Off by default. For a strong model it mostly adds
cost; for a weaker one it helps accuracy.

**What tools it gets.** A structured set (read, write, edit, search, shell), or
just a shell. Bash-only is cheaper for models that are good at shell.

## How much it costs

It calls a model, so it costs what the model costs. Two things keep that
predictable:

- Every run reports tokens used and, if you configured prices, the dollar cost.
- You can set a hard token cap per session; the run stops when it is reached.

You can also point it at a cheaper model for simple jobs and a stronger one for
hard ones.

## Where your data goes

It runs entirely on your machine. It sends your prompts and the file contents it
reads to whichever model provider you configured — nothing else. There is no
account, no telemetry, and no server of ours.

API keys are read from an environment variable, or from the credentials file that
OpenCode already uses. You can check which source was used with `--dry-run`.

## Getting started

```bash
cargo install --path crates/axolotl-cli
axolotl doctor --config config/kenari.toml --check-api
axolotl run --config config/kenari.toml --task "your job here"
```

`doctor` checks that the config parses, the key resolves, and the endpoint
answers — before you spend a turn on a typo.

Then read [how-it-works.md](how-it-works.md) if you want the mechanism, or
[install.md](install.md) for platform details.
