# Installing `axolotl`

The point of this project is a single command you can type:

```bash
axolotl run --task "fix the parser" --workspace .
```

## Build and install (Windows)

```powershell
cargo install --path crates/axolotl-cli
```

That puts `axolotl.exe` in `%USERPROFILE%\.cargo\bin`, which is already on PATH
for a standard Rust install. The ACP server and TUI are separate binaries:

```powershell
cargo install --path crates/axolotl-acp
cargo install --path crates/axolotl-tui
```

## Build and install (macOS / Linux)

```bash
cargo install --path crates/axolotl-cli
cargo install --path crates/axolotl-acp
cargo install --path crates/axolotl-tui
```

## From a release build

```bash
cargo build --release
# then copy the binary somewhere on PATH
cp target/release/axolotl ~/.local/bin/
```

## What each binary is

| Binary | Purpose |
|---|---|
| `axolotl` | the CLI: `run`, `bench`, `ablate`, `analyze`, `sessions`, `resume`, `doctor` |
| `axolotl-acp` | ACP server over stdio, for Kerenzikov and other ACP clients |
| `axolotl-tui` | live terminal view of a run |

## First run

```bash
axolotl doctor --config config/kenari.toml --check-api
```

`doctor` confirms the config parses, the credential resolves, the workspace is
writable, and the endpoint answers — before you spend a turn on a mistake.

## Credentials

Keys resolve in this order:

1. the environment variable named by `model.api_key_env`
2. `~/.local/share/opencode/auth.json`, when a matching `api` entry exists

`--dry-run` prints which source was used.
