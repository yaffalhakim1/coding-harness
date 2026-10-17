//! System prompt assembly.
//!
//! Text follows paper Appendix A.1: Figure 10 for the predefined action space,
//! Figure 11 for bash-only, Figure 12 appended when only a shell is available,
//! and Figures 13/14 for per-interface tool rules.

use crate::session::ActionSpace;

/// Base system prompt for the full action space (paper Figure 10).
pub const PREDEFINED_SYSTEM_PROMPT: &str = "\
You are a code agent operating inside a local repository.
Your job is to complete the user's coding task with high reliability.
Core workflow:
1. Explore the workspace before acting (list_files, glob_files, grep_text, read_file).
2. Before modifying an existing file you MUST read it first with read_file.
3. Prefer edit_file (targeted old_text/new_text replacement) over write_file (overwrite).
4. Run focused verification (tests, type-check) when appropriate via bash.
5. When the task is done, write a short summary describing what changed and how it was verified.
Rules:
- Never fabricate file contents or test results. Only claim a test passed if you ran it.
- Never operate outside the workspace; paths must be relative to the workspace root.
- Tool calls can be issued in parallel for read-only operations; the harness will batch them.";

/// Base system prompt for the bash-only action space (paper Figure 11).
pub const BASH_ONLY_SYSTEM_PROMPT: &str = "\
You are a code agent operating inside a local repository.
Your job is to complete the user's coding task with high reliability.
Core workflow:
1. Explore the workspace before acting.
2. Before modifying an existing file you MUST read it first.
3. Prefer surgical edits over rewriting a whole file.
4. Run focused verification (tests, type-check) when appropriate.
5. When the task is done, write a short summary describing what changed and how it was verified.
Rules:
- Never fabricate file contents or test results. Only claim a test passed if you ran it.
- Never operate outside the workspace; paths must be relative to the workspace root.
- Tool calls can be issued in parallel for read-only operations; the harness will batch them.";

/// Appended when only a shell is available (paper Figure 12).
pub const BASH_ONLY_BLOCK: &str = "\
# Working with only a shell
Every interaction with the codebase goes through `bash` — you write the commands yourself.";

/// Tool rules for the full action space (paper Figure 13).
pub const PREDEFINED_TOOL_RULES: &str = "\
# Tool Usage Rules
The available tools and their input/output schemas are provided to you via the API's `tools` field. The
rules below tell you HOW to use them well.
- Read-only tools (list_files, glob_files, grep_text, read_file) can be called in PARALLEL —
  issue multiple in a single response when exploring.
- edit_file: small targeted changes. `old_text` must be unique in the file (or pass `replace_all=true`).
  You MUST read the target with read_file first — the harness rejects edits that skip this.
- write_file: prefer for NEW files. To overwrite an existing file you must pass `overwrite=true` AND have
  read it fully first.
- bash: tests, lint, git status/diff. Not every workspace is a git repository — if `git` reports that it
  is not one, that is expected; do not retry it.
After each tool call, read the result. If it errored, FIX THE INPUT rather than retrying the same call.";

/// Tool rules for the bash-only action space (paper Figure 14).
pub const BASH_ONLY_TOOL_RULES: &str = "\
# Tool Usage Rules
The available tools and their input/output schemas are provided to you via the API's `tools` field. The
rules below tell you HOW to use them well.
- bash: tests, lint, git status/diff. This is your only way to read, search, and edit files. Not every
  workspace is a git repository — if `git` reports that it is not one, that is expected; do not retry it.
After each tool call, read the result. If it errored, FIX THE INPUT rather than retrying the same call.";

/// Planning instructions (paper Figure 15).
pub const PLANNING_BLOCK: &str = "\
# Task planning
You have an `update_plan` tool that holds a todo list. The harness shows the current plan back to you
before every turn, so use it to stay on track.
- For any non-trivial task (about 3+ steps), your FIRST action MUST be to call `update_plan` to break
  the task into concrete steps.
- Keep it current: mark exactly ONE task `in_progress` while you work on it, and mark a task
  `completed` the moment it is actually done (don't batch completions).
- Add new tasks as you discover them; re-send the COMPLETE list each time.
- Skip planning only for a single trivial step or a purely informational request.";

/// Build the full system prompt for an action space.
pub fn system_prompt(action_space: ActionSpace, planning_enabled: bool) -> String {
    let (base, rules, bash_block) = match action_space {
        ActionSpace::BashOnly => (
            BASH_ONLY_SYSTEM_PROMPT,
            BASH_ONLY_TOOL_RULES,
            Some(BASH_ONLY_BLOCK),
        ),
        ActionSpace::Predefined | ActionSpace::Hybrid => {
            (PREDEFINED_SYSTEM_PROMPT, PREDEFINED_TOOL_RULES, None)
        }
    };

    let mut prompt = String::from(base);
    if let Some(block) = bash_block {
        prompt.push_str("\n\n");
        prompt.push_str(block);
    }
    if planning_enabled {
        prompt.push_str("\n\n");
        prompt.push_str(PLANNING_BLOCK);
    }
    prompt.push_str("\n\n");
    prompt.push_str(rules);
    prompt.push_str("\n\n");
    prompt.push_str(&platform_block());
    prompt
}

/// Describe the shell dialect the `bash` tool actually runs.
///
/// Without this the model guesses: on Windows it reaches for `ls`/`cat`/`grep`
/// and burns turns on "not recognized as an internal or external command".
/// Naming the platform up front is cheaper than six failed turns.
pub fn platform_block() -> String {
    if cfg!(windows) {
        "# Platform\n\
         The `bash` tool runs commands through Windows `cmd.exe`, NOT a POSIX shell.\n\
         - Available: `dir`, `type`, `findstr`, `where`, `copy`, `move`, `del`, `mkdir`, `set`, `powershell`.\n\
         - NOT available: `ls`, `cat`, `grep`, `sed`, `awk`, `wc`, `find`, `head`, `tail`, `pwd`.\n\
         - Chain with `&&`; separate with `&`. Use `%VAR%` for variables.\n\
         - `findstr` matches whole lines with `/X /C:\"...\"`, but only on CRLF files. A file written by\n\
           `write_file` uses LF endings, so `findstr /X` will NOT match it — verify with `type`, or with\n\
           `powershell -NoProfile -Command \"...\"` instead.\n\
         - Prefer the structured tools (`read_file`, `write_file`, `edit_file`, `grep_text`) over shell\n\
           equivalents; they are platform-independent."
            .to_string()
    } else {
        "# Platform\nThe `bash` tool runs commands through `/bin/sh -c` on a POSIX system."
            .to_string()
    }
}

/// Build the first user turn: the task and how it will be judged.
pub fn task_prompt(description: &str, test_commands: &[String]) -> String {
    let mut prompt = format!(
        "## Task\n{description}\n\n## Instructions\nSolve this task using the available tools. Start by understanding the codebase, then make targeted edits, and verify your changes."
    );
    if !test_commands.is_empty() {
        prompt.push_str(
            "\n\nYour work will be judged by running these commands; they must exit successfully:\n",
        );
        for cmd in test_commands {
            prompt.push_str(&format!("- `{cmd}`\n"));
        }
    }
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predefined_prompt_mentions_structured_tools() {
        let p = system_prompt(ActionSpace::Predefined, false);
        assert!(p.contains("read_file"));
        assert!(p.contains("edit_file"));
        assert!(!p.contains("only a shell"));
        assert!(!p.contains("# Task planning"));
    }

    #[test]
    fn bash_only_prompt_mentions_only_a_shell() {
        let p = system_prompt(ActionSpace::BashOnly, false);
        assert!(p.contains("Working with only a shell"));
        assert!(!p.contains("glob_files"));
    }

    #[test]
    fn planning_block_is_opt_in() {
        let with = system_prompt(ActionSpace::Predefined, true);
        let without = system_prompt(ActionSpace::Predefined, false);
        assert!(with.contains("update_plan"));
        assert!(!without.contains("update_plan"));
    }

    #[test]
    fn task_prompt_lists_verification_commands() {
        let p = task_prompt("fix it", &["cargo test".to_string()]);
        assert!(p.contains("fix it"));
        assert!(p.contains("cargo test"));
    }

    #[test]
    fn task_prompt_omits_verification_when_absent() {
        let p = task_prompt("fix it", &[]);
        assert!(!p.contains("judged by running"));
    }

    #[test]
    fn hybrid_uses_the_predefined_prompt() {
        assert_eq!(
            system_prompt(ActionSpace::Hybrid, false),
            system_prompt(ActionSpace::Predefined, false)
        );
    }

    #[test]
    fn platform_block_names_the_actual_shell() {
        let block = platform_block();
        if cfg!(windows) {
            assert!(block.contains("cmd.exe"), "{block}");
            assert!(block.contains("findstr"), "{block}");
            // The CRLF trap is the one that silently fails verification.
            assert!(block.contains("CRLF"), "{block}");
        } else {
            assert!(block.contains("/bin/sh"), "{block}");
        }
    }

    #[test]
    fn system_prompt_includes_the_platform_block() {
        let p = system_prompt(ActionSpace::Predefined, false);
        assert!(p.contains("# Platform"), "{p}");
    }
}
