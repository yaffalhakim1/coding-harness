//! Slash commands: `/name args` handled by the harness, not the model.
//!
//! ACP models commands as ordinary prompt text that the agent recognises
//! (`/web agent client protocol`). That is the right shape: the client needs no
//! special channel, and a command is just a prompt with a prefix.
//!
//! A command here either rewrites the prompt (injecting a skill's body, say) or
//! answers it directly without calling the model at all. The second case is what
//! makes `/help` and `/skills` free.

use crate::skills::{self, Skill};

/// What a command resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    /// Not a command; run it as an ordinary prompt.
    NotACommand,
    /// Send this text to the model instead of the raw prompt.
    Prompt(String),
    /// Answer without calling the model.
    Reply(String),
}

/// A command the harness advertises to the client.
#[derive(Debug, Clone)]
pub struct Command {
    pub name: String,
    pub description: String,
    /// Hint shown before the user types an argument.
    pub input_hint: Option<String>,
}

/// A parsed command invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Invocation {
    pub name: String,
    pub arguments: String,
}

/// Parse `/name rest of the text`.
///
/// Returns `None` for anything that is not a command: a lone `/`, a path, or
/// text that merely contains a slash later on.
pub fn parse(prompt: &str) -> Option<Invocation> {
    let trimmed = prompt.trim_start();
    let rest = trimmed.strip_prefix('/')?;
    if rest.is_empty() {
        return None;
    }

    // A command name is one token; a slash inside a path must not match.
    let (name, arguments) = match rest.find(char::is_whitespace) {
        Some(idx) => (&rest[..idx], rest[idx..].trim()),
        None => (rest, ""),
    };

    if name.is_empty() || name.contains('/') {
        return None;
    }

    Some(Invocation {
        name: name.to_string(),
        arguments: arguments.to_string(),
    })
}

/// The built-in commands, plus one per discovered skill.
pub fn catalogue(skills: &[Skill]) -> Vec<Command> {
    let mut commands = vec![
        Command {
            name: "help".into(),
            description: "List the available commands".into(),
            input_hint: None,
        },
        Command {
            name: "skills".into(),
            description: "List the skills this workspace provides".into(),
            input_hint: None,
        },
        Command {
            name: "plan".into(),
            description: "Ask for a step-by-step plan before acting".into(),
            input_hint: Some("what to plan".into()),
        },
        Command {
            name: "review".into(),
            description: "Review the current changes for problems".into(),
            input_hint: Some("what to focus on".into()),
        },
        Command {
            name: "test".into(),
            description: "Run the project's tests and report failures".into(),
            input_hint: None,
        },
    ];

    for skill in skills {
        commands.push(Command {
            name: skill.name.clone(),
            description: skill.description.clone(),
            input_hint: Some("optional extra context".into()),
        });
    }

    commands
}

/// Resolve a prompt against the command catalogue.
///
/// A skill takes precedence over a built-in of the same name, so a workspace can
/// override `/test` by adding a skill rather than patching the harness.
pub fn resolve(prompt: &str, skills: &[Skill]) -> Resolution {
    let Some(invocation) = parse(prompt) else {
        return Resolution::NotACommand;
    };

    if let Some(skill) = skills::find(skills, &invocation.name) {
        let extra = invocation.arguments.trim();
        let text = if extra.is_empty() {
            skill.render()
        } else {
            format!("{}\n\n{}", skill.render(), extra)
        };
        return Resolution::Prompt(text);
    }

    match invocation.name.as_str() {
        "help" => Resolution::Reply(render_help(&catalogue(skills))),
        "skills" => Resolution::Reply(render_skills(skills)),
        "plan" => Resolution::Prompt(format!(
            "Produce a step-by-step plan before making any changes.\n\n{}",
            or_default(&invocation.arguments, "the current task")
        )),
        "review" => Resolution::Prompt(format!(
            "Review the current changes for correctness, edge cases and regressions. Report problems; do not rewrite everything.\n\nFocus: {}",
            or_default(&invocation.arguments, "everything that changed")
        )),
        "test" => Resolution::Prompt(
            "Run the project's tests, then report exactly which failed and why. Do not claim a test passed unless you ran it."
                .to_string(),
        ),
        _ => Resolution::NotACommand,
    }
}

fn or_default<'a>(value: &'a str, fallback: &'a str) -> &'a str {
    if value.trim().is_empty() {
        fallback
    } else {
        value
    }
}

/// Render the `/help` reply.
pub fn render_help(commands: &[Command]) -> String {
    let mut out = String::from("Available commands:\n");
    for command in commands {
        match &command.input_hint {
            Some(hint) => out.push_str(&format!(
                "  /{:<10} {}  <{}>\n",
                command.name, command.description, hint
            )),
            None => out.push_str(&format!(
                "  /{:<10} {}\n",
                command.name, command.description
            )),
        }
    }
    out.push_str("\nAnything else is sent to the model as a prompt.");
    out
}

/// Render the `/skills` reply.
pub fn render_skills(skills: &[Skill]) -> String {
    if skills.is_empty() {
        return "No skills found. Add one at .axolotl/skills/<name>/SKILL.md".to_string();
    }
    let mut out = String::from("Skills in this workspace:\n");
    for skill in skills {
        out.push_str(&format!("  /{:<12} {}\n", skill.name, skill.description));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn skill(name: &str, body: &str) -> Skill {
        skills::parse_skill(name, Path::new("p"), body)
    }

    #[test]
    fn parses_a_command_with_arguments() {
        let invocation = parse("/plan the auth refactor").unwrap();
        assert_eq!(invocation.name, "plan");
        assert_eq!(invocation.arguments, "the auth refactor");
    }

    #[test]
    fn parses_a_bare_command() {
        let invocation = parse("/help").unwrap();
        assert_eq!(invocation.name, "help");
        assert_eq!(invocation.arguments, "");
    }

    #[test]
    fn ordinary_text_is_not_a_command() {
        assert!(parse("fix the bug").is_none());
        assert!(parse("what does /plan do?").is_none());
    }

    #[test]
    fn a_lone_slash_is_not_a_command() {
        assert!(parse("/").is_none());
        assert!(parse("/ ").is_none());
    }

    #[test]
    fn a_path_is_not_a_command() {
        // `/usr/bin/env` must not be mistaken for a command named `usr`.
        assert!(parse("/usr/bin/env python").is_none());
    }

    #[test]
    fn help_lists_every_command() {
        let commands = catalogue(&[]);
        let reply = render_help(&commands);
        for expected in ["/help", "/skills", "/plan", "/review", "/test"] {
            assert!(reply.contains(expected), "{reply}");
        }
    }

    #[test]
    fn skills_appear_in_the_catalogue() {
        let commands = catalogue(&[skill("commit", "Write a commit message")]);
        assert!(commands.iter().any(|c| c.name == "commit"));
    }

    #[test]
    fn a_skill_resolves_to_its_body() {
        let skills = vec![skill("commit", "Write a conventional commit message.")];
        match resolve("/commit", &skills) {
            Resolution::Prompt(text) => {
                assert!(text.contains("conventional commit"), "{text}");
                assert!(text.contains("<skill name=\"commit\">"), "{text}");
            }
            other => panic!("expected a prompt, got {other:?}"),
        }
    }

    #[test]
    fn an_unknown_command_is_an_ordinary_prompt() {
        assert_eq!(resolve("/nonsense x", &[]), Resolution::NotACommand);
    }

    #[test]
    fn plan_injects_the_arguments() {
        match resolve("/plan the migration", &[]) {
            Resolution::Prompt(text) => assert!(text.contains("the migration"), "{text}"),
            other => panic!("expected a prompt, got {other:?}"),
        }
    }

    #[test]
    fn plan_without_arguments_still_produces_instructions() {
        match resolve("/plan", &[]) {
            Resolution::Prompt(text) => assert!(text.contains("step-by-step"), "{text}"),
            other => panic!("expected a prompt, got {other:?}"),
        }
    }

    #[test]
    fn help_answers_without_a_model_call() {
        match resolve("/help", &[]) {
            Resolution::Reply(text) => assert!(text.contains("Available commands")),
            other => panic!("expected a reply, got {other:?}"),
        }
    }

    #[test]
    fn skills_with_no_skills_explains_where_to_add_one() {
        match resolve("/skills", &[]) {
            Resolution::Reply(text) => assert!(text.contains(".axolotl/skills"), "{text}"),
            other => panic!("expected a reply, got {other:?}"),
        }
    }

    #[test]
    fn a_skill_can_shadow_a_builtin() {
        // A workspace overriding `/test` should win, not be ignored.
        let skills = vec![skill("test", "Run only the unit tests, not the e2e suite.")];
        match resolve("/test", &skills) {
            Resolution::Prompt(text) => assert!(text.contains("unit tests"), "{text}"),
            other => panic!("expected the skill, got {other:?}"),
        }
    }
}
