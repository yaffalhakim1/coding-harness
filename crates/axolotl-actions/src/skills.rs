//! Skills: reusable instruction files the agent loads on demand.
//!
//! Both jcode and OpenCode use a `SKILL.md` convention — a directory per skill
//! with YAML frontmatter (`name`, `description`) and a markdown body. The body
//! is injected into the conversation only when the skill is invoked, which is
//! the point: a repo can carry twenty skills without paying for twenty
//! documents in every prompt.
//!
//! Discovery is deliberately shallow and explicit: `.axolotl/skills/<name>/SKILL.md`
//! plus `.agents/skills/<name>/SKILL.md` (the convention already used by other
//! tools in this workspace). No recursive scan, no glob magic.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Directories searched for skills, relative to the workspace root.
pub const SKILL_ROOTS: &[&str] = &[".axolotl/skills", ".agents/skills"];

/// The file name that marks a skill directory.
pub const SKILL_FILE: &str = "SKILL.md";

/// A discovered skill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Invocation name, from frontmatter or the directory name.
    pub name: String,
    pub description: String,
    /// The markdown body, injected when the skill is used.
    pub body: String,
    /// Where it came from, for diagnostics.
    pub path: PathBuf,
}

impl Skill {
    /// The text injected into the conversation when this skill is invoked.
    pub fn render(&self) -> String {
        format!(
            "<skill name=\"{}\">\n{}\n</skill>",
            self.name,
            self.body.trim()
        )
    }
}

/// Parse a `SKILL.md`.
///
/// Frontmatter is optional: a file that starts with `#` is body-only, and the
/// directory name becomes the skill name. Being lenient here matters because a
/// skill with a broken header should still be usable rather than silently
/// missing.
pub fn parse_skill(name_hint: &str, path: &Path, content: &str) -> Skill {
    let (front, body) = split_frontmatter(content);

    let mut name = name_hint.to_string();
    let mut description = String::new();

    if let Some(front) = front {
        for line in front.lines() {
            let Some((key, value)) = line.split_once(':') else {
                continue;
            };
            let value = value
                .trim()
                .trim_matches('"')
                .trim_matches('\'')
                .to_string();
            match key.trim() {
                "name" if !value.is_empty() => name = value,
                "description" => description = value,
                _ => {}
            }
        }
    }

    // No frontmatter description: use the first markdown heading or sentence.
    if description.is_empty() {
        description = body
            .lines()
            .find_map(|line| {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    None
                } else {
                    Some(trimmed.chars().take(200).collect::<String>())
                }
            })
            .unwrap_or_else(|| format!("Skill '{name}'"));
    }

    Skill {
        name,
        description,
        body: body.trim().to_string(),
        path: path.to_path_buf(),
    }
}

/// Split `---\nfrontmatter\n---\nbody`.
fn split_frontmatter(content: &str) -> (Option<&str>, &str) {
    let trimmed = content.strip_prefix('\u{feff}').unwrap_or(content);
    if !trimmed.starts_with("---") {
        return (None, trimmed);
    }
    let after = &trimmed[3..];
    let after = after.strip_prefix('\r').unwrap_or(after);
    let after = after.strip_prefix('\n').unwrap_or(after);

    match after.find("\n---") {
        Some(end) => {
            let front = &after[..end];
            let rest = &after[end + 4..];
            let rest = rest.strip_prefix('\r').unwrap_or(rest);
            let rest = rest.strip_prefix('\n').unwrap_or(rest);
            (Some(front), rest)
        }
        // An unterminated block is not frontmatter; treat it all as body rather
        // than swallowing the document.
        None => (None, trimmed),
    }
}

/// Discover every skill under `root`.
pub fn discover(root: &Path) -> Vec<Skill> {
    let mut skills = Vec::new();

    for relative in SKILL_ROOTS {
        let dir = root.join(relative);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };

        let mut paths: Vec<PathBuf> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
        paths.sort();

        for entry in paths {
            let file = entry.join(SKILL_FILE);
            let Ok(content) = std::fs::read_to_string(&file) else {
                continue;
            };
            let hint = entry
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "unnamed".into());
            skills.push(parse_skill(&hint, &file, &content));
        }
    }

    // A later root overrides an earlier one with the same name, so a project
    // can shadow a shared skill without editing it.
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills.dedup_by(|a, b| a.name == b.name);
    skills
}

/// Find one skill by name.
pub fn find<'a>(skills: &'a [Skill], name: &str) -> Option<&'a Skill> {
    skills.iter().find(|s| s.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frontmatter_and_body() {
        let content =
            "---\nname: commit\ndescription: Write a commit message\n---\n# Body\nDo the thing.";
        let skill = parse_skill("dir-name", Path::new("SKILL.md"), content);
        assert_eq!(skill.name, "commit");
        assert_eq!(skill.description, "Write a commit message");
        assert!(skill.body.contains("Do the thing"));
        assert!(!skill.body.contains("---"));
    }

    #[test]
    fn falls_back_to_the_directory_name() {
        let content = "---\ndescription: no name here\n---\nBody";
        let skill = parse_skill("from-dir", Path::new("x"), content);
        assert_eq!(skill.name, "from-dir");
    }

    #[test]
    fn body_only_files_are_accepted() {
        let content = "# My skill\n\nThis does something useful.";
        let skill = parse_skill("mine", Path::new("x"), content);
        assert_eq!(skill.name, "mine");
        assert!(skill.body.contains("My skill"));
        // The description falls back to the first non-heading line.
        assert!(
            skill.description.contains("useful"),
            "{}",
            skill.description
        );
    }

    #[test]
    fn an_unterminated_frontmatter_block_is_treated_as_body() {
        let content = "---\nname: broken\nstill going";
        let skill = parse_skill("hint", Path::new("x"), content);
        assert!(skill.body.contains("still going"), "{}", skill.body);
        assert_eq!(skill.name, "hint");
    }

    #[test]
    fn quotes_are_stripped_from_values() {
        let content = "---\nname: \"quoted\"\ndescription: 'single'\n---\nBody";
        let skill = parse_skill("h", Path::new("x"), content);
        assert_eq!(skill.name, "quoted");
        assert_eq!(skill.description, "single");
    }

    #[test]
    fn discovers_skills_from_both_roots() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join(".axolotl/skills/alpha");
        std::fs::create_dir_all(&a).unwrap();
        std::fs::write(a.join(SKILL_FILE), "---\nname: alpha\n---\nA body").unwrap();

        let b = dir.path().join(".agents/skills/beta");
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(b.join(SKILL_FILE), "---\nname: beta\n---\nB body").unwrap();

        let skills = discover(dir.path());
        let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn a_directory_without_a_skill_file_is_skipped() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".axolotl/skills/empty")).unwrap();
        assert!(discover(dir.path()).is_empty());
    }

    #[test]
    fn a_missing_root_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(discover(dir.path()).is_empty());
    }

    #[test]
    fn find_locates_by_name() {
        let skills = vec![
            parse_skill("a", Path::new("x"), "A"),
            parse_skill("b", Path::new("y"), "B"),
        ];
        assert!(find(&skills, "b").is_some());
        assert!(find(&skills, "z").is_none());
    }

    #[test]
    fn rendering_wraps_the_body() {
        let skill = parse_skill("x", Path::new("p"), "Body text");
        let rendered = skill.render();
        assert!(rendered.starts_with("<skill name=\"x\">"));
        assert!(rendered.contains("Body text"));
        assert!(rendered.ends_with("</skill>"));
    }
}
