//! `axolotl analyze` and the human-readable session report.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use axolotl_core::Session;

use crate::AnalyzeArgs;

/// Process exit code: 0 only when the task's verifier passed.
pub fn exit_code_for(session: &Session) -> i32 {
    if session
        .outcome
        .as_ref()
        .is_some_and(axolotl_core::SessionOutcome::is_success)
    {
        0
    } else {
        1
    }
}

/// Print a human-readable summary of a session.
pub fn report(session: &Session) {
    let outcome = session
        .outcome
        .as_ref()
        .map(axolotl_core::SessionOutcome::label)
        .unwrap_or("none");

    println!();
    println!("outcome          {outcome}");
    println!("turns            {}", session.turn_count());
    println!(
        "tokens           {} (prompt {}, completion {})",
        session.total_usage.total_tokens,
        session.total_usage.prompt_tokens,
        session.total_usage.completion_tokens
    );
    match session.cost_usd() {
        Some(cost) => println!("cost             ${cost:.4}"),
        None => println!("cost             unknown (no prices in config)"),
    }
    println!("peak context     {}", session.peak_context_tokens);
    println!("summarizations   {}", session.summarize_calls);
    println!("duration         {}s", session.elapsed_secs());

    if let Some(result) = session.outcome.as_ref().and_then(|o| o.task_result()) {
        if !result.test_output.is_empty() {
            println!();
            println!("verification:");
            println!("{}", result.test_output.trim_end());
        }
    }
}

/// Print a summary of a saved trajectory, including failure attribution.
pub fn analyze(args: AnalyzeArgs) -> Result<i32> {
    let path = resolve_trajectory(&args.trajectory, &args.dir)?;
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading trajectory '{}'", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&text)?;

    println!(
        "session          {}",
        json["session_id"].as_str().unwrap_or("?")
    );
    println!(
        "task             {}",
        json["task_id"].as_str().unwrap_or("?")
    );
    println!(
        "outcome          {}",
        json["outcome"].as_str().unwrap_or("none")
    );
    println!("config           {}", json["config"]);
    println!("turns            {}", json["total_turns"]);
    println!("tokens           {}", json["total_tokens"]);
    println!("peak context     {}", json["peak_context_tokens"]);

    if let Some(turns) = json["turns"].as_array() {
        println!();
        println!(
            "{:<5} {:<8} {:<8} {:<14} tools",
            "turn", "in", "out", "stop"
        );
        for turn in turns {
            let tools: Vec<String> = turn["tool_calls"]
                .as_array()
                .map(|calls| {
                    calls
                        .iter()
                        .filter_map(|c| c["name"].as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            println!(
                "{:<5} {:<8} {:<8} {:<14} {}",
                turn["turn_number"],
                turn["prompt_tokens"],
                turn["output_tokens"],
                turn["stop_reason"].as_str().unwrap_or("?"),
                tools.join(",")
            );
        }
    }

    // Failure-stage attribution from the durable trajectory record.
    if let Some(analysis) = crate::trajectory_analysis::analyse_trajectory(&json) {
        crate::trajectory_analysis::print_analysis(&analysis);
    }
    Ok(0)
}

/// Resolve a trajectory reference: an existing file, the newest JSON in a
/// directory, or a session id inside `dir`.
fn resolve_trajectory(id_or_path: &str, dir: &Path) -> Result<PathBuf> {
    let direct = Path::new(id_or_path);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }

    // A directory is interpreted as "its newest trajectory", which is what a
    // user gets from passing `--trajectory-dir` by mistake.
    if direct.is_dir() {
        let mut matches: Vec<PathBuf> = std::fs::read_dir(direct)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        matches.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
        return matches
            .pop()
            .ok_or_else(|| anyhow::anyhow!("no trajectories in '{}'", direct.display()));
    }

    let candidate = dir.join(format!("{id_or_path}.json"));
    if candidate.is_file() {
        return Ok(candidate);
    }
    anyhow::bail!(
        "no trajectory found for '{id_or_path}' (looked for a file and {})",
        candidate.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_an_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.json");
        std::fs::write(&path, "{}").unwrap();
        assert_eq!(
            resolve_trajectory(path.to_str().unwrap(), dir.path()).unwrap(),
            path
        );
    }

    #[test]
    fn resolves_a_directory_to_its_newest_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("old.json"), "{}").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let newest = dir.path().join("new.json");
        std::fs::write(&newest, "{}").unwrap();

        let resolved = resolve_trajectory(dir.path().to_str().unwrap(), dir.path()).unwrap();
        assert_eq!(resolved, newest);
    }

    #[test]
    fn resolves_an_id_inside_the_directory() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("abc.json"), "{}").unwrap();
        let resolved = resolve_trajectory("abc", dir.path()).unwrap();
        assert!(resolved.ends_with("abc.json"));
    }

    #[test]
    fn unknown_reference_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_trajectory("nope", dir.path()).is_err());
    }

    #[test]
    fn an_empty_directory_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = resolve_trajectory(dir.path().to_str().unwrap(), dir.path()).unwrap_err();
        assert!(err.to_string().contains("no trajectories"), "{err}");
    }
}
