//! Failure-stage attribution from a saved trajectory.
//!
//! `harness analyze` works from the trajectory JSON, not the live session, so
//! that a sweep can be re-analysed long after it ran without re-spending tokens.

use harness_bench::{FailureStage, Phase, PhaseCounts, TrajectoryAnalysis};

/// Derive failure-stage attribution from a trajectory document.
pub fn analyse_trajectory(json: &serde_json::Value) -> Option<TrajectoryAnalysis> {
    let turns = json["turns"].as_array()?;
    let outcome = json["outcome"].as_str().unwrap_or("none");
    let passed = outcome == "success";

    let mut counts = PhaseCounts::default();
    let mut tool_calls = 0usize;
    let mut failed_tool_calls = 0usize;

    for turn in turns {
        if let Some(calls) = turn["tool_calls"].as_array() {
            tool_calls += calls.len();
            if let Some(first) = calls.first() {
                let name = first["name"].as_str().unwrap_or("");
                let args = &first["arguments"];
                match harness_bench::analysis::classify_tool(name, args) {
                    Phase::Localize => counts.localize += 1,
                    Phase::Reproduce => counts.reproduce += 1,
                    Phase::Fix => counts.fix += 1,
                    Phase::Verify => counts.verify += 1,
                    Phase::Other => counts.other += 1,
                }
            }
        }
        if let Some(results) = turn["tool_results"].as_array() {
            failed_tool_calls += results
                .iter()
                .filter(|r| r["success"].as_bool() == Some(false))
                .count();
        }
    }

    let edited = counts.fix > 0;
    let verified = counts.verify > 0;

    let failure_stage = if passed {
        FailureStage::None
    } else {
        match outcome {
            "turn_limit" | "timeout" | "context_overflow" | "stuck" => FailureStage::Exhausted,
            _ if counts.localize == 0 => FailureStage::FileLocalization,
            _ if !edited => FailureStage::LineLocalization,
            _ if !verified => FailureStage::PatchImplementation,
            _ => FailureStage::Verification,
        }
    };

    Some(TrajectoryAnalysis {
        session_id: json["session_id"].as_str().unwrap_or("?").to_string(),
        task_id: json["task_id"].as_str().unwrap_or("?").to_string(),
        outcome: outcome.to_string(),
        passed,
        turns: turns.len(),
        phases: Vec::new(),
        phase_counts: counts,
        failure_stage,
        edited,
        verified,
        tool_calls,
        failed_tool_calls,
        total_tokens: json["total_tokens"].as_u64().unwrap_or(0) as usize,
        peak_context_tokens: json["peak_context_tokens"].as_u64().unwrap_or(0) as usize,
        summarize_calls: json["summarize_calls"].as_u64().unwrap_or(0) as usize,
        termination: (outcome != "success" && outcome != "failure" && outcome != "unverified")
            .then(|| outcome.to_string()),
    })
}

/// Print the analysis block for one trajectory.
pub fn print_analysis(analysis: &TrajectoryAnalysis) {
    println!();
    println!("failure stage    {}", analysis.failure_stage.label());
    println!(
        "edited/verified  {} / {}",
        analysis.edited, analysis.verified
    );
    println!(
        "tool calls       {} ({} failed)",
        analysis.tool_calls, analysis.failed_tool_calls
    );
    println!(
        "phases           localize={} fix={} verify={} repro={} other={}",
        analysis.phase_counts.localize,
        analysis.phase_counts.fix,
        analysis.phase_counts.verify,
        analysis.phase_counts.reproduce,
        analysis.phase_counts.other
    );
    if let Some(term) = &analysis.termination {
        println!("termination      {term}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trajectory(outcome: &str, turns: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "session_id": "s1",
            "task_id": "t1",
            "outcome": outcome,
            "turns": turns,
            "total_tokens": 100,
            "peak_context_tokens": 50,
            "summarize_calls": 0,
        })
    }

    fn turn(name: &str, args: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "turn_number": 0,
            "tool_calls": [{ "name": name, "arguments": args }],
            "tool_results": [{ "success": true }],
        })
    }

    #[test]
    fn success_has_no_failure_stage() {
        let json = trajectory(
            "success",
            serde_json::json!([turn("read_file", serde_json::json!({}))]),
        );
        let analysis = analyse_trajectory(&json).unwrap();
        assert!(analysis.passed);
        assert_eq!(analysis.failure_stage, FailureStage::None);
    }

    #[test]
    fn no_localization_is_a_file_localization_failure() {
        let json = trajectory(
            "failure",
            serde_json::json!([turn("update_plan", serde_json::json!({}))]),
        );
        assert_eq!(
            analyse_trajectory(&json).unwrap().failure_stage,
            FailureStage::FileLocalization
        );
    }

    #[test]
    fn localized_but_unedited_is_line_localization() {
        let json = trajectory(
            "failure",
            serde_json::json!([turn("read_file", serde_json::json!({}))]),
        );
        assert_eq!(
            analyse_trajectory(&json).unwrap().failure_stage,
            FailureStage::LineLocalization
        );
    }

    #[test]
    fn edited_but_unverified_is_patch_implementation() {
        let json = trajectory(
            "failure",
            serde_json::json!([
                turn("read_file", serde_json::json!({})),
                turn("edit_file", serde_json::json!({})),
            ]),
        );
        assert_eq!(
            analyse_trajectory(&json).unwrap().failure_stage,
            FailureStage::PatchImplementation
        );
    }

    #[test]
    fn exhaustion_is_reported_for_terminations() {
        let json = trajectory("context_overflow", serde_json::json!([]));
        let analysis = analyse_trajectory(&json).unwrap();
        assert_eq!(analysis.failure_stage, FailureStage::Exhausted);
        assert_eq!(analysis.termination.as_deref(), Some("context_overflow"));
    }

    #[test]
    fn failed_tool_calls_are_counted() {
        let json = trajectory(
            "failure",
            serde_json::json!([{
                "turn_number": 0,
                "tool_calls": [{ "name": "read_file", "arguments": {} }],
                "tool_results": [{ "success": false }, { "success": true }],
            }]),
        );
        assert_eq!(analyse_trajectory(&json).unwrap().failed_tool_calls, 1);
    }

    #[test]
    fn missing_turns_yields_none() {
        let json = serde_json::json!({ "outcome": "success" });
        assert!(analyse_trajectory(&json).is_none());
    }
}
