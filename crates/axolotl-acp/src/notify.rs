//! Notifications the agent pushes outside a turn.
//!
//! Kept separate from the prompt path: these run on session creation and on
//! resume, and a client that renders them must not have to care how a turn is
//! driven.

use crate::rpc::Writer;

/// Advertise the slash commands available in a workspace.
///
/// Sent after `session/new` because the catalogue includes the workspace's
/// skills, and the cwd is only known then.
pub async fn emit_commands(writer: &Writer, session_id: &str, cwd: &std::path::Path) {
    let skills = axolotl_actions::skills::discover(cwd);
    let commands: Vec<serde_json::Value> = axolotl_actions::slash::catalogue(&skills)
        .into_iter()
        .map(|command| {
            let mut entry = serde_json::json!({
                "name": command.name,
                "description": command.description,
            });
            if let Some(hint) = command.input_hint {
                entry["input"] = serde_json::json!({ "hint": hint });
            }
            entry
        })
        .collect();

    let params = serde_json::json!({
        "sessionId": session_id,
        "update": {
            "sessionUpdate": "available_commands_update",
            "availableCommands": commands
        }
    });
    let _ = writer.notify("session/update", params).await;
}

/// Emit the plan as an ACP `plan` update.
pub async fn emit_plan(writer: &Writer, session_id: &str, steps: &[String]) {
    let entries: Vec<serde_json::Value> = steps
        .iter()
        .map(|step| {
            serde_json::json!({
                "content": step,
                "priority": "medium",
                "status": "pending"
            })
        })
        .collect();
    let params = serde_json::json!({
        "sessionId": session_id,
        "update": { "sessionUpdate": "plan", "entries": entries }
    });
    let _ = writer.notify("session/update", params).await;
}

/// Replay history for `session/load`.
///
/// The protocol requires the full conversation as `session/update`
/// notifications before the response, so a client that reconnects rebuilds its
/// transcript.
pub async fn replay_history(
    writer: &Writer,
    session_id: &str,
    history: &[axolotl_models::types::ChatMessage],
) {
    for message in history {
        let (kind, text) = match message.role {
            axolotl_models::types::Role::User => ("user_message_chunk", message.text()),
            axolotl_models::types::Role::Assistant => ("agent_message_chunk", message.text()),
            // System and tool messages are not part of the visible transcript.
            _ => continue,
        };
        if text.trim().is_empty() {
            continue;
        }
        let params = serde_json::json!({
            "sessionId": session_id,
            "update": {
                "sessionUpdate": kind,
                "content": { "type": "text", "text": text }
            }
        });
        let _ = writer.notify("session/update", params).await;
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn the_catalogue_includes_builtins_and_skills() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join(".axolotl/skills/commit");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: commit\ndescription: Write a commit message\n---\nBody",
        )
        .unwrap();

        let skills = axolotl_actions::skills::discover(dir.path());
        let commands = axolotl_actions::slash::catalogue(&skills);
        let names: Vec<&str> = commands.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"help"));
        assert!(names.contains(&"commit"));
    }
}
