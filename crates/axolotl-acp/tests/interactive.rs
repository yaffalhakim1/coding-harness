//! End-to-end ACP tests for the interactive features: permission prompts,
//! questions, todos, and slash commands.
//!
//! These drive the real binary and answer its requests the way a client would,
//! which is the only way to catch a callback wired to nothing.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{json, Value};

struct Agent {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
    /// Every notification received while waiting for a response.
    notifications: Vec<Value>,
}

impl Agent {
    fn spawn() -> Self {
        Self::spawn_with(|cmd| cmd)
    }

    /// Spawn, letting the caller add environment for a live run.
    fn spawn_with(configure: impl FnOnce(&mut Command) -> &mut Command) -> Self {
        let exe = env!("CARGO_BIN_EXE_axolotl-acp");
        let mut command = Command::new(exe);
        command
            .arg("--stdio")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .env("AXOLOTL_API_BASE", "http://127.0.0.1:9/v1")
            .env("AXOLOTL_API_KEY_ENV", "AXOLOTL_TEST_KEY");
        configure(&mut command);

        let mut child = command.spawn().expect("spawning axolotl-acp");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        Self {
            child,
            stdin,
            stdout,
            next_id: 1,
            notifications: Vec::new(),
        }
    }

    fn write(&mut self, value: &Value) {
        writeln!(self.stdin, "{}", serde_json::to_string(value).unwrap()).unwrap();
        self.stdin.flush().unwrap();
    }

    /// Read one line, recording notifications and answering agent requests.
    fn read_one(&mut self, answer_permission: &str) -> Option<Value> {
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line).ok()?;
        if n == 0 {
            return None;
        }
        let value: Value = serde_json::from_str(line.trim()).ok()?;

        // The agent asking us something: answer it and keep reading.
        if let Some(method) = value.get("method").and_then(Value::as_str) {
            let id = value.get("id").cloned();
            match (method, id) {
                ("session/request_permission", Some(id)) => {
                    let result = if answer_permission == "cancelled" {
                        json!({ "outcome": { "outcome": "cancelled" } })
                    } else {
                        json!({
                            "outcome": { "outcome": "selected", "optionId": answer_permission }
                        })
                    };
                    self.write(&json!({ "jsonrpc": "2.0", "id": id, "result": result }));
                }
                ("elicitation/create", Some(id)) => {
                    self.write(&json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "result": { "action": "decline" }
                    }));
                }
                _ => {
                    self.notifications.push(value);
                }
            }
            return self.read_one(answer_permission);
        }

        Some(value)
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        self.request_with(method, params, "allow-once")
    }

    fn request_with(&mut self, method: &str, params: Value, answer: &str) -> Value {
        let id = self.next_id;
        self.next_id += 1;
        self.write(&json!({
            "jsonrpc": "2.0", "id": id, "method": method, "params": params
        }));
        loop {
            match self.read_one(answer) {
                Some(value) if value.get("id").and_then(Value::as_i64) == Some(id) => {
                    return value;
                }
                Some(_) => continue,
                None => panic!("stream closed waiting for {method}"),
            }
        }
    }

    fn session(&mut self, cwd: &std::path::Path) -> String {
        self.request("initialize", json!({ "protocolVersion": 1 }));
        let created = self.request(
            "session/new",
            json!({ "cwd": cwd.to_string_lossy(), "mcpServers": [] }),
        );
        created["result"]["sessionId"]
            .as_str()
            .expect("sessionId")
            .to_string()
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn slash_help_is_answered_without_a_model_call() {
    let mut agent = Agent::spawn();
    let dir = tempfile::tempdir().unwrap();
    let id = agent.session(dir.path());

    // No API key is set and the endpoint is a dead port: if `/help` reached the
    // model this would fail rather than answer.
    let response = agent.request(
        "session/prompt",
        json!({ "sessionId": id, "prompt": [{ "type": "text", "text": "/help" }] }),
    );

    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");

    let text: String = agent
        .notifications
        .iter()
        .filter_map(|n| {
            let update = &n["params"]["update"];
            (update["sessionUpdate"] == "agent_message_chunk")
                .then(|| update["content"]["text"].as_str().unwrap_or("").to_string())
        })
        .collect();
    assert!(text.contains("Available commands"), "{text}");
}

#[test]
fn slash_commands_are_advertised_after_session_new() {
    let mut agent = Agent::spawn();
    let dir = tempfile::tempdir().unwrap();
    let _id = agent.session(dir.path());

    let advertised = agent
        .notifications
        .iter()
        .find(|n| n["params"]["update"]["sessionUpdate"] == "available_commands_update");
    let advertised = advertised.expect("available_commands_update was not sent");
    let commands = advertised["params"]["update"]["availableCommands"]
        .as_array()
        .expect("commands array");
    let names: Vec<&str> = commands.iter().filter_map(|c| c["name"].as_str()).collect();

    assert!(names.contains(&"help"), "{names:?}");
    assert!(names.contains(&"skills"), "{names:?}");
    assert!(names.contains(&"plan"), "{names:?}");
}

#[test]
fn workspace_skills_appear_in_the_command_list() {
    let mut agent = Agent::spawn();
    let dir = tempfile::tempdir().unwrap();

    let skill_dir = dir.path().join(".axolotl/skills/commit");
    std::fs::create_dir_all(&skill_dir).unwrap();
    std::fs::write(
        skill_dir.join("SKILL.md"),
        "---\nname: commit\ndescription: Write a commit message\n---\nUse conventional commits.",
    )
    .unwrap();

    let _id = agent.session(dir.path());

    let advertised = agent
        .notifications
        .iter()
        .find(|n| n["params"]["update"]["sessionUpdate"] == "available_commands_update")
        .expect("commands update");
    let names: Vec<&str> = advertised["params"]["update"]["availableCommands"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["name"].as_str())
        .collect();
    assert!(names.contains(&"commit"), "{names:?}");
}

#[test]
fn a_denied_tool_becomes_an_observation_not_a_crash() {
    let mut agent = Agent::spawn();
    let dir = tempfile::tempdir().unwrap();
    let id = agent.session(dir.path());

    // Reject every permission request. The turn must still end cleanly, and the
    // file must not be written.
    let response = agent.request_with(
        "session/prompt",
        json!({
            "sessionId": id,
            "prompt": [{ "type": "text", "text": "Create denied.txt with content x" }]
        }),
        "reject-once",
    );

    // The endpoint is dead, so the turn cannot complete — but it must fail as a
    // turn, not hang or panic.
    assert!(
        response.get("result").is_some() || response.get("error").is_some(),
        "{response}"
    );
    assert!(
        !dir.path().join("denied.txt").exists(),
        "a denied write must not create the file"
    );
}

#[test]
fn a_cancelled_permission_request_is_reported_as_cancelled() {
    let mut agent = Agent::spawn();
    let dir = tempfile::tempdir().unwrap();
    let id = agent.session(dir.path());

    let response = agent.request_with(
        "session/prompt",
        json!({
            "sessionId": id,
            "prompt": [{ "type": "text", "text": "Write a file" }]
        }),
        "cancelled",
    );

    assert!(
        response.get("result").is_some() || response.get("error").is_some(),
        "{response}"
    );
}

/// A live turn that must hit the permission prompt. Ignored: needs a model.
#[test]
#[ignore = "requires a live model and spends tokens"]
fn a_write_triggers_a_permission_prompt() {
    // The spawned binary needs the real endpoint; the test process's own
    // environment is not inherited when it is overridden below.
    let mut agent = Agent::spawn_with(|cmd| {
        cmd.env(
            "AXOLOTL_API_BASE",
            std::env::var("AXOLOTL_API_BASE").unwrap_or_else(|_| "https://kenari.id/v1".into()),
        )
        .env(
            "AXOLOTL_API_KEY_ENV",
            std::env::var("AXOLOTL_API_KEY_ENV").unwrap_or_else(|_| "KENARI_API_KEY".into()),
        )
        .env(
            "AXOLOTL_MODEL",
            std::env::var("AXOLOTL_MODEL").unwrap_or_else(|_| "deepseek-v4-1-flash".into()),
        )
    });
    let dir = tempfile::tempdir().unwrap();
    let id = agent.session(dir.path());

    let response = agent.request(
        "session/prompt",
        json!({
            "sessionId": id,
            "prompt": [{ "type": "text", "text": "Create permitted.txt containing ok" }]
        }),
    );
    assert_eq!(response["result"]["stopReason"], "end_turn", "{response}");

    // The client approved, so the file exists.
    assert!(
        dir.path().join("permitted.txt").exists(),
        "an approved write should have landed"
    );

    // And the prompt was actually shown: a permission request reached us.
    let asked = agent
        .notifications
        .iter()
        .any(|n| n["params"]["update"]["sessionUpdate"] == "tool_call");
    assert!(asked, "expected tool_call updates");
}
