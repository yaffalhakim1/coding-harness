//! End-to-end ACP test: spawn the real binary and speak the protocol to it.
//!
//! Unit tests cannot catch a transport wired to nothing. This launches
//! `axolotl-acp`, performs the documented handshake, and asserts on the bytes
//! that come back — the same check Kerenzikov's own integration tests make.
//!
//! The prompt test is `#[ignore]`d because it needs a live model.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{json, Value};

/// A running agent, with its stdin and stdout.
struct Agent {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: i64,
}

impl Agent {
    fn spawn() -> Self {
        let exe = env!("CARGO_BIN_EXE_axolotl-acp");
        let mut child = Command::new(exe)
            .arg("--stdio")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawning axolotl-acp");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = BufReader::new(child.stdout.take().expect("stdout"));
        Self {
            child,
            stdin,
            stdout,
            next_id: 1,
        }
    }

    fn write(&mut self, value: &Value) {
        let line = serde_json::to_string(value).unwrap();
        writeln!(self.stdin, "{line}").unwrap();
        self.stdin.flush().unwrap();
    }

    /// Send a request and read until its response arrives.
    ///
    /// Notifications are collected and returned alongside, because a turn
    /// interleaves updates with its final response.
    fn request(&mut self, method: &str, params: Value) -> (Value, Vec<Value>) {
        let id = self.next_id;
        self.next_id += 1;
        self.write(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params
        }));

        let mut notifications = Vec::new();
        loop {
            let mut line = String::new();
            let n = self.stdout.read_line(&mut line).expect("read");
            assert!(
                n > 0,
                "agent closed the stream while waiting for '{method}'"
            );
            let value: Value = serde_json::from_str(line.trim()).expect("valid JSON line");
            if value.get("id").and_then(Value::as_i64) == Some(id) {
                return (value, notifications);
            }
            if value.get("method").is_some() {
                notifications.push(value);
            }
        }
    }

    fn notify(&mut self, method: &str, params: Value) {
        self.write(&json!({ "jsonrpc": "2.0", "method": method, "params": params }));
    }
}

impl Drop for Agent {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn handshake_advertises_v1_and_the_implemented_capabilities() {
    let mut agent = Agent::spawn();
    let (response, _) = agent.request(
        "initialize",
        json!({
            "protocolVersion": 1,
            "clientCapabilities": { "fs": { "readTextFile": true, "writeTextFile": false } },
            "clientInfo": { "name": "acp-test", "version": "0.1.0" }
        }),
    );

    let result = &response["result"];
    assert_eq!(result["protocolVersion"], 1, "{response}");
    assert_eq!(result["agentCapabilities"]["loadSession"], true);
    assert!(result["agentCapabilities"]["sessionCapabilities"]["resume"].is_object());
    assert_eq!(result["agentInfo"]["name"], "axolotl-acp");
}

#[test]
fn session_new_returns_a_session_id() {
    let mut agent = Agent::spawn();
    agent.request("initialize", json!({ "protocolVersion": 1 }));

    let dir = tempfile::tempdir().unwrap();
    let (response, _) = agent.request(
        "session/new",
        json!({
            "cwd": dir.path().to_string_lossy(),
            "mcpServers": []
        }),
    );

    let id = response["result"]["sessionId"].as_str().expect("sessionId");
    assert!(id.starts_with("sess_"), "{id}");
}

#[test]
fn session_new_rejects_a_relative_cwd() {
    let mut agent = Agent::spawn();
    agent.request("initialize", json!({ "protocolVersion": 1 }));

    let (response, _) = agent.request(
        "session/new",
        json!({ "cwd": "relative", "mcpServers": [] }),
    );
    // A protocol error, not a crash: the connection must stay usable.
    assert!(response["error"].is_object(), "{response}");
    assert_eq!(response["error"]["code"], -32602);

    let (second, _) = agent.request("session/new", json!({ "cwd": "also-relative" }));
    assert!(
        second["error"].is_object(),
        "connection died after an error"
    );
}

#[test]
fn an_unknown_method_returns_method_not_found() {
    let mut agent = Agent::spawn();
    agent.request("initialize", json!({ "protocolVersion": 1 }));

    let (response, _) = agent.request("bogus/method", json!({}));
    assert_eq!(response["error"]["code"], -32601, "{response}");
}

#[test]
fn session_close_removes_the_session() {
    let mut agent = Agent::spawn();
    agent.request("initialize", json!({ "protocolVersion": 1 }));

    let dir = tempfile::tempdir().unwrap();
    let (created, _) = agent.request(
        "session/new",
        json!({ "cwd": dir.path().to_string_lossy() }),
    );
    let id = created["result"]["sessionId"].as_str().unwrap().to_string();

    let (closed, _) = agent.request("session/close", json!({ "sessionId": id }));
    assert!(closed["result"].is_object(), "{closed}");

    // Closing twice is an error, because the session is gone.
    let (again, _) = agent.request("session/close", json!({ "sessionId": id }));
    assert!(again["error"].is_object(), "{again}");
}

#[test]
fn session_resume_of_an_unknown_session_is_an_error() {
    let mut agent = Agent::spawn();
    agent.request("initialize", json!({ "protocolVersion": 1 }));

    let dir = tempfile::tempdir().unwrap();
    let (response, _) = agent.request(
        "session/resume",
        json!({ "sessionId": "sess_nope", "cwd": dir.path().to_string_lossy() }),
    );
    assert!(response["error"].is_object(), "{response}");
}

#[test]
fn a_malformed_line_does_not_kill_the_connection() {
    let mut agent = Agent::spawn();
    agent.request("initialize", json!({ "protocolVersion": 1 }));

    // Write garbage, then a valid request: the agent must survive.
    writeln!(agent.stdin, "this is not json").unwrap();
    agent.stdin.flush().unwrap();

    let (response, _) = agent.request("initialize", json!({ "protocolVersion": 1 }));
    assert_eq!(response["result"]["protocolVersion"], 1, "{response}");
}

#[test]
fn cancel_for_an_unknown_session_is_accepted() {
    let mut agent = Agent::spawn();
    agent.request("initialize", json!({ "protocolVersion": 1 }));

    // A cancel racing a close must not error: the client cannot know.
    agent.notify("session/cancel", json!({ "sessionId": "sess_gone" }));

    let (response, _) = agent.request("initialize", json!({ "protocolVersion": 1 }));
    assert!(response["result"].is_object(), "{response}");
}

/// Drive a real prompt turn. Ignored: needs a live model and spends tokens.
#[test]
#[ignore = "requires AXOLOTL_API_KEY_ENV to resolve and spends real tokens"]
fn prompt_turn_streams_updates_and_ends_with_a_stop_reason() {
    let mut agent = Agent::spawn();
    agent.request("initialize", json!({ "protocolVersion": 1 }));

    let dir = tempfile::tempdir().unwrap();
    let (created, _) = agent.request(
        "session/new",
        json!({ "cwd": dir.path().to_string_lossy() }),
    );
    let id = created["result"]["sessionId"].as_str().unwrap().to_string();

    let (response, notifications) = agent.request(
        "session/prompt",
        json!({
            "sessionId": id,
            "prompt": [{ "type": "text", "text": "Create p.txt containing ok, then verify it." }]
        }),
    );

    assert_eq!(
        response["result"]["stopReason"], "end_turn",
        "turn did not end cleanly: {response}"
    );

    // The turn must have reported at least one tool call and some text.
    let kinds: Vec<String> = notifications
        .iter()
        .filter_map(|n| n["params"]["update"]["sessionUpdate"].as_str())
        .map(str::to_string)
        .collect();
    assert!(
        kinds.iter().any(|k| k == "tool_call"),
        "no tool_call update in {kinds:?}"
    );
    assert!(
        kinds.iter().any(|k| k == "agent_message_chunk"),
        "no agent text in {kinds:?}"
    );
    assert!(
        dir.path().join("p.txt").exists(),
        "the agent did not create the file"
    );
}
