//! End-to-end MCP test against a real stdio server.
//!
//! Spawns a tiny JSON-RPC server as a child process and drives the full
//! handshake, tool listing and tool call through the client. This is the only
//! way to catch framing bugs: unit tests on `unwrap_result` prove nothing about
//! whether the pipe actually carries a well-formed exchange.

use std::io::{BufRead, Write};

use harness_actions::mcp::{McpClient, ServerConfig, Transport};

/// Write a small MCP server to a temp directory and return its path.
///
/// Python is used because it is present on every platform this harness runs on
/// and needs no build step, unlike compiling a helper binary per test run.
fn write_server(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("fake_mcp_server.py");
    std::fs::write(
        &path,
        r#"
import json
import sys

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue

    method = msg.get("method")
    mid = msg.get("id")

    # Notifications carry no id and expect no reply.
    if mid is None:
        continue

    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1.0"}
        }})
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
            {
                "name": "echo",
                "description": "Echo the input back",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }
            },
            {
                "name": "fail",
                "description": "Always fails",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]}})
    elif method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "echo":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": "echo:" + str(args.get("text", ""))}],
                "isError": False
            }})
        elif name == "fail":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": "boom"}],
                "isError": True
            }})
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32601, "message": "unknown tool " + str(name)
            }})
    else:
        send({"jsonrpc": "2.0", "id": mid, "error": {
            "code": -32601, "message": "unknown method " + str(method)
        }})
"#,
    )
    .unwrap();
    path
}

fn python_command() -> Option<String> {
    for candidate in ["python", "python3"] {
        let ok = std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some(candidate.to_string());
        }
    }
    None
}

fn server_config(script: &std::path::Path) -> Option<ServerConfig> {
    let python = python_command()?;
    Some(ServerConfig {
        name: "fake".into(),
        transport: Transport::Stdio {
            command: python,
            args: vec![script.to_string_lossy().to_string()],
            env: Default::default(),
        },
    })
}

#[tokio::test]
async fn connects_lists_tools_and_calls_one() {
    let dir = tempfile::tempdir().unwrap();
    let script = write_server(dir.path());
    let Some(config) = server_config(&script) else {
        eprintln!("python unavailable; skipping MCP end-to-end test");
        return;
    };

    let mut client = McpClient::connect(&config)
        .await
        .expect("handshake should succeed");

    let tools = client.tools();
    assert_eq!(tools.len(), 2, "expected the two advertised tools");
    let echo = tools.iter().find(|t| t.name == "echo").expect("echo tool");
    assert_eq!(echo.description, "Echo the input back");
    assert_eq!(echo.input_schema["type"], "object");

    let out = client
        .call_tool("echo", serde_json::json!({ "text": "hello" }))
        .await
        .expect("tool call should succeed");
    assert_eq!(out, "echo:hello");

    client.shutdown().await;
}

#[tokio::test]
async fn a_tool_level_error_is_reported_as_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let script = write_server(dir.path());
    let Some(config) = server_config(&script) else {
        eprintln!("python unavailable; skipping");
        return;
    };

    let mut client = McpClient::connect(&config).await.unwrap();
    let err = client
        .call_tool("fail", serde_json::json!({}))
        .await
        .unwrap_err();
    // `isError: true` in the result must surface as a failure, not text.
    assert!(err.to_string().contains("boom"), "{err}");
    client.shutdown().await;
}

#[tokio::test]
async fn an_unknown_tool_reports_the_jsonrpc_error() {
    let dir = tempfile::tempdir().unwrap();
    let script = write_server(dir.path());
    let Some(config) = server_config(&script) else {
        eprintln!("python unavailable; skipping");
        return;
    };

    let mut client = McpClient::connect(&config).await.unwrap();
    let err = client
        .call_tool("nope", serde_json::json!({}))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("unknown tool"), "{err}");
    client.shutdown().await;
}

#[tokio::test]
async fn a_missing_command_fails_cleanly() {
    let config = ServerConfig {
        name: "broken".into(),
        transport: Transport::Stdio {
            command: "definitely-not-a-real-binary-xyz".into(),
            args: vec![],
            env: Default::default(),
        },
    };
    let err = McpClient::connect(&config)
        .await
        .err()
        .expect("connect must fail for a missing binary");
    assert!(err.to_string().contains("spawning"), "{err}");
}

#[test]
fn fake_server_script_is_valid_python() {
    // Guard: if the fixture is broken, every test above silently skips.
    let dir = tempfile::tempdir().unwrap();
    let script = write_server(dir.path());
    let Some(python) = python_command() else {
        eprintln!("python unavailable; skipping");
        return;
    };
    let status = std::process::Command::new(python)
        .args(["-c", "import ast,sys; ast.parse(open(sys.argv[1]).read())"])
        .arg(&script)
        .status()
        .unwrap();
    assert!(status.success(), "fixture has a syntax error");
}

/// The fixture itself writes and reads line-delimited JSON, so confirm the
/// framing assumption holds for a plain pipe before blaming the client.
#[test]
fn line_delimited_json_round_trips() {
    let mut buf: Vec<u8> = Vec::new();
    writeln!(buf, "{}", serde_json::json!({"id": 1})).unwrap();
    let mut reader = std::io::BufReader::new(&buf[..]);
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
    assert_eq!(parsed["id"], 1);
}
