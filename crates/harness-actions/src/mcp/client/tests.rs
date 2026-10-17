//! Tests for the MCP client's protocol handling.

use super::*;

#[test]
fn unwraps_a_successful_result() {
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": { "tools": [] }
    });
    let result = unwrap_result(response).unwrap();
    assert!(result["tools"].is_array());
}

#[test]
fn maps_a_jsonrpc_error() {
    let response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "error": { "code": -32601, "message": "Method not found" }
    });
    let err = unwrap_result(response).unwrap_err().to_string();
    assert!(err.contains("-32601"), "{err}");
    assert!(err.contains("Method not found"), "{err}");
}

#[test]
fn renders_text_content_blocks() {
    let result = serde_json::json!({
        "content": [
            { "type": "text", "text": "line one" },
            { "type": "text", "text": "line two" }
        ]
    });
    assert_eq!(render_content(&result), "line one\nline two");
}

#[test]
fn renders_non_text_blocks_as_placeholders() {
    let result = serde_json::json!({
        "content": [{ "type": "image", "data": "..." }]
    });
    assert!(render_content(&result).contains("image content omitted"));
}

#[test]
fn renders_a_result_without_content_as_json() {
    let result = serde_json::json!({ "value": 7 });
    assert!(render_content(&result).contains("\"value\""));
}

#[test]
fn parses_tool_definitions() {
    let tool: McpTool = serde_json::from_value(serde_json::json!({
        "name": "search",
        "description": "Search the web",
        "inputSchema": { "type": "object" }
    }))
    .unwrap();
    assert_eq!(tool.name, "search");
    assert_eq!(tool.description, "Search the web");
    assert_eq!(tool.input_schema["type"], "object");
}

#[test]
fn tool_without_schema_defaults_to_null() {
    let tool: McpTool = serde_json::from_value(serde_json::json!({ "name": "bare" })).unwrap();
    assert!(tool.input_schema.is_null());
}

#[test]
fn transport_parses_from_config_shape() {
    let stdio: ServerConfig = serde_json::from_value(serde_json::json!({
        "name": "fs",
        "type": "stdio",
        "command": "npx",
        "args": ["-y", "server-filesystem"]
    }))
    .unwrap();
    match stdio.transport {
        Transport::Stdio { command, args, .. } => {
            assert_eq!(command, "npx");
            assert_eq!(args.len(), 2);
        }
        _ => panic!("expected stdio"),
    }

    let http: ServerConfig = serde_json::from_value(serde_json::json!({
        "name": "remote",
        "type": "http",
        "url": "https://example.test/mcp"
    }))
    .unwrap();
    assert!(matches!(http.transport, Transport::Http { .. }));
}

#[test]
fn a_result_that_is_not_an_object_is_returned_as_null() {
    let response = serde_json::json!({ "jsonrpc": "2.0", "id": 1 });
    assert!(unwrap_result(response).unwrap().is_null());
}
