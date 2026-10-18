//! Protocol type tests: parsing the documented shapes.

use super::*;

#[test]
fn initialize_params_parse_the_documented_shape() {
    let json = serde_json::json!({
        "protocolVersion": 1,
        "clientCapabilities": {
            "fs": { "readTextFile": true, "writeTextFile": false },
            "terminal": true
        },
        "clientInfo": { "name": "kerenzikov", "version": "1.0.0" }
    });
    let params: InitializeParams = serde_json::from_value(json).unwrap();
    assert_eq!(params.protocol_version, 1);
    assert!(params.client_capabilities.terminal);
    let fs = params.client_capabilities.fs.unwrap();
    assert!(fs.read_text_file);
    assert!(!fs.write_text_file);
    assert_eq!(params.client_info.unwrap().name, "kerenzikov");
}

#[test]
fn missing_client_capabilities_default_to_unsupported() {
    let params: InitializeParams =
        serde_json::from_value(serde_json::json!({ "protocolVersion": 1 })).unwrap();
    assert!(!params.client_capabilities.terminal);
    assert!(params.client_capabilities.fs.is_none());
}

#[test]
fn text_prompt_parses() {
    let params: PromptParams = serde_json::from_value(serde_json::json!({
        "sessionId": "sess_1",
        "prompt": [{ "type": "text", "text": "fix the bug" }]
    }))
    .unwrap();
    assert_eq!(params.prompt[0].as_text(), "fix the bug");
}

#[test]
fn resource_link_degrades_to_a_description() {
    let block: ContentBlock = serde_json::from_value(serde_json::json!({
        "type": "resource_link",
        "uri": "file:///a.rs",
        "name": "a.rs"
    }))
    .unwrap();
    let text = block.as_text();
    assert!(text.contains("a.rs"), "{text}");
    assert!(text.contains("file:///a.rs"), "{text}");
}

#[test]
fn embedded_resource_uses_its_text() {
    let block: ContentBlock = serde_json::from_value(serde_json::json!({
        "type": "resource",
        "resource": { "uri": "file:///x", "text": "contents here" }
    }))
    .unwrap();
    assert_eq!(block.as_text(), "contents here");
}

#[test]
fn image_is_reported_as_omitted_not_silently_dropped() {
    let block: ContentBlock = serde_json::from_value(serde_json::json!({
        "type": "image",
        "mimeType": "image/png"
    }))
    .unwrap();
    let text = block.as_text();
    assert!(text.contains("image omitted"), "{text}");
    // The mime type must survive: it is the only signal about what was dropped.
    assert!(text.contains("image/png"), "{text}");
}

#[test]
fn stdio_mcp_server_parses_without_a_type_field() {
    let server: McpServer = serde_json::from_value(serde_json::json!({
        "name": "fs",
        "command": "/bin/server",
        "args": ["--stdio"],
        "env": [{ "name": "K", "value": "V" }]
    }))
    .unwrap();
    match server {
        McpServer::Stdio { command, args, .. } => {
            assert_eq!(command, "/bin/server");
            assert_eq!(args.len(), 1);
        }
        _ => panic!("expected stdio"),
    }
}

#[test]
fn http_mcp_server_parses_with_a_type_field() {
    let server: McpServer = serde_json::from_value(serde_json::json!({
        "type": "http",
        "name": "remote",
        "url": "https://example.test/mcp",
        "headers": []
    }))
    .unwrap();
    assert!(matches!(server, McpServer::Remote { .. }));
}

#[test]
fn session_update_serializes_with_the_documented_discriminator() {
    let update = SessionUpdate::AgentMessageChunk {
        message_id: Some("m1".into()),
        content: OutContent::text("hello"),
    };
    let json = serde_json::to_value(update).unwrap();
    assert_eq!(json["sessionUpdate"], "agent_message_chunk");
    assert_eq!(json["content"]["type"], "text");
    assert_eq!(json["content"]["text"], "hello");
    assert_eq!(json["messageId"], "m1");
}

#[test]
fn tool_call_update_omits_unset_fields() {
    let update = SessionUpdate::ToolCallUpdate {
        tool_call_id: "c1".into(),
        status: Some(ToolStatus::Completed),
        content: None,
        raw_output: None,
    };
    let json = serde_json::to_value(update).unwrap();
    assert_eq!(json["sessionUpdate"], "tool_call_update");
    assert_eq!(json["status"], "completed");
    assert!(json.get("content").is_none());
}

#[test]
fn stop_reasons_serialize_in_snake_case() {
    let json = serde_json::to_value(StopReason::EndTurn).unwrap();
    assert_eq!(json, "end_turn");
    assert_eq!(
        serde_json::to_value(StopReason::MaxTurnRequests).unwrap(),
        "max_turn_requests"
    );
}

#[test]
fn permission_outcome_parses_both_shapes() {
    let allowed: RequestPermissionResult = serde_json::from_value(serde_json::json!({
        "outcome": { "outcome": "selected", "optionId": "allow-once" }
    }))
    .unwrap();
    assert!(allowed.outcome.allowed());

    let cancelled: RequestPermissionResult = serde_json::from_value(serde_json::json!({
        "outcome": { "outcome": "cancelled" }
    }))
    .unwrap();
    assert!(!cancelled.outcome.allowed());
}

#[test]
fn initialize_result_advertises_load_and_resume() {
    let result = InitializeResult {
        protocol_version: PROTOCOL_VERSION,
        agent_capabilities: AgentCapabilities {
            load_session: true,
            prompt_capabilities: PromptCapabilities {
                image: false,
                audio: false,
                embedded_context: true,
            },
            session_capabilities: SessionCapabilitiesOut {
                resume: serde_json::json!({}),
                close: serde_json::json!({}),
            },
        },
        agent_info: AgentInfo {
            name: "axolotl".into(),
            title: "Coding Harness".into(),
            version: "0.1.0".into(),
        },
        auth_methods: Vec::new(),
    };
    let json = serde_json::to_value(&result).unwrap();
    assert_eq!(json["protocolVersion"], 1);
    assert_eq!(json["agentCapabilities"]["loadSession"], true);
    assert!(json["agentCapabilities"]["sessionCapabilities"]["resume"].is_object());
    // The agent is text-only, and says so rather than accepting and failing.
    assert_eq!(
        json["agentCapabilities"]["promptCapabilities"]["image"],
        false
    );
}

#[test]
fn session_close_params_need_only_the_id() {
    // Regression: reusing `LoadSessionParams` here made a valid close request
    // fail with "missing field `cwd`".
    let params: CloseSessionParams =
        serde_json::from_value(serde_json::json!({ "sessionId": "sess_1" })).unwrap();
    assert_eq!(params.session_id, "sess_1");
}
