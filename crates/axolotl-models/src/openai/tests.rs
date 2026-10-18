//! Offline tests for the OpenAI adapter: wire serialization and response
//! decoding. No network access.

use super::*;
use crate::types::{Role, ToolDefinition};

#[test]
fn request_uses_wire_tool_call_shape() {
    let call = ToolCall {
        id: "call_1".into(),
        name: "read_file".into(),
        arguments: serde_json::json!({ "path": "a.txt" }),
    };
    let messages = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("go"),
        ChatMessage::assistant_with_tool_calls(None, vec![call]),
        ChatMessage::tool_result("call_1", "contents"),
    ];
    let request = ChatRequest::from_internal(&ModelConfig::default(), &messages, &[]);
    let json = serde_json::to_value(&request).unwrap();

    let assistant = &json["messages"][2];
    let call = &assistant["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "read_file");
    // arguments must be a JSON *string*, not an object
    assert!(call["function"]["arguments"].is_string());
    // a tool-call-only assistant message must not emit null content
    assert_eq!(assistant["content"], "");

    let tool = &json["messages"][3];
    assert_eq!(tool["role"], "tool");
    assert_eq!(tool["tool_call_id"], "call_1");
}

#[test]
fn empty_tools_are_omitted() {
    let messages = vec![ChatMessage::user("hi")];
    let request = ChatRequest::from_internal(&ModelConfig::default(), &messages, &[]);
    let json = serde_json::to_value(&request).unwrap();
    assert!(
        json.get("tools").is_none(),
        "empty tool list must be omitted"
    );
}

#[test]
fn tools_are_wrapped_as_functions() {
    let tools = vec![ToolDefinition {
        name: "bash".into(),
        description: "run".into(),
        parameters: serde_json::json!({ "type": "object" }),
    }];
    let request = ChatRequest::from_internal(&ModelConfig::default(), &[], &tools);
    let json = serde_json::to_value(&request).unwrap();
    assert_eq!(json["tools"][0]["type"], "function");
    assert_eq!(json["tools"][0]["function"]["name"], "bash");
}

#[test]
fn decodes_tool_call_response() {
    let body = r#"{
        "model": "gpt-4o",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": { "name": "bash", "arguments": "{\"command\":\"ls\"}" }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
    }"#;
    let completion: ChatCompletionResponse = serde_json::from_str(body).unwrap();
    let response = to_llm_response(completion).unwrap();

    assert_eq!(response.stop_reason, StopReason::ToolUse);
    assert_eq!(response.model, "gpt-4o");
    assert_eq!(response.usage.total_tokens, 15);
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].arguments["command"], "ls");
}

#[test]
fn decodes_plain_text_response_without_usage() {
    let body = r#"{
        "choices": [{ "message": { "role": "assistant", "content": "done" }, "finish_reason": "stop" }]
    }"#;
    let completion: ChatCompletionResponse = serde_json::from_str(body).unwrap();
    let response = to_llm_response(completion).unwrap();
    assert_eq!(response.message.role, Role::Assistant);
    assert_eq!(response.message.text(), "done");
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert!(response.tool_calls().is_empty());
    assert_eq!(response.usage, TokenUsage::default());
}

#[test]
fn empty_choices_is_an_error() {
    let completion: ChatCompletionResponse = serde_json::from_str(r#"{"choices": []}"#).unwrap();
    assert!(matches!(
        to_llm_response(completion),
        Err(ProviderError::EmptyResponse)
    ));
}

#[test]
fn malformed_arguments_are_surfaced_not_dropped() {
    let body = r#"{
        "choices": [{
            "message": {
                "tool_calls": [{ "id": "c1", "function": { "name": "bash", "arguments": "{not json" } }]
            },
            "finish_reason": "tool_calls"
        }]
    }"#;
    let completion: ChatCompletionResponse = serde_json::from_str(body).unwrap();
    let response = to_llm_response(completion).unwrap();
    let args = &response.tool_calls()[0].arguments;
    assert!(args["__parse_error"].is_string());
    assert_eq!(args["__raw"], "{not json");
}

#[test]
fn finish_reason_mapping() {
    assert_eq!(map_finish_reason(Some("stop")), StopReason::EndTurn);
    assert_eq!(map_finish_reason(Some("tool_calls")), StopReason::ToolUse);
    assert_eq!(map_finish_reason(Some("length")), StopReason::MaxTokens);
    assert_eq!(
        map_finish_reason(Some("content_filter")),
        StopReason::Other("content_filter".into())
    );
    assert_eq!(map_finish_reason(None), StopReason::EndTurn);
}

#[test]
fn status_mapping_drives_retry_decisions() {
    let rate = map_status(StatusCode::TOO_MANY_REQUESTS, "slow down".into());
    assert!(rate.is_retryable());

    let server = map_status(StatusCode::BAD_GATEWAY, String::new());
    assert!(server.is_retryable());

    let auth = map_status(StatusCode::UNAUTHORIZED, String::new());
    assert!(!auth.is_retryable());

    let bad = map_status(StatusCode::BAD_REQUEST, String::new());
    assert!(!bad.is_retryable());
}

#[test]
fn truncate_respects_char_boundaries() {
    let s = "αααααααααα";
    assert_eq!(truncate(s, 3), "ααα…");
    assert_eq!(truncate("abc", 10), "abc");
}
