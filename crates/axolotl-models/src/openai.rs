//! OpenAI-compatible provider adapter.
//!
//! Works with OpenAI, Azure OpenAI, and any OpenAI-compatible API
//! (e.g., local models via ollama, vllm, lmstudio).

use std::time::Duration;

use async_trait::async_trait;
use reqwest::StatusCode;
use tracing::{debug, warn};

use crate::provider::{LLMProvider, ProviderError, ProviderResult, RetryPolicy};
use crate::types::*;

use wire::{ChatCompletionResponse, ChatRequest};

/// Default endpoint used when the config does not override it.
const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAIProvider {
    client: reqwest::Client,
    default_base_url: String,
    retry_policy: RetryPolicy,
}

impl OpenAIProvider {
    pub fn new() -> ProviderResult<Self> {
        Self::with_base_url(OPENAI_BASE_URL)
    }

    pub fn with_base_url(base_url: impl Into<String>) -> ProviderResult<Self> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| ProviderError::Config(format!("failed to build HTTP client: {e}")))?;
        Ok(Self {
            client,
            default_base_url: base_url.into(),
            retry_policy: RetryPolicy::default(),
        })
    }

    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    fn resolve_api_key(&self, config: &ModelConfig) -> ProviderResult<String> {
        let var = config
            .api_key_env
            .as_deref()
            .ok_or_else(|| ProviderError::Config("no api_key_env configured".into()))?;
        std::env::var(var).map_err(|_| ProviderError::MissingApiKey(var.to_string()))
    }

    fn resolve_base_url(&self, config: &ModelConfig) -> String {
        config
            .api_base
            .clone()
            .unwrap_or_else(|| self.default_base_url.clone())
    }

    /// Perform one HTTP attempt, mapping the response into typed errors.
    async fn attempt(
        &self,
        url: &str,
        api_key: &str,
        request: &ChatRequest,
        timeout: Duration,
    ) -> ProviderResult<ChatCompletionResponse> {
        let response = self
            .client
            .post(url)
            .timeout(timeout)
            .bearer_auth(api_key)
            .json(request)
            .send()
            .await
            .map_err(|e| ProviderError::Transport(e.to_string()))?;

        let status = response.status();
        let body = response.text().await.unwrap_or_default();

        if status.is_success() {
            return serde_json::from_str(&body).map_err(|e| {
                ProviderError::Decode(format!("{e}; body: {}", truncate(&body, 500)))
            });
        }

        Err(map_status(status, body))
    }
}

/// Map a non-2xx HTTP response into a typed error.
fn map_status(status: StatusCode, body: String) -> ProviderError {
    let code = status.as_u16();
    match code {
        401 | 403 => ProviderError::Auth { status: code, body },
        429 => ProviderError::RateLimited { status: code, body },
        408 | 409 | 500..=599 => ProviderError::Server { status: code, body },
        _ => ProviderError::BadRequest { status: code, body },
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect::<String>() + "…"
}

#[async_trait]
impl LLMProvider for OpenAIProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        config: &ModelConfig,
    ) -> ProviderResult<LLMResponse> {
        let api_key = self.resolve_api_key(config)?;
        let base = self.resolve_base_url(config);
        let url = format!("{}/chat/completions", base.trim_end_matches('/'));
        let timeout = Duration::from_secs(config.timeout_secs.max(1));

        let request = ChatRequest::from_internal(config, messages, tools);

        debug!(
            model = %config.model_id,
            messages = messages.len(),
            tools = tools.len(),
            "Sending chat request"
        );

        let mut last_error: Option<ProviderError> = None;

        for attempt in 0..=config.max_retries {
            if attempt > 0 {
                let delay = self.retry_policy.delay(attempt - 1);
                debug!(
                    attempt,
                    delay_ms = delay.as_millis() as u64,
                    "Retrying LLM request"
                );
                tokio::time::sleep(delay).await;
            }

            match self.attempt(&url, &api_key, &request, timeout).await {
                Ok(completion) => return to_llm_response(completion),
                Err(err) if err.is_retryable() => {
                    warn!(attempt, error = %err, "Retryable LLM API error");
                    last_error = Some(err);
                }
                Err(err) => return Err(err),
            }
        }

        Err(ProviderError::RetriesExhausted {
            attempts: config.max_retries + 1,
            source: Box::new(last_error.unwrap_or(ProviderError::EmptyResponse)),
        })
    }

    fn name(&self) -> &str {
        "openai"
    }

    async fn health_check(&self, config: &ModelConfig) -> ProviderResult<bool> {
        let test_messages = vec![ChatMessage::user("ping")];
        match self.chat(&test_messages, &[], config).await {
            Ok(_) => Ok(true),
            Err(e) => {
                warn!(error = %e, "Health check failed");
                Ok(false)
            }
        }
    }
}

/// Convert the wire response into the harness response type.
fn to_llm_response(completion: ChatCompletionResponse) -> ProviderResult<LLMResponse> {
    wire::convert::into_llm_response(completion, parse_arguments, map_finish_reason)
        .ok_or(ProviderError::EmptyResponse)
}

/// Parse tool-call arguments.
///
/// A malformed fragment is wrapped rather than turned into `Null` so the tool
/// layer can reject it with a message the model can act on. Silently returning
/// `Null` would make the tool report "missing argument", hiding the real cause.
fn parse_arguments(raw: &str) -> serde_json::Value {
    if raw.trim().is_empty() {
        return serde_json::Value::Object(serde_json::Map::new());
    }
    serde_json::from_str(raw)
        .unwrap_or_else(|e| serde_json::json!({ "__parse_error": e.to_string(), "__raw": raw }))
}

fn map_finish_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("stop") => StopReason::EndTurn,
        Some("tool_calls") | Some("function_call") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("content_filter") => StopReason::Other("content_filter".into()),
        Some(other) => StopReason::Other(other.to_string()),
        None => StopReason::EndTurn,
    }
}

mod wire {
    //! OpenAI wire format. Kept separate from `types` because the protocol
    //! shape differs from the in-memory shape: tool calls carry a stringified
    //! `arguments` payload nested inside a `function` object, and the `tool`
    //! role requires `tool_call_id`.

    use serde::{Deserialize, Serialize};

    use crate::types::{ChatMessage, ModelConfig, Role, ToolDefinition};

    #[derive(Debug, Serialize)]
    pub struct ChatRequest {
        pub model: String,
        pub messages: Vec<WireMessage>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub tools: Vec<WireTool>,
        pub temperature: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tokens: Option<usize>,
    }

    impl ChatRequest {
        pub fn from_internal(
            config: &ModelConfig,
            messages: &[ChatMessage],
            tools: &[ToolDefinition],
        ) -> Self {
            Self {
                model: config.model_id.clone(),
                messages: messages.iter().map(WireMessage::from).collect(),
                tools: tools.iter().map(WireTool::from).collect(),
                temperature: config.temperature,
                max_tokens: config.max_tokens,
            }
        }
    }

    #[derive(Debug, Serialize)]
    pub struct WireMessage {
        pub role: &'static str,
        pub content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_call_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_calls: Option<Vec<WireToolCall>>,
    }

    impl From<&ChatMessage> for WireMessage {
        fn from(msg: &ChatMessage) -> Self {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::Tool => "tool",
            };
            // `content` is always a string: several OpenAI-compatible servers
            // reject `null` content on assistant turns that only call tools.
            Self {
                role,
                content: msg.text().to_string(),
                tool_call_id: msg.tool_call_id.clone(),
                tool_calls: msg
                    .tool_calls
                    .as_ref()
                    .map(|calls| calls.iter().map(WireToolCall::from).collect()),
            }
        }
    }

    #[derive(Debug, Serialize)]
    pub struct WireToolCall {
        pub id: String,
        pub r#type: &'static str,
        pub function: WireFunctionCall,
    }

    impl From<&crate::types::ToolCall> for WireToolCall {
        fn from(call: &crate::types::ToolCall) -> Self {
            Self {
                id: call.id.clone(),
                r#type: "function",
                function: WireFunctionCall {
                    name: call.name.clone(),
                    arguments: call.arguments.to_string(),
                },
            }
        }
    }

    #[derive(Debug, Serialize)]
    pub struct WireFunctionCall {
        pub name: String,
        pub arguments: String,
    }

    #[derive(Debug, Serialize)]
    pub struct WireTool {
        pub r#type: &'static str,
        pub function: WireFunction,
    }

    impl From<&ToolDefinition> for WireTool {
        fn from(tool: &ToolDefinition) -> Self {
            Self {
                r#type: "function",
                function: WireFunction {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: tool.parameters.clone(),
                },
            }
        }
    }

    #[derive(Debug, Serialize)]
    pub struct WireFunction {
        pub name: String,
        pub description: String,
        pub parameters: serde_json::Value,
    }

    /// Loose mirror of the response body. Only `choices` is required: usage is
    /// omitted by some compatible servers, and a strict shape would fail the
    /// whole call over a cosmetic field.
    #[derive(Debug, Deserialize)]
    pub struct ChatCompletionResponse {
        #[serde(default)]
        pub model: String,
        pub choices: Vec<WireChoice>,
        #[serde(default)]
        pub usage: Option<WireUsage>,
    }

    #[derive(Debug, Deserialize)]
    pub struct WireChoice {
        pub message: WireResponseMessage,
        #[serde(default)]
        pub finish_reason: Option<String>,
    }

    #[derive(Debug, Deserialize)]
    pub struct WireResponseMessage {
        #[serde(default)]
        pub content: Option<String>,
        #[serde(default)]
        pub tool_calls: Option<Vec<WireResponseToolCall>>,
    }

    #[derive(Debug, Deserialize)]
    pub struct WireResponseToolCall {
        #[serde(default)]
        pub id: String,
        pub function: WireResponseFunction,
    }

    #[derive(Debug, Deserialize)]
    pub struct WireResponseFunction {
        pub name: String,
        #[serde(default)]
        pub arguments: String,
    }

    #[derive(Debug, Deserialize)]
    pub struct WireUsage {
        #[serde(default)]
        pub prompt_tokens: usize,
        #[serde(default)]
        pub completion_tokens: usize,
        #[serde(default)]
        pub total_tokens: usize,
    }

    /// Re-expressed in fully-qualified `crate::types` names so the adapter's
    /// conversion stays a straight mapping.
    pub mod convert {
        use crate::types::{ChatMessage, LLMResponse, Role, StopReason, TokenUsage, ToolCall};

        use super::ChatCompletionResponse;

        pub fn into_llm_response(
            completion: ChatCompletionResponse,
            parse_args: impl Fn(&str) -> serde_json::Value,
            finish: impl Fn(Option<&str>) -> StopReason,
        ) -> Option<LLMResponse> {
            let choice = completion.choices.into_iter().next()?;

            let tool_calls = choice.message.tool_calls.map(|calls| {
                calls
                    .into_iter()
                    .map(|c| ToolCall {
                        id: c.id,
                        name: c.function.name,
                        arguments: parse_args(&c.function.arguments),
                    })
                    .collect::<Vec<_>>()
            });

            let message = ChatMessage {
                role: Role::Assistant,
                content: choice.message.content.filter(|c| !c.is_empty()),
                tool_call_id: None,
                tool_calls: tool_calls.filter(|c| !c.is_empty()),
            };

            let usage = completion
                .usage
                .map_or(TokenUsage::default(), |u| TokenUsage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                });

            Some(LLMResponse {
                message,
                usage,
                stop_reason: finish(choice.finish_reason.as_deref()),
                model: completion.model,
            })
        }
    }
}

#[cfg(test)]
mod tests;
