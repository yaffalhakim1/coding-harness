//! Multi-provider routing with failover.
//!
//! OpenCode and jcode both keep several providers configured and move to the
//! next one when the active provider fails. The paper's harness pinned one
//! model, so failover is additive here: it changes reliability, not the
//! measured behaviour, and it is opt-in.
//!
//! Failover only triggers on errors that are *not* the caller's fault
//! (rate limits, 5xx, transport). A bad request or an auth failure is returned
//! immediately: retrying those against another provider just burns money on a
//! request that is wrong regardless of who serves it.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::{debug, warn};

use crate::provider::{LLMProvider, ProviderError, ProviderResult};
use crate::types::{ChatMessage, LLMResponse, ModelConfig, ToolDefinition};

/// A provider plus the config used to reach it.
#[derive(Clone)]
pub struct Route {
    pub name: String,
    pub provider: Arc<dyn LLMProvider>,
    pub config: ModelConfig,
}

impl Route {
    pub fn new(provider: Arc<dyn LLMProvider>, config: ModelConfig) -> Self {
        let name = config.model_id.clone();
        Self {
            name,
            provider,
            config,
        }
    }
}

/// Tries each route in order until one succeeds.
pub struct FailoverProvider {
    routes: Vec<Route>,
    /// Name reported by [`LLMProvider::name`].
    label: String,
}

impl FailoverProvider {
    /// Build from an ordered list. The first route is preferred.
    pub fn new(routes: Vec<Route>) -> Self {
        assert!(
            !routes.is_empty(),
            "FailoverProvider needs at least one route"
        );
        let label = routes
            .iter()
            .map(|r| r.name.as_str())
            .collect::<Vec<_>>()
            .join("|");
        Self { routes, label }
    }

    /// Routes in preference order.
    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    /// Whether an error should move to the next route.
    fn should_fail_over(error: &ProviderError) -> bool {
        match error {
            // The route itself is unreachable or exhausted.
            ProviderError::RateLimited { .. }
            | ProviderError::Server { .. }
            | ProviderError::Transport(_)
            | ProviderError::RetriesExhausted { .. } => true,
            // Configuration and request errors follow us to every provider.
            ProviderError::Config(_)
            | ProviderError::MissingApiKey(_)
            | ProviderError::Auth { .. }
            | ProviderError::BadRequest { .. }
            | ProviderError::Decode(_)
            | ProviderError::EmptyResponse => false,
        }
    }
}

#[async_trait]
impl LLMProvider for FailoverProvider {
    async fn chat(
        &self,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        _config: &ModelConfig,
    ) -> ProviderResult<LLMResponse> {
        let mut last: Option<ProviderError> = None;

        for (index, route) in self.routes.iter().enumerate() {
            // Each route carries its own model id, base URL and key, so the
            // caller's config is intentionally ignored.
            match route.provider.chat(messages, tools, &route.config).await {
                Ok(mut response) => {
                    if index > 0 {
                        warn!(
                            provider = %route.name,
                            attempt = index + 1,
                            "Served by a failover provider"
                        );
                    }
                    // Record which model actually answered: a sweep that
                    // silently fell over to another model is not comparable.
                    if response.model.is_empty() {
                        response.model = route.name.clone();
                    }
                    return Ok(response);
                }
                Err(error) if Self::should_fail_over(&error) => {
                    warn!(
                        provider = %route.name,
                        attempt = index + 1,
                        error = %error,
                        "Provider failed; trying the next route"
                    );
                    last = Some(error);
                }
                Err(error) => {
                    debug!(provider = %route.name, error = %error, "Non-retryable failure");
                    return Err(error);
                }
            }
        }

        Err(ProviderError::RetriesExhausted {
            attempts: self.routes.len() as u32,
            source: Box::new(last.unwrap_or(ProviderError::EmptyResponse)),
        })
    }

    fn name(&self) -> &str {
        &self.label
    }

    async fn health_check(&self, _config: &ModelConfig) -> ProviderResult<bool> {
        for route in &self.routes {
            if route
                .provider
                .health_check(&route.config)
                .await
                .unwrap_or(false)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockProvider, ScriptedStep};
    use crate::types::{StopReason, TokenUsage};

    fn config(model: &str) -> ModelConfig {
        ModelConfig {
            model_id: model.to_string(),
            ..ModelConfig::default()
        }
    }

    fn ok_provider(text: &str) -> Arc<MockProvider> {
        MockProvider::new(vec![ScriptedStep::Respond(LLMResponse {
            message: ChatMessage::assistant(text),
            usage: TokenUsage::default(),
            stop_reason: StopReason::EndTurn,
            model: String::new(),
        })])
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    fn ask(provider: &FailoverProvider) -> ProviderResult<LLMResponse> {
        rt().block_on(provider.chat(&[ChatMessage::user("hi")], &[], &ModelConfig::default()))
    }

    #[test]
    fn first_route_wins_when_healthy() {
        let provider = FailoverProvider::new(vec![
            Route::new(ok_provider("first"), config("a")),
            Route::new(ok_provider("second"), config("b")),
        ]);
        let response = ask(&provider).unwrap();
        assert_eq!(response.message.text(), "first");
        // The answering model is recorded, so a silent failover is visible.
        assert_eq!(response.model, "a");
    }

    #[test]
    fn transport_failure_moves_to_the_next_route() {
        let failing = MockProvider::new(vec![ScriptedStep::Fail("connection reset".into())]);
        let provider = FailoverProvider::new(vec![
            Route::new(failing, config("a")),
            Route::new(ok_provider("second"), config("b")),
        ]);
        let response = ask(&provider).unwrap();
        assert_eq!(response.message.text(), "second");
        assert_eq!(response.model, "b");
    }

    #[test]
    fn all_routes_failing_reports_exhaustion() {
        let a = MockProvider::new(vec![ScriptedStep::Fail("down".into())]);
        let b = MockProvider::new(vec![ScriptedStep::Fail("down too".into())]);
        let provider =
            FailoverProvider::new(vec![Route::new(a, config("a")), Route::new(b, config("b"))]);
        let err = ask(&provider).unwrap_err();
        assert!(
            matches!(err, ProviderError::RetriesExhausted { attempts: 2, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn auth_failure_does_not_fail_over() {
        // A bad key is wrong for every provider; retrying wastes money.
        struct AuthFail;
        #[async_trait]
        impl LLMProvider for AuthFail {
            async fn chat(
                &self,
                _m: &[ChatMessage],
                _t: &[ToolDefinition],
                _c: &ModelConfig,
            ) -> ProviderResult<LLMResponse> {
                Err(ProviderError::Auth {
                    status: 401,
                    body: "bad key".into(),
                })
            }
            fn name(&self) -> &str {
                "authfail"
            }
            async fn health_check(&self, _c: &ModelConfig) -> ProviderResult<bool> {
                Ok(false)
            }
        }

        let provider = FailoverProvider::new(vec![
            Route::new(Arc::new(AuthFail), config("a")),
            Route::new(ok_provider("second"), config("b")),
        ]);
        let err = ask(&provider).unwrap_err();
        assert!(
            matches!(err, ProviderError::Auth { status: 401, .. }),
            "{err:?}"
        );
    }

    #[test]
    fn bad_request_does_not_fail_over() {
        struct BadRequest;
        #[async_trait]
        impl LLMProvider for BadRequest {
            async fn chat(
                &self,
                _m: &[ChatMessage],
                _t: &[ToolDefinition],
                _c: &ModelConfig,
            ) -> ProviderResult<LLMResponse> {
                Err(ProviderError::BadRequest {
                    status: 400,
                    body: "malformed".into(),
                })
            }
            fn name(&self) -> &str {
                "bad"
            }
            async fn health_check(&self, _c: &ModelConfig) -> ProviderResult<bool> {
                Ok(false)
            }
        }

        let provider = FailoverProvider::new(vec![
            Route::new(Arc::new(BadRequest), config("a")),
            Route::new(ok_provider("second"), config("b")),
        ]);
        assert!(matches!(
            ask(&provider).unwrap_err(),
            ProviderError::BadRequest { .. }
        ));
    }

    #[test]
    fn name_lists_the_routes_in_order() {
        let provider = FailoverProvider::new(vec![
            Route::new(ok_provider("a"), config("model-a")),
            Route::new(ok_provider("b"), config("model-b")),
        ]);
        assert_eq!(provider.name(), "model-a|model-b");
        assert_eq!(provider.routes().len(), 2);
    }

    #[test]
    fn single_route_behaves_like_a_plain_provider() {
        let provider = FailoverProvider::new(vec![Route::new(ok_provider("only"), config("a"))]);
        assert_eq!(ask(&provider).unwrap().message.text(), "only");
    }

    #[test]
    fn health_check_passes_if_any_route_is_healthy() {
        let provider = FailoverProvider::new(vec![
            Route::new(ok_provider("a"), config("a")),
            Route::new(ok_provider("b"), config("b")),
        ]);
        assert!(rt()
            .block_on(provider.health_check(&ModelConfig::default()))
            .unwrap());
    }

    #[test]
    fn failover_classification_is_explicit() {
        assert!(FailoverProvider::should_fail_over(
            &ProviderError::Transport("x".into())
        ));
        assert!(FailoverProvider::should_fail_over(
            &ProviderError::RateLimited {
                status: 429,
                body: String::new()
            }
        ));
        assert!(!FailoverProvider::should_fail_over(
            &ProviderError::MissingApiKey("X".into())
        ));
        assert!(!FailoverProvider::should_fail_over(
            &ProviderError::EmptyResponse
        ));
    }
}
