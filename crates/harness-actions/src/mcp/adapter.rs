//! Adapter: expose MCP tools as harness `Action`s.
//!
//! Each MCP tool becomes one registry entry named `<server>__<tool>`, so a
//! server cannot shadow a built-in like `bash` or `read_file`.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use harness_models::types::ToolDefinition;

use super::client::{McpClient, McpTool};
use crate::error::{ActionError, ActionResult};
use crate::registry::Action;

/// Separator between server and tool name in the registry key.
pub const NAME_SEPARATOR: &str = "__";

/// One MCP tool, callable through the shared client.
pub struct McpToolAction {
    /// Registry key, e.g. `filesystem__read_file`.
    registry_name: String,
    /// Name the server knows, e.g. `read_file`.
    remote_name: String,
    tool: McpTool,
    client: Arc<Mutex<McpClient>>,
}

impl McpToolAction {
    pub fn new(server: &str, tool: McpTool, client: Arc<Mutex<McpClient>>) -> Self {
        let registry_name = format!("{server}{NAME_SEPARATOR}{}", tool.name);
        Self {
            registry_name,
            remote_name: tool.name.clone(),
            tool,
            client,
        }
    }

    /// Registry key this action is registered under.
    pub fn registry_name(&self) -> &str {
        &self.registry_name
    }
}

#[async_trait]
impl Action for McpToolAction {
    fn name(&self) -> &str {
        &self.registry_name
    }

    fn definition(&self) -> ToolDefinition {
        let description = if self.tool.description.is_empty() {
            format!("MCP tool '{}'", self.remote_name)
        } else {
            self.tool.description.clone()
        };

        // A server that omits a schema still needs an object schema, or the
        // model's arguments are rejected before they reach the tool.
        let parameters = if self.tool.input_schema.is_null() {
            serde_json::json!({ "type": "object", "properties": {} })
        } else {
            self.tool.input_schema.clone()
        };

        ToolDefinition {
            name: self.registry_name.clone(),
            description,
            parameters,
        }
    }

    fn is_read_only(&self) -> bool {
        // MCP gives no read-only signal in the schema, and guessing from the
        // name is how a `write_file` ends up running in parallel with a read of
        // the same file. Treat every MCP tool as mutating: correct beats fast.
        false
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let mut client = self.client.lock().await;
        client
            .call_tool(&self.remote_name, arguments)
            .await
            .map_err(|e| ActionError::Mcp {
                tool: self.registry_name.clone(),
                reason: e.to_string(),
            })
    }
}

/// Connect to every configured server and return the actions to register.
///
/// A server that fails to connect is reported and skipped: one broken MCP
/// server must not prevent the harness from running.
pub async fn connect_all(
    servers: &[super::client::ServerConfig],
) -> (Vec<McpToolAction>, Vec<String>) {
    let mut actions = Vec::new();
    let mut failures = Vec::new();

    for config in servers {
        match McpClient::connect(config).await {
            Ok(client) => {
                let client = Arc::new(Mutex::new(client));
                let tools = client.lock().await.tools().to_vec();
                let count = tools.len();
                for tool in tools {
                    actions.push(McpToolAction::new(&config.name, tool, client.clone()));
                }
                tracing::info!(
                    server = %config.name,
                    tools = count,
                    "Connected MCP server"
                );
            }
            Err(e) => {
                tracing::warn!(server = %config.name, error = %e, "MCP server unavailable");
                failures.push(format!("{}: {e}", config.name));
            }
        }
    }

    (actions, failures)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, description: &str, schema: serde_json::Value) -> McpTool {
        McpTool {
            name: name.into(),
            description: description.into(),
            input_schema: schema,
        }
    }

    fn dummy_client() -> Arc<Mutex<McpClient>> {
        // A client that is never used: these tests only exercise the definition.
        Arc::new(Mutex::new(McpClient::disconnected_for_test()))
    }

    #[test]
    fn registry_name_is_namespaced() {
        let action = McpToolAction::new(
            "filesystem",
            tool("read_file", "read", serde_json::json!({})),
            dummy_client(),
        );
        assert_eq!(action.registry_name(), "filesystem__read_file");
    }

    #[test]
    fn definition_uses_the_namespaced_name() {
        let action = McpToolAction::new(
            "fs",
            tool(
                "grep",
                "search files",
                serde_json::json!({ "type": "object" }),
            ),
            dummy_client(),
        );
        let def = action.definition();
        assert_eq!(def.name, "fs__grep");
        assert_eq!(def.description, "search files");
        assert_eq!(def.parameters["type"], "object");
    }

    #[test]
    fn missing_schema_becomes_an_object_schema() {
        let action = McpToolAction::new(
            "fs",
            tool("bare", "", serde_json::Value::Null),
            dummy_client(),
        );
        let def = action.definition();
        assert_eq!(def.parameters["type"], "object");
        assert!(def.parameters["properties"].is_object());
    }

    #[test]
    fn empty_description_is_replaced() {
        let action = McpToolAction::new(
            "fs",
            tool("bare", "", serde_json::json!({})),
            dummy_client(),
        );
        assert!(action.definition().description.contains("bare"));
    }

    #[test]
    fn mcp_tools_are_treated_as_mutating() {
        // Read-only parallel execution must never include an MCP tool whose
        // side effects we cannot see.
        let action = McpToolAction::new(
            "fs",
            tool("read_file", "read", serde_json::json!({})),
            dummy_client(),
        );
        assert!(!action.is_read_only());
    }
}
