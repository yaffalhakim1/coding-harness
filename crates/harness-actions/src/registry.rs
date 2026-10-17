//! Action registry: the tool trait and the default registry.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use harness_models::event_store::EventStore;
use harness_models::types::{ToolCall, ToolDefinition};

use crate::error::ActionError;
use crate::file_state::FileState;
use crate::workspace::Workspace;

/// Shared, session-scoped state handed to every tool.
///
/// Cloning is cheap (the workspace is a canonical path, the state is behind a
/// mutex), so tools can each hold their own copy.
#[derive(Debug, Clone)]
pub struct ToolContext {
    pub workspace: Workspace,
    pub files: Arc<FileState>,
    /// Tool results longer than this are truncated (paper §3.1: 24k chars).
    pub max_output_chars: usize,
}

impl ToolContext {
    pub fn new(workspace: Workspace) -> Self {
        Self {
            workspace,
            files: Arc::new(FileState::new()),
            max_output_chars: 24_000,
        }
    }

    pub fn with_max_output_chars(mut self, chars: usize) -> Self {
        self.max_output_chars = chars;
        self
    }
}

/// Trait for individual tools/actions.
#[async_trait]
pub trait Action: Send + Sync {
    /// Tool name (e.g., "read_file", "bash").
    fn name(&self) -> &str;

    /// Tool definition for the LLM.
    fn definition(&self) -> ToolDefinition;

    /// Whether the tool only reads (paper Table 1). Read-only tools may run
    /// concurrently within a turn.
    fn is_read_only(&self) -> bool;

    /// Execute the tool with the given arguments.
    async fn execute(&self, arguments: serde_json::Value) -> Result<String, ActionError>;
}

/// Registry of available tools.
#[async_trait]
pub trait ActionRegistry: Send + Sync {
    /// Get all available tool definitions.
    fn get_tool_definitions(&self) -> Vec<ToolDefinition>;

    /// Execute a tool call.
    async fn execute(&self, tool_call: &ToolCall) -> Result<String, ActionError>;

    /// Check if a tool exists.
    fn has_tool(&self, name: &str) -> bool;

    /// Whether a tool is registered as read-only.
    fn is_read_only(&self, name: &str) -> bool {
        let _ = name;
        false
    }
}

/// Default registry implementation.
pub struct DefaultRegistry {
    /// `BTreeMap` so tool order in the request is stable across runs — an
    /// unstable tool order perturbs the prompt prefix and wrecks cache hits.
    tools: BTreeMap<String, Arc<dyn Action>>,
}

impl DefaultRegistry {
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    pub fn register(&mut self, action: Arc<dyn Action>) {
        self.tools.insert(action.name().to_string(), action);
    }

    /// Create a registry with the predefined tool set (paper Table 1).
    pub fn with_predefined_tools(context: ToolContext) -> Self {
        Self::with_predefined_tools_and_plan(context, None)
    }

    /// Create a registry with predefined tools and an optional plan state.
    ///
    /// When `plan_state` is `Some`, the `update_plan` tool is registered so the
    /// model can revise its plan during execution (paper §2.1).
    pub fn with_predefined_tools_and_plan(
        context: ToolContext,
        plan_state: Option<crate::update_plan::PlanState>,
    ) -> Self {
        let mut registry = Self::new();
        registry.register_all(context.clone(), PredefinedSet::Full);
        registry.register(Arc::new(crate::web_fetch::WebFetchTool::new(&context)));

        if let Some(state) = plan_state {
            registry.register(Arc::new(crate::update_plan::UpdatePlanTool::new(
                context, state,
            )));
        }

        registry
    }

    /// Create a bash-only registry (paper §2.2).
    pub fn bash_only(context: ToolContext) -> Self {
        Self::bash_only_with_plan(context, None)
    }

    /// Create a bash-only registry with an optional plan state.
    ///
    /// When `plan_state` is `Some`, the `update_plan` tool is also registered.
    pub fn bash_only_with_plan(
        context: ToolContext,
        plan_state: Option<crate::update_plan::PlanState>,
    ) -> Self {
        let mut registry = Self::new();
        registry.register_all(context.clone(), PredefinedSet::BashOnly);

        if let Some(state) = plan_state {
            registry.register(Arc::new(crate::update_plan::UpdatePlanTool::new(
                context, state,
            )));
        }

        registry
    }

    /// Register `recall_event` (M2). Only T2 and T4 keep elided originals, so
    /// only they expose it (paper §2.3).
    pub fn with_recall(mut self, context: ToolContext, store: EventStore) -> Self {
        self.register(Arc::new(crate::recall_event::RecallEventTool::new(
            context, store,
        )));
        self
    }

    /// Both predefined tools and bash (hybrid action space).
    pub fn hybrid(context: ToolContext) -> Self {
        Self::with_predefined_tools(context)
    }

    fn register_all(&mut self, ctx: ToolContext, set: PredefinedSet) {
        if set == PredefinedSet::Full {
            self.register(Arc::new(crate::file_ops::ReadFileTool::new(ctx.clone())));
            self.register(Arc::new(crate::file_ops::WriteFileTool::new(ctx.clone())));
            self.register(Arc::new(crate::file_ops::EditFileTool::new(ctx.clone())));
            self.register(Arc::new(crate::list_dir::ListDirTool::new(ctx.clone())));
            self.register(Arc::new(crate::search::SearchFilesTool::new(ctx.clone())));
            self.register(Arc::new(crate::search::GrepTool::new(ctx.clone())));
        }
        self.register(Arc::new(crate::bash::BashExecutor::new(
            ctx,
            set == PredefinedSet::BashOnly,
        )));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PredefinedSet {
    Full,
    BashOnly,
}

impl Default for DefaultRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ActionRegistry for DefaultRegistry {
    fn get_tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    async fn execute(&self, tool_call: &ToolCall) -> Result<String, ActionError> {
        if let Some(parse_error) = tool_call.arguments.get("__parse_error") {
            return Err(ActionError::MalformedArguments(
                parse_error.as_str().unwrap_or("unknown").to_string(),
            ));
        }

        let action = self
            .tools
            .get(&tool_call.name)
            .ok_or_else(|| ActionError::UnknownTool(tool_call.name.clone()))?;

        action.execute(tool_call.arguments.clone()).await
    }

    fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    fn is_read_only(&self, name: &str) -> bool {
        self.tools.get(name).is_some_and(|t| t.is_read_only())
    }
}

/// Truncate tool output on a character boundary, noting how much was dropped.
pub fn truncate_output(output: String, max_chars: usize) -> String {
    if output.chars().count() <= max_chars {
        return output;
    }
    let kept: String = output.chars().take(max_chars).collect();
    let dropped = output.chars().count() - max_chars;
    format!("{kept}\n…[truncated {dropped} chars]")
}
