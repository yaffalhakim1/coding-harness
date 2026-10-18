//! Client → agent messages.

use serde::Deserialize;

/// `initialize` params.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeParams {
    pub protocol_version: u32,
    #[serde(default)]
    pub client_capabilities: ClientCapabilities,
    #[serde(default)]
    pub client_info: Option<ClientInfo>,
}

/// What the client says it can do.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClientCapabilities {
    #[serde(default)]
    pub fs: Option<FsCapabilities>,
    /// `true` when the client can service `terminal/*`.
    #[serde(default)]
    pub terminal: bool,
    #[serde(default)]
    pub session: Option<SessionCapabilities>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FsCapabilities {
    #[serde(default)]
    pub read_text_file: bool,
    #[serde(default)]
    pub write_text_file: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SessionCapabilities {
    /// Present means the client renders `notice` updates.
    #[serde(default)]
    pub notices: Option<serde_json::Value>,
    /// Present means the client renders compaction updates.
    #[serde(default)]
    pub compaction: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClientInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
}

/// `session/new` params.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewSessionParams {
    pub cwd: String,
    #[serde(default)]
    pub mcp_servers: Vec<McpServer>,
    #[serde(default)]
    pub additional_directories: Option<Vec<String>>,
}

/// `session/load` and `session/resume` params.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoadSessionParams {
    pub session_id: String,
    pub cwd: String,
    #[serde(default)]
    pub mcp_servers: Vec<McpServer>,
    #[serde(default)]
    pub additional_directories: Option<Vec<String>>,
}

/// `session/close` params: only the session id, per the schema.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CloseSessionParams {
    pub session_id: String,
}

/// An MCP server the client asks the agent to connect to.
///
/// The stdio shape has no `type`; HTTP and SSE carry `type`. A tagged enum
/// would reject the stdio form, so the discriminator is read manually.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum McpServer {
    Stdio {
        name: String,
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: Vec<EnvVariable>,
    },
    Remote {
        #[serde(rename = "type")]
        kind: String,
        name: String,
        url: String,
        #[serde(default)]
        headers: Vec<EnvVariable>,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct EnvVariable {
    pub name: String,
    pub value: String,
}

/// `session/prompt` params.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptParams {
    pub session_id: String,
    pub prompt: Vec<ContentBlock>,
}

/// `session/cancel` params.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelParams {
    pub session_id: String,
}

/// A content block. Baseline requires text and resource_link.
///
/// `rename_all` on an enum renames variants, not their fields, so each field
/// that the protocol spells in camelCase is renamed explicitly. Getting this
/// wrong is silent: an unknown field is ignored, leaving `mime_type` as `None`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ResourceLink {
        uri: String,
        #[serde(default)]
        name: Option<String>,
    },
    Resource {
        resource: EmbeddedResource,
    },
    Image {
        #[serde(default, rename = "mimeType")]
        mime_type: Option<String>,
    },
    Audio {
        #[serde(default, rename = "mimeType")]
        mime_type: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddedResource {
    pub uri: String,
    #[serde(default)]
    pub text: Option<String>,
}

impl ContentBlock {
    /// Flatten to prompt text.
    ///
    /// The harness drives a text-only model interface, so non-text blocks
    /// degrade to a description rather than being dropped: silently discarding
    /// an attachment is how a user's screenshot becomes an unanswered question.
    pub fn as_text(&self) -> String {
        match self {
            ContentBlock::Text { text } => text.clone(),
            ContentBlock::ResourceLink { uri, name } => match name {
                Some(name) => format!("[linked resource: {name} <{uri}>]"),
                None => format!("[linked resource: {uri}]"),
            },
            ContentBlock::Resource { resource } => match &resource.text {
                Some(text) => text.clone(),
                None => format!("[embedded resource: {}]", resource.uri),
            },
            ContentBlock::Image { mime_type } => format!(
                "[image omitted: this agent is text-only ({})]",
                mime_type.as_deref().unwrap_or("unknown type")
            ),
            ContentBlock::Audio { mime_type } => format!(
                "[audio omitted: this agent is text-only ({})]",
                mime_type.as_deref().unwrap_or("unknown type")
            ),
        }
    }
}

/// `session/request_permission` result.
#[derive(Debug, Clone, Deserialize)]
pub struct RequestPermissionResult {
    pub outcome: PermissionOutcome,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PermissionOutcome {
    Cancelled,
    Selected {
        #[serde(rename = "optionId")]
        option_id: String,
    },
}

impl PermissionOutcome {
    /// Whether the user allowed the operation.
    pub fn allowed(&self) -> bool {
        matches!(self, PermissionOutcome::Selected { .. })
    }
}
