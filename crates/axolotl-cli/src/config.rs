//! Configuration loading.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use axolotl_core::{ActionSpace, SessionConfig, ThresholdSettings};
use axolotl_models::types::ModelConfig;

/// The on-disk config file (`config/default.toml`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileConfig {
    pub context_window: usize,
    /// T0-T4.
    pub context_strategy: String,
    pub planning_enabled: bool,
    /// `predefined` | `bash-only` | `hybrid`.
    pub action_space: String,
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub thresholds: Option<ThresholdSettings>,
    #[serde(default)]
    pub token_budget: Option<usize>,
    /// Expose the `task` tool so the agent can delegate to subagents.
    #[serde(default)]
    pub subagents: bool,
    pub model: FileModel,
    /// MCP servers whose tools are exposed to the agent.
    #[serde(default)]
    pub mcp: Vec<FileMcpServer>,
}

fn default_max_turns() -> u32 {
    50
}

fn default_timeout() -> u64 {
    600
}

/// Model section of the config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileModel {
    pub provider: String,
    pub model_id: String,
    #[serde(default)]
    pub api_base: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_model_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub input_cost_per_mtok: Option<f64>,
    #[serde(default)]
    pub output_cost_per_mtok: Option<f64>,
    #[serde(default = "default_retries")]
    pub max_retries: u32,
    /// Extra request headers (e.g. for gateway auth).
    #[serde(default)]
    pub headers: Vec<(String, String)>,
    /// Fallback models tried, in order, when this one is unreachable.
    ///
    /// Each entry is a full model block, so a fallback may use a different
    /// provider, endpoint or key.
    #[serde(default)]
    pub fallbacks: Vec<FileModel>,
}

/// An MCP server entry in the config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMcpServer {
    pub name: String,
    /// `stdio` or `http`.
    #[serde(rename = "type")]
    pub transport_type: String,
    /// For stdio.
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    /// For http.
    #[serde(default)]
    pub url: Option<String>,
}

impl FileMcpServer {
    /// Convert to a runtime server config, validating the transport fields.
    pub fn to_server_config(&self) -> Result<axolotl_actions::mcp::ServerConfig> {
        use axolotl_actions::mcp::Transport;

        let transport = match self.transport_type.to_ascii_lowercase().as_str() {
            "stdio" => {
                let command = self
                    .command
                    .clone()
                    .with_context(|| format!("MCP server '{}' has no command", self.name))?;
                Transport::Stdio {
                    command,
                    args: self.args.clone(),
                    env: self.env.clone(),
                }
            }
            "http" => {
                let url = self
                    .url
                    .clone()
                    .with_context(|| format!("MCP server '{}' has no url", self.name))?;
                Transport::Http { url }
            }
            other => anyhow::bail!(
                "MCP server '{}' has unknown type '{other}' (expected stdio or http)",
                self.name
            ),
        };

        Ok(axolotl_actions::mcp::ServerConfig {
            name: self.name.clone(),
            transport,
        })
    }
}

impl FileModel {
    /// Convert to a runtime [`ModelConfig`].
    pub fn to_model_config(&self) -> ModelConfig {
        ModelConfig {
            provider: self.provider.clone(),
            model_id: self.model_id.clone(),
            api_base: self.api_base.clone(),
            api_key_env: self.api_key_env.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            timeout_secs: self.timeout_secs,
            input_cost_per_mtok: self.input_cost_per_mtok,
            output_cost_per_mtok: self.output_cost_per_mtok,
            max_retries: self.max_retries,
        }
    }
}

fn default_model_timeout() -> u64 {
    120
}

fn default_retries() -> u32 {
    4
}

impl FileConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config '{}'", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing config '{}'", path.display()))
    }

    /// Convert to the runtime session config, validating enum strings.
    pub fn to_session_config(&self) -> Result<SessionConfig> {
        let context_strategy = axolotl_context::ContextTier::parse(&self.context_strategy)
            .with_context(|| {
                format!(
                    "invalid context_strategy '{}' (expected T0-T4)",
                    self.context_strategy
                )
            })?;
        let action_space = ActionSpace::parse(&self.action_space).with_context(|| {
            format!(
                "invalid action_space '{}' (expected predefined, bash-only, or hybrid)",
                self.action_space
            )
        })?;

        Ok(SessionConfig {
            context_window: self.context_window,
            context_strategy,
            planning_enabled: self.planning_enabled,
            action_space,
            model: self.model.to_model_config(),
            max_turns: self.max_turns,
            timeout_secs: self.timeout_secs,
            thresholds: self.thresholds.unwrap_or_default(),
            // Not exposed in the config file: the default is the useful value.
            max_narration_turns: 3,
            token_budget: self.token_budget,
            subagents: self.subagents,
        })
    }

    /// Every model in preference order: primary first, then fallbacks.
    pub fn model_chain(&self) -> Vec<ModelConfig> {
        std::iter::once(self.model.to_model_config())
            .chain(self.model.fallbacks.iter().map(FileModel::to_model_config))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_file_parses() {
        let cfg = FileConfig::load("../../config/default.toml").unwrap();
        let session = cfg.to_session_config().unwrap();
        assert!(matches!(
            session.context_strategy,
            axolotl_context::ContextTier::T4
        ));
        assert_eq!(session.action_space, ActionSpace::Predefined);
        assert_eq!(session.context_window, 128_000);
        assert!(session.planning_enabled);
    }

    #[test]
    fn invalid_strategy_is_rejected() {
        let mut cfg = FileConfig::load("../../config/default.toml").unwrap();
        cfg.context_strategy = "T9".into();
        assert!(cfg.to_session_config().is_err());
    }

    #[test]
    fn invalid_action_space_is_rejected() {
        let mut cfg = FileConfig::load("../../config/default.toml").unwrap();
        cfg.action_space = "magic".into();
        assert!(cfg.to_session_config().is_err());
    }

    #[test]
    fn missing_file_is_an_error() {
        assert!(FileConfig::load("does/not/exist.toml").is_err());
    }

    #[test]
    fn bash_only_parses() {
        let mut cfg = FileConfig::load("../../config/default.toml").unwrap();
        cfg.action_space = "bash-only".into();
        assert_eq!(
            cfg.to_session_config().unwrap().action_space,
            ActionSpace::BashOnly
        );
    }
}
