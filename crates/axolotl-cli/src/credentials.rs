//! Credential resolution.
//!
//! The harness reads keys from the environment, falling back to the credentials
//! OpenCode already has on disk (`~/.local/share/opencode/auth.json`). That way
//! running the harness next to OpenCode needs no copy-pasting of secrets, and no
//! key ever has to be written into a config file.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Resolved credential for a model provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Credential {
    /// Environment variable the key maps to, so the provider can read it.
    pub env_var: String,
    /// Where the key came from, for reporting.
    pub source: KeySource,
}

/// Where a key was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    Environment,
    OpenCodeAuth,
}

impl std::fmt::Display for KeySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeySource::Environment => write!(f, "environment"),
            KeySource::OpenCodeAuth => write!(f, "opencode auth.json"),
        }
    }
}

/// One provider entry in OpenCode's `auth.json`.
#[derive(Debug, Deserialize)]
struct AuthEntry {
    #[serde(rename = "type")]
    kind: Option<String>,
    key: Option<String>,
}

/// Locate OpenCode's `auth.json`.
///
/// Honours `XDG_DATA_HOME` (as OpenCode does), then the platform default.
pub fn opencode_auth_path() -> Option<PathBuf> {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        if !xdg.trim().is_empty() {
            let candidate = PathBuf::from(xdg).join("opencode").join("auth.json");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    let home = std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)?;
    let candidate = home
        .join(".local")
        .join("share")
        .join("opencode")
        .join("auth.json");
    candidate.is_file().then_some(candidate)
}

/// Read a provider's API key from an OpenCode `auth.json`.
pub fn key_from_opencode(provider: &str) -> Result<Option<String>> {
    let Some(path) = opencode_auth_path() else {
        return Ok(None);
    };
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading '{}'", path.display()))?;
    let table: serde_json::Value =
        serde_json::from_str(&text).with_context(|| format!("parsing '{}'", path.display()))?;

    let Some(entry) = table.get(provider) else {
        return Ok(None);
    };
    let parsed: AuthEntry =
        serde_json::from_value(entry.clone()).context("unexpected auth entry shape")?;

    // Only plain API keys are usable here; OAuth entries have no `key`.
    if parsed.kind.as_deref() != Some("api") {
        return Ok(None);
    }
    Ok(parsed.key.filter(|k| !k.trim().is_empty()))
}

/// Map a model provider name to the environment variable it reads.
pub fn env_var_for_provider(provider: &str) -> String {
    let upper = provider.to_ascii_uppercase().replace('-', "_");
    format!("{upper}_API_KEY")
}

/// Resolve a credential for `provider`, exporting it to the environment when it
/// came from OpenCode so the provider adapter can read it as usual.
///
/// `configured_env` is the variable named in the config file; when absent the
/// provider name is used to derive one.
pub fn resolve(provider: &str, configured_env: Option<&str>) -> Result<Credential> {
    let env_var = configured_env
        .map(str::to_string)
        .unwrap_or_else(|| env_var_for_provider(provider));

    if std::env::var(&env_var).is_ok_and(|v| !v.trim().is_empty()) {
        return Ok(Credential {
            env_var,
            source: KeySource::Environment,
        });
    }

    if let Some(key) = key_from_opencode(provider)? {
        // SAFETY: single-threaded startup path; no other thread reads the
        // environment before the agent loop begins.
        std::env::set_var(&env_var, key);
        return Ok(Credential {
            env_var,
            source: KeySource::OpenCodeAuth,
        });
    }

    anyhow::bail!(
        "no API key for '{provider}': set {env_var}, or add a 'api' credential for '{provider}' to {}",
        opencode_auth_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "~/.local/share/opencode/auth.json".to_string())
    )
}

/// Check that a key is available, without spending a request.
pub fn check_api_key(provider: &str, configured_env: Option<&str>) -> Result<Credential> {
    resolve(provider, configured_env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_var_names_follow_the_provider() {
        assert_eq!(env_var_for_provider("kenari"), "KENARI_API_KEY");
        assert_eq!(env_var_for_provider("openai"), "OPENAI_API_KEY");
        assert_eq!(
            env_var_for_provider("xiaomi-token-plan-sgp"),
            "XIAOMI_TOKEN_PLAN_SGP_API_KEY"
        );
    }

    #[test]
    fn environment_wins_over_disk_and_is_not_overwritten() {
        // A unique variable name keeps this test independent of real keys.
        let var = "AXOLOTL_TEST_ENV_WINS_KEY";
        std::env::set_var(var, "from-env");
        let cred = resolve("nonexistent-provider", Some(var)).unwrap();
        assert_eq!(cred.source, KeySource::Environment);
        assert_eq!(std::env::var(var).unwrap(), "from-env");
        std::env::remove_var(var);
    }

    #[test]
    fn missing_key_produces_an_actionable_error() {
        let err = resolve("definitely-not-a-provider", Some("AXOLOTL_MISSING_KEY_X")).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("AXOLOTL_MISSING_KEY_X"), "{msg}");
        assert!(msg.contains("auth.json"), "{msg}");
    }

    #[test]
    fn unknown_provider_in_auth_file_returns_none() {
        // Either no auth file exists (None) or it lacks this provider (None).
        assert!(key_from_opencode("definitely-not-a-provider")
            .unwrap()
            .is_none());
    }
}
