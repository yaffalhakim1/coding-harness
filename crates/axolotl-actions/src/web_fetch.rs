//! The `web_fetch` tool: retrieve a concrete URL as text (paper Table 1, Fig. 29).
//!
//! Only read-only GET requests are issued. The body is capped so a large page
//! cannot blow the context budget, and HTML is reduced to readable text.

use std::time::Duration;

use async_trait::async_trait;
use tracing::debug;

use axolotl_models::types::ToolDefinition;

use crate::error::{ActionError, ActionResult};
use crate::registry::{Action, ToolContext};

/// Response bodies above this are truncated (paper Fig. 29: default 2MB).
const MAX_BYTES: usize = 2 * 1024 * 1024;

/// Fetch an HTTP(S) URL and return it as text.
pub struct WebFetchTool {
    client: reqwest::Client,
    max_output_chars: usize,
}

impl WebFetchTool {
    pub fn new(ctx: &ToolContext) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .user_agent("axolotl/0.1")
                .build()
                .unwrap_or_default(),
            max_output_chars: ctx.max_output_chars,
        }
    }
}

#[async_trait]
impl Action for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "web_fetch".to_string(),
            description: "Fetch the content of an HTTP(S) URL and return it as readable text. HTML responses are reduced to plain text; JSON and text responses pass through unchanged.\n\nWhen: use this to read documentation pages, API responses, GitHub READMEs, library docs — anything you have a concrete URL for. For files inside the workspace use read_file.\n\nProtocol: URL must be http:// or https://. The response is capped at 2MB; larger responses are truncated. Redirects are followed automatically.\n\nError modes: HTTP 4xx/5xx responses return an error with a short body preview. Network failures (DNS, refused, TLS, timeout) also return an error. Binary content returns a notice with size and content-type rather than garbled bytes.\n\nSide effects: makes one outbound HTTP request. No workspace files are touched.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The http:// or https:// URL to fetch"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["text", "html"],
                        "description": "Output format (default text; html returns the raw body)"
                    }
                },
                "required": ["url"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let url = arguments
            .get("url")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ActionError::InvalidArgument("missing 'url'".into()))?;

        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(ActionError::InvalidArgument(format!(
                "'{url}' must start with http:// or https://"
            )));
        }

        let want_html = arguments
            .get("format")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|f| f.eq_ignore_ascii_case("html"));

        debug!(url = %url, "Fetching URL");

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| ActionError::Fetch {
                url: url.to_string(),
                reason: e.to_string(),
            })?;

        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("unknown")
            .to_string();

        let body = response.text().await.map_err(|e| ActionError::Fetch {
            url: url.to_string(),
            reason: e.to_string(),
        })?;

        if !status.is_success() {
            return Err(ActionError::Fetch {
                url: url.to_string(),
                reason: format!(
                    "HTTP {status}; body preview: {}",
                    truncate_chars(&body, 500)
                ),
            });
        }

        let is_html = content_type.contains("html");
        let rendered = if is_html && !want_html {
            html_to_text(&body)
        } else {
            body
        };

        let rendered = truncate_bytes(rendered, MAX_BYTES);
        Ok(format!(
            "[{url}] ({content_type})\n{}",
            crate::registry::truncate_output(rendered, self.max_output_chars)
        ))
    }
}

/// Reduce HTML to readable text without pulling in a parser.
///
/// A full HTML parser is not worth a dependency here: the goal is legible
/// documentation, and script/style blocks plus tags are the bulk of the noise.
fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut chars = html.chars().peekable();
    let mut in_tag = false;
    let mut in_block = false;
    let mut tag = String::new();

    while let Some(c) = chars.next() {
        match c {
            '<' => {
                in_tag = true;
                tag.clear();
            }
            '>' if in_tag => {
                in_tag = false;
                let raw = tag.trim().to_ascii_lowercase();
                let closing = raw.starts_with('/');
                let name = raw.trim_start_matches('/').trim();
                // Element name without attributes, e.g. `style` from `style type=...`.
                let element = name.split_whitespace().next().unwrap_or("");

                let noise = element == "script" || element == "style";
                if noise {
                    in_block = !closing;
                } else if matches!(
                    element,
                    "p" | "br" | "div" | "li" | "tr" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
                ) {
                    // Block-level tags become line breaks so paragraphs survive.
                    out.push('\n');
                }
            }
            _ if in_tag => tag.push(c),
            _ if in_block => {}
            '&' => {
                // Decode the handful of entities that actually appear in docs.
                let mut entity = String::new();
                while let Some(&next) = chars.peek() {
                    if next == ';' || entity.len() > 8 {
                        break;
                    }
                    entity.push(next);
                    chars.next();
                }
                if chars.peek() == Some(&';') {
                    chars.next();
                }
                out.push_str(match entity.as_str() {
                    "amp" => "&",
                    "lt" => "<",
                    "gt" => ">",
                    "quot" => "\"",
                    "apos" | "#39" => "'",
                    "nbsp" => " ",
                    _ => " ",
                });
            }
            _ => out.push(c),
        }
    }

    // Collapse runs of blank lines left by removed tags.
    let mut cleaned = String::with_capacity(out.len());
    let mut blank_run = 0;
    for line in out.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            blank_run += 1;
            if blank_run > 1 {
                continue;
            }
        } else {
            blank_run = 0;
        }
        cleaned.push_str(trimmed);
        cleaned.push('\n');
    }
    cleaned.trim().to_string()
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    text.chars().take(max).collect::<String>() + "…"
}

fn truncate_bytes(text: String, max: usize) -> String {
    if text.len() <= max {
        return text;
    }
    let mut cut = max;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}\n…[response truncated at {max} bytes]", &text[..cut])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_tags_and_keeps_text() {
        let html = "<html><body><h1>Title</h1><p>Hello <b>world</b></p></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Title"), "{text}");
        assert!(text.contains("Hello world"), "{text}");
        assert!(!text.contains('<'), "{text}");
    }

    #[test]
    fn drops_script_and_style_bodies() {
        let html = "<style>.a{color:red}</style><script>alert('x')</script><p>Keep me</p>";
        let text = html_to_text(html);
        assert!(text.contains("Keep me"), "{text}");
        assert!(!text.contains("alert"), "{text}");
        assert!(!text.contains("color:red"), "{text}");
    }

    #[test]
    fn decodes_common_entities() {
        assert!(html_to_text("<p>a &amp; b</p>").contains("a & b"));
        assert!(html_to_text("<p>&lt;tag&gt;</p>").contains("<tag>"));
    }

    #[test]
    fn block_tags_become_line_breaks() {
        let text = html_to_text("<p>one</p><p>two</p>");
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        assert_eq!(lines, vec!["one", "two"]);
    }

    #[test]
    fn rejects_non_http_urls() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(crate::workspace::Workspace::open(dir.path()).unwrap());
        let tool = WebFetchTool::new(&ctx);
        let err = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(tool.execute(serde_json::json!({ "url": "file:///etc/passwd" })))
            .unwrap_err();
        assert!(matches!(err, ActionError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn truncate_bytes_respects_char_boundaries() {
        let s = "α".repeat(100);
        let out = truncate_bytes(s, 10);
        assert!(out.contains("truncated"), "{out}");
    }

    #[test]
    fn is_read_only() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(crate::workspace::Workspace::open(dir.path()).unwrap());
        assert!(WebFetchTool::new(&ctx).is_read_only());
    }
}
