//! The `ask_user` tool: let the agent ask a question instead of guessing.
//!
//! jcode's `communicate` and OpenCode's question prompt both exist because a
//! model that has to choose between two reasonable designs should ask rather
//! than pick one and waste a turn. The tool is deliberately narrow: a question
//! and a list of options, no free-form dialogue loop.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use axolotl_models::types::ToolDefinition;

use crate::error::{ActionError, ActionResult};
use crate::registry::{Action, ToolContext};

/// A question waiting for an answer.
#[derive(Debug, Clone)]
pub struct Question {
    pub question: String,
    pub options: Vec<String>,
    /// Whether more than one option may be chosen.
    pub multi_select: bool,
}

/// The answer a human gave.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    /// Chosen options, or the free-text reply when none were offered.
    pub selections: Vec<String>,
}

impl Answer {
    /// Render for the model.
    pub fn render(&self) -> String {
        if self.selections.is_empty() {
            "The user did not answer.".to_string()
        } else {
            self.selections.join(", ")
        }
    }
}

/// Asks the user. Implemented by the front end: ACP routes it through
/// `elicitation/create`, the CLI through a terminal prompt.
#[async_trait]
pub trait Asker: Send + Sync {
    async fn ask(&self, question: &Question) -> Option<Answer>;
}

/// Records questions and returns a canned answer, for tests.
pub struct ScriptedAsker {
    answer: Option<Answer>,
    asked: Mutex<Vec<Question>>,
}

impl ScriptedAsker {
    pub fn new(answer: Option<Answer>) -> Arc<Self> {
        Arc::new(Self {
            answer,
            asked: Mutex::new(Vec::new()),
        })
    }

    /// Questions asked so far.
    pub async fn asked(&self) -> Vec<Question> {
        self.asked.lock().await.clone()
    }
}

#[async_trait]
impl Asker for ScriptedAsker {
    async fn ask(&self, question: &Question) -> Option<Answer> {
        self.asked.lock().await.push(question.clone());
        self.answer.clone()
    }
}

/// The `ask_user` tool.
pub struct AskUserTool {
    asker: Arc<dyn Asker>,
    _ctx: ToolContext,
}

impl AskUserTool {
    pub fn new(ctx: ToolContext, asker: Arc<dyn Asker>) -> Self {
        Self { asker, _ctx: ctx }
    }
}

#[async_trait]
impl Action for AskUserTool {
    fn name(&self) -> &str {
        "ask_user"
    }

    fn is_read_only(&self) -> bool {
        // It does not touch the workspace, but it does block on a human; the
        // registry's read-only fast path must not run it in parallel.
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "ask_user".to_string(),
            description: "Ask the user a question and wait for the answer.\n\nWhen: you are genuinely blocked on a decision only the user can make — two reasonable designs with different tradeoffs, an ambiguous requirement, or a destructive action that was not requested. Do NOT use it to confirm something you can verify yourself, and do not use it to ask permission for a tool (the harness handles that separately).\n\nProtocol: pass `question` (one sentence, ending in a question mark) and optionally `options` (short, mutually exclusive choices). With options the user picks; without options they reply in their own words.\n\nSide effects: blocks the turn until the user answers or dismisses the question. A dismissed question returns nothing, so phrase it so that 'no answer' is survivable.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question, in one sentence"
                    },
                    "options": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional short choices; the user picks one"
                    },
                    "multi_select": {
                        "type": "boolean",
                        "description": "Allow more than one option to be chosen (default false)"
                    }
                },
                "required": ["question"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let question = arguments
            .get("question")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ActionError::InvalidArgument("missing 'question'".into()))?;

        if question.trim().is_empty() {
            return Err(ActionError::InvalidArgument(
                "question must not be empty".into(),
            ));
        }

        let options = arguments
            .get("options")
            .and_then(serde_json::Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let multi_select = arguments
            .get("multi_select")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        let request = Question {
            question: question.to_string(),
            options,
            multi_select,
        };

        match self.asker.ask(&request).await {
            Some(answer) => Ok(format!("User answered: {}", answer.render())),
            // A dismissal is not an error: the model must continue with what it
            // has rather than retry the same question.
            None => Ok(
                "The user dismissed the question without answering. Continue with your best judgement."
                    .to_string(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::Workspace;

    fn ctx() -> ToolContext {
        let dir = tempfile::tempdir().unwrap();
        ToolContext::new(Workspace::open(dir.path()).unwrap())
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    #[test]
    fn returns_the_users_choice() {
        let asker = ScriptedAsker::new(Some(Answer {
            selections: vec!["Postgres".into()],
        }));
        let tool = AskUserTool::new(ctx(), asker.clone());

        let out = rt()
            .block_on(tool.execute(serde_json::json!({
                "question": "Which database?",
                "options": ["Postgres", "SQLite"]
            })))
            .unwrap();

        assert!(out.contains("Postgres"), "{out}");
        assert_eq!(
            rt().block_on(asker.asked())[0].options,
            vec!["Postgres", "SQLite"]
        );
    }

    #[test]
    fn a_dismissed_question_is_not_an_error() {
        // The model must be able to continue; retrying the same question would
        // burn the turn.
        let asker = ScriptedAsker::new(None);
        let tool = AskUserTool::new(ctx(), asker);

        let out = rt()
            .block_on(tool.execute(serde_json::json!({ "question": "Proceed?" })))
            .unwrap();

        assert!(out.contains("dismissed"), "{out}");
        assert!(out.contains("best judgement"), "{out}");
    }

    #[test]
    fn an_empty_question_is_rejected() {
        let tool = AskUserTool::new(ctx(), ScriptedAsker::new(None));
        let err = rt()
            .block_on(tool.execute(serde_json::json!({ "question": "  " })))
            .unwrap_err();
        assert!(matches!(err, ActionError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn options_are_optional() {
        let asker = ScriptedAsker::new(Some(Answer {
            selections: vec!["use the cache".into()],
        }));
        let tool = AskUserTool::new(ctx(), asker.clone());
        rt().block_on(tool.execute(serde_json::json!({ "question": "What now?" })))
            .unwrap();
        assert!(rt().block_on(asker.asked())[0].options.is_empty());
    }

    #[test]
    fn the_tool_is_not_read_only() {
        let tool = AskUserTool::new(ctx(), ScriptedAsker::new(None));
        assert!(!tool.is_read_only());
    }

    #[test]
    fn answer_rendering_handles_several_selections() {
        let answer = Answer {
            selections: vec!["a".into(), "b".into()],
        };
        assert_eq!(answer.render(), "a, b");
    }
}
