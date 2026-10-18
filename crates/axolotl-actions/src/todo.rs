//! The `todo` tool: a structured task list the agent maintains.
//!
//! `update_plan` takes free text and is re-injected as a system reminder. A todo
//! list is different: it is *structured*, so a client can render it as a real
//! checklist and the harness can tell whether work is actually finished.
//!
//! That second property is what makes it worth having — an agent that ends a
//! turn with items still `pending` has not finished, and saying so is cheaper
//! than discovering it from a failed test.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use axolotl_models::types::ToolDefinition;

use crate::error::{ActionError, ActionResult};
use crate::registry::{Action, ToolContext};

/// One todo item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Todo {
    /// Imperative description, e.g. "Fix the parser in foo.py".
    pub content: String,
    pub status: TodoStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Completed,
}

impl TodoStatus {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "pending" => Some(Self::Pending),
            "in_progress" | "in-progress" | "inprogress" => Some(Self::InProgress),
            "completed" | "done" => Some(Self::Completed),
            _ => None,
        }
    }
}

/// The shared list.
#[derive(Debug, Clone, Default)]
pub struct TodoList {
    inner: Arc<Mutex<Vec<Todo>>>,
}

impl TodoList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self, todos: Vec<Todo>) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = todos;
        }
    }

    pub fn get(&self) -> Vec<Todo> {
        self.inner.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Whether any item is unfinished.
    ///
    /// An empty list counts as finished: the agent never made a list, so there
    /// is nothing outstanding to hold against it.
    pub fn has_unfinished(&self) -> bool {
        self.get().iter().any(|t| t.status != TodoStatus::Completed)
    }

    /// Items still to do, for a nudge message.
    pub fn unfinished(&self) -> Vec<Todo> {
        self.get()
            .into_iter()
            .filter(|t| t.status != TodoStatus::Completed)
            .collect()
    }

    /// Render as the checklist the model sees.
    pub fn render(&self) -> String {
        let todos = self.get();
        if todos.is_empty() {
            return String::new();
        }
        todos
            .iter()
            .map(|t| {
                let mark = match t.status {
                    TodoStatus::Pending => "[ ]",
                    TodoStatus::InProgress => "[~]",
                    TodoStatus::Completed => "[x]",
                };
                format!("{mark} {}", t.content)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// The `todo` tool.
pub struct TodoTool {
    list: TodoList,
    _ctx: ToolContext,
}

impl TodoTool {
    pub fn new(ctx: ToolContext, list: TodoList) -> Self {
        Self { list, _ctx: ctx }
    }
}

#[async_trait]
impl Action for TodoTool {
    fn name(&self) -> &str {
        "todo"
    }

    fn is_read_only(&self) -> bool {
        // Mutates session state, so it must not run in a read-only batch.
        false
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "todo".to_string(),
            description: "Create or update your task list. The harness shows it back to you before every turn and refuses to call the work finished while items are still open.\n\nWhen: at the start of any task with 3 or more steps, and whenever a step's status changes. Skip it for a single trivial action.\n\nProtocol: pass the COMPLETE list every time — it replaces the previous one. Each item has `content` (imperative) and `status` (`pending` | `in_progress` | `completed`). Keep exactly one item `in_progress`; mark items `completed` as soon as they are actually done, not in a batch at the end.\n\nSide effects: in-memory only — no files are touched.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The complete list, replacing any previous one",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string" },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"]
                                }
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"],
                "additionalProperties": false
            }),
        }
    }

    async fn execute(&self, arguments: serde_json::Value) -> ActionResult<String> {
        let items = arguments
            .get("todos")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| ActionError::InvalidArgument("missing 'todos'".into()))?;

        let mut todos = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            let content = item
                .get("content")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    ActionError::InvalidArgument(format!("item {index} has no 'content'"))
                })?;
            let raw = item
                .get("status")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    ActionError::InvalidArgument(format!("item {index} has no 'status'"))
                })?;
            let status = TodoStatus::parse(raw).ok_or_else(|| {
                ActionError::InvalidArgument(format!(
                    "item {index} has unknown status '{raw}' (expected pending, in_progress or completed)"
                ))
            })?;

            if content.trim().is_empty() {
                return Err(ActionError::InvalidArgument(format!(
                    "item {index} has empty content"
                )));
            }
            todos.push(Todo {
                content: content.trim().to_string(),
                status,
            });
        }

        let in_progress = todos
            .iter()
            .filter(|t| t.status == TodoStatus::InProgress)
            .count();
        if in_progress > 1 {
            return Err(ActionError::InvalidArgument(format!(
                "{in_progress} items are in_progress; keep exactly one"
            )));
        }

        self.list.set(todos);
        let done = self
            .list
            .get()
            .iter()
            .filter(|t| t.status == TodoStatus::Completed)
            .count();
        let total = self.list.get().len();

        Ok(format!(
            "Todo list updated: {done}/{total} complete.\n{}",
            self.list.render()
        ))
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

    fn item(content: &str, status: &str) -> serde_json::Value {
        serde_json::json!({ "content": content, "status": status })
    }

    #[test]
    fn stores_a_list_and_reports_progress() {
        let list = TodoList::new();
        let tool = TodoTool::new(ctx(), list.clone());

        let out = rt()
            .block_on(tool.execute(serde_json::json!({
                "todos": [item("read the file", "completed"), item("edit it", "in_progress")]
            })))
            .unwrap();

        assert!(out.contains("1/2 complete"), "{out}");
        assert_eq!(list.get().len(), 2);
    }

    #[test]
    fn replaces_the_previous_list() {
        let list = TodoList::new();
        let tool = TodoTool::new(ctx(), list.clone());
        rt().block_on(tool.execute(serde_json::json!({
            "todos": [item("first", "pending")]
        })))
        .unwrap();
        rt().block_on(tool.execute(serde_json::json!({
            "todos": [item("second", "pending")]
        })))
        .unwrap();
        assert_eq!(list.get()[0].content, "second");
    }

    #[test]
    fn more_than_one_in_progress_is_rejected() {
        let tool = TodoTool::new(ctx(), TodoList::new());
        let err = rt()
            .block_on(tool.execute(serde_json::json!({
                "todos": [item("a", "in_progress"), item("b", "in_progress")]
            })))
            .unwrap_err();
        assert!(err.to_string().contains("exactly one"), "{err}");
    }

    #[test]
    fn an_unknown_status_is_rejected_with_the_expected_values() {
        let tool = TodoTool::new(ctx(), TodoList::new());
        let err = rt()
            .block_on(tool.execute(serde_json::json!({
                "todos": [item("a", "finished")]
            })))
            .unwrap_err();
        assert!(err.to_string().contains("pending"), "{err}");
    }

    #[test]
    fn unfinished_tracking_drives_the_completion_check() {
        let list = TodoList::new();
        list.set(vec![
            Todo {
                content: "a".into(),
                status: TodoStatus::Completed,
            },
            Todo {
                content: "b".into(),
                status: TodoStatus::Pending,
            },
        ]);
        assert!(list.has_unfinished());
        assert_eq!(list.unfinished().len(), 1);

        list.set(vec![Todo {
            content: "a".into(),
            status: TodoStatus::Completed,
        }]);
        assert!(!list.has_unfinished());
    }

    #[test]
    fn an_empty_list_is_not_unfinished() {
        // The agent never made a list, so nothing is outstanding.
        assert!(!TodoList::new().has_unfinished());
    }

    #[test]
    fn rendering_uses_checkbox_marks() {
        let list = TodoList::new();
        list.set(vec![
            Todo {
                content: "done".into(),
                status: TodoStatus::Completed,
            },
            Todo {
                content: "doing".into(),
                status: TodoStatus::InProgress,
            },
            Todo {
                content: "later".into(),
                status: TodoStatus::Pending,
            },
        ]);
        let rendered = list.render();
        assert!(rendered.contains("[x] done"), "{rendered}");
        assert!(rendered.contains("[~] doing"), "{rendered}");
        assert!(rendered.contains("[ ] later"), "{rendered}");
    }

    #[test]
    fn status_parsing_accepts_the_documented_spellings() {
        assert_eq!(TodoStatus::parse("pending"), Some(TodoStatus::Pending));
        assert_eq!(
            TodoStatus::parse("in_progress"),
            Some(TodoStatus::InProgress)
        );
        assert_eq!(
            TodoStatus::parse("in-progress"),
            Some(TodoStatus::InProgress)
        );
        assert_eq!(TodoStatus::parse("done"), Some(TodoStatus::Completed));
        assert_eq!(TodoStatus::parse("nope"), None);
    }

    #[test]
    fn a_missing_list_is_rejected() {
        let tool = TodoTool::new(ctx(), TodoList::new());
        let err = rt()
            .block_on(tool.execute(serde_json::json!({})))
            .unwrap_err();
        assert!(matches!(err, ActionError::InvalidArgument(_)), "{err:?}");
    }
}
