//! Planning strategy trait.
//!
//! Paper §2.1: planning is "an explicit, persistent representation of task
//! progress maintained by the model". This crate generates the *initial* plan;
//! the model maintains it through `update_plan` during execution.

use anyhow::Result;
use async_trait::async_trait;

use axolotl_models::provider::LLMProvider;
use axolotl_models::types::ModelConfig;

/// Strategy for generating execution plans.
///
/// Takes the task description rather than a `Task` type so this crate stays
/// independent of `axolotl-core` (which depends on it).
#[async_trait]
pub trait PlanningStrategy: Send + Sync {
    /// Generate a plan for the given task description.
    async fn generate_plan(
        &self,
        task_description: &str,
        llm: &dyn LLMProvider,
        config: &ModelConfig,
    ) -> Result<String>;

    /// Split a plan into its steps.
    ///
    /// Recognises the numbered list the default planner asks for; falls back to
    /// non-empty lines so a differently-formatted plan is still usable.
    fn parse_steps(&self, plan: &str) -> Vec<String> {
        let numbered: Vec<String> = plan
            .lines()
            .map(str::trim)
            .filter(|l| {
                l.chars().next().is_some_and(|c| c.is_ascii_digit()) && l.contains('.')
                    || l.starts_with("- ")
            })
            .map(str::to_string)
            .collect();

        if numbered.len() >= 2 {
            numbered
        } else {
            plan.lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyPlanner;

    #[async_trait]
    impl PlanningStrategy for DummyPlanner {
        async fn generate_plan(
            &self,
            _task_description: &str,
            _llm: &dyn LLMProvider,
            _config: &ModelConfig,
        ) -> Result<String> {
            Ok(String::new())
        }
    }

    #[test]
    fn parses_numbered_steps() {
        let plan = "1. Read src/main.rs\n2. Fix the bug\n3. Run tests";
        assert_eq!(DummyPlanner.parse_steps(plan).len(), 3);
    }

    #[test]
    fn falls_back_to_non_empty_lines() {
        let plan = "Look around\n\nMake an edit";
        assert_eq!(
            DummyPlanner.parse_steps(plan),
            vec!["Look around", "Make an edit"]
        );
    }

    #[test]
    fn parses_dash_bullets() {
        let plan = "- Read\n- Edit\n- Verify";
        assert_eq!(DummyPlanner.parse_steps(plan).len(), 3);
    }
}
