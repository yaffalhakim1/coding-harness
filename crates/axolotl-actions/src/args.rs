//! Argument extraction helpers shared by the tool implementations.

use crate::error::{ActionError, ActionResult};

/// Read a required string argument.
pub fn require_str<'a>(arguments: &'a serde_json::Value, key: &str) -> ActionResult<&'a str> {
    arguments
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| ActionError::InvalidArgument(format!("missing '{key}'")))
}

/// Read an optional non-negative integer argument.
pub fn optional_usize(arguments: &serde_json::Value, key: &str) -> ActionResult<Option<usize>> {
    match arguments.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(n)) => {
            n.as_u64().map(|v| Some(v as usize)).ok_or_else(|| {
                ActionError::InvalidArgument(format!("'{key}' must be a non-negative integer"))
            })
        }
        Some(_) => Err(ActionError::InvalidArgument(format!(
            "'{key}' must be an integer"
        ))),
    }
}

/// Read an optional boolean argument.
pub fn optional_bool(arguments: &serde_json::Value, key: &str) -> bool {
    arguments
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

/// Read an optional string argument.
pub fn optional_str<'a>(arguments: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    arguments.get(key).and_then(serde_json::Value::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_required_string_is_an_error() {
        let args = serde_json::json!({});
        assert!(matches!(
            require_str(&args, "path").unwrap_err(),
            ActionError::InvalidArgument(_)
        ));
    }

    #[test]
    fn wrong_type_is_an_error() {
        let args = serde_json::json!({ "path": 3 });
        assert!(require_str(&args, "path").is_err());
    }

    #[test]
    fn optional_usize_accepts_absent_null_and_number() {
        let args = serde_json::json!({ "a": null, "b": 7, "c": "x" });
        assert_eq!(optional_usize(&args, "missing").unwrap(), None);
        assert_eq!(optional_usize(&args, "a").unwrap(), None);
        assert_eq!(optional_usize(&args, "b").unwrap(), Some(7));
        assert!(optional_usize(&args, "c").is_err());
    }

    #[test]
    fn optional_bool_defaults_false() {
        assert!(!optional_bool(&serde_json::json!({}), "flag"));
        assert!(optional_bool(&serde_json::json!({ "flag": true }), "flag"));
        assert!(!optional_bool(
            &serde_json::json!({ "flag": "yes" }),
            "flag"
        ));
    }

    #[test]
    fn optional_str_returns_none_for_missing() {
        assert_eq!(optional_str(&serde_json::json!({}), "x"), None);
        assert_eq!(
            optional_str(&serde_json::json!({ "x": "v" }), "x"),
            Some("v")
        );
    }
}
