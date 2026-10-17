//! External store for elided content (paper §2.3, mechanism M2).
//!
//! "Recall (M2) stores elided observations in the file system and exposes a
//! `recall_event` tool to read them back on demand, making elision reversible."
//!
//! This is a plain id → text map. It lives in `harness-models` because both
//! `harness-context` (which writes on elision) and `harness-actions` (which
//! exposes `recall_event`) depend on it, and neither should depend on the other.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use serde::{Deserialize, Serialize};

/// One stored event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEvent {
    /// Short label for the origin, e.g. the tool name.
    pub label: String,
    /// Verbatim content.
    pub content: String,
}

/// Shared, cloneable store of elided content.
#[derive(Debug, Clone, Default)]
pub struct EventStore {
    inner: Arc<Mutex<HashMap<String, StoredEvent>>>,
}

impl EventStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `content` under `id`, replacing any previous value.
    pub fn insert(
        &self,
        id: impl Into<String>,
        label: impl Into<String>,
        content: impl Into<String>,
    ) {
        let mut guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        guard.insert(
            id.into(),
            StoredEvent {
                label: label.into(),
                content: content.into(),
            },
        );
    }

    /// Fetch an event by id.
    pub fn get(&self, id: &str) -> Option<StoredEvent> {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    /// Whether an id is present.
    pub fn contains(&self, id: &str) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .contains_key(id)
    }

    /// Number of stored events.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ids in insertion-independent (sorted) order, for diagnostics.
    pub fn ids(&self) -> Vec<String> {
        let guard = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        let mut ids: Vec<String> = guard.keys().cloned().collect();
        ids.sort();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_round_trip() {
        let store = EventStore::new();
        store.insert("e1", "bash", "full output here");
        let event = store.get("e1").expect("present");
        assert_eq!(event.label, "bash");
        assert_eq!(event.content, "full output here");
    }

    #[test]
    fn missing_id_returns_none() {
        let store = EventStore::new();
        assert!(store.get("nope").is_none());
        assert!(!store.contains("nope"));
    }

    #[test]
    fn insert_replaces() {
        let store = EventStore::new();
        store.insert("e1", "bash", "first");
        store.insert("e1", "bash", "second");
        assert_eq!(store.get("e1").unwrap().content, "second");
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn clone_shares_state() {
        let store = EventStore::new();
        let clone = store.clone();
        clone.insert("e1", "tool", "x");
        assert!(
            store.contains("e1"),
            "clones must share the same backing map"
        );
    }

    #[test]
    fn ids_are_sorted() {
        let store = EventStore::new();
        store.insert("b", "t", "x");
        store.insert("a", "t", "y");
        assert_eq!(store.ids(), vec!["a", "b"]);
    }
}
