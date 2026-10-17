//! Live event stream from the agent loop.
//!
//! The loop reports what it is doing through a [`ProgressSink`], so a front end
//! (TUI, JSON log, test recorder) can observe a run without duplicating the
//! loop. The default sink discards everything, so this costs nothing when
//! unused.

use std::sync::Arc;

use harness_models::types::TokenUsage;

/// Something the agent did.
#[derive(Debug, Clone)]
pub enum Progress {
    /// A session started.
    SessionStarted {
        session_id: String,
        task_id: String,
        resumed_turns: usize,
    },
    /// The model was called and answered.
    TurnCompleted {
        turn: u32,
        tool_calls: usize,
        usage: TokenUsage,
        /// The assistant's text, when it wrote any.
        text: String,
    },
    /// A tool call began.
    ToolStarted { name: String, arguments: String },
    /// A tool call finished.
    ToolFinished {
        name: String,
        success: bool,
        duration_ms: u64,
        /// First line of the output, for a live view.
        preview: String,
    },
    /// Context management compacted the history.
    ContextManaged {
        tokens_before: usize,
        tokens_after: usize,
    },
    /// The session ended.
    SessionFinished {
        outcome: String,
        turns: usize,
        total_tokens: usize,
    },
}

/// Receives [`Progress`] events.
///
/// Implementations must be cheap: `emit` is called inside the loop's hot path.
pub trait ProgressSink: Send + Sync {
    fn emit(&self, event: Progress);
}

/// Discards every event.
pub struct NoopSink;

impl ProgressSink for NoopSink {
    fn emit(&self, _event: Progress) {}
}

/// Records events in memory, for tests.
#[derive(Default)]
pub struct RecordingSink {
    events: std::sync::Mutex<Vec<Progress>>,
}

impl RecordingSink {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Events recorded so far.
    pub fn events(&self) -> Vec<Progress> {
        self.events.lock().map(|e| e.clone()).unwrap_or_default()
    }

    /// Whether any event matching `predicate` was seen.
    pub fn saw(&self, predicate: impl Fn(&Progress) -> bool) -> bool {
        self.events
            .lock()
            .map(|events| events.iter().any(&predicate))
            .unwrap_or(false)
    }
}

impl ProgressSink for RecordingSink {
    fn emit(&self, event: Progress) {
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
    }
}

/// A sink that forwards to an `mpsc` channel, for a UI on another task.
pub struct ChannelSink {
    sender: tokio::sync::mpsc::UnboundedSender<Progress>,
}

impl ChannelSink {
    pub fn new() -> (Arc<Self>, tokio::sync::mpsc::UnboundedReceiver<Progress>) {
        let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
        (Arc::new(Self { sender }), receiver)
    }
}

impl ProgressSink for ChannelSink {
    fn emit(&self, event: Progress) {
        // A dropped receiver means the UI closed; the run continues regardless.
        let _ = self.sender.send(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recording_sink_captures_events() {
        let sink = RecordingSink::new();
        sink.emit(Progress::ToolStarted {
            name: "bash".into(),
            arguments: "{}".into(),
        });
        assert_eq!(sink.events().len(), 1);
        assert!(sink.saw(|e| matches!(e, Progress::ToolStarted { name, .. } if name == "bash")));
    }

    #[test]
    fn recording_sink_answers_negative_queries() {
        let sink = RecordingSink::new();
        assert!(!sink.saw(|e| matches!(e, Progress::SessionFinished { .. })));
    }

    #[test]
    fn noop_sink_accepts_anything() {
        NoopSink.emit(Progress::SessionFinished {
            outcome: "success".into(),
            turns: 1,
            total_tokens: 10,
        });
    }

    #[tokio::test]
    async fn channel_sink_delivers() {
        let (sink, mut rx) = ChannelSink::new();
        sink.emit(Progress::SessionFinished {
            outcome: "success".into(),
            turns: 2,
            total_tokens: 20,
        });
        let event = rx.recv().await.expect("event");
        match event {
            Progress::SessionFinished { turns, .. } => assert_eq!(turns, 2),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_dropped_receiver_does_not_panic() {
        let (sink, rx) = ChannelSink::new();
        drop(rx);
        sink.emit(Progress::SessionFinished {
            outcome: "x".into(),
            turns: 0,
            total_tokens: 0,
        });
    }
}
