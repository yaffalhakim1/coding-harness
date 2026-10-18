//! Threshold configuration for context management.
//!
//! Values from paper §3.1 (Implementation Details):
//! "soft and hard context thresholds are 0.6 and 0.85 of the usable window, with
//! the verbatim recent window budgeted at 0.3 and floored at two turns."

/// Soft (B1) and hard (B2) thresholds plus the recent-window budget.
#[derive(Debug, Clone, Copy)]
pub struct Thresholds {
    /// B1: fraction of capacity at which elision starts.
    pub soft_ratio: f32,
    /// B2: fraction of capacity at which summarization starts.
    pub hard_ratio: f32,
    /// Fraction of capacity kept verbatim as recent turns.
    pub recent_ratio: f32,
    /// Minimum number of recent turns kept verbatim.
    pub recent_turns: usize,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            soft_ratio: 0.6,
            hard_ratio: 0.85,
            recent_ratio: 0.3,
            recent_turns: 2,
        }
    }
}

impl Thresholds {
    /// Soft threshold in tokens.
    pub fn soft(&self, capacity: usize) -> usize {
        scale(capacity, self.soft_ratio)
    }

    /// Hard threshold in tokens.
    pub fn hard(&self, capacity: usize) -> usize {
        scale(capacity, self.hard_ratio)
    }

    /// Token budget for the verbatim recent window.
    pub fn recent_budget(&self, capacity: usize) -> usize {
        scale(capacity, self.recent_ratio)
    }
}

fn scale(capacity: usize, ratio: f32) -> usize {
    let ratio = ratio.clamp(0.0, 1.0);
    (capacity as f32 * ratio) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_paper() {
        let t = Thresholds::default();
        assert!((t.soft_ratio - 0.6).abs() < f32::EPSILON);
        assert!((t.hard_ratio - 0.85).abs() < f32::EPSILON);
        assert!((t.recent_ratio - 0.3).abs() < f32::EPSILON);
        assert_eq!(t.recent_turns, 2);
    }

    #[test]
    fn thresholds_scale_with_capacity() {
        let t = Thresholds::default();
        assert_eq!(t.soft(128_000), 76_800);
        assert_eq!(t.hard(128_000), 108_800);
        assert_eq!(t.soft(32_000), 19_200);
    }

    #[test]
    fn soft_is_below_hard() {
        let t = Thresholds::default();
        for capacity in [8_000usize, 32_000, 64_000, 96_000, 128_000] {
            assert!(t.soft(capacity) < t.hard(capacity));
        }
    }

    #[test]
    fn zero_capacity_is_safe() {
        let t = Thresholds::default();
        assert_eq!(t.soft(0), 0);
        assert_eq!(t.hard(0), 0);
    }
}
