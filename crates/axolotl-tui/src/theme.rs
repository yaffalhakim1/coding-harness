//! The colour palette.
//!
//! Warm and rust-toned rather than the usual cold blue-gray: the accent is
//! Rust's own orange, the neutrals are warm dark, and the semantic colours are
//! desaturated so a transcript full of tool output stays readable.
//!
//! Truecolor is assumed. A terminal without it approximates these values, which
//! is acceptable — nothing here carries meaning through hue alone.

use ratatui::style::{Color, Modifier, Style};

/// Rust orange, the one accent colour.
pub const ACCENT: Color = Color::Rgb(0xCE, 0x42, 0x2B);
/// A lighter accent for secondary emphasis.
pub const ACCENT_SOFT: Color = Color::Rgb(0xE0, 0x7A, 0x5F);

/// Primary text.
pub const TEXT: Color = Color::Rgb(0xE8, 0xE6, 0xE3);
/// Secondary text: timings, counts, hints.
pub const DIM: Color = Color::Rgb(0x8B, 0x86, 0x80);
/// Tertiary text: rules, disabled hints.
pub const MUTED: Color = Color::Rgb(0x5C, 0x58, 0x54);

/// Success.
pub const OK: Color = Color::Rgb(0x7F, 0xB0, 0x69);
/// Failure.
pub const ERR: Color = Color::Rgb(0xD1, 0x49, 0x5B);

/// Body text.
pub fn text() -> Style {
    Style::default().fg(TEXT)
}

/// Metadata that should recede.
pub fn dim() -> Style {
    Style::default().fg(DIM)
}

/// Barely-there chrome: rules, separators.
pub fn muted() -> Style {
    Style::default().fg(MUTED)
}

/// The accent, for the marker beside an action.
pub fn accent() -> Style {
    Style::default().fg(ACCENT)
}

/// A successful outcome.
pub fn ok() -> Style {
    Style::default().fg(OK)
}

/// A failed outcome.
pub fn err() -> Style {
    Style::default().fg(ERR)
}

/// A section heading: accent, bold.
pub fn heading() -> Style {
    Style::default().fg(ACCENT).add_modifier(Modifier::BOLD)
}

/// Emphasis on body text without changing hue.
pub fn strong() -> Style {
    Style::default().fg(TEXT).add_modifier(Modifier::BOLD)
}

/// The horizontal rule used above the status bar.
pub const RULE: &str = "─";

#[cfg(test)]
mod tests {
    use super::*;

    /// The palette must stay distinguishable: two semantic colours with the
    /// same value would make a failure look like a success.
    #[test]
    fn semantic_colours_are_distinct() {
        assert_ne!(OK, ERR);
        assert_ne!(ACCENT, TEXT);
        assert_ne!(TEXT, DIM);
        assert_ne!(DIM, MUTED);
    }

    #[test]
    fn styles_carry_their_colour() {
        assert_eq!(accent().fg, Some(ACCENT));
        assert_eq!(ok().fg, Some(OK));
        assert_eq!(err().fg, Some(ERR));
        assert_eq!(dim().fg, Some(DIM));
    }

    #[test]
    fn the_heading_is_bold_and_accented() {
        let style = heading();
        assert_eq!(style.fg, Some(ACCENT));
        assert!(style.add_modifier.contains(Modifier::BOLD));
    }

    /// The accent is Rust orange: red-dominant, and warm (green above blue).
    #[test]
    fn the_accent_is_rust_orange() {
        let Color::Rgb(r, green, blue) = ACCENT else {
            panic!("accent must be an explicit RGB value");
        };
        assert!(r > green && r > blue, "orange must be red-dominant");
        assert!(green > blue, "warm: green above blue");
    }

    /// The neutrals must be warm too, or the palette fights itself.
    #[test]
    fn the_neutrals_are_warm() {
        for (name, colour) in [("TEXT", TEXT), ("DIM", DIM), ("MUTED", MUTED)] {
            let Color::Rgb(r, _g, b) = colour else {
                panic!("{name} must be RGB");
            };
            assert!(r >= b, "{name} should not be blue-tinted");
        }
    }
}
