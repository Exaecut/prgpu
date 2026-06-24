//! In-frame text rendering for GPU effect overlays.
//!
//! A single SDF glyph atlas is built from an embedded font (fontdue) on first
//! use and uploaded once per GPU device. Drawing happens in a dedicated,
//! bounding-box-limited compute pass — the vekl `DrawString` device function
//! maps each `char` to its atlas glyph and composites the SDF coverage onto the
//! destination frame. No quads, no vertex/geometry stages; pure compute.
//!
//! Effects call the high-level host API ([`draw`]) from their render path; the
//! atlas binding, layout and dispatch are handled internally.

mod atlas;
mod gpu;
mod sdf;

pub use atlas::{Atlas, GlyphMetric, build_atlas, build_default_atlas, EMBEDDED_FONT, FIRST_CHAR, GLYPH_COUNT, LAST_CHAR};
pub use gpu::{draw, TextSpec};
