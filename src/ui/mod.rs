//! Reusable UI building blocks for effect parameter panels.
//!
//! Each submodule exposes a `add_*_param` helper that wraps the standard
//! AE/Premiere popup/slider/etc. construction so individual effects don't
//! repeat boilerplate, and a typed value reader for the GPU/CPU param paths.

pub mod blend_mode;

pub use blend_mode::{add_blend_mode_param, BlendMode, BLEND_MODE_OPTIONS};
