//! Provides GPU context management, buffer upload/download, built-in media,
//! and output writers so effect crates can write integration tests in
//! `tests/*.rs` that exercise the real GPU kernel path without opening
//! Premiere Pro or After Effects.

pub mod compare;
pub mod context;
pub mod media;
pub mod output;
pub mod scene;
pub mod runner;

pub use compare::{DiffConfig, DiffReport, compute_metrics, diff_heatmap_gpu, write_heatmap_png, write_report_json, write_report_txt};
pub use context::GpuContext;
pub use media::{builtin_checkerboard, builtin_solid_color, builtin_gradient_h};
pub use output::write_png;
pub use scene::{Media, Scene, Layer, Transform, Timeline, Background};
pub use runner::{RenderTest, OutputSpec, ExecutionTarget, RenderResult, DiffPolicy, ComparisonSpec};
