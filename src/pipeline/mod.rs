//! Host-side pipeline helpers.
//!
//! Kernels run on the GPU (or the rayon CPU pool); this module holds the
//! host-side glue that wires those kernels together — mip-pyramid
//! allocation + level-0 copy + per-level downsample, source-snapshot
//! handling, sigma → downsample-resolution math for separable blurs,
//! sample-count math for radial / angular sweeps. None of these are
//! kernels themselves; they're the orchestration code that drives kernels.
//!
//! Effects compose these helpers from `lib.rs` / `gpu.rs`. The graph
//! executor (`prgpu::graph::execute`) also uses them internally.

pub mod blur;
pub mod mip;
pub mod sweep;
