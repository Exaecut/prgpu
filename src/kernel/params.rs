//! GPU-ABI marker for per-pass kernel parameter structs.
//!
//! Every type that the dispatcher hands to a generated kernel as
//! `ConstantBuffer<UserParams>` must implement [`KernelParams`]. The trait
//! carries the layout invariants the host relies on:
//!
//! - byte-stable size and alignment (via the `gpu_struct` machinery),
//! - `Copy + 'static` so the host can move the struct into a constant buffer
//!   without lifetime gymnastics,
//! - `Sync` so rayon worker threads can share a raw pointer to the params,
//! - no implicit padding that the GPU side cannot account for.
//!
//! `kernel_params! { ... }` auto-implements this trait. Manually-written
//! constant-buffer structs should annotate the type with
//! `#[prgpu::gpu_struct]` and then
//! `impl KernelParams for MyParams { const SIZE = Self::SIZE; const ALIGN = Self::ALIGN; }`.

/// Marker for a `#[repr(C)]` / `gpu_struct`-laid-out struct safe to upload
/// as a Slang `ConstantBuffer<T>` via the prgpu dispatcher.
///
/// The two associated constants must equal the values emitted by
/// `#[gpu_struct]`; mismatch will trip the `const _` size/align asserts the
/// `gpu_struct` macro plants next to the struct.
///
/// `Sync` is required because the CPU dispatcher (`render_cpu_direct`)
/// shares the params struct across rayon worker threads via a raw pointer.
/// All `#[gpu_struct]` types are Sync by construction (only scalar fields).
pub trait KernelParams: Copy + Sync + Sized + 'static {
	const SIZE: usize;
	const ALIGN: usize;
}
