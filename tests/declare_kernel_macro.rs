//! `declare_kernel!` smoke tests.
//!
//! Verifies the module form coexists with the deprecated top-level wrappers
//! and that the public surface promised in the plan §13 is reachable.
//!
//! `declare_kernel!(diff, ...)` is invoked inside `kernels/diff.rs`, so the
//! generated module sits at `prgpu::kernels::diff::diff::*`. Effects that
//! invoke `declare_kernel!` directly inside `kernel.rs` get the cleaner
//! `<crate>::kernel::<name>::*` path.

use prgpu::Kernel;

#[test]
fn diff_kernel_module_exposes_full_surface() {
	assert!(!prgpu::kernels::diff::diff::ENTRY_POINT.is_empty());
	assert_eq!(prgpu::kernels::diff::diff::ENTRY_POINT, "diff");

	let k: Kernel<prgpu::kernels::diff::DiffParams> = prgpu::kernels::diff::diff::kernel();
	assert_eq!(k.name(), "diff");
	assert_eq!(k.entry_point(), "diff");

	let _: prgpu::cpu::render::CpuDispatchFn = prgpu::kernels::diff::diff::CPU_DISPATCH;
	let _: prgpu::cpu::render::CpuDispatchTileFn = prgpu::kernels::diff::diff::CPU_DISPATCH_TILE;
}

#[test]
fn diff_kernel_legacy_aliases_still_resolve() {
	#[allow(deprecated)]
	let _: prgpu::cpu::render::CpuDispatchTileFn = prgpu::kernels::diff::DIFF_CPU_DISPATCH_TILE;
}

#[test]
fn kernel_namespace_separates_module_and_fn() {
	// The deprecated top-level fn `diff` and the generated module `diff` share
	// a name in different namespaces. Path syntax must still reach the module.
	let k = prgpu::kernels::diff::diff::kernel();
	assert_eq!(k.name(), "diff");
}
