//! `declare_kernel!` smoke tests.
//!
//! Verifies the generated module surface promised by the macro is reachable.
//! `declare_kernel!(diff, ...)` is invoked inside `kernel/builtin/mod.rs`,
//! so the generated module sits at `prgpu::kernel::builtin::diff::*`.

use prgpu::Kernel;

#[test]
fn diff_kernel_module_exposes_full_surface() {
	assert!(!prgpu::kernel::builtin::diff::ENTRY_POINT.is_empty());
	assert_eq!(prgpu::kernel::builtin::diff::ENTRY_POINT, "diff");

	let k: Kernel<prgpu::kernel::builtin::DiffParams> = prgpu::kernel::builtin::diff::kernel();
	assert_eq!(k.name(), "diff");
	assert_eq!(k.entry_point(), "diff");

	let _: prgpu::cpu::render::CpuDispatchFn = prgpu::kernel::builtin::diff::CPU_DISPATCH;
	let _: prgpu::cpu::render::CpuDispatchTileFn = prgpu::kernel::builtin::diff::CPU_DISPATCH_TILE;
}

#[test]
fn kernel_namespace_separates_module_and_name() {
	let k = prgpu::kernel::builtin::diff::kernel();
	assert_eq!(k.name(), "diff");
}
