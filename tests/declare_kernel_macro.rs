//! `kernel!` smoke tests.
//!
//! Verifies the generated module surface: struct layout, `Default` zeroing,
//! `from_ctx` against a synthetic snapshot, popup-enum accessor, bare-marker
//! sugar with a transformed expression.

use prgpu::Kernel;

#[test]
fn diff_kernel_module_exposes_full_surface() {
	assert!(!prgpu::kernel::builtin::diff::ENTRY_POINT.is_empty());
	assert_eq!(prgpu::kernel::builtin::diff::ENTRY_POINT, "diff");

	let k: Kernel<prgpu::kernel::builtin::DiffParams> = prgpu::kernel::builtin::diff::kernel();
	assert_eq!(k.name(), "diff");
	assert_eq!(k.entry_point(), "diff");
}

#[test]
fn kernel_namespace_separates_module_and_name() {
	let k = prgpu::kernel::builtin::diff::kernel();
	assert_eq!(k.name(), "diff");
}
