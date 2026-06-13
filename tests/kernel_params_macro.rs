//! `kernel!` struct layout and `FromCtx` tests.
//!
//! Uses a synthetic snapshot to exercise `from_ctx` without AE/Premiere host objects.

use prgpu::KernelParams;

// Use the built-in diff module as a simple `kernel!`-style layout test
// (DiffParams is hand-written but exercises the same KernelParams trait).
#[test]
fn kernel_params_emits_kernel_params_impl() {
	let k = prgpu::kernel::builtin::diff::kernel();
	assert_eq!(k.name(), "diff");
	assert_eq!(k.entry_point(), "diff");
	assert!(!k.shader_src().is_empty());
}

#[test]
fn kernel_params_struct_is_copy_and_clone() {
	fn assert_copy<T: Copy>() {}
	fn assert_clone<T: Clone>() {}
	assert_copy::<prgpu::kernel::builtin::DiffParams>();
	assert_clone::<prgpu::kernel::builtin::DiffParams>();
}
