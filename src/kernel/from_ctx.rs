//! Per-frame kernel parameter extraction from the typed read context.
//!
//! `FromCtx` is the one auto-generated `fn` that converts a [`crate::effect::Ctx`],
//! snapshot into a kernel's constant-buffer params struct. It is implemented by
//! `kernel!` for every declared kernel.

use crate::effect::Ctx;
use crate::kernel::params::KernelParams;
use crate::params::ParamsSpec;

pub trait FromCtx: KernelParams {
	type Spec: ParamsSpec;

	fn from_ctx(ctx: &Ctx<Self::Spec>) -> Self;
}
