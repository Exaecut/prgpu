//! `kernel_params!` smoke tests.
//!
//! These do not attempt to drive AE/Premiere — only that the macro emits a
//! struct with the GPU-ABI guarantees the dispatcher relies on
//! (`#[gpu_struct]` layout, `KernelParams` impl, byte size match).

use prgpu::KernelParams;

mod fixtures {
	use after_effects::{InData, OutData, Parameters};
	use std::fmt::Debug;
	use std::hash::Hash;

	#[repr(usize)]
	#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
	pub enum DummyParam {
		Threshold = 0,
		Strength = 1,
	}

	impl From<DummyParam> for usize {
		fn from(p: DummyParam) -> Self {
			p as usize
		}
	}

	impl prgpu::params::SetupParams for DummyParam {
		fn setup(_p: &mut Parameters<Self>, _in_data: InData, _out_data: OutData) -> Result<(), after_effects::Error> {
			Ok(())
		}
	}
}

prgpu::kernel_params! {
	SimpleScalarParams for crate::fixtures::DummyParam {
		threshold: f32 = [float(Threshold)];
		strength:  f32 = [float(Strength)];
		flag:      u32;
		_pad0:     u32;
	}
}

#[test]
fn kernel_params_emits_kernel_params_impl() {
	assert_eq!(<SimpleScalarParams as KernelParams>::SIZE, std::mem::size_of::<SimpleScalarParams>());
	assert_eq!(<SimpleScalarParams as KernelParams>::ALIGN, std::mem::align_of::<SimpleScalarParams>());
}

#[test]
fn kernel_params_size_is_byte_stable_for_4xu32() {
	// 4 × 4 bytes, vec4-aligned.
	assert_eq!(SimpleScalarParams::SIZE, 16);
	assert_eq!(SimpleScalarParams::ALIGN, 16);
}

#[test]
fn kernel_params_struct_is_copy_and_clone() {
	fn assert_copy<T: Copy>() {}
	fn assert_clone<T: Clone>() {}
	assert_copy::<SimpleScalarParams>();
	assert_clone::<SimpleScalarParams>();
}
