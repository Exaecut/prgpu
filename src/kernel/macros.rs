/// Paste-based extern block for the per-pixel and per-tile C ABI dispatch
/// functions slangc emits for a kernel. Used by the `kernel!` proc-macro.
#[macro_export]
macro_rules! __kernel_dispatch_externs {
	($name:ident) => {
		$crate::paste::paste! {
			#[doc(hidden)]
			unsafe extern "C" {
				pub fn [<$name _cpu_dispatch>](
					gid_x: u32,
					gid_y: u32,
					buffers: *const *const ::core::ffi::c_void,
					transition_params: *const ::core::ffi::c_void,
					user_params: *const ::core::ffi::c_void,
				);

				/// Tile entry: loops `y ∈ [y0, y1) × x ∈ [0, width)` in C.
				pub fn [<$name _cpu_dispatch_tile>](
					y0: u32,
					y1: u32,
					width: u32,
					buffers: *const *const ::core::ffi::c_void,
					transition_params: *const ::core::ffi::c_void,
					user_params: *const ::core::ffi::c_void,
				);
			}
		}
	};
}
