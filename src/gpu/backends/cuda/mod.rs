use after_effects::log;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::time::Duration;

use cudarc::driver::sys::{self as cuda, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyHtoD_v2, CUdeviceptr, CUresult};

pub mod buffer;
pub mod fence;
pub mod pipeline;

use crate::{Configuration, FrameParams};

#[inline]
fn check(res: cuda::CUresult, what: &str) -> Result<(), &'static str> {
	if res == cuda::CUresult::CUDA_SUCCESS {
		return Ok(());
	}
	let mut err_str: *const i8 = std::ptr::null();
	unsafe { cuda::cuGetErrorString(res, &mut err_str) };
	let msg = if err_str.is_null() {
		what.to_string()
	} else {
		unsafe { std::ffi::CStr::from_ptr(err_str).to_string_lossy().to_string() }
	};
	log::error!("[CUDA] {what} failed: {msg}");
	Err("CUDA error")
}

#[inline]
#[allow(dead_code)]
unsafe fn compute_capability(dev: cuda::CUdevice) -> Result<(i32, i32), &'static str> {
	let mut major = 0;
	let mut minor = 0;
	check(
		unsafe { cuda::cuDeviceGetAttribute(&mut major, cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev) },
		"cuDeviceGetAttribute(MAJOR)",
	)?;
	check(
		unsafe { cuda::cuDeviceGetAttribute(&mut minor, cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev) },
		"cuDeviceGetAttribute(MINOR)",
	)?;
	Ok((major, minor))
}

/// Launch a CUDA kernel on `stream`. Does NOT synchronize.
///
/// # Safety
/// - `ctx`, `stream`, `func` must be valid CUDA handles.
/// - `params` must point to device memory matching the kernel signature.
#[allow(clippy::too_many_arguments)]
unsafe fn dispatch(
	ctx: *mut c_void,
	stream: *mut c_void,
	func: cuda::CUfunction,
	grid_x: u32,
	grid_y: u32,
	block_x: u32,
	block_y: u32,
	params: &mut [*mut c_void],
) -> Result<(), &'static str> {
	if ctx.is_null() || stream.is_null() || func.is_null() {
		log::error!("[CUDA] dispatch - null handle");
		return Err("null handle");
	}
	check(unsafe { cuda::cuCtxSetCurrent(ctx as cuda::CUcontext) }, "cuCtxSetCurrent")?;
	check(
		unsafe {
			cuda::cuLaunchKernel(
				func,
				grid_x,
				grid_y,
				1,
				block_x,
				block_y,
				1,
				0,
				stream as cuda::CUstream,
				params.as_mut_ptr(),
				std::ptr::null_mut(),
			)
		},
		"cuLaunchKernel",
	)?;
	Ok(())
}

pub unsafe fn log_device_ptr_info(tag: &str, ptr: *mut c_void) {
	if ptr.is_null() {
		log::error!("[cuda] {tag}: null");
		return;
	}
	let mut mem_type: i32 = 0;
	let _ = unsafe {
		cuda::cuPointerGetAttribute(
			&mut mem_type as *mut _ as *mut c_void,
			cuda::CUpointer_attribute_enum::CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
			ptr as u64,
		)
	};
	log::info!("[cuda] {tag}: CUdeviceptr={ptr:?}, memory_type={mem_type}");
}

/// Allocate device memory and synchronously upload `bytes` into it.
/// Caller owns the returned device pointer and must free it with `cuMemFree_v2`.
unsafe fn upload_to_device(bytes: &[u8]) -> Result<CUdeviceptr, &'static str> {
	let mut devptr: CUdeviceptr = 0;
	let alloc = unsafe { cuMemAlloc_v2(&mut devptr, bytes.len()) };
	if alloc != CUresult::CUDA_SUCCESS {
		log::error!("[CUDA] cuMemAlloc_v2 ({} bytes) failed: {:?}", bytes.len(), alloc);
		return Err("cuMemAlloc_v2 failed");
	}
	let copy = unsafe { cuMemcpyHtoD_v2(devptr, bytes.as_ptr() as *const c_void, bytes.len()) };
	if copy != CUresult::CUDA_SUCCESS {
		unsafe { cuMemFree_v2(devptr) };
		log::error!("[CUDA] cuMemcpyHtoD_v2 ({} bytes) failed: {:?}", bytes.len(), copy);
		return Err("cuMemcpyHtoD_v2 failed");
	}
	Ok(devptr)
}

/// RAII guard that frees device buffers on drop. Used to keep cleanup correct
/// across early returns (kernel launch errors, stream-query errors).
struct DeviceParamScratch {
	frame: CUdeviceptr,
	user: CUdeviceptr,
}

impl Drop for DeviceParamScratch {
	fn drop(&mut self) {
		if self.frame != 0 {
			unsafe { cuMemFree_v2(self.frame) };
		}
		if self.user != 0 {
			unsafe { cuMemFree_v2(self.user) };
		}
	}
}

pub fn run<UP>(config: &Configuration, user_params: UP, shader_src: &[u8], entry: &'static str) -> Result<(), &'static str> {
	use crate::gpu;

	if config.context_handle.is_none() || config.command_queue_handle.is_null() {
		log::error!("[CUDA] invalid handles");
		return Err("Invalid CUDA handles");
	}
	if config.dest_data.is_null() {
		log::error!("[CUDA] dest_data can't be null");
		return Err("null buffers");
	}

	let ctx = config.context_handle.unwrap();

	// Slang's CUDA codegen for `ConstantBuffer<T>` produces a `.u64` kernel arg
	// that the kernel dereferences via `ld.global` to read field bytes.
	check(unsafe { cuda::cuCtxSetCurrent(ctx as cuda::CUcontext) }, "cuCtxSetCurrent")?;

	let func = unsafe { gpu::pipeline::load_kernel(ctx as _, shader_src, entry) }.map_err(|e| {
		log::error!("[CUDA] {e}");
		"kernel load failed"
	})?;

	let outgoing_data = config.outgoing_data.unwrap_or(null_mut());
	let incoming_data = config.incoming_data.unwrap_or(null_mut());

	let mut d_outgoing = outgoing_data as u64;
	let mut d_incoming = incoming_data as u64;
	let mut d_dest = config.dest_data as u64;

	let frame = FrameParams {
		out_desc: crate::types::make_outgoing_desc(config),
		in_desc: crate::types::make_in_desc(config),
		dst_desc: crate::types::make_dst_desc(config),
		width: config.width,
		height: config.height,
		time: config.time,
		progress: config.progress,
	};

	let frame_bytes = unsafe { std::slice::from_raw_parts((&frame as *const FrameParams) as *const u8, std::mem::size_of::<FrameParams>()) };
	let user_bytes = unsafe { std::slice::from_raw_parts((&user_params as *const UP) as *const u8, std::mem::size_of::<UP>()) };

	let scratch = DeviceParamScratch {
		frame: unsafe { upload_to_device(frame_bytes)? },
		user: unsafe { upload_to_device(user_bytes)? },
	};

	let mut d_frame = scratch.frame;
	let mut d_user = scratch.user;

	let mut params: [*mut c_void; 5] = [
		&mut d_outgoing as *mut _ as *mut c_void,
		&mut d_incoming as *mut _ as *mut c_void,
		&mut d_dest as *mut _ as *mut c_void,
		&mut d_frame as *mut _ as *mut c_void,
		&mut d_user as *mut _ as *mut c_void,
	];

	let block_x: u32 = 16;
	let block_y: u32 = 16;
	let grid_x: u32 = config.width.div_ceil(block_x);
	let grid_y: u32 = config.height.div_ceil(block_y);

	let stream = config.command_queue_handle as cuda::CUstream;

	unsafe {
		dispatch(ctx, config.command_queue_handle, func, grid_x, grid_y, block_x, block_y, &mut params)?;
	}

	check(unsafe { cuda::cuStreamSynchronize(stream) }, "cuStreamSynchronize")?;

	drop(scratch);
	Ok(())
}
