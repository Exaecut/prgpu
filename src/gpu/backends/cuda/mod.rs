use after_effects::log;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::OnceLock;
use std::time::Duration;

use cudarc::driver::sys as cuda;
use parking_lot::Mutex;

pub mod buffer;
pub mod fence;
pub mod pipeline;

use crate::logging::{GpuLogCursor, VeklLogBuffer};
use crate::{Configuration, FrameParams};

/// Wrapper to make CUevent Send+Sync-safe.
/// CUDA events are safe to share across threads as long as they are used
/// with proper synchronization (which cuEventRecord/cuEventElapsedTime provide).
#[derive(Copy, Clone)]
struct SafeEvent(cuda::CUevent);
unsafe impl Send for SafeEvent {}
unsafe impl Sync for SafeEvent {}

/// Cached CUDA event pairs (start, stop) per context for GPU timing.
/// Events are created once and reused across dispatches to avoid allocation overhead.
static EVENT_CACHE: OnceLock<Mutex<HashMap<usize, (SafeEvent, SafeEvent)>>> = OnceLock::new();

fn event_cache() -> &'static Mutex<HashMap<usize, (SafeEvent, SafeEvent)>> {
	EVENT_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get or create a start/stop event pair for the given context.
fn get_or_create_events(ctx_key: usize) -> (SafeEvent, SafeEvent) {
	let mut guard = event_cache().lock();
	*guard.entry(ctx_key).or_insert_with(|| {
		let mut start: cuda::CUevent = std::ptr::null_mut();
		let mut stop: cuda::CUevent = std::ptr::null_mut();
		unsafe {
			cuda::cuEventCreate(&mut start, cuda::CUevent_flags::CU_EVENT_DEFAULT as u32);
			cuda::cuEventCreate(&mut stop, cuda::CUevent_flags::CU_EVENT_DEFAULT as u32);
		}
		(SafeEvent(start), SafeEvent(stop))
	})
}

/// Destroy all cached CUDA events. Called during pipeline cleanup.
pub fn destroy_event_cache() {
	if let Some(cache) = EVENT_CACHE.get() {
		let mut guard = cache.lock();
		for (_, (SafeEvent(start), SafeEvent(stop))) in guard.drain() {
			unsafe {
				cuda::cuEventDestroy_v2(start);
				cuda::cuEventDestroy_v2(stop);
			}
		}
	}
}

struct CudaLogBufferGuard {
	ctx: *mut c_void,
	ptr: cuda::CUdeviceptr,
	host: *mut VeklLogBuffer,
}

impl CudaLogBufferGuard {
	unsafe fn allocate(ctx: *mut c_void) -> Result<Self, &'static str> {
		check(unsafe { cuda::cuCtxSetCurrent(ctx as cuda::CUcontext) }, "cuCtxSetCurrent")?;

		// Use cuMemAllocHost for pinned host memory — guaranteed CPU-accessible on Windows WDDM.
		// cuMemAllocManaged (unified memory) is NOT reliably CPU-accessible under WDDM.
		let mut host_ptr: *mut c_void = std::ptr::null_mut();
		check(
			unsafe { cuda::cuMemAllocHost_v2(&mut host_ptr, std::mem::size_of::<VeklLogBuffer>()) },
			"cuMemAllocHost_v2",
		)?;

		if host_ptr.is_null() {
			log::error!("[CUDA] cuMemAllocHost returned null");
			return Err("cuMemAllocHost returned null");
		}

		let host = host_ptr as *mut VeklLogBuffer;
		unsafe { (*host).initialize() };

		// Get the device-accessible pointer for the pinned host memory
		let mut dev_ptr: cuda::CUdeviceptr = 0;
		check(
			unsafe { cuda::cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0u32) },
			"cuMemHostGetDevicePointer_v2",
		)?;

		Ok(Self { ctx, ptr: dev_ptr, host })
	}
}

impl Drop for CudaLogBufferGuard {
	fn drop(&mut self) {
		if self.host.is_null() {
			return;
		}

		let _ = unsafe { cuda::cuCtxSetCurrent(self.ctx as cuda::CUcontext) };
		let _ = unsafe { cuda::cuMemFreeHost(self.host as *mut c_void) };
	}
}

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

/// Launches a CUDA kernel on the given stream. Does NOT synchronize.
///
/// # Safety
/// - `ctx`, `stream`, `func` must be valid CUDA handles.
/// - `params` must point to valid device memory matching the kernel signature.
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

pub fn run<UP>(config: &Configuration, user_params: UP, shader_src: &'static str, shader_src_f16: &'static str, entry: &'static str) -> Result<(), &'static str> {
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

	let (func_f32, func_f16) = unsafe { gpu::pipeline::load_kernel(ctx as _, shader_src, shader_src_f16, entry) }.map_err(|e| {
		log::error!("[CUDA] {e}");
		"kernel load failed"
	})?;
	let func = if config.is16f { func_f16 } else { func_f32 };

	let outgoing_data = config.outgoing_data.unwrap_or(null_mut());
	let incoming_data = config.incoming_data.unwrap_or(null_mut());

	let mut d_outgoing = outgoing_data as u64;
	let mut d_incoming = incoming_data as u64;
	let mut d_dest = config.dest_data as u64;
	let log_buffer = unsafe { CudaLogBufferGuard::allocate(ctx)? };
	let mut d_log = log_buffer.ptr as u64;
	let mut log_cursor = GpuLogCursor::default();

	let mut p = FrameParams {
		out_pitch: config.outgoing_pitch_px as u32,
		in_pitch: config.incoming_pitch_px as u32,
		dest_pitch: config.dest_pitch_px as u32,
		width: config.width,
		height: config.height,
		progress: config.progress,
		bpp: config.bytes_per_pixel,
		pixel_layout: config.pixel_layout,
	};
	let mut u = user_params;

	let mut params: [*mut c_void; 6] = [
		&mut d_outgoing as *mut _ as *mut c_void,
		&mut d_incoming as *mut _ as *mut c_void,
		&mut d_dest as *mut _ as *mut c_void,
		&mut p as *mut _ as *mut c_void,
		&mut u as *mut _ as *mut c_void,
		&mut d_log as *mut _ as *mut c_void,
	];

	let block_x: u32 = 16;
	let block_y: u32 = 16;
	let grid_x: u32 = config.width.div_ceil(block_x);
	let grid_y: u32 = config.height.div_ceil(block_y);

	let (SafeEvent(start_event), SafeEvent(stop_event)) = get_or_create_events(ctx as usize);
	let stream = config.command_queue_handle as cuda::CUstream;

	unsafe {
		#[cfg(feature = "timing")]
		cuda::cuEventRecord(start_event, stream);
		dispatch(ctx, config.command_queue_handle, func, grid_x, grid_y, block_x, block_y, &mut params)?;
		#[cfg(feature = "timing")]
		cuda::cuEventRecord(stop_event, stream);
	}

	loop {
		unsafe { (&*log_buffer.host).drain_into_host_log(&mut log_cursor, entry, "cuda") };

		let query = unsafe { cuda::cuStreamQuery(stream) };
		if query == cuda::CUresult::CUDA_SUCCESS {
			break;
		}
		if query != cuda::CUresult::CUDA_ERROR_NOT_READY {
			log::error!("[CUDA] cuStreamQuery failed: {query:?}");
			return Err("cuStreamQuery failed");
		}

		std::thread::sleep(Duration::from_millis(1));
	}

	unsafe { (&*log_buffer.host).drain_into_host_log(&mut log_cursor, entry, "cuda") };

	// Read GPU timing from CUDA events after stream completion.
	#[cfg(feature = "timing")]
	{
		let mut gpu_ms: f32 = 0.0;
		unsafe {
			cuda::cuEventElapsedTime_v2(&mut gpu_ms, start_event, stop_event);
		}
		crate::timing::record(entry, crate::types::Backend::Cuda, (gpu_ms * 1_000_000.0) as u64);
	}

	Ok(())
}
