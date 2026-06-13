use std::{collections::HashMap, sync::OnceLock};

use super::*;
use cudarc::driver::sys as cu;
use parking_lot::Mutex;

pub struct KernelEntry {
	pub module: cu::CUmodule,
	pub func: cu::CUfunction,
}

unsafe impl Send for KernelEntry {}
unsafe impl Sync for KernelEntry {}

static CACHE: OnceLock<Mutex<HashMap<(usize, &'static str), KernelEntry>>> = OnceLock::new();

#[inline]
fn cache() -> &'static Mutex<HashMap<(usize, &'static str), KernelEntry>> {
	CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

unsafe fn load_module_and_func(ptx_src: &[u8], fname: &str) -> Result<(cu::CUmodule, cu::CUfunction), String> {
	let mut module: cu::CUmodule = core::ptr::null_mut();

	// slangc emits a trailing NUL into the .ptx; CString::new rejects any embedded NUL,
	// so strip trailing zeros before re-wrapping (CString::new appends its own terminator).
	let ptx_trimmed: &[u8] = match ptx_src.iter().rposition(|&b| b != 0) {
		Some(end) => &ptx_src[..=end],
		None => &[],
	};

	let ptx_cstr = match std::ffi::CString::new(ptx_trimmed.to_vec()) {
		Ok(s) => s,
		Err(e) => {
			return Err(format!("NulError in kernel code. len: {}, nul_pos: {}", ptx_trimmed.len(), e.nul_position()));
		}
	};

	const JIT_ERROR_LOG_SIZE: usize = 8192;
	let mut jit_error_log: Vec<u8> = vec![0u8; JIT_ERROR_LOG_SIZE];
	let mut jit_error_log_size: usize = JIT_ERROR_LOG_SIZE;

	let mut jit_options: [cu::CUjit_option_enum; 2] = [
		cu::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER,
		cu::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
	];
	let mut jit_option_values: [*mut c_void; 2] = [
		jit_error_log.as_mut_ptr() as *mut c_void,
		&mut jit_error_log_size as *mut usize as *mut c_void,
	];

	let load_result = unsafe {
		cu::cuModuleLoadDataEx(
			&mut module,
			ptx_cstr.as_ptr() as *const c_void,
			2,
			jit_options.as_mut_ptr() as *mut cu::CUjit_option_enum,
			jit_option_values.as_mut_ptr() as *mut *mut c_void,
		)
	};

	if load_result != cu::CUresult::CUDA_SUCCESS {
		let error_log_str = jit_error_log[..jit_error_log_size.min(JIT_ERROR_LOG_SIZE)]
			.iter()
			.take_while(|&&b| b != 0)
			.map(|&b| b as char)
			.collect::<String>();
		log::error!("[CUDA] cuModuleLoadDataEx JIT error for '{fname}':\n{error_log_str}");
		super::check(load_result, "cuModuleLoadDataEx")?;
	}

	let mut func: cu::CUfunction = core::ptr::null_mut();
	let cname = std::ffi::CString::new(fname).unwrap();
	super::check(unsafe { cu::cuModuleGetFunction(&mut func, module, cname.as_ptr()) }, "cuModuleGetFunction")?;

	Ok((module, func))
}

/// Compile + cache a CUDA kernel function from PTX bytes.
///
/// `fname` must be `&'static str`: the kernel cache stores the name as part of
/// its key, so the reference has to outlive every dispatch. `declare_kernel!`
/// satisfies this via `stringify!`, which always yields a static literal.
/// Callers that need a runtime-built name should `Box::leak` it.
///
/// # Safety
/// `ctx` must be a live CUDA context. `ptx_bytes` must be valid PTX (slangc
/// output is fine; trailing NULs are stripped before submission).
pub unsafe fn load_kernel(
	ctx: cu::CUcontext,
	ptx_bytes: &[u8],
	fname: &'static str,
) -> Result<cu::CUfunction, String> {
	if ctx.is_null() {
		log::error!("[CUDA] null context");
		return Err("null context".to_string());
	}

	let key = (ctx as usize, fname);
	if let Some(k) = cache().lock().get(&key) {
		return Ok(k.func);
	}

	super::check(unsafe { cu::cuCtxSetCurrent(ctx) }, "cuCtxSetCurrent")?;

	let (module, func) = unsafe { load_module_and_func(ptx_bytes, fname) }.map_err(|e| {
		log::error!("[CUDA] module load: {e}");
		"module load failed".to_string()
	})?;

	cache().lock().insert(key, KernelEntry { module, func });

	log::info!("[CUDA] Loaded kernel '{fname}'");
	Ok(func)
}

pub unsafe fn cleanup() {
	if let Some(map) = CACHE.get() {
		let mut guard = map.lock();
		for ((_ctx, _name), k) in guard.drain() {
			if !k.module.is_null() {
				let _ = unsafe { cu::cuModuleUnload(k.module) };
			}
		}
		log::debug!("[CUDA] Module cache cleared");
	}
}
