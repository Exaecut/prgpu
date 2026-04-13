use std::{collections::HashMap, sync::OnceLock};

use super::*;
use cudarc::driver::sys as cu;
use parking_lot::Mutex;

pub struct KernelPair {
	pub module_f32: cu::CUmodule,
	pub func_f32: cu::CUfunction,
	pub module_f16: cu::CUmodule,
	pub func_f16: cu::CUfunction,
}

unsafe impl Send for KernelPair {}
unsafe impl Sync for KernelPair {}

static CACHE: OnceLock<Mutex<HashMap<(usize, &'static str), KernelPair>>> = OnceLock::new();

#[inline]
fn cache() -> &'static Mutex<HashMap<(usize, &'static str), KernelPair>> {
	CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[cfg(shader_hotreload)]
static SHADER_DIRS: OnceLock<Mutex<Option<(std::path::PathBuf, Vec<std::path::PathBuf>)>>> = OnceLock::new();

#[cfg(shader_hotreload)]
fn shader_dirs() -> &'static Mutex<Option<(std::path::PathBuf, Vec<std::path::PathBuf>)>> {
	SHADER_DIRS.get_or_init(|| Mutex::new(None))
}

pub fn set_shader_dirs(_shader_dir: std::path::PathBuf, _include_dirs: Vec<std::path::PathBuf>) {
	#[cfg(shader_hotreload)]
	{
		let (shader_dir, include_dirs) = (_shader_dir, _include_dirs);
		log::info!("[CUDA/HotReload] Shader source dir: {}", shader_dir.display());
		for d in &include_dirs {
			log::info!("[CUDA/HotReload] Include dir: {}", d.display());
		}
		*shader_dirs().lock() = Some((shader_dir, include_dirs));
	}
}

#[cfg(shader_hotreload)]
fn compile_vekl_to_ptx(name: &str, shader_dir: &std::path::Path, include_dirs: &[std::path::PathBuf], half_precision: bool) -> Result<String, String> {
	use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
	use crate::gpu::shaders::prepare_cuda_source;
	use std::time::Instant;

	let vekl_path = shader_dir.join(format!("{name}.vekl"));
	let src = std::fs::read_to_string(&vekl_path).map_err(|e| format!("Failed to read {}: {e}", vekl_path.display()))?;
	let prepared_src = prepare_cuda_source(&src, name);

	let tag = if half_precision { format!("{name} (f16)") } else { format!("{name} (f32)") };
	log::info!("[CUDA/HotReload] Compiling: {tag} ({} bytes) from {}", prepared_src.len(), vekl_path.display());

	let cuda_path = std::env::var("CUDA_HOME").or_else(|_| std::env::var("CUDA_PATH")).unwrap_or("/usr/local/cuda".into());
	let cuda_include = std::path::PathBuf::from(&cuda_path).join("include");

	let mut all_includes: Vec<String> = include_dirs
		.iter()
		.filter_map(|d| d.canonicalize().ok())
		.map(|d| d.to_string_lossy().replace("\\\\?\\", ""))
		.collect();
	if let Ok(c) = cuda_include.canonicalize() {
		all_includes.push(c.to_string_lossy().replace("\\\\?\\", ""));
	}

	let mut options = vec![
		"--std=c++14".into(),
		"--extra-device-vectorization".into(),
		"--device-as-default-execution-space".into(),
	];
	if cfg!(debug_assertions) {
		options.push("-DDEBUG=1".into());
	}
	if half_precision {
		options.push("-DUSE_HALF_PRECISION=1".into());
	}

	let opts = CompileOptions {
		ftz: Some(true),
		prec_sqrt: Some(false),
		prec_div: Some(false),
		fmad: Some(true),
		use_fast_math: None,
		include_paths: all_includes,
		arch: Some("compute_86"),
		options,
		..Default::default()
	};

	let start = Instant::now();
	let ptx = compile_ptx_with_opts(&prepared_src, opts).map_err(|e| {
		let detail = match &e {
			cudarc::nvrtc::CompileError::CompileError { log, .. } => log.to_string_lossy().into_owned(),
			other => format!("{other:#?}"),
		};
		let ms = start.elapsed().as_secs_f64() * 1000.0;
		log::error!("[CUDA/HotReload] NVRTC error for '{tag}' ({ms:.1}ms):\n{detail}");
		format!("Compilation failed for '{tag}'")
	})?;

	let elapsed = start.elapsed();
	let ptx_bytes = ptx.as_bytes().unwrap();
	let ptx_bytes = if ptx_bytes.last() == Some(&0) {
		&ptx_bytes[..ptx_bytes.len() - 1]
	} else {
		ptx_bytes
	};
	let ptx_str = std::str::from_utf8(ptx_bytes).map_err(|e| format!("PTX decode error for '{tag}': {e}"))?.to_string();

	log::info!(
		"[CUDA/HotReload] Compiled '{tag}' in {:.1}ms ({} bytes PTX)",
		elapsed.as_secs_f64() * 1000.0,
		ptx_str.len()
	);

	Ok(ptx_str)
}

unsafe fn load_module_and_func(ptx_src: String, fname: &str) -> Result<(cu::CUmodule, cu::CUfunction), String> {
	let mut module: cu::CUmodule = core::ptr::null_mut();
	let ptx_len = ptx_src.len();

	let ptx_cstr = match std::ffi::CString::new(ptx_src) {
		Ok(s) => s,
		Err(e) => {
			return Err(format!("NulError in kernel code. len: {}, nul_pos: {}", ptx_len, e.nul_position()));
		}
	};

	super::check(unsafe { cu::cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const c_void) }, "cuModuleLoadData")?;

	let mut func: cu::CUfunction = core::ptr::null_mut();
	let cname = std::ffi::CString::new(fname).unwrap();
	super::check(unsafe { cu::cuModuleGetFunction(&mut func, module, cname.as_ptr()) }, "cuModuleGetFunction")?;

	Ok((module, func))
}

pub unsafe fn load_kernel(
	ctx: cu::CUcontext,
	shader_src_f32: &'static str,
	shader_src_f16: &'static str,
	fname: &'static str,
) -> Result<(cu::CUfunction, cu::CUfunction), String> {
	if ctx.is_null() {
		log::error!("[CUDA] null context");
		return Err("null context".to_string());
	}

	let key = (ctx as usize, fname);
	if let Some(k) = cache().lock().get(&key) {
		return Ok((k.func_f32, k.func_f16));
	}

	super::check(unsafe { cu::cuCtxSetCurrent(ctx) }, "cuCtxSetCurrent")?;

	// Resolve the PTX source for each precision variant independently.
	let ptx_f32: std::borrow::Cow<'static, str>;
	let ptx_f16: std::borrow::Cow<'static, str>;

	#[cfg(shader_hotreload)]
	{
		let guard = shader_dirs().lock();
		if let Some((shader_dir, include_dirs)) = guard.as_ref() {
			ptx_f32 = match compile_vekl_to_ptx(fname, shader_dir, include_dirs, false) {
				Ok(ptx) => {
					log::info!("[CUDA/HotReload] Runtime-compiled f32 PTX for '{fname}'");
					std::borrow::Cow::Owned(ptx)
				}
				Err(e) => {
					log::error!("[CUDA/HotReload] {e}");
					log::warn!("[CUDA/HotReload] Falling back to embedded f32 PTX for '{fname}'");
					std::borrow::Cow::Borrowed(shader_src_f32)
				}
			};
			ptx_f16 = match compile_vekl_to_ptx(fname, shader_dir, include_dirs, true) {
				Ok(ptx) => {
					log::info!("[CUDA/HotReload] Runtime-compiled f16 PTX for '{fname}'");
					std::borrow::Cow::Owned(ptx)
				}
				Err(e) => {
					log::error!("[CUDA/HotReload] {e}");
					log::warn!("[CUDA/HotReload] Falling back to embedded f16 PTX for '{fname}'");
					std::borrow::Cow::Borrowed(shader_src_f16)
				}
			};
		} else {
			log::warn!("[CUDA/HotReload] No shader dirs registered — using embedded PTX for '{fname}'");
			ptx_f32 = std::borrow::Cow::Borrowed(shader_src_f32);
			ptx_f16 = std::borrow::Cow::Borrowed(shader_src_f16);
		}
	}

	#[cfg(not(shader_hotreload))]
	{
		ptx_f32 = std::borrow::Cow::Borrowed(shader_src_f32);
		ptx_f16 = std::borrow::Cow::Borrowed(shader_src_f16);
	}

	let (module_f32, func_f32) = unsafe { load_module_and_func(ptx_f32.into_owned(), fname) }.map_err(|e| {
		log::error!("[CUDA] f32 module: {e}");
		"module load failed".to_string()
	})?;
	let (module_f16, func_f16) = unsafe { load_module_and_func(ptx_f16.into_owned(), fname) }.map_err(|e| {
		log::error!("[CUDA] f16 module: {e}");
		"module load failed".to_string()
	})?;

	cache().lock().insert(
		key,
		KernelPair {
			module_f32,
			func_f32,
			module_f16,
			func_f16,
		},
	);

	log::info!("[CUDA] Loaded kernel pair '{fname}' (f32 + f16)");
	Ok((func_f32, func_f16))
}

/// # Safety
/// Must be called with no active CUDA contexts locked; clears global cache.
pub unsafe fn cleanup() {
	if let Some(map) = CACHE.get() {
		let mut guard = map.lock();
		for ((_ctx, _name), k) in guard.drain() {
			if !k.module_f32.is_null() {
				let _ = unsafe { cu::cuModuleUnload(k.module_f32) };
			}
			if !k.module_f16.is_null() {
				let _ = unsafe { cu::cuModuleUnload(k.module_f16) };
			}
		}
		log::info!("[CUDA] Module cache cleared");
	}
}

pub fn hot_reload() {
	unsafe { cleanup() };
	#[cfg(shader_hotreload)]
	log::info!("[CUDA/HotReload] Cache cleared - next dispatch will recompile from disk.");
	#[cfg(not(shader_hotreload))]
	log::info!("[CUDA] Cache cleared.");
}
