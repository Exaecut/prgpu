use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};
use cudarc::driver::sys as cu;
use super::*;

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

pub unsafe fn get_pso_pair(
	ctx: cu::CUcontext,
	shader_src: &'static str,
	fname: &'static str,
	device_handle: *mut c_void,
) -> Result<(cu::CUfunction, cu::CUfunction), &'static str> {
	if ctx.is_null() {
		log::error!("[CUDA] null context");
		return Err("null context");
	}
	let key = (ctx as usize, fname);
	if let Some(k) = cache().lock().get(&key) {
		return Ok((k.func_f32, k.func_f16));
	}

	super::check(unsafe { cu::cuCtxSetCurrent(ctx) }, "cuCtxSetCurrent")?;

	let raw_src: Cow<'static, str> = {
		#[cfg(all(debug_assertions, shader_hotreload))]
		{
			use crate::gpu::shaders::expand_includes_runtime;
			let manifest_dir = env!("CARGO_MANIFEST_DIR");
			let plugin_root = std::path::PathBuf::from(manifest_dir).join("shaders");
			let ws_utils = std::path::PathBuf::from(manifest_dir).join("../shaders/utils");

			let mut path = plugin_root.clone();
			path.push(format!("{fname}.cu"));

			match std::fs::read_to_string(&path) {
				Ok(s) => match expand_includes_runtime(&s, &plugin_root, &[ws_utils]) {
					Ok(expanded) => {
						log::info!("[CUDA] Hot-reloading shader (flattened) from {}", path.display());
						Cow::Owned(expanded)
					}
					Err(e) => {
						log::warn!("[CUDA] Hot reload include expansion failed: {e}. Using embedded source.");
						Cow::Borrowed(shader_src)
					}
				},
				Err(e) => {
					log::warn!("[CUDA] Hot file not found/failed to read ({}). Using embedded source.", e);
					Cow::Borrowed(shader_src)
				}
			}
		}
		#[cfg(not(all(debug_assertions, shader_hotreload)))]
		{
			Cow::Borrowed(shader_src)
		}
	};

	// Deux variantes: 32f et 16f (macro USE_HALF_PRECISION)
	let src_f32 = raw_src.as_ref().to_string();
	let src_f16 = format!("#define USE_HALF_PRECISION 1\n{}", raw_src.as_ref());

	let ptx_f32 = compile_ptx(&src_f32, fname, device_handle)?;
	let ptx_f16 = compile_ptx(&src_f16, fname, device_handle)?;

	let (module_f32, func_f32) = unsafe { load_module_and_func(ptx_f32.to_src(), fname) }?;
	let (module_f16, func_f16) = unsafe { load_module_and_func(ptx_f16.to_src(), fname) }?;

	cache().lock().insert(
		key,
		KernelPair {
			module_f32,
			func_f32,
			module_f16,
			func_f16,
		},
	);

	log::info!("[CUDA] Built kernels '{fname}' (f32 + f16)");
	Ok((func_f32, func_f16))
}

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
	log::info!("[CUDA] Hot reload requested - cache cleared; next frame will recompile.");
}

fn compile_ptx(src: &str, fname: &str, dev_handle: *mut c_void) -> Result<Ptx, &'static str> {
	if dev_handle.is_null() {
		return Err("null device handle");
	}
	let dev = dev_handle as cu::CUdevice;

	let (major, minor) = unsafe { super::compute_capability(dev)? };
	let arch = format!("compute_{}{}", major, minor);

	let mut opts = CompileOptions::default();
	// Ajoute le flag d’arch
	opts.options.push(format!("--gpu-architecture={arch}"));
	// Pour debug : donner un nom symbolique
	opts.name = Some(format!("{fname}.cu"));

	match compile_ptx_with_opts(src, opts) {
		Ok(ptx) => Ok(ptx),
		Err(e) => {
			log::warn!("[CUDA] NVRTC compile with arch={} failed: {e:?}. Retrying without arch flag...", arch);

			let mut fallback_opts = CompileOptions::default();
			fallback_opts.name = Some(format!("{fname}.cu"));
			compile_ptx_with_opts(src, fallback_opts).map_err(|e2| {
				log::error!("[CUDA] NVRTC compile fallback also failed: {e2:?}");
				"NVRTC compile error"
			})
		}
	}
}

unsafe fn load_module_and_func(ptx_src: String, fname: &str) -> Result<(cu::CUmodule, cu::CUfunction), &'static str> {
	let mut module: cu::CUmodule = core::ptr::null_mut();
	let ptx_cstr = std::ffi::CString::new(ptx_src).unwrap();
	super::check(unsafe { cu::cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const c_void) }, "cuModuleLoadData")?;
	let mut func: cu::CUfunction = core::ptr::null_mut();
	let cname = std::ffi::CString::new(fname).unwrap();
	super::check(unsafe { cu::cuModuleGetFunction(&mut func, module, cname.as_ptr()) }, "cuModuleGetFunction")?;
	Ok((module, func))
}
