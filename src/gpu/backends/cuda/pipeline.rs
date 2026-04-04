use super::*;
use cudarc::driver::sys as cu;

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

/// Retrieves or compiles a pair of CUDA kernels (f32 and f16 variants) for the given shader source and function name.
///
/// # Safety
/// - `ctx` must be a valid, non-null `CUcontext` owned by the calling thread.
/// - `device_handle` must be a valid pointer to a `CUdevice`.
/// - `shader_src` and `fname` must be `'static` strings (NVRTC requires persistent source).
/// - Caller must ensure `ctx` remains valid for the kernel's lifetime; use `cleanup()` before destroying `ctx`.
/// - Modules are cached per `(ctx, fname)`; safe for repeated calls but leaks if not cleaned up.
/// - In debug+hotreload mode, reads from filesystem; ensure shader files are accessible.
/// - No automatic error recovery; check `Result` and log errors.
/// - Violating these may cause segfaults, GPU hangs, or memory leaks.
pub unsafe fn get_pso_pair(
    ctx: cu::CUcontext,
    shader_src: &'static str,
    fname: &'static str,
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
                        log::info!(
                            "[CUDA] Hot-reloading shader (flattened) from {}",
                            path.display()
                        );
                        Cow::Owned(expanded)
                    }
                    Err(e) => {
                        log::warn!(
                            "[CUDA] Hot reload include expansion failed: {e}. Using embedded source."
                        );
                        Cow::Borrowed(shader_src)
                    }
                },
                Err(e) => {
                    log::warn!(
                        "[CUDA] Hot file not found/failed to read ({}). Using embedded source.",
                        e
                    );
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

    let (module_f32, func_f32) = unsafe { load_module_and_func(src_f32, fname) }?;
    let (module_f16, func_f16) = unsafe { load_module_and_func(src_f16, fname) }?;

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

/// # Safety
/// - Must be called with no active CUDA contexts locked; clears global cache.
/// - Modules must not be in use by kernels during cleanup.
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

/// # Safety
/// - `ptx_src` must be valid PTX from NVRTC.
/// - Caller owns returned `module`; unload with `cuModuleUnload`.
unsafe fn load_module_and_func(
    ptx_src: String,
    fname: &str,
) -> Result<(cu::CUmodule, cu::CUfunction), &'static str> {
    let mut module: cu::CUmodule = core::ptr::null_mut();
    let ptx_cstr = std::ffi::CString::new(ptx_src).unwrap();
    super::check(
        unsafe { cu::cuModuleLoadData(&mut module, ptx_cstr.as_ptr() as *const c_void) },
        "cuModuleLoadData",
    )?;
    let mut func: cu::CUfunction = core::ptr::null_mut();
    let cname = std::ffi::CString::new(fname).unwrap();
    super::check(
        unsafe { cu::cuModuleGetFunction(&mut func, module, cname.as_ptr()) },
        "cuModuleGetFunction",
    )?;
    Ok((module, func))
}
