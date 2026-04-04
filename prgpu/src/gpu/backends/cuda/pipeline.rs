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

/// # Safety
/// - `ptx_src` must be valid PTX from NVRTC.
/// - Caller owns returned `module`; unload with `cuModuleUnload`.
unsafe fn load_module_and_func(
    ptx_src: String,
    fname: &str,
) -> Result<(cu::CUmodule, cu::CUfunction), String> {
    let mut module: cu::CUmodule = core::ptr::null_mut();
    let ptx_len = ptx_src.len();

    let ptx_cstr = match std::ffi::CString::new(ptx_src) {
        Ok(s) => s,
        Err(e) => {
            return Err(format!(
                "NulError in kernel code. len: {}, nul_pos: {}",
                ptx_len,
                e.nul_position()
            ));
        }
    };

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

/// Retrieves or load a pair of CUDA kernels (f32 and f16 variants) for the given shader source and function name.
///
/// # Safety
pub unsafe fn get_or_load_kernel(
    ctx: cu::CUcontext,
    shader_src: &'static str,
    fname: &'static str,
) -> Result<(cu::CUfunction, cu::CUfunction), String> {
    if ctx.is_null() {
        log::error!("[CUDA] null context");
        return Err("null context".to_string());
    }

    // Check if already cached
    let key = (ctx as usize, fname);
    if let Some(k) = cache().lock().get(&key) {
        return Ok((k.func_f32, k.func_f16));
    }

    // Switch CUDA Context
    super::check(unsafe { cu::cuCtxSetCurrent(ctx) }, "cuCtxSetCurrent")?;

    // Load module full and half precision
    let (module_f32, func_f32) = unsafe { load_module_and_func(shader_src.to_string(), fname) }
        .map_err(|e| {
            log::error!("[CUDA] {e}");
            "module load failed"
        })?;
    let (module_f16, func_f16) = unsafe { load_module_and_func(shader_src.to_string(), fname) }
        .map_err(|e| {
            log::error!("[CUDA] {e}");
            "module load failed"
        })?;

    // Cache module
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
