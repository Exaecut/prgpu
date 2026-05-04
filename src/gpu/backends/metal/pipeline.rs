use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use after_effects::log;
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use parking_lot::Mutex;

use super::ns_error;

pub struct Pipeline {
    pub pso: *mut Object,
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

#[derive(Clone, Copy, Eq)]
struct Key {
    device: usize,
    src_hash: u64,
    name_hash: u64,
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device && self.src_hash == other.src_hash && self.name_hash == other.name_hash
    }
}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.device.hash(state);
        self.src_hash.hash(state);
        self.name_hash.hash(state);
    }
}

fn hash_bytes(data: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    data.hash(&mut h);
    h.finish()
}

static CACHE: OnceLock<Mutex<HashMap<Key, Pipeline>>> = OnceLock::new();

pub unsafe fn load_kernel(device: *mut Object, metallib_bytes: &[u8], fname: &str) -> Result<*mut Object, &'static str> {
    let key = Key {
        device: device as usize,
        src_hash: hash_bytes(metallib_bytes),
        name_hash: {
            use std::collections::hash_map::DefaultHasher;
            let mut h = DefaultHasher::new();
            fname.hash(&mut h);
            h.finish()
        },
    };

    let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let guard = map.lock();
        if let Some(p) = guard.get(&key) {
            return Ok(p.pso);
        }
    }

    let data: *mut Object = msg_send![class!(NSData), dataWithBytesNoCopy: metallib_bytes.as_ptr() as *const std::ffi::c_void
        length: metallib_bytes.len()
        freeWhenDone: false];
    if data.is_null() {
        log::error!("[Metal] failed to create NSData from metallib bytes");
        return Err("NSData creation failed");
    }

    let mut error: *mut Object = std::ptr::null_mut();
    let library: *mut Object = msg_send![device, newLibraryWithData: data error: &mut error];
    if library.is_null() {
        if let Some(msg) = unsafe { ns_error(error) } {
            log::error!("[Metal] newLibraryWithData failed: {msg}");
        }
        return Err("library load from metallib failed");
    }

    let fname_ns = unsafe { super::nsstring_utf8(fname) };
    let func: *mut Object = msg_send![library, newFunctionWithName: fname_ns];
    if func.is_null() {
        let _: () = msg_send![library, release];
        log::error!("[Metal] function '{fname}' not found in library");
        return Err("function not found");
    }

    let mut err: *mut Object = std::ptr::null_mut();
    let pso: *mut Object = msg_send![device, newComputePipelineStateWithFunction: func error: &mut err];
    let _: () = msg_send![func, release];
    let _: () = msg_send![library, release];

    if pso.is_null() {
        if let Some(msg) = unsafe { ns_error(err) } {
            log::error!("[Metal] pipeline creation failed: {msg}");
        }
        return Err("pipeline failed");
    }

    {
        let mut guard = map.lock();
        guard.insert(key, Pipeline { pso });
    }

    log::info!("[Metal] Built pipeline for device={device:p} entry='{fname}'");
    Ok(pso)
}

pub unsafe fn cleanup() {
    if let Some(map) = CACHE.get() {
        let mut guard = map.lock();
        for (_k, p) in guard.drain() {
            if !p.pso.is_null() {
                let _: () = msg_send![p.pso, release];
            }
        }
        log::info!("[Metal] Pipeline cache cleared");
    }
}

pub fn hot_reload() {
    unsafe { cleanup() };
    log::info!("[Metal] Cache cleared.");
}
