// src/gpu/backends/metal/pipeline.rs  (or wherever your `gpu::pipeline` lives)
use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use after_effects::log;
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use parking_lot::Mutex;

use super::{ns_error, nsstring_utf8};

pub struct Pipelines {
	pub library: *mut Object,  // keep one lib alive for symbols
	pub pso_full: *mut Object, // float4 pipeline
	pub pso_half: *mut Object, // half4 pipeline with USE_HALF_PRECISION=1
}

// Raw ObjC pointers are not Send/Sync by default - we fence access with a Mutex.
// Declare these as safe because we only move pointers across threads, not the objects.
unsafe impl Send for Pipelines {}
unsafe impl Sync for Pipelines {}

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

fn hash_str(s: &str) -> u64 {
	use std::collections::hash_map::DefaultHasher;
	let mut h = DefaultHasher::new();
	s.hash(&mut h);
	h.finish()
}

static CACHE: OnceLock<Mutex<HashMap<Key, Pipelines>>> = OnceLock::new();

/// Retrieves a pair of pipeline state objects (PSOs) for the given device and shader source.
///
/// # Arguments
/// - `device`: A pointer to the Metal device.
/// - `shader_src`: The source code of the shader.
/// - `fname`: The name of the function to retrieve.
///
/// # Returns
/// A `Result` containing a tuple of two raw pointers to the PSOs (`pso_full` and `pso_half`) on success,
/// or an error message on failure.
///
/// # Safety
/// - The `device` pointer must be a valid Metal device pointer.
/// - The `shader_src` and `fname` must point to valid static strings.
/// - The caller must ensure that the returned pointers are used correctly and released appropriately.
pub unsafe fn get_pso_pair(device: *mut Object, shader_src: &'static str, fname: &'static str) -> Result<(*mut Object, *mut Object), &'static str> {
	let key = Key {
		device: device as usize,
		src_hash: hash_str(shader_src),
		name_hash: hash_str(fname),
	};

	let map = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
	{
		let guard = map.lock();
		if let Some(p) = guard.get(&key) {
			return Ok((p.pso_full, p.pso_half));
		}
	}

	// Optional hot-reload: when enabled you may rebuild `shader_src` from file + includes.
	let raw_src: Cow<'static, str> = {
		#[cfg(all(debug_assertions, shader_hotreload))]
		{
			use crate::gpu::shaders::expand_includes_runtime;
			let manifest_dir = env!("CARGO_MANIFEST_DIR");
			let mut path = std::path::PathBuf::from(&manifest_dir);
			path.push("shaders");
			path.push(format!("{fname}.metal"));

			match std::fs::read_to_string(&path) {
				Ok(s) => {
					let plugin_root = std::path::PathBuf::from(&manifest_dir).join("shaders");
					let ws_utils = std::path::PathBuf::from(&manifest_dir).join("../shaders/utils");
					match expand_includes_runtime(&s, &plugin_root, &[ws_utils]) {
						Ok(expanded) => {
							log::info!("[Metal] Hot-reloading shader (flattened) from {}", path.display());
							Cow::Owned(expanded)
						}
						Err(e) => {
							log::warn!("[Metal] Hot reload include expansion failed: {e}. Using embedded source.");
							Cow::Borrowed(shader_src)
						}
					}
				}
				Err(e) => {
					log::warn!("[Metal] Hot file not found/failed to read ({}). Using embedded source.", e);
					Cow::Borrowed(shader_src)
				}
			}
		}
		#[cfg(not(all(debug_assertions, shader_hotreload)))]
		{
			Cow::Borrowed(shader_src)
		}
	};

	let src = unsafe { nsstring_utf8(&raw_src) };
	let mut error: *mut Object = std::ptr::null_mut();

	// Compile full precision
	let opts_f32: *mut Object = msg_send![class!(MTLCompileOptions), alloc];
	let opts_f32: *mut Object = msg_send![opts_f32, init];

	let lib_f32: *mut Object = msg_send![device, newLibraryWithSource: src options: opts_f32 error: &mut error];
	if lib_f32.is_null() {
		if let Some(msg) = unsafe { ns_error(error) } {
			log::error!("[Metal] newLibraryWithSource (f32) failed: {msg}");
		}
		return Err("library f32 compile failed");
	}

	// Compile half precision with macro
	let opts_f16: *mut Object = msg_send![class!(MTLCompileOptions), alloc];
	let opts_f16: *mut Object = msg_send![opts_f16, init];

	let macros: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
	let key_macro: *mut Object = unsafe { nsstring_utf8("USE_HALF_PRECISION") };
	let val_macro: *mut Object = msg_send![class!(NSNumber), numberWithInt: 1];
	let _: () = msg_send![macros, setObject: val_macro forKey: key_macro];
	let _: () = msg_send![opts_f16, setPreprocessorMacros: macros];

	let lib_f16: *mut Object = msg_send![device, newLibraryWithSource: src options: opts_f16 error: &mut error];
	if lib_f16.is_null() {
		if let Some(msg) = unsafe { ns_error(error) } {
			log::error!("[Metal] newLibraryWithSource (f16) failed: {msg}");
		}
		return Err("library f16 compile failed");
	}

	// Create function objects for requested entry point
	let fname_ns = unsafe { nsstring_utf8(fname) };
	let func_f32: *mut Object = msg_send![lib_f32, newFunctionWithName: fname_ns];
	let func_f16: *mut Object = msg_send![lib_f16, newFunctionWithName: fname_ns];
	if func_f32.is_null() || func_f16.is_null() {
		log::error!("[Metal] function '{fname}' not found in libraries");
		return Err("function not found");
	}

	// Build pipelines
	let mut err1: *mut Object = std::ptr::null_mut();
	let mut err2: *mut Object = std::ptr::null_mut();
	let pso_f32: *mut Object = msg_send![device, newComputePipelineStateWithFunction: func_f32 error: &mut err1];
	let pso_f16: *mut Object = msg_send![device, newComputePipelineStateWithFunction: func_f16 error: &mut err2];

	if pso_f32.is_null() || pso_f16.is_null() {
		log::error!("[Metal] pipeline creation failed: {err1:?} / {err2:?}");
		return Err("pipeline failed");
	}

	// Functions are retained by PSOs - release them
	let _: () = msg_send![func_f32, release];
	let _: () = msg_send![func_f16, release];

	// Insert into cache
	{
		let mut guard = map.lock();
		guard.insert(
			key,
			Pipelines {
				library: lib_f32, // keep one lib alive
				pso_full: pso_f32,
				pso_half: pso_f16,
			},
		);
	}

	log::info!("[Metal] Built pipelines for device={device:p} entry='{fname}'");
	Ok((pso_f32, pso_f16))
}

/// Cleans up the pipeline cache by releasing all retained Metal objects.
///
/// # Safety
/// - This function must only be called when no other threads are accessing the pipeline cache.
/// - The caller must ensure that the Metal objects being released are no longer in use elsewhere.
pub unsafe fn cleanup() {
	if let Some(map) = CACHE.get() {
		let mut guard = map.lock();
		for (_k, p) in guard.drain() {
			if !p.pso_full.is_null() {
				let _: () = msg_send![p.pso_full, release];
			}
			if !p.pso_half.is_null() {
				let _: () = msg_send![p.pso_half, release];
			}
			if !p.library.is_null() {
				let _: () = msg_send![p.library, release];
			}
		}
		log::info!("[Metal] Pipeline cache cleared");
	}
}

pub fn hot_reload() {
	unsafe { cleanup() };
	log::info!("[Metal] Hot reload requested - cache cleared; next request will recompile.");
}
