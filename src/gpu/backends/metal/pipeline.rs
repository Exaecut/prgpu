use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use after_effects::log;
use objc::{class, msg_send, runtime::Object, sel, sel_impl};
use parking_lot::Mutex;

use super::{ns_error, nsstring_utf8};

pub struct Pipelines {
	pub library_f32: *mut Object,
	pub library_f16: *mut Object,
	pub pso_full: *mut Object,
	pub pso_half: *mut Object,
}

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

#[cfg(shader_hotreload)]
static SHADER_DIRS: OnceLock<Mutex<Option<(std::path::PathBuf, Vec<std::path::PathBuf>)>>> = OnceLock::new();

#[cfg(shader_hotreload)]
fn shader_dirs() -> &'static Mutex<Option<(std::path::PathBuf, Vec<std::path::PathBuf>)>> {
	SHADER_DIRS.get_or_init(|| Mutex::new(None))
}

#[cfg(shader_hotreload)]
pub fn set_shader_dirs(shader_dir: std::path::PathBuf, include_dirs: Vec<std::path::PathBuf>) {
	log::info!("[Metal/HotReload] Shader source dir: {}", shader_dir.display());
	for d in &include_dirs {
		log::info!("[Metal/HotReload] Include dir: {}", d.display());
	}
	*shader_dirs().lock() = Some((shader_dir, include_dirs));
}

/// Retrieves a pair of pipeline state objects (PSOs) for the given device and shader source.
///
/// Under `shader_hotreload`, reads .vekl from disk on cache miss, flattens includes,
/// and passes the expanded source to Metal's runtime compiler.
/// Falls back to the build-time embedded source on failure.
///
/// # Safety
/// - `device` must be a valid Metal device pointer.
/// - `shader_src` and `fname` must point to valid static strings.
fn inject_metal_dispatch_params(src: &str) -> String {
	let needle = "kernel void ";
	let mut out = String::with_capacity(src.len() + 256);
	let mut pos = 0;
	while let Some(off) = src[pos..].find(needle) {
		let abs = pos + off;
		out.push_str(&src[pos..abs]);
		out.push_str(needle);
		let after = abs + needle.len();
		if let Some(paren) = src[after..].find('(') {
			let paren_abs = after + paren;
			out.push_str(&src[after..=paren_abs]);
			out.push_str("\n    uint2 __vekl_dispatch_id [[thread_position_in_grid]],\n    uint2 __vekl_dispatch_size [[grid_size]],");
			pos = paren_abs + 1;
		} else {
			out.push_str(&src[after..]);
			pos = src.len();
			break;
		}
	}
	out.push_str(&src[pos..]);
	out
}

pub unsafe fn load_kernel(device: *mut Object, shader_src: &'static str, fname: &'static str) -> Result<(*mut Object, *mut Object), &'static str> {
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

	let raw_src: Cow<'static, str> = {
		#[cfg(shader_hotreload)]
		{
			use crate::gpu::shaders::expand_includes_runtime;
			use std::time::Instant;

			let guard = shader_dirs().lock();
			if let Some((shader_dir, include_dirs)) = guard.as_ref() {
				let vekl_path = shader_dir.join(format!("{fname}.vekl"));
				match std::fs::read_to_string(&vekl_path) {
					Ok(src) => {
						log::info!("[Metal/HotReload] Compiling: {fname} ({} bytes) from {}", src.len(), vekl_path.display());
						let start = Instant::now();
						match expand_includes_runtime(&src, shader_dir, include_dirs) {
							Ok(expanded) => {
								let elapsed = start.elapsed();
								log::info!(
									"[Metal/HotReload] Flattened '{fname}' in {:.1}ms ({} bytes expanded)",
									elapsed.as_secs_f64() * 1000.0,
									expanded.len()
								);
								Cow::Owned(expanded)
							}
							Err(e) => {
								log::error!("[Metal/HotReload] Include expansion failed for '{fname}': {e}");
								log::warn!("[Metal/HotReload] Falling back to embedded source for '{fname}'");
								Cow::Borrowed(shader_src)
							}
						}
					}
					Err(e) => {
						log::warn!("[Metal/HotReload] Failed to read {}: {e} — using embedded source", vekl_path.display());
						Cow::Borrowed(shader_src)
					}
				}
			} else {
				log::warn!("[Metal/HotReload] No shader dirs registered — using embedded source for '{fname}'");
				Cow::Borrowed(shader_src)
			}
		}
		#[cfg(not(shader_hotreload))]
		{
			Cow::Borrowed(shader_src)
		}
	};

	let injected = inject_metal_dispatch_params(&raw_src);
	let src = unsafe { nsstring_utf8(&injected) };
	let mut error: *mut Object = std::ptr::null_mut();

	// Compile f32 library (+1 retained: alloc+init on opts, new* on lib)
	let opts_f32: *mut Object = msg_send![class!(MTLCompileOptions), alloc];
	let opts_f32: *mut Object = msg_send![opts_f32, init];

	let lib_f32: *mut Object = msg_send![device, newLibraryWithSource: src options: opts_f32 error: &mut error];
	let _: () = msg_send![opts_f32, release];
	if lib_f32.is_null() {
		if let Some(msg) = unsafe { ns_error(error) } {
			log::error!("[Metal] newLibraryWithSource (f32) failed: {msg}");
		}
		return Err("library f32 compile failed");
	}

	// Compile f16 library with USE_HALF_PRECISION=1 preprocessor macro
	let opts_f16: *mut Object = msg_send![class!(MTLCompileOptions), alloc];
	let opts_f16: *mut Object = msg_send![opts_f16, init];

	// dictionary is autoreleased; numberWithInt is autoreleased
	let macros: *mut Object = msg_send![class!(NSMutableDictionary), dictionary];
	let key_vekl: *mut Object = unsafe { nsstring_utf8("VEKL_METAL") };
	let val_one: *mut Object = msg_send![class!(NSNumber), numberWithInt: 1];
	let _: () = msg_send![macros, setObject: val_one forKey: key_vekl];
	let key_macro: *mut Object = unsafe { nsstring_utf8("USE_HALF_PRECISION") };
	let val_macro: *mut Object = msg_send![class!(NSNumber), numberWithInt: 1];
	let _: () = msg_send![macros, setObject: val_macro forKey: key_macro];
	let _: () = msg_send![opts_f16, setPreprocessorMacros: macros];

	let lib_f16: *mut Object = msg_send![device, newLibraryWithSource: src options: opts_f16 error: &mut error];
	let _: () = msg_send![opts_f16, release];
	if lib_f16.is_null() {
		let _: () = msg_send![lib_f32, release];
		if let Some(msg) = unsafe { ns_error(error) } {
			log::error!("[Metal] newLibraryWithSource (f16) failed: {msg}");
		}
		return Err("library f16 compile failed");
	}

	// Extract kernel functions (+1 retained from newFunctionWithName)
	let fname_ns = unsafe { nsstring_utf8(fname) };
	let func_f32: *mut Object = msg_send![lib_f32, newFunctionWithName: fname_ns];
	let func_f16: *mut Object = msg_send![lib_f16, newFunctionWithName: fname_ns];
	if func_f32.is_null() || func_f16.is_null() {
		if !func_f32.is_null() {
			let _: () = msg_send![func_f32, release];
		}
		if !func_f16.is_null() {
			let _: () = msg_send![func_f16, release];
		}
		let _: () = msg_send![lib_f32, release];
		let _: () = msg_send![lib_f16, release];
		log::error!("[Metal] function '{fname}' not found in libraries");
		return Err("function not found");
	}

	// Build pipeline state objects (+1 retained from new*)
	let mut err1: *mut Object = std::ptr::null_mut();
	let mut err2: *mut Object = std::ptr::null_mut();
	let pso_f32: *mut Object = msg_send![device, newComputePipelineStateWithFunction: func_f32 error: &mut err1];
	let pso_f16: *mut Object = msg_send![device, newComputePipelineStateWithFunction: func_f16 error: &mut err2];

	// Functions retained by PSOs — release our ref
	let _: () = msg_send![func_f32, release];
	let _: () = msg_send![func_f16, release];

	if pso_f32.is_null() || pso_f16.is_null() {
		if !pso_f32.is_null() {
			let _: () = msg_send![pso_f32, release];
		}
		if !pso_f16.is_null() {
			let _: () = msg_send![pso_f16, release];
		}
		let _: () = msg_send![lib_f32, release];
		let _: () = msg_send![lib_f16, release];
		log::error!("[Metal] pipeline creation failed: {err1:?} / {err2:?}");
		return Err("pipeline failed");
	}

	{
		let mut guard = map.lock();
		guard.insert(
			key,
			Pipelines {
				library_f32: lib_f32,
				library_f16: lib_f16,
				pso_full: pso_f32,
				pso_half: pso_f16,
			},
		);
	}

	log::info!("[Metal] Built pipelines for device={device:p} entry='{fname}'");
	Ok((pso_f32, pso_f16))
}

/// # Safety
/// Must be called when no other threads are accessing the pipeline cache.
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
			if !p.library_f32.is_null() {
				let _: () = msg_send![p.library_f32, release];
			}
			if !p.library_f16.is_null() {
				let _: () = msg_send![p.library_f16, release];
			}
		}
		log::info!("[Metal] Pipeline cache cleared");
	}
}

pub fn hot_reload() {
	unsafe { cleanup() };
	#[cfg(shader_hotreload)]
	log::info!("[Metal/HotReload] Cache cleared - next dispatch will recompile from disk.");
	#[cfg(not(shader_hotreload))]
	log::info!("[Metal] Cache cleared.");
}
