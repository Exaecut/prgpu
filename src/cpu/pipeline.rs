
use after_effects::log;

use std::{collections::HashMap, sync::OnceLock};

use libloading::Library;

use parking_lot::Mutex;

use crate::cpu::render::CpuDispatchFns;

struct KernelEntry {

	_library: Library,
	dispatch_fns: CpuDispatchFns,
}

static CACHE: OnceLock<Mutex<HashMap<&'static str, KernelEntry>>> = OnceLock::new();

static SHADER_DIRS: OnceLock<Mutex<Option<(std::path::PathBuf, Vec<std::path::PathBuf>)>>> = OnceLock::new();

static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
fn cache() -> &'static Mutex<HashMap<&'static str, KernelEntry>> {
	CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[inline]
fn shader_dirs() -> &'static Mutex<Option<(std::path::PathBuf, Vec<std::path::PathBuf>)>> {
	SHADER_DIRS.get_or_init(|| Mutex::new(None))
}

pub fn set_shader_dirs(shader_dir: std::path::PathBuf, include_dirs: Vec<std::path::PathBuf>) {
	log::info!("[CPU/HotReload] Shader source dir: {}", shader_dir.display());
	for d in &include_dirs {
		log::info!("[CPU/HotReload] Include dir: {}", d.display());
	}
	*shader_dirs().lock() = Some((shader_dir, include_dirs));
}

pub fn get_dispatch_fn(kernel_name: &'static str, static_fallback: CpuDispatchFns) -> CpuDispatchFns {
	{
		let guard = cache().lock();
		if let Some(entry) = guard.get(kernel_name) {
			return entry.dispatch_fns;
		}
	}

	match compile_kernel(kernel_name) {
		Ok(entry) => {
			let fns = entry.dispatch_fns;
			let mut guard = cache().lock();

			if let Some(existing) = guard.get(kernel_name) {
				return existing.dispatch_fns;
			}
			guard.insert(kernel_name, entry);
			log::info!("[CPU/HotReload] Using runtime-compiled kernel '{kernel_name}'");
			fns
		}
		Err(e) => {
			log::error!("[CPU/HotReload] {e}");
			log::warn!("[CPU/HotReload] Falling back to statically linked '{kernel_name}'");
			static_fallback
		}
	}
}

pub fn hot_reload() {
	cleanup();
	log::info!("[CPU/HotReload] Cache cleared - next dispatch will recompile from disk.");
}

fn cleanup() {
	GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

	if let Some(map) = CACHE.get() {
		let count = map.lock().drain().count();
		log::info!("[CPU/HotReload] Unloaded {count} kernel(s).");
	}
}

#[cfg(target_os = "windows")]
fn lib_filename(kernel_name: &str, generation: u64) -> String {
	format!("{kernel_name}_cpu_dispatch_gen{generation}.dll")
}

#[cfg(target_os = "macos")]
fn lib_filename(kernel_name: &str, generation: u64) -> String {
	format!("lib{kernel_name}_cpu_dispatch_gen{generation}.dylib")
}

fn compile_kernel(kernel_name: &'static str) -> Result<KernelEntry, String> {
	use std::time::Instant;

	let guard = shader_dirs().lock();
	let (shader_dir, include_dirs) = guard.as_ref().ok_or_else(|| {
		format!(
			"[CPU/HotReload] No shader dirs registered for '{kernel_name}'. \
		         Ensure declare_kernel! is called before the first render."
		)
	})?;

	let vekl_path = shader_dir.join(format!("{kernel_name}.vekl"));
	let src = std::fs::read_to_string(&vekl_path).map_err(|e| format!("[CPU/HotReload] Failed to read '{}': {e}", vekl_path.display()))?;

	log::info!("[CPU/HotReload] Compiling: {kernel_name} ({} bytes) from {}", src.len(), vekl_path.display());

	let sig = crate::cpu::codegen::parse_kernel_signature(&src).ok_or_else(|| format!("[CPU/HotReload] Could not parse kernel signature in '{kernel_name}.vekl'"))?;

	let shader_abs = vekl_path
		.canonicalize()
		.map_err(|e| format!("[CPU/HotReload] Canonicalize '{}': {e}", vekl_path.display()))?;
	let shader_abs_str = shader_abs.to_string_lossy().replace("\\\\?\\", "").replace("//?/", "").replace('\\', "/");

	let wrapper_code = crate::cpu::codegen::generate_cpu_dispatch_wrapper(&shader_abs_str, &sig);

	let hotreload_dir = std::env::temp_dir().join("exaecut_hotreload");
	std::fs::create_dir_all(&hotreload_dir).map_err(|e| format!("[CPU/HotReload] Create temp dir '{}': {e}", hotreload_dir.display()))?;

	let wrapper_path = hotreload_dir.join(format!("{kernel_name}_cpu_dispatch.cpp"));
	std::fs::write(&wrapper_path, &wrapper_code).map_err(|e| format!("[CPU/HotReload] Write wrapper '{}': {e}", wrapper_path.display()))?;

	let generation = GENERATION.load(std::sync::atomic::Ordering::Relaxed);
	let lib_name = lib_filename(kernel_name, generation);
	let lib_path = hotreload_dir.join(&lib_name);

	let shader_dir_owned = shader_dir.clone();
	let include_dirs_owned = include_dirs.clone();
	drop(guard);

	let start = Instant::now();
	compile_to_shared_lib(&wrapper_path, &lib_path, &shader_dir_owned, &include_dirs_owned)?;
	let elapsed = start.elapsed();

	log::info!(
		"[CPU/HotReload] Compiled '{kernel_name}' in {:.1}ms → {}",
		elapsed.as_secs_f64() * 1000.0,
		lib_path.display()
	);

	let lib = unsafe { Library::new(&lib_path) }.map_err(|e| format!("[CPU/HotReload] Load '{}': {e}", lib_path.display()))?;

	let per_pixel_name = format!("{kernel_name}_cpu_dispatch\0");
	let per_pixel_fn: crate::cpu::render::CpuDispatchFn = unsafe {
		let sym: libloading::Symbol<crate::cpu::render::CpuDispatchFn> = lib
			.get(per_pixel_name.as_bytes())
			.map_err(|e| format!("[CPU/HotReload] Per-pixel symbol '{}' not found: {e}", kernel_name))?;

		std::mem::transmute(sym.into_raw())
	};

	let row_batch_name = format!("{kernel_name}_cpu_row_dispatch\0");
	let row_batch_fn: crate::cpu::render::CpuRowBatchFn = unsafe {
		let sym: libloading::Symbol<crate::cpu::render::CpuRowBatchFn> = lib
			.get(row_batch_name.as_bytes())
			.map_err(|e| format!("[CPU/HotReload] Row-batch symbol '{}' not found: {e}", kernel_name))?;

		std::mem::transmute(sym.into_raw())
	};

	Ok(KernelEntry {
		_library: lib,
		dispatch_fns: crate::cpu::render::CpuDispatchFns {
			per_pixel: per_pixel_fn,
			row_batch: row_batch_fn,
		},
	})
}

#[cfg(target_os = "windows")]
fn compile_to_shared_lib(
	wrapper_path: &std::path::Path,
	lib_path: &std::path::Path,
	shader_dir: &std::path::Path,
	include_dirs: &[std::path::PathBuf],
) -> Result<(), String> {

	let tool = cc::Build::new().cpp(true).get_compiler();

	let mut cmd = std::process::Command::new(tool.path());
	for (k, v) in tool.env() {
		cmd.env(k, v);
	}

	let canon = |p: &std::path::Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());

	for dir in include_dirs {
		cmd.arg(format!("/I{}", canon(dir).display()));
	}
	cmd.arg(format!("/I{}", canon(shader_dir).display()));

	cmd.arg("/nologo")
		.arg("/O2")
		.arg("/fp:fast")
		.arg("/std:c++14")
		.arg("/DVEKL_CPU=1")
		.arg("/LD")
		.arg(wrapper_path)
		.arg(format!("/Fe:{}", lib_path.display()));

	if cfg!(debug_assertions) {
		cmd.arg("/DDEBUG=1");
	}

	log::info!("[CPU/HotReload] Invoking compiler '{}'", tool.path().display());
	
	let output = cmd
		.output()
		.map_err(|e| format!("[CPU/HotReload] Invoke compiler '{}': {e}", tool.path().display()))?;

	if !output.status.success() {
		let stderr = String::from_utf8_lossy(&output.stderr);
		let stdout = String::from_utf8_lossy(&output.stdout);
		let diag = if !stderr.is_empty() { stderr.into_owned() } else { stdout.into_owned() };
		return Err(format!(
			"[CPU/HotReload] Compilation failed for '{}' (exit {:?}):\n{}",
			wrapper_path.display(),
			output.status.code(),
			diag
		));
	}

	Ok(())
}

#[cfg(target_os = "macos")]
fn compile_to_shared_lib(
	wrapper_path: &std::path::Path,
	lib_path: &std::path::Path,
	shader_dir: &std::path::Path,
	include_dirs: &[std::path::PathBuf],
) -> Result<(), String> {

	let tool = cc::Build::new().cpp(true).get_compiler();

	let mut cmd = std::process::Command::new(tool.path());
	for (k, v) in tool.env() {
		cmd.env(k, v);
	}

	let canon = |p: &std::path::Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());

	for dir in include_dirs {
		cmd.arg("-I").arg(canon(dir));
	}
	cmd.arg("-I").arg(canon(shader_dir));

	cmd.arg("-shared")
		.arg("-std=c++14")
		.arg("-DVEKL_CPU=1")
		.arg("-O2")
		.arg("-ffast-math")
		.arg("-o")
		.arg(lib_path)
		.arg(wrapper_path);

	if cfg!(debug_assertions) {
		cmd.arg("-DDEBUG=1");
	}

	let output = cmd
		.output()
		.map_err(|e| format!("[CPU/HotReload] Invoke compiler '{}': {e}", tool.path().display()))?;

	if !output.status.success() {
		let stderr = String::from_utf8_lossy(&output.stderr);
		let stdout = String::from_utf8_lossy(&output.stdout);
		let diag = if !stderr.is_empty() { stderr.into_owned() } else { stdout.into_owned() };
		return Err(format!(
			"[CPU/HotReload] Compilation failed for '{}' (exit {:?}):\n{}",
			wrapper_path.display(),
			output.status.code(),
			diag
		));
	}

	Ok(())
}
