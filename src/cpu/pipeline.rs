// CPU kernel hot-reload pipeline.
//
// Mirrors the GPU backends (`cuda::pipeline` and `metal::pipeline`) using the
// same public API surface:
//
//   pub fn hot_reload()          — clears cache; always callable
//   pub fn set_shader_dirs(...)  — registers shader / include dirs (hotreload only)
//   pub fn get_dispatch_fn(...)  — returns cached or freshly compiled fn ptr (hotreload only)
//
// `hot_reload()` is always public and callable — `vignette/src/lib.rs` calls
// `prgpu::cpu::pipeline::hot_reload()` unconditionally alongside the GPU counterpart.
// Under non-hotreload builds it is a no-op that logs a message.
//
// Under `shader_hotreload`:
//   - `CACHE` maps kernel names → `KernelEntry { _library, dispatch_fn }`.
//   - `SHADER_DIRS` holds the .vekl source dir + include paths, registered once
//     per kernel via a `Once` guard inside the `declare_kernel!` macro expansion.
//   - `GENERATION` is an incrementing counter stamped into each compiled DLL/dylib
//     filename, avoiding Windows file-locking across reloads entirely.
//   - On `hot_reload()` the cache is drained — dropping all `Library` handles
//     triggers `FreeLibrary`/`dlclose` — and the generation counter is bumped.
//   - On the next render frame `get_dispatch_fn()` sees a cache miss, reads the
//     .vekl from disk, compiles a shared library, loads it via `libloading`, and
//     resolves the `{name}_cpu_dispatch` symbol.
//   - Any error returns the static fallback (build-time linked symbol).

use after_effects::log;

use std::{collections::HashMap, sync::OnceLock};

use libloading::Library;

use parking_lot::Mutex;

use crate::cpu::render::CpuDispatchFn;

/// A loaded kernel entry: keeps the shared library alive for as long as the
/// function pointer is in use.  Dropping this calls `FreeLibrary`/`dlclose`.

struct KernelEntry {
	/// Keeps the shared library mapped in memory.  Must outlive `dispatch_fn`.
	_library: Library,
	dispatch_fn: CpuDispatchFn,
}

/// Per-kernel cache: maps kernel name → loaded entry.
/// Key is `&'static str` from `stringify!($name)` inside `declare_kernel!`.

static CACHE: OnceLock<Mutex<HashMap<&'static str, KernelEntry>>> = OnceLock::new();

/// Registered shader source directory and extra include paths.
/// Set once per effect on the first CPU kernel dispatch.

static SHADER_DIRS: OnceLock<Mutex<Option<(std::path::PathBuf, Vec<std::path::PathBuf>)>>> = OnceLock::new();

/// Generation counter — stamped into compiled shared-library filenames.
/// Incremented by `hot_reload()` so new DLLs have fresh names, side-stepping
/// the Windows file-locking problem entirely.

static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
fn cache() -> &'static Mutex<HashMap<&'static str, KernelEntry>> {
	CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[inline]
fn shader_dirs() -> &'static Mutex<Option<(std::path::PathBuf, Vec<std::path::PathBuf>)>> {
	SHADER_DIRS.get_or_init(|| Mutex::new(None))
}

/// Registers the shader source directory and extra include paths for this effect.
///
/// Called once per kernel (via a `std::sync::Once` guard in the `declare_kernel!`
/// macro expansion) on the first CPU render frame.  All kernels in one effect share
/// the same `CARGO_MANIFEST_DIR`, so repeated calls are idempotent.

pub fn set_shader_dirs(shader_dir: std::path::PathBuf, include_dirs: Vec<std::path::PathBuf>) {
	log::info!("[CPU/HotReload] Shader source dir: {}", shader_dir.display());
	for d in &include_dirs {
		log::info!("[CPU/HotReload] Include dir: {}", d.display());
	}
	*shader_dirs().lock() = Some((shader_dir, include_dirs));
}

/// Returns the hot-reloaded dispatch function for `kernel_name`, compiling from
/// disk on first use (or after a `hot_reload()` cache clear).
///
/// Falls back to `static_fallback` — the build-time linked symbol — on any error,
/// so the render continues with the last good code.

pub fn get_dispatch_fn(kernel_name: &'static str, static_fallback: CpuDispatchFn) -> CpuDispatchFn {
	// Fast path: cache hit.
	{
		let guard = cache().lock();
		if let Some(entry) = guard.get(kernel_name) {
			return entry.dispatch_fn;
		}
	} // lock released before the slow compilation step

	// Slow path: compile from disk (~100–500 ms).
	match compile_kernel(kernel_name) {
		Ok(entry) => {
			let fn_ptr = entry.dispatch_fn;
			let mut guard = cache().lock();
			// Double-check: a concurrent render may have compiled the same kernel.
			// Prefer theirs; our KernelEntry drops here (Library unloaded).
			if let Some(existing) = guard.get(kernel_name) {
				return existing.dispatch_fn;
			}
			guard.insert(kernel_name, entry);
			log::info!("[CPU/HotReload] Using runtime-compiled kernel '{kernel_name}'");
			fn_ptr
		}
		Err(e) => {
			log::error!("[CPU/HotReload] {e}");
			log::warn!("[CPU/HotReload] Falling back to statically linked '{kernel_name}'");
			static_fallback
		}
	}
}

/// Clears the CPU shader cache, triggering recompilation on the next render frame.
///
/// Mirrors `cuda::pipeline::hot_reload()` and `metal::pipeline::hot_reload()`.
pub fn hot_reload() {
	cleanup();
	log::info!("[CPU/HotReload] Cache cleared - next dispatch will recompile from disk.");
}

fn cleanup() {
	// Bump the generation so the next compile produces a fresh filename.
	// The old dll/dylib is released below when all KernelEntry drops complete.
	GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

	if let Some(map) = CACHE.get() {
		let count = map.lock().drain().count();
		log::info!("[CPU/HotReload] Unloaded {count} kernel(s).");
	}
}

// Returns the platform shared-library filename for the given kernel and generation.
// Defined as separate cfg-gated functions to avoid #[cfg] on let statements.
// The outer `#[cfg(shader_hotreload)] pub mod pipeline;` in cpu/mod.rs means this
// whole file is only compiled in hotreload builds, so no shader_hotreload gate needed here.
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

	// 1. Read .vekl source from disk.
	let vekl_path = shader_dir.join(format!("{kernel_name}.vekl"));
	let src = std::fs::read_to_string(&vekl_path).map_err(|e| format!("[CPU/HotReload] Failed to read '{}': {e}", vekl_path.display()))?;

	log::info!("[CPU/HotReload] Compiling: {kernel_name} ({} bytes) from {}", src.len(), vekl_path.display());

	// 2. Parse kernel signature and generate the C++ wrapper.
	let sig = crate::cpu::codegen::parse_kernel_signature(&src).ok_or_else(|| format!("[CPU/HotReload] Could not parse kernel signature in '{kernel_name}.vekl'"))?;

	let shader_abs = vekl_path
		.canonicalize()
		.map_err(|e| format!("[CPU/HotReload] Canonicalize '{}': {e}", vekl_path.display()))?;
	let shader_abs_str = shader_abs.to_string_lossy().replace("\\\\?\\", "").replace("//?/", "").replace('\\', "/");

	let wrapper_code = crate::cpu::codegen::generate_cpu_dispatch_wrapper(&shader_abs_str, &sig);

	// 3. Write wrapper .cpp to temp directory.
	let hotreload_dir = std::env::temp_dir().join("exaecut_hotreload");
	std::fs::create_dir_all(&hotreload_dir).map_err(|e| format!("[CPU/HotReload] Create temp dir '{}': {e}", hotreload_dir.display()))?;

	let wrapper_path = hotreload_dir.join(format!("{kernel_name}_cpu_dispatch.cpp"));
	std::fs::write(&wrapper_path, &wrapper_code).map_err(|e| format!("[CPU/HotReload] Write wrapper '{}': {e}", wrapper_path.display()))?;

	// 4. Compile to a shared library with a generation-stamped filename.
	let generation = GENERATION.load(std::sync::atomic::Ordering::Relaxed);
	let lib_name = lib_filename(kernel_name, generation);
	let lib_path = hotreload_dir.join(&lib_name);

	// Clone paths before releasing the SHADER_DIRS lock — compile is the slow part.
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

	// 5. Load the shared library and resolve `{name}_cpu_dispatch`.
	let lib = unsafe { Library::new(&lib_path) }.map_err(|e| format!("[CPU/HotReload] Load '{}': {e}", lib_path.display()))?;

	// libloading requires a NUL-terminated byte slice for the symbol name.
	let symbol_name = format!("{kernel_name}_cpu_dispatch\0");
	let dispatch_fn: CpuDispatchFn = unsafe {
		let sym: libloading::Symbol<CpuDispatchFn> = lib
			.get(symbol_name.as_bytes())
			.map_err(|e| format!("[CPU/HotReload] Symbol '{}' not found: {e}", kernel_name))?;
		// SAFETY: We transmute the Symbol's lifetime away so the fn pointer can
		// outlive the borrow.  Safety is upheld by storing the Library in
		// KernelEntry — the library stays mapped as long as dispatch_fn can be called.
		std::mem::transmute(sym.into_raw())
	};

	Ok(KernelEntry { _library: lib, dispatch_fn })
}

// Platform-specific shared-library compilation.
#[cfg(target_os = "windows")]
fn compile_to_shared_lib(
	wrapper_path: &std::path::Path,
	lib_path: &std::path::Path,
	shader_dir: &std::path::Path,
	include_dirs: &[std::path::PathBuf],
) -> Result<(), String> {
	// `cc::Build::get_compiler()` detects MSVC via registry / vcvars and
	// returns the full path to cl.exe plus the VC environment variables
	// (LIB, INCLUDE, PATH) required to run it outside a Developer Command Prompt.
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

	// /nologo  — suppress banner
	// /O1      — minimal optimization (fast iteration compile)
	// /std:c++14
	// /DVEKL_CPU=1
	// /LD      — produce a DLL
	// /Fe:     — DLL output path (lib + exp also emitted alongside it)
	cmd.arg("/nologo")
		.arg("/O1")
		.arg("/std:c++14")
		.arg("/DVEKL_CPU=1")
		.arg("/LD")
		.arg(wrapper_path)
		.arg(format!("/Fe:{}", lib_path.display()));

	if cfg!(debug_assertions) {
		cmd.arg("/DDEBUG=1");
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

#[cfg(target_os = "macos")]
fn compile_to_shared_lib(
	wrapper_path: &std::path::Path,
	lib_path: &std::path::Path,
	shader_dir: &std::path::Path,
	include_dirs: &[std::path::PathBuf],
) -> Result<(), String> {
	// On macOS, `cc::Build::get_compiler()` returns the Xcode-selected clang++
	// (via xcrun).  All extern "C" symbols are exported by default with -shared.
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

	// -shared    — produce a dylib
	// -std=c++14
	// -DVEKL_CPU=1
	// -O1        — minimal optimization for fast iteration
	// -o         — output path
	cmd.arg("-shared")
		.arg("-std=c++14")
		.arg("-DVEKL_CPU=1")
		.arg("-O1")
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
