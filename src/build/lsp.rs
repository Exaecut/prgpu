//! Bootstrap slang-lsp (the Slang VSCode extension) for a downstream effect.
//!
//! Effect crates consume prgpu via `[build-dependencies]` and call
//! [`write_slang_lsp_config`] from their own `build.rs`. The helper:
//!
//! 1. Locates vekl (vendored in prgpu's tarball, or sibling checkout in dev).
//! 2. Syncs it into `<user_crate>/.slang-deps/vekl/` — a gitignored mirror
//!    so the Slang extension's `slang.additionalSearchPaths` can reference a
//!    stable `${workspaceFolder}/.slang-deps/vekl` path that works on every
//!    developer's machine, not a machine-specific
//!    `~/.cargo/registry/src/...` path.
//! 3. Merges `"slang.additionalSearchPaths"` into `.vscode/settings.json`
//!    (preserving any other settings the user already has).
//! 4. Appends `.slang-deps/` to the crate's `.gitignore` idempotently.
//!
//! Because the mirror lives under `.slang-deps/` and the settings file uses
//! `${workspaceFolder}` substitution, **the resulting `.vscode/settings.json`
//! is portable** — committing it works across machines.
//!
//! If the user's `.vscode/settings.json` contains comments (JSONC) or fails
//! to parse as strict JSON, the helper leaves it untouched and emits a
//! `cargo:warning` with the exact snippet to paste manually.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use super::DynError;

/// Return the absolute path to the vekl shader module, if it can be found.
///
/// Checks `CARGO_MANIFEST_DIR/vekl` (vendored copy inside prgpu's published
/// tarball) first, then `CARGO_MANIFEST_DIR/../vekl` (workspace sibling used
/// during Exaecut development). Returns `None` if neither is present.
pub fn vekl_include_path() -> Option<PathBuf> {
	let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
	let vendored = manifest_dir.join("vekl");
	if vendored.is_dir() {
		return Some(vendored);
	}
	manifest_dir
		.parent()
		.map(|p| p.join("vekl"))
		.filter(|p| p.is_dir())
}

/// Wire slang-lsp (the Slang VSCode extension) so editing `.slang` files in
/// this crate gets full autocomplete + hover for vekl imports.
///
/// Call from `build.rs` after [`super::compile_shaders`]:
///
/// ```ignore
/// fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///     prgpu::build::compile_shaders("./shaders")?;
///     prgpu::build::write_slang_lsp_config("./shaders")?;
///     Ok(())
/// }
/// ```
///
/// `shader_dir` is currently advisory — the mirror always lives at
/// `<user_crate>/.slang-lsp/vekl/` regardless. Accepted as a parameter so a
/// future version can write a `.slangd` alongside shaders for editors that
/// prefer the clangd-style config.
///
/// Does nothing if `CARGO_MANIFEST_DIR` is unset (not called from a build
/// script) or if vekl cannot be located.
pub fn write_slang_lsp_config(_shader_dir: &str) -> Result<(), DynError> {
	let user_manifest = match std::env::var("CARGO_MANIFEST_DIR") {
		Ok(v) => PathBuf::from(v),
		Err(_) => {
			// Called outside a build script — silently no-op so docs examples
			// that invoke this at runtime don't panic.
			return Ok(());
		}
	};

	let vekl_src = match vekl_include_path() {
		Some(p) => p,
		None => {
			println!("cargo:warning=[prgpu] write_slang_lsp_config: no vekl include dir found, skipping LSP setup");
			return Ok(());
		}
	};

	let deps_dir = user_manifest.join(".slang-deps");
	let vekl_dst = deps_dir.join("vekl");

	sync_vekl_mirror(&vekl_src, &vekl_dst)?;
	merge_vscode_settings(&user_manifest)?;
	ensure_gitignore(&user_manifest, ".slang-deps/")?;

	// Re-run if the vendored vekl source changes (e.g. prgpu bump).
	println!("cargo:rerun-if-changed={}", vekl_src.display());

	Ok(())
}

/// Recursively mirror `src` to `dst`, filtering to slang / license / readme
/// only. Always overwrites — the mirror is gitignored and small enough
/// (~200 KB) that a fresh copy per build stays under a few ms.
fn sync_vekl_mirror(src: &Path, dst: &Path) -> io::Result<()> {
	// Clean slate so stale files from a previous prgpu version never leak.
	if dst.exists() {
		fs::remove_dir_all(dst)?;
	}
	fs::create_dir_all(dst)?;
	copy_filtered(src, dst)
}

fn copy_filtered(src: &Path, dst: &Path) -> io::Result<()> {
	for entry in fs::read_dir(src)? {
		let entry = entry?;
		let ft = entry.file_type()?;
		let from = entry.path();
		let name = entry.file_name();
		let name_str = name.to_string_lossy();
		let to = dst.join(&name);

		if ft.is_dir() {
			// Skip VCS + node_modules-style junk if the source tree is a
			// full git clone (CI scenario); we only want source files.
			if matches!(name_str.as_ref(), ".git" | "target" | "node_modules") {
				continue;
			}
			fs::create_dir_all(&to)?;
			copy_filtered(&from, &to)?;
		} else if ft.is_file() {
			let keep = name_str.ends_with(".slang")
				|| name_str == "LICENSE"
				|| name_str == "README.md";
			if keep {
				fs::copy(&from, &to)?;
			}
		}
	}
	Ok(())
}

const SEARCH_PATH_VALUE: &str = "${workspaceFolder}/.slang-deps/vekl";

/// Ensure `.vscode/settings.json` contains our search-path entry, preserving
/// every other key the user may have set. JSONC with comments is not
/// supported — if parsing fails we emit a warning rather than clobber.
fn merge_vscode_settings(user_manifest: &Path) -> io::Result<()> {
	let vscode_dir = user_manifest.join(".vscode");
	let settings_path = vscode_dir.join("settings.json");

	if !settings_path.exists() {
		fs::create_dir_all(&vscode_dir)?;
		let initial = format!(
			"{{\n  \"slang.additionalSearchPaths\": [\"{SEARCH_PATH_VALUE}\"]\n}}\n"
		);
		fs::write(&settings_path, initial)?;
		return Ok(());
	}

	let content = fs::read_to_string(&settings_path)?;
	let mut parsed: serde_json::Value = match serde_json::from_str(&content) {
		Ok(v) => v,
		Err(_) => {
			println!(
				"cargo:warning=[prgpu] .vscode/settings.json is not strict JSON (likely has comments). Add {:?} to `slang.additionalSearchPaths` manually.",
				SEARCH_PATH_VALUE
			);
			return Ok(());
		}
	};

	let obj = match parsed.as_object_mut() {
		Some(o) => o,
		None => {
			println!(
				"cargo:warning=[prgpu] .vscode/settings.json root is not a JSON object, leaving it alone"
			);
			return Ok(());
		}
	};

	let entry = obj
		.entry("slang.additionalSearchPaths")
		.or_insert_with(|| serde_json::Value::Array(Vec::new()));

	let arr = match entry.as_array_mut() {
		Some(a) => a,
		None => {
			println!(
				"cargo:warning=[prgpu] slang.additionalSearchPaths in .vscode/settings.json is not an array, leaving it alone"
			);
			return Ok(());
		}
	};

	let already_present = arr
		.iter()
		.any(|v| v.as_str() == Some(SEARCH_PATH_VALUE));
	if already_present {
		return Ok(());
	}

	arr.push(serde_json::Value::String(SEARCH_PATH_VALUE.to_string()));

	let pretty = serde_json::to_string_pretty(&parsed)
		.map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
	// Keep a trailing newline so the file matches common editor conventions.
	let mut pretty = pretty;
	if !pretty.ends_with('\n') {
		pretty.push('\n');
	}
	fs::write(&settings_path, pretty)
}

/// Append `entry` to the crate's `.gitignore` if it isn't already there.
/// Creates the file if it doesn't exist.
fn ensure_gitignore(user_manifest: &Path, entry: &str) -> io::Result<()> {
	let path = user_manifest.join(".gitignore");
	if path.exists() {
		let current = fs::read_to_string(&path)?;
		for line in current.lines() {
			let trimmed = line.trim_end_matches('/');
			if trimmed == entry.trim_end_matches('/') {
				return Ok(());
			}
		}
		let mut next = current;
		if !next.ends_with('\n') {
			next.push('\n');
		}
		next.push_str(entry);
		next.push('\n');
		fs::write(&path, next)
	} else {
		fs::write(&path, format!("{entry}\n"))
	}
}
