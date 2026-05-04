use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const SLANG_VERSION: &str = "2026.8";
pub const SLANG_TAG: &str = "v2026.8";

fn detect_platform() -> (&'static str, &'static str) {
	let os = if cfg!(target_os = "macos") { "macos" }
		else if cfg!(target_os = "windows") { "windows" }
		else if cfg!(target_os = "linux") { "linux" }
		else { panic!("Unsupported OS for Slang SDK") };

	let arch = if cfg!(target_arch = "aarch64") { "aarch64" }
		else if cfg!(target_arch = "x86_64") { "x86_64" }
		else { panic!("Unsupported architecture for Slang SDK") };

	(os, arch)
}

/// SDK directory at `{workspace_root}/target/.slang-sdk/{version}/`.
/// Shared across all workspace members — downloaded once.
/// Auto-downloads if missing.
pub fn sdk_dir() -> PathBuf {
	let target_dir = find_shared_target_dir();
	let sdk = target_dir.join(".slang-sdk").join(SLANG_VERSION);

	if sdk.join("bin").exists() && sdk.join("include").exists() {
		return sdk;
	}

	println!("cargo:warning=[slang] Slang SDK v{SLANG_VERSION} not found, downloading...");
	download_sdk(&sdk);
	sdk
}

/// Find the shared `target/` directory for the workspace.
///
/// In a build.rs, `OUT_DIR` is `<root>/target/debug/build/<crate-hash>/out`.
/// Walk up from OUT_DIR to find the `target/` directory. This ensures every
/// workspace member resolves to the same target directory, so the Slang SDK
/// is downloaded exactly once.
///
/// Works for both workspaces (shared target/) and standalone crates.
fn find_shared_target_dir() -> PathBuf {
	let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
	let mut dir = out_dir.as_path();
	while let Some(parent) = dir.parent() {
		if parent.file_name().is_some_and(|n| n == "target") {
			return parent.to_path_buf();
		}
		dir = parent;
	}
	panic!(
		"Could not locate workspace target/ directory from OUT_DIR={}",
		out_dir.display()
	);
}

pub fn slangc_bin(sdk: &Path) -> PathBuf {
	let bin_name = if cfg!(target_os = "windows") { "slangc.exe" } else { "slangc" };
	sdk.join("bin").join(bin_name)
}

fn download_sdk(dest: &Path) {
	let (os, arch) = detect_platform();
	let filename = format!("slang-{SLANG_VERSION}-{os}-{arch}.tar.gz");
	let url = format!("https://github.com/shader-slang/slang/releases/download/{SLANG_TAG}/{filename}");

	println!("cargo:warning=[slang] Downloading {url}");

	let tmp_dir = dest.with_extension("downloading");
	if tmp_dir.exists() { let _ = fs::remove_dir_all(&tmp_dir); }
	fs::create_dir_all(&tmp_dir).expect("Failed to create temp download dir");

	let archive_path = tmp_dir.join(&filename);
	download_file(&url, &archive_path);

	println!("cargo:warning=[slang] Extracting {filename}...");
	extract_tar_gz(&archive_path, &tmp_dir);

	let sdk_root = find_sdk_root(&tmp_dir)
		.unwrap_or_else(|| panic!("Could not find include/ in extracted archive at {}", tmp_dir.display()));

	if dest.exists() { let _ = fs::remove_dir_all(dest); }
	fs::create_dir_all(dest.parent().unwrap()).ok();

	match fs::rename(&sdk_root, dest) {
		Ok(_) => {}
		Err(_) => {
			println!("cargo:warning=[slang] rename failed, copying...");
			copy_dir_recursive(&sdk_root, dest);
		}
	}

	let _ = fs::remove_dir_all(&tmp_dir);
	println!("cargo:warning=[slang] Slang SDK v{SLANG_VERSION} installed at {}", dest.display());
}

fn download_file(url: &str, dest: &Path) {
	let response = ureq::get(url).call().unwrap_or_else(|e| panic!(
		"Failed to download: {e}\nURL: {url}\nManually extract to: {}",
		dest.parent().unwrap_or(dest).display()
	));

	let total_size: usize = response
		.headers().get("content-length")
		.and_then(|v| v.to_str().ok())
		.and_then(|v| v.parse().ok())
		.unwrap_or(0);

	if total_size > 0 {
		println!("cargo:warning=[slang]   Download size: {:.1} MB", total_size as f64 / (1024.0 * 1024.0));
	}

	let mut file = fs::File::create(dest).expect("Failed to create temp archive file");
	let mut reader = response.into_body().into_reader();
	let mut buf = [0u8; 64 * 1024];
	let mut downloaded: usize = 0;
	let start = Instant::now();
	let mut last_report = Instant::now();

	loop {
		let n = reader.read(&mut buf).expect("Failed to read download chunk");
		if n == 0 { break; }
		file.write_all(&buf[..n]).expect("Failed to write download chunk");
		downloaded += n;

		let now = Instant::now();
		if now.duration_since(last_report).as_millis() >= 500 {
			last_report = now;
			let elapsed = start.elapsed().as_secs_f64();
			let speed = (downloaded as f64 / (1024.0 * 1024.0)) / elapsed.max(0.001);
			if total_size > 0 {
				let pct = downloaded * 100 / total_size;
				println!("cargo:warning=[slang]   {pct}% ({:.1}/{:.1} MB, {:.1} MB/s)",
					downloaded as f64 / (1024.0 * 1024.0), total_size as f64 / (1024.0 * 1024.0), speed);
			} else {
				println!("cargo:warning=[slang]   {:.1} MB downloaded ({:.1} MB/s)",
					downloaded as f64 / (1024.0 * 1024.0), speed);
			}
		}
	}

	let elapsed = start.elapsed().as_secs_f64();
	let speed = (downloaded as f64 / (1024.0 * 1024.0)) / elapsed.max(0.001);
	println!("cargo:warning=[slang]   Download complete: {:.1} MB in {:.1}s ({:.1} MB/s)",
		downloaded as f64 / (1024.0 * 1024.0), elapsed, speed);
}

fn extract_tar_gz(archive_path: &Path, dest: &Path) {
	let file = fs::File::open(archive_path).expect("Failed to open tar.gz archive");
	let gz = flate2::read::GzDecoder::new(file);
	let mut archive = tar::Archive::new(gz);
	#[cfg(unix)] { archive.set_preserve_permissions(true); }
	archive.unpack(dest).expect("Failed to extract tar.gz archive");
}

fn find_sdk_root(dir: &Path) -> Option<PathBuf> {
	if dir.join("include").exists() { return Some(dir.to_path_buf()); }
	for entry in fs::read_dir(dir).ok()? {
		let entry = entry.ok()?;
		if entry.file_type().ok()?.is_dir() && entry.path().join("include").exists() {
			return Some(entry.path());
		}
	}
	None
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
	fs::create_dir_all(dst).ok();
	for entry in fs::read_dir(src).unwrap() {
		let entry = entry.unwrap();
		let src_path = entry.path();
		let dst_path = dst.join(entry.file_name());
		if src_path.is_dir() { copy_dir_recursive(&src_path, &dst_path); }
		else { fs::copy(&src_path, &dst_path).ok(); }
	}
}
