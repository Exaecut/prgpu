use std::path::Path;

use crate::reflection::Reflection;

/// Generate a C++ bridge wrapper that translates the Rust `extern "C"` dispatch
/// convention into Slang's CPU ABI (`ComputeVaryingInput` + `EntryPointParams`).
///
/// The bridge exposes:
/// - `extern "C" {name}_cpu_dispatch(gid_x, gid_y, buffers, transition_params, user_params)`
/// - `extern "C" {name}_cpu_dispatch_tile(y0, y1, width, buffers, transition_params, user_params)`
///
/// These match the symbols declared by `declare_kernel!` on the Rust side.
pub fn generate_bridge(
	name: &str,
	reflection: &Reflection,
	sdk_path: &Path,
	out_dir: &Path,
) -> std::path::PathBuf {
	let sdk_include = sdk_path.join("include");
	let sdk_include_str = sdk_include.to_str().unwrap_or(".");

	let ep = reflection.entry_points.first().expect("no entry points in reflection");
	let tg = ep.thread_group_size;
	let tg_x = tg[0] as u32;
	let tg_y = tg[1] as u32;

	// Collect bindable params (those with "uniform" kind and non-zero size in EntryPointParams)
	let bindable: Vec<_> = ep.parameters.iter()
		.filter(|p| p.binding.as_ref().map_or(false, |b| b.kind == "uniform" && b.size.unwrap_or(0) > 0))
		.collect();

	let ep_total_size: u64 = bindable.iter()
		.map(|p| p.binding.as_ref().unwrap().size.unwrap_or(0))
		.sum();

	// Separate into structured buffers and constant buffer pointers.
	// Buffers are mapped sequentially to the `buffers[]` array from Rust.
	// CBufs are mapped: first → transition_params (FrameParams), second → user_params.
	let mut buffer_fills: Vec<String> = Vec::new();
	let mut cbuf_offsets: Vec<u64> = Vec::new();
	let mut buffer_slot = 0usize;

	for param in &bindable {
		let b = param.binding.as_ref().unwrap();
		let offset = b.offset.unwrap_or(0);

		if param.ty.kind == "resource" {
			// StructuredBuffer / RWStructuredBuffer: {T* data, size_t count} = 16 bytes
			buffer_fills.push(format!(
				"    *reinterpret_cast<uint32_t**>(ep + {offset}) = reinterpret_cast<uint32_t*>(const_cast<void*>(buffers[{slot}]));\n    \
				 *reinterpret_cast<size_t*>(ep + {count_off}) = static_cast<size_t>(-1) / sizeof(uint32_t);",
				offset = offset,
				count_off = offset + 8,
				slot = buffer_slot,
			));
			buffer_slot += 1;
		} else if param.ty.kind == "constantBuffer" {
			cbuf_offsets.push(offset);
		}
	}

	let bridge_path = out_dir.join(format!("{name}_cpu_dispatch.cpp"));

	let mut out = String::new();

	out.push_str(&format!("// Auto-generated bridge wrapper for Slang CPU kernel '{name}'\n"));
	out.push_str("#include <cstddef>\n");
	out.push_str("#include <cstdint>\n");
	out.push_str(&format!("#include \"{sdk_include_str}/slang-cpp-types.h\"\n\n"));

	out.push_str("#ifdef SLANG_PRELUDE_NAMESPACE\n");
	out.push_str("using namespace SLANG_PRELUDE_NAMESPACE;\n");
	out.push_str("#endif\n\n");

	// Forward-declare the Slang entry functions
	out.push_str(&format!("extern \"C\" void {name}_Thread(ComputeThreadVaryingInput*, void*, void*);\n"));
	out.push_str(&format!("extern \"C\" void {name}(ComputeVaryingInput*, void*, void*);\n\n"));

	// Helper: fill EntryPointParams from flat buffer/param pointers
	out.push_str(&format!("static inline void fill_entry_params_{name}(\n"));
	out.push_str("    uint8_t* ep,\n");
	out.push_str("    const void* const* buffers,\n");
	out.push_str("    const void* transition_params,\n");
	out.push_str("    const void* user_params) {\n");
	out.push_str("    (void)transition_params; (void)user_params;\n");

	for fill in &buffer_fills {
		out.push_str(fill);
		out.push('\n');
	}

	for (i, &offset) in cbuf_offsets.iter().enumerate() {
		let src = if i == 0 { "transition_params" } else { "user_params" };
		out.push_str(&format!(
			"    *reinterpret_cast<const void**>(ep + {offset}) = {src};\n",
		));
	}

	out.push_str("}\n\n");

	// Per-pixel dispatch (used by AE iterate_with path)
	out.push_str(&format!("extern \"C\" void {name}_cpu_dispatch(\n"));
	out.push_str("    unsigned int gid_x,\n");
	out.push_str("    unsigned int gid_y,\n");
	out.push_str("    const void* const* buffers,\n");
	out.push_str("    const void* transition_params,\n");
	out.push_str("    const void* user_params) {\n");
	out.push_str(&format!("    uint8_t ep_bytes[{ep_total_size}] = {{}};\n"));
	out.push_str(&format!("    fill_entry_params_{name}(ep_bytes, buffers, transition_params, user_params);\n\n"));
	out.push_str("    ComputeThreadVaryingInput vi;\n");
	out.push_str(&format!("    vi.groupID = uint3{{gid_x / {tg_x}, gid_y / {tg_y}, 0}};\n"));
	out.push_str(&format!("    vi.groupThreadID = uint3{{gid_x % {tg_x}, gid_y % {tg_y}, 0}};\n"));
	out.push_str(&format!("    {name}_Thread(&vi, ep_bytes, nullptr);\n"));
	out.push_str("}\n\n");

	// Tile dispatch (used by rayon path)
	out.push_str(&format!("extern \"C\" void {name}_cpu_dispatch_tile(\n"));
	out.push_str("    unsigned int y0,\n");
	out.push_str("    unsigned int y1,\n");
	out.push_str("    unsigned int width,\n");
	out.push_str("    const void* const* buffers,\n");
	out.push_str("    const void* transition_params,\n");
	out.push_str("    const void* user_params) {\n");
	out.push_str(&format!("    uint8_t ep_bytes[{ep_total_size}] = {{}};\n"));
	out.push_str(&format!("    fill_entry_params_{name}(ep_bytes, buffers, transition_params, user_params);\n\n"));
	out.push_str("    ComputeVaryingInput vi;\n");
	out.push_str(&format!("    vi.startGroupID = uint3{{0, y0 / {tg_y}, 0}};\n"));
	out.push_str(&format!("    vi.endGroupID = uint3{{(width + {tg_x} - 1) / {tg_x}, (y1 + {tg_y} - 1) / {tg_y}, 1}};\n"));
	out.push_str(&format!("    {name}(&vi, ep_bytes, nullptr);\n"));
	out.push_str("}\n");

	std::fs::write(&bridge_path, &out)
		.unwrap_or_else(|e| panic!("failed to write bridge {}: {e}", bridge_path.display()));
	bridge_path
}

/// Compile all C++ sources (Slang-generated + bridge wrappers) into one static library.
pub fn compile_cpu_all(cpp_paths: &[&Path], sdk_path: &Path) {
	let sdk_include = sdk_path.join("include");

	let mut build = cc::Build::new();
	build
		.cpp(true)
		.opt_level(2)
		.flag_if_supported("-std=c++17")
		.include(&sdk_include);

	for path in cpp_paths {
		build.file(path);
	}

	let pkg_name = std::env::var("CARGO_PKG_NAME").unwrap_or("unknown".into());
	let lib_name = format!("{}_slang_cpu", pkg_name);
	build.compile(&lib_name);

	println!("cargo:warning=[slang] C++ compiled {} files to static .a — zero runtime deps", cpp_paths.len());
}
