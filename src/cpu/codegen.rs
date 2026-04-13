// Shared kernel codegen: shared between the build-time `build/mod.rs` and the
// runtime `cpu/pipeline.rs` (when `shader_hotreload` is active).
//
// Gated in `cpu/mod.rs` by:
//   #[cfg(any(feature = "build", all(debug_assertions, feature = "shader_hotreload")))]

#[derive(Debug, Clone)]
pub(crate) enum ParamKind {
	Ro,
	Rw,
	Wo,
	Cbuf,
}

#[derive(Debug, Clone)]
pub(crate) struct KernelParam {
	pub kind: ParamKind,
	pub type_name: String,
	pub name: String,
	#[allow(dead_code)]
	pub slot: u32,
}

#[derive(Debug)]
pub(crate) struct KernelSignature {
	pub name: String,
	pub params: Vec<KernelParam>,
}

pub(crate) fn parse_kernel_signature(src: &str) -> Option<KernelSignature> {
	let re_kernel = regex_lite::Regex::new(r"kernel\s+void\s+(\w+)\s*\(([\s\S]*?)\)\s*\{").ok()?;

	let caps = re_kernel.captures(src)?;
	let name = caps.get(1)?.as_str().to_string();
	let raw_params = caps.get(2)?.as_str();

	let re_param =
		regex_lite::Regex::new(r"param_dev_(ro|rw|wo|cbuf)\s*\(\s*([\w:]+)\s*,\s*(\w+)\s*,\s*(\d+)\s*\)").ok()?;

	let mut params = Vec::new();
	for cap in re_param.captures_iter(raw_params) {
		let kind = match &cap[1] {
			"ro" => ParamKind::Ro,
			"rw" => ParamKind::Rw,
			"wo" => ParamKind::Wo,
			"cbuf" => ParamKind::Cbuf,
			_ => continue,
		};
		params.push(KernelParam {
			kind,
			type_name: cap[2].to_string(),
			name: cap[3].to_string(),
			slot: cap[4].parse().unwrap_or(0),
		});
	}

	Some(KernelSignature { name, params })
}

const PIXEL_TYPE_NAMES: &[&str] = &["pixel", "pixel_format"];

fn is_pixel_type(type_name: &str) -> bool {
	PIXEL_TYPE_NAMES.contains(&type_name)
}

/// Generates the C++ dispatch wrapper for the given kernel signature.
///
/// The emitted wrapper:
/// - is `extern "C"` for stable C ABI
/// - exports the symbol on Windows via `__declspec(dllexport)` (harmless for
///   static-lib builds; required for shared-lib hot-reload builds)
/// - on non-Windows the symbol is exported by default with `-shared`
///
/// Per-pixel CPU dispatch ABI:
/// ```cpp
/// void <name>_cpu_dispatch(
///     unsigned int gid_x,
///     unsigned int gid_y,
///     const void* const* buffers,
///     const void* transition_params,   // FrameParams*
///     const void* user_params          // effect-specific UserParams*
/// );
/// ```
pub(crate) fn generate_cpu_dispatch_wrapper(shader_abs_path: &str, sig: &KernelSignature) -> String {
	let mut out = String::new();

	out.push_str("// AUTO-GENERATED CPU dispatch wrapper. Do not edit.\n");
	out.push_str(&format!("#define VEKL_KERNEL_NAME \"{}\"\n", sig.name));
	if cfg!(debug_assertions) {
		out.push_str("#define DEBUG 1\n");
	}
	out.push_str(&format!("#include \"{}\"\n\n", shader_abs_path.replace('\\', "/")));

	// Portable export decorator.
	// Required when building as a shared library on Windows (hot-reload path).
	// __declspec(dllexport) on a static-lib symbol is harmless on MSVC.
	// On non-Windows, -shared exports all extern "C" symbols by default.
	out.push_str("#ifdef _WIN32\n");
	out.push_str("#  define VEKL_EXPORT __declspec(dllexport)\n");
	out.push_str("#else\n");
	out.push_str("#  define VEKL_EXPORT\n");
	out.push_str("#endif\n\n");

	out.push_str("#ifdef __cplusplus\n");
	out.push_str("extern \"C\" {\n");
	out.push_str("#endif\n\n");

	out.push_str(&format!("VEKL_EXPORT void {}_cpu_dispatch(\n", sig.name));
	out.push_str("    unsigned int __gid_x,\n");
	out.push_str("    unsigned int __gid_y,\n");
	out.push_str("    const void* const* __buffers,\n");
	out.push_str("    const void* __transition_params,\n");
	out.push_str("    const void* __user_params\n");
	out.push_str(") {\n");

	let mut forward_args = Vec::new();
	let mut buf_idx = 0u32;
	let mut tp_name = String::new();

	for p in &sig.params {
		match p.kind {
			ParamKind::Ro => {
				if is_pixel_type(&p.type_name) {
					out.push_str(&format!("    const void * __restrict {} = (const void *)__buffers[{}];\n", p.name, buf_idx));
				} else {
					out.push_str(&format!(
						"    const {} * __restrict {} = (const {} *)__buffers[{}];\n",
						p.type_name, p.name, p.type_name, buf_idx
					));
				}
				buf_idx += 1;
				forward_args.push(p.name.clone());
			}
			ParamKind::Rw => {
				if is_pixel_type(&p.type_name) {
					out.push_str(&format!("    void * __restrict {} = (void *)__buffers[{}];\n", p.name, buf_idx));
				} else {
					out.push_str(&format!("    {} * __restrict {} = ({} *)__buffers[{}];\n", p.type_name, p.name, p.type_name, buf_idx));
				}
				buf_idx += 1;
				forward_args.push(p.name.clone());
			}
			ParamKind::Wo => {
				if is_pixel_type(&p.type_name) {
					out.push_str(&format!("    void * __restrict {} = (void *)__buffers[{}];\n", p.name, buf_idx));
				} else {
					out.push_str(&format!("    {} * __restrict {} = ({} *)__buffers[{}];\n", p.type_name, p.name, p.type_name, buf_idx));
				}
				buf_idx += 1;
				forward_args.push(p.name.clone());
			}
			ParamKind::Cbuf if p.type_name == "FrameParams" => {
				out.push_str(&format!("    const {} {} = *(const {} *)__transition_params;\n", p.type_name, p.name, p.type_name));
				tp_name = p.name.clone();
				forward_args.push(p.name.clone());
			}
			ParamKind::Cbuf => {
				out.push_str(&format!("    const {} {} = *(const {} *)__user_params;\n", p.type_name, p.name, p.type_name));
				forward_args.push(p.name.clone());
			}
		}
	}

	if tp_name.is_empty() {
		panic!("Kernel '{}' has no param_dev_cbuf(FrameParams, ...) — required for CPU dispatch", sig.name);
	}

	out.push('\n');
	out.push_str(&format!("    __cpu_dispatch_w = {}.width;\n", tp_name));
	out.push_str(&format!("    __cpu_dispatch_h = {}.height;\n", tp_name));
	out.push_str(&format!("    __cpu_format = {}.bpp;\n", tp_name));
	out.push_str("    __cpu_gid_x = __gid_x;\n");
	out.push_str("    __cpu_gid_y = __gid_y;\n");
	out.push_str(&format!("    {}({});\n", sig.name, forward_args.join(", ")));
	out.push_str("}\n\n");

	out.push_str("#ifdef __cplusplus\n");
	out.push_str("}\n");
	out.push_str("#endif\n");

	out
}
