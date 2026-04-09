use std::{error::Error, path::PathBuf};

use cudarc::nvrtc::{CompileError, CompileOptions};

type DynError = Box<dyn Error + Send + Sync>;

pub fn parse_nvrtc_error(err: &CompileError) -> String {
    match err {
        CompileError::CompileError { log, .. } => {
            let log = log.to_string_lossy();

            let mut out = String::new();
            let mut current_block = Vec::new();

            for line in log.lines() {
                let line = line.trim_end();

                if line.contains("note #") {
                    continue;
                }

                if line.contains("): error:") && !current_block.is_empty() {
                    out.push_str(&format_block(&current_block));
                    current_block.clear();
                }

                current_block.push(line.to_string());
            }

            if !current_block.is_empty() {
                out.push_str(&format_block(&current_block));
            }

            if out.is_empty() {
                log.into_owned()
            } else {
                out
            }
        }

        other => format!("{:#?}", other),
    }
}

fn format_block(block: &[String]) -> String {
    let mut out = String::new();

    if let Some(header) = block.first()
        && let Some((path_part, rest)) = header.split_once("): ")
        && let Some((path, line)) = path_part.rsplit_once('(')
    {
        let file = path.split('\\').next_back().unwrap_or(path);

        out.push_str(&format!("\nerror: {}\n", rest));
        out.push_str(&format!(" --> {}:{}\n", file, line));
        out.push_str("  |\n");

        for l in &block[1..] {
            out.push_str(&format!("  {}\n", l));
        }

        return out;
    }

    for l in block {
        out.push_str(l);
        out.push('\n');
    }

    out
}

#[derive(Debug, Clone)]
enum ParamKind {
    Ro,
    Rw,
    Wo,
    Cbuf,
}

#[derive(Debug, Clone)]
struct KernelParam {
    kind: ParamKind,
    type_name: String,
    name: String,
    #[allow(dead_code)]
    slot: u32,
}

#[derive(Debug)]
struct KernelSignature {
    name: String,
    params: Vec<KernelParam>,
}

fn parse_kernel_signature(src: &str) -> Option<KernelSignature> {
    let re_kernel = regex_lite::Regex::new(r"kernel\s+void\s+(\w+)\s*\(([\s\S]*?)\)\s*\{").ok()?;

    let caps = re_kernel.captures(src)?;
    let name = caps.get(1)?.as_str().to_string();
    let raw_params = caps.get(2)?.as_str();

    let re_param = regex_lite::Regex::new(
        r"param_dev_(ro|rw|wo|cbuf)\s*\(\s*([\w:]+)\s*,\s*(\w+)\s*,\s*(\d+)\s*\)",
    )
    .ok()?;

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

/// Generic CPU dispatch ABI:
///   void <name>_cpu_dispatch(
///       const void* const* buffers,          // [outgoing, incoming, dest, ...]
///       const void* transition_params,       // FrameParams* (contains width/height for loop)
///       const void* user_params              // effect-specific UserParams*
///   );
///
/// Width/height are extracted from FrameParams.width/.height for dispatch loop bounds.
const PIXEL_TYPE_NAMES: &[&str] = &["pixel", "pixel_format"];

fn is_pixel_type(type_name: &str) -> bool {
    PIXEL_TYPE_NAMES.contains(&type_name)
}

fn generate_cpu_dispatch_wrapper(shader_abs_path: &str, sig: &KernelSignature) -> String {
    let mut out = String::new();

    out.push_str("// AUTO-GENERATED CPU dispatch wrapper. Do not edit.\n");
    out.push_str(&format!(
        "#include \"{}\"\n\n",
        shader_abs_path.replace('\\', "/")
    ));

    out.push_str("#ifdef __cplusplus\n");
    out.push_str("extern \"C\" {\n");
    out.push_str("#endif\n\n");

    out.push_str(&format!("void {}_cpu_dispatch(\n", sig.name));
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
                    out.push_str(&format!(
                        "    const void * __restrict {} = (const void *)__buffers[{}];\n",
                        p.name, buf_idx
                    ));
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
                    out.push_str(&format!(
                        "    void * __restrict {} = (void *)__buffers[{}];\n",
                        p.name, buf_idx
                    ));
                } else {
                    out.push_str(&format!(
                        "    {} * __restrict {} = ({} *)__buffers[{}];\n",
                        p.type_name, p.name, p.type_name, buf_idx
                    ));
                }
                buf_idx += 1;
                forward_args.push(p.name.clone());
            }
            ParamKind::Wo => {
                if is_pixel_type(&p.type_name) {
                    out.push_str(&format!(
                        "    void * __restrict {} = (void *)__buffers[{}];\n",
                        p.name, buf_idx
                    ));
                } else {
                    out.push_str(&format!(
                        "    {} * __restrict {} = ({} *)__buffers[{}];\n",
                        p.type_name, p.name, p.type_name, buf_idx
                    ));
                }
                buf_idx += 1;
                forward_args.push(p.name.clone());
            }
            ParamKind::Cbuf if p.type_name == "FrameParams" => {
                out.push_str(&format!(
                    "    const {} {} = *(const {} *)__transition_params;\n",
                    p.type_name, p.name, p.type_name
                ));
                tp_name = p.name.clone();
                forward_args.push(p.name.clone());
            }
            ParamKind::Cbuf => {
                out.push_str(&format!(
                    "    const {} {} = *(const {} *)__user_params;\n",
                    p.type_name, p.name, p.type_name
                ));
                forward_args.push(p.name.clone());
            }
        }
    }

    if tp_name.is_empty() {
        panic!(
            "Kernel '{}' has no param_dev_cbuf(FrameParams, ...) — required for CPU dispatch",
            sig.name
        );
    }

    out.push('\n');
    out.push_str(&format!("    __cpu_dispatch_w = {}.width;\n", tp_name));
    out.push_str(&format!("    __cpu_dispatch_h = {}.height;\n", tp_name));
    out.push_str(&format!("    __cpu_format = {}.bpp;\n", tp_name));
    out.push_str(&format!(
        "    for (unsigned int __y = 0; __y < {}.height; ++__y) {{\n",
        tp_name
    ));
    out.push_str(&format!(
        "        for (unsigned int __x = 0; __x < {}.width; ++__x) {{\n",
        tp_name
    ));
    out.push_str("            __cpu_gid_x = __x;\n");
    out.push_str("            __cpu_gid_y = __y;\n");
    out.push_str(&format!(
        "            {}({});\n",
        sig.name,
        forward_args.join(", ")
    ));
    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    out.push_str("#ifdef __cplusplus\n");
    out.push_str("}\n");
    out.push_str("#endif\n");

    out
}

pub fn compile_shaders(shader_dir: &str) -> Result<(), DynError> {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let utils = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("vekl")
        .canonicalize()
        .unwrap();
    let utils_str = utils.to_string_lossy().replace("\\\\?\\", "");

    let cuda_path = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or("/usr/local/cuda".into());
    let cuda_include = PathBuf::from(&cuda_path)
        .join("include")
        .canonicalize()
        .unwrap();

    let mut cpu_sources: Vec<(String, PathBuf)> = Vec::new();

    for entry in std::fs::read_dir(shader_dir).unwrap() {
        let path = entry.unwrap().path();

        if path.extension().and_then(|s| s.to_str()) != Some("vekl") {
            continue;
        }

        let name = path.file_stem().unwrap().to_str().unwrap().to_string();
        let src = std::fs::read_to_string(&path).unwrap();

        // GPU: compile f32 and f16 PTX variants via NVRTC.
        for (suffix, half_precision) in [("", false), ("_f16", true)] {
            let mut extra_opts = vec![
                "--std=c++14".into(),
                "--extra-device-vectorization".into(),
                "--device-as-default-execution-space".into(),
                "-DVEKL_CUDA=1".into(),
            ];
            if half_precision {
                extra_opts.push("-DUSE_HALF_PRECISION=1".into());
            }

            let opts = CompileOptions {
                ftz: Some(true),
                prec_sqrt: Some(false),
                prec_div: Some(false),
                fmad: Some(true),
                use_fast_math: None,
                include_paths: vec![
                    utils_str.clone(),
                    cuda_include.to_string_lossy().replace("\\\\?\\", ""),
                ],
                arch: Some("compute_86"),
                options: extra_opts,
                ..Default::default()
            };

            let tag = format!("{name}{suffix}");
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(&src, opts).map_err(|e| {
                let pretty = parse_nvrtc_error(&e);
                eprintln!("Compile failed [{tag}]:\n{pretty}");
                println!("cargo:warning=Compile failed [{tag}]:\n{pretty}");
                Box::new(e) as DynError
            })?;

            let ptx_path = PathBuf::from(&out_dir).join(format!("{tag}.ptx"));
            let ptx_bytes = ptx.as_bytes().unwrap();
            let ptx_bytes = if ptx_bytes.last() == Some(&0) {
                &ptx_bytes[..ptx_bytes.len() - 1]
            } else {
                ptx_bytes
            };
            std::fs::write(&ptx_path, ptx_bytes)?;
            println!(
                "cargo:warning=Shader compiled successfully to -> {}",
                ptx_path.to_str().unwrap()
            );
        }

        // CPU: Generate dispatch wrapper .cpp
        let sig = parse_kernel_signature(&src)
            .ok_or_else(|| format!("Failed to parse kernel signature in {}", path.display()))?;

        let shader_abs = path.canonicalize().unwrap();
        let shader_abs_str = shader_abs
            .to_string_lossy()
            .replace("\\\\?\\", "")
            .replace("//?/", "")
            .replace("\\", "/");
        let wrapper_code = generate_cpu_dispatch_wrapper(&shader_abs_str, &sig);

        let wrapper_path = PathBuf::from(&out_dir).join(format!("{}_cpu_dispatch.cpp", name));
        std::fs::write(&wrapper_path, &wrapper_code)?;

        println!(
            "cargo:warning=CPU dispatch wrapper generated -> {}",
            wrapper_path.to_str().unwrap()
        );

        cpu_sources.push((name, wrapper_path));
    }

    // CPU: Compile all wrappers into a static library via cc
    if !cpu_sources.is_empty() {
        let shader_dir_abs = PathBuf::from(shader_dir).canonicalize().unwrap();

        let mut build = cc::Build::new();
        build
            .cpp(true)
            .opt_level(3)
            .include(&utils_str)
            .include(shader_dir_abs.to_str().unwrap())
            .define("VEKL_CPU", Some("1"))
            .flag_if_supported("/std:c++14") // MSVC
            .flag_if_supported("-std=c++14") // Clang/GCC
            .flag_if_supported("/fp:fast") // MSVC fast math
            .flag_if_supported("-ffast-math") // Clang/GCC fast math
            .flag_if_supported("/Oi") // MSVC intrinsics
            .flag_if_supported("/arch:AVX2") // MSVC SIMD
            .flag_if_supported("-mavx2") // Clang/GCC SIMD
            .flag_if_supported("-mfma"); // Clang/GCC FMA

        for (name, wrapper_path) in &cpu_sources {
            build.file(wrapper_path);
            println!("cargo:warning=Compiling CPU kernel: {}", name);
        }

        let pkg_name = std::env::var("CARGO_PKG_NAME").unwrap_or("unknown".into());
        let lib_name = format!("{}_cpu_kernels", pkg_name);
        build.compile(&lib_name);
        println!("cargo:warning=CPU shader library compiled: {}", lib_name);
    }

    Ok(())
}
