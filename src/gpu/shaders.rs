use std::{
    fs,
    path::{Path, PathBuf},
};

#[macro_export]
macro_rules! include_shader {
    ($name:ident, cuda) => {{ include_str!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".ptx")) }};

    ($name:literal, cuda) => {{ include_str!(concat!(env!("OUT_DIR"), "/", $name, ".ptx")) }};

    ($name:ident, cuda, halfprecision) => {{
        include_str!(concat!(env!("OUT_DIR"), "/", stringify!($name), "_f16.ptx"))
    }};

    ($name:literal, cuda, halfprecision) => {{
        include_str!(concat!(env!("OUT_DIR"), "/", $name, "_f16.ptx"))
    }};

    ($name:ident, metal) => {{ include_str!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".metal")) }};

    ($name:literal, metal) => {{ include_str!(concat!(env!("OUT_DIR"), "/", $name, ".metal")) }};

    ($name:ident, opencl) => {{ include_str!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".cl")) }};

    ($name:literal, opencl) => {{ include_str!(concat!(env!("OUT_DIR"), "/", $name, ".cl")) }};

    ($name:ident, cpu) => {{ include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/", stringify!($name), ".vekl")) }};

    ($name:literal, cpu) => {{ include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/", $name, ".vekl")) }};
}

fn find_kernel_signature_span(src: &str) -> Option<(usize, usize, usize)> {
    let kernel_off = src.find("kernel void ")?;
    let open_paren = src[kernel_off..].find('(')? + kernel_off;
    let mut depth = 0usize;
    let mut close_paren = None;

    for (idx, ch) in src[open_paren..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    close_paren = Some(open_paren + idx);
                    break;
                }
            }
            _ => {}
        }
    }

    let close_paren = close_paren?;
    let body_start = src[close_paren..].find('{')? + close_paren;
    Some((open_paren, close_paren, body_start))
}

pub fn inject_kernel_params(src: &str, params: &str) -> String {
    let Some((open_paren, close_paren, _body_start)) = find_kernel_signature_span(src) else {
        return src.to_string();
    };

    let existing = src[open_paren + 1..close_paren].trim();
    let insertion = if existing.is_empty() {
        format!("\n{params}\n")
    } else {
        format!(",\n{params}\n")
    };

    let mut out = String::with_capacity(src.len() + insertion.len() + 64);
    out.push_str(&src[..close_paren]);
    out.push_str(&insertion);
    out.push_str(&src[close_paren..]);
    out
}

pub fn inject_kernel_prologue(src: &str, prologue: &str) -> String {
    let Some((_open_paren, _close_paren, body_start)) = find_kernel_signature_span(src) else {
        return src.to_string();
    };

    let mut out = String::with_capacity(src.len() + prologue.len() + 16);
    out.push_str(&src[..body_start + 1]);
    out.push_str(prologue);
    out.push_str(&src[body_start + 1..]);
    out
}

pub fn prefix_kernel_name_define(src: &str, kernel_name: &str) -> String {
    format!("#define VEKL_KERNEL_NAME \"{kernel_name}\"\n{src}")
}

pub fn prepare_cuda_source(src: &str, kernel_name: &str) -> String {
    let src = prefix_kernel_name_define(src, kernel_name);
    let src = inject_kernel_params(&src, "    param_dev_rw(VeklLogBuffer, __vekl_log_buffer, 5)");
    inject_kernel_prologue(&src, "\n    __vekl_bind_log_buffer(__vekl_log_buffer);\n")
}

pub fn prepare_metal_source(src: &str, kernel_name: &str) -> String {
    let src = prefix_kernel_name_define(src, kernel_name);
    let src = inject_kernel_params(
        &src,
        "    device VeklLogBuffer* __vekl_log_buffer [[buffer(5)]],\n    uint2 __vekl_dispatch_id [[thread_position_in_grid]],\n    uint2 __vekl_dispatch_size [[grid_size]]",
    );
    inject_kernel_prologue(&src, "\n    __vekl_bind_log_buffer(__vekl_log_buffer);\n")
}

pub fn expand_includes_runtime(
    src: &str,
    base_dir: &Path,
    include_dirs: &[PathBuf],
) -> Result<String, String> {
    fn expand_one(
        path: &Path,
        include_dirs: &[PathBuf],
        stack: &mut Vec<PathBuf>,
    ) -> Result<String, String> {
        let path = path
            .canonicalize()
            .map_err(|e| format!("canonicalize {path:?}: {e}"))?;
        if stack.contains(&path) {
            return Err(format!("circular include at {path:?}"));
        }
        let text = fs::read_to_string(&path).map_err(|e| format!("read {path:?}: {e}"))?;
        stack.push(path.clone());

        let parent = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        let mut out = String::new();
        for line in text.lines() {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("#include") {
                if let Some(inc) = rest
                    .trim()
                    .strip_prefix('"')
                    .and_then(|s| s.strip_suffix('"'))
                {
                    let mut candidates = vec![parent.join(inc)];
                    candidates.extend(include_dirs.iter().map(|d| d.join(inc)));
                    let found = candidates
                        .into_iter()
                        .find(|p| p.exists())
                        .ok_or_else(|| format!("include not found: {inc}"))?;
                    let chunk = expand_one(&found, include_dirs, stack)?;
                    out.push_str(&chunk);
                    out.push('\n');
                } else {
                    out.push_str(line);
                    out.push('\n');
                }
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }

        stack.pop();
        Ok(out)
    }

    // Write src to a temp file to reuse the same walker
    let tmp = base_dir.join("__temp_runtime_root__.metal");
    fs::create_dir_all(base_dir).ok();
    fs::write(&tmp, src).map_err(|e| format!("write tmp: {e}"))?;
    let mut stack = Vec::new();
    let result = expand_one(&tmp, include_dirs, &mut stack);
    let _ = fs::remove_file(&tmp);
    result
}
