use std::{
    fs,
    path::{Path, PathBuf},
};

#[macro_export]
macro_rules! include_shader {
    ($name:ident) => {{
        include_str!(concat!(
            env!("OUT_DIR"),
            "/",
            stringify!($name),
            "_flat.metal"
        ))
    }};

    ($name:literal) => {{ include_str!(concat!(env!("OUT_DIR"), "/", $name, "_flat.metal")) }};

    ($name:literal, metal) => {
        include_str!(concat!(env!("OUT_DIR"), "/", $name, "_flat.metal"))
    };

    ($name:literal, cuda) => {
        include_str!(concat!(env!("OUT_DIR"), "/", $name, "_flat.cu"))
    };
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
