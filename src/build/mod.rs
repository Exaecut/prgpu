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

                // Ignore bruit inutile
                if line.contains("note #") {
                    continue;
                }

                // Détection début d’erreur
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

    // fallback brut
    for l in block {
        out.push_str(l);
        out.push('\n');
    }

    out
}

pub fn compile_shaders(shader_dir: &str) -> Result<(), DynError> {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    for entry in std::fs::read_dir(shader_dir).unwrap() {
        let path = entry.unwrap().path();

        if path.extension().and_then(|s| s.to_str()) == Some("shader") {
            let name = path.file_stem().unwrap().to_str().unwrap();

            let src = std::fs::read_to_string(&path).unwrap();

            let utils = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("shaders/utils")
                .canonicalize()
                .unwrap();

            let cuda_path = std::env::var("CUDA_HOME")
                .or_else(|_| std::env::var("CUDA_PATH"))
                .unwrap_or("/usr/local/cuda".into());
            let cuda_include = PathBuf::from(cuda_path)
                .join("include")
                .canonicalize()
                .unwrap();

            let opts: CompileOptions = CompileOptions {
                ftz: Some(true),
                prec_sqrt: Some(false),
                prec_div: Some(false),
                fmad: Some(true),
                use_fast_math: None,
                include_paths: vec![
                    utils.to_string_lossy().replace("\\\\?\\", ""),
                    cuda_include.to_string_lossy().replace("\\\\?\\", ""),
                ],
                arch: Some("compute_86"),
                options: vec![
                    "--std=c++14".into(),
                    "--extra-device-vectorization".into(),
                    "--device-as-default-execution-space".into(),
                ],
                ..Default::default()
            };

            let ptx = cudarc::nvrtc::compile_ptx_with_opts(&src, opts).map_err(|e| {
                let pretty = parse_nvrtc_error(&e);

                eprintln!("Compile failed [{name}]:\n{pretty}");
                println!("cargo:warning=Compile failed [{name}]:\n{pretty}");
                Box::new(e) as DynError
            })?;

            if let Err(write_err) = std::fs::write(
                PathBuf::from(&out_dir).join(format!("{}.ptx", name)),
                ptx.as_bytes().unwrap(),
            ) {
                eprintln!("Shader write failed [{name}]: {write_err}");
                println!("cargo:warning=Shader write failed [{name}]: {write_err}");
                return Err(Box::new(write_err));
            }
        }
    }

    Ok(())
}
