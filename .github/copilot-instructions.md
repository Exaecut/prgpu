# Copilot Instructions for Exaecut Effects

This workspace contains a collection of GPU-accelerated video effects (plugins) for After Effects and Premiere Pro, built with Rust and WGPU/WGSL, with some components using a custom Metal/CUDA DSL.

## Architecture & Patterns

### Plugin Structure
Each effect is a standalone crate in the workspace (e.g., `chromaticaberration/`, `retrovhs/`) with a consistent internal structure:
- `src/lib.rs`: Entry point, implements `AdobePluginGlobal`. Handles lifecycle and command routing.
- `src/params.rs`: Defines the UI parameters using the `after_effects` crate.
- `src/wgpu_procs.rs`: WGPU boilerplate for GPU processing. Defines `KernelParams` (C-repr struct) passed to shaders.
- `shaders/*.cu`: The actual GPU kernel code.

### Data Flow
1. **Host (AE/Premiere)** calls the plugin.
2. **Rust** parses parameters in `params.rs`.
3. **WGPU** (in `wgpu_procs.rs`) uploads parameters and textures to the GPU.
4. **WGSL** shaders process the frames.
5. **Rust** downloads the result back to the host buffer.

### Shader Hot-Reloading
The project supports shader hot-reloading in debug builds.
- Enabled via `EX_SHADER_HOTRELOAD=1` environment variable.
- In `lib.rs`, the `Plugin` struct checks `cfg!(shader_hotreload)` to decide whether to load shaders from disk or via `include_str!`.

### Cross-Platform Shaders (`shaders/utils/`)
The `shaders/utils/` directory contains a "Metal-first" DSL (`dsl.h`) that allows writing kernels that compile for both Metal and CUDA.
- **CUDA**: Define `SUPPORT_CUDA`.
- **Metal**: Default behavior.
- Use `float2/3/4` and `thread_position_in_grid` even in CUDA contexts via the shims.

## Critical Workflows

### Building
Use the provided shell scripts for building. Do not use raw `cargo build` unless you know what you are doing, as the scripts handle plugin packaging.
- **Build All (Debug)**: `./build_all.sh debug`
- **Build All (Release)**: `./build_all.sh release`
- **Build Specific Plugin**: `./build_all.sh debug from <path_to_file_in_plugin>`
- **Hot-Reloading**: Set `EX_SHADER_HOTRELOAD=1` before building/running.

### Adding a New Effect
Use the scaffolding script:
```bash
./new_effect.sh <effect_name>
```

## Coding Standards
- **C-Repr**: Always use `#[repr(C)]` for structs passed to shaders (e.g., `KernelParams`).
- **Error Handling**: Use the `after_effects::Error` type for plugin commands.
- **Licensing**: Most plugins integrate with `themis` for license validation in `params_setup`.
- **WGSL**: Prefer WGSL for new effects unless cross-platform Metal/CUDA parity is specifically required via the `shader-utils` DSL.
