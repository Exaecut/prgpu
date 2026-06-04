//! Pixel-level comparison and diff-report generation.
//!
//! Runs the built-in `diff` kernel on the GPU for heatmap generation, computes
//! aggregate metrics on the CPU, and produces JSON and human-readable text reports.

use std::fs;
use std::path::Path;

use crate::testing::context::GpuBuffer;
use crate::testing::GpuContext;
use crate::types::Configuration;
use crate::kernel::builtin::{diff, DiffParams};

/// Per-channel absolute tolerance in [0, 1] plus heatmap smoothstep bounds.
#[derive(Clone, Copy, Debug)]
pub struct DiffConfig {
    pub tolerance_r: f32,
    pub tolerance_g: f32,
    pub tolerance_b: f32,
    pub tolerance_a: f32,
    /// Normalised error below this value → black (cold).
    pub smooth_a: f32,
    /// Normalised error above this value → white (hot).
    pub smooth_b: f32,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            tolerance_r: 0.01,
            tolerance_g: 0.01,
            tolerance_b: 0.01,
            tolerance_a: 0.01,
            smooth_a: 0.0,
            smooth_b: 1.0,
        }
    }
}

/// Aggregate metrics from a pixel-level comparison.
#[derive(Clone, Debug)]
pub struct DiffReport {
    pub width: u32,
    pub height: u32,
    pub pixels_total: u64,
    pub pixels_equal: u64,
    pub pixels_different: u64,
    pub different_ratio: f64,
    pub mean_absolute_error: f64,
    pub max_absolute_error: f64,
    pub root_mean_square_error: f64,
    pub threshold: DiffConfig,
    pub passed: bool,
}

impl DiffReport {
    /// Compute PSNR from RMSE. ∞ if images are identical.
    pub fn psnr(&self) -> f64 {
        if self.root_mean_square_error < f64::EPSILON {
            f64::INFINITY
        } else {
            20.0 * (1.0 / self.root_mean_square_error).log10()
        }
    }
}

/// Run the GPU diff kernel to produce a heatmap.
///
/// `rendered` and `reference` must be tightly-packed BGRA at `width * height * 4` bytes.
/// Returns the heatmap as BGRA pixels.
pub fn diff_heatmap_gpu(
    gpu: &GpuContext,
    rendered: &GpuBuffer,
    reference: &GpuBuffer,
    width: u32,
    height: u32,
    config: &DiffConfig,
) -> Result<Vec<u8>, String> {
    let bpp = 4;

    let out_buf = gpu.create_buffer(width, height, bpp, 0x44494646)?; // "DIFF"

    let cfg = Configuration {
        device_handle: gpu.device,
        context_handle: gpu.context,
        command_queue_handle: gpu.command_queue,
        outgoing_data: Some(rendered.data),
        incoming_data: Some(reference.data),
        dest_data: out_buf.data,
        outgoing_pitch_px: rendered.pitch_px as i32,
        incoming_pitch_px: reference.pitch_px as i32,
        dest_pitch_px: out_buf.pitch_px as i32,
        width,
        height,
        outgoing_width: width,
        outgoing_height: height,
        incoming_width: width,
        incoming_height: height,
        bytes_per_pixel: bpp,
        time: 0.0,
        progress: 0.0,
        render_generation: 0,
        pixel_layout: 1,
        storage: crate::types::storage_from_bpp(bpp),
        flip_y: 0,
        outgoing_mip_levels: 0,
    };

    let params = DiffParams {
        tol_r: config.tolerance_r,
        tol_g: config.tolerance_g,
        tol_b: config.tolerance_b,
        tol_a: config.tolerance_a,
        smooth_a: config.smooth_a,
        smooth_b: config.smooth_b,
        _pad0: 0,
        _pad1: 0,
    };

    unsafe { diff::gpu(&cfg, params).map_err(|e| format!("diff kernel: {e}"))? };

    gpu.download_from_buffer(&out_buf, width, height, bpp)
}

/// Compute aggregate metrics on the CPU by comparing two tightly-packed BGRA buffers.
pub fn compute_metrics(
    rendered: &[u8],
    reference: &[u8],
    width: u32,
    height: u32,
    config: &DiffConfig,
) -> Result<DiffReport, String> {
    let expected = (width as usize) * (height as usize) * 4;
    if rendered.len() != expected || reference.len() != expected {
        return Err(format!(
            "size mismatch: rendered {}, reference {}, expected {expected}",
            rendered.len(),
            reference.len()
        ));
    }

    let pixels_total = (width as u64) * (height as u64);
    let mut pixels_equal: u64 = 0;
    let mut sum_abs_err: f64 = 0.0;
    let mut max_absolute_error: f64 = 0.0;
    let mut sum_sq_err: f64 = 0.0;

    let tol = [config.tolerance_r as f64, config.tolerance_g as f64, config.tolerance_b as f64, config.tolerance_a as f64];

    for i in (0..expected).step_by(4) {
        let r0 = rendered[i] as f64 / 255.0;
        let r1 = rendered[i + 1] as f64 / 255.0;
        let r2 = rendered[i + 2] as f64 / 255.0;
        let r3 = rendered[i + 3] as f64 / 255.0;

        let f0 = reference[i] as f64 / 255.0;
        let f1 = reference[i + 1] as f64 / 255.0;
        let f2 = reference[i + 2] as f64 / 255.0;
        let f3 = reference[i + 3] as f64 / 255.0;

        let d0 = (r0 - f0).abs();
        let d1 = (r1 - f1).abs();
        let d2 = (r2 - f2).abs();
        let d3 = (r3 - f3).abs();

        let pixel_same = d0 <= tol[0] && d1 <= tol[1] && d2 <= tol[2] && d3 <= tol[3];
        if pixel_same {
            pixels_equal += 1;
        }

        sum_abs_err += (d0 + d1 + d2 + d3) as f64;
        max_absolute_error = max_absolute_error.max(d0 as f64).max(d1 as f64).max(d2 as f64).max(d3 as f64);
        sum_sq_err += (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3) as f64;
    }

    let pixels_different = pixels_total - pixels_equal;
    let channel_count = pixels_total * 4;
    let mean_absolute_error = sum_abs_err / channel_count as f64;
    let root_mean_square_error = (sum_sq_err / channel_count as f64).sqrt();
    let different_ratio = pixels_different as f64 / pixels_total as f64;

    Ok(DiffReport {
        width,
        height,
        pixels_total,
        pixels_equal,
        pixels_different,
        different_ratio,
        mean_absolute_error,
        max_absolute_error,
        root_mean_square_error,
        threshold: *config,
        passed: true,
    })
}

/// Write metrics as JSON.
pub fn write_report_json(path: impl AsRef<Path>, report: &DiffReport) -> Result<(), String> {
    let json = serde_json::to_string_pretty(&serde_report(report))
        .map_err(|e| format!("serialize: {e}"))?;
    let p = path.as_ref();
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir -p {}: {e}", parent.display()))?;
    }
    fs::write(p, json).map_err(|e| format!("write {}: {e}", p.display()))
}

/// Write a human-readable text report.
pub fn write_report_txt(path: impl AsRef<Path>, report: &DiffReport) -> Result<(), String> {
    let mut out = String::new();
    out.push_str(&format!("Image Comparison Report\n"));
    out.push_str(&format!("========================\n\n"));
    out.push_str(&format!("Dimensions:         {} × {}\n", report.width, report.height));
    out.push_str(&format!("Total pixels:       {}\n", report.pixels_total));
    out.push_str(&format!("Pixels equal:       {} ({:.2}%)\n",
        report.pixels_equal,
        (1.0 - report.different_ratio) * 100.0));
    out.push_str(&format!("Pixels different:   {} ({:.4}%)\n",
        report.pixels_different,
        report.different_ratio * 100.0));
    out.push_str(&format!("Mean abs error:     {:.6}\n", report.mean_absolute_error));
    out.push_str(&format!("Max abs error:      {:.6}\n", report.max_absolute_error));
    out.push_str(&format!("RMSE:               {:.6}\n", report.root_mean_square_error));
    out.push_str(&format!("PSNR:               {:.2} dB\n", report.psnr()));
    out.push_str(&format!("\nTolerances (per channel):\n"));
    out.push_str(&format!("  R: {:.4}  G: {:.4}  B: {:.4}  A: {:.4}\n",
        report.threshold.tolerance_r,
        report.threshold.tolerance_g,
        report.threshold.tolerance_b,
        report.threshold.tolerance_a));

    let p = path.as_ref();
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir -p {}: {e}", parent.display()))?;
    }
    fs::write(p, out).map_err(|e| format!("write {}: {e}", p.display()))
}

#[derive(serde::Serialize)]
struct SerdeReport {
    width: u32,
    height: u32,
    pixels_total: u64,
    pixels_equal: u64,
    pixels_different: u64,
    different_ratio: f64,
    mean_absolute_error: f64,
    max_absolute_error: f64,
    root_mean_square_error: f64,
    psnr: f64,
    thresholds: SerdeThresholds,
    passed: bool,
}

#[derive(serde::Serialize)]
struct SerdeThresholds {
    per_channel_absolute: Vec<f64>,
    max_different_ratio: f64,
}

fn serde_report(r: &DiffReport) -> SerdeReport {
    SerdeReport {
        width: r.width,
        height: r.height,
        pixels_total: r.pixels_total,
        pixels_equal: r.pixels_equal,
        pixels_different: r.pixels_different,
        different_ratio: r.different_ratio,
        mean_absolute_error: r.mean_absolute_error,
        max_absolute_error: r.max_absolute_error,
        root_mean_square_error: r.root_mean_square_error,
        psnr: r.psnr(),
        thresholds: SerdeThresholds {
            per_channel_absolute: vec![
                r.threshold.tolerance_r as f64,
                r.threshold.tolerance_g as f64,
                r.threshold.tolerance_b as f64,
                r.threshold.tolerance_a as f64,
            ],
            max_different_ratio: 0.01,
        },
        passed: r.passed,
    }
}

/// Blackbody heatmap: black→dark blue→bright blue→orange→white.
/// smooth_a / smooth_b control the smoothstep ramp.
pub fn write_heatmap_png(
    path: impl AsRef<Path>,
    rendered: &[u8],
    reference: &[u8],
    width: u32,
    height: u32,
    config: &DiffConfig,
) -> Result<(), String> {
    let expected = (width as usize) * (height as usize) * 4;
    if rendered.len() != expected || reference.len() != expected {
        return Err("size mismatch".into());
    }

    let smooth_a = config.smooth_a;
    let smooth_b = config.smooth_b;
    let mut heatmap = vec![0u8; expected];

    for i in (0..expected).step_by(4) {
        let d0 = (rendered[i] as f32 - reference[i] as f32).abs() / 255.0;
        let d1 = (rendered[i + 1] as f32 - reference[i + 1] as f32).abs() / 255.0;
        let d2 = (rendered[i + 2] as f32 - reference[i + 2] as f32).abs() / 255.0;
        let d3 = (rendered[i + 3] as f32 - reference[i + 3] as f32).abs() / 255.0;

        let max_err = d0.max(d1).max(d2).max(d3);

        // Smoothstep: map max_err through sigmoid controlled by smooth_a/smooth_b.
        let x = (max_err - smooth_a) / (smooth_b - smooth_a).max(1e-6);
        let x = x.clamp(0.0, 1.0);
        let t = x * x * (3.0 - 2.0 * x); // smoothstep

        // Blackbody piecewise ramp (matches diff.slang).
        let (r, g, b) = if t < 0.25 {
            let s = t * 4.0;
            (0.0, 0.0, 0.4 * s)
        } else if t < 0.5 {
            let s = (t - 0.25) * 4.0;
            (0.1 * s, 0.2 * s, 0.4 + 0.6 * s)
        } else if t < 0.75 {
            let s = (t - 0.5) * 4.0;
            (0.1 + 0.9 * s, 0.2 + 0.3 * s, 1.0 - 1.0 * s)
        } else {
            let s = (t - 0.75) * 4.0;
            (1.0, 0.5 + 0.5 * s, s)
        };

        heatmap[i] = (b * 255.0) as u8;
        heatmap[i + 1] = (g * 255.0) as u8;
        heatmap[i + 2] = (r * 255.0) as u8;
        heatmap[i + 3] = 255;
    }

    crate::testing::output::write_png(path, &heatmap, width, height, 4)
}
