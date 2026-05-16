//! Renders a sequence of frames through the GPU pipeline and writes outputs.

use std::path::PathBuf;

use crate::testing::context::GpuBuffer;
use crate::testing::scene::Timeline;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionTarget {
    Gpu,
    #[allow(dead_code)]
    Cpu,
}

#[derive(Clone, Debug)]
pub enum OutputSpec {
    PngSequence {
        dir: PathBuf,
        /// Base filename ("frame" → frame_000000.png).
        basename: String,
    },
    SinglePng(PathBuf),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffPolicy {
    ReportOnly,
    FailAboveThreshold,
}

#[derive(Clone, Debug)]
pub struct ComparisonSpec {
    pub reference: PathBuf,
    /// 0-based frame index.
    pub frame_number: u32,
    /// Per-channel absolute tolerance [0, 1].
    pub tolerance: f32,
    pub policy: DiffPolicy,
}

#[derive(Clone, Debug)]
pub struct RenderResult {
    pub frame_count: u32,
    pub output_dir: PathBuf,
    pub width: u32,
    pub height: u32,
    pub execution: ExecutionTarget,
}

/// For multi-frame tests the callers iterate frame indices via `run()`.
pub struct RenderTest {
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    pub timeline: Timeline,
    pub output: OutputSpec,
    pub execution: ExecutionTarget,
    pub comparison: Option<ComparisonSpec>,
    /// Tightly-packed BGRA source data for every frame.
    pub input_data: Vec<u8>,
    frame_outputs: Vec<Vec<u8>>,
}

impl RenderTest {
    pub fn new(
        width: u32,
        height: u32,
        input_data: Vec<u8>,
        output: OutputSpec,
    ) -> Self {
        Self {
            width,
            height,
            bytes_per_pixel: 4,
            timeline: Timeline::default(),
            output,
            execution: ExecutionTarget::Gpu,
            comparison: None,
            input_data,
            frame_outputs: Vec::new(),
        }
    }

    pub fn with_timeline(mut self, timeline: Timeline) -> Self {
        self.timeline = timeline;
        self
    }

    pub fn with_comparison(mut self, comparison: ComparisonSpec) -> Self {
        self.comparison = Some(comparison);
        self
    }

    /// Default is 4 (BGRA8).
    pub fn with_bpp(mut self, bpp: u32) -> Self {
        self.bytes_per_pixel = bpp;
        self
    }

    /// Calls `render_frame_fn(input, output, frame_index, clip_time, config)`
    /// for each frame, downloads the result, and writes output PNG(s).
    pub fn run<F>(
        &mut self,
        gpu: &crate::testing::GpuContext,
        mut render_frame_fn: F,
    ) -> Result<RenderResult, String>
    where
        F: FnMut(&GpuBuffer, &GpuBuffer, u32, i64, &crate::types::Configuration) -> Result<(), String>,
    {
        use crate::testing::output::write_png;

        let frame_count = self.timeline.frame_count;
        let w = self.width;
        let h = self.height;
        let bpp = self.bytes_per_pixel;

        let expected = (w as u64) * (h as u64) * (bpp as u64);
        if self.input_data.len() as u64 != expected {
            return Err(format!(
                "input data length {} != expected {} ({}x{}x{})",
                self.input_data.len(),
                expected,
                w,
                h,
                bpp
            ));
        }

        let (in_buf, out_buf) = gpu.create_io_buffers(w, h, bpp)?;
        gpu.upload_to_buffer(&in_buf, &self.input_data, w, h, bpp)?;

        let config_base = gpu.build_config(&in_buf, &out_buf, w, h, bpp);

        self.frame_outputs.clear();

        for fi in 0..frame_count {
            let clip_time = self.timeline.clip_time(fi);
            let mut config = config_base.clone();
            config.time = clip_time as f32 / self.timeline.time_scale as f32;

            render_frame_fn(&in_buf, &out_buf, fi, clip_time, &config)?;

            let output = gpu.download_from_buffer(&out_buf, w, h, bpp)?;
            self.frame_outputs.push(output.clone());

            match &self.output {
                OutputSpec::SinglePng(path) => {
                    write_png(path, &output, w, h, bpp)?;
                }
                OutputSpec::PngSequence { dir, basename } => {
                    let filename = format!("{}_{:06}.png", basename, fi);
                    let filepath = dir.join(&filename);
                    write_png(&filepath, &output, w, h, bpp)?;
                }
            }
        }

        let output_dir = match &self.output {
            OutputSpec::SinglePng(p) => p.parent().map(|p| p.to_path_buf()).unwrap_or_default(),
            OutputSpec::PngSequence { dir, .. } => dir.clone(),
        };

        Ok(RenderResult {
            frame_count,
            output_dir,
            width: w,
            height: h,
            execution: self.execution,
        })
    }

    pub fn frame_outputs(&self) -> &[Vec<u8>] {
        &self.frame_outputs
    }
}
