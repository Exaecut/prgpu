//! Cross-host AE PF effect contract.
//!
//! `CrossHostEffect<P>` factors the parts of an AE PF plugin that are
//! identical between AE-standalone and AE-in-Premiere into a single trait,
//! so plugin authors only describe the effect once.
//!
//! Scope: the AE PF dispatcher (FrameSetup, Render, SmartPreRender,
//! SmartRender, GpuDeviceSetup, SmartRenderGpu) on AE. On Premiere, the
//! same selectors are dispatched but real GPU rendering is expected to go
//! through a separate `pr::GpuFilter` / `xGPUFilterEntry`. The trait is
//! intentionally unaware of the Pr GPU path.
//!
//! Usage in an effect's `handle_command`:
//!
//! ```ignore
//! impl CrossHostEffect<Params> for Plugin { /* ... */ }
//!
//! match command {
//!     Command::GlobalSetup => {
//!         self.plugin_id = effect::handle_global_setup::<Plugin, Params>(&in_data)?;
//!     }
//!     Command::FrameSetup { in_layer, .. } => {
//!         effect::handle_frame_setup(self, &in_data, &in_layer, &mut out_data, params)?;
//!     }
//!     Command::FrameSetdown => { in_data.destroy_frame_data::<FrameData>(); }
//!     Command::Render { mut in_layer, mut out_layer } => {
//!         effect::handle_legacy_render(self, &in_data, &mut in_layer, &mut out_layer)?;
//!     }
//!     Command::SmartPreRender { extra } => {
//!         effect::handle_smart_pre_render(self, &in_data, extra, params)?;
//!     }
//!     Command::SmartRender { extra } => {
//!         effect::handle_smart_render(self, &in_data, extra, false)?;
//!     }
//!     Command::SmartRenderGpu { extra } => {
//!         effect::handle_smart_render(self, &in_data, extra, true)?;
//!     }
//!     Command::GpuDeviceSetup { extra } => {
//!         effect::handle_gpu_device_setup::<Plugin, Params>(extra, &mut out_data)?;
//!     }
//!     _ => {}
//! }
//! ```

use after_effects::{
	self as ae, aegp,
	pf::{Layer, OutFlags2, Parameters, PreRenderExtra, SmartRenderExtra},
	Error, GpuFramework, InData, OutData, Point, Rect,
};
use std::ffi::c_void;
use std::fmt::Debug;
use std::hash::Hash;

/// Per-side pixel inflation applied uniformly to the input layer to compute
/// the rendered output rect.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExpansionExtent {
	pub left: i32,
	pub top: i32,
	pub right: i32,
	pub bottom: i32,
}

impl ExpansionExtent {
	pub fn symmetric(px: i32) -> Self {
		Self { left: px, top: px, right: px, bottom: px }
	}

	pub fn is_zero(&self) -> bool {
		self.left == 0 && self.top == 0 && self.right == 0 && self.bottom == 0
	}

	pub fn total_width(&self) -> i32 {
		self.left + self.right
	}

	pub fn total_height(&self) -> i32 {
		self.top + self.bottom
	}

	pub fn inflate_rect(&self, r: Rect) -> Rect {
		let mut out = Rect::empty();
		out.left = r.left - self.left;
		out.top = r.top - self.top;
		out.right = r.right + self.right;
		out.bottom = r.bottom + self.bottom;
		out
	}
}

/// Declares which GPU frameworks the effect's `render_unified` is prepared
/// to handle on the AE PF GPU path. Effects on macOS Apple Silicon should
/// return `metal: true`; everything else stays disabled.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuSupport {
	pub metal: bool,
	pub cuda: bool,
	pub opencl: bool,
	pub directx: bool,
}

impl GpuSupport {
	pub fn from_cfg() -> Self {
		Self {
			metal: cfg!(gpu_backend = "metal"),
			cuda: cfg!(gpu_backend = "cuda"),
			opencl: cfg!(gpu_backend = "opencl"),
			directx: cfg!(gpu_backend = "directx"),
		}
	}

	pub fn supports(&self, framework: GpuFramework) -> bool {
		match framework {
			GpuFramework::Metal => self.metal,
			GpuFramework::Cuda => self.cuda,
			GpuFramework::OpenCl => self.opencl,
			GpuFramework::DirectX => self.directx,
			GpuFramework::None => false,
		}
	}
}

/// Resolved AE PF GPU device handles for one render call.
#[derive(Clone, Copy)]
pub struct GpuDeviceContext {
	pub device_handle: *mut c_void,
	pub command_queue_handle: *mut c_void,
	pub context_handle: Option<*mut c_void>,
	pub device_index: usize,
	pub framework: GpuFramework,
}

/// Arguments handed to `render_unified` for one frame.
pub struct RenderArgs<'a, FD: 'static> {
	pub in_data: &'a InData,
	pub in_layer: &'a mut Layer,
	pub out_layer: &'a mut Layer,
	pub frame_data: &'a FD,
	pub is_gpu: bool,
	pub gpu_device: Option<GpuDeviceContext>,
}

/// AE PF cross-host effect contract.
pub trait CrossHostEffect<P>
where
	P: Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	type FrameData: Copy + 'static;

	fn gpu_support() -> GpuSupport {
		GpuSupport::from_cfg()
	}

	fn aegp_display_name() -> &'static str;

	fn premiere_pixel_formats() -> &'static [ae::pr::PixelFormat] {
		&[ae::pr::PixelFormat::Bgra4444_32f, ae::pr::PixelFormat::Bgra4444_8u]
	}

	fn render_ready(&self, _params: &Parameters<P>, _in_data: &InData) -> bool {
		true
	}

	fn compute_expansion(&self, params: &Parameters<P>, in_data: &InData, layer_w: u32, layer_h: u32) -> Result<ExpansionExtent, Error>;

	fn resolve_frame_data(&self, params: &Parameters<P>, in_data: &InData, out_w: u32, out_h: u32) -> Result<Self::FrameData, Error>;

	fn render_unified(&mut self, args: RenderArgs<Self::FrameData>) -> Result<(), Error>;
}

pub fn handle_global_setup<E, P>(in_data: &InData) -> Result<aegp::PluginId, Error>
where
	E: CrossHostEffect<P>,
	P: Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	if in_data.is_premiere() {
		let suite = ae::pf::suites::PixelFormat::new()?;
		suite.clear_supported_pixel_formats(in_data.effect_ref())?;
		for fmt in E::premiere_pixel_formats() {
			suite.add_supported_pixel_format(in_data.effect_ref(), *fmt)?;
		}
		ae::pf::suites::Utility::new()?.effect_wants_checked_out_frames_to_match_render_pixel_format(in_data.effect_ref())?;
	}

	let plugin_id = match aegp::suites::Utility::new() {
		Ok(suite) => suite.register_with_aegp(E::aegp_display_name())?,
		Err(_) => aegp::PluginId::default(),
	};
	Ok(plugin_id)
}

pub fn handle_gpu_device_setup<E, P>(extra: ae::pf::GpuDeviceSetupExtra, out_data: &mut OutData) -> Result<(), Error>
where
	E: CrossHostEffect<P>,
	P: Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	let what = extra.what_gpu();
	if E::gpu_support().supports(what) {
		out_data.set_out_flag2(OutFlags2::SupportsGpuRenderF32, true);
	}
	Ok(())
}

pub fn handle_frame_setup<E, P>(effect: &mut E, in_data: &InData, in_layer: &Layer, out_data: &mut OutData, params: &Parameters<P>) -> Result<(), Error>
where
	E: CrossHostEffect<P>,
	P: Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	if !effect.render_ready(params, in_data) {
		return Ok(());
	}

	let layer_w = in_layer.width() as u32;
	let layer_h = in_layer.height() as u32;
	let ext = effect.compute_expansion(params, in_data, layer_w, layer_h)?;

	let out_w = (layer_w as i32 + ext.total_width()).max(1) as u32;
	let out_h = (layer_h as i32 + ext.total_height()).max(1) as u32;

	if !ext.is_zero() {
		out_data.set_width(out_w);
		out_data.set_height(out_h);
		out_data.set_origin(Point { h: ext.left, v: ext.top });
	}

	let frame_data = effect.resolve_frame_data(params, in_data, out_w, out_h)?;
	out_data.set_frame_data::<E::FrameData>(frame_data);
	Ok(())
}

pub fn handle_smart_pre_render<E, P>(effect: &mut E, in_data: &InData, mut extra: PreRenderExtra, params: &Parameters<P>) -> Result<(), Error>
where
	E: CrossHostEffect<P>,
	P: Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	let what_gpu = extra.what_gpu();
	if !effect.render_ready(params, in_data) {
		return Ok(());
	}

	let req = extra.output_request();
	let req_rect = Rect::from(req.rect);
	let layer_w = req_rect.width().max(1) as u32;
	let layer_h = req_rect.height().max(1) as u32;
	let ext = effect.compute_expansion(params, in_data, layer_w, layer_h)?;

	let mut src_request = req;
	let inflated = ext.inflate_rect(req_rect);
	src_request.rect = inflated.into();

	let in_result = extra
		.callbacks()
		.checkout_layer(0, 0, &src_request, in_data.current_time(), in_data.time_step(), in_data.time_scale())?;

	let layer_max = Rect::from(in_result.max_result_rect);
	let layer_result = Rect::from(in_result.result_rect);
	let max_rect = ext.inflate_rect(layer_max);
	let result_rect = if ext.is_zero() { layer_result } else { ext.inflate_rect(layer_result) };

	let _ = extra.union_result_rect(result_rect);
	let _ = extra.union_max_result_rect(max_rect);

	if !ext.is_zero() {
		extra.set_returns_extra_pixels(true);
	}

	if E::gpu_support().supports(what_gpu) {
		extra.set_gpu_render_possible(true);
	}

	let out_w = result_rect.width().max(1) as u32;
	let out_h = result_rect.height().max(1) as u32;
	let frame_data = effect.resolve_frame_data(params, in_data, out_w, out_h)?;
	extra.set_pre_render_data::<E::FrameData>(frame_data);
	Ok(())
}

pub fn handle_smart_render<E, P>(effect: &mut E, in_data: &InData, extra: SmartRenderExtra, is_gpu: bool) -> Result<(), Error>
where
	E: CrossHostEffect<P>,
	P: Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	let cb = extra.callbacks();
	let Some(input_world) = cb.checkout_layer_pixels(0)? else {
		return Ok(());
	};

	let mut input_world = input_world;
	let render_result = (|| -> Result<(), Error> {
		let Some(mut output_world) = cb.checkout_output()? else {
			return Ok(());
		};

		let frame_data = match extra.pre_render_data::<E::FrameData>() {
			Some(fd) => fd,
			None => in_data.frame_data::<E::FrameData>().ok_or(Error::Generic)?,
		};

		let gpu_device = if is_gpu {
			let gpu_suite = ae::pf::suites::GPUDevice::new()?;
			let device_index = extra.device_index();
			let info = gpu_suite.device_info(in_data.effect_ref(), device_index)?;
			Some(GpuDeviceContext {
				device_handle: info.devicePV as *mut c_void,
				command_queue_handle: info.command_queuePV as *mut c_void,
				context_handle: if info.contextPV.is_null() { None } else { Some(info.contextPV as *mut c_void) },
				device_index,
				framework: extra.what_gpu(),
			})
		} else {
			None
		};

		effect.render_unified(RenderArgs {
			in_data,
			in_layer: &mut input_world,
			out_layer: &mut output_world,
			frame_data,
			is_gpu,
			gpu_device,
		})
	})();

	cb.checkin_layer_pixels(0)?;
	render_result
}

pub fn handle_legacy_render<E, P>(effect: &mut E, in_data: &InData, in_layer: &mut Layer, out_layer: &mut Layer) -> Result<(), Error>
where
	E: CrossHostEffect<P>,
	P: Eq + Hash + Copy + Debug + Into<usize> + 'static,
{
	let frame_data = in_data.frame_data::<E::FrameData>().ok_or(Error::Generic)?;
	effect.render_unified(RenderArgs {
		in_data,
		in_layer,
		out_layer,
		frame_data,
		is_gpu: false,
		gpu_device: None,
	})
}
