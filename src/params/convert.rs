//! Snapshot extraction helpers shared by `ParamsSpec::snapshot_cpu` /
//! `snapshot_gpu` codegen. Each resolves one host quirk into a [`ParamValue`]
//! so the generated code is a flat list of calls.
//!
//! Built on the legacy [`CpuParams`] / [`get_param`] extractors as a phase-3
//! bridge; the direct host reads move here when `FrameDataContext` is removed.

use after_effects::Parameters;
use premiere as pr;

use super::legacy::{CpuParams, SetupParams, get_param};
use super::value::{Color, ParamValue, Point2};
use crate::types::Pixel;

pub fn cpu_float<P: SetupParams>(p: &Parameters<P>, id: P) -> Result<ParamValue, after_effects::Error> {
	Ok(ParamValue::Float(p.float(id)?))
}

pub fn cpu_angle<P: SetupParams>(p: &Parameters<P>, id: P) -> Result<ParamValue, after_effects::Error> {
	Ok(ParamValue::Float(p.angle(id)?))
}

pub fn cpu_checkbox<P: SetupParams>(p: &Parameters<P>, id: P) -> Result<ParamValue, after_effects::Error> {
	Ok(ParamValue::Bool(p.checkbox(id)?))
}

pub fn cpu_color<P: SetupParams>(p: &Parameters<P>, id: P) -> Result<ParamValue, after_effects::Error> {
	let px = p.color(id)?;
	Ok(ParamValue::Color(Color::from_u8(px.red, px.green, px.blue, px.alpha)))
}

/// AE PF popups are 1-based at the SDK; subtract 1 to match the 0-based index
/// every other host (and the kernel) sees.
pub fn cpu_popup<P: SetupParams>(p: &Parameters<P>, id: P) -> Result<ParamValue, after_effects::Error> {
	Ok(ParamValue::Index((p.popup(id)? as u32).saturating_sub(1)))
}

/// AE/Premiere CPU `point` is in pixel space; normalize to 0–1 against the
/// layer dimensions.
pub fn cpu_point<P: SetupParams>(p: &Parameters<P>, id: P, layer_w: u32, layer_h: u32) -> Result<ParamValue, after_effects::Error> {
	let (x, y) = p.point(id)?;
	Ok(ParamValue::Point(Point2::new(x / layer_w.max(1) as f32, y / layer_h.max(1) as f32)))
}

pub fn gpu_float<P: SetupParams>(f: &pr::GpuFilterData, rp: &pr::RenderParams, id: P) -> ParamValue {
	ParamValue::Float(get_param::<f32, _>(f, id, rp))
}

pub fn gpu_checkbox<P: SetupParams>(f: &pr::GpuFilterData, rp: &pr::RenderParams, id: P) -> ParamValue {
	ParamValue::Bool(get_param::<bool, _>(f, id, rp))
}

pub fn gpu_color<P: SetupParams>(f: &pr::GpuFilterData, rp: &pr::RenderParams, id: P) -> ParamValue {
	let c: Pixel = get_param(f, id, rp);
	ParamValue::Color(Color::from_u8(c.red, c.green, c.blue, c.alpha))
}

/// Premiere GPU popups arrive already 0-based; clamp negatives defensively.
pub fn gpu_popup<P: SetupParams>(f: &pr::GpuFilterData, rp: &pr::RenderParams, id: P) -> ParamValue {
	ParamValue::Index(get_param::<i32, _>(f, id, rp).max(0) as u32)
}

/// Premiere GPU points are pre-normalized 0–1; pass through.
pub fn gpu_point<P: SetupParams>(f: &pr::GpuFilterData, rp: &pr::RenderParams, id: P) -> ParamValue {
	let (x, y): (f32, f32) = get_param(f, id, rp);
	ParamValue::Point(Point2::new(x, y))
}
