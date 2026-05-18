//! Built-in mip-chain downsampler.
//!
//! Effects opt in via `Configuration::outgoing_mip_levels`; full host walkthrough
//! in [`prgpu/docs/mip_chain.md`](../../../docs/mip_chain.md), shader-side reference
//! in [`vekl/docs/reference/texture/view.md`](../../../../vekl/docs/reference/texture/view.md).
//!
//! `prgpu/shaders/mip_downsample.slang` is compiled by prgpu's `build.rs`; the
//! `declare_kernel!` below wires `include_shader!` + CPU FFI + GPU dispatch.
//!
//! Host recipe (identical on CPU/Metal/CUDA aside from the level-0 copy):
//!
//! ```ignore
//! config.outgoing_mip_levels = 4;
//! let _mip = unsafe { prepare_mip_source(&mut config, MY_TAG)? };
//! unsafe { generate_mips(&config)?; }
//! unsafe { effect_kernel(&config, params)?; }
//! ```
//!
//! `generate_mips` routes on `config.context_handle` — `Some(_)` = GPU dispatch,
//! `None` = `render_cpu_direct`. `device_handle` can't be the sentinel: CUDA
//! Premiere stores a `CUdevice` *ordinal* there, and ordinal `0` (single-GPU
//! host) is indistinguishable from a null pointer. Allocators
//! (`{cpu,metal,cuda}::buffer::get_or_create_with_mips`) key on
//! `(device, w, h, bpp, mip_levels, tag)` so mip and plain buffers at the same
//! dims don't share a slot.

use crate::declare_kernel;
use crate::types::{Configuration, ImageBuffer, MAX_MIP};

/// Uniforms for the mip-downsample kernel.
///
/// `_pad*` aligns the slang ConstantBuffer to a 16-byte vec4 boundary, matching
/// the `uint _pad0; uint _pad1; uint _pad2;` fields in `prgpu/shaders/mip_downsample.slang`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MipDownsampleParams {
	pub src_lod: u32,
	pub _pad0: u32,
	pub _pad1: u32,
	pub _pad2: u32,
}

declare_kernel!(mip_downsample, MipDownsampleParams);

/// Fill levels `1..N` from level 0 in `config.outgoing_data`.
///
/// `outgoing_mip_levels <= 1` short-circuits to a no-op so callers can call this
/// unconditionally before every kernel dispatch. Caller must allocate via
/// `cpu::buffer::get_or_create_with_mips` or the Metal/CUDA equivalents and
/// populate level 0 (see `prepare_mip_source` for the convenience helper).
///
/// # Safety
/// `config.outgoing_data` must hold at least `mip_buffer_size_bytes` bytes laid
/// out per `fill_mip_desc`.
pub unsafe fn generate_mips(config: &Configuration) -> Result<(), &'static str> {
	let levels = config.outgoing_mip_levels.max(1).min(MAX_MIP);
	if levels <= 1 {
		return Ok(());
	}

	let Some(mip_ptr) = config.outgoing_data else {
		return Err("generate_mips: outgoing_data is None");
	};

	for lod in 0..(levels - 1) {
		let dst_w = (config.outgoing_width >> (lod + 1)).max(1);
		let dst_h = (config.outgoing_height >> (lod + 1)).max(1);

		// Downsample reads from and writes to the same buffer; the shader only reads
		// `dst` (slot 2), but `outgoing` and `incoming` are bound to satisfy the
		// 5-buffer Metal/CUDA calling convention.
		let mut pass_cfg = *config;
		pass_cfg.width = dst_w;
		pass_cfg.height = dst_h;
		pass_cfg.outgoing_data = Some(mip_ptr);
		pass_cfg.incoming_data = Some(mip_ptr);
		pass_cfg.dest_data = mip_ptr;
		pass_cfg.dest_pitch_px = config.outgoing_pitch_px;

		let params = MipDownsampleParams {
			src_lod: lod,
			_pad0: 0,
			_pad1: 0,
			_pad2: 0,
		};

		if pass_cfg.context_handle.is_some() {
			unsafe { mip_downsample(&pass_cfg, params)? };
		} else {
			// CPU dispatch — direct rayon tile loop, no AE plumbing.
			unsafe {
				crate::cpu::render::render_cpu_direct(
					"mip_downsample",
					&pass_cfg,
					MIP_DOWNSAMPLE_CPU_DISPATCH_TILE,
					&params,
				);
			}
		}
	}

	Ok(())
}

/// Allocate a mip-capable buffer, copy `config.outgoing_data` into level 0, and
/// redirect `config.outgoing_data` to it. Returns the `ImageBuffer` for the
/// caller to keep alive across the frame (it's owned by the prgpu cache).
///
/// Routes on `config.context_handle`: `None` → CPU (`cpu::buffer` +
/// `copy_nonoverlapping`), `Some(_)` → Metal blit / CUDA `cuMemcpy*`.
/// (CUDA `device_handle` is a `CUdevice` ordinal, so `0` is a valid GPU
/// and can't be used as the CPU sentinel.)
///
/// `tag` participates in the cache key with `(w, h, bpp, mip_levels)` so distinct
/// effect instances don't stomp on each other's mip buffers (convention: upper
/// half = effect namespace, lower half = role).
///
/// On return, `outgoing_data` points at the mip buffer and `outgoing_pitch_px =
/// outgoing_width` (tight). Call `generate_mips` next.
///
/// # Safety
/// - `config.outgoing_data` must hold at least
///   `outgoing_pitch_px * outgoing_height * bytes_per_pixel` bytes (GPU buffer or host memory).
/// - On Metal, `config.command_queue_handle` must match `config.device_handle`.
/// - No other CPU/GPU work may touch the returned buffer until the effect kernel completes.
pub unsafe fn prepare_mip_source(config: &mut Configuration, tag: u32) -> Result<ImageBuffer, &'static str> {
	let levels = config.outgoing_mip_levels.max(1).min(MAX_MIP);
	if levels <= 1 {
		return Err("prepare_mip_source: outgoing_mip_levels must be >= 2");
	}

	let w = config.outgoing_width;
	let h = config.outgoing_height;
	let bpp = config.bytes_per_pixel;
	let src_ptr = config.outgoing_data.ok_or("prepare_mip_source: outgoing_data is None")?;
	let src_pitch_bytes = (config.outgoing_pitch_px as u32).saturating_mul(bpp);
	let dst_pitch_bytes = w.saturating_mul(bpp); // mip buffer level 0 is tight-packed

	if config.context_handle.is_none() {
		let buf = crate::cpu::buffer::get_or_create_with_mips(w, h, bpp, levels, tag);
		if buf.buf.raw.is_null() {
			return Err("prepare_mip_source: CPU allocator returned null");
		}
		unsafe {
			if src_pitch_bytes == dst_pitch_bytes {
				std::ptr::copy_nonoverlapping(
					src_ptr as *const u8,
					buf.buf.raw as *mut u8,
					(dst_pitch_bytes as usize) * (h as usize),
				);
			} else {
				for y in 0..(h as usize) {
					std::ptr::copy_nonoverlapping(
						(src_ptr as *const u8).add(y * src_pitch_bytes as usize),
						(buf.buf.raw as *mut u8).add(y * dst_pitch_bytes as usize),
						dst_pitch_bytes as usize,
					);
				}
			}
		}
		config.outgoing_data = Some(buf.buf.raw);
		config.outgoing_pitch_px = w as i32;
		return Ok(buf);
	}

	#[cfg(gpu_backend = "metal")]
	unsafe {
		use crate::DeviceHandleInit;
		let buf = crate::gpu::backends::metal::buffer::get_or_create_with_mips(
			DeviceHandleInit::FromPtr(config.device_handle),
			w,
			h,
			bpp,
			levels,
			tag,
		);
		if buf.buf.raw.is_null() {
			return Err("prepare_mip_source: Metal allocator returned null");
		}
		crate::gpu::backends::metal::buffer::copy_buffer(
			config,
			src_ptr,
			0,
			src_pitch_bytes,
			buf.buf.raw,
			0,
			dst_pitch_bytes,
			dst_pitch_bytes,
			h,
		)?;
		config.outgoing_data = Some(buf.buf.raw);
		config.outgoing_pitch_px = w as i32;
		return Ok(buf);
	}

	#[cfg(gpu_backend = "cuda")]
	unsafe {
		use crate::DeviceHandleInit;
		// CUDA `allocate` calls `cuCtxSetCurrent` on whatever pointer it gets, so it
		// needs the CUcontext (`context_handle`) — `device_handle` is a CUdevice
		// ordinal here. Routing above guarantees `context_handle.is_some()`.
		let ctx = config.context_handle.expect("CUDA path requires context_handle");
		let buf = crate::gpu::backends::cuda::buffer::get_or_create_with_mips(
			DeviceHandleInit::FromPtr(ctx),
			w,
			h,
			bpp,
			levels,
			tag,
		);
		if buf.buf.raw.is_null() {
			return Err("prepare_mip_source: CUDA allocator returned null");
		}
		crate::gpu::backends::cuda::buffer::copy_buffer(
			config,
			src_ptr,
			0,
			src_pitch_bytes,
			buf.buf.raw,
			0,
			dst_pitch_bytes,
			dst_pitch_bytes,
			h,
		)?;
		config.outgoing_data = Some(buf.buf.raw);
		config.outgoing_pitch_px = w as i32;
		return Ok(buf);
	}

	#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
	{
		Err("prepare_mip_source: no GPU backend enabled")
	}
}

/// Allocate a tight private GPU/CPU buffer, copy `config.outgoing_data` into it,
/// and redirect `config.outgoing_data`. Returns the `ImageBuffer` to keep alive.
///
/// Use when an effect's main pass would otherwise read from Premiere's outgoing
/// PPix while writing to its destination PPix — Premiere can hand the same buffer
/// as both source and destination, and a multi-tap kernel reading and writing the
/// same memory races on neighbour pixels (visible flicker / corruption at corners
/// of curved distortions).
///
/// Effects already chaining through prgpu LRU buffers get this protection for
/// free; only the all-intermediate-passes-disabled case needs the explicit copy.
///
/// `tag` participates in the cache key with `(w, h, bpp)` (convention as in
/// `prepare_mip_source`).
///
/// # Safety
/// Same preconditions as `prepare_mip_source`.
pub unsafe fn prepare_source_copy(config: &mut Configuration, tag: u32) -> Result<ImageBuffer, &'static str> {
	let w = config.outgoing_width;
	let h = config.outgoing_height;
	let bpp = config.bytes_per_pixel;
	let src_ptr = config.outgoing_data.ok_or("prepare_source_copy: outgoing_data is None")?;
	let src_pitch_bytes = (config.outgoing_pitch_px as u32).saturating_mul(bpp);
	let dst_pitch_bytes = w.saturating_mul(bpp); // private buffer is tight-packed

	if config.context_handle.is_none() {
		let buf = crate::cpu::buffer::get_or_create(w, h, bpp, tag);
		if buf.buf.raw.is_null() {
			return Err("prepare_source_copy: CPU allocator returned null");
		}
		unsafe {
			if src_pitch_bytes == dst_pitch_bytes {
				std::ptr::copy_nonoverlapping(
					src_ptr as *const u8,
					buf.buf.raw as *mut u8,
					(dst_pitch_bytes as usize) * (h as usize),
				);
			} else {
				for y in 0..(h as usize) {
					std::ptr::copy_nonoverlapping(
						(src_ptr as *const u8).add(y * src_pitch_bytes as usize),
						(buf.buf.raw as *mut u8).add(y * dst_pitch_bytes as usize),
						dst_pitch_bytes as usize,
					);
				}
			}
		}
		config.outgoing_data = Some(buf.buf.raw);
		config.outgoing_pitch_px = w as i32;
		return Ok(buf);
	}

	#[cfg(gpu_backend = "metal")]
	unsafe {
		use crate::DeviceHandleInit;
		let buf = crate::gpu::backends::metal::buffer::get_or_create(
			DeviceHandleInit::FromPtr(config.device_handle),
			w,
			h,
			bpp,
			tag,
		);
		if buf.buf.raw.is_null() {
			return Err("prepare_source_copy: Metal allocator returned null");
		}
		crate::gpu::backends::metal::buffer::copy_buffer(
			config,
			src_ptr,
			0,
			src_pitch_bytes,
			buf.buf.raw,
			0,
			dst_pitch_bytes,
			dst_pitch_bytes,
			h,
		)?;
		config.outgoing_data = Some(buf.buf.raw);
		config.outgoing_pitch_px = w as i32;
		return Ok(buf);
	}

	#[cfg(gpu_backend = "cuda")]
	unsafe {
		use crate::DeviceHandleInit;
		// `device_handle` here is a CUdevice ordinal; `cuda::buffer::allocate`
		// passes its pointer to `cuCtxSetCurrent`, so it needs the CUcontext.
		// Routing above guarantees `context_handle.is_some()`.
		let ctx = config.context_handle.expect("CUDA path requires context_handle");
		let buf = crate::gpu::backends::cuda::buffer::get_or_create(
			DeviceHandleInit::FromPtr(ctx),
			w,
			h,
			bpp,
			tag,
		);
		if buf.buf.raw.is_null() {
			return Err("prepare_source_copy: CUDA allocator returned null");
		}
		crate::gpu::backends::cuda::buffer::copy_buffer(
			config,
			src_ptr,
			0,
			src_pitch_bytes,
			buf.buf.raw,
			0,
			dst_pitch_bytes,
			dst_pitch_bytes,
			h,
		)?;
		config.outgoing_data = Some(buf.buf.raw);
		config.outgoing_pitch_px = w as i32;
		return Ok(buf);
	}

	#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
	{
		Err("prepare_source_copy: no GPU backend enabled")
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::cpu::buffer::get_or_create_with_mips;
	use crate::types::fill_mip_desc;

	/// Generate a 2-level chain from a 32x32 Bgra8 pattern and verify level 1
	/// matches the expected 2x2 box average. Per-pixel x/y gradient gives every
	/// 2x2 block four distinct values, so rounding-mode drift would show up.
	#[test]
	fn box_downsamples_known_32x32_pattern() {
		const W: u32 = 32;
		const H: u32 = 32;
		const BPP: u32 = 4;
		const LEVELS: u32 = 3;

		let buf = get_or_create_with_mips(W, H, BPP, LEVELS, 0xBEEF);

		let mut expected_l0 = vec![0u8; (W * H * BPP) as usize];
		for y in 0..H {
			for x in 0..W {
				let off = ((y * W + x) * BPP) as usize;
				expected_l0[off] = x as u8; 
				expected_l0[off + 1] = y as u8; 
				expected_l0[off + 2] = (x ^ y) as u8; 
				expected_l0[off + 3] = 255; 
			}
		}

		unsafe {
			std::ptr::copy_nonoverlapping(
				expected_l0.as_ptr(),
				buf.buf.raw as *mut u8,
				expected_l0.len(),
			);
		}

		let mut config = Configuration::cpu(
			buf.buf.raw,
			buf.buf.raw,
			W as i32,
			W as i32,
			W,
			H,
			BPP,
			1, 
		);
		config.outgoing_data = Some(buf.buf.raw);
		config.outgoing_pitch_px = W as i32;
		config.outgoing_width = W;
		config.outgoing_height = H;
		config.outgoing_mip_levels = LEVELS;

		unsafe { generate_mips(&config).expect("generate_mips failed"); }

		let base = buf.buf.raw as *const u8;
		let mut desc = crate::types::make_texture_desc(W, H, W, BPP, 1);
		fill_mip_desc(&mut desc, W, H, W, BPP, LEVELS);

		let l1_off = desc.mip_offset_bytes[1] as usize;
		let l1_w = desc.mip_width[1];
		let l1_h = desc.mip_height[1];
		for y in 0..l1_h {
			for x in 0..l1_w {
				let off = l1_off + ((y * l1_w + x) * BPP) as usize;
				let actual_b = unsafe { *base.add(off) };
				let actual_g = unsafe { *base.add(off + 1) };
				let actual_r = unsafe { *base.add(off + 2) };
				let actual_a = unsafe { *base.add(off + 3) };

				// Box average of 4 source px (f32 for parity with the shader's `* 0.25`).
				let p = |sx: u32, sy: u32| {
					let o = ((sy * W + sx) * BPP) as usize;
					(expected_l0[o] as f32, expected_l0[o + 1] as f32, expected_l0[o + 2] as f32, expected_l0[o + 3] as f32)
				};
				let sx = x * 2;
				let sy = y * 2;
				let (b0, g0, r0, a0) = p(sx, sy);
				let (b1, g1, r1, a1) = p(sx + 1, sy);
				let (b2, g2, r2, a2) = p(sx, sy + 1);
				let (b3, g3, r3, a3) = p(sx + 1, sy + 1);
				// Shader normalises bytes, averages, saturates, then re-quantises:
				//   byte_out = round(saturate(avg_float) * 255), avg_float = (b0+b1+b2+b3)/(4*255).
				let expect = |v0: f32, v1: f32, v2: f32, v3: f32| -> u8 {
					let avg = (v0 + v1 + v2 + v3) / 4.0 / 255.0;
					(avg.clamp(0.0, 1.0) * 255.0) as u8
				};
				let exp_b = expect(b0, b1, b2, b3);
				let exp_g = expect(g0, g1, g2, g3);
				let exp_r = expect(r0, r1, r2, r3);
				let exp_a = expect(a0, a1, a2, a3);
				let diff = |a: u8, b: u8| a.abs_diff(b);
				assert!(diff(actual_b, exp_b) <= 1, "L1 ({x},{y}) B: got {} expected {}", actual_b, exp_b);
				assert!(diff(actual_g, exp_g) <= 1, "L1 ({x},{y}) G: got {} expected {}", actual_g, exp_g);
				assert!(diff(actual_r, exp_r) <= 1, "L1 ({x},{y}) R: got {} expected {}", actual_r, exp_r);
				assert!(diff(actual_a, exp_a) <= 1, "L1 ({x},{y}) A: got {} expected {}", actual_a, exp_a);
			}
		}

		// Spot-check L2 corner only: level-of-level deltas compound rounding, ε ≈ 2 is acceptable.
		let l2_off = desc.mip_offset_bytes[2] as usize;
		let corner_b = unsafe { *base.add(l2_off) };
		assert!(corner_b <= 3, "L2 corner B drift: {}", corner_b);
	}

	/// End-to-end check on the CPU path: padded source → tight mip buffer copy +
	/// generate_mips. Validates the row-copy when source pitch > width * bpp
	/// (the common Premiere case).
	#[test]
	fn prepare_mip_source_copies_and_swaps_config() {
		const W: u32 = 16;
		const H: u32 = 16;
		const BPP: u32 = 4;
		const LEVELS: u32 = 3;
		// Padded source: pitch_px = 20 (80 bytes), tight = 16 (64 bytes); the helper must row-copy correctly.
		const SRC_PITCH_PX: u32 = 20;

		let mut src = vec![0u8; (SRC_PITCH_PX * H * BPP) as usize];
		for y in 0..H {
			for x in 0..W {
				let o = ((y * SRC_PITCH_PX + x) * BPP) as usize;
				src[o] = (x + 1) as u8;
				src[o + 1] = (y + 1) as u8;
				src[o + 2] = ((x ^ y) + 1) as u8;
				src[o + 3] = 255;
			}
		}

		let mut config = Configuration::cpu(
			src.as_mut_ptr() as *mut std::ffi::c_void,
			src.as_mut_ptr() as *mut std::ffi::c_void,
			SRC_PITCH_PX as i32,
			SRC_PITCH_PX as i32,
			W,
			H,
			BPP,
			1,
		);
		config.outgoing_data = Some(src.as_mut_ptr() as *mut std::ffi::c_void);
		config.outgoing_pitch_px = SRC_PITCH_PX as i32;
		config.outgoing_width = W;
		config.outgoing_height = H;
		config.outgoing_mip_levels = LEVELS;

		let _mip = unsafe { prepare_mip_source(&mut config, 0xF00D).expect("prepare_mip_source failed") };

		assert_eq!(config.outgoing_pitch_px, W as i32);
		let mip_ptr = config.outgoing_data.expect("outgoing_data lost");
		assert_ne!(mip_ptr as *const _, src.as_ptr() as *const _);

		// Mip buffer level 0 must match the source minus the trailing pitch pixels.
		let mip_base = mip_ptr as *const u8;
		for y in 0..H {
			for x in 0..W {
				let src_o = ((y * SRC_PITCH_PX + x) * BPP) as usize;
				let dst_o = ((y * W + x) * BPP) as usize;
				for c in 0..4 {
					let s = src[src_o + c];
					let d = unsafe { *mip_base.add(dst_o + c) };
					assert_eq!(s, d, "lod 0 pixel mismatch at ({x},{y}) channel {c}");
				}
			}
		}

		unsafe { generate_mips(&config).expect("generate_mips failed"); }
		let mut desc = crate::types::make_texture_desc(W, H, W, BPP, 1);
		crate::types::fill_mip_desc(&mut desc, W, H, W, BPP, LEVELS);
		let l1_off = desc.mip_offset_bytes[1] as usize;
		let p = |sx: u32, sy: u32| {
			let o = ((sy * SRC_PITCH_PX + sx) * BPP) as usize;
			(src[o] as u32, src[o + 1] as u32, src[o + 2] as u32, src[o + 3] as u32)
		};
		let (b0, g0, r0, a0) = p(0, 0);
		let (b1, g1, r1, a1) = p(1, 0);
		let (b2, g2, r2, a2) = p(0, 1);
		let (b3, g3, r3, a3) = p(1, 1);
		let expect_b = (b0 + b1 + b2 + b3) / 4;
		let expect_g = (g0 + g1 + g2 + g3) / 4;
		let expect_r = (r0 + r1 + r2 + r3) / 4;
		let expect_a = (a0 + a1 + a2 + a3) / 4;
		let actual_b = unsafe { *mip_base.add(l1_off) as u32 };
		let actual_g = unsafe { *mip_base.add(l1_off + 1) as u32 };
		let actual_r = unsafe { *mip_base.add(l1_off + 2) as u32 };
		let actual_a = unsafe { *mip_base.add(l1_off + 3) as u32 };
		let diff = |a: u32, b: u32| a.max(b) - a.min(b);
		assert!(diff(actual_b, expect_b) <= 1);
		assert!(diff(actual_g, expect_g) <= 1);
		assert!(diff(actual_r, expect_r) <= 1);
		assert!(diff(actual_a, expect_a) <= 1);
	}

	#[test]
	fn prepare_mip_source_rejects_single_level() {
		let mut config = Configuration::cpu(
			std::ptr::null_mut(),
			std::ptr::null_mut(),
			1,
			1,
			1,
			1,
			4,
			1,
		);
		config.outgoing_mip_levels = 1;
		let res = unsafe { prepare_mip_source(&mut config, 0) };
		match res {
			Err(msg) => assert!(msg.contains(">= 2"), "unexpected error: {msg}"),
			Ok(_) => panic!("prepare_mip_source should reject single-level configs"),
		}
	}

	#[test]
	fn generate_mips_is_noop_for_single_level() {
		let mut config = Configuration::cpu(
			std::ptr::null_mut(),
			std::ptr::null_mut(),
			1,
			1,
			1,
			1,
			4,
			1,
		);
		config.outgoing_mip_levels = 1;
		unsafe { generate_mips(&config).unwrap(); }
		config.outgoing_mip_levels = 0;
		unsafe { generate_mips(&config).unwrap(); }
	}
}
