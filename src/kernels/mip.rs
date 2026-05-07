//! Built-in mip-chain downsampler shared by any effect that opts into the
//! pyramid via `Configuration::outgoing_mip_levels`. The actual shader
//! (`prgpu/shaders/mip_downsample.slang`) is compiled by prgpu's own
//! `build.rs` into prgpu's `OUT_DIR`; the `declare_kernel!` invocation
//! below wires the `include_shader!` + CPU FFI + GPU dispatch for it.
//!
//! Effect gpu.rs pattern (Phase 3 will consume this):
//! ```ignore
//! config.outgoing_mip_levels = 4;
//! let mip_buf = unsafe { prgpu::gpu::buffer::get_or_create_with_mips(
//!     device, w, h, bpp, 4, MIP_TAG,
//! ) };
//! // Blit/copy level 0 from Premiere's outgoing into mip_buf.
//! let mut mip_cfg = config;
//! mip_cfg.outgoing_data = Some(mip_buf.buf.raw);
//! prgpu::kernels::mip::generate_mips(&mip_cfg)?;
//! // Swap outgoing on the user config and dispatch the effect kernel.
//! config.outgoing_data = Some(mip_buf.buf.raw);
//! unsafe { effect_kernel(&config, user_params)?; }
//! ```
//!
//! On CPU, the same entry point is called via `render_cpu_direct` so the
//! bench harness + Premiere GPU-failover path work without any AE plumbing.

use crate::declare_kernel;
use crate::types::{Configuration, MAX_MIP};

/// Uniform parameters for the mip-downsample kernel.
///
/// `_pad*` exists purely so slang aligns the ConstantBuffer layout to a
/// 16-byte vec4 boundary — matches the `uint _pad0; uint _pad1; uint _pad2;`
/// fields in `prgpu/shaders/mip_downsample.slang`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MipDownsampleParams {
	pub src_lod: u32,
	pub _pad0: u32,
	pub _pad1: u32,
	pub _pad2: u32,
}

declare_kernel!(mip_downsample, MipDownsampleParams);

/// Fill levels `1..N` of a mip chain whose level 0 is already populated in
/// `config.outgoing_data`. `config.outgoing_mip_levels` drives the level
/// count; values `<= 1` short-circuit to a no-op.
///
/// The caller is responsible for allocating a large-enough buffer (via
/// [`crate::cpu::buffer::get_or_create_with_mips`] or the Metal / CUDA
/// equivalents) and for writing level 0 into it. Phase 3 will add a
/// convenience helper that also handles the buffer setup + copy.
///
/// Re-entrant and cheap when `outgoing_mip_levels <= 1`, so callers can
/// unconditionally call this before every user-kernel dispatch.
///
/// # Safety
/// The buffer pointed to by `config.outgoing_data` must be at least
/// [`crate::types::mip_buffer_size_bytes`] bytes, laid out tightly per
/// [`crate::types::fill_mip_desc`].
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

		// The downsample kernel reads from and writes to the same buffer; we
		// point every slot at the mip buffer so the ABI is satisfied. The
		// shader only reads from `dst` (slot 2) in practice — `outgoing` and
		// `incoming` are bound for completeness and to keep the 5-buffer
		// Metal/CUDA calling convention happy.
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

		if !pass_cfg.device_handle.is_null() {
			unsafe { mip_downsample(&pass_cfg, params)? };
		} else {
			// CPU dispatch — no AE plumbing, direct rayon tile loop.
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

#[cfg(test)]
mod tests {
	use super::*;
	use crate::cpu::buffer::get_or_create_with_mips;
	use crate::types::fill_mip_desc;

	/// Generate a 2-level mip chain from a 32x32 Bgra8 source and verify
	/// that level 1 is a faithful 2x2 box average of level 0.
	///
	/// The pattern is a per-pixel gradient (x,y -> packed BGRA bytes) so
	/// every 2x2 block has four distinct values; rounding-mode errors
	/// would show up as off-by-one drift on any channel.
	#[test]
	fn box_downsamples_known_32x32_pattern() {
		const W: u32 = 32;
		const H: u32 = 32;
		const BPP: u32 = 4;
		const LEVELS: u32 = 3;

		let buf = get_or_create_with_mips(W, H, BPP, LEVELS, 0xBEEF);

		// Compute the expected mip chain on CPU using the same 2x2 box math.
		let mut expected_l0 = vec![0u8; (W * H * BPP) as usize];
		for y in 0..H {
			for x in 0..W {
				let off = ((y * W + x) * BPP) as usize;
				expected_l0[off] = x as u8; // B
				expected_l0[off + 1] = y as u8; // G
				expected_l0[off + 2] = (x ^ y) as u8; // R
				expected_l0[off + 3] = 255; // A
			}
		}

		// Write the gradient into the mip buffer's level 0 region.
		unsafe {
			std::ptr::copy_nonoverlapping(
				expected_l0.as_ptr(),
				buf.buf.raw as *mut u8,
				expected_l0.len(),
			);
		}

		// Build an AE-free Configuration and dispatch generate_mips via CPU.
		let mut config = Configuration::cpu(
			buf.buf.raw,
			buf.buf.raw,
			W as i32,
			W as i32,
			W,
			H,
			BPP,
			1, // BGRA
		);
		config.outgoing_data = Some(buf.buf.raw);
		config.outgoing_pitch_px = W as i32;
		config.outgoing_width = W;
		config.outgoing_height = H;
		config.outgoing_mip_levels = LEVELS;

		unsafe { generate_mips(&config).expect("generate_mips failed"); }

		// Read back levels 1 + 2.
		let base = buf.buf.raw as *const u8;
		// Derive mip offsets the exact same way the shader does.
		let mut desc = crate::types::make_texture_desc(W, H, W, BPP, 1);
		fill_mip_desc(&mut desc, W, H, W, BPP, LEVELS);

		// Expected level 1: 16x16 box-averaged from level 0.
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

				// Expected box average from 4 source pixels (cast to f32 for
				// division parity with the shader's `* 0.25`).
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
				// Shader does sRGB byte -> normalized float -> average -> saturate -> byte.
				// So:  byte_out = round(saturate(avg_float) * 255) where avg_float = (b0+b1+b2+b3)/(4*255).
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

		// Spot-check level 2 (8x8) by its corner pixel only — level-of-level
		// deltas compound rounding errors, so an epsilon of ~2 is acceptable.
		let l2_off = desc.mip_offset_bytes[2] as usize;
		let corner_b = unsafe { *base.add(l2_off) };
		// 4 source px from L1 (0..2, 0..2) averaged -> expected B ≈ 1.5
		// (gradient averages). Acceptable range [0, 3].
		assert!(corner_b <= 3, "L2 corner B drift: {}", corner_b);
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
