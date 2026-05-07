//! Built-in mip-chain downsampler shared by any effect that opts into the
//! pyramid via `Configuration::outgoing_mip_levels`. Host API +
//! full CPU / GPU walkthrough: [`prgpu/docs/mip_chain.md`](../../../docs/mip_chain.md).
//! Shader-side reference: [`vekl/docs/reference/texture/view.md`](../../../../vekl/docs/reference/texture/view.md).
//!
//! The actual shader (`prgpu/shaders/mip_downsample.slang`) is compiled by
//! prgpu's own `build.rs` into prgpu's `OUT_DIR`; the `declare_kernel!`
//! invocation below wires the `include_shader!` + CPU FFI + GPU dispatch.
//!
//! # Three-step host recipe
//!
//! Every effect that wants a mip-chain source does exactly this, same
//! shape on CPU and GPU paths (the only backend-specific bit is the
//! allocator + the level-0 copy in step 2):
//!
//! ```ignore
//! // 1. Request a chain.
//! config.outgoing_mip_levels = 4;
//!
//! // 2. Allocate a mip-capable buffer for the chosen backend, copy
//! //    Premiere's outgoing into level 0, and redirect config.outgoing_data.
//! let mip_buf = unsafe {
//!     prgpu::gpu::backends::metal::buffer::get_or_create_with_mips(
//!         device, w, h, bpp, config.outgoing_mip_levels, MIP_TAG,
//!     )
//! };
//! // (Metal: MTLBlitCommandEncoder copyFromBuffer:...)
//! // (CUDA:  cuMemcpyDtoD_v2(...))
//! // (CPU:   std::ptr::copy_nonoverlapping(...))
//! config.outgoing_data = Some(mip_buf.buf.raw);
//!
//! // 3. Fill levels 1..N-1 and run the effect kernel. prgpu's dispatcher
//! //    calls make_outgoing_desc(&config), which auto-populates the
//! //    mip metadata in frame.outDesc — no shader-side setup.
//! unsafe { prgpu::kernels::mip::generate_mips(&config)?; }
//! unsafe { effect_kernel(&config, user_params)?; }
//! ```
//!
//! # CPU vs GPU dispatch routing
//!
//! `generate_mips` auto-routes based on `config.device_handle`:
//!
//! - `device_handle != null`  → GPU path (`mip_downsample(&pass_cfg, params)`)
//!   which goes through `backends::dispatch_kernel` (Metal or CUDA).
//! - `device_handle == null`  → CPU path (`render_cpu_direct`) that reuses
//!   the bench harness's pure-rayon tile loop. Same code path as
//!   Premiere's CPU failover.
//!
//! Effects don't need to fork on backend — one call site works for bench,
//! GPU render, and CPU failover alike.
//!
//! # Cache / allocator
//!
//! Use `{cpu,gpu::backends::metal,gpu::backends::cuda}::buffer::get_or_create_with_mips`;
//! the legacy 4-arg `get_or_create` still works (it delegates with
//! `mip_levels = 1`). Every allocator is a 12-slot LRU keyed on
//! `(device, w, h, bpp, mip_levels, tag)` so mip buffers and plain
//! buffers at the same dims don't share a slot.

use crate::declare_kernel;
use crate::types::{Configuration, ImageBuffer, MAX_MIP};

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

/// Allocate a mip-capable buffer, copy `config.outgoing_data` into level
/// 0, and redirect `config.outgoing_data` to the new buffer. Returns the
/// allocated `ImageBuffer` so the caller can keep it alive for the
/// remainder of the frame (dropping it has no GPU effect — the buffer is
/// owned by the prgpu cache).
///
/// This is the Phase 3 one-call convenience helper that makes mips
/// "invisible to kernel authors":
///
/// ```ignore
/// config.outgoing_mip_levels = 4;
/// let _mip = unsafe { prepare_mip_source(&mut config, MY_MIP_TAG)? };
/// unsafe { generate_mips(&config)?; }
/// unsafe { my_effect_kernel(&config, params)?; }
/// ```
///
/// The host pattern is identical on CPU / Metal / CUDA; this function
/// routes to the correct backend allocator + buffer-to-buffer copy
/// based on `config.device_handle`:
///
/// - `device_handle == null` → CPU: `cpu::buffer::get_or_create_with_mips`
///   + `std::ptr::copy_nonoverlapping` (row-by-row when the source has
///   pitch padding, single flat copy otherwise).
/// - `device_handle != null` on macOS → Metal:
///   `gpu::backends::metal::buffer::get_or_create_with_mips` +
///   `copy_buffer` (MTLBlitCommandEncoder).
/// - `device_handle != null` on Windows with CUDA → CUDA:
///   `gpu::backends::cuda::buffer::get_or_create_with_mips` +
///   `copy_buffer` (`cuMemcpy2D_v2` / `cuMemcpyDtoD_v2`).
///
/// `tag` participates in the cache key along with `(w, h, bpp, mip_levels)`
/// so distinct effect instances don't stomp on each other's mip buffers.
/// Convention: upper half = effect namespace, lower half = role
/// (e.g. `0x5242_0001` for "RB" / radialblur's primary mip source).
///
/// After `prepare_mip_source` returns, `config.outgoing_data` points at
/// the mip buffer, `config.outgoing_pitch_px = outgoing_width` (tight),
/// and the caller's original source pointer is no longer referenced by
/// the config. Call [`generate_mips`] next to fill levels 1..N-1.
///
/// # Safety
/// - `config.outgoing_data` must point to a readable buffer of at least
///   `config.outgoing_pitch_px * config.outgoing_height * config.bytes_per_pixel`
///   bytes (GPU buffer on device paths, host memory on CPU path).
/// - On Metal, `config.command_queue_handle` must be a valid `MTLCommandQueue`
///   on the same device as `config.device_handle`.
/// - No other CPU / GPU work may touch the returned buffer until the
///   caller's effect kernel dispatch completes.
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

	if config.device_handle.is_null() {
		// ---- CPU path -------------------------------------------------------
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

	// ---- GPU path -----------------------------------------------------------
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
			config.command_queue_handle as *mut objc::runtime::Object,
			src_ptr as *mut objc::runtime::Object,
			0,
			src_pitch_bytes,
			buf.buf.raw as *mut objc::runtime::Object,
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
		let buf = crate::gpu::backends::cuda::buffer::get_or_create_with_mips(
			DeviceHandleInit::FromPtr(config.device_handle),
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
			config.device_handle,
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

	/// End-to-end coverage of the three-step Phase 3 host recipe on the CPU
	/// path: `prepare_mip_source` copies a padded source into a tight mip
	/// buffer, swaps the config, and `generate_mips` fills subsequent lods.
	/// Validates that the copy handles padded-row sources correctly (source
	/// pitch > width * bpp is the common Premiere case).
	#[test]
	fn prepare_mip_source_copies_and_swaps_config() {
		const W: u32 = 16;
		const H: u32 = 16;
		const BPP: u32 = 4;
		const LEVELS: u32 = 3;
		// Simulate a source with padded rows: pitch_px = 20 (80 bytes) vs
		// tight = 16 (64 bytes). The helper must row-copy correctly.
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

		// After the swap the config points at the mip buffer with a tight pitch.
		assert_eq!(config.outgoing_pitch_px, W as i32);
		let mip_ptr = config.outgoing_data.expect("outgoing_data lost");
		assert_ne!(mip_ptr as *const _, src.as_ptr() as *const _);

		// Level 0 inside the mip buffer must match the source, row-stripped
		// of the trailing pitch pixels.
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

		// generate_mips fills level 1 with a 2x2 box average of level 0.
		unsafe { generate_mips(&config).expect("generate_mips failed"); }
		let mut desc = crate::types::make_texture_desc(W, H, W, BPP, 1);
		crate::types::fill_mip_desc(&mut desc, W, H, W, BPP, LEVELS);
		let l1_off = desc.mip_offset_bytes[1] as usize;
		// Spot-check (0, 0) at level 1: average of src[(0..2, 0..2)].
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
