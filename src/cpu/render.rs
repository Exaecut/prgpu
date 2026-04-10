use std::ffi::c_void;

use after_effects as ae;

use crate::types::{Configuration, FrameParams};

/// Per-pixel VEKL dispatch function type.
///
/// Signature matches the generated C++ per-pixel entry point:
/// ```cpp
/// void name_cpu_dispatch(
///     unsigned int gid_x, unsigned int gid_y,
///     const void* const* buffers,
///     const void* transition_params,
///     const void* user_params
/// );
/// ```
pub type CpuDispatchFn = unsafe extern "C" fn(
    u32,                    // gid_x
    u32,                    // gid_y
    *const *const c_void,   // buffers
    *const c_void,          // transition_params (FrameParams*)
    *const c_void,          // user_params
);

/// Wrapper to make buffer pointer array `Send + Sync`.
///
/// SAFETY: The buffer pointers are valid for the duration of the dispatch call.
/// They point to Layer-backed or intermediate buffers that outlive the iteration.
#[derive(Copy, Clone)]
struct SafeBuffers([*const c_void; 3]);
unsafe impl Send for SafeBuffers {}
unsafe impl Sync for SafeBuffers {}

/// Auto-compute bytes-per-pixel from the Layer based on host.
///
/// - **After Effects**: uses `world_type()` → U8=4, U15=8, F32=16
/// - **Premiere**: uses `pr_pixel_format()` to match BGRA/VUYA × 8u/16u/32f
pub fn compute_bpp(in_data: &ae::InData, layer: &ae::Layer) -> Result<u32, ae::Error> {
    if in_data.is_premiere() {
        let fmt = layer.pr_pixel_format()?;
        match fmt {
            // 8-bit: 4 bytes per pixel (4 channels × 1 byte)
            ae::pr::PixelFormat::Bgra4444_8u
            | ae::pr::PixelFormat::Vuya4444_8u
            | ae::pr::PixelFormat::Vuya4444_8u709
            | ae::pr::PixelFormat::Argb4444_8u
            | ae::pr::PixelFormat::Bgrx4444_8u
            | ae::pr::PixelFormat::Vuyx4444_8u
            | ae::pr::PixelFormat::Vuyx4444_8u709
            | ae::pr::PixelFormat::Xrgb4444_8u
            | ae::pr::PixelFormat::Bgrp4444_8u
            | ae::pr::PixelFormat::Vuyp4444_8u
            | ae::pr::PixelFormat::Vuyp4444_8u709
            | ae::pr::PixelFormat::Prgb4444_8u => Ok(4),

            // 16-bit: 8 bytes per pixel (4 channels × 2 bytes)
            ae::pr::PixelFormat::Bgra4444_16u
            | ae::pr::PixelFormat::Vuya4444_16u
            | ae::pr::PixelFormat::Argb4444_16u
            | ae::pr::PixelFormat::Bgrx4444_16u
            | ae::pr::PixelFormat::Xrgb4444_16u
            | ae::pr::PixelFormat::Bgrp4444_16u
            | ae::pr::PixelFormat::Prgb4444_16u => Ok(8),

            // 32-bit float: 16 bytes per pixel (4 channels × 4 bytes)
            ae::pr::PixelFormat::Bgra4444_32f
            | ae::pr::PixelFormat::Vuya4444_32f
            | ae::pr::PixelFormat::Vuya4444_32f709
            | ae::pr::PixelFormat::Argb4444_32f
            | ae::pr::PixelFormat::Bgrx4444_32f
            | ae::pr::PixelFormat::Vuyx4444_32f
            | ae::pr::PixelFormat::Vuyx4444_32f709
            | ae::pr::PixelFormat::Xrgb4444_32f
            | ae::pr::PixelFormat::Bgrp4444_32f
            | ae::pr::PixelFormat::Vuyp4444_32f
            | ae::pr::PixelFormat::Vuyp4444_32f709
            | ae::pr::PixelFormat::Prgb4444_32f
            | ae::pr::PixelFormat::Bgra4444_32fLinear
            | ae::pr::PixelFormat::Bgrp4444_32fLinear
            | ae::pr::PixelFormat::Bgrx4444_32fLinear
            | ae::pr::PixelFormat::Argb4444_32fLinear
            | ae::pr::PixelFormat::Prgb4444_32fLinear
            | ae::pr::PixelFormat::Xrgb4444_32fLinear => Ok(16),

            _ => Err(ae::Error::InvalidParms),
        }
    } else {
        match layer.world_type() {
            ae::aegp::WorldType::U8 => Ok(4),
            ae::aegp::WorldType::U15 => Ok(8),
            ae::aegp::WorldType::F32 => Ok(16),
            _ => Err(ae::Error::Generic),
        }
    }
}

/// Unified CPU render dispatch.
///
/// Iterates over every pixel, calling `dispatch_fn` for each one.
/// - **After Effects** with matching Layer dimensions: uses `iterate_with`
///   (AE multi-threaded iterate suites, supports 8/16/32-bit via
///   Iterate8/Iterate16/IterateFloat)
/// - **Premiere** or mismatched dimensions (blur intermediate buffers):
///   uses rayon parallel row iteration (Premiere has no IterateFloat)
pub fn render_cpu<P: Copy + Sync>(
    in_data: &ae::InData,
    in_layer: &ae::Layer,
    out_layer: &mut ae::Layer,
    config: &Configuration,
    dispatch_fn: CpuDispatchFn,
    user_params: &P,
) -> Result<(), ae::Error> {
    let w = config.width;
    let h = config.height;
    if w == 0 || h == 0 {
        return Ok(());
    }

    let buffers = SafeBuffers([
        config.outgoing_data.unwrap_or(std::ptr::null_mut()) as *const c_void,
        config.incoming_data.unwrap_or(std::ptr::null_mut()) as *const c_void,
        config.dest_data as *const c_void,
    ]);

    let tp = FrameParams {
        out_pitch: config.outgoing_pitch_px as u32,
        in_pitch: config.incoming_pitch_px as u32,
        dest_pitch: config.dest_pitch_px as u32,
        width: w,
        height: h,
        progress: config.progress,
        bpp: config.bytes_per_pixel,
    };

    // Use iterate_with when: AE host AND config dimensions match the output Layer.
    // This ensures the AE iterate suites iterate over the correct pixel range.
    // For blur intermediate buffers (dimensions differ), fall through to rayon.
    let can_iterate_with = !in_data.is_premiere()
        && w == out_layer.width() as u32
        && h == out_layer.height() as u32;

    if can_iterate_with {
        ae_dispatch(in_layer, out_layer, buffers, tp, user_params, dispatch_fn)
    } else {
        rayon_dispatch(w, h, buffers, tp, user_params, dispatch_fn)
    }
}

/// AE path: `iterate_with` for multi-threaded dispatch.
///
/// Supports 8-bit (Iterate8Suite), 16-bit (Iterate16Suite), and
/// 32-bit float (IterateFloatSuite) automatically via `GenericPixel` dispatch.
fn ae_dispatch<P: Copy + Sync>(
    in_layer: &ae::Layer,
    out_layer: &mut ae::Layer,
    buffers: SafeBuffers,
    tp: FrameParams,
    user_params: &P,
    dispatch_fn: CpuDispatchFn,
) -> Result<(), ae::Error> {
    in_layer.iterate_with(
        out_layer,
        0,
        tp.height as i32,
        None,
        move |x: i32, y: i32, _pixel: ae::GenericPixel, _out_pixel: ae::GenericPixelMut| {
            unsafe {
                dispatch_fn(
                    x as u32,
                    y as u32,
                    buffers.0.as_ptr(),
                    &tp as *const _ as *const c_void,
                    user_params as *const _ as *const c_void,
                );
            }
            Ok(())
        },
    )
}

/// Rayon path: parallel row iteration.
///
/// Used for Premiere (no IterateFloat support — sdk_noise pattern) and for
/// intermediate buffer dispatch where config dimensions don't match Layer dimensions
/// (e.g., blur downsampled buffers).
fn rayon_dispatch<P: Copy + Sync>(
    width: u32,
    height: u32,
    buffers: SafeBuffers,
    tp: FrameParams,
    user_params: &P,
    dispatch_fn: CpuDispatchFn,
) -> Result<(), ae::Error> {
    use rayon::prelude::*;

    // Convert pointers to usize so the closure captures plain usize values
    // (raw pointers are not Send/Sync, but usize is).
    let buf_ptr = buffers.0.as_ptr() as usize;
    let tp_ptr = &tp as *const _ as usize;
    let up_ptr = user_params as *const _ as usize;

    (0..height).into_par_iter().for_each(move |y| {
        for x in 0..width {
            unsafe {
                dispatch_fn(
                    x,
                    y,
                    buf_ptr as *const *const c_void,
                    tp_ptr as *const c_void,
                    up_ptr as *const c_void,
                );
            }
        }
    });

    Ok(())
}
