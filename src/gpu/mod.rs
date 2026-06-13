use premiere::{self as pr};
use std::slice;

pub mod backends;
pub mod metrics;
pub mod render_properties;
pub mod scheduling;
pub mod shaders;

#[inline]
fn frames_as_slice<'a>(frames: *const pr::sys::PPixHand, frame_count: usize) -> Result<&'a [pr::sys::PPixHand], pr::Error> {
	if frames.is_null() || frame_count == 0 {
		return Err(pr::Error::Fail);
	}

	Ok(unsafe { slice::from_raw_parts(frames, frame_count) })
}

pub(crate) fn gpu_bytes_per_pixels(pixel_format: pr::PixelFormat) -> i32 {
	match pixel_format {
		pr::PixelFormat::GpuBgra4444_32f => 16,
		pr::PixelFormat::GpuBgra4444_16f => 8,
		pr::PixelFormat::Bgra4444_32f => 16, // same layout as GpuBgra4444_32f
		_ => panic!("Unsupported pixel format"),
	}
}

/// Vekl `PixelStorage` tag for a Premiere GPU pixel format. `GpuBgra4444_16f`
/// is **half-float** (not unorm16), so it cannot be inferred from bpp alone —
/// the host format is the only reliable signal.
pub(crate) fn gpu_storage(pixel_format: pr::PixelFormat) -> u32 {
	match pixel_format {
		pr::PixelFormat::GpuBgra4444_16f => crate::types::PIXEL_STORAGE_FLOAT16X4,
		_ => crate::types::PIXEL_STORAGE_FLOAT32X4, // GpuBgra4444_32f, Bgra4444_32f
	}
}

pub mod buffer {
	pub use imp::*;

	#[cfg(gpu_backend = "metal")]
	mod imp {
		pub use crate::gpu::backends::metal::buffer::*;
	}

	#[cfg(gpu_backend = "cuda")]
	mod imp {
		pub use crate::gpu::backends::cuda::buffer::*;
	}

	#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
	mod imp {
		compile_error!("Unsupported gpu_backend");
	}
}

pub mod pipeline {
	pub use imp::*;

	#[cfg(gpu_backend = "metal")]
	mod imp {
		pub use crate::gpu::backends::metal::pipeline::*;
	}

	#[cfg(gpu_backend = "cuda")]
	mod imp {
		pub use crate::gpu::backends::cuda::pipeline::*;
	}

	#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
	mod imp {
		compile_error!("Unsupported gpu_backend");
	}
}

/// Per-frame submission scope shared by both GPU backends: the adapter
/// brackets `run_graph` with `begin`/`end`, passes enqueue into the frame's
/// stream / command buffer with no per-pass sync, and `end` performs the one
/// sync Adobe's buffer lifecycle requires.
pub mod frame_scope {
	pub use imp::*;

	#[cfg(gpu_backend = "metal")]
	mod imp {
		pub use crate::gpu::backends::metal::frame_scope::*;
	}

	#[cfg(gpu_backend = "cuda")]
	mod imp {
		pub use crate::gpu::backends::cuda::frame_scope::*;
	}

	#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
	mod imp {
		use crate::types::FrameScopeDesc;

		pub const ERR_WATCHDOG: &str = "metal frame watchdog";

		pub fn begin(_desc: &FrameScopeDesc) {}
		pub fn end(_desc: &FrameScopeDesc) -> Result<(), &'static str> {
			Ok(())
		}
	}
}

pub mod fence {
	pub use imp::*;

	#[cfg(gpu_backend = "metal")]
	mod imp {
		pub use crate::gpu::backends::metal::fence::*;
	}

	#[cfg(gpu_backend = "cuda")]
	mod imp {
		pub use crate::gpu::backends::cuda::fence::*;
	}

	#[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
	mod imp {
		compile_error!("Unsupported gpu_backend");
	}
}
