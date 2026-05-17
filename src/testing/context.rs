//! Creates a real GPU device and command queue, wraps prgpu's buffer
//! allocators, and provides host↔GPU transfer helpers.

use std::ffi::c_void;

use crate::types::{Configuration, DeviceHandleInit, ImageBuffer};
use crate::gpu::backends;

/// Buffer from prgpu's LRU cache. Keep this value alive to prevent eviction.
pub struct GpuBuffer {
    /// Raw GPU pointer (`MTLBuffer*` on Metal, `CUdeviceptr` on CUDA).
    pub data: *mut c_void,
    /// Row pitch in pixels. Tight (= width) when allocated through this module.
    pub pitch_px: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_pixel: u32,
    #[allow(dead_code)]
    _img: ImageBuffer, // keeps LRU entry alive
}

pub struct GpuContext {
    pub device: *mut c_void,
    pub command_queue: *mut c_void,
    #[allow(dead_code)]
    pub context: Option<*mut c_void>,
}

impl GpuContext {
    /// Selects the default system GPU device.
    ///
    /// macOS → `MTLCreateSystemDefaultDevice`. Windows → CUDA device 0.
    /// Returns an error when no supported GPU is available.
    pub fn create() -> Result<Self, String> {
        #[cfg(gpu_backend = "metal")]
        {
            create_metal_context()
        }
        #[cfg(gpu_backend = "cuda")]
        {
            create_cuda_context()
        }
        #[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
        {
            Err("no GPU backend compiled in (metal or cuda)".into())
        }
    }

    pub fn is_available() -> bool {
        Self::create().is_ok()
    }

    pub fn create_io_buffers(
        &self,
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> Result<(GpuBuffer, GpuBuffer), String> {
        let tag = 0x54455354; // "TEST"
        let input = self.create_buffer_inner(width, height, bytes_per_pixel, tag)?;
        let tag_out = 0x54455355; // "TEST"+1
        let output = self.create_buffer_inner(width, height, bytes_per_pixel, tag_out)?;
        Ok((input, output))
    }

    pub fn create_buffer(
        &self,
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
        tag: u32,
    ) -> Result<GpuBuffer, String> {
        self.create_buffer_inner(width, height, bytes_per_pixel, tag)
    }

    fn create_buffer_inner(
        &self,
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
        tag: u32,
    ) -> Result<GpuBuffer, String> {
        let init = DeviceHandleInit::FromPtr(self.device);
        #[cfg(gpu_backend = "metal")]
        {
            let img = unsafe {
                backends::metal::buffer::get_or_create(init, width, height, bytes_per_pixel, tag)
            };
            if img.buf.raw.is_null() {
                return Err("Metal buffer allocation returned null".into());
            }
            Ok(GpuBuffer {
                data: img.buf.raw,
                pitch_px: img.pitch_px,
                width: img.width,
                height: img.height,
                bytes_per_pixel: img.bytes_per_pixel,
                _img: img,
            })
        }
        #[cfg(gpu_backend = "cuda")]
        {
            let img = unsafe {
                backends::cuda::buffer::get_or_create(init, width, height, bytes_per_pixel, tag)
            };
            if img.buf.raw.is_null() {
                return Err("CUDA buffer allocation returned null".into());
            }
            Ok(GpuBuffer {
                data: img.buf.raw,
                pitch_px: img.pitch_px,
                width: img.width,
                height: img.height,
                bytes_per_pixel: img.bytes_per_pixel,
                _img: img,
            })
        }
        #[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
        {
            let _ = (init, width, height, bytes_per_pixel, tag);
            Err("no GPU backend".into())
        }
    }

    /// `host_data` must be tightly-packed BGRA at `width * height * bpp` bytes.
    pub fn upload_to_buffer(
        &self,
        dst: &GpuBuffer,
        host_data: &[u8],
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> Result<(), String> {
        let expected = (width as u64) * (height as u64) * (bytes_per_pixel as u64);
        if host_data.len() as u64 != expected {
            return Err(format!(
                "upload: data length {} != expected {} ({}x{}x{})",
                host_data.len(),
                expected,
                width,
                height,
                bytes_per_pixel
            ));
        }

        #[cfg(gpu_backend = "metal")]
        {
            upload_metal(self, dst, host_data, width, height, bytes_per_pixel)
        }
        #[cfg(gpu_backend = "cuda")]
        {
            upload_cuda(self, dst, host_data, width, height, bytes_per_pixel)
        }
        #[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
        {
            let _ = (self, dst, host_data, width, height, bytes_per_pixel);
            Err("no GPU backend".into())
        }
    }

    pub fn download_from_buffer(
        &self,
        src: &GpuBuffer,
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> Result<Vec<u8>, String> {
        self.download_raw(src.data, src.pitch_px, width, height, bytes_per_pixel)
    }

    /// Download from a raw GPU pointer without a `GpuBuffer` wrapper.
    pub fn download_raw(
        &self,
        data: *mut c_void,
        pitch_px: u32,
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> Result<Vec<u8>, String> {
        #[cfg(gpu_backend = "metal")]
        {
            let tmp = GpuBuffer {
                data,
                pitch_px,
                width,
                height,
                bytes_per_pixel,
                _img: crate::types::ImageBuffer {
                    buf: crate::types::BufferObj { raw: data },
                    pitch_px,
                    width,
                    height,
                    bytes_per_pixel,
                    row_bytes: pitch_px * bytes_per_pixel,
                },
            };
            download_metal(self, &tmp, width, height, bytes_per_pixel)
        }
        #[cfg(gpu_backend = "cuda")]
        {
            let tmp = GpuBuffer {
                data,
                pitch_px,
                width,
                height,
                bytes_per_pixel,
                _img: crate::types::ImageBuffer {
                    buf: crate::types::BufferObj { raw: data },
                    pitch_px,
                    width,
                    height,
                    bytes_per_pixel,
                    row_bytes: pitch_px * bytes_per_pixel,
                },
            };
            download_cuda(self, &tmp, width, height, bytes_per_pixel)
        }
        #[cfg(not(any(gpu_backend = "metal", gpu_backend = "cuda")))]
        {
            let _ = (self, data, pitch_px, width, height, bytes_per_pixel);
            Err("no GPU backend".into())
        }
    }

    /// Sets `pixel_layout = 1` (BGRA, matching Premiere GPU path).
    pub fn build_config(
        &self,
        input: &GpuBuffer,
        output: &GpuBuffer,
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> Configuration {
        let ctx = self.context;

        Configuration {
            device_handle: self.device,
            context_handle: ctx,
            command_queue_handle: self.command_queue,
            outgoing_data: Some(input.data),
            incoming_data: Some(input.data),
            dest_data: output.data,
            outgoing_pitch_px: input.pitch_px as i32,
            incoming_pitch_px: input.pitch_px as i32,
            dest_pitch_px: output.pitch_px as i32,
            width,
            height,
            outgoing_width: width,
            outgoing_height: height,
            incoming_width: width,
            incoming_height: height,
            bytes_per_pixel,
            time: 0.0,
            progress: 0.0,
            render_generation: 0,
            pixel_layout: 1, // BGRA — GPU path convention
            outgoing_mip_levels: 0,
        }
    }
}

#[cfg(gpu_backend = "metal")]
fn create_metal_context() -> Result<GpuContext, String> {
    use objc::{class, msg_send, runtime::Object, sel, sel_impl};

    let device: *mut Object = unsafe { msg_send![class!(MTLCreateSystemDefaultDevice), retain] };
    if device.is_null() {
        return Err("MTLCreateSystemDefaultDevice returned null — no Metal-capable GPU".into());
    }
    let queue: *mut Object = unsafe { msg_send![device, newCommandQueue] };
    if queue.is_null() {
        unsafe { let _: () = msg_send![device, release]; }
        return Err("newCommandQueue returned null".into());
    }
    Ok(GpuContext {
        device: device as *mut c_void,
        command_queue: queue as *mut c_void,
        context: None,
    })
}

#[cfg(gpu_backend = "metal")]
fn upload_metal(
    gpu: &GpuContext,
    dst: &GpuBuffer,
    host_data: &[u8],
    width: u32,
    height: u32,
    bpp: u32,
) -> Result<(), String> {
    use objc::{msg_send, runtime::Object, sel, sel_impl};
    use crate::gpu::backends::metal::buffer::copy_buffer;

    let device = gpu.device as *mut Object;
    let row_bytes = width * bpp;
    let length = (row_bytes * height) as u64;

    let options: u64 = 0; // MTLStorageModeShared = 0
    let staging: *mut Object = unsafe {
        msg_send![device, newBufferWithBytes: host_data.as_ptr()
                                     length: length as usize
                                    options: options]
    };
    if staging.is_null() {
        return Err("failed to allocate Metal staging buffer".into());
    }

    let config = Configuration {
        device_handle: gpu.device,
        context_handle: None,
        command_queue_handle: gpu.command_queue,
        outgoing_data: None,
        incoming_data: None,
        dest_data: dst.data,
        outgoing_pitch_px: 0,
        incoming_pitch_px: 0,
        dest_pitch_px: dst.pitch_px as i32,
        width,
        height,
        outgoing_width: 0,
        outgoing_height: 0,
        incoming_width: 0,
        incoming_height: 0,
        bytes_per_pixel: bpp,
        time: 0.0,
        progress: 0.0,
        render_generation: 0,
        pixel_layout: 1,
        outgoing_mip_levels: 0,
    };

    let result = unsafe {
        copy_buffer(
            &config,
            staging as *mut c_void,
            0,
            row_bytes,
            dst.data,
            0,
            row_bytes,
            row_bytes,
            height,
        )
    };

    unsafe { let _: () = msg_send![staging, release]; }

    result.map_err(|e| format!("Metal upload blit failed: {e}"))
}

#[cfg(gpu_backend = "metal")]
fn download_metal(
    gpu: &GpuContext,
    src: &GpuBuffer,
    width: u32,
    height: u32,
    bpp: u32,
) -> Result<Vec<u8>, String> {
    use objc::{msg_send, runtime::Object, sel, sel_impl};
    use std::ptr;

    let device = gpu.device as *mut Object;
    let row_bytes = width * bpp;
    let length = (row_bytes * height) as u64;

    let options: u64 = 0; // MTLStorageModeShared = 0
    let staging: *mut Object = unsafe {
        msg_send![device, newBufferWithLength: length as usize options: options]
    };
    if staging.is_null() {
        return Err("failed to allocate Metal staging buffer for download".into());
    }

    // Manual blit because we need src→staging (copy_buffer goes the other way).
    let queue = gpu.command_queue as *mut Object;
    let cmd: *mut Object = unsafe { msg_send![queue, commandBuffer] };
    if cmd.is_null() {
        unsafe { let _: () = msg_send![staging, release]; }
        return Err("download: commandBuffer returned null".into());
    }

    let enc: *mut Object = unsafe { msg_send![cmd, blitCommandEncoder] };
    if enc.is_null() {
        unsafe { let _: () = msg_send![staging, release]; }
        return Err("download: blitCommandEncoder returned null".into());
    }

    unsafe {
        let _: () = msg_send![enc,
            copyFromBuffer: (src.data as *mut Object)
                sourceOffset: 0u64
                toBuffer: staging
           destinationOffset: 0u64
                        size: length as usize];
        let _: () = msg_send![enc, endEncoding];
        let _: () = msg_send![cmd, commit];
        let _: () = msg_send![cmd, waitUntilCompleted];
    }

    let contents: *const u8 = unsafe { msg_send![staging, contents] };
    if contents.is_null() {
        unsafe { let _: () = msg_send![staging, release]; }
        return Err("download: staging buffer contents is null".into());
    }

    let mut out = vec![0u8; length as usize];
    if row_bytes == (src.pitch_px * bpp) {
        unsafe { ptr::copy_nonoverlapping(contents, out.as_mut_ptr(), length as usize); }
    } else {
        let tight_row = row_bytes as usize;
        let src_row = src.pitch_px as usize * bpp as usize;
        for y in 0..height as usize {
            let src_off = y * src_row;
            let dst_off = y * tight_row;
            unsafe {
                ptr::copy_nonoverlapping(
                    contents.add(src_off),
                    out.as_mut_ptr().add(dst_off),
                    tight_row,
                );
            }
        }
    }

    unsafe { let _: () = msg_send![staging, release]; }
    Ok(out)
}

#[cfg(gpu_backend = "cuda")]
fn create_cuda_context() -> Result<GpuContext, String> {
    use cudarc::driver::sys::{
        cuCtxSetCurrent, cuDeviceGet, cuDevicePrimaryCtxRetain, cuInit, cuStreamCreate,
        CUcontext, CUdevice, CUresult, CUstream,
    };

    // Bail early if the CUDA driver DLL is missing — cudarc's fallback
    // dynamic loading can segfault when the DLL is absent.
    #[cfg(target_os = "windows")]
    {
        let sys32 = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".into());
        let dll = std::path::PathBuf::from(sys32).join("System32").join("nvcuda.dll");
        if !dll.exists() {
            return Err(format!("CUDA driver not found at {}", dll.display()));
        }
    }

    let result = unsafe { cuInit(0) };
    if result != CUresult::CUDA_SUCCESS {
        return Err(format!("cuInit failed: {:?}", result));
    }

    let mut device: CUdevice = 0;
    let result = unsafe { cuDeviceGet(&mut device, 0) };
    if result != CUresult::CUDA_SUCCESS {
        return Err(format!("cuDeviceGet(0) failed: {:?} — no CUDA GPU", result));
    }

    let mut cu_ctx: CUcontext = std::ptr::null_mut();
    let result = unsafe { cuDevicePrimaryCtxRetain(&mut cu_ctx, device) };
    if result != CUresult::CUDA_SUCCESS {
        return Err(format!("cuDevicePrimaryCtxRetain failed: {:?}", result));
    }

    unsafe { cuCtxSetCurrent(cu_ctx) };

    let mut stream: CUstream = std::ptr::null_mut();
    let result = unsafe { cuStreamCreate(&mut stream, 0) };
    if result != CUresult::CUDA_SUCCESS {
        return Err(format!("cuStreamCreate failed: {:?}", result));
    }

    Ok(GpuContext {
        device: cu_ctx as *mut c_void,
        command_queue: stream as *mut c_void,
        context: Some(cu_ctx as *mut c_void),
    })
}

#[cfg(gpu_backend = "cuda")]
fn upload_cuda(
    gpu: &GpuContext,
    dst: &GpuBuffer,
    host_data: &[u8],
    width: u32,
    height: u32,
    bpp: u32,
) -> Result<(), String> {
    use cudarc::driver::sys::{
        cuCtxSetCurrent, cuMemcpyHtoD_v2, CUcontext, CUdeviceptr, CUresult,
    };

    let ctx = gpu.device as CUcontext;
    unsafe { cuCtxSetCurrent(ctx) };

    let row_bytes = (width * bpp) as usize;
    let dst_pitch = dst.pitch_px as usize * bpp as usize;

    if dst_pitch == row_bytes {
        let result = unsafe {
            cuMemcpyHtoD_v2(
                dst.data as CUdeviceptr,
                host_data.as_ptr() as *const c_void,
                row_bytes * height as usize,
            )
        };
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemcpyHtoD failed: {:?}", result));
        }
    } else {
        for y in 0..height as usize {
            let src_off = y * row_bytes;
            let dst_off = (y * dst_pitch) as u64;
            let result = unsafe {
                cuMemcpyHtoD_v2(
                    (dst.data as CUdeviceptr).wrapping_add(dst_off),
                    host_data.as_ptr().add(src_off) as *const c_void,
                    row_bytes,
                )
            };
            if result != CUresult::CUDA_SUCCESS {
                return Err(format!("cuMemcpyHtoD row {y} failed: {:?}", result));
            }
        }
    }

    Ok(())
}

#[cfg(gpu_backend = "cuda")]
fn download_cuda(
    gpu: &GpuContext,
    src: &GpuBuffer,
    width: u32,
    height: u32,
    bpp: u32,
) -> Result<Vec<u8>, String> {
    use cudarc::driver::sys::{
        cuCtxSetCurrent, cuMemcpyDtoH_v2, CUcontext, CUdeviceptr, CUresult,
    };

    let ctx = gpu.device as CUcontext;
    unsafe { cuCtxSetCurrent(ctx) };

    let row_bytes = (width * bpp) as usize;
    let src_pitch = src.pitch_px as usize * bpp as usize;
    let total = row_bytes * height as usize;
    let mut out = vec![0u8; total];

    if src_pitch == row_bytes {
        let result = unsafe {
            cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, src.data as CUdeviceptr, total)
        };
        if result != CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemcpyDtoH failed: {:?}", result));
        }
    } else {
        for y in 0..height as usize {
            let dst_off = y * row_bytes;
            let src_off = (y * src_pitch) as u64;
            let result = unsafe {
                cuMemcpyDtoH_v2(
                    out.as_mut_ptr().add(dst_off) as *mut c_void,
                    (src.data as CUdeviceptr).wrapping_add(src_off),
                    row_bytes,
                )
            };
            if result != CUresult::CUDA_SUCCESS {
                return Err(format!("cuMemcpyDtoH row {y} failed: {:?}", result));
            }
        }
    }

    Ok(out)
}
