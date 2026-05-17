//! Premiere Pro host simulation.
//!
//! Builds mock FFI objects (GpuFilterData, RenderParams, PPixHand, suites)
//! so that `PremiereGPU::render()` can be called from `cargo test` exactly as
//! Premiere Pro would call it.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::ptr;

use crate::testing::context::GpuContext;
use premiere as pr;
use premiere::sys as pr_sys;

// ── Public ergonomic API ──────────────────────────────────────────────────

/// Pixel format constants matching Adobe `PrPixelFormat` values.
pub mod pixel_format {
    use premiere as pr;
    /// 4×float BGRA (same layout as GpuBgra4444_32f, 16 bpp).
    /// Uses the non-GPU variant so `PixelFormat::from()` handles the raw value
    /// correctly across all platforms.
    pub const BGRA8: pr::PixelFormat = pr::PixelFormat::Bgra4444_32f;
    pub const BGRA32F: pr::PixelFormat = pr::PixelFormat::Bgra4444_32f;
}

/// An ergonomic parameter value. Maps to the underlying `pr::Param` / `PrParam`
/// encoding that `from_gpu()` expects.
#[derive(Clone, Debug)]
pub enum ParamValue {
    Float(f32),
    Bool(bool),
    Int32(i32),
    /// RGBA colour. Premiere stores PF_Pixel as a big-endian-packed u64:
    /// `x[0]=A, x[2]=R, x[4]=G, x[6]=B`.
    Color { r: u8, g: u8, b: u8, a: u8 },
}

impl ParamValue {
    pub fn float(v: f32) -> Self { Self::Float(v) }
    pub fn bool(v: bool) -> Self { Self::Bool(v) }
    pub fn color(r: u8, g: u8, b: u8, a: u8) -> Self { Self::Color { r, g, b, a } }
}

impl From<ParamValue> for pr_sys::PrParam {
    fn from(v: ParamValue) -> pr_sys::PrParam {
        match v {
            ParamValue::Float(f) => pr::Param::Float32(f).into(),
            ParamValue::Bool(b) => pr::Param::Bool(b).into(),
            ParamValue::Int32(i) => pr::Param::Int32(i).into(),
            ParamValue::Color { r, g, b, a } => {
                let u = ((a as u64) << 56) | ((r as u64) << 40) | ((g as u64) << 24) | ((b as u64) << 8);
                pr::Param::Int64(u as i64).into()
            }
        }
    }
}

/// Maps from an effect's `Params` enum discriminants to `ParamValue`s.
pub type ParamMap = Vec<(usize, pr_sys::PrParam)>;

/// Builds a `HostContext` with named parameters.
pub struct HostBuilder<F: pr::GpuFilter, P: Into<usize> + Copy> {
    gpu_filter: F,
    width: u32,
    height: u32,
    input_data: Vec<u8>,
    params: Vec<(usize, pr_sys::PrParam)>,
    time_seconds: f32,
    pixel_format: pr::PixelFormat,
    _phantom: std::marker::PhantomData<P>,
}

impl<F: pr::GpuFilter, P: Into<usize> + Copy> HostBuilder<F, P> {
    pub fn new(gpu_filter: F, input_data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            gpu_filter,
            width,
            height,
            input_data,
            params: Vec::new(),
            time_seconds: 0.0,
            pixel_format: pixel_format::BGRA8,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn param(mut self, index: P, value: ParamValue) -> Self {
        self.params.push((index.into(), value.into()));
        self
    }

    pub fn pixel_format(mut self, fmt: pr::PixelFormat) -> Self { self.pixel_format = fmt; self }
    pub fn time_seconds(mut self, t: f32) -> Self { self.time_seconds = t; self }

    pub fn build(self) -> Result<HostContext<F>, String> {
        HostContext::create(
            self.gpu_filter,
            self.width,
            self.height,
            &self.input_data,
            self.params,
            self.time_seconds,
            self.pixel_format,
        )
    }
}

// ── Mock state held across FFI calls ──────────────────────────────────────

struct MockState {
    param_values: Vec<(usize, pr_sys::PrParam)>,
    gpu_device: *mut c_void,
    gpu_queue: *mut c_void,
}

fn make_ppix(gpu_data: *mut c_void, width: u32, height: u32, bpp: u32, pixel_format: pr::PixelFormat) -> pr_sys::PPix {
    let mut ppix: pr_sys::PPix = unsafe { std::mem::zeroed() };
    ppix.pix = gpu_data;
    ppix.rowbytes = (width * bpp) as i32;
    ppix.bitsperpixel = (bpp * 8) as i32;
    ppix.bounds = pr_sys::prRect { top: 0, left: 0, bottom: height as i32, right: width as i32 };
    // Store the C constant value, not the Rust enum discriminant.
    let raw_pf: pr_sys::PrPixelFormat = pixel_format.into();
    ppix.reserved[0] = raw_pf as usize as *mut c_void;
    ppix.reserved[1] = 1usize as *mut c_void;
    ppix.reserved[2] = 1usize as *mut c_void;
    ppix.reserved[3] = 1usize as *mut c_void;
    ppix
}

fn ppix_to_hand(ppix: Box<pr_sys::PPix>) -> pr_sys::PPixHand {
    let raw = Box::into_raw(ppix);
    Box::into_raw(Box::new(raw))
}

// ── Vtable registry ───────────────────────────────────────────────────────

std::thread_local! {
    static MOCK_VTABLES: std::cell::RefCell<MockVtables> = const {
        std::cell::RefCell::new(MockVtables {
            gpu_device: ptr::null(),
            ppix: ptr::null(),
            ppix2: ptr::null(),
            video_segment: ptr::null(),
        })
    };
}

struct MockVtables {
    gpu_device: *const c_void,
    ppix: *const c_void,
    ppix2: *const c_void,
    video_segment: *const c_void,
}

unsafe extern "C" fn mock_acquire_suite(
    name: *const c_char,
    _version: c_int,
    suite: *mut *const c_void,
) -> i32 {
    let name_str = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    let vt = MOCK_VTABLES.with(|cell| cell.borrow().gpu_device);
    let ptr = if name_str.contains("GPU Device") {
        vt
    } else if name_str.contains("PPix 2") {
        MOCK_VTABLES.with(|cell| cell.borrow().ppix2)
    } else if name_str.contains("PPix") {
        MOCK_VTABLES.with(|cell| cell.borrow().ppix)
    } else if name_str.contains("Video Segment") {
        MOCK_VTABLES.with(|cell| cell.borrow().video_segment)
    } else {
        ptr::null()
    };
    if ptr.is_null() { return -1; }
    unsafe { *suite = ptr };
    0
}

unsafe extern "C" fn mock_release_suite(_name: *const c_char, _version: c_int) -> i32 { 0 }

// ── GPUDevice mocks ───────────────────────────────────────────────────────

unsafe extern "C" fn mock_get_gpu_ppix_data(
    ppix_handle: pr_sys::PPixHand,
    out_data: *mut *mut c_void,
) -> i32 {
    if ppix_handle.is_null() || unsafe { *ppix_handle }.is_null() { return -1; }
    unsafe { *out_data = (**ppix_handle).pix };
    0
}

unsafe extern "C" fn mock_gpu_ppix_device_index(
    _ppix_handle: pr_sys::PPixHand,
    out_device_index: *mut u32,
) -> i32 {
    unsafe { *out_device_index = 0 };
    0
}

unsafe extern "C" fn mock_create_gpu_ppix(
    _device_index: u32,
    _pixel_format: pr_sys::PrPixelFormat,
    _width: c_int,
    _height: c_int,
    _par_num: c_int,
    _par_den: c_int,
    _field_type: i32,
    out_ppix_hand: *mut pr_sys::PPixHand,
) -> i32 {
    unsafe { *out_ppix_hand = ptr::null_mut() };
    -1
}

unsafe extern "C" fn mock_get_device_info(
    _suite_version: u32,
    _device_index: u32,
    out_device_info: *mut pr_sys::PrGPUDeviceInfo,
) -> i32 {
    unsafe { *out_device_info = std::mem::zeroed() };
    let state_ptr = MOCK_STATE_PTR.with(|cell| *cell.borrow());
    if !state_ptr.is_null() {
        let state = unsafe { &*state_ptr };
        unsafe {
            (*out_device_info).outDeviceHandle = state.gpu_device;
            (*out_device_info).outCommandQueueHandle = state.gpu_queue;
            (*out_device_info).outContextHandle = state.gpu_device;
        }
    }
    0
}

fn make_gpu_device_vtable() -> Box<pr_sys::PrSDKGPUDeviceSuite> {
    let mut v = unsafe { Box::<pr_sys::PrSDKGPUDeviceSuite>::new_zeroed().assume_init() };
    v.GetGPUPPixData = Some(mock_get_gpu_ppix_data);
    v.GetGPUPPixDeviceIndex = Some(mock_gpu_ppix_device_index);
    v.CreateGPUPPix = Some(mock_create_gpu_ppix);
    v.GetDeviceInfo = Some(mock_get_device_info);
    v
}

// ── PPix mocks ────────────────────────────────────────────────────────────

unsafe extern "C" fn mock_pixel_format(
    ppix_handle: pr_sys::PPixHand,
    out_format: *mut pr_sys::PrPixelFormat,
) -> i32 {
    if ppix_handle.is_null() || unsafe { *ppix_handle }.is_null() { return -1; }
    // The pixel format is stored in reserved[0] during PPix construction.
    // We read it back directly — no enum conversion needed since the raw
    // discriminant was stored there.
    unsafe { *out_format = (**ppix_handle).reserved[0] as usize as pr_sys::PrPixelFormat };
    0
}

unsafe extern "C" fn mock_row_bytes(
    ppix_handle: pr_sys::PPixHand,
    out_row_bytes: *mut i32,
) -> i32 {
    if ppix_handle.is_null() || unsafe { *ppix_handle }.is_null() { return -1; }
    unsafe { *out_row_bytes = (**ppix_handle).rowbytes };
    0
}

unsafe extern "C" fn mock_bounds(
    ppix_handle: pr_sys::PPixHand,
    inout_rect: *mut pr_sys::prRect,
) -> i32 {
    if ppix_handle.is_null() || unsafe { *ppix_handle }.is_null() { return -1; }
    unsafe { *inout_rect = (**ppix_handle).bounds };
    0
}

unsafe extern "C" fn mock_pixel_aspect_ratio(
    ppix_handle: pr_sys::PPixHand,
    out_num: *mut u32,
    out_den: *mut u32,
) -> i32 {
    if ppix_handle.is_null() || unsafe { *ppix_handle }.is_null() { return -1; }
    unsafe {
        *out_num = (**ppix_handle).reserved[1] as u32;
        *out_den = (**ppix_handle).reserved[2] as u32;
    }
    0
}

unsafe extern "C" fn mock_dispose(_ppix_handle: pr_sys::PPixHand) -> i32 { 0 }

fn make_ppix_vtable() -> Box<pr_sys::PrSDKPPixSuite> {
    let mut v = unsafe { Box::<pr_sys::PrSDKPPixSuite>::new_zeroed().assume_init() };
    v.GetPixelFormat = Some(mock_pixel_format);
    v.GetRowBytes = Some(mock_row_bytes);
    v.GetBounds = Some(mock_bounds);
    v.GetPixelAspectRatio = Some(mock_pixel_aspect_ratio);
    v.Dispose = Some(mock_dispose);
    v
}

// ── PPix2 mocks ───────────────────────────────────────────────────────────

unsafe extern "C" fn mock_field_order(
    ppix_handle: pr_sys::PPixHand,
    out_field_type: *mut i32,
) -> i32 {
    if ppix_handle.is_null() || unsafe { *ppix_handle }.is_null() { return -1; }
    unsafe { *out_field_type = (**ppix_handle).reserved[3] as i32 };
    0
}

fn make_ppix2_vtable() -> Box<pr_sys::PrSDKPPix2Suite> {
    let mut v = unsafe { Box::<pr_sys::PrSDKPPix2Suite>::new_zeroed().assume_init() };
    v.GetFieldOrder = Some(mock_field_order);
    v
}

// ── VideoSegment mocks ────────────────────────────────────────────────────

unsafe extern "C" fn mock_get_param(
    _node_id: i32,
    index: i32,
    _time: i64,
    out_param: *mut pr_sys::PrParam,
) -> i32 {
    let state_ptr = MOCK_STATE_PTR.with(|cell| *cell.borrow());
    if state_ptr.is_null() { return -1; }
    let state = unsafe { &*state_ptr };
    for (idx, val) in &state.param_values {
        if *idx == index as usize {
            unsafe { *out_param = *val };
            return 0;
        }
    }
    let raw: pr_sys::PrParam = pr::Param::Float32(0.0).into();
    unsafe { *out_param = raw };
    0
}

unsafe extern "C" fn mock_get_node_property(
    _node_id: i32,
    _key: *const c_char,
    out_value: *mut *mut c_char,
) -> i32 {
    // Property values are returned as C strings. 30 s in Adobe ticks.
    static DURATION: &[u8] = b"1800000\0";
    unsafe { *out_value = DURATION.as_ptr() as *mut c_char };
    0
}

std::thread_local! {
    static MOCK_STATE_PTR: std::cell::RefCell<*const MockState> = const { std::cell::RefCell::new(ptr::null()) };
}

fn make_video_segment_vtable() -> Box<pr_sys::PrSDKVideoSegmentSuite> {
    let mut v = unsafe { Box::<pr_sys::PrSDKVideoSegmentSuite>::new_zeroed().assume_init() };
    v.GetParam = Some(mock_get_param);
    v.GetNodeProperty = Some(mock_get_node_property);
    v
}

// ── Filter construction ───────────────────────────────────────────────────

pub struct MockFilter {
    instance: *mut pr_sys::PrGPUFilterInstance,
    gpu_device_vtable: *mut pr_sys::PrSDKGPUDeviceSuite,
    ppix_vtable: *mut pr_sys::PrSDKPPixSuite,
    ppix2_vtable: *mut pr_sys::PrSDKPPix2Suite,
    video_vtable: *mut pr_sys::PrSDKVideoSegmentSuite,
    sp_basic: *mut pr_sys::SPBasicSuite,
    state: *mut MockState,
    _pica_guard: pr::PicaBasicSuite,
    gpu_info: pr_sys::PrGPUDeviceInfo,
    input_hand: pr_sys::PPixHand,
    output_hand: pr_sys::PPixHand,
}

impl MockFilter {
    fn build(
        input_pix: *mut c_void,
        output_pix: *mut c_void,
        width: u32,
        height: u32,
        bpp: u32,
        pixel_format: pr::PixelFormat,
        param_values: Vec<(usize, pr_sys::PrParam)>,
        gpu_device: *mut c_void,
        gpu_queue: *mut c_void,
    ) -> Result<Self, String> {
        let input_ppix = make_ppix(input_pix, width, height, bpp, pixel_format);
        let output_ppix = make_ppix(output_pix, width, height, bpp, pixel_format);

        let input_hand = ppix_to_hand(Box::new(input_ppix));
        let output_hand = ppix_to_hand(Box::new(output_ppix));

        let gpu_device_vtable = Box::into_raw(make_gpu_device_vtable());
        let ppix_vtable = Box::into_raw(make_ppix_vtable());
        let ppix2_vtable = Box::into_raw(make_ppix2_vtable());
        let video_vtable = Box::into_raw(make_video_segment_vtable());

        let sp_basic = Box::into_raw(Box::new(pr_sys::SPBasicSuite {
            AcquireSuite: Some(mock_acquire_suite),
            ReleaseSuite: Some(mock_release_suite),
            IsEqual: None,
            AllocateBlock: None,
            FreeBlock: None,
            ReallocateBlock: None,
            Undefined: None,
        }));

        MOCK_VTABLES.with(|cell| {
            *cell.borrow_mut() = MockVtables {
                gpu_device: gpu_device_vtable as *const c_void,
                ppix: ppix_vtable as *const c_void,
                ppix2: ppix2_vtable as *const c_void,
                video_segment: video_vtable as *const c_void,
            };
        });

        let _pica_guard = pr::PicaBasicSuite::from_sp_basic_suite_raw(sp_basic);

        let state = Box::into_raw(Box::new(MockState { param_values, gpu_device, gpu_queue }));
        MOCK_STATE_PTR.with(|cell| *cell.borrow_mut() = state);

        let gpu_info = pr_sys::PrGPUDeviceInfo {
            outDeviceFramework: 0,
            outMeetsMinimumRequirementsForAcceleration: 1,
            outPlatformHandle: ptr::null_mut(),
            outDeviceHandle: gpu_device,
            outContextHandle: gpu_device,
            outCommandQueueHandle: gpu_queue,
            outOffscreenOpenGLContextHandle: ptr::null_mut(),
            outOffscreenOpenGLDeviceHandle: ptr::null_mut(),
        };

        let instance = Box::into_raw(Box::new(pr_sys::PrGPUFilterInstance {
            piSuites: ptr::null_mut(),
            inDeviceIndex: 0,
            inTimelineID: 0,
            inNodeID: 0,
            ioPrivatePluginData: ptr::null_mut(),
            outIsRealtime: 0,
        }));

        Ok(MockFilter {
            instance,
            gpu_device_vtable,
            ppix_vtable,
            ppix2_vtable,
            video_vtable,
            sp_basic,
            state,
            _pica_guard,
            gpu_info,
            input_hand,
            output_hand,
        })
    }

    pub fn filter_data(&self) -> Result<pr::GpuFilterData, String> {
        let gpu_device_suite = pr::suites::GPUDevice::new().map_err(|e| format!("GPUDevice: {e:?}"))?;
        let ppix_suite = pr::suites::PPix::new().map_err(|e| format!("PPix: {e:?}"))?;
        let ppix2_suite = pr::suites::PPix2::new().map_err(|e| format!("PPix2: {e:?}"))?;
        let video_suite = pr::suites::VideoSegment::new().map_err(|e| format!("VideoSegment: {e:?}"))?;

        let gpu_image = unsafe { std::mem::ManuallyDrop::new(pr::suites::GPUImageProcessing::new().unwrap_or_else(|_| std::mem::zeroed())) };
        let mem_mgr   = unsafe { std::mem::ManuallyDrop::new(pr::suites::MemoryManager::new().unwrap_or_else(|_| std::mem::zeroed())) };

        Ok(pr::GpuFilterData {
            instance_ptr: self.instance,
            gpu_device_suite,
            gpu_image_processing_suite: unsafe { std::ptr::read(&*gpu_image) },
            memory_manager_suite: unsafe { std::ptr::read(&*mem_mgr) },
            ppix_suite,
            ppix2_suite,
            video_segment_suite: video_suite,
            gpu_info: self.gpu_info,
        })
    }

    pub fn input_hand(&self) -> pr_sys::PPixHand { self.input_hand }
    pub fn output_hand(&self) -> pr_sys::PPixHand { self.output_hand }
}

impl Drop for MockFilter {
    fn drop(&mut self) {
        unsafe {
            if !self.instance.is_null() { drop(Box::from_raw(self.instance)); }
            if !self.gpu_device_vtable.is_null() { drop(Box::from_raw(self.gpu_device_vtable)); }
            if !self.ppix_vtable.is_null() { drop(Box::from_raw(self.ppix_vtable)); }
            if !self.ppix2_vtable.is_null() { drop(Box::from_raw(self.ppix2_vtable)); }
            if !self.video_vtable.is_null() { drop(Box::from_raw(self.video_vtable)); }
            if !self.sp_basic.is_null() { drop(Box::from_raw(self.sp_basic)); }
            if !self.state.is_null() { drop(Box::from_raw(self.state)); }
            if !self.input_hand.is_null() {
                let ppix = *self.input_hand;
                if !ppix.is_null() { drop(Box::from_raw(ppix)); }
                drop(Box::from_raw(self.input_hand));
            }
            if !self.output_hand.is_null() {
                let ppix = *self.output_hand;
                if !ppix.is_null() { drop(Box::from_raw(ppix)); }
                drop(Box::from_raw(self.output_hand));
            }
        }
    }
}

// ── HostContext ────────────────────────────────────────────────────────────

pub struct HostContext<F: pr::GpuFilter> {
    pub gpu: GpuContext,
    gpu_filter: F,
    mock: MockFilter,
    width: u32,
    height: u32,
    bpp: u32,
    render_params_heap: *const pr_sys::PrGPUFilterRenderParams,
}

impl<F: pr::GpuFilter> HostContext<F> {
    pub fn create(
        gpu_filter: F,
        width: u32,
        height: u32,
        input_data: &[u8],
        param_values: Vec<(usize, pr_sys::PrParam)>,
        time_seconds: f32,
        pixel_format: pr::PixelFormat,
    ) -> Result<Self, String> {
        let input_bpp = 4;
        let gpu_bpp = 16; // GpuBgra4444_32f → 4 floats → 16 bytes
        let expected = (width as u64) * (height as u64) * (input_bpp as u64);
        if input_data.len() as u64 != expected {
            return Err("input data size mismatch".into());
        }

        // Convert BGRA8 → packed 32f for the GPU path.
        let mut float_data = vec![0u8; (width as usize) * (height as usize) * (gpu_bpp as usize)];
        for i in 0..(width as usize * height as usize) {
            let src = &input_data[i * 4..i * 4 + 4];
            let dst = &mut float_data[i * 16..i * 16 + 16];
            let b = src[0] as f32 / 255.0;
            let g = src[1] as f32 / 255.0;
            let r = src[2] as f32 / 255.0;
            let a = src[3] as f32 / 255.0;
            dst[0..4].copy_from_slice(&b.to_le_bytes());
            dst[4..8].copy_from_slice(&g.to_le_bytes());
            dst[8..12].copy_from_slice(&r.to_le_bytes());
            dst[12..16].copy_from_slice(&a.to_le_bytes());
        }

        let gpu = GpuContext::create()?;

        let (in_buf, out_buf) = gpu.create_io_buffers(width, height, gpu_bpp)?;
        gpu.upload_to_buffer(&in_buf, &float_data, width, height, gpu_bpp)?;

        let mock = MockFilter::build(
            in_buf.data, out_buf.data,
            width, height, gpu_bpp,
            pixel_format, param_values,
            gpu.device, gpu.command_queue,
        )?;

        let render_params_heap = Box::into_raw(Box::new(pr_sys::PrGPUFilterRenderParams {
            inClipTime: (time_seconds * 60_000.0) as i64,
            inSequenceTime: 0,
            inQuality: 0,
            inDownsampleFactorX: 1.0,
            inDownsampleFactorY: 1.0,
            inRenderWidth: width,
            inRenderHeight: height,
            inRenderPARNum: 1,
            inRenderPARDen: 1,
            inRenderFieldType: 1,
            inRenderTicksPerFrame: 1000, // 60_000 ticks / 60 fps = 1000
            inRenderField: 0,
        }));

        Ok(HostContext { gpu, gpu_filter, mock, width, height, bpp: gpu_bpp, render_params_heap })
    }

    pub fn start(&self) -> Result<Vec<u8>, String> {
        let filter_data = self.mock.filter_data()?;
        let render_params = pr::RenderParams::from_raw(self.render_params_heap);

        F::global_init();

        {
            let mut query_index: i32 = 0;
            let _ = self.gpu_filter.get_frame_dependencies(&filter_data, render_params.clone(), &mut query_index);
        }

        let _ = self.gpu_filter.precompute(&filter_data, render_params.clone(), 0, self.mock.input_hand());

        let frames = [self.mock.input_hand()];
        let mut out_hand = self.mock.output_hand();
        self.gpu_filter.render(&filter_data, render_params.clone(), frames.as_ptr(), 1, &mut out_hand)
            .map_err(|e| format!("render: {e:?}"))?;

        let output = self.download_output();

        F::global_destroy();

        output
    }

    fn download_output(&self) -> Result<Vec<u8>, String> {
        let hand = self.mock.output_hand();
        if hand.is_null() { return Err("output PPixHand is null".into()); }
        let ppix = unsafe { *hand };
        if ppix.is_null() { return Err("output PPix* is null".into()); }
        let pix = unsafe { (*ppix).pix };
        if pix.is_null() { return Err("output PPix.pix is null".into()); }
        let float_out = self.gpu.download_raw(pix, self.width, self.width, self.height, self.bpp)?;
        // Convert 32f BGRA → BGRA8
        let mut out = vec![0u8; (self.width as usize) * (self.height as usize) * 4];
        for i in 0..(self.width as usize * self.height as usize) {
            let src = &float_out[i * 16..i * 16 + 16];
            let b = f32::from_le_bytes(src[0..4].try_into().unwrap()).clamp(0.0, 1.0);
            let g = f32::from_le_bytes(src[4..8].try_into().unwrap()).clamp(0.0, 1.0);
            let r = f32::from_le_bytes(src[8..12].try_into().unwrap()).clamp(0.0, 1.0);
            let a = f32::from_le_bytes(src[12..16].try_into().unwrap()).clamp(0.0, 1.0);
            out[i * 4] = (b * 255.0) as u8;
            out[i * 4 + 1] = (g * 255.0) as u8;
            out[i * 4 + 2] = (r * 255.0) as u8;
            out[i * 4 + 3] = (a * 255.0) as u8;
        }
        Ok(out)
    }
}

impl<F: pr::GpuFilter> Drop for HostContext<F> {
    fn drop(&mut self) {
        if !self.render_params_heap.is_null() {
            unsafe { drop(Box::from_raw(self.render_params_heap as *mut pr_sys::PrGPUFilterRenderParams)) };
        }
    }
}
