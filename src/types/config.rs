use std::ffi::c_void;

use premiere::suites::GPUDevice;

use crate::GPURenderProperties;

pub enum DeviceHandleInit<'a> {
    FromPtr(*mut c_void),
    FromSuite((u32, &'a GPUDevice)),
}

#[repr(C)]
pub struct MTLSize {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(unused)]
pub struct Configuration {
    pub device_handle: *mut c_void,
    pub context_handle: Option<*mut c_void>,
    pub command_queue_handle: *mut c_void,
    pub outgoing_data: Option<*mut c_void>,
    pub incoming_data: Option<*mut c_void>,
    pub dest_data: *mut c_void,
    pub outgoing_pitch_px: i32,
    pub incoming_pitch_px: i32,
    pub dest_pitch_px: i32,
    pub width: u32,
    pub height: u32,
    pub is16f: bool,
    pub progress: f32,
}

impl Configuration {
    /// # Safety
    /// `out_frame` must be a valid, non-null GPU frame pointer whose memory remains alive and writable.
    /// `bytes_per_pixel` and `row_bytes` must match the actual pixel format and layout.
    /// No concurrent access or invalid GPU context usage is allowed.
    pub unsafe fn effect(
        render_properties: &GPURenderProperties,
        out_frame: *mut premiere::sys::PPixHand,
    ) -> Result<Self, premiere::Error> {
        let filter = render_properties.get_filter();
        let bytes_per_pixel = render_properties.bytes_per_pixel;

        // Get destination frame data
        let (dest_data, dest_row_bytes) = (
            filter
                .gpu_device_suite
                .gpu_ppix_data(unsafe { *out_frame })?,
            filter.ppix_suite.row_bytes(unsafe { *out_frame })?,
        );

        let dest_pitch_px = dest_row_bytes / bytes_per_pixel;

        let width = render_properties.bounds.width();
        let height = render_properties.bounds.height();

        Ok(Self {
            device_handle: filter.gpu_info.outDeviceHandle,
            context_handle: Some(filter.gpu_info.outContextHandle),
            command_queue_handle: filter.gpu_info.outCommandQueueHandle,
            outgoing_data: None,
            incoming_data: None,
            dest_data,
            outgoing_pitch_px: 0,
            incoming_pitch_px: 0,
            dest_pitch_px,
            width: width as u32,
            height: height as u32,
            is16f: render_properties.half_precision,
            progress: render_properties.progress,
        })
    }

    /// # Safety
    /// `out_frame` must be a valid, non-null GPU frame pointer whose memory remains alive and writable.
    /// `bytes_per_pixel` and `row_bytes` must match the actual pixel format and layout.
    /// No concurrent access or invalid GPU context usage is allowed.
    pub unsafe fn transition(
        render_properties: &GPURenderProperties,
        out_frame: *mut premiere::sys::PPixHand,
    ) -> Result<Self, premiere::Error> {
        let filter = render_properties.get_filter();
        let bytes_per_pixel = render_properties.bytes_per_pixel;

        let (incoming, outgoing) = render_properties.frames;

        // Get incoming frame data
        let (incoming_data, incoming_row_bytes) = (
            Some(filter.gpu_device_suite.gpu_ppix_data(incoming)?),
            filter.ppix_suite.row_bytes(incoming)?,
        );
        let incoming_pitch_px = incoming_row_bytes / bytes_per_pixel;

        // Get outgoing frame data
        let (outgoing_data, outgoing_row_bytes) = (
            Some(filter.gpu_device_suite.gpu_ppix_data(outgoing)?),
            filter.ppix_suite.row_bytes(outgoing)?,
        );
        let outgoing_pitch_px = outgoing_row_bytes / bytes_per_pixel;

        // Get destination frame data
        let (dest_data, dest_row_bytes) = (
            filter
                .gpu_device_suite
                .gpu_ppix_data(unsafe { *out_frame })?,
            filter.ppix_suite.row_bytes(unsafe { *out_frame })?,
        );

        let dest_pitch_px = dest_row_bytes / bytes_per_pixel;

        let width = render_properties.bounds.width();
        let height = render_properties.bounds.height();

        Ok(Self {
            device_handle: filter.gpu_info.outDeviceHandle,
            context_handle: Some(filter.gpu_info.outContextHandle),
            command_queue_handle: filter.gpu_info.outCommandQueueHandle,
            outgoing_data,
            incoming_data,
            dest_data,
            outgoing_pitch_px,
            incoming_pitch_px,
            dest_pitch_px,
            width: width as u32,
            height: height as u32,
            is16f: render_properties.half_precision,
            progress: render_properties.progress,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TransitionParams {
    pub out_pitch: u32,
    pub in_pitch: u32,
    pub dest_pitch: u32,
    pub width: u32,
    pub height: u32,
    pub progress: f32,
}
