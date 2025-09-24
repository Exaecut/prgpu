use std::ffi::c_void;

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
    pub outgoing_data: *mut c_void,
    pub incoming_data: *mut c_void,
    pub dest_data: *mut c_void,
    pub outgoing_pitch_px: i32,
    pub incoming_pitch_px: i32,
    pub dest_pitch_px: i32,
    pub width: u32,
    pub height: u32,
    pub is16f: bool,
    pub progress: f32,
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
