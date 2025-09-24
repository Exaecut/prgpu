use after_effects::{log, Parameters};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use wgpu::*;

use crate::{Params, RepeatMode};

#[derive(Debug)]
#[repr(C)]
pub struct KernelParams {
	anchor_point_x: f32,
	anchor_point_y: f32,
	time: f32,
	amplitude: f32,
	frequency: f32,
	h_amplitude: f32,
	v_amplitude: f32,
	h_frequency: f32,
	v_frequency: f32,
	phase: f32,
	seed: u32,
	xframe_size_x: f32,
	xframe_size_y: f32,
	repeat_mode_x: u32,
	repeat_mode_y: u32,
	style: u32,
	clip: u32,
	motion_blur: u32,
	motion_blur_time_offset: f32,
	motion_blur_length: f32,
	motion_blur_samples: i32,
	tilt_amplitude: f32,
	tilt_frequency: f32,
	tilt_phase: f32,
	debug: u32,
	is_premiere: u32,
}

impl KernelParams {
	pub fn from_params(
		params: &mut Parameters<Params>,
		xframe: Option<f32>,
		checkout_result: Option<after_effects::sys::PF_CheckoutResult>,
		downsampling: (f32, f32),
		time: f32,
		in_data: after_effects::InData,
	) -> Result<Self, after_effects::Error> {
		let amplitude = params.get(Params::Amplitude)?.as_float_slider()?.value() as f32;
		let frequency = params.get(Params::Frequency)?.as_float_slider()?.value() as f32;
		let h_amplitude = params.get(Params::HorizontalShakeAmplitude)?.as_float_slider()?.value() as f32;
		let v_amplitude = params.get(Params::VerticalShakeAmplitude)?.as_float_slider()?.value() as f32;

		let style = params.get(Params::Style)?.as_popup()?.value() as u32;

		let motion_blur = params.get(Params::MotionBlur)?.as_checkbox()?.value() as u32;
		let motion_blur_length = (params.get(Params::MotionBlurLength)?.as_float_slider()?.value() / (if style == 2 { 20.0 } else { 1.0 })) as f32;

		let hq_xframe_size = match xframe {
			Some(xframe) => xframe,
			None => amplitude * (h_amplitude + v_amplitude),
		};

		let mut xframe_size = (hq_xframe_size * downsampling.0, hq_xframe_size * downsampling.1);

		let mut anchors = (0.0, 0.0);

		if !in_data.is_premiere() {
			if let Some(checkout_result) = checkout_result {
				xframe_size.0 += checkout_result.max_result_rect.left as f32;
				xframe_size.1 += checkout_result.max_result_rect.top as f32;

				anchors = (
					(checkout_result.max_result_rect.right - checkout_result.max_result_rect.left) as f32 / 2.0,
					(checkout_result.max_result_rect.bottom - checkout_result.max_result_rect.top) as f32 / 2.0,
				);
			}
		} else {
			anchors = (in_data.width() as f32 / 2.0 * downsampling.0, in_data.height() as f32 / 2.0 * downsampling.1);
		}

		let repeat_mode_x = params.get(Params::RepeatModeX)?.as_popup()?.value() as u32;
		let repeat_mode_y = params.get(Params::RepeatModeY)?.as_popup()?.value() as u32;

		let debug_flag = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		Ok(Self {
			anchor_point_x: anchors.0,
			anchor_point_y: anchors.1,
			time,
			amplitude,
			frequency,
			h_amplitude: h_amplitude * downsampling.0,
			v_amplitude: v_amplitude * downsampling.1,
			h_frequency: params.get(Params::HorizontalShakeFrequency)?.as_float_slider()?.value() as f32,
			v_frequency: params.get(Params::VerticalShakeFrequency)?.as_float_slider()?.value() as f32,
			phase: params.get(Params::Phase)?.as_angle()?.value(),
			seed: params.get(Params::Seed)?.as_float_slider()?.value() as u32,
			xframe_size_x: xframe_size.0,
			xframe_size_y: xframe_size.1,
			repeat_mode_x: params.get(Params::RepeatModeX)?.as_popup()?.value() as u32,
			repeat_mode_y: params.get(Params::RepeatModeY)?.as_popup()?.value() as u32,
			style: style,
			clip: !(RepeatMode::from(repeat_mode_x) == RepeatMode::None || RepeatMode::from(repeat_mode_y) == RepeatMode::None) as u32,
			motion_blur,
			motion_blur_time_offset: (time + in_data.local_time_step() as f32 / in_data.time_scale() as f32),
			motion_blur_length,
			motion_blur_samples: params.get(Params::MotionBlurSamples)?.as_float_slider()?.value() as i32,
			tilt_amplitude: params.get(Params::TiltAmplitude)?.as_float_slider()?.value() as f32,
			tilt_frequency: params.get(Params::TiltFrequency)?.as_float_slider()?.value() as f32,
			tilt_phase: params.get(Params::TiltPhase)?.as_angle()?.value(),
			debug: debug_flag,
			is_premiere: in_data.is_premiere() as u8 as u32,
		})
	}
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct BufferKey {
	in_size: (usize, usize, usize),
	out_size: (usize, usize, usize),
}

#[derive(Debug)]
pub struct BufferState {
	pub in_texture: Texture,
	pub out_texture: Texture,
	pub bind_group: BindGroup,
	pub params: Buffer,
	pub staging_buffer: Buffer,
	pub padded_out_stride: u32,
	pub last_access: AtomicUsize,
}

#[derive(Debug)]
pub struct WgpuProcessing<T: Sized> {
	pub device: Device,
	pub queue: Queue,
	pub pipeline: ComputePipeline,
	pub state: RwLock<HashMap<BufferKey, BufferState>>,
	_marker: std::marker::PhantomData<T>,
}

#[allow(dead_code)]
pub enum ProcShaderSource<'a> {
	Wgsl(&'a str),
	SpirV(&'a [u8]),
}

impl<T: Sized> WgpuProcessing<T> {
	pub fn new(shader: ProcShaderSource) -> Self {
		let power_preference = PowerPreference::from_env().unwrap_or(PowerPreference::HighPerformance);
		let mut instance_desc = InstanceDescriptor::default();

		// We can't use the DX12 backend with validation turned on, otherwise it
		// will remove AE's internal D3D12 device:
		// https://learn.microsoft.com/en-us/windows/win32/api/d3d12sdklayers/nf-d3d12sdklayers-id3d12debug-enabledebuglayer
		if instance_desc.backends.contains(Backends::DX12) && instance_desc.flags.contains(InstanceFlags::VALIDATION) {
			instance_desc.backends.remove(Backends::DX12);
			log::info!("Disabling {:?} because {:?} is on", Backends::DX12, InstanceFlags::VALIDATION);
		}

		let instance = Instance::new(&instance_desc);

		let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
			power_preference,
			..Default::default()
		}))
		.unwrap();

		let (device, queue) = pollster::block_on(adapter.request_device(
			&DeviceDescriptor {
				label: None,
				required_features: adapter.features(),
				required_limits: adapter.limits(),
				memory_hints: wgpu::MemoryHints::Performance,
			},
			None,
		))
		.unwrap();

		let info = adapter.get_info();
		log::info!("Using {} ({}) - {:#?}.", info.name, info.device, info.backend);

		let shader = device.create_shader_module(ShaderModuleDescriptor {
			label: None,
			source: match shader {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
			},
		});

		let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			entries: &[
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: BufferSize::new(std::mem::size_of::<T>() as _),
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 1,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Texture {
						sample_type: TextureSampleType::Uint,
						view_dimension: TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 2,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::StorageTexture {
						access: StorageTextureAccess::ReadWrite,
						format: TextureFormat::Rgba8Uint,
						view_dimension: TextureViewDimension::D2,
					},
					count: None,
				},
			],
			label: None,
		});

		let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: None,
			bind_group_layouts: &[&layout],
			push_constant_ranges: &[],
		});

		let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &shader,
			entry_point: Some("main"),
			label: None,
			layout: Some(&pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		Self {
			device,
			queue,
			pipeline,
			_marker: std::marker::PhantomData,
			state: RwLock::new(HashMap::new()),
		}
	}

	pub fn create_buffers(&self, in_size: (usize, usize, usize), out_size: (usize, usize, usize)) -> BufferState {
		let (iw, ih, _) = (in_size.0 as u32, in_size.1 as u32, in_size.2 as u32);
		let (ow, oh, os) = (out_size.0 as u32, out_size.1 as u32, out_size.2 as u32);

		let align = COPY_BYTES_PER_ROW_ALIGNMENT;
		let padding = (align - os % align) % align;
		let padded_out_stride = os + padding;
		let staging_size = padded_out_stride * oh;

		let in_desc = TextureDescriptor {
			label: None,
			size: Extent3d {
				width: iw,
				height: ih,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: TextureFormat::Rgba8Uint,
			usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
			view_formats: &[],
		};
		let out_desc = TextureDescriptor {
			label: None,
			size: Extent3d {
				width: ow,
				height: oh,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: TextureFormat::Rgba8Uint,
			usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
			view_formats: &[],
		};

		let in_texture = self.device.create_texture(&in_desc);
		let out_texture = self.device.create_texture(&out_desc);
		let staging_buffer = self.device.create_buffer(&BufferDescriptor {
			size: staging_size as u64,
			usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
			label: None,
			mapped_at_creation: false,
		});

		let in_view = in_texture.create_view(&TextureViewDescriptor::default());
		let out_view = out_texture.create_view(&TextureViewDescriptor::default());

		let params = self.device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<T>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: None,
			mapped_at_creation: false,
		});

		let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
			label: None,
			layout: &self.pipeline.get_bind_group_layout(0),
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: params.as_entire_binding(),
				},
				BindGroupEntry {
					binding: 1,
					resource: BindingResource::TextureView(&in_view),
				},
				BindGroupEntry {
					binding: 2,
					resource: BindingResource::TextureView(&out_view),
				},
			],
		});

		log::info!("Creating buffers {in_size:?} {out_size:?}, thread: {:?}", std::thread::current().id());

		BufferState {
			in_texture,
			out_texture,
			bind_group,
			params,
			staging_buffer,
			padded_out_stride,
			last_access: AtomicUsize::new(Self::timestamp()),
		}
	}

	fn timestamp() -> usize {
		std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as usize
	}

	fn get_buffer_for_thread(
		&self,
		in_size: (usize, usize, usize),
		out_size: (usize, usize, usize),
	) -> parking_lot::lock_api::RwLockUpgradableReadGuard<'_, parking_lot::RawRwLock, HashMap<BufferKey, BufferState>> {
		let key = BufferKey { in_size, out_size };
		let mut lock = self.state.upgradable_read();
		if !lock.contains_key(&key) {
			lock.with_upgraded(|x| {
				x.insert(key, self.create_buffers(in_size, out_size));
			});
		}
		let state = lock.get(&key).unwrap();
		state.last_access.store(Self::timestamp(), std::sync::atomic::Ordering::Relaxed);
		lock
	}

	pub fn run_compute(&self, params: &T, in_size: (usize, usize, usize), out_size: (usize, usize, usize), in_buffer: &[u8], out_buffer: &mut [u8]) -> bool {
		let key = BufferKey { in_size, out_size };
		let lock = self.get_buffer_for_thread(in_size, out_size);
		let state = lock.get(&key).unwrap();

		let width = out_size.0 as u32;
		let height = out_size.1 as u32;

		let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: None });

		// Write params uniform
		self.queue.write_buffer(&state.params, 0, unsafe {
			std::slice::from_raw_parts(params as *const _ as _, std::mem::size_of::<T>())
		});

		// Write input texture
		self.queue.write_texture(
			state.in_texture.as_image_copy(),
			in_buffer,
			TexelCopyBufferLayout {
				offset: 0,
				bytes_per_row: Some(in_size.2 as u32),
				rows_per_image: None,
			},
			Extent3d {
				width: in_size.0 as u32,
				height: in_size.1 as u32,
				depth_or_array_layers: 1,
			},
		);

		// Run the compute pass
		{
			let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
				label: None,
				timestamp_writes: None,
			});
			cpass.set_pipeline(&self.pipeline);
			cpass.set_bind_group(0, &state.bind_group, &[]);
			cpass.dispatch_workgroups((width as f32 / 16.0).ceil() as u32, (height as f32 / 16.0).ceil() as u32, 1);
		}

		// Copy output texture to buffer that we can read
		encoder.copy_texture_to_buffer(
			TexelCopyTextureInfo {
				texture: &state.out_texture,
				mip_level: 0,
				origin: Origin3d::ZERO,
				aspect: TextureAspect::All,
			},
			TexelCopyBufferInfo {
				buffer: &state.staging_buffer,
				layout: TexelCopyBufferLayout {
					offset: 0,
					bytes_per_row: Some(state.padded_out_stride),
					rows_per_image: None,
				},
			},
			Extent3d {
				width,
				height,
				depth_or_array_layers: 1,
			},
		);

		self.queue.submit(Some(encoder.finish()));

		// Read the output buffer
		let buffer_slice = state.staging_buffer.slice(..);
		let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
		buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

		self.device.poll(Maintain::Wait);

		if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
			let out_stride = out_size.2;

			let data = buffer_slice.get_mapped_range();
			if state.padded_out_stride == out_stride as u32 {
				// Fast path
				out_buffer[..height as usize * out_stride].copy_from_slice(data.as_ref());
			} else {
				data.as_ref()
					.chunks(state.padded_out_stride as usize)
					.zip(out_buffer.chunks_mut(out_stride))
					.for_each(|(src, dest)| {
						dest.copy_from_slice(&src[0..out_stride]);
					});
			}

			// We have to make sure all mapped views are dropped before we unmap the buffer.
			drop(data);
			state.staging_buffer.unmap();
		} else {
			log::error!("failed to run compute on wgpu!");
			return false;
		}
		true
	}
}
