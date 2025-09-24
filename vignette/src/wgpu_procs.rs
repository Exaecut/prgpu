use after_effects::{log, Parameters};
use core::f32;
use crevice::std140::{AsStd140, Vec2, Vec4};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use wgpu::*;

use crate::Params;

const WG_SIZE: u32 = 8; // Workgroup size for compute shaders
const MAX_BUFFER_STATES: usize = 15; // Limit number of BufferState instances

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, AsStd140)]
#[repr(C)]
pub struct KernelParams {
	time: f32,
	debug: u32,
	tint: Vec4,
	darken_strength: f32,
	blur_strength: f32,
	anchor: Vec2,
	scale: Vec2,
	noise: f32,
	noise_timeoffset: f32,
	noise_size: f32,
	darken_min: f32,
	darken_max: f32,
	blur_quality: f32,
	blur_radius: f32,
	blur_min: f32,
	blur_max: f32,
	is_premiere: u32,
}

impl KernelParams {
	pub fn from_params(params: &mut Parameters<Params>, time: f32, in_data: after_effects::InData, downsamples: (f32, f32)) -> Result<Self, after_effects::Error> {
		let debug_flag = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		let tint_color = params.get(Params::Tint)?.as_color()?.value();
		let tint = Vec4 {
			x: tint_color.red as f32 / 255.0,
			y: tint_color.green as f32 / 255.0,
			z: tint_color.blue as f32 / 255.0,
			w: tint_color.alpha as f32 / 255.0,
		};

		Ok(Self {
			time,
			debug: debug_flag,
			tint,
			anchor: Vec2 {
				x: params.get(Params::Anchor)?.as_point()?.value().0,
				y: params.get(Params::Anchor)?.as_point()?.value().1,
			},
			scale: Vec2 {
				x: params.get(Params::ScaleX)?.as_float_slider()?.value() as f32,
				y: params.get(Params::ScaleY)?.as_float_slider()?.value() as f32,
			},
			noise_timeoffset: params.get(Params::NoiseTimeOffset)?.as_angle()?.value(),
			darken_strength: params.get(Params::DarkenStrength)?.as_float_slider()?.value() as f32,
			blur_strength: params.get(Params::BlurStrength)?.as_float_slider()?.value() as f32,
			noise: params.get(Params::Noise)?.as_float_slider()?.value() as f32,
			noise_size: params.get(Params::NoiseSize)?.as_float_slider()?.value() as f32,
			darken_min: params.get(Params::DarkenMin)?.as_float_slider()?.value() as f32,
			darken_max: params.get(Params::DarkenMax)?.as_float_slider()?.value() as f32,
			blur_quality: params.get(Params::BlurQuality)?.as_float_slider()?.value() as f32,
			blur_radius: params.get(Params::BlurRadius)?.as_float_slider()?.value() as f32,
			blur_min: params.get(Params::BlurMin)?.as_float_slider()?.value() as f32,
			blur_max: params.get(Params::BlurMax)?.as_float_slider()?.value() as f32,
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
#[allow(dead_code)]
pub struct BufferState {
	pub in_texture: Texture,
	pub out_texture: Texture,
	pub out_view: TextureView,
	pub main_params: Buffer,
	pub main_bind_group: BindGroup,
	pub staging_buffer: Buffer,
	pub padded_out_stride: u32,
	pub last_access: AtomicUsize,
}

#[derive(Debug)]
pub struct WgpuProcessing<T: Sized> {
	pub device: Device,
	pub queue: Queue,
	pub main_pipeline: ComputePipeline,
	pub mip_count: AtomicUsize,
	pub state: RwLock<HashMap<BufferKey, BufferState>>,
	_marker: std::marker::PhantomData<T>,
}

#[allow(dead_code)]
pub enum ProcShaderSource<'a> {
	Wgsl(String),
	SpirV(&'a [u8]),
}

impl<T: Sized> WgpuProcessing<T> {
	pub fn new(main_shader: ProcShaderSource) -> Self {
		log::info!("Initializing wgpu processing");
		let power_preference = PowerPreference::from_env().unwrap_or(PowerPreference::HighPerformance);
		let mut instance_desc = InstanceDescriptor::default();

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

		let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
			label: None,
			required_features: adapter.features(),
			required_limits: adapter.limits(),
			memory_hints: wgpu::MemoryHints::Performance,
			trace: wgpu::Trace::Off,
		}))
		.unwrap();

		let info = adapter.get_info();
		log::info!("Using {} ({}) - {:#?}.", info.name, info.device, info.backend);

		let main_shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("main_shader"),
			source: match main_shader {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Owned(wgsl)),
			},
		});

		let main_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
						sample_type: TextureSampleType::Float { filterable: true },
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
						format: TextureFormat::Rgba8Unorm,
						view_dimension: TextureViewDimension::D2,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 3,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Sampler(SamplerBindingType::Filtering),
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 4,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Texture {
						sample_type: TextureSampleType::Float { filterable: true },
						view_dimension: TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
			],
			label: Some("combine_bind_group_layout"),
		});

		let main_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("combine_pipeline_layout"),
			bind_group_layouts: &[&main_bind_group_layout],
			push_constant_ranges: &[],
		});

		let main_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &main_shader_module,
			entry_point: Some("main"),
			label: Some("combine_pipeline"),
			layout: Some(&main_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		Self {
			device,
			queue,
			main_pipeline,
			mip_count: AtomicUsize::new(8),
			state: RwLock::new(HashMap::new()),
			_marker: std::marker::PhantomData,
		}
	}

	pub fn create_buffers(&self, in_size: (usize, usize, usize), out_size: (usize, usize, usize)) -> BufferState {
		let mip_count = self.mip_count.load(std::sync::atomic::Ordering::SeqCst) as u32;
		let (iw, ih, _) = (in_size.0 as u32, in_size.1 as u32, in_size.2 as u32);
		let (mut ow, mut oh, os) = (out_size.0 as u32, out_size.1 as u32, out_size.2 as u32);

		let limits = self.device.limits();
		if ow > limits.max_texture_dimension_2d || oh > limits.max_texture_dimension_2d {
			log::error!("Texture size exceeds GPU limits: {}x{} > {}", ow, oh, limits.max_texture_dimension_2d);
			// Fallback: Cap dimensions to max allowed
			ow = ow.min(limits.max_texture_dimension_2d);
			oh = oh.min(limits.max_texture_dimension_2d);
		}

		let align = COPY_BYTES_PER_ROW_ALIGNMENT;
		let padding = (align - os % align) % align;
		let padded_out_stride = os + padding;
		let staging_size = padded_out_stride * oh;

		let in_texture = self.device.create_texture(&TextureDescriptor {
			label: Some("in_texture"),
			size: Extent3d {
				width: iw,
				height: ih,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: TextureFormat::Rgba8Unorm,
			usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
			view_formats: &[],
		});

		let in_view = in_texture.create_view(&TextureViewDescriptor::default());

		let out_texture = self.device.create_texture(&TextureDescriptor {
			label: None,
			size: Extent3d {
				width: ow,
				height: oh,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: TextureFormat::Rgba8Unorm,
			usage: TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC,
			view_formats: &[],
		});

		let out_view = out_texture.create_view(&TextureViewDescriptor::default());

		let staging_buffer = self.device.create_buffer(&BufferDescriptor {
			size: staging_size as u64,
			usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
			label: None,
			mapped_at_creation: false,
		});

		let main_params = self.device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<T>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: Some("main_params"),
			mapped_at_creation: false,
		});

		let sampler = self.device.create_sampler(&SamplerDescriptor {
			label: None,
			mag_filter: FilterMode::Linear,
			min_filter: FilterMode::Linear,
			mipmap_filter: FilterMode::Linear,
			lod_min_clamp: 0.0,
			lod_max_clamp: (mip_count - 1) as f32,
			address_mode_u: AddressMode::ClampToBorder,
			address_mode_v: AddressMode::ClampToBorder,
			address_mode_w: AddressMode::ClampToBorder,
			border_color: Some(SamplerBorderColor::TransparentBlack),
			..Default::default()
		});

		let main_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
			label: Some("main_bind_group"),
			layout: &self.main_pipeline.get_bind_group_layout(0),
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: main_params.as_entire_binding(),
				},
				BindGroupEntry {
					binding: 1,
					resource: BindingResource::TextureView(&in_view),
				},
				BindGroupEntry {
					binding: 2,
					resource: BindingResource::TextureView(&out_view),
				},
				BindGroupEntry {
					binding: 3,
					resource: BindingResource::Sampler(&sampler),
				},
				BindGroupEntry {
					binding: 4,
					resource: BindingResource::TextureView(&in_view),
				},
			],
		});

		log::info!("Creating buffers {in_size:?} {out_size:?}, thread: {:?}", std::thread::current().id());

		BufferState {
			in_texture,
			out_texture,
			main_params,
			main_bind_group,
			out_view,
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
			if lock.len() >= MAX_BUFFER_STATES {
				// Remove least recently used BufferState
				if let Some(lru_key) = lock
					.iter()
					.min_by_key(|(_, v)| v.last_access.load(std::sync::atomic::Ordering::SeqCst))
					.map(|(k, _)| *k)
				{
					lock.with_upgraded(|x| {
						x.remove(&lru_key);
						log::info!("Evicted BufferState for key: {:?}", lru_key);
					});
				}
			}
			lock.with_upgraded(|x| {
				x.insert(key, self.create_buffers(in_size, out_size));
			});
		}
		let state = lock.get(&key).unwrap();
		state.last_access.store(Self::timestamp(), std::sync::atomic::Ordering::SeqCst);
		lock
	}

	fn run_main_pass(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &Std140KernelParams, out_size: (usize, usize, usize)) {
		self.queue.write_buffer(&state.main_params, 0, bytemuck::cast_slice(&[*params]));
		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("main_pass"),
			timestamp_writes: None,
		});

		cpass.set_pipeline(&self.main_pipeline);
		cpass.set_bind_group(0, &state.main_bind_group, &[]);

		let out_w = out_size.0 as u32;
		let out_h = out_size.1 as u32;
		cpass.dispatch_workgroups(out_w.div_ceil(WG_SIZE), out_h.div_ceil(WG_SIZE), 1);
	}

	#[allow(clippy::too_many_arguments)]
	pub fn run_compute(
		&self,
		params: &Std140KernelParams,
		in_size: (usize, usize, usize),
		out_size: (usize, usize, usize),
		in_buffer: &[u8],
		out_buffer: &mut [u8],
	) -> bool {
		let key = BufferKey { in_size, out_size };
		let lock = self.get_buffer_for_thread(in_size, out_size);
		let state = lock.get(&key).unwrap();

		let (ow, oh, _) = (out_size.0 as u32, out_size.1 as u32, out_size.2 as u32);

		let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor { label: None });

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

		{
			self.run_main_pass(&mut encoder, state, params, out_size);
		}

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
				width: ow,
				height: oh,
				depth_or_array_layers: 1,
			},
		);

		self.queue.submit(Some(encoder.finish()));

		let buffer_slice = state.staging_buffer.slice(..);
		let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
		buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

		let _ = self.device.poll(PollType::Wait);

		if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
			let out_stride = out_size.2;
			let data = buffer_slice.get_mapped_range();
			if state.padded_out_stride == out_stride as u32 {
				out_buffer[..oh as usize * out_stride].copy_from_slice(data.as_ref());
			} else {
				data.as_ref()
					.chunks(state.padded_out_stride as usize)
					.zip(out_buffer.chunks_mut(out_stride))
					.for_each(|(src, dest)| {
						dest.copy_from_slice(&src[0..out_stride]);
					});
			}
			drop(data);
			state.staging_buffer.unmap();
		} else {
			log::error!("failed to run compute on wgpu!");
			return false;
		}
		true
	}
}
