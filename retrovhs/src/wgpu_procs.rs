use after_effects::{log, Parameters};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use wgpu::*;

use crate::Params;

const LAYER_FLAG_DISTORTION: u32 = 1 << 0;
const LAYER_FLAG_COMPRESSION: u32 = 1 << 1;
const LAYER_FLAG_SIGNAL_NOISE: u32 = 1 << 2;
const LAYER_FLAG_FILTER: u32 = 1 << 3;

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SignalNoiseParams {
	time: f32,
	tape_noise_lowfreq_glitch: f32,
	tape_noise_highfreq_glitch: f32,
	tape_noise_horizontal_offset: f32,
	tape_noise_vertical_offset: f32,
	crease_phase_frequency: f32,
	crease_speed: f32,
	crease_height: f32,
	crease_depth: f32,
	crease_intensity: f32,
	crease_noise_frequency: f32,
	crease_stability: f32,
	crease_minimum: f32,
	extremis_noise_height_proportion: f32,
	side_leak_intensity: f32,
	bloom_exposure: f32,
	enabled_layers: u32,
	is_premiere: u32,
}

impl SignalNoiseParams {
	pub fn from_params(params: &mut Parameters<Params>, time: f32, in_data: after_effects::InData) -> Result<Self, after_effects::Error> {
		let mut enabled_layers = 0u32;
		if params.get(Params::Distortion)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_DISTORTION;
		}
		if params.get(Params::Compression)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_COMPRESSION;
		}
		if params.get(Params::SignalNoise)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_SIGNAL_NOISE;
		}
		if params.get(Params::Filter)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_FILTER;
		}

		let is_premiere = in_data.is_premiere() as u8 as u32;

		Ok(Self {
			time,
			tape_noise_lowfreq_glitch: params.get(Params::SNLowfreqGlitch)?.as_float_slider()?.value() as f32,
			tape_noise_highfreq_glitch: params.get(Params::SNHighfreqGlitch)?.as_float_slider()?.value() as f32,
			tape_noise_horizontal_offset: params.get(Params::SNHorizontalOffset)?.as_float_slider()?.value() as f32,
			tape_noise_vertical_offset: params.get(Params::SNVerticalOffset)?.as_float_slider()?.value() as f32,
			crease_phase_frequency: params.get(Params::SNTapeCreasePhaseFreq)?.as_float_slider()?.value() as f32,
			crease_speed: params.get(Params::SNTapeCreaseSpeed)?.as_float_slider()?.value() as f32,
			crease_height: params.get(Params::SNTapeCreaseHeight)?.as_float_slider()?.value() as f32,
			crease_depth: params.get(Params::SNTapeCreaseDepth)?.as_float_slider()?.value() as f32,
			crease_intensity: params.get(Params::SNTapeCreaseIntensity)?.as_float_slider()?.value() as f32,
			crease_noise_frequency: params.get(Params::SNTapeCreaseNoiseFreq)?.as_float_slider()?.value() as f32,
			crease_stability: params.get(Params::SNTapeCreaseStability)?.as_float_slider()?.value() as f32,
			crease_minimum: params.get(Params::SNTapeCreaseMinimum)?.as_float_slider()?.value() as f32,
			extremis_noise_height_proportion: params.get(Params::SNExtremisNoiseHFrac)?.as_float_slider()?.value() as f32,
			side_leak_intensity: params.get(Params::SNBorderLeakIntensity)?.as_float_slider()?.value() as f32,
			bloom_exposure: params.get(Params::SNBloomExposure)?.as_float_slider()?.value() as f32,
			enabled_layers,
			is_premiere,
		})
	}
}

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompressionParams {
	time: f32,
	enabled_layers: u32,
	is_premiere: u32,
}

impl CompressionParams {
	pub fn from_params(params: &mut Parameters<Params>, time: f32, in_data: after_effects::InData) -> Result<Self, after_effects::Error> {
		let mut enabled_layers = 0u32;
		if params.get(Params::Distortion)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_DISTORTION;
		}
		if params.get(Params::Compression)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_COMPRESSION;
		}
		if params.get(Params::SignalNoise)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_SIGNAL_NOISE;
		}
		if params.get(Params::Filter)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_FILTER;
		}

		let is_premiere = in_data.is_premiere() as u8 as u32;

		Ok(Self {
			time,
			enabled_layers,
			is_premiere,
		})
	}
}

#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KernelParams {
	// Time in seconds
	time: f32,
	debug: u32,

	uv_mode: u32, // 0: normal, 1: 4:3
	horizontal_distortion: f32,
	vertical_distortion: f32,
	vignette_strength: f32,
	tint_color_r: u32,
	tint_color_g: u32,
	tint_color_b: u32,
	tint_color_a: u32,
	bloom_exposure: f32,
	pixel_cell_size: f32,
	scanline_hardness: f32,
	pixel_hardness: f32,
	bloom_scanline_hardness: f32,
	bloom_pixel_hardness: f32,
	crt_contrast: f32,
	enabled_layers: u32,
	is_premiere: u32,
}

impl KernelParams {
	pub fn from_params(params: &mut Parameters<Params>, time: f32, _downsampling: (f32, f32), in_data: after_effects::InData) -> Result<Self, after_effects::Error> {
		let debug_flag = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		let mut enabled_layers = 0u32;
		if params.get(Params::Distortion)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_DISTORTION;
		}
		if params.get(Params::Compression)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_COMPRESSION;
		}
		if params.get(Params::SignalNoise)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_SIGNAL_NOISE;
		}
		if params.get(Params::Filter)?.as_checkbox()?.value() {
			enabled_layers |= LAYER_FLAG_FILTER;
		}

		let color = params.get(Params::TintColor)?.as_color()?.value();
		let tint_color = [color.red as u32, color.green as u32, color.blue as u32, color.alpha as u32];

		Ok(Self {
			time,
			debug: debug_flag,

			uv_mode: params.get(Params::DistortionAspectRatio)?.as_popup()?.value() as u32,
			horizontal_distortion: params.get(Params::DistortionHorizontal)?.as_float_slider()?.value() as f32,
			vertical_distortion: params.get(Params::DistortionVertical)?.as_float_slider()?.value() as f32,
			vignette_strength: params.get(Params::DistortionVignetteStrength)?.as_float_slider()?.value() as f32,
			tint_color_r: tint_color[0],
			tint_color_g: tint_color[1],
			tint_color_b: tint_color[2],
			tint_color_a: tint_color[3],
			bloom_exposure: params.get(Params::SNBloomExposure)?.as_float_slider()?.value() as f32,
			pixel_cell_size: params.get(Params::PixelCellSize)?.as_float_slider()?.value() as f32,
			scanline_hardness: -params.get(Params::ScanlineHardness)?.as_float_slider()?.value() as f32,
			pixel_hardness: -params.get(Params::PixelHardness)?.as_float_slider()?.value() as f32,
			bloom_scanline_hardness: -params.get(Params::BloomScanlineHardness)?.as_float_slider()?.value() as f32,
			bloom_pixel_hardness: -params.get(Params::BloomPixelHardness)?.as_float_slider()?.value() as f32,
			crt_contrast: params.get(Params::CRTContrast)?.as_float_slider()?.value() as f32,

			enabled_layers,
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
	pub main_bind_group: BindGroup,
	pub signal_noise_bind_group: BindGroup,
	pub compress_bind_group: BindGroup,
	pub params: Buffer,
	pub signal_noise_params: Buffer,
	pub compress_params: Buffer,
	pub staging_buffer: Buffer,
	pub padded_out_stride: u32,
	pub last_access: AtomicUsize,
}

#[derive(Debug)]
pub struct WgpuProcessing<T: Sized + bytemuck::Pod> {
	pub device: Device,
	pub queue: Queue,
	pub signal_noise_pipeline: ComputePipeline,
	pub compress_pipeline: ComputePipeline,
	pub main_pipeline: ComputePipeline,
	pub state: RwLock<HashMap<BufferKey, BufferState>>,
	_marker: std::marker::PhantomData<T>,
}

#[allow(dead_code)]
pub enum ProcShaderSource<'a> {
	Wgsl(&'a str),
	SpirV(&'a [u8]),
}

impl<T: Sized + bytemuck::Pod> WgpuProcessing<T> {
	pub fn new(main_shader: ProcShaderSource, compress_shader: ProcShaderSource, signal_shader: ProcShaderSource) -> Self {
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

		let signal_noise_shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("signal_noise_shader"),
			source: match signal_shader {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
			},
		});

		let signal_noise_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			entries: &[
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Texture {
						sample_type: TextureSampleType::Float { filterable: true },
						view_dimension: TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 1,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Sampler(SamplerBindingType::Filtering),
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
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: BufferSize::new(std::mem::size_of::<SignalNoiseParams>() as _),
					},
					count: None,
				},
			],
			label: Some("signal_noise_bind_group_layout"),
		});

		let compress_shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("compress_shader"),
			source: match compress_shader {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
			},
		});

		let compress_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			entries: &[
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Texture {
						sample_type: TextureSampleType::Float { filterable: true },
						view_dimension: TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 1,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Sampler(SamplerBindingType::Filtering),
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
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: BufferSize::new(std::mem::size_of::<CompressionParams>() as _),
					},
					count: None,
				},
			],
			label: Some("compress_bind_group_layout"),
		});

		let main_shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("main_shader"),
			source: match main_shader {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Borrowed(wgsl)),
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
			label: Some("main_bind_group_layout"),
		});

		let signal_noise_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("signal_noise_pipeline_layout"),
			bind_group_layouts: &[&signal_noise_bind_group_layout],
			push_constant_ranges: &[],
		});

		let signal_noise_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &signal_noise_shader_module,
			entry_point: Some("main"),
			label: Some("signal_noise_pipeline"),
			layout: Some(&signal_noise_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		let compress_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("compress_pipeline_layout"),
			bind_group_layouts: &[&compress_bind_group_layout],
			push_constant_ranges: &[],
		});

		let compress_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &compress_shader_module,
			entry_point: Some("main"),
			label: Some("compress_pipeline"),
			layout: Some(&compress_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		let main_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("main_pipeline_layout"),
			bind_group_layouts: &[&main_bind_group_layout],
			push_constant_ranges: &[],
		});

		let main_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &main_shader_module,
			entry_point: Some("main"),
			label: Some("main_pipeline"),
			layout: Some(&main_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		Self {
			device,
			queue,
			main_pipeline,
			signal_noise_pipeline,
			compress_pipeline,
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
			usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
			view_formats: &[],
		});

		let signal_noise_texture = self.device.create_texture(&TextureDescriptor {
			label: Some("signal_noise_texture"),
			size: Extent3d {
				width: iw,
				height: ih,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: TextureFormat::Rgba8Unorm,
			usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
			view_formats: &[],
		});

		let compress_texture = self.device.create_texture(&TextureDescriptor {
			label: Some("compress_texture"),
			size: Extent3d {
				width: iw,
				height: ih,
				depth_or_array_layers: 1,
			},
			mip_level_count: 1,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: TextureFormat::Rgba8Unorm,
			usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
			view_formats: &[],
		});

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

		let staging_buffer = self.device.create_buffer(&BufferDescriptor {
			size: staging_size as u64,
			usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
			label: None,
			mapped_at_creation: false,
		});

		let signal_noise_params = self.device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<SignalNoiseParams>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: Some("signal_noise_params"),
			mapped_at_creation: false,
		});

		let compress_params = self.device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<CompressionParams>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: Some("compress_params"),
			mapped_at_creation: false,
		});

		let params = self.device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<T>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: Some("main_params"),
			mapped_at_creation: false,
		});

		let in_view = in_texture.create_view(&TextureViewDescriptor::default());
		let signal_noise_view = signal_noise_texture.create_view(&TextureViewDescriptor {
			label: Some("compress_view"),
			..Default::default()
		});
		let compress_view = compress_texture.create_view(&TextureViewDescriptor {
			label: Some("compress_view"),
			..Default::default()
		});
		let out_view = out_texture.create_view(&TextureViewDescriptor::default());

		let sampler = self.device.create_sampler(&SamplerDescriptor {
			label: None,
			mag_filter: FilterMode::Linear,
			min_filter: FilterMode::Linear,
			mipmap_filter: FilterMode::Linear,
			lod_min_clamp: 0.0,
			lod_max_clamp: 6.0,
			address_mode_u: AddressMode::ClampToBorder,
			address_mode_v: AddressMode::ClampToBorder,
			address_mode_w: AddressMode::ClampToBorder,
			border_color: Some(SamplerBorderColor::TransparentBlack),
			..Default::default()
		});

		let signal_noise_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
			label: None,
			layout: &self.signal_noise_pipeline.get_bind_group_layout(0),
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: BindingResource::TextureView(&in_view),
				},
				BindGroupEntry {
					binding: 1,
					resource: BindingResource::Sampler(&sampler),
				},
				BindGroupEntry {
					binding: 2,
					resource: BindingResource::TextureView(&signal_noise_view),
				},
				BindGroupEntry {
					binding: 3,
					resource: signal_noise_params.as_entire_binding(),
				},
			],
		});

		let compress_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
			label: None,
			layout: &self.compress_pipeline.get_bind_group_layout(0),
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: BindingResource::TextureView(&signal_noise_view),
				},
				BindGroupEntry {
					binding: 1,
					resource: BindingResource::Sampler(&sampler),
				},
				BindGroupEntry {
					binding: 2,
					resource: BindingResource::TextureView(&compress_view),
				},
				BindGroupEntry {
					binding: 3,
					resource: compress_params.as_entire_binding(),
				},
			],
		});

		let main_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
			label: None,
			layout: &self.main_pipeline.get_bind_group_layout(0),
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: params.as_entire_binding(),
				},
				BindGroupEntry {
					binding: 1,
					resource: BindingResource::TextureView(&compress_view),
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
					resource: BindingResource::TextureView(&compress_view),
				},
			],
		});

		log::info!("Creating buffers {in_size:?} {out_size:?}, thread: {:?}", std::thread::current().id());

		BufferState {
			in_texture,
			out_texture,
			main_bind_group,
			signal_noise_bind_group,
			compress_bind_group,
			params,
			signal_noise_params,
			compress_params,
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

	fn run_signal_noise_pass(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &SignalNoiseParams, out_size: (usize, usize, usize)) {
		self.queue.write_buffer(&state.signal_noise_params, 0, bytemuck::cast_slice(&[*params]));
		let width = out_size.0 as u32;
		let height = out_size.1 as u32;

		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: None,
			timestamp_writes: None,
		});
		cpass.set_pipeline(&self.signal_noise_pipeline);
		cpass.set_bind_group(0, &state.signal_noise_bind_group, &[]);
		cpass.dispatch_workgroups((width as f32 / 16.0).ceil() as u32, (height as f32 / 16.0).ceil() as u32, 1);
	}

	fn run_compress_pass(&self, encoder: &mut CommandEncoder, state: &BufferState, compress_params: &CompressionParams, out_size: (usize, usize, usize)) {
		self.queue.write_buffer(&state.compress_params, 0, bytemuck::cast_slice(&[*compress_params]));
		let width = out_size.0 as u32;
		let height = out_size.1 as u32;

		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: None,
			timestamp_writes: None,
		});
		cpass.set_pipeline(&self.compress_pipeline);
		cpass.set_bind_group(0, &state.compress_bind_group, &[]);
		cpass.dispatch_workgroups((width as f32 / 16.0).ceil() as u32, (height as f32 / 16.0).ceil() as u32, 1);
	}

	fn run_main_pass(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &T, out_size: (usize, usize, usize)) {
		self.queue.write_buffer(&state.params, 0, bytemuck::cast_slice(&[*params]));
		let width = out_size.0 as u32;
		let height = out_size.1 as u32;

		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: None,
			timestamp_writes: None,
		});
		cpass.set_pipeline(&self.main_pipeline);
		cpass.set_bind_group(0, &state.main_bind_group, &[]);
		cpass.dispatch_workgroups((width as f32 / 16.0).ceil() as u32, (height as f32 / 16.0).ceil() as u32, 1);
	}

	#[allow(clippy::too_many_arguments)]
	pub fn run_compute(
		&self,
		params: &T,
		signal_noise_params: &SignalNoiseParams,
		compress_params: &CompressionParams,
		in_size: (usize, usize, usize),
		out_size: (usize, usize, usize),
		in_buffer: &[u8],
		out_buffer: &mut [u8],
	) -> bool {
		let key = BufferKey { in_size, out_size };
		let lock = self.get_buffer_for_thread(in_size, out_size);
		let state = lock.get(&key).unwrap();

		let width = out_size.0 as u32;
		let height = out_size.1 as u32;

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

		self.run_signal_noise_pass(&mut encoder, state, signal_noise_params, out_size);
		self.run_compress_pass(&mut encoder, state, compress_params, out_size);
		self.run_main_pass(&mut encoder, state, params, out_size);

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

		let buffer_slice = state.staging_buffer.slice(..);
		let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
		buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

		self.device.poll(Maintain::Wait);

		if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
			let out_stride = out_size.2;
			let data = buffer_slice.get_mapped_range();
			if state.padded_out_stride == out_stride as u32 {
				out_buffer[..height as usize * out_stride].copy_from_slice(data.as_ref());
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
