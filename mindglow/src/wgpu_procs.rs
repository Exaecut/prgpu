use after_effects::{log, Parameters};
use core::f32;
use crevice::std140::{AsStd140, Vec2};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use wgpu::*;

use crate::{utils, Params};

const WG_SIZE: u32 = 8; // Workgroup size for compute shaders
const MAX_BUFFER_STATES: usize = 15; // Limit number of BufferState instances

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(4))]
pub struct DownsampleParams {
	pub time: f32,
	pub debug: u32,
	strength: f32,
	threshold: f32,
	threshold_smoothness: f32,
	pub is_premiere: u32,
}

impl DownsampleParams {
	pub fn from_params(params: &mut Parameters<Params>, time: f32, in_data: after_effects::InData) -> Result<Self, after_effects::Error> {
		let debug_flag = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		let threshold = params.get(Params::Threshold)?.as_float_slider()?.value() as f32;
		let threshold_smoothness = params.get(Params::ThresholdSmoothness)?.as_float_slider()?.value() as f32;

		Ok(Self {
			time,
			debug: debug_flag,
			strength: params.get(Params::Strength)?.as_float_slider()?.value() as f32,
			threshold,
			threshold_smoothness,
			is_premiere: in_data.is_premiere() as u8 as u32,
		})
	}
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(4))]
pub struct UpsampleParams {
	pub time: f32,
	pub debug: u32,
	pub threshold: f32,
	pub threshold_smoothness: f32,
	pub is_premiere: u32,
}

impl UpsampleParams {
	pub fn from_params(params: &mut Parameters<Params>, time: f32, in_data: after_effects::InData) -> Result<Self, after_effects::Error> {
		let debug_flag = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		Ok(Self {
			time,
			debug: debug_flag,
			threshold: params.get(Params::Threshold)?.as_float_slider()?.value() as f32,
			threshold_smoothness: params.get(Params::ThresholdSmoothness)?.as_float_slider()?.value() as f32,
			is_premiere: in_data.is_premiere() as u8 as u32,
		})
	}
}

#[derive(Debug, Clone, Copy, AsStd140)]
#[repr(C)]
pub struct KernelParams {
	time: f32,
	bloom_strength: f32,
	radius: f32,
	real_radius: f32,
	downsampling_factor: Vec2,
	tint_color_r: f32,
	tint_color_g: f32,
	tint_color_b: f32,
	chromatic_aberration: f32,
	flicker: u32,
	flicker_frequency: f32,
	flicker_randomness: f32,
	flicker_bias: f32,
	debug: u32,
	is_premiere: u32,
	preview_layer: u32,
}

impl KernelParams {
	pub fn from_params(
		params: &mut Parameters<Params>,
		time: f32,
		downsampling: (f32, f32),
		in_data: after_effects::InData,
		real_radius: f32,
	) -> Result<Self, after_effects::Error> {
		let strength = params.get(Params::Strength)?.as_float_slider()?.value() as f32;

		let debug_flag = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		let tint_color = params.get(Params::TintColor)?.as_color()?.value();
		let tint_color = (tint_color.red as f32 / 255.0, tint_color.green as f32 / 255.0, tint_color.blue as f32 / 255.0);

		let radius = params.get(Params::Radius)?.as_float_slider()?.value() as f32;

		Ok(Self {
			time,
			bloom_strength: strength,
			radius,
			real_radius,
			downsampling_factor: Vec2 {
				x: downsampling.0,
				y: downsampling.1,
			},
			tint_color_r: tint_color.0,
			tint_color_g: tint_color.1,
			tint_color_b: tint_color.2,
			chromatic_aberration: params.get(Params::ChromaticAberration)?.as_float_slider()?.value() as f32,
			flicker: params.get(Params::Flicker)?.as_checkbox()?.value() as u8 as u32,
			flicker_frequency: params.get(Params::FlickerFrequency)?.as_float_slider()?.value() as f32,
			flicker_randomness: params.get(Params::FlickerRandomness)?.as_float_slider()?.value() as f32,
			flicker_bias: params.get(Params::FlickerBias)?.as_float_slider()?.value() as f32,
			debug: debug_flag,
			is_premiere: in_data.is_premiere() as u8 as u32,
			preview_layer: params.get(Params::PreviewLayer)?.as_popup()?.value() as u32,
		})
	}
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlurParams {
	pub is_horizontal: u32, // 1 for horizontal, 0 for vertical
	pub radius: f32,        // Blur radius from parameters
	pub debug: u32,         // Debug flag
}

impl BlurParams {
	pub fn from_params(params: &mut Parameters<Params>) -> Result<Self, after_effects::Error> {
		let radius = params.get(Params::Radius)?.as_float_slider()?.value() as f32;
		let debug = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		Ok(Self { is_horizontal: 1, radius, debug })
	}
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct DownsampleConstants {
	current_mip: u32,
	user_brightness_factor: f32,
}

#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
pub struct BufferKey {
	in_size: (usize, usize, usize),
	out_size: (usize, usize, usize),
	mip_length: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct BufferState {
	pub in_texture: Texture,
	pub downsample_texture: Texture,
	pub downsample_mipmap_views: Vec<TextureView>,
	pub downsample_bind_groups: Vec<BindGroup>,
	pub upsample_bind_groups: Vec<BindGroup>,
	pub bloom_texture: Texture,
	pub bloom_mipmap_views: Vec<TextureView>,
	pub out_texture: Texture,
	pub out_view: TextureView,
	pub downsample_params: Buffer,
	pub upsample_params: Buffer,
	pub combine_params: Buffer,
	pub combine_bind_group: BindGroup,
	pub copy_bind_group: BindGroup,
	pub staging_buffer: Buffer,
	pub base_width: u32,
	pub base_height: u32,
	pub out_width: u32,
	pub out_height: u32,
	pub padded_out_stride: u32,
	pub last_access: AtomicUsize,
	pub blurred_texture: Texture,
	pub blurred_mip_views: Vec<TextureView>,
	pub temp_mip_textures: Vec<Texture>,
	pub temp_mip_views: Vec<TextureView>,
	pub hblur_params: Buffer,
	pub vblur_params: Buffer,
	pub blur_horizontal_bind_groups: Vec<BindGroup>,
	pub blur_vertical_bind_groups: Vec<BindGroup>,
}

#[derive(Debug)]
pub struct WgpuProcessing<T: Sized> {
	pub device: Device,
	pub queue: Queue,
	pub copy_pipeline: ComputePipeline,
	pub downsample_pipeline: ComputePipeline,
	pub upsample_pipeline: ComputePipeline,
	pub combine_pipeline: ComputePipeline,
	pub blur_pipeline: ComputePipeline,
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
	pub fn new(
		downsample_shader: ProcShaderSource,
		upsample_shader: ProcShaderSource,
		combine_shader: ProcShaderSource,
		blur_shader: ProcShaderSource,
		copy_shader: ProcShaderSource,
	) -> Self {
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

		let copy_shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("copy_to_16f_shader"),
			source: match copy_shader {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Owned(wgsl)),
			},
		});

		let copy_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
					ty: BindingType::StorageTexture {
						access: StorageTextureAccess::WriteOnly,
						format: TextureFormat::Rgba16Float,
						view_dimension: TextureViewDimension::D2,
					},
					count: None,
				},
			],
			label: Some("copy_to_16f_layout"),
		});

		let copy_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			label: Some("copy_to_16f_pipeline"),
			layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
				label: Some("copy_to_16f_layout"),
				bind_group_layouts: &[&copy_layout],
				push_constant_ranges: &[PushConstantRange {
					range: 0..(std::mem::size_of::<u32>() * 2) as _,
					stages: ShaderStages::COMPUTE,
				}],
			})),
			module: &copy_shader_module,
			entry_point: Some("main"),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		let downsample_shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("downsample_shader"),
			source: match downsample_shader {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Owned(wgsl)),
			},
		});

		let downsample_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			entries: &[
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: BufferSize::new(std::mem::size_of::<DownsampleParams>() as _),
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
						format: TextureFormat::Rgba16Float,
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
			],
			label: Some("downsample_bind_group_layout"),
		});

		let downsample_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("downsample_pipeline_layout"),
			bind_group_layouts: &[&downsample_bind_group_layout],
			push_constant_ranges: &[PushConstantRange {
				range: 0..(std::mem::size_of::<DownsampleConstants>()) as _,
				stages: ShaderStages::COMPUTE,
			}],
		});

		let downsample_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &downsample_shader_module,
			entry_point: Some("main"),
			label: Some("downsample_pipeline"),
			layout: Some(&downsample_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		let upsample_shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("upsample_shader"),
			source: match upsample_shader {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Owned(wgsl)),
			},
		});

		let upsample_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			entries: &[
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: BufferSize::new(std::mem::size_of::<UpsampleParams>() as _),
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
					ty: BindingType::Texture {
						sample_type: TextureSampleType::Float { filterable: true },
						view_dimension: TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 3,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::StorageTexture {
						access: StorageTextureAccess::ReadWrite,
						format: TextureFormat::Rgba16Float,
						view_dimension: TextureViewDimension::D2,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 4,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Sampler(SamplerBindingType::Filtering),
					count: None,
				},
			],
			label: Some("upsample_bind_group_layout"),
		});

		let upsample_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("upsample_pipeline_layout"),
			bind_group_layouts: &[&upsample_bind_group_layout],
			push_constant_ranges: &[PushConstantRange {
				stages: ShaderStages::COMPUTE,
				range: 0..std::mem::size_of::<f32>() as _,
			}],
		});

		let upsample_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &upsample_shader_module,
			entry_point: Some("main"),
			label: Some("upsample_pipeline"),
			layout: Some(&upsample_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		let combine_shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("combine_shader"),
			source: match combine_shader {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Owned(wgsl)),
			},
		});

		let combine_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
					ty: BindingType::Texture {
						sample_type: TextureSampleType::Float { filterable: true },
						view_dimension: TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 3,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::StorageTexture {
						access: StorageTextureAccess::ReadWrite,
						format: TextureFormat::Rgba8Unorm,
						view_dimension: TextureViewDimension::D2,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 4,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Sampler(SamplerBindingType::Filtering),
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 5,
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

		let combine_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("combine_pipeline_layout"),
			bind_group_layouts: &[&combine_bind_group_layout],
			push_constant_ranges: &[],
		});

		let combine_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &combine_shader_module,
			entry_point: Some("main"),
			label: Some("combine_pipeline"),
			layout: Some(&combine_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		let blur_shader_module = device.create_shader_module(ShaderModuleDescriptor {
			label: Some("blur_shader"),
			source: match blur_shader {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Owned(wgsl)),
			},
		});

		let blur_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			entries: &[
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: BufferSize::new(std::mem::size_of::<BlurParams>() as _),
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
					ty: BindingType::Texture {
						sample_type: TextureSampleType::Float { filterable: true },
						view_dimension: TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 3,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Texture {
						sample_type: TextureSampleType::Float { filterable: true },
						view_dimension: TextureViewDimension::D2,
						multisampled: false,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 4,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::StorageTexture {
						access: StorageTextureAccess::WriteOnly,
						format: TextureFormat::Rgba16Float,
						view_dimension: TextureViewDimension::D2,
					},
					count: None,
				},
				BindGroupLayoutEntry {
					binding: 5,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Sampler(SamplerBindingType::Filtering),
					count: None,
				},
			],
			label: Some("blur_bind_group_layout"),
		});

		let blur_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("blur_pipeline_layout"),
			bind_group_layouts: &[&blur_bind_group_layout],
			push_constant_ranges: &[PushConstantRange {
				stages: ShaderStages::COMPUTE,
				range: 0..std::mem::size_of::<f32>() as u32,
			}],
		});

		let blur_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &blur_shader_module,
			entry_point: Some("main"),
			label: Some("blur_pipeline"),
			layout: Some(&blur_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		Self {
			device,
			queue,
			copy_pipeline,
			downsample_pipeline,
			upsample_pipeline,
			combine_pipeline,
			blur_pipeline,
			mip_count: AtomicUsize::new(8),
			state: RwLock::new(HashMap::new()),
			_marker: std::marker::PhantomData,
		}
	}

	pub fn create_buffers(&self, in_size: (usize, usize, usize), out_size: (usize, usize, usize)) -> BufferState {
		let mip_count = self.mip_count.load(std::sync::atomic::Ordering::SeqCst) as u32;
		let (iw, ih, _) = (in_size.0 as u32, in_size.1 as u32, in_size.2 as u32);
		let (ow, oh, os) = (out_size.0 as u32, out_size.1 as u32, out_size.2 as u32);

		let limits = self.device.limits();
		if ow > limits.max_texture_dimension_2d || oh > limits.max_texture_dimension_2d {
			log::error!("Texture size exceeds GPU limits: {}x{} > {}", ow, oh, limits.max_texture_dimension_2d);
			// Fallback: Cap dimensions to max allowed
			let ow = ow.min(limits.max_texture_dimension_2d);
			let oh = oh.min(limits.max_texture_dimension_2d);
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

		let downsample_texture = self.device.create_texture(&TextureDescriptor {
			label: Some("downsample_texture"),
			size: Extent3d {
				width: ow,
				height: oh,
				depth_or_array_layers: 1,
			},
			mip_level_count: mip_count,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: TextureFormat::Rgba16Float,
			usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
			view_formats: &[],
		});

		let downsample_mip_views = (0..mip_count)
			.map(|mip| {
				downsample_texture.create_view(&TextureViewDescriptor {
					label: Some(format!("downsample_mipmap_{}", mip).as_str()),
					format: Some(TextureFormat::Rgba16Float),
					aspect: TextureAspect::All,
					base_mip_level: mip,
					mip_level_count: Some(1),
					array_layer_count: Some(1),
					..Default::default()
				})
			})
			.collect::<Vec<_>>();

		let bloom_texture = self.device.create_texture(&TextureDescriptor {
			label: Some("bloom_texture"),
			size: Extent3d {
				width: ow,
				height: oh,
				depth_or_array_layers: 1,
			},
			mip_level_count: mip_count,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: TextureFormat::Rgba16Float,
			usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
			view_formats: &[],
		});

		let bloom_mip_views = (0..mip_count)
			.map(|mip| {
				bloom_texture.create_view(&TextureViewDescriptor {
					label: Some(format!("bloom_mipmap_{}", mip).as_str()),
					format: Some(TextureFormat::Rgba16Float),
					aspect: TextureAspect::All,
					base_mip_level: mip,
					mip_level_count: Some(1),
					array_layer_count: Some(1),
					..Default::default()
				})
			})
			.collect::<Vec<_>>();

		let blurred_texture = self.device.create_texture(&TextureDescriptor {
			label: Some("blurred_texture"),
			size: Extent3d {
				width: ow,
				height: oh,
				depth_or_array_layers: 1,
			},
			mip_level_count: mip_count,
			sample_count: 1,
			dimension: TextureDimension::D2,
			format: TextureFormat::Rgba16Float,
			usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
			view_formats: &[],
		});

		let blurred_mip_views = (0..mip_count)
			.map(|mip| {
				blurred_texture.create_view(&TextureViewDescriptor {
					label: Some(format!("blurred_mipmap_{}", mip).as_str()),
					format: Some(TextureFormat::Rgba16Float),
					aspect: TextureAspect::All,
					base_mip_level: mip,
					mip_level_count: Some(1),
					array_layer_count: Some(1),
					..Default::default()
				})
			})
			.collect::<Vec<_>>();

		let temp_mip_textures = (0..mip_count)
			.map(|i| {
				let w = (ow >> i).max(1);
				let h = (oh >> i).max(1);
				self.device.create_texture(&TextureDescriptor {
					label: Some(format!("temp_mip_texture_{}", i).as_str()),
					size: Extent3d {
						width: w,
						height: h,
						depth_or_array_layers: 1,
					},
					mip_level_count: 1,
					sample_count: 1,
					dimension: TextureDimension::D2,
					format: TextureFormat::Rgba16Float,
					usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC | TextureUsages::COPY_DST,
					view_formats: &[],
				})
			})
			.collect::<Vec<_>>();

		let temp_mip_views = temp_mip_textures
			.iter()
			.map(|tex| tex.create_view(&TextureViewDescriptor::default()))
			.collect::<Vec<_>>();

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

		let downsample_params = self.device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<DownsampleParams>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: Some("downsample_params"),
			mapped_at_creation: false,
		});

		let upsample_params = self.device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<UpsampleParams>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: Some("upsample_params"),
			mapped_at_creation: false,
		});

		let combine_params = self.device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<T>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: Some("main_params"),
			mapped_at_creation: false,
		});

		let hblur_params = self.device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<BlurParams>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: Some("hblur_params"),
			mapped_at_creation: false,
		});

		let vblur_params = self.device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<BlurParams>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: Some("vblur_params"),
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

		let copy_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
			layout: &self.copy_pipeline.get_bind_group_layout(0),
			label: Some("copy_bind_group"),
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: BindingResource::TextureView(&in_view),
				},
				BindGroupEntry {
					binding: 1,
					resource: BindingResource::TextureView(&downsample_mip_views[0]),
				},
			],
		});

		let downsample_bind_groups = (0..(mip_count as usize) - 1)
			.map(|i| {
				self.device.create_bind_group(&BindGroupDescriptor {
					label: Some(format!("downsample_bind_group_{}/{}", i, i + 1).as_str()),
					layout: &self.downsample_pipeline.get_bind_group_layout(0),
					entries: &[
						BindGroupEntry {
							binding: 0,
							resource: downsample_params.as_entire_binding(),
						},
						BindGroupEntry {
							binding: 1,
							resource: BindingResource::TextureView(&downsample_mip_views[i]),
						},
						BindGroupEntry {
							binding: 2,
							resource: BindingResource::TextureView(&downsample_mip_views[i + 1]),
						},
						BindGroupEntry {
							binding: 3,
							resource: BindingResource::Sampler(&sampler),
						},
					],
				})
			})
			.collect::<Vec<_>>();

		let blur_horizontal_bind_groups = (0..mip_count as usize)
			.map(|i| {
				self.device.create_bind_group(&BindGroupDescriptor {
					label: Some(format!("blur_horizontal_bind_group_mip_{}", i).as_str()),
					layout: &self.blur_pipeline.get_bind_group_layout(0),
					entries: &[
						BindGroupEntry {
							binding: 0,
							resource: hblur_params.as_entire_binding(),
						},
						BindGroupEntry {
							binding: 1,
							resource: BindingResource::TextureView(&downsample_mip_views[i]),
						},
						BindGroupEntry {
							binding: 2,
							resource: BindingResource::TextureView(&downsample_mip_views[i]),
						},
						BindGroupEntry {
							binding: 3,
							resource: BindingResource::TextureView(&in_view),
						},
						BindGroupEntry {
							binding: 4,
							resource: BindingResource::TextureView(&temp_mip_views[i]),
						},
						BindGroupEntry {
							binding: 5,
							resource: BindingResource::Sampler(&sampler),
						},
					],
				})
			})
			.collect::<Vec<_>>();

		let blur_vertical_bind_groups = (0..mip_count as usize)
			.map(|i| {
				self.device.create_bind_group(&BindGroupDescriptor {
					label: Some(format!("blur_vertical_bind_group_mip_{}", i).as_str()),
					layout: &self.blur_pipeline.get_bind_group_layout(0),
					entries: &[
						BindGroupEntry {
							binding: 0,
							resource: vblur_params.as_entire_binding(),
						},
						BindGroupEntry {
							binding: 1,
							resource: BindingResource::TextureView(&downsample_mip_views[i]),
						},
						BindGroupEntry {
							binding: 2,
							resource: BindingResource::TextureView(&temp_mip_views[i]),
						},
						BindGroupEntry {
							binding: 3,
							resource: BindingResource::TextureView(&in_view),
						},
						BindGroupEntry {
							binding: 4,
							resource: BindingResource::TextureView(&blurred_mip_views[i]),
						},
						BindGroupEntry {
							binding: 5,
							resource: BindingResource::Sampler(&sampler),
						},
					],
				})
			})
			.collect::<Vec<_>>();

		let upsample_bind_groups = (0..(mip_count - 1) as usize)
			.map(|i| {
				self.device.create_bind_group(&BindGroupDescriptor {
					label: Some(format!("upsample_bind_group_{}/{}", i, i + 1).as_str()),
					layout: &self.upsample_pipeline.get_bind_group_layout(0),
					entries: &[
						BindGroupEntry {
							binding: 0,
							resource: upsample_params.as_entire_binding(),
						},
						BindGroupEntry {
							binding: 1,
							resource: BindingResource::TextureView(&bloom_mip_views[i + 1]),
						},
						BindGroupEntry {
							binding: 2,
							resource: BindingResource::TextureView(&blurred_mip_views[i]),
						},
						BindGroupEntry {
							binding: 3,
							resource: BindingResource::TextureView(&bloom_mip_views[i]),
						},
						BindGroupEntry {
							binding: 4,
							resource: BindingResource::Sampler(&sampler),
						},
					],
				})
			})
			.collect::<Vec<_>>();

		let combine_bind_group = self.device.create_bind_group(&BindGroupDescriptor {
			label: Some("combine_bind_group"),
			layout: &self.combine_pipeline.get_bind_group_layout(0),
			entries: &[
				BindGroupEntry {
					binding: 0,
					resource: combine_params.as_entire_binding(),
				},
				BindGroupEntry {
					binding: 1,
					resource: BindingResource::TextureView(&in_view),
				},
				BindGroupEntry {
					binding: 2,
					resource: BindingResource::TextureView(&bloom_mip_views[0]),
				},
				BindGroupEntry {
					binding: 3,
					resource: BindingResource::TextureView(&out_view),
				},
				BindGroupEntry {
					binding: 4,
					resource: BindingResource::Sampler(&sampler),
				},
				BindGroupEntry {
					binding: 5,
					resource: BindingResource::TextureView(&bloom_mip_views[0]),
				},
			],
		});

		log::info!("Creating buffers {in_size:?} {out_size:?}, thread: {:?}", std::thread::current().id());

		BufferState {
			in_texture,
			downsample_texture,
			downsample_mipmap_views: downsample_mip_views,
			downsample_bind_groups,
			upsample_bind_groups,
			bloom_texture,
			bloom_mipmap_views: bloom_mip_views,
			out_texture,
			out_view,
			downsample_params,
			upsample_params,
			combine_params,
			combine_bind_group,
			copy_bind_group,
			staging_buffer,
			base_width: iw,
			base_height: ih,
			out_width: ow,
			out_height: oh,
			padded_out_stride,
			last_access: AtomicUsize::new(Self::timestamp()),
			blurred_texture,
			blurred_mip_views,
			temp_mip_textures,
			temp_mip_views,
			hblur_params,
			vblur_params,
			blur_horizontal_bind_groups,
			blur_vertical_bind_groups,
		}
	}

	fn timestamp() -> usize {
		std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as usize
	}

	fn get_buffer_for_thread(
		&self,
		in_size: (usize, usize, usize),
		out_size: (usize, usize, usize),
		mip_length: usize,
	) -> parking_lot::lock_api::RwLockUpgradableReadGuard<'_, parking_lot::RawRwLock, HashMap<BufferKey, BufferState>> {
		let key = BufferKey { in_size, out_size, mip_length };
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

	fn run_downsample(&self, encoder: &mut CommandEncoder, state: &BufferState, downsample_params: &DownsampleParams, effect_params: &Parameters<Params>) {
		self.queue.write_buffer(&state.downsample_params, 0, bytemuck::cast_slice(&[*downsample_params]));
		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("downsample_pass"),
			timestamp_writes: None,
		});

		cpass.set_pipeline(&self.downsample_pipeline);

		state.downsample_bind_groups.iter().enumerate().for_each(|(i, bind_group)| {
			let dst_w = (state.out_width >> (i as u32 + 1)).max(1);
			let dst_h = (state.out_height >> (i as u32 + 1)).max(1);
			cpass.set_bind_group(0, bind_group, &[]);

			let layer_scale_size = effect_params.get(Params::LayerSizeIndex(i)).unwrap().as_float_slider().unwrap().value() as f32;
			cpass.set_push_constants(
				0,
				bytemuck::bytes_of(&[DownsampleConstants {
					current_mip: i as u32,
					user_brightness_factor: layer_scale_size,
				}]),
			);

			cpass.dispatch_workgroups(dst_w.div_ceil(WG_SIZE), dst_h.div_ceil(WG_SIZE), 1);
		});
	}

	fn run_horizontalblur_pass(&self, encoder: &mut CommandEncoder, state: &BufferState, blur_params: &BlurParams) {
		self.queue.write_buffer(
			&state.hblur_params,
			0,
			bytemuck::cast_slice(&[BlurParams {
				is_horizontal: 1,
				radius: blur_params.radius,
				debug: blur_params.debug,
			}]),
		);

		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("horizontal_blur_pass"),
			timestamp_writes: None,
		});

		cpass.set_pipeline(&self.blur_pipeline);

		for i in 0..state.blur_horizontal_bind_groups.len() {
			let bind_group = &state.blur_horizontal_bind_groups[i];
			let w = (state.out_width >> i as u32).max(1);
			let h = (state.out_height >> i as u32).max(1);
			cpass.set_bind_group(0, bind_group, &[]);
			cpass.set_push_constants(0, &((i + 1) as u32).to_le_bytes());
			cpass.dispatch_workgroups(w.div_ceil(WG_SIZE), h.div_ceil(WG_SIZE), 1);
		}
	}

	fn run_verticalblur_pass(&self, encoder: &mut CommandEncoder, state: &BufferState, blur_params: &BlurParams) {
		self.queue.write_buffer(
			&state.vblur_params,
			0,
			bytemuck::cast_slice(&[BlurParams {
				is_horizontal: 0,
				radius: blur_params.radius,
				debug: blur_params.debug,
			}]),
		);

		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("vertical_blur_pass"),
			timestamp_writes: None,
		});

		cpass.set_pipeline(&self.blur_pipeline);

		for i in 0..state.blur_vertical_bind_groups.len() {
			let bind_group = &state.blur_vertical_bind_groups[i];
			let w = (state.out_width >> i as u32).max(1);
			let h = (state.out_height >> i as u32).max(1);

			let radius_multiplier = 0.1 * f32::powf(f32::exp(1.0), 1.151 * (i + 1) as f32);
			let adjusted_radius = radius_multiplier * blur_params.radius;

			log::info!("Radius multiplier: {}, adjusted radius: {}", radius_multiplier, adjusted_radius);

			cpass.set_bind_group(0, bind_group, &[]);
			cpass.set_push_constants(0, &((i + 1) as u32).to_le_bytes());
			cpass.dispatch_workgroups(w.div_ceil(WG_SIZE), h.div_ceil(WG_SIZE), 1);
		}
	}

	fn copy_lowest_mip_to_bloom(&self, encoder: &mut CommandEncoder, state: &BufferState) {
		let mip_count = self.mip_count.load(std::sync::atomic::Ordering::SeqCst) as u32;
		encoder.copy_texture_to_texture(
			TexelCopyTextureInfo {
				texture: &state.blurred_texture,
				mip_level: mip_count - 1,
				origin: Origin3d::ZERO,
				aspect: TextureAspect::All,
			},
			TexelCopyTextureInfo {
				texture: &state.bloom_texture,
				mip_level: mip_count - 1,
				origin: Origin3d::ZERO,
				aspect: TextureAspect::All,
			},
			Extent3d {
				width: (state.out_width >> (mip_count - 1)).max(1),
				height: (state.out_height >> (mip_count - 1)).max(1),
				depth_or_array_layers: 1,
			},
		);
	}

	fn run_upsample(&self, encoder: &mut CommandEncoder, state: &BufferState, effect_params: &Parameters<Params>, upsample_params: &UpsampleParams) {
		self.queue.write_buffer(&state.upsample_params, 0, bytemuck::cast_slice(&[*upsample_params]));
		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("upsample_pass"),
			timestamp_writes: None,
		});

		cpass.set_pipeline(&self.upsample_pipeline);

		for i in (0..(self.mip_count.load(std::sync::atomic::Ordering::SeqCst) - 1)).rev() {
			let bind_group = &state.upsample_bind_groups[i];
			let dst_w = (state.out_width >> (i as u32)).max(1);
			let dst_h = (state.out_height >> (i as u32)).max(1);

			cpass.set_push_constants(0, &((i + 1) as u32).to_le_bytes());
			cpass.set_bind_group(0, bind_group, &[]);
			cpass.dispatch_workgroups(dst_w.div_ceil(WG_SIZE), dst_h.div_ceil(WG_SIZE), 1);
		}
	}

	fn run_combine(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &KernelParams, out_size: (usize, usize, usize)) {
		self.queue.write_buffer(&state.combine_params, 0, params.as_std140().as_bytes());
		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("combine_pass"),
			timestamp_writes: None,
		});

		cpass.set_pipeline(&self.combine_pipeline);
		cpass.set_bind_group(0, &state.combine_bind_group, &[]);

		let out_w = out_size.0 as u32;
		let out_h = out_size.1 as u32;
		cpass.dispatch_workgroups(out_w.div_ceil(WG_SIZE), out_h.div_ceil(WG_SIZE), 1);
	}

	#[allow(clippy::too_many_arguments)]
	pub fn run_compute(
		&self,
		effect_params: &Parameters<Params>,
		params: &KernelParams,
		downsample_params: &DownsampleParams,
		upsample_params: &UpsampleParams,
		blur_params: &BlurParams,
		mip_length: usize,
		in_size: (usize, usize, usize),
		out_size: (usize, usize, usize),
		in_buffer: &[u8],
		out_buffer: &mut [u8],
	) -> bool {
		self.mip_count
			.store(mip_length.min(utils::calculate_mipmap_levels(in_size) as usize), std::sync::atomic::Ordering::SeqCst);
		let key = BufferKey { in_size, out_size, mip_length };
		let lock = self.get_buffer_for_thread(in_size, out_size, mip_length);
		let state = lock.get(&key).unwrap();

		let (iw, ih, _) = (in_size.0 as u32, in_size.1 as u32, in_size.2 as u32);
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
			let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
				label: Some("copy_to_16f_pass"),
				timestamp_writes: None,
			});

			cpass.set_pipeline(&self.copy_pipeline);
			let offsets = [(ow - iw) / 2, (oh - ih) / 2];
			cpass.set_push_constants(0, bytemuck::cast_slice(&offsets));
			cpass.set_bind_group(0, &state.copy_bind_group, &[]);
			cpass.dispatch_workgroups(state.out_width.div_ceil(WG_SIZE), state.out_height.div_ceil(WG_SIZE), 1);
		}

		{
			self.run_downsample(&mut encoder, state, downsample_params, effect_params);
			self.run_horizontalblur_pass(&mut encoder, state, blur_params);
			self.run_verticalblur_pass(&mut encoder, state, blur_params);
			self.copy_lowest_mip_to_bloom(&mut encoder, state);
			self.run_upsample(&mut encoder, state, effect_params, upsample_params);
			self.run_combine(&mut encoder, state, params, out_size);
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
