use after_effects::{log, Parameters};
use core::f32;
use crevice::std140::{AsStd140, Vec4};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::AtomicUsize;
use wgpu::*;

use crate::Params;

const WG_SIZE: u32 = 8; // Workgroup size for compute shaders
const MAX_BUFFER_STATES: usize = 15; // Limit number of BufferState instances

pub trait ComputeShader<P> {
	fn run(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &P, out_size: (usize, usize, usize));
}

impl<T> ComputeShader<KernelParams> for WgpuProcessing<T> {
	fn run(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &KernelParams, out_size: (usize, usize, usize)) {
		self.run_turbulent_depth(encoder, state, params, out_size);
	}
}

impl<T> ComputeShader<EnergyFluxParams> for WgpuProcessing<T> {
	fn run(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &EnergyFluxParams, out_size: (usize, usize, usize)) {
		self.run_energy_flux(encoder, state, params, out_size);
	}
}

impl<T> ComputeShader<TheSunParams> for WgpuProcessing<T> {
	fn run(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &TheSunParams, out_size: (usize, usize, usize)) {
		self.run_the_sun(encoder, state, params, out_size);
	}
}

impl<T> ComputeShader<VelvetParams> for WgpuProcessing<T> {
	fn run(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &VelvetParams, out_size: (usize, usize, usize)) {
		self.run_sweet_velvet(encoder, state, params, out_size);
	}
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, AsStd140)]
#[repr(C)]
pub struct VelvetParams {
	time: f32,        // Time in seconds
	time_factor: f32, // Multiplier for animation speed
	time_offset: f32, // Added offset to time
	debug: u32,
	is_premiere: u32,

	color: Vec4, // Final color multiplier

	height_cos_divisor: f32,             // divisor for cos(p.y)
	height_base_offset: f32,             // additive offset in height()
	vertical_rot_amplitude_scale: f32,   // divisor for sin(time) in first rot
	vertical_rot_base_offset: f32,       // additive offset in first rot
	horizontal_rot_amplitude_scale: f32, // divisor for sin(time) in second rot
	horizontal_rot_base_offset: f32,     // additive offset in second rot
	horizontal_rot_angle_scale: f32,     // scale divisor after second rot
	ray_steps: u32,                      // number of iterations
	distance_accum_scale: f32,           // multiplier in t accumulation
	fog_density: f32,                    // multiplier in fog denominator
}

impl VelvetParams {
	pub fn from_params(params: &mut Parameters<Params>, time: f32, in_data: after_effects::InData) -> Result<Self, after_effects::Error> {
		let debug_flag = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		let color = params.get(Params::SVColor)?.as_color()?.value();

		Ok(Self {
			time,
			time_factor: params.get(Params::Speed)?.as_float_slider()?.value() as f32,
			time_offset: params.get(Params::TimeOffset)?.as_float_slider()?.value() as f32,
			debug: debug_flag,
			is_premiere: in_data.is_premiere() as u8 as u32,
			color: Vec4 {
				x: color.red as f32 / 255.0,
				y: color.green as f32 / 255.0,
				z: color.blue as f32 / 255.0,
				w: color.alpha as f32 / 255.0,
			},
			height_cos_divisor: params.get(Params::SVHeight)?.as_float_slider()?.value() as f32,
			height_base_offset: params.get(Params::SVHeightOffset)?.as_float_slider()?.value() as f32,
			vertical_rot_amplitude_scale: params.get(Params::SVVertScale)?.as_float_slider()?.value() as f32,
			vertical_rot_base_offset: params.get(Params::SVVertOffset)?.as_float_slider()?.value() as f32,
			horizontal_rot_amplitude_scale: params.get(Params::SVHorizontalScale)?.as_float_slider()?.value() as f32,
			horizontal_rot_base_offset: params.get(Params::SVHorizontalOffset)?.as_float_slider()?.value() as f32,
			horizontal_rot_angle_scale: params.get(Params::SVHorizontalAngle)?.as_float_slider()?.value() as f32,
			ray_steps: params.get(Params::SVIterations)?.as_slider()?.value() as u32,
			distance_accum_scale: params.get(Params::SVAccumulation)?.as_float_slider()?.value() as f32,
			fog_density: params.get(Params::SVFogDensity)?.as_float_slider()?.value() as f32,
		})
	}
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct TheSunParams {
	time: f32,
	time_factor: f32,
	time_offset: f32,
	debug: u32,
	is_premiere: u32,
	iterations: u32,
	warp_base: f32,
}

impl TheSunParams {
	pub fn from_params(params: &mut Parameters<Params>, time: f32, in_data: after_effects::InData) -> Result<Self, after_effects::Error> {
		let debug_flag = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		Ok(Self {
			time,
			time_factor: params.get(Params::Speed)?.as_float_slider()?.value() as f32,
			time_offset: params.get(Params::TimeOffset)?.as_float_slider()?.value() as f32,
			debug: debug_flag,
			is_premiere: in_data.is_premiere() as u32,
			iterations: params.get(Params::TSIterations)?.as_slider()?.value() as u32,
			warp_base: params.get(Params::TSWarpBase)?.as_float_slider()?.value() as f32,
		})
	}
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct KernelParams {
	time: f32, // Time in seconds
	debug: u32,
	is_premiere: u32,
	scale: f32,       // Scaling factor for pattern frequency
	time_factor: f32, // Multiplier for animation speed
	time_offset: f32,
	fractal_repetition: f32,
	twist_frequency: f32,
	twist_amplitude: f32,

	_padding: [u32; 3],

	color1: [f32; 4],
	color2: [f32; 4],
	color3: [f32; 4],
	color4: [f32; 4],
}

impl KernelParams {
	pub fn from_params(params: &mut Parameters<Params>, time: f32, in_data: after_effects::InData, _downsamples: (f32, f32)) -> Result<Self, after_effects::Error> {
		let debug_flag = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		let color_a = params.get(Params::TDColor1)?.as_color()?.value();
		let color_b = params.get(Params::TDColor2)?.as_color()?.value();
		let color_c = params.get(Params::TDColor3)?.as_color()?.value();
		let color_d = params.get(Params::TDColor4)?.as_color()?.value();

		Ok(Self {
			time,
			debug: debug_flag,
			is_premiere: in_data.is_premiere() as u8 as u32,
			scale: params.get(Params::TDDepth)?.as_float_slider()?.value() as f32,
			time_factor: params.get(Params::Speed)?.as_float_slider()?.value() as f32,
			time_offset: params.get(Params::TimeOffset)?.as_float_slider()?.value() as f32,
			fractal_repetition: params.get(Params::TDFractalRep)?.as_float_slider()?.value() as f32,
			twist_frequency: params.get(Params::TDTwistFreq)?.as_float_slider()?.value() as f32,
			twist_amplitude: params.get(Params::TDTwistAmp)?.as_float_slider()?.value() as f32,
			_padding: [0; 3],
			color1: [
				color_a.red as f32 / 255.0,
				color_a.green as f32 / 255.0,
				color_a.blue as f32 / 255.0,
				color_a.alpha as f32 / 255.0,
			],
			color2: [
				color_b.red as f32 / 255.0,
				color_b.green as f32 / 255.0,
				color_b.blue as f32 / 255.0,
				color_b.alpha as f32 / 255.0,
			],
			color3: [
				color_c.red as f32 / 255.0,
				color_c.green as f32 / 255.0,
				color_c.blue as f32 / 255.0,
				color_c.alpha as f32 / 255.0,
			],
			color4: [
				color_d.red as f32 / 255.0,
				color_d.green as f32 / 255.0,
				color_d.blue as f32 / 255.0,
				color_d.alpha as f32 / 255.0,
			],
		})
	}
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct EnergyFluxParams {
	time: f32, // Time in seconds
	debug: u32,
	is_premiere: u32,
	time_factor: f32, // Multiplier for animation speed
	time_offset: f32,

	_padding: [u32; 3],

	color: [f32; 4],
	pattern_determinism: f32,
	frequency: f32,
	_p1: [u32; 2],
}

impl EnergyFluxParams {
	pub fn from_params(params: &mut Parameters<Params>, time: f32, in_data: after_effects::InData) -> Result<Self, after_effects::Error> {
		let debug_flag = if cfg!(debug_assertions) {
			params.get(Params::Debug)?.as_checkbox()?.value() as u8 as u32
		} else {
			0
		};

		let color = params.get(Params::EFColor)?.as_color()?.value();

		Ok(Self {
			time,
			debug: debug_flag,
			is_premiere: in_data.is_premiere() as u8 as u32,
			time_factor: params.get(Params::Speed)?.as_float_slider()?.value() as f32,
			time_offset: params.get(Params::TimeOffset)?.as_float_slider()?.value() as f32,
			_padding: [0; 3],
			color: [
				color.red as f32 / 255.0,
				color.green as f32 / 255.0,
				color.blue as f32 / 255.0,
				color.alpha as f32 / 255.0,
			],
			pattern_determinism: params.get(Params::EFPatDeter)?.as_float_slider()?.value() as f32,
			frequency: params.get(Params::EFFrequency)?.as_float_slider()?.value() as f32,
			_p1: [0; 2],
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
	pub energy_flux_params: Buffer,
	pub energy_flux_bind_group: BindGroup,
	pub the_sun_params: Buffer,
	pub the_sun_bind_group: BindGroup,
	pub sweet_velvet_params: Buffer,
	pub sweet_velvet_bind_group: BindGroup,
	pub staging_buffer: Buffer,
	pub padded_out_stride: u32,
	pub last_access: AtomicUsize,
}

#[derive(Debug)]
pub struct WgpuProcessing<T: Sized> {
	pub device: Device,
	pub queue: Queue,
	pub main_pipeline: ComputePipeline,
	pub energy_flux_pipeline: ComputePipeline,
	pub the_sun_pipeline: ComputePipeline,
	pub sweet_velvet_pipeline: ComputePipeline,
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
	pub fn create_bind_group_layout<S>(device: &Device) -> BindGroupLayout {
		device.create_bind_group_layout(&BindGroupLayoutDescriptor {
			entries: &[
				BindGroupLayoutEntry {
					binding: 0,
					visibility: ShaderStages::COMPUTE,
					ty: BindingType::Buffer {
						ty: BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: BufferSize::new(std::mem::size_of::<S>() as _),
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
		})
	}

	pub fn create_shader_module(device: &Device, source: ProcShaderSource) -> ShaderModule {
		device.create_shader_module(ShaderModuleDescriptor {
			label: Some("main_shader"),
			source: match source {
				ProcShaderSource::SpirV(bytes) => util::make_spirv(bytes),
				ProcShaderSource::Wgsl(wgsl) => ShaderSource::Wgsl(std::borrow::Cow::Owned(wgsl)),
			},
		})
	}

	pub fn new(main_shader: ProcShaderSource, energy_flux_shader: ProcShaderSource, the_sun_shader: ProcShaderSource, velvet_shader: ProcShaderSource) -> Self {
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

		let main_shader_module = Self::create_shader_module(&device, main_shader);
		let main_bind_group_layout = Self::create_bind_group_layout::<KernelParams>(&device);

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

		let energy_flux_shader_module = Self::create_shader_module(&device, energy_flux_shader);
		let energy_bind_group_layout = Self::create_bind_group_layout::<EnergyFluxParams>(&device);

		let energy_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("energy_pipeline_layout"),
			bind_group_layouts: &[&energy_bind_group_layout],
			push_constant_ranges: &[],
		});

		let energy_flux_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &energy_flux_shader_module,
			entry_point: Some("main"),
			label: Some("energy_flux_pipeline"),
			layout: Some(&energy_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		let the_sun_shader_module = Self::create_shader_module(&device, the_sun_shader);
		let the_sun_bind_group_layout = Self::create_bind_group_layout::<TheSunParams>(&device);

		let the_sun_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("the_sun_pipeline_layout"),
			bind_group_layouts: &[&the_sun_bind_group_layout],
			push_constant_ranges: &[],
		});

		let the_sun_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &the_sun_shader_module,
			entry_point: Some("main"),
			label: Some("the_sun_pipeline"),
			layout: Some(&the_sun_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		let sweet_velvet_shader_module = Self::create_shader_module(&device, velvet_shader);
		let sweet_velvet_bind_group_layout = Self::create_bind_group_layout::<Std140VelvetParams>(&device);

		let sweet_velvet_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
			label: Some("sweet_velvet_pipeline_layout"),
			bind_group_layouts: &[&sweet_velvet_bind_group_layout],
			push_constant_ranges: &[],
		});

		let sweet_velvet_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
			module: &sweet_velvet_shader_module,
			entry_point: Some("main"),
			label: Some("sweet_velvet_pipeline"),
			layout: Some(&sweet_velvet_pipeline_layout),
			compilation_options: Default::default(),
			cache: Default::default(),
		});

		Self {
			device,
			queue,
			main_pipeline,
			energy_flux_pipeline,
			the_sun_pipeline,
			sweet_velvet_pipeline,
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

		let (main_bind_group, main_params) = Self::create_bind_group::<KernelParams>(&self.device, &self.main_pipeline, &in_view, &out_view, &sampler);
		let (energy_flux_bind_group, energy_flux_params) =
			Self::create_bind_group::<EnergyFluxParams>(&self.device, &self.energy_flux_pipeline, &in_view, &out_view, &sampler);

		let (the_sun_bind_group, the_sun_params) = Self::create_bind_group::<TheSunParams>(&self.device, &self.the_sun_pipeline, &in_view, &out_view, &sampler);
		let (sweet_velvet_bind_group, sweet_velvet_params) =
			Self::create_bind_group::<Std140VelvetParams>(&self.device, &self.sweet_velvet_pipeline, &in_view, &out_view, &sampler);

		log::info!("Creating buffers {in_size:?} {out_size:?}, thread: {:?}", std::thread::current().id());

		BufferState {
			in_texture,
			out_texture,
			main_params,
			main_bind_group,
			energy_flux_bind_group,
			energy_flux_params,
			the_sun_bind_group,
			the_sun_params,
			sweet_velvet_bind_group,
			sweet_velvet_params,
			out_view,
			staging_buffer,
			padded_out_stride,
			last_access: AtomicUsize::new(Self::timestamp()),
		}
	}

	pub fn create_bind_group<S>(device: &Device, pipeline: &ComputePipeline, in_view: &TextureView, out_view: &TextureView, sampler: &Sampler) -> (BindGroup, Buffer) {
		let params = device.create_buffer(&BufferDescriptor {
			size: std::mem::size_of::<S>() as u64,
			usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
			label: Some(format!("{}_params_buffer", std::any::type_name::<S>()).as_str()),
			mapped_at_creation: false,
		});

		(
			device.create_bind_group(&BindGroupDescriptor {
				label: Some(format!("{} bind group", std::any::type_name::<S>()).as_str()),
				layout: &pipeline.get_bind_group_layout(0),
				entries: &[
					BindGroupEntry {
						binding: 0,
						resource: params.as_entire_binding(),
					},
					BindGroupEntry {
						binding: 1,
						resource: BindingResource::TextureView(in_view),
					},
					BindGroupEntry {
						binding: 2,
						resource: BindingResource::TextureView(out_view),
					},
					BindGroupEntry {
						binding: 3,
						resource: BindingResource::Sampler(sampler),
					},
					BindGroupEntry {
						binding: 4,
						resource: BindingResource::TextureView(in_view),
					},
				],
			}),
			params,
		)
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

	fn run_turbulent_depth(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &KernelParams, out_size: (usize, usize, usize)) {
		self.queue.write_buffer(&state.main_params, 0, bytemuck::cast_slice(&[*params]));
		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("turbulent_pass"),
			timestamp_writes: None,
		});

		cpass.set_pipeline(&self.main_pipeline);
		cpass.set_bind_group(0, &state.main_bind_group, &[]);

		let out_w = out_size.0 as u32;
		let out_h = out_size.1 as u32;
		cpass.dispatch_workgroups(out_w.div_ceil(WG_SIZE), out_h.div_ceil(WG_SIZE), 1);
	}

	fn run_energy_flux(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &EnergyFluxParams, out_size: (usize, usize, usize)) {
		self.queue.write_buffer(&state.energy_flux_params, 0, bytemuck::cast_slice(&[*params]));
		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("energy_flux_pass"),
			timestamp_writes: None,
		});

		cpass.set_pipeline(&self.energy_flux_pipeline);
		cpass.set_bind_group(0, &state.energy_flux_bind_group, &[]);

		let out_w = out_size.0 as u32;
		let out_h = out_size.1 as u32;
		cpass.dispatch_workgroups(out_w.div_ceil(WG_SIZE), out_h.div_ceil(WG_SIZE), 1);
	}

	fn run_the_sun(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &TheSunParams, out_size: (usize, usize, usize)) {
		self.queue.write_buffer(&state.the_sun_params, 0, bytemuck::cast_slice(&[*params]));
		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("the_sun_pass"),
			timestamp_writes: None,
		});

		cpass.set_pipeline(&self.the_sun_pipeline);
		cpass.set_bind_group(0, &state.the_sun_bind_group, &[]);

		let out_w = out_size.0 as u32;
		let out_h = out_size.1 as u32;
		cpass.dispatch_workgroups(out_w.div_ceil(WG_SIZE), out_h.div_ceil(WG_SIZE), 1);
	}

	fn run_sweet_velvet(&self, encoder: &mut CommandEncoder, state: &BufferState, params: &VelvetParams, out_size: (usize, usize, usize)) {
		self.queue.write_buffer(&state.sweet_velvet_params, 0, params.as_std140().as_bytes());
		let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
			label: Some("velvet_pass"),
			timestamp_writes: None,
		});

		cpass.set_pipeline(&self.sweet_velvet_pipeline);
		cpass.set_bind_group(0, &state.sweet_velvet_bind_group, &[]);

		let out_w = out_size.0 as u32;
		let out_h = out_size.1 as u32;
		cpass.dispatch_workgroups(out_w.div_ceil(WG_SIZE), out_h.div_ceil(WG_SIZE), 1);
	}

	#[allow(clippy::too_many_arguments)]
	pub fn run_compute<P>(&self, params: &P, in_size: (usize, usize, usize), out_size: (usize, usize, usize), in_buffer: &[u8], out_buffer: &mut [u8]) -> bool
	where
		Self: ComputeShader<P>,
	{
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

		<Self as ComputeShader<P>>::run(self, &mut encoder, state, params, out_size);

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
