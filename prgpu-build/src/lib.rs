use std::path::{Path, PathBuf};

use ::pipl::{OutFlags, OutFlags2, Property};

pub mod backend;
pub mod bindings;
pub mod compile;
pub mod cpu_dispatch;
pub mod lsp;
pub mod meta_gen;
pub mod metadata;
pub mod pipl;
pub mod reflection;
pub mod sdk;

pub type DynError = Box<dyn std::error::Error + Send + Sync>;

pub fn effect() -> EffectBuild {
	EffectBuild::from_cargo_env()
}

/// Compile the `.slang` shaders in `shader_dir` for the active GPU backend,
/// generate CPU dispatch bridges, and emit the backend cfg for `prgpu`.
pub fn compile_builtin_shaders(shader_dir: &Path) -> Result<(), DynError> {
	let backend = backend::resolve_backend();
	backend::emit_backend_cfg(backend);
	println!("cargo:rerun-if-changed=build.rs");

	let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
	if shader_dir.is_dir() {
		let include_dirs = compile::resolve_include_dirs(shader_dir, None)?;
		compile::compile_shaders(shader_dir, &out_dir, &include_dirs, backend)?;
	}

	Ok(())
}

pub struct EffectBuild {
	shader_dir: PathBuf,
	slang_include: Option<PathBuf>,
	metadata: metadata::EffectMetadata,
	out_flags: Option<OutFlags>,
	extra_out_flags: OutFlags,
	extra_out_flags_2: OutFlags2,
	extra_properties: Vec<Property>,
}

impl EffectBuild {
	fn from_cargo_env() -> Self {
		let metadata = metadata::EffectMetadata::from_cargo_manifest()
			.unwrap_or_else(|e| panic!("failed to read [package.metadata.prgpu]: {e}"));

		Self {
			shader_dir: PathBuf::from("shaders"),
			slang_include: None,
			metadata,
			out_flags: None,
			extra_out_flags: OutFlags::None,
			extra_out_flags_2: OutFlags2::None,
			extra_properties: Vec::new(),
		}
	}

	pub fn shader_dir(mut self, dir: impl Into<PathBuf>) -> Self {
		self.shader_dir = dir.into();
		self
	}

	pub fn slang_include(mut self, dir: impl Into<PathBuf>) -> Self {
		self.slang_include = Some(dir.into());
		self
	}

	pub fn match_name(mut self, name: &str) -> Self {
		self.metadata.match_name = Box::leak(name.to_owned().into_boxed_str()) as &'static str;
		self
	}

	pub fn display_name(mut self, name: &str) -> Self {
		self.metadata.display_name = Box::leak(name.to_owned().into_boxed_str()) as &'static str;
		self
	}

	pub fn out_flags(mut self, f: OutFlags) -> Self {
		self.out_flags = Some(f);
		self
	}

	pub fn extra_out_flags(mut self, f: OutFlags) -> Self {
		self.extra_out_flags |= f;
		self
	}

	pub fn extra_out_flags_2(mut self, f: OutFlags2) -> Self {
		self.extra_out_flags_2 |= f;
		self
	}

	pub fn pipl_property(mut self, p: Property) -> Self {
		self.extra_properties.push(p);
		self
	}

	pub fn build(self) {
		if let Err(e) = self.run() {
			panic!("prgpu_build::effect().build() failed: {e}");
		}
	}

	fn run(self) -> Result<(), DynError> {
		let backend = backend::resolve_backend();
		backend::emit_backend_cfg(backend);

		// TRANSITIONAL(plan-04): effect crates only get the cfg transitionally;
		// phase 4 removes the need for this emission.
		println!("cargo:rustc-cfg=with_premiere");

		let out_dir = PathBuf::from(std::env::var("OUT_DIR")?);
		let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
		let shader_dir_abs = manifest_dir.join(&self.shader_dir);

		if shader_dir_abs.is_dir() {
			let include_dirs = compile::resolve_include_dirs(&shader_dir_abs, self.slang_include.as_deref())?;
			compile::compile_shaders(&shader_dir_abs, &out_dir, &include_dirs, backend)?;
		}

		let metadata = self.metadata;
		if let Some(replace_flags) = self.out_flags {
			let (_, base_f2) = pipl::derive_flags(&metadata);
			let mut metadata = metadata;
			metadata.expansion = replace_flags.contains(OutFlags::IExpandBuffer);
			let f2 = base_f2 | self.extra_out_flags_2;
			let f = replace_flags | self.extra_out_flags;
			let mut props = pipl::build_properties(&metadata, self.extra_properties);
			replace_flag_prop(&mut props, Property::AE_Effect_Global_OutFlags(f));
			replace_flag_prop(&mut props, Property::AE_Effect_Global_OutFlags_2(f2));
			emit_pipl(props)?;
			meta_gen::write_effect_meta(&out_dir, &metadata);
		} else {
			let (base_f, base_f2) = pipl::derive_flags(&metadata);
			let f = base_f | self.extra_out_flags;
			let f2 = base_f2 | self.extra_out_flags_2;
			let mut props = pipl::build_properties(&metadata, self.extra_properties);
			replace_flag_prop(&mut props, Property::AE_Effect_Global_OutFlags(f));
			replace_flag_prop(&mut props, Property::AE_Effect_Global_OutFlags_2(f2));
			emit_pipl(props)?;
			meta_gen::write_effect_meta(&out_dir, &metadata);
		}

		Ok(())
	}
}

fn replace_flag_prop(props: &mut Vec<Property>, new_prop: Property) {
	let discriminant = std::mem::discriminant(&new_prop);
	if let Some(pos) = props.iter().position(|p| std::mem::discriminant(p) == discriminant) {
		props[pos] = new_prop;
	}
}

fn emit_pipl(props: Vec<Property>) -> Result<(), DynError> {
	if std::env::var_os("PRGPU_BUILD_DUMP_PIPL").is_some() {
		eprintln!("{}", pipl::dump_pipl(&props));
	}
	::pipl::plugin_build(props);
	Ok(())
}
