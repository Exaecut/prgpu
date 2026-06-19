use serde::Deserialize;
use std::{fs, path::PathBuf};

use crate::DynError;

#[derive(Debug, Clone)]
pub struct EffectMetadata {
	pub match_name: &'static str,
	pub display_name: &'static str,
	pub category: &'static str,
	pub support_url: Option<&'static str>,
	pub expansion: bool,
	pub gpu: bool,
	pub custom_ui: bool,
}

#[derive(Deserialize)]
struct PackageManifest {
	package: Package,
}

#[derive(Deserialize)]
struct Package {
	#[serde(rename = "metadata")]
	metadata: Option<PackageMetadata>,
}

#[derive(Deserialize)]
struct PackageMetadata {
	prgpu: Option<RawEffectMetadata>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawEffectMetadata {
	#[serde(rename = "match-name")]
	match_name: Option<String>,
	#[serde(rename = "display-name")]
	display_name: Option<String>,
	category: Option<String>,
	#[serde(rename = "support-url")]
	support_url: Option<String>,
	expansion: Option<bool>,
	gpu: Option<bool>,
	#[serde(rename = "custom-ui")]
	custom_ui: Option<bool>,
}

impl EffectMetadata {
	pub fn from_cargo_manifest() -> Result<Self, DynError> {
		let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
		let manifest_path = manifest_dir.join("Cargo.toml");
		let contents = fs::read_to_string(&manifest_path)?;
		let parsed: PackageManifest = toml::from_str(&contents)?;

		let raw = parsed
			.package
			.metadata
			.as_ref()
			.and_then(|m| m.prgpu.as_ref())
			.ok_or("missing [package.metadata.prgpu] table")?;

		let match_name = raw
			.match_name
			.as_ref()
			.ok_or("[package.metadata.prgpu] missing required field 'match-name'")?
			.clone();
		let display_name = raw
			.display_name
			.as_ref()
			.ok_or("[package.metadata.prgpu] missing required field 'display-name'")?
			.clone();
		let category = raw
			.category
			.as_ref()
			.ok_or("[package.metadata.prgpu] missing required field 'category'")?
			.clone();
		let support_url = raw.support_url.as_ref().map(|s| s.clone());

		Ok(EffectMetadata {
			match_name: Box::leak(match_name.into_boxed_str()),
			display_name: Box::leak(display_name.into_boxed_str()),
			category: Box::leak(category.into_boxed_str()),
			support_url: support_url.map(|s| Box::leak(s.into_boxed_str()) as &'static str),
			expansion: raw.expansion.unwrap_or(false),
			gpu: raw.gpu.unwrap_or(true),
			custom_ui: raw.custom_ui.unwrap_or(false),
		})
	}
}
