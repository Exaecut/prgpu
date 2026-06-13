/// Build-time effect identity. Constructed only by the generated
/// `prgpu_effect_meta.rs` — never by hand.
pub struct EffectMeta {
	pub match_name: &'static str,
	pub display_name: &'static str,
	pub category: &'static str,
	pub version: (u32, u32, u32),
	pub support_url: Option<&'static str>,
	pub out_flags: u64,
	pub out_flags_2: u64,
	pub expansion: bool,
	pub gpu: bool,
}
