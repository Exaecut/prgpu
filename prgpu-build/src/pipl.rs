use ::pipl::{OutFlags, OutFlags2, Property, Stage};

use crate::metadata::EffectMetadata;

pub fn derive_flags(m: &EffectMetadata) -> (OutFlags, OutFlags2) {
	let mut f = OutFlags::PixIndependent | OutFlags::NonParamVary | OutFlags::SendUpdateParamsUI;
	if m.expansion {
		f |= OutFlags::IExpandBuffer | OutFlags::UseOutputExtent;
	}
	let mut f2 = OutFlags2::SupportsThreadedRendering
		| OutFlags2::SupportsSmartRender
		| OutFlags2::SupportsGetFlattenedSequenceData
		| OutFlags2::ParamGroupStartCollapsedFlag;
	if m.gpu {
		f2 |= OutFlags2::SupportsGpuRenderF32;
	}
	(f, f2)
}

pub fn build_properties(m: &EffectMetadata, mut extra: Vec<Property>) -> Vec<Property> {
	let (out_flags, out_flags_2) = derive_flags(m);

	let mut props = vec![
		Property::Kind(::pipl::PIPLType::AEEffect),
		Property::Name(&m.display_name),
		Property::Category(&m.category),
	];

	#[cfg(target_os = "windows")]
	props.push(Property::CodeWin64X86("EffectMain"));
	#[cfg(target_os = "macos")]
	{
		props.push(Property::CodeMacIntel64("EffectMain"));
		props.push(Property::CodeMacARM64("EffectMain"));
	}

	let version = parse_version();

	props.extend([
		Property::AE_PiPL_Version { major: 2, minor: 0 },
		Property::AE_Effect_Spec_Version {
			major: PF_PLUG_IN_VERSION,
			minor: PF_PLUG_IN_SUBVERS,
		},
		Property::AE_Effect_Version {
			version: version.0,
			subversion: version.1,
			bugversion: version.2,
			stage: version.3,
			build: 0,
		},
		Property::AE_Effect_Info_Flags(3),
		Property::AE_Effect_Global_OutFlags(out_flags),
		Property::AE_Effect_Global_OutFlags_2(out_flags_2),
		Property::AE_Effect_Match_Name(&m.match_name),
		Property::AE_Reserved_Info(0),
	]);

	if let Some(url) = m.support_url {
		props.push(Property::AE_Effect_Support_URL(url));
	}

	props.append(&mut extra);
	props
}

const PF_PLUG_IN_VERSION: u16 = 13;
const PF_PLUG_IN_SUBVERS: u16 = 28;

fn parse_version() -> (u32, u32, u32, Stage) {
	let vers = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into());
	let mut parts = vers.split('.');
	let major = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
	let minor = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);

	let mut patch_and_pre = parts.next().unwrap_or("0").split('-');
	let patch = patch_and_pre.next().and_then(|s| s.parse().ok()).unwrap_or(0);
	let stage = if patch_and_pre.next().is_some() {
		Stage::Develop
	} else {
		Stage::Release
	};

	(major, minor, patch, stage)
}

pub fn dump_pipl(properties: &[Property]) -> String {
	format!("{:#?}", properties)
}
