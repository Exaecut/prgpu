use std::error::Error;

use pipl::*;

use prgpu::build::compile_shaders;

const PF_PLUG_IN_VERSION: u16 = 13;
const PF_PLUG_IN_SUBVERS: u16 = 28;

#[rustfmt::skip]
fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let shader_hotreload = std::env::var("EX_SHADER_HOTRELOAD").unwrap_or("false".to_string());
    println!("cargo:warning=Shader hot reload flag: {}", shader_hotreload);

    compile_shaders("./shaders")?;

    if shader_hotreload == "true" {
        println!("cargo:rustc-check-cfg=cfg(shader_hotreload)");
        println!("cargo:rustc-cfg=shader_hotreload");
        println!("cargo:rerun-if-env-changed=EX_SHADER_HOTRELOAD");
        println!("cargo:warning=Hot reloading shaders is enabled. This is not recommended for production builds.");
    } else {
        println!("cargo:warning=Hot reloading shaders is disabled.");
    }

    pipl::plugin_build(vec![
        Property::Kind(PIPLType::AEEffect),
        Property::Name("EX Vignette"),
        Property::Category("Exaecut"),

        #[cfg(target_os = "windows")]
        Property::CodeWin64X86("EffectMain"),
        #[cfg(target_os = "macos")]
        Property::CodeMacIntel64("EffectMain"),
        #[cfg(target_os = "macos")]
        Property::CodeMacARM64("EffectMain"),
        
        Property::AE_PiPL_Version { major: 2, minor: 0 },
        Property::AE_Effect_Spec_Version {
            major: PF_PLUG_IN_VERSION,
            minor: PF_PLUG_IN_SUBVERS,
        },
        Property::AE_Effect_Version {
            version: 0,
            subversion: 1,
            bugversion: 0,
            stage: Stage::Develop,
            build: 0,
        },
        Property::AE_Effect_Info_Flags(3),
        Property::AE_Effect_Global_OutFlags(
            OutFlags::PixIndependent
                | OutFlags::UseOutputExtent
                | OutFlags::IExpandBuffer
                | OutFlags::NonParamVary
                | OutFlags::SendUpdateParamsUI,
        ),
        Property::AE_Effect_Global_OutFlags_2(
            OutFlags2::SupportsThreadedRendering
            | OutFlags2::SupportsSmartRender
            | OutFlags2::SupportsGetFlattenedSequenceData
            | OutFlags2::ParamGroupStartCollapsedFlag,
        ),
        Property::AE_Effect_Match_Name("EXAE Vignette"),
        Property::AE_Reserved_Info(0),
        Property::AE_Effect_Support_URL("https://exaecut.io/vignette/issues"),
    ]);
    
    Ok(())
}
