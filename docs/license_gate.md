# `LicenseGate` — opt-in licence checks

`prgpu` does not depend on any licensing backend. Effects that ship
unlocked declare `type License = NoLicenseGate;` and the adapter never
runs a licence check. Effects that need a licence implement the
`LicenseGate` trait against their own backend (`themis`, in the Exaecut
suite).

## Trait

```rust
pub trait LicenseGate: Default + 'static {
    fn initialize(&self) -> Result<(), &'static str> { Ok(()) }
    fn is_valid(&self) -> bool { true }
    fn retry(&self) -> Result<(), &'static str> { Ok(()) }
    fn debug_label(&self) -> Option<String> { None }
}

#[derive(Default)]
pub struct NoLicenseGate;
impl LicenseGate for NoLicenseGate {}
```

Every method has a safe default. Implementors override only what they
care about.

## Adapter integration

The AE adapter calls:

| Selector             | LicenseGate method   |
|----------------------|----------------------|
| `Cmd_GlobalSetup`    | `initialize`         |
| `Cmd_FrameSetup`     | `is_valid` (skip frame setup if false) |
| `Cmd_Render` / `SmartRender` / `SmartRenderGpu` | `is_valid` (skip render if false) |

`retry` and `debug_label` are not called automatically — surface them
through a parameter button + `ParamApi::actions` if you want a user-driven
retry.

## Example: `themis`

```rust
use themis::{license::InitializationOptions, types::LicenseState};

const PRODUCT_ID: usize = 36;
const LICENSE_SERVER: &str = "https://localhost:11444/license";

#[derive(Default)]
pub struct MindglowLicense;

impl prgpu::effect::LicenseGate for MindglowLicense {
    fn initialize(&self) -> Result<(), &'static str> {
        let ok = themis::license::initialize(InitializationOptions {
            product_id: PRODUCT_ID,
            reset: false,
            server_url: LICENSE_SERVER.to_string(),
            cert_fingerprint: None,
        });
        if ok { Ok(()) } else { Err("themis init failed") }
    }

    fn is_valid(&self) -> bool {
        themis::license::is_valid(false)
    }

    fn retry(&self) -> Result<(), &'static str> {
        let ok = themis::license::initialize(InitializationOptions {
            product_id: PRODUCT_ID,
            reset: true,
            server_url: LICENSE_SERVER.to_string(),
            cert_fingerprint: None,
        });
        if ok { Ok(()) } else { Err("themis retry failed") }
    }

    fn debug_label(&self) -> Option<String> {
        Some(LicenseState::debug_string_from_bits(
            themis::license::get_license_state().bits()))
    }
}
```

Then declare the type alias on your effect:

```rust
impl Effect for Mindglow {
    type License = MindglowLicense;
    // ...
}
```

## User-driven retry button

Wire a parameter as a button, hide it through `ParamApi::visibility` while
the licence is valid, and call your retry logic from `ParamApi::actions`.
Mindglow's `lib.rs` is the canonical example.

```rust
fn ui(api: &mut ParamApi<Params>) -> Result<(), ae::Error> {
    api.visibility(|v| {
        v.show(Params::NoLicense, |_p, _host| !themis::license::is_valid(false));
    });
    api.actions(|a| {
        a.on_click(Params::NoLicense, |ctx| {
            let ok = themis::license::initialize(/* reset: true */);
            if ok { ctx.hot_reload_shaders(); }
            Ok(())
        });
    });
    Ok(())
}
```

The visibility predicate re-runs on every `Cmd_UpdateParamsUi`, so the
button hides automatically the next UI tick after a successful retry.
