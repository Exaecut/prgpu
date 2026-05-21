# `LicenseGate` — opt-in licence checks

`prgpu` does not depend on any licensing backend. Effects that ship
unlocked declare `type License = NoLicenseGate;` and the adapter never
runs a licence check. Effects that need a licence check implement the
`LicenseGate` trait against their own backend.

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

## Example shape

The actual licence backend is yours — an HTTP token check, a hardware
dongle, a hashed key file, anything. The trait only cares about the four
methods. A typical implementation routes each method to the backend's
equivalent call:

```rust
#[derive(Default)]
pub struct MyLicense;

impl prgpu::effect::LicenseGate for MyLicense {
    fn initialize(&self) -> Result<(), &'static str> {
        // Talk to your licence backend; return Ok(()) when the user is
        // authorised. Called once at Cmd_GlobalSetup.
        my_backend::start().map_err(|_| "license init failed")
    }

    fn is_valid(&self) -> bool {
        // Cheap synchronous check the adapter calls before every render.
        my_backend::is_valid()
    }

    fn retry(&self) -> Result<(), &'static str> {
        // Force-refresh credentials. Called from a parameter button click.
        my_backend::reauth().map_err(|_| "license retry failed")
    }

    fn debug_label(&self) -> Option<String> {
        // Optional human-readable status to surface on a retry button.
        Some(my_backend::status_string())
    }
}
```

Then declare the type alias on your effect:

```rust
impl Effect for MyEffect {
    type License = MyLicense;
    // ...
}
```

## User-driven retry button

Wire a parameter as a button, hide it through `ParamApi::visibility` while
the licence is valid, and call your retry logic from `ParamApi::actions`.

```rust
fn ui(api: &mut ParamApi<Params>) -> Result<(), ae::Error> {
    api.visibility(|v| {
        v.show(Params::NoLicense, |_p, _host| !my_backend::is_valid());
    });
    api.actions(|a| {
        a.on_click(Params::NoLicense, |ctx| {
            if my_backend::reauth().is_ok() {
                ctx.hot_reload_shaders();
            }
            Ok(())
        });
    });
    Ok(())
}
```

The visibility predicate re-runs on every `Cmd_UpdateParamsUi`, so the
button hides automatically the next UI tick after a successful retry.
