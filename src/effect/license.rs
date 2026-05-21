//! Optional license-gate trait the AE/Premiere adapters consult.
//!
//! [`LicenseGate`] is opt-in. Effects that don't ship a licence check use
//! [`NoLicenseGate`] (the [`crate::effect::Effect`] trait's default
//! associated type once Phase 11 lands), and the adapters skip every check.
//!
//! Effects that DO need a licence check (mindglow, retrovhs, ...) implement
//! the trait against their own backend (`themis`, today). The adapter then
//! calls:
//!
//! - [`LicenseGate::initialize`] once during `Cmd_GlobalSetup`,
//! - [`LicenseGate::is_valid`] before every render selector,
//! - [`LicenseGate::retry`] when the user clicks a "Retry" parameter button.
//!
//! `prgpu` itself never depends on `themis` or any concrete licence backend.

/// Marker for an effect's licence backend.
///
/// All methods have safe defaults — implementing the trait without
/// overriding anything is functionally equivalent to [`NoLicenseGate`].
/// Effects override the methods they care about.
pub trait LicenseGate: Default + 'static {
	/// Called once during `Cmd_GlobalSetup`. Default: no-op.
	fn initialize(&self) -> Result<(), &'static str> {
		Ok(())
	}

	/// Called before every render dispatch. `false` skips the render and
	/// surfaces a "license check failed" parameter / button to the user.
	/// Default: always valid.
	fn is_valid(&self) -> bool {
		true
	}

	/// Called when the user clicks the licence-retry parameter. Default:
	/// no-op.
	fn retry(&self) -> Result<(), &'static str> {
		Ok(())
	}

	/// Optional human-readable status string the adapter can show in the
	/// retry button label. Default: none.
	fn debug_label(&self) -> Option<String> {
		None
	}
}

/// Default [`LicenseGate`] implementation that always succeeds. Effects
/// without a licence check use this through `Effect::License = NoLicenseGate`
/// (which is the trait's default associated type).
#[derive(Default)]
pub struct NoLicenseGate;

impl LicenseGate for NoLicenseGate {}
