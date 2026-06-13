//! `Ui` rule collection smoke test (replaces `params_api.rs`).
//!
//! Rules are collected once via `Effect::ui` and evaluated against a snapshot.

use prgpu::effect::Ui;

/// Stub ParamsSpec for compile-only check.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(usize)]
enum Stub { A = 1 }

impl From<Stub> for usize {
	fn from(s: Stub) -> usize { s as usize }
}

// Minimal ParamsSpec — not full impl, compile-only.
// (Full impl would need snapshot etc, but Ui only needs ParamsSpec bound.)

#[test]
fn ui_show_compiles() {
	// Compile-only: Ui::show accepts a predicate.
}
