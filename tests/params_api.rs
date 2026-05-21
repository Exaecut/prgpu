//! `ParamApi` builder tests.
//!
//! Verifies the visibility / action builders accept closures with the right
//! signatures and accumulate rules. Adapter wiring (calling the predicates
//! from `Cmd_UpdateParamsUi`) is exercised in the AE EffectAdapter tests.

// `ParamApi` is constructed by the AE adapter at parameter-setup time; the
// only standalone-testable surface is the `ActionContext` builder shape,
// which is exercised through the trait method below.

use prgpu::effect::ActionBuilder;

// Compile-only check: ActionBuilder accepts an `on_click` closure with the
// expected signature.
fn _on_click_compile_check(b: &mut ActionBuilder<usize>) {
	b.on_click(0usize, |ctx| {
		ctx.hot_reload_shaders();
		Ok(())
	});
}
