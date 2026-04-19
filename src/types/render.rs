pub struct RenderContext<'a> {
	pub in_data: &'a after_effects::InData,
	pub in_layer: &'a after_effects::Layer,
	pub out_layer: &'a mut after_effects::Layer,
}