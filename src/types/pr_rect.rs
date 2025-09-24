use premiere as pr;

pub struct PrRect(pr::sys::prRect);

impl From<pr::sys::prRect> for PrRect {
	fn from(rect: pr::sys::prRect) -> Self {
		PrRect(rect)
	}
}

impl From<PrRect> for after_effects::Rect {
	fn from(rect: PrRect) -> Self {
		after_effects::Rect {
			bottom: rect.0.bottom as i32,
			left: rect.0.left as i32,
			right: rect.0.right as i32,
			top: rect.0.top as i32,
		}
	}
}
