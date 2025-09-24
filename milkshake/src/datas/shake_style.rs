#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShakeStyle {
	Perlin = 1,
	Wave,
}

impl From<i32> for ShakeStyle {
	fn from(value: i32) -> Self {
		match value {
			1 => ShakeStyle::Perlin,
			2 => ShakeStyle::Wave,
			_ => unreachable!("Invalid value for ShakeStyle"),
		}
	}
}