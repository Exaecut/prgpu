#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RepeatMode {
	None = 1,
	Tile = 2,
	Mirror = 3,
}

impl From<u32> for RepeatMode {
	fn from(value: u32) -> Self {
		match value {
			1 => RepeatMode::None,
			2 => RepeatMode::Tile,
			3 => RepeatMode::Mirror,
			_ => unreachable!("Invalid value for RepeatMode"),
		}
	}
}