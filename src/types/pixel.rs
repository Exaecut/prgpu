use std::ops::{Deref, DerefMut};

use after_effects::sys::PF_Pixel;

#[derive(Clone, Copy)]
pub struct Pixel(PF_Pixel);

impl Pixel {
	pub fn from_pf_pixel(pf_pixel: PF_Pixel) -> Self {
		Pixel(pf_pixel)
	}

	pub fn debug_print_color(v: i64) {
		let raw64 = v as u64;
		let x = raw64.to_be_bytes();

		println!(
			"Param Int64 = {v}, hex64 = {raw64:#018x}, bytes = {x:?}, Decoded RGBA = ({},{},{},{})",
			x[2], x[4], x[6], x[0]
		);
	}

	pub fn from_u64_color(raw64: u64) -> Self {
		let x = raw64.to_be_bytes();

		Pixel(PF_Pixel {
			red: x[2],
			green: x[4],
			blue: x[6],
			alpha: x[0],
		})
	}

	pub fn from_bytes32(raw: u32) -> Self {
		let b = raw.to_be_bytes();
		Pixel(PF_Pixel {
			red: b[1],
			green: b[2],
			blue: b[3],
			alpha: b[0],
		})
	}
}

impl Default for Pixel {
	fn default() -> Self {
		Pixel(PF_Pixel {
			alpha: 255,
			red: 0,
			green: 0,
			blue: 0,
		})
	}
}

impl From<Pixel> for PF_Pixel {
	fn from(wrapper: Pixel) -> Self {
		wrapper.0
	}
}

impl Deref for Pixel {
	type Target = PF_Pixel;
	fn deref(&self) -> &Self::Target {
		&self.0
	}
}

impl DerefMut for Pixel {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.0
	}
}
