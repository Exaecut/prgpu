prgpu::params! {
	pub enum Params {
		#[slider(label = "A", range = 0.0..=1.0, default = 0.5)]
		Dup,
		#[checkbox(label = "B", default = false)]
		Dup,
	}
}

fn main() {}
