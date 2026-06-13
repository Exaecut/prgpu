use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
	Cpu,
	Cuda,
	Metal,
}

impl Backend {
	pub(crate) fn from_premiere_framework(v: u32) -> Option<Backend> {
		match v {
			0 => Some(Backend::Cuda),
			2 => Some(Backend::Metal),
			_ => None,
		}
	}
}

impl Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
           Backend::Cpu => "CPU",
           Backend::Cuda => "CUDA",
           Backend::Metal => "Metal",
        };

        f.write_str(str)
    }
}