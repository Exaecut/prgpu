use std::fmt::Display;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
	Cpu,
	Cuda,
	Metal,
    OpenCL
}

impl Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
           Backend::Cpu => "CPU",
           Backend::Cuda => "CUDA",
           Backend::Metal => "Metal",
           Backend::OpenCL => "OpenCL"
        };

        f.write_str(str)
    }
}