// CUDA: PTX is text but the kernel constant is `&[u8]` (matches Metal's metallib).
// slangc emits a trailing NUL on Windows; the loader trims it before CString::new.
#[macro_export]
macro_rules! include_shader {
    ($name:ident, cuda) => {{ include_bytes!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".ptx")) }};

    ($name:literal, cuda) => {{ include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".ptx")) }};

    ($name:ident, metal) => {{ include_bytes!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".metallib")) }};

    ($name:literal, metal) => {{ include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib")) }};
}
