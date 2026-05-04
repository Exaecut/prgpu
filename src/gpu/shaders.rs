#[macro_export]
macro_rules! include_shader {
    ($name:ident, cuda) => {{ include_str!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".ptx")) }};

    ($name:literal, cuda) => {{ include_str!(concat!(env!("OUT_DIR"), "/", $name, ".ptx")) }};

    ($name:ident, metal) => {{ include_bytes!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".metallib")) }};

    ($name:literal, metal) => {{ include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".metallib")) }};
}
