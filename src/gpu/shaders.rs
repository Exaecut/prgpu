// TRANSITIONAL(plan-04): macro deleted when kernel! lands.
#[macro_export]
macro_rules! include_shader {
	($name:ident) => { include_bytes!(concat!(env!("OUT_DIR"), "/", stringify!($name), ".shader")) };

	($name:literal) => { include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".shader")) };
}
