#[test]
fn params_compile_fail() {
	let t = trybuild::TestCases::new();
	t.compile_fail("tests/params/compile-fail/*.rs");
}
