#[test]
fn compile_fail_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/gpu_struct/compile-fail/*.rs");
}
