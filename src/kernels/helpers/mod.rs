//! Reusable building blocks for kernel host code (CPU + GPU dispatch sites).
//!
//! These helpers contain *no* backend-specific code — they're pure logic that
//! every effect can pull in regardless of how it ultimately runs the kernel.

pub mod blur;
pub mod sweep;
