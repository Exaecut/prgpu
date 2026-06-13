//! Source-binding policy.
//!
//! # Why this exists
//!
//! Premiere's GPU path may hand the effect the *same* buffer as both source
//! and output (`Capability::SourceOutputMayAlias`). A pass that reads the
//! source at a *displaced* coordinate after the dispatch has begun writing the
//! output then reads pixels other threads in the same dispatch have already
//! overwritten. The result is 16×16 block corruption (threads race their tile
//! neighbours) and an echo/feedback smear (the image samples its own partial
//! output). After Effects and Premiere CPU never alias, so the hazard is
//! Premiere-GPU-only and invisible in CPU tests.
//!
//! # When you must care
//!
//! You do **not** need to think about this at all: the default
//! [`SourcePolicy::Auto`] detects the hazard for you. A snapshot is taken only
//! when **both** hold:
//!
//! - the host signals `Capability::SourceOutputMayAlias`, and
//! - a pass reads [`Slot::Source`] and writes [`Slot::Output`].
//!
//! That covers every effect that samples the source anywhere other than the
//! pixel it is writing — shakes, blurs, distortions, glows, echoes, godrays.
//! A strict 1:1 colour op (reads only its own output pixel) is safe even when
//! aliased; `Auto` still snapshots it, which is correct but costs one buffer
//! copy. Such effects may opt out with [`SourcePolicy::Direct`].
//!
//! # Examples
//!
//! ```ignore
//! // Typical effect: nothing to do — Auto snapshots iff it detects the hazard.
//! fn pipeline(g: &mut RenderGraph<Self::FrameData>) {
//!     g.add_pass("shake", k::shake::kernel(), Slot::Source, Slot::Output, |c| c.frame_data().main);
//! }
//!
//! // 1:1 colour grade that only reads its own pixel: skip the copy.
//! fn pipeline(g: &mut RenderGraph<Self::FrameData>) {
//!     g.set_source_policy(SourcePolicy::Direct);
//!     g.add_pass("grade", k::grade::kernel(), Slot::Source, Slot::Output, |c| c.frame_data().main);
//! }
//!
//! // Pipeline that reads the source back through a private pyramid and wants the
//! // snapshot on every host, not just aliasing ones:
//! fn pipeline(g: &mut RenderGraph<Self::FrameData>) {
//!     g.set_source_policy(SourcePolicy::AlwaysSnapshot { tag: MY_TAG });
//!     // ...
//! }
//! ```

/// Buffer-pool tag for the snapshot taken automatically by
/// [`SourcePolicy::Auto`]. Pools are keyed by `(dims, tag)` and a snapshot is
/// written and consumed within a single `execute`, so one shared tag is safe.
pub(crate) const AUTO_SOURCE_SNAPSHOT_TAG: u32 = 0x4155_544F;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourcePolicy {
	/// Default. prgpu snapshots the source automatically when the host may
	/// alias source/output **and** a pass reads `Source` while writing
	/// `Output`. No action required from the effect author.
	Auto,
	/// Never snapshot. Bind the host's source buffer directly. Use only for a
	/// pass you know reads solely its own output pixel (1:1), to save the copy
	/// on aliasing hosts.
	Direct,
	/// Snapshot only when the host signals aliasing, keyed by `tag`. Equivalent
	/// to `Auto` but with a caller-chosen pool tag; prefer `Auto` unless you
	/// need a dedicated tag.
	SnapshotIfAliased { tag: u32 },
	/// Always snapshot, regardless of host. Useful when the pipeline reads the
	/// source back after writing it through a private pyramid.
	AlwaysSnapshot { tag: u32 },
}

impl Default for SourcePolicy {
	fn default() -> Self {
		SourcePolicy::Auto
	}
}
