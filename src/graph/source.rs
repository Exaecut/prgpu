//! Source-binding policy.
//!
//! Premiere may hand the same PPix as both source and output (alias). Passes
//! that read source after writing output need a private snapshot. Phase 6
//! moves the snapshot allocation logic into the executor; for now this enum
//! just carries the user's intent so adapters can act on it.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourcePolicy {
	/// Bind the host's source buffer directly. Safe when source/output are
	/// guaranteed distinct (After Effects, Premiere CPU).
	Direct,
	/// Take a private snapshot only when the host signals
	/// `Capability::SourceOutputMayAlias`. Standard recipe for Premiere GPU.
	SnapshotIfAliased { tag: u32 },
	/// Always take a snapshot, regardless of host. Useful when the pipeline
	/// reads source after writing it back through the bloom pyramid.
	AlwaysSnapshot { tag: u32 },
}

impl Default for SourcePolicy {
	fn default() -> Self {
		SourcePolicy::Direct
	}
}
