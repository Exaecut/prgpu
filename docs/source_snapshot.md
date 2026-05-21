# Source snapshot policy

Premiere may hand the same PPix as both source and output. A pipeline
that reads source after writing output corrupts its own next-frame
source. The graph's `SourcePolicy` lets you opt into a private snapshot
without writing the alloc / copy plumbing per effect.

## Variants

```rust
pub enum SourcePolicy {
    Direct,
    SnapshotIfAliased { tag: u32 },
    AlwaysSnapshot { tag: u32 },
}
```

| Variant                              | When the executor takes a snapshot |
|--------------------------------------|-------------------------------------|
| `Direct`                             | never                               |
| `SnapshotIfAliased { tag }`          | only when the host signals `Capability::SourceOutputMayAlias` (Premiere GPU) |
| `AlwaysSnapshot { tag }`             | always, regardless of host          |

`tag` keys the snapshot in the prgpu buffer pool — same `tag` reuses the
same physical buffer across renders.

## Recommended recipe

For effects whose composite pass writes the output buffer and whose
prefilter / mip chain reads the source:

```rust
fn pipeline(g: &mut RenderGraph<FrameData>) {
    g.set_source_policy(SourcePolicy::SnapshotIfAliased { tag: SOURCE_TAG });
    // ... rest of the pipeline
}
```

The executor:

1. Builds a private buffer keyed by `tag` (CPU pool or GPU pool, by backend).
2. Copies `base.main_source` into it via the existing `mip::prepare_source_copy`
   (CPU memcpy / Metal `copy_buffer` / CUDA `cuMemcpy`).
3. Rebinds `base.main_source` to the snapshot for the rest of the graph.

`Slot::MainSource` in every pass now resolves to the snapshot, not the
original host buffer.

## Tag conventions

Use a 32-bit value with the upper half encoding the effect namespace and
the lower half encoding the role:

| Tag           | Meaning                |
|---------------|------------------------|
| `0xAA_0001`   | effect "AA" — bloom pyramid |
| `0xAA_0002`   | effect "AA" — source snapshot |
| `0xBB_0001`   | effect "BB" — source mip pyramid |

Distinct lower halves keep the bloom pyramid and the source snapshot from
sharing a slot. The upper-half byte is just an effect-side convention so
two effects loaded into the same host don't collide on tag values.

## Why not always snapshot?

Snapshots cost a copy + extra memory. On hosts where source/output never
alias (After Effects standalone, Premiere CPU), the snapshot is pure
overhead. `SnapshotIfAliased` defers the cost to the only host that
actually needs it.

## Performance note

The snapshot allocator is keyed by `(host pointer, tag)` (via
`prepare_source_snapshot`) when an effect needs once-per-clip semantics
(see `pipeline::mip::prepare_source_snapshot` source for the host-pointer
folding). The current executor uses the simpler `prepare_source_copy`,
which re-copies every frame — fine for the standard alias case where the
source PPix is freshly populated each frame anyway. Effects that need the
once-per-clip variant should drop down to `prepare_source_snapshot`
manually for now.
