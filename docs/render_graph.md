# `RenderGraph<F>` — declarative pipeline

Effects describe their multi-pass pipeline once in `Effect::pipeline`. The
adapter caches the graph for the effect instance's lifetime and runs it
against each frame's `FrameData` via `prgpu::graph::execute()`.

```rust
fn pipeline(g: &mut RenderGraph<FrameData>) {
    g.set_source_policy(SourcePolicy::SnapshotIfAliased { tag: SOURCE_TAG });

    let bloom = g.declare_mip_pyramid("bloom", |ctx| {
        let q = Quality::from_index(ctx.frame_data().quality);
        let (scale, levels) = q.pyramid_layout();
        let (w, h) = base_dims(ctx.output_width(), ctx.output_height(), scale);
        MipPyramidDesc::new(w, h)
            .levels(actual_levels(levels, w, h))
            .tag(BLOOM_TAG)
    });

    g.add_pass("prefilter",
        k::bloom_prefilter::kernel(),
        Slot::MainSource, bloom.mip(0),
        |ctx| ctx.frame_data().prefilter);

    g.add_mip_chain("downsample", bloom, MipDirection::Down,
        k::bloom_downsample::kernel(),
        |level, _ctx| BloomDownsampleParams { src_lod: level, .. });

    g.add_mip_chain("upsample", bloom, MipDirection::Up,
        k::bloom_upsample::kernel(),
        |level, ctx| { let mut up = ctx.frame_data().upsample; up.dst_lod = level; up });

    g.add_pass_with_input("composite",
        k::mindglow_composite::kernel(),
        Slot::MainSource, bloom.mip(0), Slot::Output,
        |ctx| ctx.frame_data().composite);
}
```

## Resources

| Method                         | Returns                       | Purpose |
|--------------------------------|-------------------------------|---------|
| `declare_mip_pyramid(name, f)` | `ResourceHandle<MipPyramid>`  | N-level pyramid sized by per-frame closure |

The descriptor closure receives a `MipPyramidCtx<F>` exposing
`frame_data()`, `output_width/height()`, `bytes_per_pixel()`. Returning
`MipPyramidDesc::new(w, h).levels(N).tag(T)` keys the allocation in the
prgpu buffer pool — same `tag` reuses the same physical buffer across
renders.

## Slots

A `Slot` resolves to a `FrameBinding` at execution time:

| Variant                        | Resolves to                                      |
|--------------------------------|--------------------------------------------------|
| `Slot::MainSource`             | `base.main_source` (after source-snapshot policy) |
| `Slot::Output`                 | `base.output`                                    |
| `bloom.mip(lod)`               | A specific mip level of the pyramid resource     |
| `bloom.whole()`                | The whole pyramid (mip-aware sampling)           |
| `Slot::Inline(FrameBinding)`   | A pre-built binding (test fixtures, snapshots)   |

## Single-pass

```rust
g.add_pass(name, kernel, source, target, |ctx| params);
g.add_pass_with_input(name, kernel, source, input, target, |ctx| params);
```

`source` binds slot 0 (`outgoing`), `input` (when given) binds slot 1
(`incoming`), `target` binds slot 2 (`dest`). When no `input` is given,
slot 1 mirrors slot 0 — matches the "either source-only or
source+secondary" pattern Slang kernels expect.

The `params` closure receives a `PassContext<F>` and returns the kernel's
constant-buffer params for this frame.

## Mip chains

```rust
g.add_mip_chain(name, resource, MipDirection::Down, kernel, |level, ctx| params);
g.add_mip_chain(name, resource, MipDirection::Up,   kernel, |level, ctx| params);
```

For an N-level pyramid, the chain runs N-1 dispatches. `Down` walks
`lod 0 → 1 → ... → N-1`; `Up` walks the reverse. The closure receives the
source `level` for `Down` and the destination `level` for `Up`.

## Source-snapshot policy

```rust
g.set_source_policy(SourcePolicy::SnapshotIfAliased { tag });
```

| Variant                              | Behaviour |
|--------------------------------------|-----------|
| `SourcePolicy::Direct`               | No snapshot. Use when source/output are guaranteed distinct. |
| `SourcePolicy::SnapshotIfAliased { tag }` | Snapshot only when host signals `Capability::SourceOutputMayAlias` (Premiere GPU). |
| `SourcePolicy::AlwaysSnapshot { tag }`    | Snapshot unconditionally. Useful for pipelines that read source after writing back through it. |

See [`source_snapshot.md`](source_snapshot.md) for details.

## Execution

```rust
prgpu::graph::execute(graph, frame_data, base) -> Result<(), GraphError>
```

The executor:

1. Clones `base` (so it can rebind `main_source` to a snapshot).
2. Applies `source_policy`.
3. Allocates each declared resource via `cpu::buffer` or `gpu::buffer`
   based on `base.backend`.
4. Walks `passes` in declaration order. Each pass:
   - resolves slot bindings against `base` + the resource table,
   - builds a `Configuration` via `ConfigBuilder`,
   - calls the type-erased dispatcher closure (which knows the kernel +
     param closure + backend branch).

`GraphError` short-circuits the whole render — a failed pass aborts cleanly
instead of producing a partial output.

## Why static + dynamic

The graph **structure** (resources, passes, mip chains, source policy) is
declared once. Per-frame **behaviour** comes from closures: param
extractors, mip-pyramid sizing, and (future) `enabled_when` predicates.
Don't rebuild the graph every frame just to disable a pass — wire it
through a closure that returns zero-strength params, or guard with an
`enabled_when` predicate when that lands.

## Validation

The executor validates lazily: missing target, unknown resource, mip
level out of range, or bad config build all return a `GraphError`
identifying the offending pass.
