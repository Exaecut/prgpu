# Publishing prgpu + Exaecut fork of after-effects to crates.io

## Overview

Six crates to publish, across two repositories:

| Order | Crate                        | Path (repo → dir)                                   | Version |
|-------|------------------------------|-----------------------------------------------------|---------|
| 1     | `exaecut-after-effects-sys`  | `Exaecut/after-effects` → `after-effects-sys/`      | 0.5.0   |
| 2     | `exaecut-premiere-sys`       | `Exaecut/after-effects` → `premiere-sys/`           | 0.5.0   |
| 3     | `exaecut-pipl`               | `Exaecut/after-effects` → `pipl/`                   | 0.2.0   |
| 4     | `exaecut-after-effects`      | `Exaecut/after-effects` → `after-effects/`          | 0.5.0   |
| 5     | `exaecut-premiere`           | `Exaecut/after-effects` → `premiere/`               | 0.5.0   |
| 6     | `prgpu-macro`                | `Exaecut/prgpu` → `prgpu-macro/`                    | 0.1.0   |
| 7     | `prgpu`                      | `Exaecut/prgpu` → `.`                               | 0.1.2   |

Rows 1–3 and 6 have no inter-dep and can publish in parallel. Rows 4 and 5
require rows 1 and 2 respectively to be indexed on crates.io first. Row 7
requires rows 4, 5, and 6.

## Why the `exaecut-` prefix

- `after-effects`, `after-effects-sys`, `premiere`, `premiere-sys`, `pipl` are
  all taken on crates.io by virtualritz (upstream we fork from).
- Each Exaecut crate keeps `[lib] name = "after_effects"` (etc.) so every
  downstream `use after_effects::*` / `use premiere::*` / `use pipl::*`
  compiles unchanged. The rename only surfaces in `Cargo.toml`.

## Authentication

You'll need a crates.io API token with publish scope on each name. Claim the
names first (`cargo owner --add` does this implicitly on the first publish).

```
cargo login <token>
```

## Step-by-step

### 1. Publish the 3 independent fork crates (can run in parallel)

```
cd /Users/marvincano/Projects/Exaecut/after-effects
cargo publish -p exaecut-after-effects-sys
cargo publish -p exaecut-premiere-sys
cargo publish -p exaecut-pipl
```

Expect ~30 s each (mostly compile + crates.io upload).

### 2. Wait for the crates.io index to sync (~30-60 s)

`cargo search exaecut-after-effects-sys` should return the new crate. If it
doesn't after two minutes, crates.io is slow; wait longer before step 3.

### 3. Publish the two high-level fork crates

```
cd /Users/marvincano/Projects/Exaecut/after-effects
cargo publish -p exaecut-after-effects
cargo publish -p exaecut-premiere
```

### 4. Publish prgpu-macro (independent of the fork)

```
cd /Users/marvincano/Projects/Exaecut/effects/prgpu
cargo publish -p prgpu-macro
```

### 5. Wait again for index sync, then publish prgpu itself

```
cd /Users/marvincano/Projects/Exaecut/effects/prgpu
cargo publish -p prgpu
```

## Dry-runs (already validated)

All four independent crates pass `cargo publish --dry-run --allow-dirty`:

- `exaecut-after-effects-sys`: ✔
- `exaecut-premiere-sys`: ✔
- `exaecut-pipl`: ✔
- `prgpu-macro`: ✔

`exaecut-after-effects`, `exaecut-premiere`, and `prgpu` have passing dry-runs
modulo the "no matching package named X" error for their upstream deps,
which resolves automatically once steps 1-4 land.

## Post-publish sanity check

After every crate is live, from a **clean** clone of the Exaecut workspace
(no `[patch.crates-io]`), `cargo check -p radialblur` should resolve every
dep from crates.io and build. You can temporarily comment out the
`[patch.crates-io]` block in `effects/Cargo.toml` to test this locally.

## Downstream effects / transitions

All 11 effects and 7 transitions in this monorepo have already been migrated
to the new dep names. Once publishing is complete:

- `effects/Cargo.toml` `[patch.crates-io]` can optionally be removed; builds
  will then resolve entirely from crates.io.
- Same for `transitions/Cargo.toml`.
- Leaving the patch in place is fine — it only takes effect during local
  workspace builds, not for external consumers.

## Create-video-effect CLI (generator)

Templates in `create-video-effect/templates/effect/*/Cargo.toml.tera` already
reference `prgpu = "0.1"` and `exaecut-* = "0.5" / "0.2"`. Once prgpu is live,
the generator produces projects that compile out of the box on any machine
that can resolve crates.io (no local workspace required).

## Rolling back

If a publish is wrong, yank it:

```
cargo yank --version 0.5.0 -p exaecut-after-effects
```

Yanking doesn't delete; it just prevents *new* projects from resolving to
that version. Bump the version for the next attempt.

## Future upstream sync

The Exaecut fork at `Exaecut/after-effects` stays synced with
`virtualritz/after-effects` periodically. When the upstream releases a new
version (e.g. 0.5.0), rebase/merge into the Exaecut fork, bump our version
to 0.6.0 (or `0.5.N+1` if compatible), and republish the 5 crates in the same
order. No downstream code changes required — `[lib] name` preservation means
the Rust API is stable across Exaecut releases.
