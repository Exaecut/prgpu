#!/usr/bin/env bash
set -euo pipefail

# Regenerate Cargo.lock in every sub-crate that has a Cargo.toml
# but no Cargo.lock yet.
find . -mindepth 2 -maxdepth 2 -type f -name Cargo.toml | while read -r manifest; do
    crate_dir="$(dirname "$manifest")"
    if [[ ! -f "$crate_dir/Cargo.lock" ]]; then
        echo "🔧 Generating lockfile for $crate_dir"
        cargo generate-lockfile --manifest-path "$manifest"
    fi
done
