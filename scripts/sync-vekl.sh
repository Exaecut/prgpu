#!/usr/bin/env bash
# Refresh the vendored vekl snapshot inside prgpu's crate tree so the next
# `cargo publish` ships the latest vekl. Prefers the sibling workspace
# checkout when available (no network), falls back to a shallow clone.
#
# Usage: ./scripts/sync-vekl.sh [VEKL_REF]
#   VEKL_REF defaults to `main`. Pass a tag or sha to pin a specific version.

set -euo pipefail

cd "$(dirname "$0")/.."

VEKL_REF="${1:-main}"
SIBLING_VEKL="../vekl"
UPSTREAM_URL="https://github.com/Exaecut/vekl.git"

rm -rf vekl

if [ -d "$SIBLING_VEKL/.git" ] || [ -d "$SIBLING_VEKL" ]; then
	echo "Syncing vekl from sibling $SIBLING_VEKL (ref ignored — taking current tree)"
	mkdir -p vekl
	# rsync slang + license + readme only; skip .git, docs, tests, CI, etc.
	rsync -a \
		--include='*/' \
		--include='*.slang' \
		--include='LICENSE' \
		--include='README.md' \
		--exclude='*' \
		"$SIBLING_VEKL/" vekl/
else
	echo "Cloning vekl@$VEKL_REF from $UPSTREAM_URL"
	git clone --depth 1 --branch "$VEKL_REF" "$UPSTREAM_URL" vekl
	rm -rf vekl/.git
fi

count=$(find vekl -type f -name '*.slang' | wc -l | tr -d ' ')
echo "vendored $count .slang files into prgpu/vekl/"
