#!/usr/bin/env bash
# Bump every prgpu workspace crate to one shared version, with the root `prgpu`
# crate as the version leader. Intra-workspace dependency requirements
# (prgpu-macro, prgpu-build) are rewritten to the same number so a minor/major
# bump that crosses a caret boundary (e.g. 0.1 -> 0.2) still resolves when prgpu
# verifies against its just-published siblings on crates.io.
#
# Usage: scripts/bump-versions.sh [patch|minor|major]   (default: patch)
# Prints the new version to stdout; all other chatter goes to stderr.
set -euo pipefail

kind="${1:-patch}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# First `version = "x.y.z"` inside the [package] table of a manifest.
package_version() {
	awk '
		/^\[/ { in_pkg = ($0 ~ /^\[package\]/) }
		in_pkg && $1 == "version" { gsub(/[^0-9.]/, "", $0); print; exit }
	' "$1"
}

# Rewrite the [package] version line in place.
set_package_version() {
	local file="$1" v="$2"
	awk -v v="$v" '
		/^\[/ { in_pkg = ($0 ~ /^\[package\]/) }
		in_pkg && !done && $1 == "version" { print "version = \"" v "\""; done = 1; next }
		{ print }
	' "$file" > "$file.tmp"
	mv "$file.tmp" "$file"
}

# Rewrite `<dep> = { ... version = "..." ... }` requirement in place.
set_dep_version() {
	local file="$1" dep="$2" v="$3"
	perl -pi -e "s/(\\Q$dep\\E\\s*=\\s*\\{[^}]*?\\bversion\\s*=\\s*\")[^\"]+(\")/\${1}$v\${2}/g" "$file"
}

cur="$(package_version "$root/Cargo.toml")"
[[ -n "$cur" ]] || { echo "bump: could not read prgpu version" >&2; exit 1; }

IFS=. read -r MA MI PA <<< "$cur"
case "$kind" in
	major) MA=$((MA + 1)); MI=0; PA=0 ;;
	minor) MI=$((MI + 1)); PA=0 ;;
	patch) PA=$((PA + 1)) ;;
	*) echo "bump: unknown kind '$kind' (want patch|minor|major)" >&2; exit 1 ;;
esac
new="$MA.$MI.$PA"

# Enumerate members: root prgpu + every prgpu-* member directory.
members=( prgpu )
manifests=( "$root/Cargo.toml" )
for d in "$root"/prgpu-*/; do
	[[ -f "${d}Cargo.toml" ]] || continue
	members+=( "$(basename "$d")" )
	manifests+=( "${d}Cargo.toml" )
done

for m in "${manifests[@]}"; do
	set_package_version "$m" "$new"
done

for m in "${manifests[@]}"; do
	for dep in "${members[@]}"; do
		set_dep_version "$m" "$dep" "$new"
	done
done

echo "bump: $cur -> $new ($kind) across ${members[*]}" >&2
echo "$new"
