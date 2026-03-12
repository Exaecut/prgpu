#!/usr/bin/env bash

set -euo pipefail

CHANGED_PLUGIN_DIRS="${CHANGED_PLUGIN_DIRS:-}"

TARGET=""

relpath() {
    local target=$1
    local base=$2
    
    target=$(cd "$target" && pwd)
    base=$(cd "$base" && pwd)
    
    local common=$base
    local result=""
    
    while [[ "$target" != "$common"* ]]; do
        common=$(dirname "$common")
        result="../$result"
    done
    
    result+="${target#$common/}"
    echo "${result%/}"
}

# ----------------------
# Build plugin
# ----------------------
build_plugin() {
    local manifest="$1"
    local dir
    dir="$(dirname "$manifest")"
    local name
    name="$(basename "$dir")"
    
    if [ ! -f Cargo.lock ]; then
        echo "⚠️  Root Cargo.lock missing, generating..."
        cargo generate-lockfile --manifest-path "$WORKSPACE_TOML"
    fi
    
    echo ">>> Building plugin: $name"
    
    if [ -n "$TARGET" ]; then
        echo "🎯 Target: $TARGET"
        bash "$PLUGIN_BUILD" \
        --manifest "$manifest" \
        --profile "$PROFILE" \
        --target "$TARGET" || {
            echo "❌ Build failed for $name"
            exit 1
        }
    else
        bash "$PLUGIN_BUILD" \
        --manifest "$manifest" \
        --profile "$PROFILE" || {
            echo "❌ Build failed for $name"
            exit 1
        }
    fi
    
    echo "----------------------------"
}

PROFILE="debug"
export CARGO_TARGET_DIR="$(pwd)/target"
FROM_FILE=""
EX_SHADER_HOTRELOAD="${EX_SHADER_HOTRELOAD:-false}"

for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    case "$arg" in
        -hotreload)
            EX_SHADER_HOTRELOAD=1
        ;;
        release|debug)
            PROFILE="$arg"
        ;;
        --target)
            next_index=$((i+1))
            if [ $next_index -le $# ]; then
                TARGET="${!next_index}"
                i=$next_index
            else
                echo "Error: --target requires a target triple" >&2
                exit 1
            fi
        ;;
        from)
            next_index=$((i+1))
            if [ $next_index -le $# ]; then
                FROM_FILE="${!next_index}"
                i=$next_index
            else
                echo "Error: 'from' requires a file path" >&2
                exit 1
            fi
        ;;
        *)
        ;;
    esac
done

export EX_SHADER_HOTRELOAD

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
WORKSPACE_TOML="$SCRIPT_DIR/Cargo.toml"
PLUGIN_BUILD="$SCRIPT_DIR/plugin_build.sh"

if [ ! -f "$PLUGIN_BUILD" ]; then
    echo "❌ plugin_build.sh not found next to build_all.sh"
    exit 1
fi

# ----------------------
# Parse workspace members
# ----------------------
WORKSPACE_MEMBERS=$(awk '/members = \[/,/\]/' "$WORKSPACE_TOML" | sed 's/members = \[//' | tr -d '[]",' | xargs)

echo "📦 Workspace members: $WORKSPACE_MEMBERS"

if [ -n "$TARGET" ]; then
    echo "🧱 Cross compilation target: $TARGET"
fi

is_in_workspace() {
    local candidate="$1"
    for m in $WORKSPACE_MEMBERS; do
        if [ "$m" = "$candidate" ]; then
            return 0
        fi
    done
    return 1
}

# ----------------------
# Fast path: changed plugins
# ----------------------
if [ -n "${CHANGED_PLUGIN_DIRS}" ]; then
    echo "🎯 Building only changed plugins: ${CHANGED_PLUGIN_DIRS}"
    for dir in ${CHANGED_PLUGIN_DIRS}; do
        dir="${dir#./}"
        if [ ! -f "$dir/Cargo.toml" ]; then
            echo "⚠️  Skipping $dir - no Cargo.toml found"
            continue
        fi
        
        if ! is_in_workspace "$dir"; then
            echo "⚠️  Skipping $dir - not in [workspace].members"
            continue
        fi
        
        build_plugin "$dir/Cargo.toml"
    done
    
    echo "✅ Selected plugins built for profile: $PROFILE"
    exit 0
fi

if [ -n "$FROM_FILE" ]; then
    TARGET_DIR="$(dirname "$(realpath "$FROM_FILE")")"
    FOUND=""
    
    while [ "$TARGET_DIR" != "/" ]; do
        if [ -f "$TARGET_DIR/Cargo.toml" ]; then
            FOUND="$TARGET_DIR"
            break
        fi
        TARGET_DIR="$(dirname "$TARGET_DIR")"
    done
    
    if [ -z "$FOUND" ]; then
        echo "❌ No Cargo.toml found in parent directories of $FROM_FILE"
        exit 1
    fi
    
    RELATIVE_PATH=$(relpath "$FOUND" "$(pwd)")
    
    if ! is_in_workspace "$RELATIVE_PATH"; then
        echo "⚠️  Warning: $RELATIVE_PATH is not listed in [workspace].members, skipping"
        exit 0
    fi
    
    echo "============================"
    echo ">>> Building plugin from file: $(basename "$FOUND")"
    echo "============================"
    
    build_plugin "$FOUND/Cargo.toml"
    
    echo "✅ Built plugin: $(basename "$FOUND")"
    
else
    PLUGINS=$(find . -mindepth 2 -maxdepth 2 -type f -name Cargo.toml | sort)
    
    for manifest in $PLUGINS; do
        DIR=$(dirname "$manifest")
        RELATIVE_PATH=$(relpath "$DIR" "$(pwd)")
        
        if ! is_in_workspace "$RELATIVE_PATH"; then
            echo "⚠️  Warning: $RELATIVE_PATH is not listed in [workspace].members, skipping"
            continue
        fi
        
        echo "============================"
        echo ">>> Building plugin: $(basename "$DIR")"
        echo "============================"
        
        build_plugin "$DIR/Cargo.toml"
        
        echo "----------------------------"
    done
    
    echo "✅ All plugins built for profile: $PROFILE"
fi