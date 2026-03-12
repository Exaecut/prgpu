#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./plugin_build.sh --manifest /path/to/Cargo.toml --profile release
# Options:
#   --manifest PATH   Path to Cargo.toml of the plugin (required)
#   --profile PROFILE Build profile (debug|release), default: debug
#   --target TARGET   Rust target triple (optional)
#   --no-install      Skip installation step
#   --no-sign         Skip codesigning step (macOS only)

# ----------------------
# Parse args
# ----------------------
PROFILE="debug"
TARGET=""
NO_INSTALL="${NO_INSTALL:-false}"
NO_SIGN="${NO_SIGN:-false}"
MANIFEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)
            MANIFEST="$2"
            shift 2
        ;;
        --profile)
            PROFILE="$2"
            shift 2
        ;;
        --target)
            TARGET="$2"
            shift 2
        ;;
        --no-install)
            NO_INSTALL=true
            shift
        ;;
        --no-sign)
            NO_SIGN=true
            shift
        ;;
        *)
            echo "❌ Unknown argument: $1" >&2
            exit 1
        ;;
    esac
done

if [[ -z "$MANIFEST" ]]; then
    echo "❌ Missing --manifest argument"
    exit 1
fi

SCRIPT_DIR="$(dirname "$(realpath "$MANIFEST")")"

# ----------------------
# Check ignore flag
# ----------------------
IGNORE=$(awk '
    /^\[package.metadata.plugin\]/ { in_meta=1; next }
    /^\[/ && in_meta { exit }
    in_meta && /^ignore/ {
        val=$3
        gsub(/#.*/, "", val)
        print val
        exit
    }
' "$SCRIPT_DIR/Cargo.toml")

if [[ "$IGNORE" == "true" ]]; then
    echo "⏭️  Skipping build for $(basename "$SCRIPT_DIR") (metadata.ignore = true)"
    exit 0
fi

# ----------------------
# Get plugin info
# ----------------------
PKG_ID=$(cargo pkgid --manifest-path "$MANIFEST")
PLUGIN_NAME="${PKG_ID%#*}"
PLUGIN_NAME="${PLUGIN_NAME##*/}"
BINARY_NAME=$(echo "$PLUGIN_NAME" | tr '[:upper:]' '[:lower:]')
BUNDLE_IDENTIFIER="io.exaecut.transitions.${BINARY_NAME}"

echo "🔧 Parsed plugin name     : $PLUGIN_NAME"
echo "🔧 Derived binary name    : $BINARY_NAME"
echo "🔧 Bundle identifier      : $BUNDLE_IDENTIFIER"

TARGET_DIR="${CARGO_TARGET_DIR:-target}"

BUILD_DIR="$TARGET_DIR"
if [[ -n "$TARGET" ]]; then
    BUILD_DIR="$TARGET_DIR/$TARGET"
fi

HOST_OS="$(uname -s)"

# ----------------------
# Determine target platform
# ----------------------
TARGET_PLATFORM=""

if [[ -n "$TARGET" ]]; then
    case "$TARGET" in
        *windows*) TARGET_PLATFORM="windows" ;;
        *darwin*) TARGET_PLATFORM="macos" ;;
        *) TARGET_PLATFORM="linux" ;;
    esac
else
    case "$HOST_OS" in
        Darwin) TARGET_PLATFORM="macos" ;;
        MINGW*|CYGWIN*|MSYS*) TARGET_PLATFORM="windows" ;;
        *) TARGET_PLATFORM="linux" ;;
    esac
fi

# ----------------------
# Cargo target args
# ----------------------
CARGO_TARGET_ARGS=()

if [[ -n "$TARGET" ]]; then
    CARGO_TARGET_ARGS+=(--target "$TARGET")
fi

# ----------------------
# Build
# ----------------------
if [[ "$TARGET_PLATFORM" == "macos" && "$PROFILE" == "release" && -z "$TARGET" ]]; then
    rustup target add aarch64-apple-darwin x86_64-apple-darwin
    
    cargo build \
    --release \
    --manifest-path "$MANIFEST" \
    --target-dir "$TARGET_DIR" \
    --target x86_64-apple-darwin \
    
    cargo build \
    --release \
    --manifest-path "$MANIFEST" \
    --target-dir "$TARGET_DIR" \
    --target aarch64-apple-darwin \
    
else
    if [[ "$PROFILE" == "release" ]]; then
        cargo build \
        --release \
        --manifest-path "$MANIFEST" \
        --target-dir "$TARGET_DIR" \
        "${CARGO_TARGET_ARGS[@]}" \
        --no-default-features \
        
    else
        cargo build \
        --manifest-path "$MANIFEST" \
        --target-dir "$TARGET_DIR" \
        "${CARGO_TARGET_ARGS[@]}" \
        --no-default-features \
        
    fi
fi

# ----------------------
# macOS packaging
# ----------------------
if [[ "$TARGET_PLATFORM" == "macos" ]]; then
    
    OUTDIR="$BUILD_DIR/$PROFILE/$PLUGIN_NAME.plugin"
    
    echo "Creating bundle at $OUTDIR"
    
    rm -rf "$OUTDIR"
    mkdir -p "$OUTDIR/Contents/MacOS" "$OUTDIR/Contents/Resources"
    
    PLIST="$OUTDIR/Contents/Info.plist"
    
    /usr/libexec/PlistBuddy -c "add CFBundlePackageType string SPFX" "$PLIST"
    /usr/libexec/PlistBuddy -c "add CFBundleSignature string PrMr" "$PLIST"
    /usr/libexec/PlistBuddy -c "add CFBundleIdentifier string $BUNDLE_IDENTIFIER" "$PLIST"
    
    echo "SPFXPrMr" > "$OUTDIR/Contents/PkgInfo"
    
    if [[ "$PROFILE" == "release" && -z "$TARGET" ]]; then
        cp "$TARGET_DIR/x86_64-apple-darwin/release/${BINARY_NAME}.rsrc" \
        "$OUTDIR/Contents/Resources/${PLUGIN_NAME}.rsrc"
        
        lipo \
        "$TARGET_DIR/x86_64-apple-darwin/release/lib${BINARY_NAME}.dylib" \
        "$TARGET_DIR/aarch64-apple-darwin/release/lib${BINARY_NAME}.dylib" \
        -create \
        -output "$OUTDIR/Contents/MacOS/$PLUGIN_NAME"
        
    else
        cp "$BUILD_DIR/$PROFILE/${BINARY_NAME}.rsrc" \
        "$OUTDIR/Contents/Resources/${PLUGIN_NAME}.rsrc"
        
        cp "$BUILD_DIR/$PROFILE/lib${BINARY_NAME}.dylib" \
        "$OUTDIR/Contents/MacOS/$PLUGIN_NAME"
    fi
    
    if [[ "$NO_SIGN" == false ]]; then
        CERT=$(security find-identity -v -p codesigning | grep -m 1 "Apple Development" | awk '{print $2}' || true)
        
        if [[ -n "$CERT" ]]; then
            echo "Signing..."
            codesign --options runtime --timestamp -strict --sign "$CERT" "$OUTDIR"
        fi
    fi
    
    if [[ "$NO_INSTALL" == false ]]; then
        DEST="/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/Exaecut/${PLUGIN_NAME}.plugin"
        
        echo "Installing to $DEST"
        
        sudo rm -rf "$DEST"
        sudo cp -R "$OUTDIR" "$DEST"
    fi
    
    echo "✅ macOS build complete: $OUTDIR"
fi

# ----------------------
# Windows packaging
# ----------------------
if [[ "$TARGET_PLATFORM" == "windows" ]]; then
    
    OUTDIR="$BUILD_DIR/$PROFILE"
    DLL="$OUTDIR/${BINARY_NAME}.dll"
    AEX="$OUTDIR/${PLUGIN_NAME}.aex"
    
    cp "$DLL" "$AEX"
    
    if [[ "$NO_INSTALL" == false && "$HOST_OS" != "Linux" ]]; then
        DEST="/c/Program Files/Adobe/Common/Plug-ins/7.0/MediaCore/Exaecut"
        
        echo "Installing to $DEST"
        
        mkdir -p "$DEST"
        
        cp "$AEX" "$DEST/${PLUGIN_NAME}.aex"
        cp "$OUTDIR/${BINARY_NAME}.pdb" "$DEST/${PLUGIN_NAME}.pdb" || true
    fi
    
    echo "✅ Windows build complete: $AEX"
fi