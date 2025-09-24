#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./plugin_build.sh --manifest /path/to/Cargo.toml --profile release
# Options:
#   --manifest PATH   Path to Cargo.toml of the plugin (required)
#   --profile PROFILE Build profile (debug|release), default: debug
#   --no-install      Skip installation step
#   --no-sign         Skip codesigning step (macOS only)

# ----------------------
# Parse args
# ----------------------
PROFILE="debug"
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
    /^\[/ && in_meta { exit }   # stop if another section begins
    in_meta && /^ignore/ {
        val=$3
        gsub(/#.*/, "", val)  # strip comments
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
OS="$(uname -s)"

# ----------------------
# Select backend
# ----------------------
case "$OS" in
    Darwin) BACKEND="metal" ;;
    MINGW*|CYGWIN*|MSYS*) BACKEND="cuda" ;;
    *) echo "❌ Unsupported OS: $OS" && exit 1 ;;
esac

echo "🔧 Selected GPU backend   : $BACKEND"

# ----------------------
# Build
# ----------------------
if [[ "$OS" == "Darwin" && "$PROFILE" == "release" ]]; then
    rustup target add aarch64-apple-darwin x86_64-apple-darwin
    cargo build --release --manifest-path "$MANIFEST" --target-dir "$TARGET_DIR" --target x86_64-apple-darwin --features "prgpu/$BACKEND"
    cargo build --release --manifest-path "$MANIFEST" --target-dir "$TARGET_DIR" --target aarch64-apple-darwin --features "prgpu/$BACKEND"
else
    if [[ "$PROFILE" == "release" ]]; then
        cargo build --release --manifest-path "$MANIFEST" --target-dir "$TARGET_DIR" --no-default-features --features "prgpu/$BACKEND"
    else
        cargo build --manifest-path "$MANIFEST" --target-dir "$TARGET_DIR" --no-default-features --features "prgpu/$BACKEND"
    fi
fi

# ----------------------
# macOS packaging
# ----------------------
if [[ "$OS" == "Darwin" ]]; then
    OUTDIR="$TARGET_DIR/$PROFILE/$PLUGIN_NAME.plugin"
    echo "Creating bundle at $OUTDIR"
    rm -rf "$OUTDIR"
    mkdir -p "$OUTDIR/Contents/MacOS" "$OUTDIR/Contents/Resources"
    
    PLIST="$OUTDIR/Contents/Info.plist"
    /usr/libexec/PlistBuddy -c "add CFBundlePackageType string SPFX" "$PLIST"
    /usr/libexec/PlistBuddy -c "add CFBundleSignature string PrMr" "$PLIST"
    /usr/libexec/PlistBuddy -c "add CFBundleIdentifier string $BUNDLE_IDENTIFIER" "$PLIST"
    echo "SPFXPrMr" > "$OUTDIR/Contents/PkgInfo"
    
    if [[ "$PROFILE" == "release" ]]; then
        cp "$TARGET_DIR/x86_64-apple-darwin/release/${BINARY_NAME}.rsrc" "$OUTDIR/Contents/Resources/${PLUGIN_NAME}.rsrc"
        lipo "$TARGET_DIR/x86_64-apple-darwin/release/lib${BINARY_NAME}.dylib" \
             "$TARGET_DIR/aarch64-apple-darwin/release/lib${BINARY_NAME}.dylib" \
             -create -output "$OUTDIR/Contents/MacOS/$PLUGIN_NAME"
    else
        cp "$TARGET_DIR/$PROFILE/${BINARY_NAME}.rsrc" "$OUTDIR/Contents/Resources/${PLUGIN_NAME}.rsrc"
        cp "$TARGET_DIR/$PROFILE/lib${BINARY_NAME}.dylib" "$OUTDIR/Contents/MacOS/$PLUGIN_NAME"
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
if [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* ]]; then
    OUTDIR="$TARGET_DIR/$PROFILE"
    DLL="$OUTDIR/${BINARY_NAME}.dll"
    AEX="$OUTDIR/${PLUGIN_NAME}.aex"
    cp "$DLL" "$AEX"
    
    if [[ "$NO_INSTALL" == false ]]; then
        DEST="/c/Program Files/Adobe/Common/Plug-ins/7.0/MediaCore/Exaecut"
        echo "Installing to $DEST"
        mkdir -p "$DEST"
        cp "$AEX" "$DEST/${PLUGIN_NAME}.aex"
        cp "$OUTDIR/${BINARY_NAME}.pdb" "$DEST/${PLUGIN_NAME}.pdb" || true
    fi
    
    echo "✅ Windows build complete: $AEX"
fi
