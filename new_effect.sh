#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --name PascalCaseName"
  exit 1
}

NEW_NAME_PASCAL=""

# arg parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      shift
      NEW_NAME_PASCAL="${1:-}"
      shift || true
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

if [[ -z "${NEW_NAME_PASCAL}" ]]; then
  echo "Error: --name is required"
  usage
fi

# enforce PascalCase: start with A-Z, then letters or digits
enforce_pascal() {
  local name="$1"
  if ! [[ "$name" =~ ^[A-Z][A-Za-z0-9]*$ ]]; then
    echo "Error: name must be PascalCase like ExposureBlur or Crossfade"
    exit 2
  fi
}
enforce_pascal "${NEW_NAME_PASCAL}"

# PascalCase -> snake_case (macOS compatible)
to_snake() {
  awk '{
    s=$0; out="";
    for(i=1;i<=length(s);i++){
      c=substr(s,i,1);
      if (c ~ /[A-Z]/) {
        if (i>1) {
          p=substr(s,i-1,1);
          if (p ~ /[a-z0-9]/) { out=out "_" }
          else if (p ~ /[A-Z]/ && i<length(s)) {
            n=substr(s,i+1,1);
            if (n ~ /[a-z]/) { out=out "_" }
          }
        }
        out=out tolower(c);
      } else {
        out=out c;
      }
    }
    print out
  }' <<<"$1"
}

# PascalCase -> UPPER (no underscores)
to_upper_flat() {
  tr '[:lower:]' '[:upper:]' <<<"$1"
}

NEW_NAME_SNAKE="$(to_snake "${NEW_NAME_PASCAL}")"
NEW_NAME_UPPER="$(to_upper_flat "${NEW_NAME_PASCAL}")"

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_DIR="${SCRIPT_DIR}/crossfade"
DEST_DIR="${SCRIPT_DIR}/${NEW_NAME_SNAKE}"

if [[ ! -d "${TEMPLATE_DIR}" ]]; then
  echo "Error: template folder '${TEMPLATE_DIR}' not found"
  exit 3
fi
if [[ -e "${DEST_DIR}" ]]; then
  echo "Error: destination '${DEST_DIR}' already exists"
  exit 3
fi

# copy template folder
cp -R "${TEMPLATE_DIR}" "${DEST_DIR}"

# sed -i wrapper for GNU and BSD
sed_inplace() {
  local expr="$1"
  local file="$2"
  if sed --version >/dev/null 2>&1; then
    sed -i -e "${expr}" "$file"
  else
    sed -i '' -e "${expr}" "$file"
  fi
}

# replace rules in one file
# crossfade -> <snake>
# Crossfade -> <Pascal>
# CROSSFADE -> <UPPER>
apply_replacements() {
  local file="$1"
  local snake="$2"
  local pascal="$3"
  local upper="$4"
  if [[ ! -f "$file" ]]; then
    echo "Warning: file not found: $file"
    return 0
  fi
  sed_inplace "s/crossfade/${snake}/g" "$file"
  sed_inplace "s/Crossfade/${pascal}/g" "$file"
  sed_inplace "s/CROSSFADE/${upper}/g" "$file"
}

# rename kernels/crossfade.rs to snake_case
KERNELS_DIR="${DEST_DIR}/src/kernels"
SRC_KERNEL_FILE="${KERNELS_DIR}/crossfade.rs"
DST_KERNEL_FILE="${KERNELS_DIR}/${NEW_NAME_SNAKE}.rs"
if [[ -f "${SRC_KERNEL_FILE}" ]]; then
  mv "${SRC_KERNEL_FILE}" "${DST_KERNEL_FILE}"
else
  echo "Warning: ${SRC_KERNEL_FILE} not found. Skipping rename."
fi

# rename shaders/crossfade.metal to snake_case
SHADERS_DIR="${DEST_DIR}/shaders"
SRC_SHADER_FILE="${SHADERS_DIR}/crossfade.metal"
DST_SHADER_FILE="${SHADERS_DIR}/${NEW_NAME_SNAKE}.metal"
if [[ -f "${SRC_SHADER_FILE}" ]]; then
  mv "${SRC_SHADER_FILE}" "${DST_SHADER_FILE}"
else
  echo "Warning: ${SRC_SHADER_FILE} not found. Skipping rename."
fi

# apply replacements
apply_replacements "${DST_KERNEL_FILE}"                   "${NEW_NAME_SNAKE}" "${NEW_NAME_PASCAL}" "${NEW_NAME_UPPER}"
apply_replacements "${KERNELS_DIR}/mod.rs"               "${NEW_NAME_SNAKE}" "${NEW_NAME_PASCAL}" "${NEW_NAME_UPPER}"
apply_replacements "${DEST_DIR}/src/premiere.rs"         "${NEW_NAME_SNAKE}" "${NEW_NAME_PASCAL}" "${NEW_NAME_UPPER}"
apply_replacements "${DST_SHADER_FILE}"                  "${NEW_NAME_SNAKE}" "${NEW_NAME_PASCAL}" "${NEW_NAME_UPPER}"
apply_replacements "${DEST_DIR}/build.rs"                "${NEW_NAME_SNAKE}" "${NEW_NAME_PASCAL}" "${NEW_NAME_UPPER}"
apply_replacements "${DEST_DIR}/src/lib.rs"              "${NEW_NAME_SNAKE}" "${NEW_NAME_PASCAL}" "${NEW_NAME_UPPER}"
apply_replacements "${DEST_DIR}/Cargo.toml"              "${NEW_NAME_SNAKE}" "${NEW_NAME_PASCAL}" "${NEW_NAME_UPPER}"

echo "Created '${DEST_DIR}'"
echo "Renamed:"
echo "  crossfade.rs    -> ${NEW_NAME_SNAKE}.rs"
echo "  crossfade.metal -> ${NEW_NAME_SNAKE}.metal"
echo "Replaced identifiers in:"
echo "  ${DST_KERNEL_FILE}"
echo "  ${KERNELS_DIR}/mod.rs"
echo "  ${DEST_DIR}/src/premiere.rs"
echo "  ${DST_SHADER_FILE}"
echo "  ${DEST_DIR}/build.rs"
echo "  ${DEST_DIR}/src/lib.rs"
echo "  ${DEST_DIR}/Cargo.toml"
echo "crossfade -> ${NEW_NAME_SNAKE}"
echo "Crossfade -> ${NEW_NAME_PASCAL}"
echo "CROSSFADE -> ${NEW_NAME_UPPER}"
