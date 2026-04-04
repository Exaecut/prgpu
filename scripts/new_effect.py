import argparse
import shutil
from pathlib import Path

from utils import replace_in_file
from naming import enforce_pascal, to_snake, to_upper_flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    pascal = args.name
    enforce_pascal(pascal)

    snake = to_snake(pascal)
    upper = to_upper_flat(pascal)

    script_dir = Path(__file__).parent
    template_dir = script_dir / "crossfade"
    dest_dir = script_dir / snake

    if not template_dir.exists():
        raise RuntimeError(f"Template not found: {template_dir}")

    if dest_dir.exists():
        raise RuntimeError(f"Destination exists: {dest_dir}")

    shutil.copytree(template_dir, dest_dir)

    kernels_dir = dest_dir / "src/kernels"
    shaders_dir = dest_dir / "shaders"

    src_kernel = kernels_dir / "crossfade.rs"
    dst_kernel = kernels_dir / f"{snake}.rs"
    if src_kernel.exists():
        src_kernel.rename(dst_kernel)

    src_shader = shaders_dir / "crossfade.metal"
    dst_shader = shaders_dir / f"{snake}.metal"
    if src_shader.exists():
        src_shader.rename(dst_shader)

    replacements = {
        "crossfade": snake,
        "Crossfade": pascal,
        "CROSSFADE": upper,
    }

    files = [
        dst_kernel,
        kernels_dir / "mod.rs",
        dest_dir / "src/premiere.rs",
        dst_shader,
        dest_dir / "build.rs",
        dest_dir / "src/lib.rs",
        dest_dir / "Cargo.toml",
    ]

    for f in files:
        replace_in_file(f, replacements)

    print(f"Created '{dest_dir}'")


if __name__ == "__main__":
    main()
