from __future__ import annotations

import argparse
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from scripts.common import bool_env, package_ignored, package_name, run


@dataclass(slots=True)
class PluginBuildOptions:
    manifest: Path
    profile: str = "debug"
    target: str = ""
    no_sign: bool = False


def target_platform(target: str) -> str:
    if target:
        if "windows" in target:
            return "windows"
        if "darwin" in target:
            return "macos"
        return "linux"

    host = platform.system()
    if host == "Darwin":
        return "macos"
    if host in {"Windows", "MINGW", "MSYS", "CYGWIN"}:
        return "windows"
    return "linux"


def cargo_build_base(manifest: Path, target_dir: Path, target: str) -> list[str]:
    cmd = [
        "cargo",
        "build",
        "--manifest-path",
        str(manifest),
        "--target-dir",
        str(target_dir),
        "--no-default-features",
    ]
    if target:
        cmd += ["--target", target]
    return cmd


def build_plugin(options: PluginBuildOptions) -> dict | None:
    manifest = options.manifest.resolve()
    crate_dir = manifest.parent

    if package_ignored(manifest):
        print(f"Skipping {crate_dir.name} (ignored)")
        return None

    name = package_name(manifest)
    binary_name = name.lower()

    target_dir = Path(os.environ.get("CARGO_TARGET_DIR", "target")).resolve()
    build_dir = target_dir / options.target if options.target else target_dir
    platform_name = target_platform(options.target)

    if platform_name == "macos" and options.profile == "release" and not options.target:
        run(["rustup", "target", "add", "aarch64-apple-darwin", "x86_64-apple-darwin"])

        run([*cargo_build_base(manifest, target_dir, "x86_64-apple-darwin"), "--release"])
        run([*cargo_build_base(manifest, target_dir, "aarch64-apple-darwin"), "--release"])
    else:
        cmd = cargo_build_base(manifest, target_dir, options.target)
        if options.profile == "release":
            cmd.append("--release")
        run(cmd)

    if platform_name == "windows":
        outdir = build_dir / options.profile
        dll = outdir / f"{binary_name}.dll"
        aex = outdir / f"{name}.aex"

        shutil.copy2(dll, aex)

        return {
            "platform": "windows",
            "name": name,
            "aex": str(aex),
            "pdb": str(outdir / f"{binary_name}.pdb"),
        }

    if platform_name == "macos":
        outdir = build_dir / options.profile / f"{name}.plugin"

        shutil.rmtree(outdir, ignore_errors=True)
        (outdir / "Contents" / "MacOS").mkdir(parents=True, exist_ok=True)
        (outdir / "Contents" / "Resources").mkdir(parents=True, exist_ok=True)

        return {
            "platform": "macos",
            "name": name,
            "plugin": str(outdir),
        }

    return None


def parse_args() -> PluginBuildOptions:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--target", default="")
    parser.add_argument("--no-sign", action="store_true")
    args = parser.parse_args()

    return PluginBuildOptions(
        manifest=Path(args.manifest),
        profile=args.profile,
        target=args.target,
        no_sign=args.no_sign or bool_env("NO_SIGN", False),
    )


def main() -> None:
    build_plugin(parse_args())


if __name__ == "__main__":
    main()