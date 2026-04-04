import os
import sys
import subprocess
import platform
from pathlib import Path


def run(cmd, cwd=None, env=None):
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_file(path: Path, content: str):
    path.write_text(content, encoding="utf-8")


def replace_in_file(path: Path, replacements: dict[str, str]):
    if not path.exists():
        print(f"Warning: file not found: {path}")
        return

    content = read_file(path)
    for k, v in replacements.items():
        content = content.replace(k, v)
    write_file(path, content)


def is_windows():
    return platform.system().lower().startswith("win")


def is_macos():
    return platform.system() == "Darwin"


def find_cargo_manifests(root: Path):
    return list(root.glob("*/*/Cargo.toml"))


def ensure_lockfile(manifest: Path):
    crate_dir = manifest.parent
    lockfile = crate_dir / "Cargo.lock"
    if not lockfile.exists():
        print(f"🔧 Generating lockfile for {crate_dir}")
        run(f'cargo generate-lockfile --manifest-path "{manifest}"')
