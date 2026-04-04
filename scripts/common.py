from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any
import tomllib


def run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    print("+", " ".join(str(part) for part in cmd))
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def workspace_members(workspace_toml: Path) -> set[str]:
    data = read_toml(workspace_toml)
    members = data.get("workspace", {}).get("members", [])
    return {Path(m).as_posix().rstrip("/") for m in members}


def manifest_dirs(root: Path) -> list[Path]:
    return sorted(
        [
            p / "Cargo.toml"
            for p in root.iterdir()
            if p.is_dir() and (p / "Cargo.toml").exists()
        ]
    )


def relpath(target: Path, base: Path) -> str:
    return target.resolve().relative_to(base.resolve()).as_posix()


def find_manifest_ancestor(path: Path) -> Path | None:
    current = path.resolve()
    if current.is_file():
        current = current.parent

    while True:
        manifest = current / "Cargo.toml"
        if manifest.exists():
            return manifest
        if current.parent == current:
            return None
        current = current.parent


def package_ignored(manifest: Path) -> bool:
    data = read_toml(manifest)
    plugin_meta = data.get("package", {}).get("metadata", {}).get("plugin", {})
    return bool(plugin_meta.get("ignore", False))


def package_name(manifest: Path) -> str:
    data = read_toml(manifest)
    pkg = data.get("package", {})
    name = pkg.get("name")
    if not name:
        raise RuntimeError(f"Missing [package].name in {manifest}")
    return name


def bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
