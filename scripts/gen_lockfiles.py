from pathlib import Path
from utils import find_cargo_manifests, ensure_lockfile


def main():
    root = Path(".").resolve()

    for manifest in find_cargo_manifests(root):
        ensure_lockfile(manifest)


if __name__ == "__main__":
    main()
