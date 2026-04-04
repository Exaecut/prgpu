from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from scripts.common import manifest_dirs, relpath, workspace_members
from scripts.plugin_build import PluginBuildOptions, build_plugin


@dataclass(slots=True)
class BuildAllOptions:
    profile: str = "debug"
    target: str = ""
    from_file: str = ""
    display_effects_selector: bool = False


def parse_args(argv: list[str]) -> BuildAllOptions:
    opts = BuildAllOptions()

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg in {"debug", "release"}:
            opts.profile = arg

        elif arg == "--target":
            i += 1
            if i >= len(argv):
                raise SystemExit("Error: --target requires a value")
            opts.target = argv[i]

        elif arg == "from":
            i += 1
            if i >= len(argv):
                raise SystemExit("Error: from requires a file path")
            opts.from_file = argv[i]

        elif arg in {"true", "false"}:
            opts.display_effects_selector = arg == "true"

        elif arg == "--effects-selector":
            opts.display_effects_selector = True

        elif arg == "--no-effects-selector":
            opts.display_effects_selector = False

        i += 1

    return opts


def find_manifest_from_file(file: Path) -> Path | None:
    current = file.resolve()

    if current.is_file():
        current = current.parent

    while True:
        manifest = current / "Cargo.toml"
        if manifest.exists():
            return manifest

        if current.parent == current:
            return None

        current = current.parent


def _manifest_relpath(manifest: Path, root: Path) -> str:
    return relpath(manifest.parent, root)


def _sorted_workspace_manifests(manifests: Iterable[Path], root: Path) -> list[Path]:
    return sorted(manifests, key=lambda m: _manifest_relpath(m, root))


def _truncate(text: str, max_len: int) -> str:
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len == 1:
        return "…"
    return text[: max_len - 1].rstrip() + "…"


def _humanize_selection(items: list[str]) -> str:
    if not items:
        return "aucun"

    names = [Path(item).name for item in items]
    preview = ", ".join(names[:6])
    if len(names) > 6:
        preview += f" +{len(names) - 6}"
    return preview


def _choose_manifests_prompt_toolkit(
    candidates: list[Path],
    root: Path,
    initial_selected: set[str],
) -> list[Path]:
    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.formatted_text import AnyFormattedText
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.output import ColorDepth
        from prompt_toolkit.styles import Style
    except ImportError as exc:
        raise SystemExit(
            "Interactive selector requires 'prompt_toolkit'. "
            "Install it with: pip install prompt_toolkit"
        ) from exc

    class SelectorState:
        def __init__(self, items: list[Path], selected_relpaths: set[str]) -> None:
            self.items = items
            self.selected = [
                _manifest_relpath(item, root) in selected_relpaths for item in items
            ]
            self.cursor = 0
            self.scroll = 0
            self.max_visible = min(8, len(items))
            self.result: list[Path] | None = None

        def selected_items(self) -> list[Path]:
            return [item for item, enabled in zip(self.items, self.selected) if enabled]

        def all_selected(self) -> bool:
            return bool(self.items) and all(self.selected)

        def toggle_current(self) -> None:
            if not self.items:
                return
            self.selected[self.cursor] = not self.selected[self.cursor]

        def toggle_all(self) -> None:
            if not self.items:
                return
            target = not self.all_selected()
            self.selected = [target] * len(self.items)

        def move(self, delta: int) -> None:
            if not self.items:
                return

            self.cursor = max(0, min(len(self.items) - 1, self.cursor + delta))

            if self.cursor < self.scroll:
                self.scroll = self.cursor
            elif self.cursor >= self.scroll + self.max_visible:
                self.scroll = self.cursor - self.max_visible + 1

        def visible_slice(self) -> tuple[int, int]:
            start = self.scroll
            end = min(len(self.items), start + self.max_visible)
            return start, end

    state = SelectorState(candidates, initial_selected)

    def term_width() -> int:
        try:
            from prompt_toolkit.application.current import get_app

            return max(50, get_app().output.get_size().columns)
        except Exception:
            return 100

    def fit(text: str, limit: int) -> str:
        if limit <= 0:
            return ""
        text = text.replace("\n", " ")
        if len(text) <= limit:
            return text
        if limit == 1:
            return "…"
        return text[: limit - 1].rstrip() + "…"

    def render_title() -> AnyFormattedText:
        width = term_width()
        title = " Sélection des effets "
        subtitle = "Espace: bascule | A: tout basculer | Entrée: valider | Q: quitter"
        available = max(0, width - len(title) - 2)
        return [
            ("class:title", title),
            ("", fit(subtitle, available)),
        ]

    def render_selection_info() -> AnyFormattedText:
        width = term_width()
        selected = state.selected_items()
        base = f"{len(selected)}/{len(state.items)} sélectionné(s)"
        if len(state.items) <= 4:
            return [("class:muted", fit(base, width))]

        recap = _humanize_selection(
            [_manifest_relpath(item, root) for item in selected]
        )
        line = f"{base} | {recap}"
        return [("class:muted", fit(line, width))]

    def render_list() -> AnyFormattedText:
        width = term_width()
        start, end = state.visible_slice()
        lines: list[tuple[str, str]] = []

        for index in range(start, end):
            manifest = state.items[index]
            rel = _manifest_relpath(manifest, root)
            label = rel
            prefix = "❯" if index == state.cursor else " "
            checkbox = "[x]" if state.selected[index] else "[ ]"

            available = max(10, width - 6)
            text = fit(label, available)

            if index == state.cursor:
                lines.append(("class:cursor", f"{prefix} {checkbox} "))
                lines.append(("class:cursor", text))
            elif state.selected[index]:
                lines.append(("class:selected", f"{prefix} {checkbox} "))
                lines.append(("class:selected", text))
            else:
                lines.append(("class:normal", f"{prefix} {checkbox} "))
                lines.append(("class:normal", text))

            if index != end - 1:
                lines.append(("", "\n"))

        if not lines:
            return [("", "Aucun membre de l'espace de travail détecté.")]

        return lines

    def render_footer() -> AnyFormattedText:
        width = term_width()
        line1 = "↑ ↓ naviguer | Espace sélectionner | A tout basculer | Entrée valider | Q quitter"

        return [
            ("class:footer", fit(line1, width)),
        ]

    kb = KeyBindings()

    @kb.add("up")
    def _up(event) -> None:
        state.move(-1)

    @kb.add("down")
    def _down(event) -> None:
        state.move(1)

    @kb.add("space")
    def _space(event) -> None:
        state.toggle_current()

    @kb.add("a")
    def _a(event) -> None:
        state.toggle_all()

    @kb.add("enter")
    def _enter(event) -> None:
        state.result = state.selected_items()
        event.app.exit(result=state.result)

    @kb.add("q")
    @kb.add("escape")
    def _quit(event) -> None:
        state.result = []
        event.app.exit(result=[])

    style = Style.from_dict(
        {
            "title": "bold",
            "muted": "ansigray",
            "cursor": "bold",
            "selected": "",
            "normal": "",
            "footer": "bold",
            "footer_dim": "ansigray",
        }
    )

    root_container = HSplit(
        [
            Window(
                FormattedTextControl(render_title), height=1, dont_extend_height=True
            ),
            Window(
                FormattedTextControl(render_selection_info),
                height=1,
                dont_extend_height=True,
            ),
            Window(
                FormattedTextControl(render_list),
                height=state.max_visible,
                dont_extend_height=True,
            ),
            Window(
                FormattedTextControl(render_footer), height=2, dont_extend_height=True
            ),
        ]
    )

    app = Application(
        layout=Layout(root_container),
        key_bindings=kb,
        style=style,
        full_screen=True,
        mouse_support=False,
        color_depth=ColorDepth.TRUE_COLOR,
    )

    result = app.run()
    return result or []


def choose_manifests(
    candidates: list[Path],
    root: Path,
    initial_selected: set[str],
    enable_ui: bool,
) -> list[Path]:
    if not candidates:
        return []

    if not enable_ui or len(candidates) == 1:
        return [m for m in candidates if _manifest_relpath(m, root) in initial_selected]

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print("Terminal non interactif, sélection automatique appliquée.")
        return [m for m in candidates if _manifest_relpath(m, root) in initial_selected]

    selected = _choose_manifests_prompt_toolkit(candidates, root, initial_selected)
    return selected


def install_all(built: list[dict]):
    if not built:
        return

    platform_name = built[0]["platform"]

    if platform_name == "windows":
        dest = Path(r"C:\Program Files\Adobe\Common\Plug-ins\7.0\MediaCore\Exaecut")

        def install():
            dest.mkdir(parents=True, exist_ok=True)

            for item in built:
                aex = Path(item["aex"])
                pdb = Path(item["pdb"])

                if aex.exists():
                    shutil.copy2(aex, dest / f"{item['name']}.aex")

                if pdb.exists():
                    shutil.copy2(pdb, dest / f"{item['name']}.pdb")

        try:
            install()
            print("Install OK (sans élévation)")
        except PermissionError:
            print("Élévation requise")

            script = Path("scripts/_install_batch.py")

            script.write_text(
                "import shutil\n"
                "from pathlib import Path\n\n"
                f"dest = Path(r'{dest}')\n"
                "dest.mkdir(parents=True, exist_ok=True)\n\n"
                f"built = {built}\n\n"
                "for item in built:\n"
                "    aex = Path(item['aex'])\n"
                "    pdb = Path(item['pdb'])\n"
                "    if aex.exists(): shutil.copy2(aex, dest / f\"{item['name']}.aex\")\n"
                "    if pdb.exists(): shutil.copy2(pdb, dest / f\"{item['name']}.pdb\")\n"
            )

            subprocess.run(
                [
                    "powershell",
                    "-Command",
                    f'Start-Process "{sys.executable}" -ArgumentList "{script}" -Verb RunAs',
                ],
                check=True,
            )

            print("Install OK (avec élévation)")

    elif platform_name == "macos":
        dest_root = Path(
            "/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore/Exaecut"
        )

        for item in built:
            src = Path(item["plugin"])
            dest = dest_root / f"{item['name']}.plugin"

            shutil.rmtree(dest, ignore_errors=True)
            shutil.copytree(src, dest)

        print("Installation macOS OK")


def main() -> None:
    opts = parse_args(sys.argv[1:])

    root = Path(__file__).resolve().parent.parent
    workspace = workspace_members(root / "Cargo.toml")
    manifests = _sorted_workspace_manifests(manifest_dirs(root), root)

    workspace_manifests = []
    for manifest in manifests:
        rel = _manifest_relpath(manifest, root)
        if rel in workspace:
            workspace_manifests.append(manifest)

    if not workspace_manifests:
        print("No plugins selected")
        return

    selected: list[Path] = []

    if opts.from_file:
        target_file = Path(opts.from_file)
        manifest = find_manifest_from_file(target_file)

        if manifest is None:
            print(
                f"Aucun Cargo.toml trouvé pour {opts.from_file}. "
                "Compilation classique de tous les membres de l'espace de travail."
            )
            selected = workspace_manifests
        else:
            rel = _manifest_relpath(manifest, root)

            if rel not in workspace:
                print(
                    f"Le fichier {opts.from_file} n'appartient à aucun membre de l'espace de travail. "
                    "Compilation classique de tous les membres de l'espace de travail."
                )
                selected = workspace_manifests
            else:
                if opts.display_effects_selector:
                    selected = workspace_manifests
                else:
                    selected = [manifest]
    else:
        selected = workspace_manifests

    if opts.display_effects_selector:
        chosen = choose_manifests(
            candidates=workspace_manifests,
            root=root,
            initial_selected=set(),
            enable_ui=True,
        )
        if not chosen:
            print("No plugins selected")
            return
        selected = chosen

    if not selected:
        print("No plugins selected")
        return

    built = []

    for manifest in selected:
        print(f"Building {manifest.parent.name}")

        result = build_plugin(
            PluginBuildOptions(
                manifest=manifest,
                profile=opts.profile,
                target=opts.target,
            )
        )

        if result:
            built.append(result)

    install_all(built)

    print("Build complete")


if __name__ == "__main__":
    main()
