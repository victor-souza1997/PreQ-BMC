"""Import isolation for the CROWN diagnostic driver. Standard library only.

The diagnostic must run with working directory /home/joao/code/preqbmc (the
loaders read relative output/ paths) while importing every PreQ-BMC module from
the selected worktree. Two things defeat PYTHONPATH there:

- crown_finder/affine_export.py runs sys.path.insert(0, "src"), which resolves
  against the working directory, i.e. the main checkout's src;
- the conda env has an editable install whose .pth file appends the main
  checkout's src.

utils/, symbolic_pp/ and benchmark/ have no __init__.py, so they are namespace
packages whose __path__ is recomputed from sys.path; preloading cannot pin them.
This module therefore installs a meta-path finder that resolves every
project-local name (each top-level directory or module under the worktree src,
plus the flat crown_finder modules) from the worktree only, with fixed search
locations, and refuses names that do not exist there instead of falling through
to sys.path. audit() then lists every loaded project module; anything outside
the worktree fails preflight.

The driver loads this file by path (importlib.util.spec_from_file_location),
never through sys.path, so the guard itself cannot be shadowed.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import site
import sys

CROWN_FINDER_MODULES = ("affine_export", "box_ledger", "crown_chain_export", "crown_chain_export_v2")


def project_names(src: Path) -> dict[str, Path]:
    """Every importable top-level name under src: directories and .py modules."""

    names: dict[str, Path] = {}
    for entry in sorted(src.iterdir()):
        if entry.name.startswith((".", "_")):
            continue
        if entry.is_dir() and entry.name.isidentifier():
            names[entry.name] = entry
        elif entry.suffix == ".py" and entry.stem.isidentifier():
            names[entry.stem] = entry
    return names


class WorktreeFinder(importlib.abc.MetaPathFinder):
    """Resolve project-local modules from one source tree, never from sys.path."""

    def __init__(self, src: Path) -> None:
        self.src = src
        self.tops = project_names(src)
        finder_dir = src / "scripts" / "crown_finder"
        self.flat = {name: finder_dir / f"{name}.py" for name in CROWN_FINDER_MODULES}

    def owns(self, name: str) -> bool:
        return name in self.flat or name.partition(".")[0] in self.tops

    def find_spec(self, name, path=None, target=None):
        if name in self.flat:
            return importlib.util.spec_from_file_location(name, self.flat[name])
        if name.partition(".")[0] not in self.tops:
            return None
        base = self.src.joinpath(*name.split("."))
        if base.is_dir():
            init = base / "__init__.py"
            if init.is_file():
                return importlib.util.spec_from_file_location(
                    name, init, submodule_search_locations=[str(base)])
            spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
            spec.submodule_search_locations = [str(base)]   # fixed list, not a recomputed namespace path
            return spec
        module = base.with_suffix(".py")
        if module.is_file():
            return importlib.util.spec_from_file_location(name, module)
        raise ModuleNotFoundError(f"{name} is project-local but does not exist under {self.src}", name=name)


def editable_pth_files(main_src: Path) -> list[dict[str, str]]:
    found = []
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for root in roots:
        directory = Path(root)
        if not directory.is_dir():
            continue
        for pth in sorted(directory.glob("*.pth")):
            text = pth.read_text(encoding="utf-8", errors="replace")
            if "preqbmc" in text or "editable" in pth.name:
                found.append({"path": str(pth), "content": text.strip(),
                              "points_to_main_checkout": str(main_src) in text})
    return found


def install(worktree_src: Path, main_src: Path) -> WorktreeFinder:
    """Pin project imports to worktree_src. Must run before any project import."""

    worktree_src, main_src = worktree_src.resolve(), main_src.resolve()
    finder = WorktreeFinder(worktree_src)
    early = sorted(name for name in sys.modules if finder.owns(name))
    if early:
        raise RuntimeError(f"project modules imported before the guard: {early}")

    def resolves_to_main(entry: str) -> bool:
        try:
            return Path(entry or ".").resolve() == main_src
        except OSError:
            return False

    sys.path[:] = [str(worktree_src)] + [p for p in sys.path if not resolves_to_main(p)
                                         and Path(p or ".").resolve() != worktree_src]
    sys.meta_path.insert(0, finder)
    return finder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _search_paths(module) -> list[Path]:
    raw = getattr(module, "__path__", None)
    try:
        entries = list(raw) if raw is not None else []
    except TypeError:          # some third-party modules expose a non-iterable __path__
        return []
    return [Path(p).resolve() for p in entries if isinstance(p, str)]


def audit(finder: WorktreeFinder, main_src: Path) -> dict:
    """Every loaded project module with its file and hash; violations outside the worktree."""

    main_src = main_src.resolve()
    modules, violations = [], []
    for name, module in sorted(sys.modules.items()):
        file = getattr(module, "__file__", None)
        paths = _search_paths(module)
        located = [Path(file).resolve()] if file else paths
        touches_main = any(p.is_relative_to(main_src) for p in located)
        if not (finder.owns(name) or touches_main):
            continue
        outside = [str(p) for p in located if not p.is_relative_to(finder.src)]
        record = {"module": name, "file": str(Path(file).resolve()) if file else None,
                  "search_locations": [str(p) for p in paths] or None,
                  "sha256": _sha256(Path(file)) if file and Path(file).is_file() else None}
        modules.append(record)
        if outside or not located:
            violations.append({"module": name, "outside_worktree": outside or ["<no location>"]})
    return {"worktree_src": str(finder.src), "main_checkout_src": str(main_src),
            "sys_path": list(sys.path), "meta_path_guard_first": sys.meta_path[0] is finder,
            "editable_pth": editable_pth_files(main_src),
            "modules": modules, "violations": violations, "passed": not violations}
