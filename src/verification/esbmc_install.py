from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from typing import Any


GITHUB_LATEST_RELEASE_API = "https://api.github.com/repos/esbmc/esbmc/releases/latest"
ENV_ESBMC_EXECUTABLE = "PREQBMC_ESBMC"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_bin_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".local" / "bin"


def local_esbmc_path(root: Path | None = None) -> Path:
    name = "esbmc.exe" if os.name == "nt" else "esbmc"
    return local_bin_dir(root) / name


def resolve_esbmc_executable(
    executable: str | None = None,
    *,
    root: Path | None = None,
) -> str | None:
    """Resolve ESBMC from explicit config, repo-local install, or PATH."""

    env_path = os.environ.get(ENV_ESBMC_EXECUTABLE)
    if env_path:
        return env_path
    if executable and executable != "esbmc":
        return executable
    local_path = local_esbmc_path(root)
    if local_path.exists():
        return str(local_path)
    return shutil.which(executable or "esbmc")


def _platform_asset_pattern() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "*linux*.zip"
    if system == "darwin":
        return "*macos*.zip"
    if system == "windows":
        return "*windows*.zip"
    raise RuntimeError(
        "Cannot infer an ESBMC release asset for "
        f"{platform.system()} {platform.machine()}. Pass --asset-pattern explicitly."
    )


def _read_json_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "preqbmc-esbmc-installer",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _select_asset(release: dict[str, Any], asset_pattern: str) -> dict[str, Any]:
    assets = release.get("assets", [])
    matches = [
        asset
        for asset in assets
        if fnmatch.fnmatch(str(asset.get("name", "")).lower(), asset_pattern.lower())
    ]
    if not matches:
        available = ", ".join(str(asset.get("name")) for asset in assets) or "(none)"
        raise RuntimeError(f"No ESBMC release asset matched {asset_pattern!r}. Available assets: {available}")
    return matches[0]


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "preqbmc-esbmc-installer"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _find_esbmc_binary(directory: Path) -> Path:
    candidates = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower() not in {"esbmc", "esbmc.exe"}:
            continue
        candidates.append(path)
    if not candidates:
        raise RuntimeError(f"Could not find an ESBMC executable in {directory}")
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates[0]


def _write_shim(target: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if os.name == "nt":
        shutil.copy2(target, destination)
        return
    destination.write_text(f"#!/bin/sh\nexec {str(target)!r} \"$@\"\n", encoding="utf-8")
    destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def install_esbmc(
    *,
    root: Path | None = None,
    asset_pattern: str | None = None,
    release_api_url: str = GITHUB_LATEST_RELEASE_API,
    force: bool = False,
) -> dict[str, Any]:
    """Download the latest ESBMC release and expose it under .local/bin."""

    root = root or repo_root()
    local_binary = local_esbmc_path(root)
    if local_binary.exists() and not force:
        return {
            "installed": False,
            "reason": "already-installed",
            "esbmc_path": str(local_binary),
        }

    release = _read_json_url(release_api_url)
    selected_pattern = asset_pattern or _platform_asset_pattern()
    asset = _select_asset(release, selected_pattern)
    tag = str(release.get("tag_name") or release.get("name") or "latest")
    asset_name = str(asset["name"])
    download_url = str(asset["browser_download_url"])
    install_dir = root / ".local" / "esbmc" / tag / Path(asset_name).stem
    archive_dir = root / ".local" / "downloads"
    archive_path = archive_dir / asset_name

    if install_dir.exists() and force:
        shutil.rmtree(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="preqbmc-esbmc-") as temp_name:
        temp_archive = Path(temp_name) / asset_name
        _download(download_url, temp_archive)
        shutil.copy2(temp_archive, archive_path)
        with zipfile.ZipFile(temp_archive) as archive:
            archive.extractall(install_dir)

    binary = _find_esbmc_binary(install_dir)
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _write_shim(binary, local_binary)

    version = subprocess.run(
        [str(local_binary), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
        check=False,
    )
    return {
        "installed": True,
        "release": tag,
        "asset": asset_name,
        "download_url": download_url,
        "install_dir": str(install_dir),
        "archive_path": str(archive_path),
        "esbmc_path": str(local_binary),
        "version_output": version.stdout.strip(),
        "return_code": int(version.returncode),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install ESBMC into this repository under .local/.")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--asset-pattern", default=None, help="Release asset glob, e.g. '*linux*.zip'.")
    parser.add_argument("--force", action="store_true", help="Replace an existing repo-local ESBMC install.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = install_esbmc(
        root=args.repo_root.resolve(),
        asset_pattern=args.asset_pattern,
        force=bool(args.force),
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"ESBMC path: {result['esbmc_path']}")
        if result.get("installed"):
            print(f"Installed {result['release']} from {result['asset']}")
            if result.get("version_output"):
                print(result["version_output"])
        else:
            print(f"Skipped: {result.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
