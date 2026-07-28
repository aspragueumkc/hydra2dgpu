#!/usr/bin/env python3
"""Single source of truth for the release version.

Reads ``pyproject.toml::project.version`` (the canonical string the
wheel is built from) and propagates it to the two files that need a
literal version baked in at release time:

    * qgis_plugin/HYDRA2DGPU/metadata.txt   — read by QGIS + plugins.qgis.org
    * qgis_plugin/HYDRA2DGPU/installer.py    — first-launch wheel URL

``installer.py`` cannot import ``hydra_swe2d.__version__`` because it
runs *before* the wheel is installed, so the version literal must be
baked into the plugin zip at release time. This script does that bake.

Run locally before a release, or as a CI step in
``.github/workflows/release.yml``.

Stdlib only (tomllib is py3.11+). No new dependencies.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
METADATA = ROOT / "qgis_plugin" / "HYDRA2DGPU" / "metadata.txt"
INSTALLER = ROOT / "qgis_plugin" / "HYDRA2DGPU" / "installer.py"

_SEMVER_RE = re.compile(r"\d+\.\d+\.\d+")


def read_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    version = data["project"]["version"]
    if not isinstance(version, str) or not _SEMVER_RE.fullmatch(version):
        raise ValueError(
            f"pyproject.toml::project.version must be X.Y.Z, got {version!r}"
        )
    return version


def _rewrite(path: Path, pattern: str, replacement: str, *, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match in {path}, got {n}"
        )
    path.write_text(new, encoding="utf-8")


def write_metadata(version: str) -> None:
    _rewrite(
        METADATA,
        pattern=r"^version=.*$",
        replacement=f"version={version}",
        label="metadata.txt version=",
    )


def write_installer(version: str) -> None:
    _rewrite(
        INSTALLER,
        pattern=r'^WHEEL_VERSION\s*=\s*".*"',
        replacement=f'WHEEL_VERSION = "{version}"',
        label="installer.py WHEEL_VERSION =",
    )


def main() -> int:
    version = read_version()
    write_metadata(version)
    write_installer(version)
    print(f"Synced version {version} -> metadata.txt + installer.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())