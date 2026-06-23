#!/usr/bin/env python3
"""
package_release.py — Package the plugin for distribution.

Builds the release ZIP containing:
  - Full Python plugin tree (no dev/test/build artifacts)
  - Pre-compiled native .so/.pyd binaries in the plugin root

Usage:
    python tools/package_release.py [--build-dir BUILD] [--output-dir DIST]

The script:
  1. Validates that native binaries exist in the build directory.
  2. Copies the clean plugin tree into a staging directory.
  3. Drops the native binaries into the staging root.
  4. Creates a platform-tagged ZIP ready for GitHub Releases.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import zipfile
from pathlib import Path

# ── Directories to exclude from the distribution ZIP ──────────────────────
EXCLUDE_DIRS = {
    "build",
    "build/",
    ".git",
    "__pycache__",
    "tests",
    "tools",
    "docs",
    "reference",
    "typings",
    "cpp",
    "report_output",
    "2d_example",
    "example_project",
    "qgis_testing_project",
    ".github",
}

EXCLUDE_FILES = {
    "*.pyc",
    "*.pyo",
    "AGENTS.md",
    "GPU_AUDIT_REPORT.md",
    "MOMENTUM_CAP_FIX.md",
    "coupling_diag.log",
    "package_release.py",
    "test_gpu_debug.py",
    "test_workbench_persistence.py",
    "stacked_bridge_coupling.py",
    "stacked_bridge_toy.py",
    "swe2d_results_panel.py.sprint0_bak",
}

# Native module basenames (without extension)
NATIVE_MODULES = [
    "hydra_swe2d",
    "hydra_hybridmesh",
    "hydra_meshing_native",
    "hydra_overlay",
]


def _platform_tag() -> str:
    """Return a platform tag for the release ZIP filename."""
    system = platform.system().lower()          # linux, windows, darwin
    machine = platform.machine().lower()        # x86_64, amd64, aarch64
    tag_map = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "arm64",
    }
    arch = tag_map.get(machine, machine)
    if system == "windows":
        system = "windows"
    return f"{system}-{arch}"


def _find_native_binaries(build_dir: Path) -> list[Path]:
    """Locate compiled .so or .pyd files in the build directory."""
    exts = (".so", ".pyd")
    found = []
    for mod in NATIVE_MODULES:
        hits = list(build_dir.rglob(f"{mod}*"))
        hits = [h for h in hits if h.suffix in exts]
        if not hits:
            print(f"  WARNING: {mod} not found in {build_dir}")
            continue
        # Pick the first match (prefer Release/ build)
        best = sorted(hits, key=lambda p: "Release" in str(p), reverse=True)[0]
        found.append(best)
        print(f"  Found: {best.name}")
    return found


def _copy_plugin_tree(src: Path, dst: Path) -> None:
    """Copy the plugin tree excluding dev/test/build artifacts."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        name = item.name

        # Skip excluded directories
        if item.is_dir() and name in EXCLUDE_DIRS:
            continue

        # Skip excluded files
        if item.is_file() and any(
            name == pat or name.endswith(pat.lstrip("*"))
            for pat in EXCLUDE_FILES
            if pat.startswith("*")
        ):
            continue
        if item.is_file() and name in EXCLUDE_FILES:
            continue

        if item.is_dir():
            shutil.copytree(item, dst / name, ignore=shutil.ignore_patterns(*EXCLUDE_FILES))
        else:
            shutil.copy2(item, dst / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package plugin for release")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Path to CMake build directory (default: build/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist"),
        help="Output directory for the ZIP (default: dist/)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Version tag for the ZIP filename (e.g. v1.0.0)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    build_dir = repo_root / args.build_dir
    staging_dir = args.output_dir / "staging"
    output_dir = args.output_dir

    print("=== HYDRA2DGPU Release Packager ===")
    print(f"  Repo root:  {repo_root}")
    print(f"  Build dir:  {build_dir}")
    print(f"  Platform:   {_platform_tag()}")

    # 1. Find native binaries
    print("\n[1/4] Locating native binaries...")
    binaries = _find_native_binaries(build_dir)
    if not binaries:
        print("\nERROR: No native binaries found. Build the project first.")
        sys.exit(1)

    # 2. Copy plugin tree
    print("\n[2/4] Copying plugin tree...")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    _copy_plugin_tree(repo_root, staging_dir)
    print(f"  Staged to: {staging_dir}")

    # 3. Drop binaries into staging root
    print("\n[3/4] Installing native binaries...")
    for binary in binaries:
        dest = staging_dir / binary.name
        shutil.copy2(binary, dest)
        print(f"  Installed: {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")

    # 4. Create ZIP
    print("\n[4/4] Creating release ZIP...")
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "dev"
    zip_name = f"hydra2gpu-{_platform_tag()}-{tag}.zip"
    zip_path = output_dir / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(staging_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = Path("plugin") / file_path.relative_to(staging_dir)
                zf.write(file_path, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n  Created: {zip_path} ({size_mb:.1f} MB)")

    # Cleanup staging
    shutil.rmtree(staging_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
