#!/usr/bin/env python3
"""
check_deps.py — Verify and install HYDRA2DGPU Python dependencies.

Run this after installing the plugin to ensure all required packages
are available in your QGIS Python environment.

Usage (standalone):
    python check_deps.py                   # Check only (report missing packages)
    python check_deps.py --install         # Install missing required packages
    python check_deps.py --all             # Install all packages (required + optional)

Usage (from inside QGIS Python console):
    exec(open('path/to/check_deps.py').read())
    check_all(install=True)                # returns (missing_required, missing_optional)
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ── Package registry ──────────────────────────────────────────────────────
# (import_name, pip_name, required, description)
@dataclass
class Package:
    import_name: str
    pip_name: str
    required: bool = True
    description: str = ""
    installed: bool = False
    version: str = ""


PACKAGES: List[Package] = [
    # Core — always required
    Package(
        import_name="numpy",
        pip_name="numpy>=1.24",
        required=True,
        description="Numerical arrays and linear algebra",
    ),
    # Mesh generation — required for the gmsh meshing backend
    Package(
        import_name="gmsh",
        pip_name="gmsh>=4.12",
        required=True,
        description="Unstructured mesh generation (Gmsh backend)",
    ),
    # Optional: result export
    Package(
        import_name="h5py",
        pip_name="h5py>=3.8",
        required=False,
        description="HEC-RAS HDF5 result export",
    ),
    Package(
        import_name="netCDF4",
        pip_name="netCDF4>=1.6",
        required=False,
        description="UGRID NetCDF result export",
    ),
    # Optional: visualization
    Package(
        import_name="matplotlib",
        pip_name="matplotlib>=3.7",
        required=False,
        description="In-plugin plotting and visualization",
    ),
]

# Packages that should NEVER be installed via pip (provided by QGIS)
QGIS_BUNDLED = {"qgis", "PyQt5", "PyQt6", "osgeo", "qgis.PyQt"}


def _check_package(pkg: Package) -> None:
    """Check if a package is importable and get its version."""
    try:
        mod = importlib.import_module(pkg.import_name)
        pkg.installed = True
        pkg.version = getattr(mod, "__version__", "unknown")
    except ImportError:
        pkg.installed = False
        pkg.version = ""


def _install_package(pkg: Package) -> bool:
    """Install a package via pip. Returns True on success.

    Uses sys.executable to run pip, which inside QGIS automatically uses
    the QGIS-bundled Python interpreter — no need to hunt down the right
    Python path manually.
    """
    cmd = [sys.executable, "-m", "pip", "install", pkg.pip_name]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_all(
    install: bool = False,
    install_optional: bool = False,
    quiet: bool = False,
) -> Tuple[int, int]:
    """Check all dependencies.

    Args:
        install: Install missing required packages.
        install_optional: Also install missing optional packages.
        quiet: Suppress console output (for use from inside QGIS).

    Returns:
        (missing_required_count, missing_optional_count)
    """
    if not quiet:
        print("=" * 60)
        print("  HYDRA2DGPU Dependency Check")
        print("=" * 60)
        print()

    missing_required = 0
    missing_optional = 0

    for pkg in PACKAGES:
        _check_package(pkg)

        if pkg.installed:
            status = f"✅ {pkg.version}"
        elif pkg.required:
            status = "❌ MISSING"
            missing_required += 1
        else:
            status = "⚠️  not installed (optional)"
            missing_optional += 1

        if not quiet:
            label = "required" if pkg.required else "optional"
            print(f"  {pkg.import_name:<15} {status:<25} [{label}] {pkg.description}")

    if not quiet:
        print()

    # Install missing packages if requested
    if install or install_optional:
        to_install = []
        for pkg in PACKAGES:
            if pkg.installed:
                continue
            if pkg.required or (install_optional and not pkg.required):
                to_install.append(pkg)

        if to_install and not quiet:
            print("Installing missing packages...")
        for pkg in to_install:
            if not quiet:
                print(f"  pip install {pkg.pip_name} ... ", end="", flush=True)
            ok = _install_package(pkg)
            if ok:
                if not quiet:
                    print("✅")
            else:
                if not quiet:
                    print("❌ failed")
                if pkg.required:
                    missing_required += 1
        if not quiet:
            print()

    # Summary
    if not quiet:
        print("-" * 60)
        if missing_required == 0:
            print("✅ All required dependencies are installed.")
        else:
            print(f"❌ {missing_required} required package(s) missing.")
            print("   Run: pip install -r requirements.txt")
        if missing_optional > 0:
            print(f"⚠️  {missing_optional} optional package(s) not installed.")
            print("   Run: pip install -r requirements.txt  (for full functionality)")

        # CUDA check
        print()
        try:
            import hydra_swe2d
            if hasattr(hydra_swe2d, "swe2d_gpu_available") and hydra_swe2d.swe2d_gpu_available():
                print("✅ CUDA GPU solver available.")
            else:
                print("⚠️  Native module loaded but no CUDA GPU detected.")
        except ImportError:
            print("⚠️  Native CUDA module (hydra_swe2d) not found.")
            print("   Download pre-compiled binary from:")
            print("   https://github.com/aspragueumkc/hydra2dgpu/releases")
        print("-" * 60)

    return missing_required, missing_optional


def main() -> None:
    parser = argparse.ArgumentParser(description="Check HYDRA2DGPU dependencies")
    parser.add_argument(
        "--install", action="store_true",
        help="Install missing required packages via pip",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Install all missing packages (required + optional)",
    )
    args = parser.parse_args()

    missing_req, missing_opt = check_all(
        install=args.install,
        install_optional=args.all,
    )
    sys.exit(1 if missing_req > 0 else 0)


if __name__ == "__main__":
    main()
