#!/usr/bin/env python3
"""Pack a pre-built hydra_swe2d tree into a Windows wheel.

Used by the raw-cmake Windows build path in
``.github/workflows/build-wheels.yml`` (the cibuildwheel path on Linux
still uses scikit-build-core, which calls scikit-build-core.wheel.pack
itself — this script is only for the manual assembly on Windows).

Inputs (CLI args):

    --src-root DIR           Repo root (contains pyproject.toml).
    --staging-dir DIR        Tree produced by ``cmake --install`` —
                             already contains hydra_swe2d/*.pyd,
                             hydra_swe2d/include/, swe2d/, etc.
                             Anything the CMake install rules don't
                             emit (tools/, tests/, hydra_swe2d/__init__.py)
                             is added here from the source tree.
    --output-wheel PATH      Destination .whl path.

The script mirrors the layout that scikit-build-core produces for the
Linux manylinux wheel (see the working
hydra_swe2d-0.3.0-cp312-cp312-manylinux_2_28_x86_64.whl for reference):

    hydra_swe2d/
        __init__.py
        hydra_swe2d.cp312-win_amd64.pyd
        hydra_meshing_native.cp312-win_amd64.pyd
        hydra_overlay.cp312-win_amd64.pyd
        include/swe2d_solver.hpp
    swe2d/...
    tools/...
    tests/...
    hydra_swe2d-<ver>.dist-info/
        METADATA
        WHEEL
        RECORD
        entry_points.txt
        top_level.txt
        licenses/LICENSE

The version, dependencies, and console-script entry point are read from
pyproject.toml so the Windows wheel stays in sync with the Linux one.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tomllib
import zipfile
from pathlib import Path

WHEEL_NAME = "hydra_swe2d"


def _project_metadata(pyproject: Path) -> dict:
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    proj = data["project"]
    return {
        "name": proj["name"],
        "version": proj["version"],
        "summary": proj.get("summary", ""),
        "requires_python": proj.get("requires-python", ""),
        "requires_dist": proj.get("dependencies", []),
        "homepage": proj.get("urls", {}).get("Homepage", ""),
        "repository": proj.get("urls", {}).get("Repository", ""),
        "issues": proj.get("urls", {}).get("Issues", ""),
        "scripts": proj.get("scripts", {}),
    }


def _stage_top_level_packages(src_root: Path, staging_dir: Path) -> None:
    """Copy swe2d/, tools/, tests/, and hydra_swe2d/__init__.py into
    the staging tree. ``cmake --install`` already emitted
    hydra_swe2d/*.pyd + swe2d/ (via the CMake install rules) but it
    does NOT emit tools/, tests/, or the __init__.py shim."""
    for pkg in ("swe2d", "tools", "tests"):
        src = src_root / pkg
        if not src.is_dir():
            continue
        dst = staging_dir / pkg
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(
            src, dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    # The __init__.py shim: cmake's install(FILES hydra_swe2d/__init__.py)
    # handles this when OPTIONAL is satisfied, but defensively copy if
    # the install step somehow skipped it.
    src_init = src_root / "hydra_swe2d" / "__init__.py"
    if src_init.is_file():
        dst_init = staging_dir / "hydra_swe2d" / "__init__.py"
        if not dst_init.exists():
            dst_init.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_init, dst_init)


def _stage_license(src_root: Path, dist_info: Path) -> None:
    src = src_root / "LICENSE"
    if not src.is_file():
        return
    licenses_dir = dist_info / "licenses"
    licenses_dir.mkdir(exist_ok=True)
    shutil.copy2(src, licenses_dir / "LICENSE")


def _stage_cudart(staging_dir: Path, cudart_dll: Path | None) -> None:
    """Bundle the CUDA runtime DLL into hydra_swe2d/ next to the .pyd.

    The Windows .pyd links cudart64_12.dll. It is NOT present on end-user
    machines (or CI runners) without a system CUDA install, so the wheel
    must ship it — Python finds DLLs in the package directory before the
    system load path. Without this the import fails with
    'DLL load failed ... The specified module could not be found'.
    """
    if cudart_dll is None:
        return
    dll = Path(cudart_dll)
    if not dll.is_file():
        raise SystemExit(f"cudart DLL not found at {dll}")
    dst = staging_dir / "hydra_swe2d" / dll.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dll, dst)
    print(f"  bundled {dll.name} into hydra_swe2d/")


def _write_dist_info(
    staging_dir: Path,
    src_root: Path,
    meta: dict,
    python_tag: str,
    platform_tag: str,
) -> Path:
    """Generate the dist-info directory + WHEEL, METADATA, RECORD,
    entry_points.txt, top_level.txt."""
    dist_info = staging_dir / f"{WHEEL_NAME}-{meta['version']}.dist-info"
    if dist_info.exists():
        shutil.rmtree(dist_info)
    dist_info.mkdir(parents=True)

    # METADATA
    requires_dist = "\n".join(f"Requires-Dist: {r}" for r in meta["requires_dist"])
    md = (
        f"Metadata-Version: 2.2\n"
        f"Name: {meta['name']}\n"
        f"Version: {meta['version']}\n"
        f"Summary: {meta['summary']}\n"
    )
    if meta["homepage"]:
        md += f"Project-URL: Homepage, {meta['homepage']}\n"
    if meta["repository"]:
        md += f"Project-URL: Repository, {meta['repository']}\n"
    if meta["issues"]:
        md += f"Project-URL: Issues, {meta['issues']}\n"
    if meta["requires_python"]:
        md += f"Requires-Python: {meta['requires_python']}\n"
    if requires_dist:
        md += requires_dist + "\n"
    (dist_info / "METADATA").write_text(md, encoding="utf-8")

    # WHEEL
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\n"
        "Generator: hydra-raw-cmake 1.0\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {python_tag}-{python_tag}-{platform_tag}\n",
        encoding="utf-8",
    )

    # entry_points.txt — swe2d-cli
    if meta["scripts"]:
        ep = "[console_scripts]\n"
        for name, target in meta["scripts"].items():
            ep += f"{name} = {target}\n"
        (dist_info / "entry_points.txt").write_text(ep, encoding="utf-8")

    # top_level.txt
    (dist_info / "top_level.txt").write_text(
        "\n".join(sorted({WHEEL_NAME, "swe2d", "tools", "tests"})) + "\n",
        encoding="utf-8",
    )

    _stage_license(src_root, dist_info)

    # RECORD — sha256+size for every file in the wheel. Per PEP 427 every
    # file must be listed except RECORD itself (which appears with an empty
    # hash+size as the last entry). dist-info/* files (METADATA, WHEEL,
    # entry_points.txt, top_level.txt, licenses/LICENSE) ARE included — a
    # wheel whose RECORD omits them makes pip reject the install.
    record_lines: list[str] = []
    record_rel = f"{dist_info.name}/RECORD"
    for path in sorted(p for p in staging_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(staging_dir).as_posix()
        if rel == record_rel:
            continue
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        record_lines.append(f"{rel},sha256={sha},{len(data)}")
    record_lines.append(f"{record_rel},,")
    (dist_info / "RECORD").write_text(
        "\n".join(record_lines) + "\n", encoding="utf-8",
    )

    return dist_info


def pack_wheel(
    *,
    src_root: Path,
    staging_dir: Path,
    output_wheel: Path,
    platform_tag: str = "win_amd64",
    python_tag: str = "cp312",
    cudart_dll: Path | None = None,
) -> Path:
    """Assemble + pack the wheel from the staging tree. Returns the
    output wheel path."""
    _stage_top_level_packages(src_root, staging_dir)
    _stage_cudart(staging_dir, cudart_dll)
    meta = _project_metadata(src_root / "pyproject.toml")
    dist_info = _write_dist_info(staging_dir, src_root, meta, python_tag, platform_tag)

    output_wheel.parent.mkdir(parents=True, exist_ok=True)
    if output_wheel.exists():
        output_wheel.unlink()
    with zipfile.ZipFile(output_wheel, "w", zipfile.ZIP_DEFLATED) as zf:
        # Standard wheel ordering: package files first, then dist-info
        for path in sorted(p for p in staging_dir.rglob("*") if p.is_file()):
            if path.is_relative_to(dist_info):
                continue
            arcname = path.relative_to(staging_dir).as_posix()
            zf.write(path, arcname)
        for path in sorted(dist_info.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(staging_dir).as_posix()
                zf.write(path, arcname)

    return output_wheel


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--staging-dir", type=Path, required=True)
    p.add_argument("--output-wheel", type=Path, required=True)
    p.add_argument("--platform-tag", default="win_amd64")
    p.add_argument("--cudart-dll", type=Path, default=None,
                   help="Path to cudart64_*.dll to bundle (Windows runtime dep)")
    args = p.parse_args(argv)

    wheel = pack_wheel(
        src_root=args.src_root.resolve(),
        staging_dir=args.staging_dir.resolve(),
        output_wheel=args.output_wheel.resolve(),
        platform_tag=args.platform_tag,
        cudart_dll=args.cudart_dll.resolve() if args.cudart_dll else None,
    )
    print(f"Packed {wheel} ({wheel.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
