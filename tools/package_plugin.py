#!/usr/bin/env python3
"""Package HYDRA2DGPU QGIS plugin for plugins.qgis.org.

Produces dist/HYDRA2DGPU.zip containing ONLY the Python sources under
qgis_plugin/HYDRA2DGPU/ (no .so, no build artifacts, no docs/tests/tools).
Output is asserted to be <20MB (QGIS plugin repo limit).
"""
from __future__ import annotations
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_SRC = ROOT / "qgis_plugin" / "HYDRA2DGPU"
DOCS_SRC = ROOT / "docs"
OUT_DIR = ROOT / "dist"


def _plugin_version() -> str:
    """Read the canonical plugin version from pyproject.toml.

    Used to embed the version in the zip filename so release assets are
    distinguishable at a glance. ``tomllib`` is Python 3.11+; ``tools/``
    runs against whatever Python QGIS ships, so this stays stdlib-only.
    """
    try:
        from tomllib import load as _toml_load  # py3.11+
    except ImportError:  # pragma: no cover
        try:
            from tomli import load as _toml_load  # py3.10 backport
        except ImportError as exc:
            raise SystemExit(
                "package_plugin.py requires Python 3.11+ or the tomli "
                "backport on the import path"
            ) from exc
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return str(_toml_load(fh)["project"]["version"])

# User-facing guides referenced by swe2d/workbench/views/doc_viewer.py
# (AVAILABLE_DOCS). Shipped under HYDRA2DGPU/docs/ so the workbench can
# find them in a production-installed plugin folder. Images, archives,
# and project-internal docs are intentionally NOT shipped — see
# the production-path fallback in doc_viewer._load_markdown().
USER_GUIDE_DOCS = (
    "USER_GUIDE.md",
    "GMSH_MESHING_GUIDE.md",
    "DEVELOPER_GUIDE.md",
    "STUDIO_GUI_API.md",
    "UI_COMPONENT_GUIDE.md",
)


def _plugin_name() -> str:
    """Read the canonical plugin name from metadata.txt so the zip's root
    folder always matches what QGIS's plugin manager expects (and matches
    what plugins.qgis.org will validate)."""
    for line in (PLUGIN_SRC / "metadata.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("name="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"name= not found in {PLUGIN_SRC / 'metadata.txt'}")

EXCLUDE_DIRS = {
    "__pycache__", ".git", ".github", ".opencode", ".agents",
    ".vscode", ".idea", ".worktrees", ".pytest_cache", ".superpowers",
    "build", "build_asan", "build_debug", "_deps", "CMakeFiles", "dist",
    "tests", "tools", "docs", "graphify-out", "report_output", "reference",
    "marketing", "swmm_canonical", "swmm524_gui", "Stormwater-Management-Model-develop",
    "anuga_validation_tests", "example_test_project",
}
EXCLUDE_EXT = {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".o", ".obj",
               ".exe", ".egg-info", ".png"}  # png: kept only via resources/icon.png whitelist below
EXCLUDE_FILENAMES = {"sqlite3"}                  # spurious empty file at repo root


def should_include(p: Path) -> bool:
    try:
        rel = p.relative_to(PLUGIN_SRC)
    except ValueError:
        return False
    if set(rel.parts) & EXCLUDE_DIRS:
        return False
    if p.is_file():
        if p.suffix in EXCLUDE_EXT:
            # whitelist: icon is the only allowed .png in qgis_plugin/HYDRA2DGPU/resources/
            if p.name == "icon.png" and "resources" in rel.parts:
                return True
            return False
        if p.name.startswith("."):
            return False
        if p.name in EXCLUDE_FILENAMES:
            return False
    return True


def main() -> int:
    if not PLUGIN_SRC.exists():
        print(f"ERROR: {PLUGIN_SRC} not found", file=sys.stderr)
        return 1
    name = _plugin_name()
    version = _plugin_version()
    OUT_DIR.mkdir(exist_ok=True)
    # The versioned filename (e.g. HYDRA2DGPU-0.3.0.zip) is the single
    # release asset. QGIS identifies the plugin by the top-level folder
    # inside the zip (which always matches metadata.txt::name=), not by
    # the zip's own filename. Stale zips from previous versions are
    # removed so the dist/ directory doesn't accumulate junk.
    out_zip = OUT_DIR / f"HYDRA2DGPU-{version}.zip"
    for stale in OUT_DIR.glob("HYDRA2DGPU*.zip"):
        if stale != out_zip:
            stale.unlink()

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(PLUGIN_SRC.rglob("*")):
            if not should_include(item):
                continue
            if item.is_file():
                # QGIS plugin manager (and plugins.qgis.org) require a
                # top-level folder whose name matches metadata.txt::name=.
                # Without it, "Install from ZIP" reports
                # "No root folder was found inside the zip file".
                arcname = Path(name) / item.relative_to(PLUGIN_SRC)
                zf.write(item, arcname)

        # Ship the user-facing guides under HYDRA2DGPU/docs/ so the
        # workbench can find them after a production install. The doc
        # viewer tries PLUGIN_ROOT/docs/ first (production path) and
        # falls back to PLUGIN_ROOT/../docs/ (dev path); only files
        # listed in USER_GUIDE_DOCS are bundled.
        for doc_name in USER_GUIDE_DOCS:
            src = DOCS_SRC / doc_name
            if not src.exists():
                print(f"WARNING: doc {src} not found, skipping", file=sys.stderr)
                continue
            zf.write(src, Path(name) / "docs" / doc_name)

        # Ship the layer-style QML files under HYDRA2DGPU/QML/. The style
        # service resolves QML/ relative to the repo root in dev and via
        # sys.path (plugin dir) in production; without this bundle the
        # production install silently loses layer styling.
        for qml in sorted((ROOT / "QML").glob("*.qml")):
            zf.write(qml, Path(name) / "QML" / qml.name)

    mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Packaged: {out_zip} ({mb:.2f} MB)")
    if mb > 20:
        print("ERROR: zip exceeds 20 MB QGIS plugin repo limit", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
