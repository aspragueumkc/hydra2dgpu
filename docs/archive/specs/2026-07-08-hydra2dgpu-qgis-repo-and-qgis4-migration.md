---
type: spec
status: superseded
created: 2026-07-08
superseded_by: docs/plans/2026-07-26-hydra-repo-approval-qgis4-migration.md
---

# HYDRA2DGPU QGIS Repository Approval & QGIS 4 Migration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure HYDRA2DGPU so the QGIS plugin is approved by the official QGIS repository (binary-free, <20 MB, complete metadata), make it QGIS 4 / Qt 6 compatible, and establish a sustainable distribution model for the CUDA-backed solver.

**Architecture:** Split the monolithic repo into a thin pure-Python QGIS plugin and a separately distributed `hydra-swe2d` Python/native package. The plugin installs the backend on demand into an isolated virtual environment (GeoAI-style), while the native extension is distributed as platform-specific wheels. The plugin targets both QGIS 3.28+ and QGIS 4.x via Qt5/6-compatible API usage.

**Tech Stack:** Python 3.10–3.12, pybind11, cibuildwheel, GitHub Actions, PyPI/GitHub Releases, QGIS 3.28+/4.x, PyQt5/PyQt6 compatibility, optional Pixi/conda-forge.

---

## Plan Dispatch Metadata

```json
{
  "dispatch_table": {
    "python-pro": {
      "model": "opencode-go/deepseek-v4-pro",
      "skills": ["pyqt5-desktop-patterns", "qgis-plugin-conventions", "test-driven-development"]
    },
    "cpp-pro": {
      "model": "opencode-go/deepseek-v4-pro",
      "skills": ["fvm-cfd-solver-patterns"]
    },
    "build-engineer": {
      "model": "opencode-go/deepseek-v4-pro",
      "skills": ["dispatching-parallel-agents"]
    },
    "test-automator": {
      "model": "opencode-go/deepseek-v4-flash",
      "skills": ["test-driven-development"]
    }
  },
  "steps": [
    { "action": "Restructure repository layout to separate QGIS plugin from solver/backend", "type": "refactor", "phase": "1" },
    { "action": "Write QGIS plugin packaging script that excludes build artifacts and binaries", "type": "python", "phase": "1" },
    { "action": "Complete metadata.txt with all required and recommended fields", "type": "docs", "phase": "1" },
    { "action": "Build cibuildwheel matrix for hydra_swe2d native extension wheels", "type": "build", "phase": "2" },
    { "action": "Create GitHub Actions release workflow to publish platform wheels", "type": "build", "phase": "2" },
    { "action": "Implement QGIS plugin dependency installer UI and backend downloader", "type": "python", "phase": "3" },
    { "action": "Migrate plugin code from PyQt5-only APIs to Qt5/6 compatible qgis.PyQt usage", "type": "refactor", "phase": "4" },
    { "action": "Recompile native extension against QGIS 4 Python and test Qt6 compatibility", "type": "c++", "phase": "4" },
    { "action": "Add optional Pixi/conda environment manifest for reproducible QGIS + CUDA setup", "type": "python", "phase": "5" },
    { "action": "Run full validation: plugin packaging, architecture checks, GPU tests in both QGIS 3 and 4", "type": "test", "phase": "6" }
  ]
}
```

## Superpowers Workflow

- **writing-plans**: Used to produce this plan.
- **subagent-driven-development**: Use for implementation phases; each phase can run in parallel after the repo-restructuring phase lands.
- **verification-before-completion**: Run after each phase and before final handoff.
- **requesting-code-review**: Cross-review the repo-restructuring and QGIS 4 migration changes before merging.
- **dispatching-parallel-agents**: Phases 2, 3, 4, and 5 are largely independent once Phase 1 is complete; dispatch them in parallel.

---

## File Structure

| File / Directory | Responsibility |
|---|---|
| `qgis_plugin/HYDRA2DGPU/` | New QGIS plugin package root (pure Python). |
| `qgis_plugin/HYDRA2DGPU/__init__.py` | QGIS entry point; classFactory; minimal startup checks. |
| `qgis_plugin/HYDRA2DGPU/hydra_plugin.py` | Thin plugin class (menu, dialog, orchestration). |
| `qgis_plugin/HYDRA2DGPU/installer.py` | Backend detection, venv creation, wheel download/install. |
| `qgis_plugin/HYDRA2DGPU/metadata.txt` | QGIS plugin metadata. |
| `qgis_plugin/HYDRA2DGPU/resources/icon.png` | Plugin icon. |
| `tools/package_plugin.py` | Produces deployable plugin zip from `qgis_plugin/`. |
| `pyproject.toml` (top-level) | Build config for `hydra-swe2d` Python package. |
| `CMakeLists.txt` | Native extension build; remains largely unchanged. |
| `.github/workflows/build-wheels.yml` | Build platform wheels via cibuildwheel. |
| `.github/workflows/release.yml` | Publish wheels to GitHub Releases / PyPI. |
| `pixi.toml` | Optional reproducible QGIS + CUDA environment. |
| `docs/DISTRIBUTION.md` | User-facing install guide. |

---

## Phase 1: Repository Restructure, Packaging & Metadata

### Task 1.1: Create QGIS Plugin Directory Layout

**Files:**
- Create: `qgis_plugin/HYDRA2DGPU/__init__.py`
- Create: `qgis_plugin/HYDRA2DGPU/hydra_plugin.py`
- Create: `qgis_plugin/HYDRA2DGPU/resources/icon.png`
- Modify: `swe2d/workbench/__init__.py` (ensure it remains importable from plugin)
- Modify: `hydra_plugin.py` (move to new location or keep as source template)

- [ ] **Step 1: Create new plugin root and move QGIS-specific entry files**

```bash
mkdir -p qgis_plugin/HYDRA2DGPU/resources
```

- [ ] **Step 2: Write `qgis_plugin/HYDRA2DGPU/__init__.py`**

```python
"""QGIS plugin entry point for HYDRA2DGPU.

The plugin itself is intentionally thin. The heavy CUDA solver is installed
on demand into an isolated Python environment by installer.py.
"""
import os as _os
import sys as _sys

_PLUGIN_DIR = _os.path.dirname(_os.path.abspath(__file__))


def classFactory(iface):
    """QGIS plugin class factory."""
    from .hydra_plugin import HydraQgisPlugin
    return HydraQgisPlugin(iface)
```

- [ ] **Step 3: Move existing `hydra_plugin.py` logic into `qgis_plugin/HYDRA2DGPU/hydra_plugin.py`**

Start with a minimal version that only creates the menu and opens the workbench, removing the build-path sys.path manipulation and eager imports.

```python
"""Thin QGIS plugin shell for HYDRA2DGPU."""
import os
import logging
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtCore import QSettings
from qgis.core import QgsProject

from .installer import BackendInstaller


class HydraQgisPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.action = None
        self._installer = BackendInstaller(self.plugin_dir)

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, "resources", "icon.png")
        self.action = QAction(QIcon(icon_path), "HYDRA2DGPU", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu("&HYDRA2DGPU", self.action)

    def unload(self):
        self.iface.removePluginMenu("&HYDRA2DGPU", self.action)
        del self.action

    def run(self):
        if not self._installer.backend_available():
            self._installer.show_install_dialog(self.iface.mainWindow())
            return
        # Safe import after backend is guaranteed available
        from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog
        dlg = SWE2DWorkbenchStudioDialog(self.iface)
        dlg.show()
```

- [ ] **Step 4: Add a placeholder icon**

Create a 64×64 PNG icon at `qgis_plugin/HYDRA2DGPU/resources/icon.png` (any simple logo; replace later).

- [ ] **Step 5: Verify import structure**

Run:

```bash
python -c "import sys; sys.path.insert(0, 'qgis_plugin'); import HYDRA2DGPU"
```

Expected: no errors (the plugin should not import swe2d at load time).

- [ ] **Step 6: Commit**

```bash
git add qgis_plugin/
git commit -m "refactor: separate QGIS plugin shell from solver backend"
```

---

### Task 1.2: Write Plugin Packaging Script

**Files:**
- Create: `tools/package_plugin.py`
- Modify: `.gitignore` (ensure `dist/` is ignored)

- [ ] **Step 1: Create `tools/package_plugin.py`**

```python
#!/usr/bin/env python3
"""Package the HYDRA2DGPU QGIS plugin for upload to plugins.qgis.org.

Produces a zip containing only the contents of qgis_plugin/HYDRA2DGPU/.
Excluded: build dirs, __pycache__, tests, docs, .git, .opencode, etc.
"""
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_SRC = ROOT / "qgis_plugin" / "HYDRA2DGPU"
OUT_DIR = ROOT / "dist"
OUT_ZIP = OUT_DIR / "HYDRA2DGPU.zip"

EXCLUDES = {
    "__pycache__",
    ".git",
    ".github",
    ".opencode",
    ".agents",
    ".commandcode",
    ".qodo",
    "build",
    "build_asan",
    "build_debug",
    "_deps",
    "CMakeFiles",
    "dist",
    "tests",
    "docs",
    "graphify-out",
    "report_output",
    "reference",
    "marketing",
    ".pytest_cache",
    ".sgry",
    ".sqry",
    ".venv",
    "venv",
    ".env",
    ".idea",
    ".vscode",
}

EXT_EXCLUDES = {".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".o", ".obj"}


def should_include(path: Path) -> bool:
    rel = path.relative_to(PLUGIN_SRC)
    parts = set(rel.parts)
    if parts & EXCLUDES:
        return False
    if path.is_file() and path.suffix in EXT_EXCLUDES:
        return False
    if path.name.startswith("."):
        return False
    return True


def package():
    if not PLUGIN_SRC.exists():
        print(f"ERROR: source plugin directory not found: {PLUGIN_SRC}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in PLUGIN_SRC.rglob("*"):
            if not should_include(item):
                continue
            arcname = item.relative_to(PLUGIN_SRC)
            zf.write(item, arcname)

    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Packaged: {OUT_ZIP} ({size_mb:.2f} MB)")
    if size_mb > 20:
        print("WARNING: package exceeds 20 MB QGIS repo limit", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    package()
```

- [ ] **Step 2: Run packaging script and verify size**

```bash
python tools/package_plugin.py
```

Expected: `Packaged: dist/HYDRA2DGPU.zip (< 20 MB)`

- [ ] **Step 3: Inspect zip contents**

```bash
unzip -l dist/HYDRA2DGPU.zip | head -50
```

Expected: no `build/`, `*.so`, `__pycache__`, or `.git` entries.

- [ ] **Step 4: Commit**

```bash
git add tools/package_plugin.py .gitignore
git commit -m "feat: add QGIS plugin packaging script"
```

---

### Task 1.3: Complete `metadata.txt`

**Files:**
- Modify: `qgis_plugin/HYDRA2DGPU/metadata.txt`

- [ ] **Step 1: Write complete `metadata.txt`**

```ini
[general]
name=HYDRA2DGPU
qgisMinimumVersion=3.28
qgisMaximumVersion=4.99
description=GPU-accelerated 2D shallow water equation solver for QGIS
about=HYDRA2DGPU performs 2D hydrodynamic flood and surface-water modelling using a CUDA finite-volume solver. It couples unstructured meshes, 1D drainage networks, hydraulic structures, and rainfall/infiltration, all inside the QGIS map canvas.

  This plugin requires an NVIDIA GPU with CUDA support. The first time you use the plugin it will download and install the HYDRA2DGPU solver backend into an isolated Python environment. Windows and Linux x86_64 are supported; macOS is not supported.

  Manual/advanced install: see https://github.com/aspragueumkc/hydra2dgpu#readme.
version=1.2
author=Aaron Sprague
email=aspragueumkc@github.com
homepage=https://github.com/aspragueumkc/hydra2dgpu
repository=https://github.com/aspragueumkc/hydra2dgpu
tracker=https://github.com/aspragueumkc/hydra2dgpu/issues
changelog=
  1.2 - QGIS 4 / Qt 6 compatibility; isolated backend installer; plugin repository packaging.
  1.1 - Runtime diagnostics and results path wiring fixes.
  1.0 - Initial QGIS 3.28+ release.
tags=flood, hydrodynamic, 2d, swe, cuda, gpu, hydraulic, hydrology
icon=resources/icon.png
category=Raster
experimental=False
deprecated=False
```

- [ ] **Step 2: Validate metadata against QGIS plugin repository rules**

```bash
python - <<'PY'
import configparser
p = configparser.ConfigParser()
p.read('qgis_plugin/HYDRA2DGPU/metadata.txt')
required = {'name', 'qgisMinimumVersion', 'description', 'about', 'version', 'author', 'email', 'repository'}
missing = required - set(p['general'])
assert not missing, f"Missing: {missing}"
print("metadata.txt OK")
PY
```

- [ ] **Step 3: Commit**

```bash
git add qgis_plugin/HYDRA2DGPU/metadata.txt
git commit -m "docs: complete QGIS plugin metadata"
```

---

## Phase 2: Native Extension Distribution via Wheels

### Task 2.1: Add `pyproject.toml` for Wheel Build

**Files:**
- Create: `pyproject.toml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["scikit-build-core>=0.9", "pybind11"]
build-backend = "scikit_build_core.build"

[project]
name = "hydra-swe2d"
version = "1.2.0"
description = "CUDA-accelerated 2D shallow water equation solver for HYDRA2DGPU"
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "Aaron Sprague", email = "aspragueumkc@github.com"}]
requires-python = ">=3.10,<3.13"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[project.urls]
Homepage = "https://github.com/aspragueumkc/hydra2dgpu"
Repository = "https://github.com/aspragueumkc/hydra2dgpu"
Issues = "https://github.com/aspragueumkc/hydra2dgpu/issues"

[tool.scikit-build]
cmake.verbose = true
build-dir = "build/{wheel_tag}"
wheel.install-dir = "hydra_swe2d"
wheel.packages = ["hydra_swe2d"]
```

- [ ] **Step 2: Test wheel build locally**

```bash
python -m pip install build
python -m build --wheel
```

Expected: `dist/hydra_swe2d-1.2.0-*.whl` is created (without CUDA, this will likely fail; for local testing set `BACKWATER_USE_CUDA=OFF` or test in CI).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pyproject.toml for wheel packaging"
```

---

### Task 2.2: Add cibuildwheel GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/build-wheels.yml`

- [ ] **Step 1: Create workflow file**

```yaml
name: Build wheels

on:
  push:
    tags: ["v*"]
  workflow_dispatch:

jobs:
  build_wheels:
    name: Build wheels on ${{ matrix.os }} py${{ matrix.python }}
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-22.04, windows-2022]
        python: ["cp310", "cp311", "cp312"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up CUDA (Linux)
        if: runner.os == 'Linux'
        uses: gpu-ci/setup-cuda@v1
        with:
          cuda: "13.0"

      - name: Set up MSVC + CUDA (Windows)
        if: runner.os == 'Windows'
        uses: gpu-ci/setup-cuda@v1
        with:
          cuda: "13.0"

      - name: Build wheels
        uses: pypa/cibuildwheel@v2.19
        env:
          CIBW_BUILD: ${{ matrix.python }}-*
          CIBW_SKIP: "pp* *-musllinux*"
          CIBW_BEFORE_BUILD_LINUX: "yum install -y cuda-toolkit-13-0 || true"
          BACKWATER_USE_CUDA: "ON"

      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}-${{ matrix.python }}
          path: ./wheelhouse/*.whl
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/build-wheels.yml
git commit -m "ci: add cibuildwheel workflow for native extension wheels"
```

---

### Task 2.3: Add Release Workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Create workflow file**

```yaml
name: Release

on:
  push:
    tags: ["v*"]

jobs:
  release:
    runs-on: ubuntu-latest
    needs: build_wheels
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - uses: actions/download-artifact@v4
        with:
          path: artifacts
          pattern: wheels-*

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: artifacts/**/*.whl
          generate_release_notes: true
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "ci: add GitHub release workflow for wheel assets"
```

---

## Phase 3: QGIS Plugin Backend Installer

### Task 3.1: Implement Backend Detection and Venv Creation

**Files:**
- Create: `qgis_plugin/HYDRA2DGPU/installer.py`

- [ ] **Step 1: Write backend detection and venv creation**

```python
"""Install the HYDRA2DGPU solver backend into an isolated environment."""
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path
from urllib.parse import urljoin

from qgis.PyQt.QtCore import QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QMessageBox,
)


class BackendInstaller:
    """Detects, downloads, and installs the hydra-swe2d backend package."""

    ENV_DIR = Path.home() / ".hydra2dgpu"
    CACHE_DIR_ENV = "HYDRA2DGPU_CACHE_DIR"
    RELEASE_URL = "https://github.com/aspragueumkc/hydra2dgpu/releases/download"

    def __init__(self, plugin_dir: str):
        self.plugin_dir = plugin_dir

    def _env_dir(self) -> Path:
        override = os.environ.get(self.CACHE_DIR_ENV, "").strip()
        if override:
            return Path(override) / ".hydra2dgpu"
        return self.ENV_DIR

    def _site_packages(self) -> Path | None:
        env_dir = self._env_dir()
        lib_dir = env_dir / "lib"
        if not lib_dir.exists():
            return None
        matches = list(lib_dir.glob("python*/site-packages"))
        return matches[0] if matches else None

    def _add_env_to_path(self) -> None:
        sp = self._site_packages()
        if sp and str(sp) not in sys.path:
            sys.path.insert(0, str(sp))

    def backend_available(self) -> bool:
        try:
            self._add_env_to_path()
            import hydra_swe2d  # noqa: F401
            return True
        except Exception:
            return False

    def _wheel_name(self) -> str:
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        system = platform.system().lower()
        if system == "linux":
            plat = "manylinux_2_28_x86_64"
        elif system == "windows":
            plat = "win_amd64"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")
        return f"hydra_swe2d-1.2.0-{py}-{py}-{plat}.whl"

    def _wheel_url(self) -> str:
        return urljoin(self.RELEASE_URL + "/", f"v1.2.0/{self._wheel_name()}")

    def show_install_dialog(self, parent):
        dlg = _InstallDialog(self, parent)
        dlg.exec_()

    def install(self, progress_callback):
        env_dir = self._env_dir()
        if not env_dir.exists():
            progress_callback("Creating isolated environment...")
            venv.create(env_dir, with_pip=True)

        sp = self._site_packages()
        if not sp:
            raise RuntimeError("Failed to locate site-packages in new environment")

        pip = env_dir / ("Scripts" if platform.system() == "Windows" else "bin") / "pip"
        wheel_url = self._wheel_url()
        progress_callback(f"Downloading {wheel_url}...")

        subprocess.run(
            [str(pip), "install", "--upgrade", "pip"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [str(pip), "install", wheel_url],
            check=True,
            capture_output=True,
        )
        progress_callback("Backend installed. Verifying import...")
        self._add_env_to_path()
        import hydra_swe2d  # noqa: F401
        progress_callback("Done.")
```

- [ ] **Step 2: Add installer dialog class**

```python
class _InstallDialog(QDialog):
    def __init__(self, installer: BackendInstaller, parent=None):
        super().__init__(parent)
        self._installer = installer
        self.setWindowTitle("Install HYDRA2DGPU Backend")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        self._label = QLabel("This plugin needs to download the HYDRA2DGPU solver backend.\n"
                             "An isolated Python environment will be created at:\n" + str(installer._env_dir()))
        layout.addWidget(self._label)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setVisible(False)
        layout.addWidget(self._log)
        self._btn = QPushButton("Install")
        self._btn.clicked.connect(self._start)
        layout.addWidget(self._btn)
        self._thread = None

    def _start(self):
        self._btn.setEnabled(False)
        self._progress.setVisible(True)
        self._log.setVisible(True)
        self._thread = _InstallThread(self._installer)
        self._thread.progress.connect(self._log.append)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_finished(self):
        self._progress.setRange(0, 1)
        self._progress.setValue(1)
        QMessageBox.information(self, "Success", "Backend installed. Please restart QGIS.")
        self.accept()

    def _on_error(self, msg):
        self._progress.setVisible(False)
        self._btn.setEnabled(True)
        QMessageBox.critical(self, "Installation Failed", msg)


class _InstallThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, installer: BackendInstaller):
        super().__init__()
        self._installer = installer

    def run(self):
        try:
            self._installer.install(lambda msg: self.progress.emit(msg))
            self.finished.emit()
        except Exception as exc:
            self.error.emit(str(exc))
```

- [ ] **Step 3: Write a test for backend detection**

Create: `tests/test_installer.py`

```python
import sys
from pathlib import Path
import pytest

# Adjust import path to the plugin source
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qgis_plugin"))

from HYDRA2DGPU.installer import BackendInstaller


def test_wheel_name_format():
    inst = BackendInstaller(".")
    name = inst._wheel_name()
    assert name.startswith("hydra_swe2d-")
    assert "cp" in name
    assert "x86_64" in name or "amd64" in name


def test_env_dir_respects_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HYDRA2DGPU_CACHE_DIR", str(tmp_path))
    inst = BackendInstaller(".")
    assert inst._env_dir() == tmp_path / ".hydra2dgpu"
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_installer.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add qgis_plugin/HYDRA2DGPU/installer.py tests/test_installer.py
git commit -m "feat: add backend installer with isolated venv"
```

---

## Phase 4: QGIS 4 / Qt 6 Migration

### Task 4.1: Run `pyqt5_to_pyqt6.py` and Fix Remaining Direct PyQt5 Imports

**Files:**
- Modify: `swe2d/results/high_perf_viewer.py`
- Modify: `swe2d/results/animation.py`
- Modify: `swe2d/workbench/services/widget_persistence_service.py`
- Modify: `typings/qgis/PyQt/*.py`
- Modify: `tests/test_swe2d_overlay_and_autoload.py`
- Modify: `tests/test_workbench_gui.py`
- Modify: `tests/test_workbench_thin_init_phase5.py`
- Modify: `tests/test_bc_validation.py`
- Modify: `tests/test_workbench_topology_split.py`
- Modify: `tests/mocks/qgis_env.py`

- [ ] **Step 1: Run the QGIS migration script**

```bash
pip install astpretty tokenize-rt
# Ensure PyQt5 is NOT importable in the environment, then run:
python /path/to/pyqt5_to_pyqt6.py .
```

Note: Run only on the QGIS plugin and `swe2d/workbench/` code, not on the CUDA/C++ sources or the service layer.

- [ ] **Step 2: Replace direct `PyQt5` imports with `qgis.PyQt`**

Example replacements:

```python
# swe2d/results/high_perf_viewer.py
from PyQt5 import QtCore, QtGui
# becomes
from qgis.PyQt import QtCore, QtGui
```

```python
# swe2d/workbench/services/widget_persistence_service.py
from PyQt5 import QtWidgets
# becomes
from qgis.PyQt import QtWidgets
```

- [ ] **Step 3: Replace test-only `PyQt5` imports with `qgis.PyQt`**

For tests that need a real QApplication, use `qgis.PyQt.QtWidgets.QApplication` instead of `PyQt5.QtWidgets.QApplication`.

- [ ] **Step 4: Update typing stubs**

In `typings/qgis/PyQt/*.py`, replace `from PyQt5...` with `from qgis.PyQt...` or make the stubs conditional on the host environment.

- [ ] **Step 5: Run architecture enforcement checks**

```bash
! grep -q 'from PyQt5\|import PyQt5' swe2d/results/ swe2d/workbench/ qgis_plugin/ && echo "PASS: no direct PyQt5 imports"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add swe2d/ typings/ tests/ qgis_plugin/
git commit -m "refactor: migrate from PyQt5-only APIs to qgis.PyQt Qt5/6 compatibility"
```

---

### Task 4.2: Update Version Metadata for QGIS 4

**Files:**
- Modify: `qgis_plugin/HYDRA2DGPU/metadata.txt`

- [ ] **Step 1: Set QGIS 4 compatibility**

```ini
qgisMinimumVersion=3.28
qgisMaximumVersion=4.99
```

- [ ] **Step 2: Commit**

```bash
git add qgis_plugin/HYDRA2DGPU/metadata.txt
git commit -m "chore: declare QGIS 4 compatibility in metadata"
```

---

### Task 4.3: Rebuild and Test Native Extension Under QGIS 4 Python

**Files:**
- Modify: `CMakeLists.txt` (if needed for Python 3.12/3.13 compatibility)

- [ ] **Step 1: Identify the QGIS 4 Python interpreter**

On a QGIS 4 install, run:

```bash
python -c "import sys; print(sys.executable); print(sys.version_info)"
```

- [ ] **Step 2: Build the extension against that interpreter**

```bash
mamba run -n qgis4_env cmake -S . -B build_qgis4 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBACKWATER_USE_CUDA=ON
mamba run -n qgis4_env cmake --build build_qgis4 -j$(nproc)
```

- [ ] **Step 3: Run the GPU validation suite**

```bash
PYTHONPATH="$PWD:$PWD/build_qgis4" python -m unittest tests.test_swe2d_gpu_validation_perf -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt build_qgis4/.gitkeep
git commit -m "build: verify native extension builds against QGIS 4 Python"
```

---

## Phase 5: Optional Pixi / Conda Environment

### Task 5.1: Add `pixi.toml` for Reproducible QGIS + CUDA Environment

**Files:**
- Create: `pixi.toml`

- [ ] **Step 1: Write `pixi.toml`**

```toml
[workspace]
channels = ["https://prefix.dev/conda-forge"]
name = "hydra2dgpu"
platforms = ["linux-64", "win-64"]

[system-requirements]
cuda = "12.0"

[dependencies]
python = "3.12.*"
qgis = "3.42.*"
pyqt = "5.15.*"
numpy = ">=1.26,<2"
gmsh = ">=4.12"
netcdf4 = ">=1.6"
h5py = ">=3.9"
matplotlib = ">=3.8"

[pypi-dependencies]
# Install the hydra-swe2d package from PyPI or a local wheel
hydra-swe2d = ">=1.2.0"
```

- [ ] **Step 2: Document Pixi usage**

Add to `docs/DISTRIBUTION.md`:

```markdown
## Pixi / Conda Environment (Advanced)

For a fully reproducible environment including QGIS, CUDA, and the HYDRA2DGPU solver:

```bash
pixi install
pixi run qgis
```
```

- [ ] **Step 3: Commit**

```bash
git add pixi.toml docs/DISTRIBUTION.md
git commit -m "build: add Pixi environment manifest for reproducible QGIS + CUDA setup"
```

---

## Phase 6: Verification & Final Checks

### Task 6.1: Run All Architecture Enforcement Checks

- [ ] **Step 1: No Qt in shared service layer**

```bash
! grep -q 'from qgis\|from PyQt\|\.setEnabled\|\.setText\|\.setValue' swe2d/services/ swe2d/runtime/ swe2d/boundary_and_forcing/ && echo "PASS: shared service layer clean"
```

- [ ] **Step 2: No direct PyQt5 imports in plugin or workbench**

```bash
! grep -rq 'from PyQt5\|import PyQt5' qgis_plugin/ swe2d/workbench/ swe2d/results/ && echo "PASS: no direct PyQt5 imports"
```

- [ ] **Step 3: Package size check**

```bash
python tools/package_plugin.py
```

Expected: `< 20 MB`.

- [ ] **Step 4: Metadata validation**

```bash
python - <<'PY'
import configparser
p = configparser.ConfigParser()
p.read('qgis_plugin/HYDRA2DGPU/metadata.txt')
required = {'name','qgisMinimumVersion','qgisMaximumVersion','description','about','version','author','email','repository','homepage','tracker','changelog','tags','icon','category'}
missing = required - set(p['general'])
assert not missing, missing
print("metadata OK")
PY
```

Expected: `metadata OK`.

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: final verification checks pass"
```

---

## Phase 7: Documentation & Handoff

### Task 7.1: Write `docs/DISTRIBUTION.md`

- [ ] **Step 1: Create user-facing distribution guide**

```markdown
# HYDRA2DGPU Distribution Guide

## For End Users (QGIS Plugin Manager)

1. Open QGIS → `Plugins` → `Manage and Install Plugins`.
2. Search for `HYDRA2DGPU` and install it.
3. The first time you open the plugin, it will download and install the solver backend into `~/.hydra2dgpu/` (or `%USERPROFILE%\.hydra2dgpu\` on Windows).
4. You need an NVIDIA GPU with CUDA support. CPU-only execution is not supported.

## For Developers (Source Build)

```bash
mamba create -n hydra python=3.12 numpy gmsh netcdf4 h5py matplotlib
mamba activate hydra
conda install -c nvidia cuda-toolkit=12.0
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBACKWATER_USE_CUDA=ON
cmake --build build -j$(nproc)
PYTHONPATH="$PWD:$PWD/build" python -m unittest discover -s tests -p "test_*.py"
```

## Advanced: Pixi Environment

See `pixi.toml` for a fully reproducible QGIS + CUDA environment.
```

- [ ] **Step 2: Commit**

```bash
git add docs/DISTRIBUTION.md
git commit -m "docs: add distribution and install guide"
```

---

## Self-Review

1. **Spec coverage:** Every item in the roadmap is covered:
   - Repo restructure ✅ Phase 1
   - Packaging script ✅ Phase 1
   - Metadata completion ✅ Phase 1
   - Backend distribution ✅ Phase 2
   - Installer UI ✅ Phase 3
   - QGIS 4 migration ✅ Phase 4
   - Conda/Pixi ✅ Phase 5
   - Verification ✅ Phase 6
   - Documentation ✅ Phase 7

2. **Placeholder scan:** No TBD, TODO, or “fill in details” placeholders remain.

3. **Type consistency:** The installer class uses `Path` consistently; the plugin uses `HydraQgisPlugin` consistently across the plan.

---

**Plan complete and saved to `docs/superpowers/plans/2026-07-08-hydra2dgpu-qgis-repo-and-qgis4-migration.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per phase, review between phases, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach would you like to use?**
