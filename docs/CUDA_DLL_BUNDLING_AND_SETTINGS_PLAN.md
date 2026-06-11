# CUDA Runtime DLL Bundling & Settings UI Plan

**Created**: 2026-06-11
**Status**: Planned

## Problem

Users on Windows without CUDA installed system-wide cannot use the pre-compiled release ZIPs. The native `hydra_swe2d.pyd` module fails to load because `cudart64_12.dll` is not on `PATH`.

Additionally, `gmsh` and other Python dependencies must be installed into the **QGIS-bundled Python interpreter**, which is a hassle on Windows via standalone pip.

## Phase 1 — CUDA DLL Bundling (CI)

### Steps

1. **Copy `cudart64_12.dll` into the Windows release ZIP**
   - In the `Collect binaries` step of `.github/workflows/build-release.yml`, add:
     ```powershell
     Copy-Item "$env:CUDA_PATH\bin\cudart64_*.dll" dist\lib\
     ```
   - This bundles the CUDA runtime DLL alongside the `.pyd` files.

2. **Copy DLL to plugin root during packaging**
   - In the `Package` step, copy `cudart64_12.dll` to `dist\hydra2dgpu\` (the plugin root), so it sits alongside `__init__.py` and is on the default DLL search path.

### Considerations

| Concern | Answer |
|---------|--------|
| **License** | NVIDIA EULA allows redistributing the CUDA runtime DLL |
| **DLL size** | ~850 KB — negligible |
| **CUDA driver dependency** | `cudart` requires the NVIDIA driver, which must be installed separately — expected |
| **Target runtime** | `cudart64_12.dll` from CUDA 12.4 (matches CI build) |

## Phase 2 — Python-side DLL Path Resolution

### Mechanism

Before importing `hydra_swe2d`, add the plugin root to the Windows DLL search path:

```python
import os
import sys

if sys.platform == "win32":
    # Python 3.8+ API (QGIS bundles Python 3.12)
    dll_dir = os.path.dirname(__file__)  # plugin root
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll_dir)
    else:
        # Fallback for older Python
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
```

### Search Order

1. Custom path from `QSettings("hydra2dgpu", "cuda_dll_path")` (if set by user)
2. Plugin root (bundled `cudart64_12.dll`)
3. System `PATH` (system-wide CUDA install)

### Files to Modify

| File | Change |
|------|--------|
| `swe2d/runtime/backend.py` | Add DLL path resolution before native module load; check `QSettings` for custom path |

## Phase 3 — Settings Dialog

We need a simple settings dialog accessible from the HYDRA2DGPU menu.

### Dialog Contents

- **CUDA DLL Path**: `QLineEdit` showing current path + `Browse...` button (`QFileDialog.getOpenFileName`)
- **Reset to Default** button
- OK / Cancel buttons
- Stored via `QSettings("hydra2dgpu", "cuda_dll_path")`

### Menu Action

Add a `Settings...` action to the HYDRA2DGPU dropdown menu in `hydra_plugin.py`:

```python
action_specs = [
    ('HYDRA2DMenuOpenPanelAction', 'Open HYDRA2DGPU Panel', lambda: self.run()),
    ('HYDRA2DMenuSettingsAction', 'Settings...', lambda: self.open_settings()),
]
```

### Files to Modify

| File | Change |
|------|--------|
| `hydra_plugin.py` | Add `Settings...` action and `open_settings()` method |
| `forms/` or `swe2d_workbench_qt.py` | New `CudaSettingsDialog` class (simple `QDialog`) |

## Phase 4 — In-QGIS Dependency Installer

### Problem

On Windows, `pip install gmsh` must run inside the QGIS Python interpreter, not a standalone one. QGIS ships its own Python, and the user's system Python may differ.

### Solution

Run the dependency checker **from inside QGIS after plugin initialization**.

1. **Auto-detect missing deps on first load** — `__init__.py` already logs missing optional packages. Extend this to also detect **required** packages (`numpy`, `gmsh`).

2. **Install button in settings** — The Settings dialog gains a **"Check & Install Dependencies"** button that runs:
   ```python
   import subprocess, sys
   subprocess.check_call(
       [sys.executable, "-m", "pip", "install", "-r", requirements_path]
   )
   ```
   Because this runs inside QGIS, `sys.executable` points to the **QGIS-bundled Python**, so packages land in the right place.

3. **Post-install verification** — After install, re-check imports and report success/failure.

### Files to Modify

| File | Change |
|------|--------|
| `tools/check_deps.py` | Add a `--qgis-mode` flag that runs inside QGIS (uses `sys.executable` for pip) |
| `__init__.py` | After the optional-dep check, also detect missing **required** deps and log an actionable warning |
| `hydra_plugin.py` | Wire the "Check & Install Dependencies" action into the settings dialog |

## Verification

1. **CI**: Push new tag → verify Windows ZIP contains `cudart64_12.dll` in plugin root
2. **Manual**: Extract ZIP on clean Windows → launch QGIS → verify `swe2d_gpu_available()` returns `True`
3. **Settings**: Open Settings → set custom CUDA path → restart → verify custom path used
4. **Deps**: Click "Check & Install Dependencies" → verify `gmsh` installed into QGIS Python
5. **Reset**: Reset CUDA path to default → verify bundled DLL restored
