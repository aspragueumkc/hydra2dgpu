# HYDRA — GPU-Accelerated 2D Shallow Water Equation Plugin for QGIS

[MIT License](LICENSE)

HYDRA is a QGIS plugin for 2D shallow water equation (SWE) modeling with a CUDA-accelerated finite-volume solver. It couples surface hydrodynamics, 1D urban drainage networks, hydraulic structures (weirs, culverts, gates, bridges, pumps), and rainfall/infiltration — all within the QGIS map canvas.

## Features

- **GPU-accelerated solver** — Full CUDA path with graph caching for high throughput
- **Unstructured mesh FVM** — Triangles, quads, and general polygons via Gmsh,  or built in backend (triangles only)
- **Multiple spatial schemes** — First-order, MUSCL (Fast/MinMod/MC/Van Leer), WENO5
- **Multiple temporal schemes** — Euler, RK2, RK4, Graph-safe RK4/RK5
- **Boundary conditions** — Wall, inflow, stage, open, normal depth, hydrograph timeseries
- **1D drainage coupling** — SWMM-style pipe networks (EGL, Diffusion, Dynamic wave)
- **Hydraulic structures** — FHWA HDS-5 culverts, weirs, gates, bridges, pumps
- **Rainfall & infiltration** — Rain-on-grid with SCS Curve Number
- **Results export** — GeoPackage, UGRID NetCDF, GeoTIFF, CSV

## Requirements

| Component | Requirement |
|---|---|
| QGIS | 3.28+ |
| Python | 3.12+ |
| CUDA Toolkit | 11.x or 12.x |
| NVIDIA GPU | Compute Capability ≥ 7.5 |
| C++ Compiler | GCC 10+ or Clang 12+ (C++17) |
| CMake | 3.16+ |

> **Platform support:** Linux (x86_64) and Windows (x86_64) only. NVIDIA CUDA GPU
> is **required** — there is no CPU fallback path. macOS is not currently supported.
> Future releases may add Intel/AMD GPU compatibility via SYCL or HIP; Apple Silicon
> support could follow from those backends given its unified memory architecture, but
> there are no concrete plans at this time.

## Quick Start

Install the latest release without compiling anything — QGIS downloads
the GPU backend for you on first launch.

1. **Download** `HYDRA2DGPU-<version>.zip` from
   [GitHub Releases](https://github.com/aspragueumkc/hydra2dgpu/releases).
   The asset is named with the plugin version (e.g.
   `HYDRA2DGPU-1.3.0.zip`) so you can tell which release it belongs to.

   > The plugin is **not** in the official QGIS plugin repository yet; use
   > the GitHub release zip and QGIS's *Install from ZIP* flow described
   > below.

2. **Install in QGIS.** Open QGIS → **Plugins → Manage and Install
   Plugins → Install from ZIP** and select the downloaded zip.

3. **Restart QGIS.** The first launch of HYDRA2DGPU opens the
   *Check & Install Dependencies* dialog, which downloads the matching
   `hydra_swe2d-<version>-cp<python>-cp<python>-<platform>.whl` into
   `~/.hydra2dgpu/` (or `%USERPROFILE%\.hydra2dgpu\` on Windows) and
   installs it into an isolated virtual environment so it does not
   interfere with your system Python. Click *Install* once.

4. **Verify.** Open QGIS's Python Console (`Plugins → Python Console`)
   and run:
   ```python
   from swe2d.runtime.backend import swe2d_gpu_available
   print(f"GPU available: {swe2d_gpu_available()}")
   ```

The plugin source ships **Python only**; the native solver comes from
the downloaded wheel, which keeps the install zip small (well under
the 20 MB QGIS plugin repo limit) and lets the same plugin zip work
on Linux and Windows — the platform-specific wheel is fetched at
first launch.

### Dependencies

| Component | Installed by | Notes |
|---|---|---|
| `numpy`, `gmsh` | QGIS Python env (`pip install -r requirements.txt` from the plugin source) | Required for mesh I/O and mesh generation |
| `h5py`, `netCDF4`, `matplotlib` | Optional | Only if you want HEC-RAS HDF5 / UGRID NetCDF result export or in-plugin plots |
| `hydra_swe2d-<ver>-<plat>.whl` | Auto on first launch | Native GPU/CUDA backend; lives in `~/.hydra2dgpu/lib/python*/site-packages/` |
| `QGIS`, `PyQt5`, `osgeo` (GDAL) | QGIS itself | Do **not** `pip install` these |

If the first-launch installer cannot reach GitHub Releases (offline,
rate-limited, or behind a corporate proxy), re-run it from
**HYDRA2DGPU → Settings → Check & Install Dependencies**. You can
override the download URL with the `HYDRA_SWE2D_WHEEL_URL` environment
variable for air-gapped or mirror deployments.

## Build from Source (Advanced)

The pre-compiled install path covers the public install. Building from
source is only needed if you are developing the GPU solver or
debugging a wheel build locally.

```bash
git clone https://github.com/aspragueumkc/hydra2dgpu.git
cd hydra2dgpu

# Build the wheel against your local CUDA toolkit
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

> **Mixed precision (experimental):** Add `-DSWE2D_STATE_FP32=ON` to the
> cmake command for ~35% less GPU memory traffic. The precompiled
> binaries use full `double` precision.

Then symlink or install the plugin root into your QGIS plugins
directory and restart QGIS. The plugin will prefer the locally-built
wheel over a downloaded one only if you also update
`qgis_plugin/HYDRA2DGPU/installer.py::WHEEL_VERSION` to match your
build.

## Documentation

**[Documentation Index](docs/INDEX.md)** — All guides organized by audience (users, developers, C++ engineers).

- **[User Guide](docs/USER_GUIDE.md)** — Installation, Studio UI, running your first simulation
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** — Architecture, module reference, style guide, test suite
- **[GPU Architecture Report](docs/SWE2D_GPU_ARCHITECTURE_REPORT.md)** — Deep-dive on the GPU solver
- **[Model GeoPackage Schema](docs/MODEL_GEOPACKAGE_SCHEMA.md)** — Input GPKG tables
- **[Results GeoPackage Schema](docs/RESULTS_GEOPACKAGE_SCHEMA.md)** — Output GPKG tables

A pre-built [knowledge graph](graphify-out/GRAPH_REPORT.md) of the codebase is also available.

## Repository Layout

```
swe2d/                   Python package (solver API, extensions, workbench)
  runtime/               Backend creation and GPU interface
  extensions/            Drainage, structures, rainfall modules
  boundary_and_forcing/  BC sampling and hydrograph handling
  mesh/                  Mesh I/O and topology
  results/               Result queries, export, run management
  plotting/              Qt-free figure dispatch service
  workbench/             QGIS workbench (views, controllers, dialogs)
cpp/src/                 CUDA/C++ solver, mesh, numerics, and bindings
tests/                   Solver validation and GPU performance tests
tools/                   Build helpers and dev utilities
docs/                    Design notes, guides, Doxygen API reference
```

## Testing

```bash
# GPU validation suite (primary acceptance gate)
PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v \
  tests.test_swe2d_gpu_validation_perf \
  tests.test_swe2d_gpu_unstructured \
  tests.test_swe2d_gpu_dambreak \
  tests.test_workbench_gui
```

Additional tests are listed in `.github/workflows/test.yml`.

## License

MIT — see [LICENSE](LICENSE).
