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

## Quick Start

```bash
# Clone
git clone https://github.com/aspragueumkc/hydra2dgpu.git
cd hydra2dgpu

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

> **Mixed precision (experimental):** Add `-DSWE2D_STATE_FP32=ON` to the cmake command for ~35% less GPU memory traffic. Precompiled binaries use full `double` precision.

Then symlink or install the plugin root into your QGIS plugins directory and restart QGIS.

### Python Dependencies

After installing the plugin, install Python dependencies into your QGIS Python environment:

```bash
# Check which packages are installed
python tools/check_deps.py

# Install required + optional packages
pip install -r requirements.txt
```

| Package | Required | Purpose |
|---------|----------|---------|
| `numpy` | ✅ | Array operations, mesh data |
| `gmsh` | ✅ | Unstructured mesh generation (Gmsh backend) |
| `h5py` | ❌ | HEC-RAS HDF5 result export |
| `netCDF4` | ❌ | UGRID NetCDF result export |
| `matplotlib` | ❌ | In-plugin plotting |

> **Note**: `QGIS`, `PyQt5`, and `osgeo` (GDAL) are provided by QGIS itself — do **not** install these via pip.

### Pre-compiled Binaries

*(Updated after each tagged release via GitHub Actions — see [GitHub Releases](https://github.com/aspragueumkc/hydra2dgpu/releases).)*

If you don't want to build from source, download the pre-compiled binary for your platform from the releases page:

1. Download `hydra2gpu-linux-x86_64.zip` or `hydra2gpu-windows-x86_64.zip`
2. Extract `hydra2dgpu/` into your QGIS plugins directory
3. Restart QGIS
4. Run `pip install -r requirements.txt` in your QGIS Python environment

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
