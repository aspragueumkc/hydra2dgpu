# HYDRA — GPU-Accelerated 2D Shallow Water Equation Plugin for QGIS

[MIT License](LICENSE)

HYDRA is a QGIS plugin for 2D shallow water equation (SWE) modeling with a CUDA-accelerated finite-volume solver. It couples surface hydrodynamics, 1D urban drainage networks, hydraulic structures (weirs, culverts, gates, bridges, pumps), and rainfall/infiltration — all within the QGIS map canvas.

## Features

- **GPU-accelerated solver** — Full CUDA path with graph caching for high throughput
- **Unstructured mesh FVM** — Triangles, quads, and general polygons via Gmsh, TQMesh, or structured backends
- **Multiple spatial schemes** — First-order, MUSCL (Fast/MinMod/MC/Van Leer), WENO5
- **Multiple temporal schemes** — Euler, RK2, RK4, Graph-safe RK4/RK5
- **Boundary conditions** — Wall, inflow, stage, open, normal depth, hydrograph timeseries
- **1D drainage coupling** — SWMM-style pipe networks (EGL, Diffusion, Dynamic wave)
- **Hydraulic structures** — FHWA HDS-5 culverts, weirs, gates, bridges, pumps
- **Rainfall & infiltration** — Rain-on-grid with SCS Curve Number
- **Results export** — GeoPackage, HEC-RAS HDF5, UGRID NetCDF, GeoTIFF, CSV

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

Then symlink or install the plugin root into your QGIS plugins directory and restart QGIS.

## Documentation

**[User Guide](docs/USER_GUIDE.md)** — Full documentation including installation, model setup, solver configuration, hydraulic theory, and API reference.

## Repository Layout

```
swe2d/                   Python package (solver API, extensions, workbench)
  core/                  Solver configuration and enums
  runtime/               Backend creation and GPU interface
  extensions/            Drainage, structures, rainfall modules
  boundary_and_forcing/  BC sampling and hydrograph handling
  mesh/                  Mesh I/O and topology
  results/               Export (HDF5, NetCDF, GeoTIFF, CSV)
  workbench/             QGIS workbench GUI methods
cpp/src/                 CUDA/C++ solver, mesh, numerics, and bindings
forms/                   Qt Designer UI files
tests/                   Solver validation and GPU performance tests
tools/                   Build helpers and dev utilities
docs/                    Design notes, architecture, and user guide
```

## Testing

```bash
# GPU validation suite (primary acceptance gate)
PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v tests.test_swe2d_gpu_validation_perf tests.test_swe2d_gpu_unstructured
```

## License

MIT — see [LICENSE](LICENSE).
