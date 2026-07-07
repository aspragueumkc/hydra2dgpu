# Changelog

All notable changes to HYDRA2DGPU are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- New `open_bc_relaxation` stability knob (Numerics / Stability) that damps reflections at OPEN, REFLECT, NORMAL_DEPTH, and NORMAL_DEPTH_SLOPE boundaries. Per-edge overrides can be supplied via a `bc_relax` field on the BC line layer.

---

## [1.2.0] — 2026-07-06

### Added

- **Headless CLI** (`swe2d-cli`) — QGIS-free batch execution with GPKG adapter
  for mesh/BC reads and sweep-parameter expansion
- **1D Pipe Network GPU solver** — Three solver modes (EGL, Diffusion Wave,
  Fully Dynamic) with pybind11 bindings and HEC-22 inlet/exit loss support
- **Batch simulation dialog** — GUI parameter grid with subprocess pool
  execution and multi-GPKG output
- **In-memory results viewing** — Live runs display without intermediate GPKG
  writes; final result set persisted at finalize via `SWE2DResultsData`
- **GPU max-tracking** — Per-step h/hu/hv maxima captured on-device and
  persisted alongside baked BLOB results
- **On-device line metrics** — GPU ring buffer kernels for fully on-device
  line sampling, eliminating PCIe transfers during the solve loop
- **Save-max-only mode** — Trim snapshot history to terminal state + max-
  tracking maxima to conserve disk space
- **Reverse Cuthill-McKee cell renumbering** — Improved GPU cache locality
  for unstructured meshes
- **Min-depth threshold and manual color ramp controls** for overlay
  visualization
- **View protocol dialog/message/file-picker methods** (Tier 2 MVP)
- **QML-generated portable layer styles** — Attribute forms with conditional
  visibility for all 18 structure layer schemas
- **Session persistence** across QGIS restarts
- **HYDRA toolbar** with keyboard shortcuts for run/cancel/save/open/refresh
- **Settings dialog** for feature-flag configuration
- **Dedicated Run dock** — All Run/Output controls in their own QDockWidget
- **Batch results plotter** (`tools/plot_baked_results.py`)
- **`fix_bare_excepts.py` tool** and elimination of all 83 bare `except:`
  patterns

### Changed

- **Threading architecture** — Solver loop moved from main thread to
  `SimulationWorker` (QThread) with `RunController` orchestration; GeoPackage
  persistence delegated to `PersistenceWorker`
- **MVP Tier 1 refactoring** — 10+ service extractions (GeoTIFF export, mesh
  persistence, line sampling, BC preprocessing, coupling controller setup,
  RCMK permutation, engine class placement, etc.)
- **MVP Tier 2 refactoring** — View protocols for dialog interaction; all
  controllers (topology, run, mesh, overlay) rewired to use protocol methods
- **Results visualization** — Consolidated to pyqtgraph; removed dead
  matplotlib Network/Profile paths
- **Overlay controls** — Renamed Display → Overlay tab; Runs split into
  its own Results toolbox page
- **Model tab** — GroupBox-organized parameters; Config button row split
  into labeled rows
- **Topology tab** — Group boxes + filter pattern matches simulation tab;
  Gmsh-only widgets flagged as advanced
- **Boundary Conditions** — Moved from Map tab to Solver Parameters section
- **Temporal schemes** — RK2/3/4/5 fully corrected (textbook RK4, Cash-Karp
  RK5, stale-coupling fix, buffer management)
- **Rainfall-CN** — Interval-based SCS-CN source computation (60 s windows);
  removed RK save/restore overhead
- **Save-mesh flow** — Select database first, then name entry
- **Structures attribute form** — Python editor config replaced with
  portable QML-based layer styles

### Performance

- **Atomics-free Green-Gauss gradient** — Edge-scratch + cell-gather pattern
- **Gradient AoS struct** — 48B/cell (one cache line) packing 6 gradient arrays
- **Edge CSR split** — Owned/peer arrays for improved memory layout
- **Sorted-vector edge dedup** — 3–5× faster mesh build
- **On-device face-flux redistribution** — Zero PCIe transfers or Python loops
- **Incremental line metrics** — O(n²) → O(n) for time-series assembly
- **Single-transaction persistence** — All baked results written in one
  SQLite connection + one transaction
- **CUDA graph fallback retry** — Automatic graph rebuild on launch failure

### Fixed

- **7 threading bugs** — Cross-thread dialog access, race conditions in
  worker finish handler, stale event wiring, missing `_results_data`
  initialization, finalizer calling view methods from worker thread, sample-
  line map built on worker thread
- **Green-Gauss gradient** — Well-balanced for all 7 spatial schemes on
  unstructured meshes
- **Pipe1D uninitialized variables** — `d_A_new`, `d_Q_new`, GPU node depth
  management
- **HEC-22 exit loss** — Uses actual flow area, not full pipe area
- **Culvert outlet-control dead zone** — No longer stalls on HDS-5 validation
  cases
- **Culvert velocity head** — Fetched from upstream face cell, not enquiry
  cell
- **GPU coupling** — Per-snap indexing for coupling snapshots; capture
  coupling arrays pre-permutation
- **Coupling readback** — Shared `SWE2DRunFinalizer` path between CLI and GUI
- **Rainfall silently dropped** with native device coupling active
- **RK4/RK5** — Stale-coupling in inner stages; correct buffer management;
  textbook coefficients for RK4
- **Constraint enforcement** — DistMin/DistMax cap for large polygons;
  `target_size` not being enforced
- **Topology tab filter** — Flag 74 niche Gmsh widgets as advanced
- **`is_cancel_requested` on destroyed backend** — `__del__` guard with
  `getattr`
- **CSR prefix-sum overflow** in cell renumbering
- **`model_to_ft` not wired** for culvert geometry in runtime coupling
- **Rain mm-to-model conversion** for USC unit systems in CLI
- **Overlay mesh loading** from per-run GPKG instead of stale `data.gpkg_path`
- **`StopIteration`** bug in `query_bc_arrays` (column TYPE not NAME)

### Removed

- **CPU solver fallback** — Dead GPU-host code, IMEX helpers, Python coupling
  path, CPU structure flow functions, old-format persistence (~500 LOC)
- **5 silent robustness fallbacks** in results read path (fail fast now)
- **`WorkbenchController` facade** — Eliminated circular delegation
- **Redundant `current_line_results_storage_path`** — Unified to
  View protocol
- **Unwired `extended_outputs_chk`** from Output Options
- **`_rain_rate_si_to_model()` and `_flow_si_to_model()`** — Broken calls in
  `_build_run_context`
- **`_WorkerResultsData`** — Parallel implementation eliminated in favor of
  shared `SWE2DResultsData`
- **All 83 bare `except:` patterns** — Replaced with specific exception
  types
- **GPU memory precision toggle** — Removed broken `SWE2D_STATE_FP32` option
- **Dead Python structure computation** — `compute_cell_source_terms`,
  `compute_cell_source_rate`, `compute_flux_adjustments`
- **Old-format GPKG persistence** — After baked BLOB migration

### Documentation

- Comprehensive spec documents added:
  - `docs/BAKED_MESH_RESULTS_SPEC.md` — Baked BLOB persistence design
  - `docs/SIMULATION_CONFIG_TABLE.md` — Run parameter reference
  - `docs/GUI_UX_RECOMMENDATIONS.md` — UX audit findings
  - `docs/TEMPORAL_SCHEME_FIX_SPEC.md` — Temporal scheme corrections
  - `docs/RESULTS_PERSISTENCE_BUGFIX_SPEC.md` — Results path audit
  - `docs/STRUCTURAL_PLACEMENT_AUDIT.md` — Structure coupling audit
  - `docs/CLI_GPKG_ADAPTER_AUDIT.md` — CLI/GPKG adapter review
  - `docs/DRAINAGE_EQUATION_PLAN.md` — Drainage equation design
  - `docs/PIPE1D_SOLVER_PLAN.md` — 1D pipe solver architecture
  - `docs/WORKBENCH_PERSISTENCE_PLAN.md` — Session persistence design
  - `docs/superpowers/` — Multi-plan simulation threading specification
- `AGENTS.md` — Spec-against-plan verification rule added

### Testing

- MVP boundary regression tests
- Batch orchestrator extraction tests
- Mesh persistence service round-trip tests
- CLI unit tests (sweep expansion, mesh round-trip)
- Pipe1d GPU path validation (daylighted pipe horizontal reservoir)
- HDS-5 culvert outlet-control validation
- Coupling-controller factory tests (Tier 1 #6b)
- RCMK permutation extraction tests (Tier 1 #6a)
- BC classification + coupling validation tests (Tier 1 #6c)
- Engine class placement tests (Tier 1 #9)
- Tier 2 protocol completeness tests for dialog methods
- Dead-code removal regression test
- Error-path regression tests for `log_fn` wiring
- GMSH timing benchmarks
- ANUGA validation comparison tests and documentation

---

## [1.1.0] — 2025-06-xx

### Added
- GPU finite-volume solver with unstructured mesh FVM
- Multiple spatial schemes (First-order, MUSCL, WENO5)
- Multiple temporal schemes (Euler, RK2, RK4)
- Boundary conditions (Wall, inflow, stage, open, normal depth, hydrograph)
- 1D drainage coupling (SWMM-style)
- Hydraulic structures (FHWA HDS-5 culverts, weirs, gates, bridges, pumps)
- Rainfall & infiltration with SCS Curve Number
- Results export (GeoPackage, UGRID NetCDF, GeoTIFF, CSV)
- QGIS plugin integration

[1.2.0]: https://github.com/aspragueumkc/hydra2dgpu/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/aspragueumkc/hydra2dgpu/releases/tag/v1.1.0
