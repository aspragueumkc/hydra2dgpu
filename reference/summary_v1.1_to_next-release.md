# HYDRA2DGPU — Changes Since v1.1.0

**413 commits · 344 files changed · +568,960 / −19,806 lines**

---

## 1. Threading & Concurrency Architecture (Complete Rewrite)

The simulation pipeline was fundamentally re-architected to use `QThread` workers,
eliminating GUI freezes during long runs.

| Component | Description |
|-----------|-------------|
| `SimulationWorker` | Solver runs in a background `QThread` with progress signals, safe cancellation, and CUDA graph fallback retry |
| `PersistenceWorker` | GeoPackage writes happen off the main thread so QGIS stays responsive |
| `RunContext` | Captures all immutable simulation parameters on the main thread before the worker starts |
| `RunController` | Orchestrates the full lifecycle — worker creation → simulation → persistence → UI refresh |
| Live snapshot signals | Per-step data (coupling, line metrics) transferred from worker to main thread via `SnapshotData` signals |

**All 7 threading bugs fixed** — cross-thread dialog access, race conditions,
stale event wiring, missing `_results_data` initialization.

---

## 2. Headless CLI & Batch Simulation

A complete CLI interface enables QGIS-free batch runs.

- **`swe2d-cli` entry point** — Full headless runner (`swe2d/cli/` package)
  with GPKG adapter for QGIS-free mesh/BC reads
- **Batch runner** — Subprocess pool with parameter sweep expansion
  (`BatchOrchestrator`)
- **Batch simulation dialog** — GUI parameter grid with JSON config generation,
  subprocess execution, and multi-GPKG output
- **Pip-installable** via `pyproject.toml` — `pip install .` gives you `swe2d-cli`
- **Batch results plotter** — `tools/plot_baked_results.py` for headless
  output visualization

---

## 3. 1D Pipe Network (Pipe1D) GPU Solver

A complete GPU-accelerated 1D pipe solver was added.

- **Three solver modes**: EGL, Diffusion Wave, Fully Dynamic
- **Three GPU kernels**: `swe2d_pipe1d_step` with minor loss, wave speed, and
  boundary flux fixes
- **Pybind11 bindings** for all pipe solver functions
- **Inlet/exit loss** applied at pipe-end cells, not uniformly
- **HEC-22 exit loss** using actual flow area
- **GPU node depth management** and uninitialized variable fixes
- **Structures coupling** (weirs, culverts, gates, bridges, pumps) wired into
  headless runner

---

## 4. In-Memory Results & Live Viewing

Results visualization was transformed.

- **In-memory results** during live runs — no GPKG write until finalize
- **`SWE2DResultsData`** — Unified state object replacing scattered dialog
  attributes
- **pyqtgraph consolidation** — Removed dead matplotlib Network/Profile paths
- **Live snapshot ring buffer** — Memory-aware auto-dump with GPU-direct
  viewer design
- **Min-depth threshold & manual color ramp controls** for overlay
- **Overlay tab renamed** Display → Overlay; Runs split into its own Results
  toolbox page
- **Configurable `save_max_only`** — Trim intermediate snapshots, storing
  only terminal state + GPU max-tracking maxima

---

## 5. UX/UI Overhaul

Major GUI improvements across every tab.

- **Dedicated Run dock** — All Run/Output controls in their own dock widget
- **HYDRA toolbar** — Keyboard shortcuts for run/cancel/save/open/refresh
- **Settings dialog** — Feature flags for HYDRA functionality
- **Results controls** — Full Overlay redesign with separated Runs page
- **Topology tab** — Matches simulation tab pattern (group boxes + filter);
  Gmsh-only widgets flagged as advanced
- **Model tab** — GroupBox-organized parameters; Config button row split into
  labeled rows
- **Human-readable combo labels** — Everywhere
- **Tooltips on all widgets** — Complete coverage
- **Session persistence** across QGIS restarts
- **QML-based layer styles** — Portable attribute forms with conditional
  visibility for structures (18 layer schemas unified)

---

## 6. MVP Architecture (Model-View-Presenter)

A systematic Tier 1 + Tier 2 refactoring to enforce clean architecture.

- **View protocols** — Typed interfaces for all view interactions (dialog
  methods, file pickers, message boxes)
- **Controllers** — All UI logic moved out of views: topology, run, mesh,
  overlay controllers
- **Services extracted** — GeoTIFF export, mesh persistence, line sampling,
  BC preprocessing, coupling controller setup, RCMK permutation (10+
  extractions)
- **Dead code elimination** — ~500 LOC of confirmed dead code removed; 5
  silent robustness fallbacks removed from results read path
- **Qt-free logic split** — Line sampling, flow distribution, external
  sources logic extracted into services usable from worker threads

---

## 7. GPU Performance Optimizations

Significant CUDA kernel improvements.

- **Atomics-free Green-Gauss gradient** — Edge-scratch + cell-gather pattern
  eliminates warp divergence
- **Gradient AoS struct** — 48B/cell (one cache line) packing 6 gradient
  arrays
- **Edge CSR split** — Owned/peer arrays for better memory layout
- **Sorted-vector edge dedup** — 3–5× faster mesh build
- **On-device face-flux redistribution** — Zero PCIe transfers or Python loops
- **Cell renumbering** — Reverse Cuthill-McKee for GPU cache locality
- **Incremental line metrics** — O(n²) → O(n); skip array round-trip in
  profile persistence
- **Single-transaction persistence** — All baked results (mesh + line +
  coupling) written in one SQLite transaction

---

## 8. Solver Enhancements

- **On-device line metrics** — GPU ring buffer kernels with Python bindings
  for fully on-device line sampling
- **GPU max-tracking** (h/hu/hv maxima) — Persisted alongside baked results
- **Temporal scheme fix** — RK2/3/4/5 all corrected (stale coupling, buffer
  management, textbook RK4, Cash-Karp RK5)
- **Green-Gauss gradient fix** — Well-balanced for all 7 schemes on
  unstructured meshes
- **Rainfall-CN optimization** — Interval-based SCS-CN source computation
  (60 s windows), removed RK save/restore
- **Rain-on-GPU** — Nuked CPU rainfall path; constant-rate native rain with
  GPU sync timing
- **Polygon mesh support** — Works in both workbench and CLI runner
- **Constraint fixes** — DistMin/DistMax for large polygons; target_size
  enforcement

---

## 9. HDS-5 Culvert & Structure Validation

- **Dead zone fix** — Culvert outlet control no longer stalls
- **Velocity head** — Taken from upstream face cell, not enquiry cell
- **HDS-5 validation tests** — Added and passing
- **Structures attribute form** — QML-generated with conditional visibility,
  FHWA culvert codes, NULL-invert fallback

---

## 10. Code Quality & Testing

- **344 files changed**, 568K insertions, 19K deletions
- Extensive test suite: MVP boundary regression, batch orchestrator, mesh
  persistence roundtrip, CLI unit tests, pipe1d tests on GPU, HDS-5
  validation, coupling controller factory, RCMK permutation, engine class
  placement
- **All 83 bare `except:` patterns** eliminated via `fix_bare_excepts.py`
- `_call_workbench` delegation — Methods found on controllers with
  underscore-prefixed dialog method support
- Dead Python structure computation code, CPU fallback paths, old-format
  persistence functions all removed
