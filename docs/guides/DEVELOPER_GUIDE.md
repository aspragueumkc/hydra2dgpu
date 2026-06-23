# HYDRA 2D GPU — Developer Guide

**Document version**: 2.0  
**Last updated**: 2026-06-14

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Module Reference](#2-module-reference)
3. [C++ Native Module Reference](#3-c-native-module-reference)
4. [Style Guide](#4-style-guide)
5. [Test Suite Guide](#5-test-suite-guide)
6. [Test Coverage Gaps](#6-test-coverage-gaps)
7. [Common Development Workflows](#7-common-development-workflows)

---

## 1. Architecture Overview

### 1.1 High-Level Layering

```
┌──────────────────────────────────────────────────────────┐
│                   QGIS Plugin Layer                       │
│  hydra_plugin.py, swe2d/workbench/studio_dialog.py       │
│  swe2d/workbench/views/ (8+ view modules)                │
│  swe2d/workbench/controllers/ (5 MVP controllers)        │
│  forms/*.ui (~12 designers)                              │
├──────────────────────────────────────────────────────────┤
│                  swe2d/ Python Package                    │
│  ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌───────────┐ │
│  │  core/   │ │  mesh/   │ │  runtime/  │ │ results/  │ │
│  │ (re-     │ │ (mesh    │ │ (solver    │ │ (post-    │ │
│  │  export) │ │  gen)    │ │  orchest.) │ │  process) │ │
│  ├──────────┤ ├──────────┤ ├────────────┤ ├───────────┤ │
│  │boundary  │ │extensions│ │  units.py  │ │workbench/ │ │
│  │_and_     │ │(drainage,│ │  (CRS      │ │(services, │ │
│  │forcing/  │ │ struct.) │ │   conv)    │ │  widgets) │ │
│  └──────────┘ └──────────┘ └────────────┘ └───────────┘ │
├──────────────────────────────────────────────────────────┤
│              C++ Native Modules (hydra_swe2d)              │
│  swe2d_bindings.cpp — solver/mesh/boundary/coupling       │
│  swe2d_mesh.cpp/hpp — mesh construction                   │
│  swe2d_solver.cpp/hpp — CPU solver backend                │
│  swe2d_gpu.cu/.cuh — CUDA kernels                        │
│  swe2d_numerics.cpp/hpp — Riemann solvers, flux schemes    │
│  hybrid_mesh_bindings.cpp — hybrid mesh builder           │
│  meshing_native_bindings.cpp — native mesh helpers        │
│  overlay_backend.cpp — high-performance canvas overlay    │
└──────────────────────────────────────────────────────────┘
```

### 1.2 Inter-Module Dependency Graph

```
swe2d/units.py  (standalone — CRS-derived unit conversion)
   ↓
   ├──→ swe2d/runtime/backend.py
   ├──→ swe2d/runtime/coupling.py  ──→ swe2d/extensions/drainage_network.py
   │                                     └──→ swe2d/extensions/structures.py
   │                                          └──→ swe2d/extensions/extension_models.py
   ├──→ swe2d/mesh/bridge_stacked_mesh.py ──→ swe2d/mesh/mesh_runtime_logic.py
   ├──→ swe2d/results/panel.py
   └──→ swe2d/workbench/startup_state.py
   └──→ swe2d/workbench/views/studio_viewer_plot.py

swe2d/boundary_and_forcing/
   ├── bc_logic.py  (standalone — core BC math)
   ├── boundary_runtime_logic.py  (standalone)
   ├── hydrograph_logic.py  (standalone)
   ├── rainfall.py  (standalone)
   ├── internal_flow_logic.py  (standalone)
   ├── boundary_qgis_adapter.py  ──→ (QGIS)
   ├── internal_flow_qgis_adapter.py  ──→ internal_flow_logic.py, internal_flow_qgis_geometry.py
   ├── spatial_forcing_qgis_adapter.py  (standalone)
   └── runtime_source_logic.py  ──→ swe2d.runtime.backend (via arg)

swe2d/runtime/
   ├── backend.py  ──→ native_binding_compat.py, swe2d/units.py, swe2d/extensions/extension_models.py
   ├── coupling.py  ──→ swe2d/extensions/*, swe2d/units.py, bridge_stacked_runtime.py
   ├── runtime_step_executor.py  ──→ swe2d/boundary_and_forcing/bc_logic.py
   ├── run_orchestrator.py  ──→ (coordinates all other runtime modules)
   └── (8 other runtime seam modules)

swe2d/mesh/
   ├── meshing.py  (standalone — optionally loads hydra_meshing_native)
   ├── mesh_runtime_logic.py  (standalone)
   └── bridge_stacked_mesh.py  ──→ swe2d/extensions/extension_models.py

swe2d/results/
   ├── queries.py  ──→ db_utils.py
   ├── velocity_layer.py  ──→ db_utils.py
   ├── panel.py  ──→ animation.py, db_utils.py, swe2d/units.py
   └── db_utils.py  (standalone)

swe2d/workbench/
   ├── controllers/  (5 MVP controllers: run, overlay, topology, etc.)
   ├── services/     (Qt-free business logic + persistence)
   ├── views/        (PlotViewWidget, TopologyTabView, studio_viewer, etc.)
   ├── studio_dialog.py  (main UI host)
   └── studio_results_panel.py  (results panel orchestration)
```

### 1.3 Key Design Decisions

| Decision | Rationale |
|---|---|
| **GPU-first** | `SWE2DBackend` requires `hydra_swe2d` CUDA module. CPU-only paths are deprecated. |
| **CRS-agnostic units** | All geometry in model units. `swe2d/units.py` converts at the coupling boundary. HDS-5 culvert is the only internal USC path. |
| **Conceptual topology → mesh** | Inspired by HEC-RAS 2025. Users define regions/arcs/constraints; backends (Gmsh/Structured/TQMesh) translate to computational mesh. |
| **CSR topology storage** | `cell_face_offsets` + `cell_face_nodes` for general polygon solver support. |
| **MVP architecture** | View layer (Qt widgets) → Controller → Service layer (Qt-free). See `.opencode/rules/MVP_ARCHITECTURE.md`. |
| **Native module compatibility** | `native_binding_compat.py` handles three-tier fallback for `swe2d_create_solver` signature changes across build versions. |

---

## 2. Module Reference

### 2.1 `swe2d/units.py` — CRS-Derived Unit Conversion

Called once at startup via `configure(length_scale_si_to_model)`.

```python
configure(length_scale_si_to_model: float) -> None   # Call once
si_m_per_model() -> float                              # SI m per model unit
model_per_si_m() -> float                              # Model units per SI m
si_m2_per_model_area() -> float                        # SI m² per model area
si_m3_per_model_volume() -> float                      # SI m³ per model volume
gravity() -> float                                     # Model units (9.81 or 32.17)
model_to_ft() -> float                                 # For HDS-5 culverts
manning_factor() -> float                              # 1.0 (SI) or 1.486 (USC)
compute_length_factor() -> float                       # Deprecated
```

**No tests exist** — this is a critical gap (see §6).

### 2.2 `swe2d/mesh/` — Meshing Pipeline

#### `meshing.py` (10,337 LOC)

The core meshing module. Three backends share a common abstract base:

```
MeshingBackend (ABC)
  ├── GmshBackend           — Gmsh 4.x (default, requires pip install gmsh)
  ├── StructuredFaceCentricBackend — Deterministic grid (no deps, fallback)
  └── TQMeshBackend         — Advancing-front (requires hydra_tqmesh native)
```

**Data model classes:**

| Class | Purpose |
|---|---|
| `ConceptualNode` | Topology node (node_id, x, y) |
| `ConceptualArc` | Breakline/channel arc (arc_id, nodes, region_id, arc_role, polyline) |
| `ConceptualRegion` | Polygonal domain region (ring, default_size, cell_type, holes) |
| `CellConstraint` | Local refinement zone (polygon, target_size, cell_type) |
| `QuadEdgeControl` | Boundary-layer edge spacing (region_id, edge_id, n_layers, growth_rate) |
| `ConceptualModel` | Container for all conceptual geometry |
| `MeshResult` | Output: node_x/y/z, cell_face_offsets/nodes, cell_type, region_id, target_size |

**Entry points:**

```python
def generate_face_centric_mesh(
    model: ConceptualModel,
    backend: str = "gmsh",
    options: Optional[Dict[str, object]] = None,
) -> MeshResult

def conceptual_from_qgis_layers(
    nodes_layer, arcs_layer, regions_layer,
    constraints_layer=None, quad_edges_layer=None,
    default_size=20.0, default_cell_type="triangular",
) -> ConceptualModel
```

#### `mesh_runtime_logic.py` — Runtime Mesh Analytics

```python
mesh_cell_centroids(mesh) -> (cx, cy)
mesh_cell_areas(mesh, node_x, node_y) -> areas
mesh_cell_min_bed(mesh) -> min_bed     # Per-cell min node z
mesh_cell_solver_bed(mesh) -> bed       # Per-cell solver bed elevation
inflow_adjacent_cells(mesh) -> cells    # Cells adjacent to boundary
boundary_buffer_cells(mesh, n_layers) -> cells  # Boundary buffer zones
initial_state(h0, bed, wse_or_depth) -> h0_per_cell  # Initial water depth
```

**No tests exist** — all 7 functions are untested (§6).

#### `bridge_stacked_mesh.py` — Bridge Stacked Geometry

```python
build_bridge_stacked_plan(mesh, structure, ...) -> BridgeStackedPlan
bridge_specs_from_structure_config(config) -> List[BridgeStackedGeometrySpec]
```

### 2.3 `swe2d/boundary_and_forcing/` — BC & Source Forcing

#### `bc_logic.py` — Core Boundary Condition Math

```python
interp_hydrograph(times, values, query_t) -> float
distribute_total_flow_to_unit_q(total_q, edge_lengths) -> ndarray
normalize_inflow_to_uniform_velocity(h, bed, ...) -> ndarray
apply_timeseries_bc_values(bc_type, bc_value, t) -> (type, value)
```

#### `boundary_runtime_logic.py` — Mesh Edge Extraction

```python
mesh_boundary_edges(mesh) -> (n0, n1, edge_len)  # All boundary edges
collect_boundary_arrays(mesh, ...) -> (bc_n0, bc_n1, bc_tp, bc_vl)  # Classified
```

#### `hydrograph_logic.py` — Hydrograph Parsing

```python
parse_time_hours(text) -> float   # "HH:MM:SS" or decimal hours
parse_hydrograph_text(text) -> Hydrograph  # Multi-line hydrograph
hydrograph_from_layer(layer, ...) -> Hydrograph
```

#### `internal_flow_logic.py` — Source/Sink Processing

```python
resolve_internal_flow_field_name(layer, prefix) -> str
first_matching_field(layer, candidates) -> str
build_hydrograph_lookup_from_features(features, ...) -> dict
resolve_layer_hydrograph_for_feature(feature, ...) -> Hydrograph
build_internal_flow_forcing_from_features(features, ...) -> list
```

**No tests exist** — all 5 functions untested.

#### `spatial_forcing_qgis_adapter.py` — Spatial Manning/CN/Rain

```python
build_spatial_manning_array_qgis(mesh, manning_layer, ...) -> ndarray
build_spatial_cn_array_qgis(mesh, cn_layer, ...) -> ndarray
build_thiessen_rain_cn_forcing_qgis(mesh, gage_layer, ...) -> forcing
```

**No tests exist** — all 3 functions untested (QGIS-dependent).

#### `rainfall.py` — Rain-On-Grid

```python
class SWE2DRainfallModule(RainfallSourceEngine):
    def cell_source_term(self, cell_idx, t) -> float
```

### 2.4 `swe2d/runtime/` — Solver Orchestration

#### `backend.py` — Python ↔ C++ Bridge

```python
class SWE2DBackend:
    def __init__(self, use_gpu=True)           # Load native module
    def build_mesh(self, node_x, y, z, cell_nodes, ...) -> None
    def initialize(self, h0, n_mann=0.03, cfl=0.45, g=9.81, ...) -> None
    def step(self, dt_request=-1.0) -> SWE2DStepDiag
    def run_to_time(self, t_end, dt_max, ...) -> None
    def get_state(self) -> (h, hu, hv)
    def set_state(self, h, hu, hv) -> None
    def set_boundary_conditions(self, edge_n0, n1, bc_type, bc_val) -> None
    def set_boundary_hydrographs_native(self, edge_ids, times, values, codes) -> None
    def set_rain_cn_forcing_native(self, cell_gage_idx, ...) -> None
    def set_external_sources_native(self, source_mps) -> None
    def destroy(self) -> None

def swe2d_available() -> bool
def swe2d_gpu_available() -> bool
def load_swe2d_native_module()
```

#### `coupling.py` — Surface + Drainage + Structure Coupling

```python
class SWE2DCouplingController:
    def __init__(self, cell_area, cell_bed, drainage=None, structures=None, ...)
    def set_cell_centroids(self, cx, cy) -> None
    def compute_source_rates(self, t_s, dt_s, h, hu, hv) -> ndarray
    def save_coupling_pred(self) -> ndarray
    def average_coupling_sources(self, h_pred, ...) -> ndarray
    def restore_state_from_backup(self) -> None

    # SoA packing (all untested):
    pack_pipe_network_soa(...) -> SWE2DDrainageSoA
    pack_structures_soa(...) -> SWE2DStructuresSoA
    pack_coupling_soa(...) -> SWE2DCouplingSoA

class SWE2DCouplingDiagnostics:
    @property
    def total_drainage_inflow(self) -> float
    def total_structure_inflow(self) -> float
    # ... etc
```

#### `native_binding_compat.py` — pybind11 Compatibility

```python
call_solver_create_compat(mod, py_mesh, ...) -> handle  # 3-tier signature fallback
log_feature_unavailable(name) -> None
```

#### Run Seam Classes

| Class | File | Key Method |
|---|---|---|
| `SWE2DBackendInitializer` | `backend_initializer.py` | `build_and_initialize()` |
| `SWE2DRunController` | `run_controller.py` | `run_preflight()` |
| `SWE2DRunDataBuilder` | `run_data_builder.py` | `build() -> SWE2DRunInputData` |
| `SWE2DRunOptionsBuilder` | `run_options_builder.py` | `build() -> SWE2DRunOptionsData` |
| `SWE2DRunOrchestrator` | `run_orchestrator.py` | `run(request) -> bool` |
| `SWE2DRunFinalizer` | `run_finalizer.py` | `finalize_and_persist()` |
| `SWE2DRunLifecycle` | `run_lifecycle.py` | `handle_run_failure()`, `finalize_cleanup()` |
| `SWE2DRuntimeReporter` | `runtime_reporting.py` | `process_step() -> diag` |
| `SWE2DRuntimeStepExecutor` | `runtime_step_executor.py` | `execute_step() -> diag` |
| `SWE2DRuntimeSourceManager` | `runtime_sources.py` | `cell_source_model_at_time()`, `accumulate_source_volume_model()` |
| `SWE2DRunSetupConfigurator` | `runtime_setup_configurator.py` | `configure_native_rain_cn_forcing()`, `configure_native_source_injection()` |

### 2.5 `swe2d/extensions/` — Multi-Physics Models

#### `extension_models.py` — Data Models & Enums

```python
# Enum definitions:
class SpatialDiscretization(IntEnum): FV_FIRST_ORDER=0, FV_MUSCL_FAST=1, ...
class TemporalScheme(IntEnum): EULER_1ST=1, SSP_RK2=2, GRAPH_SAFE_RK4=5, ...
class TurbulenceModel(IntEnum): NONE=0, SMAGORINSKY=1, ...
class BedFrictionModel(IntEnum): MANNING=0, CHEZY=1, DARCY_WEISBACH=2, ...
class GodunovSolverMode(IntEnum): RUSANOV=0, HLL=1, HLLC=2
class DrainageSolverMode(IntEnum): EGL=0, DIFFUSION=1, DYNAMIC=2
class StructureType(IntEnum): WEIR=1, CULVERT=2, GATE=3, BRIDGE=4, PUMP=5

# Hydraulic helper functions (all untested):
circular_area_from_diameter(d) -> float
equivalent_circular_diameter_from_area(a) -> float
circular_wet_perimeter_full(d) -> float
circular_section_from_depth(d, depth) -> (area, perimeter, top_width)
compute_orifice_flow(...) -> float
compute_weir_flow(...) -> float
compute_pipe_manning_capacity_full(d, slope, n) -> float
convert_cell_flows_to_depth_rates(flows, areas) -> ndarray

# Config classes:
@dataclass class SolverModelOptions
@dataclass class PipeNetworkConfig
@dataclass class HydraulicStructureConfig
@dataclass class HydraulicStructure
@dataclass class DrainageNode, DrainageLink, InletExchange, OutfallExchange, PipeEndExchange
@dataclass class RainFieldConfig

# Engine base classes:
class RainfallSourceEngine     # skeleton
class DrainageCouplingEngine   # skeleton
class HydraulicStructureEngine # skeleton
```

#### `drainage_network.py` — 1D Drainage Network Solver

```python
class SWE2DUrbanDrainageModule(DrainageCouplingEngine):
    def __init__(self, cfg: PipeNetworkConfig)
    def initialize(self) -> None
    def step(self, dt_s, cell_wse_2d=None) -> (sinks, sources)
```

#### `structures.py` — Hydraulic Structures

```python
class SWE2DStructureModule(HydraulicStructureEngine):
    def __init__(self, cfg, model_to_ft=1.0)
    def structure_flows(self, cell_wse, dt_s) -> List[float]
    def structure_details(self, cell_wse) -> List[dict]
```

### 2.6 `swe2d/results/` — Post-Processing

```python
class ResultsAnimationController(QtCore.QObject):
    # signals: current_timestep_changed, play_state_changed

class SWE2DResultsPanel:    # Full featured dockable panel (1800 LOC)

class VelocityVectorBuilder:
    def load_snapshot(...) -> VelocitySnapshot
    def build_streamline_traces(...) -> traces

# Data layer:
class ResultsDataset
list_available_runs(gpkg_path) -> list
load_timeseries(gpkg_path, run_name, ...) -> DataFrame
load_profile(gpkg_path, run_name, ...) -> DataFrame

# DB utilities:
open_ro(path) -> sqlite3.Connection
table_exists(conn, name) -> bool
table_columns(conn, name) -> list
```

### 2.7 `swe2d/workbench/` — Non-GUI Utilities

```python
# QGIS utilities (non_gui_qgis.py):
resolve_layer_field_name(layer, candidates) -> str
parse_feature_float(feature, field, default) -> float
infer_obj_path_from_layer_3d_renderer(layer) -> str
build_patch_terrain_surface(...)

# Runtime utilities (non_gui_runtime.py):
build_mesh_snapshot_rows(mesh, ...) -> list
parse_obj_scale_value(text) -> float
# Environment variable helpers:
_env_bool(name, default) -> bool
_env_float(name, default) -> float
_env_int(name, default) -> int

# Project settings (project_settings.py):
read_project_entry_text(key) -> str
load_project_json(key) -> dict
write_project_json(key, data) -> bool
```

### 2.8 `swe2d/core/` — Backward-Compat Re-Export Shim

Re-exports from `swe2d.runtime`, `swe2d.extensions`, and `swe2d.mesh`:

```python
SWE2DBackend, swe2d_available, swe2d_gpu_available
SWE2DCouplingController, pack_coupling_soa
SWE2DUrbanDrainageModule
SWE2DStructureModule
# plus wildcard from extension_models
```

---

## 3. C++ Native Module Reference

### 3.1 Module: `hydra_swe2d` (swe2d_bindings.cpp)

Built by CMake, produces `hydra_swe2d.cpython-*.so`.

**Mesh construction:**
```cpp
swe2d_build_mesh(node_x, node_y, node_z, cell_nodes) -> MeshHandle
swe2d_build_mesh_poly(node_x, node_y, node_z, cell_face_offsets, cell_face_nodes) -> MeshHandle
swe2d_mesh_info(handle) -> (n_nodes, n_cells)
swe2d_boundary_edges(handle) -> (edge_n0, edge_n1)
```

**Solver lifecycle:**
```cpp
swe2d_create_solver(mesh_handle, ...) -> SolverHandle
swe2d_step(solver, dt_request) -> (h, hu, hv, diag)
swe2d_run_to_time(solver, t_end, dt_max) -> void
swe2d_get_state(solver) -> (h, hu, hv)
swe2d_set_state(solver, h, hu, hv) -> void
swe2d_destroy(solver) -> void
```

**Boundary conditions:**
```cpp
swe2d_set_boundary_values(solver, bc_n0, bc_n1, bc_type, bc_val) -> void
swe2d_solver_set_boundary_values(solver, bc_type, bc_val, ...) -> void
swe2d_solver_set_boundary_hydrographs(solver, edge_ids, times, values, codes) -> void
swe2d_solver_set_progressive_bc_data(solver, ...) -> void
```

**Forcing:**
```cpp
swe2d_solver_set_rain_cn_forcing(solver, cell_gage_idx, ...) -> void
swe2d_solver_set_external_sources(solver, source_mps) -> void
```

**Coupling:**
```cpp
swe2d_gpu_compute_coupling_sources(...) -> void
swe2d_gpu_compute_bridge_coupling_sources(...) -> void
swe2d_gpu_set_coupling_dt(...) -> void
swe2d_gpu_upload_culvert_face_flux_params(...) -> void
swe2d_gpu_redistribute_structure_sources(...) -> void
swe2d_gpu_drainage_step(...) -> void
swe2d_gpu_set_culvert_solver_mode(mode) -> void
swe2d_gpu_build_culvert_tables(input_np) -> dict
```

**GPU queries:**
```cpp
swe2d_gpu_available() -> bool
```

### 3.2 Module: `hydra_meshing_native` (meshing_native_bindings.cpp)

Optional C++ extensions for Gmsh-related polyline distance/overlap computations.

### 3.3 Module: `hydra_hybridmesh` (hybrid_mesh_bindings.cpp)

Alternative channel-guided meshing with constrained edge recovery.

### 3.4 Module: `hydra_overlay_backend` (overlay_backend.cpp)

High-performance canvas overlay rendering.

### 3.5 Build System

```cmake
cmake_minimum_required(VERSION 3.16)
project(hydra2dgpu)
# C++17, CUDA optional via option(USE_CUDA "..." ON)
# Auto-fetches pybind11 v2.13.6 via FetchContent
```

Build targets: `hydra_swe2d`, `hydra_meshing_native`, `hydra_hybridmesh`, `hydra_overlay_backend`.

---

## 4. Style Guide

### 4.1 Imports

```python
# ALWAYS absolute, never relative
from swe2d.runtime.backend import SWE2DBackend
from swe2d import units as _u

# Wildcard imports: avoid in new code. Use explicit re-exports.
# Existing wildcards use # noqa: F403 for flake8 compatibility.

# Order: future → stdlib → third-party → swe2d (groups separated by blank line)
from __future__ import annotations

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from swe2d.runtime.backend import SWE2DBackend
```

### 4.2 Type Annotations

```python
# @dataclass with full type annotations
@dataclass
class SWE2DDrainageSoA:
    node_x: np.ndarray
    solver_mode: int = int(DrainageSolverMode.EGL)

# Every function MUST have return type annotation
def si_m_per_model() -> float: ...
def build_mesh(self, ...) -> None: ...
def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

# Optional for nullable parameters
def initialize(self, hu0: Optional[np.ndarray] = None) -> None: ...

# Keyword-only arguments after * for optional parameters
def distribute_total_flow_to_unit_q(
    total_q: float,
    edge_lengths: np.ndarray,
    *,
    _side_idx: Optional[np.ndarray] = None,
) -> np.ndarray: ...
```

### 4.3 Naming

| Category | Convention | Examples |
|---|---|---|
| Classes | PascalCase with `SWE2D` prefix | `SWE2DBackend`, `SWE2DRunController` |
| Functions/methods | snake_case | `build_mesh`, `set_boundary_conditions` |
| Module constants | UPPER_CASE | `USC_FT_PER_SI_M`, `BW2D_BIG` |
| Private helpers | `_` prefix | `_repair_mesh_result`, `_polygon_area_xy` |
| Type aliases | PascalCase | `Hydrograph = Tuple[np.ndarray, np.ndarray]` |
| Enum values | UPPER_CASE | `FV_FIRST_ORDER`, `SSP_RK2` |
| Test classes | `Test` prefix | `TestGPUUnstructuredDamBreak` |
| Test methods | `test_` prefix | `test_stoker_linf_error` |

### 4.4 Logging

```python
import logging
logger = logging.getLogger(__name__)

# Use %s formatting (NOT f-strings) for deferred evaluation
logger.warning("[BACKEND] GPU availability check failed: %s", exc)
logger.info("mesh built: %d nodes, %d cells", n_nodes, n_cells)

# Include exc_info=True for exception logging
logger.warning("hydra_meshing_native import failed: %s", e, exc_info=True)

# Diagnostics mode: BACKWATER_SWE2D_DIAG_MODE=1 enables DEBUG level
```

### 4.5 Error Handling

```python
# Precondition guards: check then raise
if self._solver_h is None:
    raise RuntimeError("initialize() must be called before step().")

# Try/except for graceful degradation
try:
    self._boundary_edge_cells = ...
except Exception:
    self._boundary_edge_index_by_nodes = {}

# Avoid assert in production code — use explicit if/raise
# Custom exception classes: NOT used — prefer standard exceptions
```

### 4.6 Docstrings

Use **NumPy-style** docstrings with `Parameters`, `Returns`, `Raises` sections:

```python
def build_mesh(self, node_x, node_y, node_z, cell_nodes) -> None:
    """
    Build the unstructured mesh.

    Parameters
    ----------
    node_x, node_y, node_z : array_like, shape (N,)
        Node coordinates and bed elevations.
    cell_nodes : array_like
        Triangular connectivity (3 nodes per row) or polygon CSR.

    Raises
    ------
    ValueError
        If cell_face_offsets[-1] != len(cell_nodes).
    """
```

### 4.7 Section Separators

Use `# ── Section Name ──` with Unicode em-dash (U+2500) for visual separation:

```python
# ── Mesh ─────────────────────────────────────────────────────────────────
# ── Solver init ──────────────────────────────────────────────────────────
```

### 4.8 `__all__` Exports

Define at the **bottom** of the module:

```python
__all__ = [
    "SWE2DBackend",
    "swe2d_available",
    "swe2d_gpu_available",
]
```

### 4.9 C++ pybind11 Binding Style

```cpp
// Every binding uses py::arg(...) named arguments
m.def("swe2d_build_mesh", &swe2d_build_mesh,
    py::arg("node_x"), py::arg("node_y"), ...);

// Opaque handles with shared_ptr for Python lifecycle
py::class_<PyMesh, std::shared_ptr<PyMesh>>(m, "SWE2DMeshHandle")
    .def("__repr__", ...);

// numpy arrays with c_style + forcecast
py::array_t<double, py::array::c_style | py::array::forcecast> arr;

// Error handling: std::invalid_argument, std::runtime_error
if (arr.size() != expected)
    throw std::invalid_argument("array size mismatch");

// Default argument values in binding
py::arg("dt_request") = -1.0,
```

---

## 5. Test Suite Guide

### 5.1 Test Framework

The project uses **`unittest`** (Python standard library). Two files use pytest-style bare functions. There is **no pytest infrastructure** — `conftest.py` is essentially empty.

### 5.2 Running Tests

```bash
# From the repo root:
PYTHONPATH="$PWD:$PWD/build:$PWD/tests" python3 -m unittest discover -s tests -p "test_*.py"

# Run a single file:
PYTHONPATH="$PWD:$PWD/build:$PWD/tests" python3 -m unittest tests.test_swe2d_dambreak

# Run with verbose output (preferred):
PYTHONPATH="$PWD:$PWD/build:$PWD/tests" python3 -m unittest -v \
    tests.test_swe2d_gpu_validation_perf \
    tests.test_swe2d_gpu_unstructured

# Run GPU validation (primary gate):
PYTHONPATH="$PWD:$PWD/build:$PWD/tests" python3 -m unittest -v \
    tests.test_swe2d_gpu_validation_perf \
    tests.test_swe2d_gpu_unstructured
```

Per `AGENTS.md`, the primary validation gate is `test_swe2d_gpu_validation_perf` + `test_swe2d_gpu_unstructured`.

### 5.3 Conditional Skips

```python
# GPU-required tests
@unittest.skipUnless(swe2d_gpu_available(), "CUDA GPU not available")
class TestGPUFeature(unittest.TestCase): ...

# Native module required
@unittest.skipUnless(swe2d_available(), "hydra_swe2d not built")
class TestCPUSolver(unittest.TestCase): ...

# Gmsh required
@unittest.skipUnless(_gmsh_available(), "gmsh Python package not installed")
class TestGmshMesh(unittest.TestCase): ...

# Environment-gated benchmarks
@unittest.skipUnless(os.environ.get("BACKWATER_RUN_GPU_PERF") == "1",
    "benchmark skipped (set BACKWATER_RUN_GPU_PERF=1)")
class TestBenchmark(unittest.TestCase): ...
```

### 5.4 Test File Catalog

| File | Tests | Requires | Category |
|---|---|---|---|
| `test_swe2d_gpu_validation_perf.py` | 3 | hydra_swe2d + CUDA | **Primary GPU validation** |
| `test_swe2d_gpu_unstructured.py` | 9 | hydra_swe2d + CUDA + gmsh | **Primary GPU unstructured** |
| `test_swe2d_gpu.py` | 5 | hydra_swe2d + CUDA | Legacy parity (informational only) |
| `test_swe2d_gpu_unstructured_rain.py` | 3 | hydra_swe2d + CUDA + gmsh | GPU rain-on-grid |
| `test_swe2d_gpu_graph_higher_order.py` | 2 | hydra_swe2d + CUDA | GPU higher-order schemes |
| `test_swe2d_gpu_coupling_kernel.py` | 1 | hydra_swe2d + CUDA | GPU coupling kernel |
| `test_swe2d_gpu_bridge_coupling_kernel.py` | 1 | hydra_swe2d + CUDA | GPU bridge coupling kernel |
| `test_bridge_coupling_stability_conservation_modes.py` | 2 | None | Bridge coupling modes |
| `test_swe2d_runtime_bridge_cuda_coupling.py` | 1 | None | Bridge CUDA routing |
| `test_bridge_stacked_mesh.py` | 3 | None (pytest-style) | Bridge stacked mesh |
| `test_bridge_stacked_runtime.py` | 2 | None (pytest-style) | Bridge stacked runtime |
| `test_swe2d_dambreak.py` | 1 | hydra_swe2d | CPU dam-break (Stoker) |
| `test_swe2d_lakerest.py` | 2 | hydra_swe2d | CPU lake-at-rest |
| `test_swe2d_unstructured.py` | 11 | hydra_swe2d + gmsh | CPU unstructured mesh |
| `test_swe2d_channel_flow.py` | 5 | hydra_swe2d + CUDA | Channel flow validation |
| `test_swe2d_compound_channel.py` | 5 | hydra_swe2d + CUDA | Compound channel |
| `test_swe2d_compound_channel_gmsh_multiscale.py` | 3 | hydra_swe2d + CUDA + gmsh | Gmsh compound channel |
| `test_swe2d_malpasset.py` | 6 | hydra_swe2d + CUDA | Malpasset-scale dam-break |
| `test_swe2d_mesh.py` | 8 | hydra_swe2d | Mesh construction |
| `test_bc_validation.py` | ~22 | varies | Boundary condition validation |
| `test_swe2d_drainage_structures.py` | ~35 | varies | Drainage & structures |
| `test_swe2d_backend_tiny_mode_config.py` | 4 | None | Tiny-mode config |
| `test_swe2d_tiny_mode_dispatch.py` | 6 | hydra_swe2d + CUDA | Tiny-mode dispatch |
| `test_swe2d_nonorth_vs_orth_channel.py` | 1 | hydra_swe2d | Non-orth CPU |
| `test_swe2d_nonorth_vs_orth_channel_gpu_100cells.py` | 1 | hydra_swe2d | Non-orth GPU sweep |
| `test_swe2d_nonorth_vs_orth_channel_gpu_1000cells.py` | 1 | hydra_swe2d | Non-orth GPU sweep |
| `test_swe2d_weno5_convergence.py` | 5 | hydra_swe2d + CUDA + gmsh | WENO5 convergence |
| `test_swe2d_results_queries.py` | 2 | None | Results queries |
| `test_swe2d_velocity_layer.py` | 3 | None | Velocity layer |
| `test_swe2d_run_options_builder.py` | 1 | None | Run options |
| `test_hydrograph_bc_native.py` | 5 | varies | Hydrograph BC |
| `test_gmsh_flow_aligned_quads.py` | ~12 | None | Gmsh flow-aligned quads |
| `test_tqmesh_quad_edges.py` | ~14 | varies | TQMesh + Gmsh mesh |
| `test_hybrid_cpp_channel_transition.py` | 6 | hybridmesh | Hybrid C++ mesh |
| `test_workbench_imports.py` | ~15 | None | Workbench imports |

**Shared helper modules:**
- `tests/swe2d_nonorth_gpu_sweep_common.py` — `run_gpu_nonorth_vs_orth_sweep()` shared sweep logic
- `tests/test_swe2d_unstructured.py` — `_make_gmsh_triangle_mesh()`, `_build_mesh()`, `stoker_dam_break()` reused by many tests

### 5.5 Test Structure Pattern

```python
import unittest
import numpy as np
import sys; sys.path.insert(0, ...)  # Path setup

from swe2d.runtime.backend import SWE2DBackend, swe2d_gpu_available

class TestFeature(unittest.TestCase):
    """Group docstring."""

    NX = 100       # Class-level constants
    LY = 50.0
    T_END = 10.0

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_module()

    def test_something(self):
        # Arrange
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(...)
        backend = SWE2DBackend(use_gpu=True)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        backend.initialize(h0, ...)

        # Act
        diag = backend.step()

        # Assert
        self.assertLess(diag.max_courant, 1.0)
        self.assertTrue(np.isfinite(diag.h).all())
```

---

## 6. Test Coverage Gaps

### 6.1 Untested Source Modules (Priority: High)

| Module | Untested Functions | Impact |
|---|---|---|
| `swe2d/units.py` | `configure()`, `gravity()`, `model_to_ft()`, all getters | **CRITICAL** — affects all CRS-dependent calculations. A units bug corrupts solver physics. |
| `swe2d/mesh/mesh_runtime_logic.py` | `mesh_cell_centroids`, `areas`, `min_bed`, `solver_bed`, `inflow_adjacent_cells`, `boundary_buffer_cells`, `initial_state` | **HIGH** — used by runtime setup, coupling, and results. |
| `swe2d/boundary_and_forcing/internal_flow_logic.py` | 5 public functions | **HIGH** — internal flow forcing is a user-facing feature. |
| `swe2d/boundary_and_forcing/runtime_source_logic.py` | `internal_flow_source_cms_at_time`, `apply_external_sources` | **HIGH** — runtime source assembly. |
| `swe2d/extensions/extension_models.py` | 9 standalone hydraulic functions | **HIGH** — these compute orifice/weir/pipe flows used by structures. |

### 6.2 Untested Source Modules (Priority: Medium)

| Module | Untested Classes/Functions |
|---|---|
| `swe2d/runtime/native_binding_compat.py` | `call_solver_create_compat()` fallback paths |
| `swe2d/runtime/runtime_reporting.py` | `SWE2DRuntimeReporter.process_step()` |
| `swe2d/runtime/runtime_sources.py` | `SWE2DRuntimeSourceManager` volume accounting |
| `swe2d/runtime/runtime_setup_configurator.py` | All 3 configure methods |
| `swe2d/runtime/backend_initializer.py` | `SWE2DBackendInitializer.build_and_initialize()` |
| `swe2d/runtime/run_controller.py` | `run_preflight()` |
| `swe2d/runtime/run_data_builder.py` | `build()` |
| `swe2d/runtime/run_finalizer.py` | `finalize_and_persist()` |
| `swe2d/runtime/run_orchestrator.py` | `run()` — the top-level entry point |
| `swe2d/results/animation.py` | `ResultsAnimationController` |
| `swe2d/results/panel.py` | `SWE2DResultsPanel` (1800 LOC, complex UI) |
| `swe2d/results/db_utils.py` | 3 small functions |

### 6.3 Untested Coupling & Structure Paths

| Feature | Priority |
|---|---|
| **Culvert face-flux mode** (`culvert_face_flux_mode="face_flux"`) | High |
| **Influence-width redistribution** (`_build_redistribution_data`, `_apply_redistribution`) | High |
| **SoA packing** (`pack_pipe_network_soa`, `pack_structures_soa`, `pack_coupling_soa`) | Medium |
| **Pump structures** (`StructureType.PUMP`) | Medium |
| **Gate structures** (`StructureType.GATE`) | Medium |
| **Culvert embankment overflow** | Medium |
| **Culvert table-based solver** (`culvert_solver_mode=1`) | Medium |
| **Drainage DIFFUSION and DYNAMIC solver modes** (only EGL tested) | High |
| **Pipe end exchange** routing | Medium |
| **Stage BC (type 3)** | Medium |
| **Normal depth BC (type 6)** | Medium |
| **Open BC (type 4)** | Medium |
| **Spatial Manning** | Medium |
| **Spatial CN / rain** | Medium |
| **Bridge stacked phase3 spatial mode** | Medium |
| **Implicit coupling iterations** (`implicit_coupling_iterations > 1`) | Low |

### 6.4 Infrastructure Gaps

| Gap | Impact |
|---|---|
| **No CI test execution** | `.github/workflows/` is empty. Tests must be run manually. |
| **No coverage measurement** | No `.coveragerc`, no `pytest-cov`, no per-line coverage data. |
| **No pytest configuration** | `conftest.py` is empty. No fixtures, no markers, no plugins. |
| **No test dependency specification** | `requirements.txt` has only runtime deps. |
| **No mocking strategy** | QGIS-dependent code cannot be unit-tested. |
| **Duplicate mesh helpers** | Every test file reinvents `_make_rect_mesh`. |
| **No tox.ini** | No multi-python-version test matrix. |

### 6.5 Recommended Test Targets (Priority Order)

1. **`swe2d/units.py`** — Unit conversion correctness for SI, USC, gravity, Manning factor, model_to_ft.
2. **`swe2d/mesh/mesh_runtime_logic.py`** — Centroids, areas, min_bed, solver_bed, boundary buffers, initial_state.
3. **`swe2d/boundary_and_forcing/internal_flow_logic.py`** — Field name resolution, hydrograph lookup, forcing building.
4. **`swe2d/extensions/extension_models.py`** — Standalone hydraulic math (orifice, weir, pipe capacity, circular section).
5. **`swe2d/runtime/coupling.py` SoA packing** — `pack_pipe_network_soa`, `pack_structures_soa`, `pack_coupling_soa`.
6. **Culvert face-flux mode** — Integration test with `SWE2DCouplingController`.
7. **Drainage DIFFUSION and DYNAMIC modes** — Mode-specific network step.
8. **Pump and gate structure types** — Structure flow computation.
9. **All missing BC types** — Stage, normal depth, open BC.
10. **CI + coverage infrastructure** — Add `.github/workflows/test.yml`, `.coveragerc`.

---

## 7. Common Development Workflows

### 7.1 Adding a New Test

```python
# tests/test_swe2d_mymodule.py
import unittest
import sys; sys.path.insert(0, "..."); sys.path.insert(0, ".../build")

from swe2d.runtime.backend import swe2d_available, swe2d_gpu_available, SWE2DBackend

@unittest.skipUnless(swe2d_available(), "hydra_swe2d not built")
class TestMyFeature(unittest.TestCase):
    """Tests for MyFeature."""

    def test_expected_behavior(self):
        # Arrange
        mod = ...

        # Act

        # Assert
        self.assertLess(result, tolerance)
```

### 7.2 Adding a New Module Under `swe2d/`

1. Create the file with: imports, `logger = logging.getLogger(__name__)`, `__all__` at bottom.
2. Follow the style guide (§4): absolute imports, NumPy docstrings, `_` prefix private helpers.
3. Update parent `__init__.py` to re-export new symbols (explicitly, no wildcards).
4. Create corresponding test file in `tests/`.
5. Purge `__pycache__`:
   ```bash
   find . -type d -name __pycache__ -exec rm -rf {} +
   ```

### 7.3 Adding a New C++ pybind11 Binding

1. Add function in `cpp/src/swe2d_bindings.cpp` (or appropriate file).
2. Use `py::arg("name")` named arguments.
3. Use `py::array_t<T, py::array::c_style | py::array::forcecast>` for numpy arrays.
4. Wrap opaque handles in `py::class_<T, std::shared_ptr<T>>`.
5. Rebuild: `cd build && cmake .. && make -j$(nproc)`.
6. Test in Python via `import hydra_swe2d`.

### 7.4 Debugging GPU Tests

```bash
# Enable diagnostics logging
export BACKWATER_SWE2D_DIAG_MODE=1

# Run specific GPU test with verbose output
PYTHONPATH="$PWD:$PWD/build" python3 -m unittest -v \
    tests.test_swe2d_gpu_unstructured.TestGPUUnstructuredDamBreak.test_stability_all_schemes

# Enable tiny-mode diagnostics for solver step info
export BACKWATER_SWE2D_DIAG_MODE=1
```

### 7.5 Debugging Mesh Generation

```bash
# Use the standalone CLI mesher
python3 tools/gmsh_topology_mesher.py \
    --source /path/to/model.gpkg \
    --regions-layer swe2d_topo_regions \
    --out-prefix /tmp/debug_mesh \
    --verbosity 5 \
    --write-msh

# Check the output in Gmsh GUI or ParaView
gmsh /tmp/debug_mesh.msh
```

### 7.6 Code Style Audit

```bash
# Run the built-in style audit tool
python3 tools/python_style_audit.py swe2d/
```

This checks for missing docstrings, missing return annotations, and missing parameter annotations — no external linter needed.

### 7.7 UI Binding Sync

After editing a `.ui` file, verify all widgets have Python bindings:

```bash
python3 tools/ui_bind_sync.py forms/swe2d_<name>.ui <py_files> --missing
```

---

## 8. Studio UI API Reference

### 8.1 Overview

The Studio UI provides a **component-based API** for adding, removing, and
managing dock widgets and left-pane tabs.  All docks go through a registry
that ensures they survive the host-window teardown process automatically.

**Key files:**

| File | Role |
|------|------|
| `swe2d/workbench/studio_component.py` | `StudioComponent` dataclass + tab registry |
| `swe2d_workbench_qt.py` | `SWE2DWorkbenchStudioDialog` — the UI host; contains inlined dock extraction methods |
| `swe2d/workbench/doc_viewer.py` | `DocHubWidget` — embedded documentation viewer with cross-document search |

### 8.2 `StudioComponent` Dataclass

```python
@dataclass
class StudioComponent:
    name: str                          # Unique key (e.g. "results")
    dock: QDockWidget                  # The staging dock widget
    area: Qt.DockWidgetArea            # Left, Right, Bottom, Top
    title: str = ""                    # Title bar text (defaults to name.title())
    object_name: str = ""              # Qt objectName (defaults to "SWE2DStudio{Name}Dock")
    tab_with: Optional[str] = None     # Tabify with another component name
    collapsible: bool = True           # Allow user to close/hide
    populate: Optional[Callable] = None # Callback to fill dock with widgets
```

### 8.3 Adding a New Dock (Recommended Pattern)

Use ``_build_component()`` — a single call handles creation + population +
registration:

```python
def _populate_my_dock(self, dock: QDockWidget) -> None:
    """Fill the dock with widgets."""
    content = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(content)
    layout.addWidget(QtWidgets.QLabel("My panel content"))
    dock.setWidget(content)

# In _build_ui():
self._build_component(
    name="my_panel",
    title="My Panel",
    area=QtCore.Qt.RightDockWidgetArea,
    tab_with="inspector",          # optional: tab with CFD Inspector
    populate=self._populate_my_dock,
)
```

That's it.  No edits to the extraction pipeline.  No double-extraction bugs.
The dock automatically survives into QGIS.

### 8.4 Adding a Dock Manually (Two-Step Pattern)

If you need more control over the dock before registering:

```python
# Step 1: Create the dock
self._studio_my_dock = QtWidgets.QDockWidget("My Dock", self._studio_main_window)
self._studio_my_dock.setObjectName("SWE2DStudioMyDock")
self._studio_my_dock.setFeatures(...)
# ... populate ...
self._studio_main_window.addDockWidget(area, self._studio_my_dock)

# Step 2: Register it
self._register_component(StudioComponent(
    name="my_panel",
    dock=self._studio_my_dock,
    area=area,
    tab_with="inspector",
))
```

### 8.5 Removing a Dock at Runtime

```python
self._destroy_component("my_panel")
```

This disconnects signals, removes the dock from the staging window, and
schedules Qt cleanup.

### 8.6 Adding a Left-Pane Tab

```python
def _build_my_tab_page(self) -> QtWidgets.QWidget:
    page = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(page)
    layout.addWidget(QtWidgets.QLabel("My tab content"))
    return page

# In _compose_left_pane():
self._register_left_tab("My Tab", self._build_my_tab_page)
```

The tab is automatically added to ``self._left_tabs`` during composition.
The tab order follows registration order.

### 8.7 Migration Guide (Old → New API)

| Old Pattern | New API |
|-------------|---------|
| ``self._studio_docks["x"] = dock`` | ``self._register_component(StudioComponent(...))`` |
| Manual ``getattr`` in ``_build_studio_component_docks`` | Registry auto-iteration |
| Manual ``_mkdock`` + ``_attach_host_dock_widget`` per dock | Handled by ``_build_component()`` |
| Manual ``tabifyDockWidget`` per dock | Handled by ``tab_with`` parameter |
| ``self._left_tabs.addTab(self._wrap_left_tab_page(...), "Label")`` | ``self._register_left_tab("Label", builder)`` |

**Backward compatibility:** The old ``_studio_docks`` dict is still updated
by ``_register_component()``, so existing extraction code continues to work
until fully migrated.

### 8.8 Architecture Diagram

```
SWE2DWorkbenchStudioDialog._build_ui()
    │
    ├── self._build_component("setup",   ...)  ──→  _register_component()
    ├── self._build_component("inspector",...)  ──→  _register_component()
    ├── self._build_component("results", ...)  ──→  _register_component()
    │
    └── _build_studio_component_docks()
            │
            ├── _extract_registered_docks()     # Iterates _studio_components
            ├── _mkdock() per component         # Creates QGIS host docks
            ├── _attach_host_dock_widget()      # Adds to QGIS window
            └── _tabify_registered_docks()      # Handles tab_with links
```

### 8.9 Best Practices

1. **Always use ``_build_component()``** for new docks — it's one call with
   no risk of forgetting to populate or register.

2. **Use the ``populate`` callback pattern** — keeps dock construction in a
   focused method rather than inline in ``_build_ui()``.

3. **Set ``tab_with`` for related panels** — the results dock is tabbed with
   the inspector by default.  New analysis panels should follow the same
   pattern so QGIS doesn't fill with floating docks.

4. **Name docks consistently** — use ``SWE2DStudio{Name}Dock`` for the staging
   dock and ``SWE2DStudio{Name}HostDock`` for the extracted host dock.
   The ``StudioComponent`` dataclass auto-generates these defaults.

5. **Use ``destroy_component()`` for cleanup** — never call ``dock.close()``
   or ``dock.deleteLater()`` directly.  Always go through the registry so
   the ``_studio_docks`` fallback dict stays in sync.

6. **Left-pane tabs go through ``_register_left_tab()``** — this ensures
   the tab is properly wrapped with ``_wrap_left_tab_page()`` and registered
   with ``_register_detachable_tab_widget()``.

7. **Signal safety** — before destroying any widget, call
   ``widget.blockSignals(True)``.  The ``_destroy_component()`` method does
   this automatically.
```
