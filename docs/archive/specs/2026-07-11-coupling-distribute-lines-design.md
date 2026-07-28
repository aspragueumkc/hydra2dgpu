---
type: spec
status: complete
created: 2026-07-11
completed: 2026-07-25
---

# Coupling Distribute Lines — Design Spec

**Date:** 2026-07-11
**Status:** Approved
**Approach:** Option C — Unified coupling line mechanism for culverts, pipe-ends, and outfalls

## 1. Problem

Currently, pipe-end and culvert coupling injects flow as a single-cell mass source/sink at one cell center. This has two limitations:

1. **No momentum coupling** — flow appears/disappears at a point with no directional momentum transfer
2. **No spatial distribution** — all flow goes through one cell regardless of pipe/outfall geometry

The existing structure coupling has optional face-flux mode and redistribution for culverts, but pipe-end coupling has neither. This spec unifies both under a single "coupling distribute line" mechanism.

## 2. Overview

A new `swe2d_coupling_distribute_lines` GeoPackage layer allows users to draw a LineString that:
- Defines **which cells** receive coupled inflow/outflow (redistribution corridor)
- Defines the **momentum direction** at the face (face normal from line geometry)
- Links to an existing **pipe-end node, outfall node, or culvert structure**

Without a coupling distribute line, existing single-cell mass-source behavior is preserved.

## 3. Data Model

### New GeoPackage Layer: `swe2d_coupling_distribute_lines`

| Field | Type | Nullable | Default | Description |
|---|---|---|---|---|
| `coupling_id` | TEXT(64) | N | — | Unique identifier |
| `linked_type` | TEXT(16) | N | — | `"pipe_end"`, `"outfall"`, or `"culvert"` |
| `linked_id` | TEXT(64) | N | — | ID of the linked node or structure |
| `enabled` | INTEGER | N | 1 | 0/1 |
| `influence_width` | DOUBLE | Y | NULL | Corridor width for redistribution (model units) |
| `face_width_override` | DOUBLE | Y | NULL | Override face width (default = pipe diameter or culvert span) |
| `depth_safety_factor` | DOUBLE | Y | 0.5 | Depth limiter safety factor |

Geometry: **LineString**. Start vertex = upstream end, end vertex = downstream end (relative to flow direction).

### Face-Flux Normal Convention

```
flow_dir = normalize(x1 - x0, y1 - y0)   # unit vector along line
face_normal = (flow_dir.y, -flow_dir.x)    # 90° clockwise rotation
```

For a horizontal line drawn left-to-right:
- `flow_dir = (1, 0)`, `face_normal = (0, -1)` — momentum in -y direction

### New Python Dataclass

```python
@dataclass
class CouplingDistributeLine:
    coupling_id: str
    linked_type: str           # "pipe_end", "outfall", "culvert"
    linked_id: str
    line_x0: float
    line_y0: float
    line_x1: float
    line_y1: float
    influence_width: float = 0.0
    face_width_override: Optional[float] = None
    depth_safety_factor: float = 0.5
    enabled: bool = True
```

## 4. CUDA Changes

### 4.1 Unified Face-Flux Kernel

Generalize `swe2d_culvert_face_flux_kernel` (`swe2d_gpu.cu:3144`) to accept flow from two sources:

| Source Type | Flow Array | Entry |
|---|---|---|
| 0 (structure) | `structure_flow[si]` (HDS-5 output) | Existing culvert entries |
| 1 (pipe node) | `node_net_q[n]` (mass balance) | New pipe-end/outfall entries |

New parameter: `source_type` int32 array (0=structure, 1=pipe node). The kernel dispatches per entry.

Face normal, width, donor/receiver, and depth limiter logic are identical regardless of source type.

### 4.2 Workspace Rename

`CulvertFaceFluxWorkspace` → `CouplingFaceFluxWorkspace` in `swe2d_gpu.cuh`. Contains entries for both culvert structures and pipe-end/outfall nodes that have coupling lines.

### 4.3 Exchange Path Changes

**With coupling line:**
- Skip `swe2d_drainage_pipe_end_exchange_kernel` (single-cell) for pipe-end/outfall nodes
- Skip single-cell source kernel for culvert structures
- Unified face-flux kernel handles flow via `node_net_q` or `structure_flow`

**Without coupling line:**
- Existing single-cell mass-source behavior unchanged

### 4.4 Redistribution

Unified `swe2d_redistribute_face_flux_kernel` operates on `d_ext_struct_flux_h` regardless of whether the entry came from a culvert or pipe-end. Corridor computed from coupling line geometry + `influence_width`.

## 5. Execution Flow

```
Per timestep:
  1. swe2d_gpu_apply_pipe_end_bc()              — unchanged
  2. swe2d_pipe1d_step()                         — unchanged
  3. Zero d_ext_struct_flux_h / hu / hv
  4. swe2d_coupling_face_flux_kernel()          — unified: culverts + pipe-ends with lines
  5. swe2d_gpu_redistribute_face_flux()           — redistribute if influence_width > 0
  6. For entities WITHOUT coupling line:
     a. swe2d_drainage_pipe_end_exchange_kernel  — single-cell (fallback)
     b. swe2d_coupling_structure_source_kernel   — single-cell (existing fallback)
  7. swe2d_update_kernel reads d_ext_struct_flux_*
```

## 6. Python Changes

### 6.1 Runtime Coupling (`swe2d/runtime/coupling.py`)

- Accept `coupling_distribute_lines` list in constructor
- `_build_coupling_distribute_line_data()`: read lines, resolve linked nodes/structures, compute face normals and corridor cells (reuses `_build_redistribution_data` pattern from line 738)
- `_ensure_coupling_face_flux_preloaded()`: upload unified face-flux params (merged from existing `_ensure_culvert_face_flux_preloaded`, renamed)
- `apply_native_device_sources()`: skip single-cell exchange for entities with coupling lines

### 6.2 Extension Models (`swe2d/extensions/extension_models.py`)

- Add `CouplingDistributeLine` dataclass (Section 3)
- Add `coupling_distribute_lines: list` to `PipeNetworkConfig` (~line 285)

### 6.3 JSON Config Builders

- `swe2d/extensions/drainage_network.py`: parse `coupling_distribute_lines` in `build_drainage_config_from_json()` (~line 76)
- `swe2d/extensions/structures.py`: accept coupling distribute lines in structure config builder (~line 27)

## 7. GUI Changes

### 7.1 Model Tab (`swe2d/workbench/views/model_tab_view.py`)

- Add `coupling_distribute_lines_layer_combo` QComboBox in Layer Setup group (~line 1170, after `structures_layer_combo`)
- Add `get_coupling_distribute_lines_layer()` protocol method (~line 1436)

### 7.2 Map Tab (`swe2d/workbench/views/map_tab_view.py`)

- Add corresponding `coupling_distribute_lines_layer_combo` (~lines 161-192)

### 7.3 View Protocols (`swe2d/workbench/views/view_protocols.py`)

- Add `get_coupling_distribute_lines_layer()` to `ModelTabViewProtocol` and `MapTabViewProtocol`

### 7.4 Layer Controller (`swe2d/workbench/controllers/layer_controller.py`)

- Add `v.populate_layer_combo("coupling_distribute_lines_layer_combo", layers, "line")` (~line 37)

### 7.5 Studio Dialog (`swe2d/workbench/studio_dialog.py`)

- Add `"coupling_distribute"` to feature keywords (~line 1577)
- Wire new combo in `_build_pipe_network_config()` (~line 1903)

### 7.6 Batch Dialog (`swe2d/workbench/dialogs/batch_simulation_dialog.py`)

- Add widget-to-CLI mapping (~line 59)

### 7.7 Run Context (`swe2d/workbench/workers/run_context.py`)

- Add `coupling_distribute_lines_layer` field (~line 75)

## 8. Geopackage Schema

### 8.1 Schema Definitions (`swe2d/workbench/services/schema_definitions.py`)

- Add `swe2d_coupling_distribute_lines` to `LAYER_SCHEMAS` dict (~after line 330)
- Add display name to `_LAYER_DISPLAY_NAMES`

### 8.2 Topology Template (`swe2d/workbench/services/topology_template_service.py`)

- Add to `_TOPOLOGY_TEMPLATE_KEYS` (~line 30)

### 8.3 Mesh Controller (`swe2d/workbench/controllers/mesh_controller.py`)

- Add layer classification (~line 678)

### 8.4 Schema Doc (`docs/MODEL_GEOPACKAGE_SCHEMA.md`)

- Document new layer as #19

## 9. CLI Changes

### 9.1 Headless Runner (`swe2d/cli/headless_runner.py`)

- Read coupling distribute lines from GPKG (~line 429)
- Pass to coupling controller (~line 446)

### 9.2 GPKG Adapter (`swe2d/cli/gpkg_adapter.py`)

- Add `read_coupling_distribute_lines_from_gpkg()` function (~line 525)
- Read layer features, resolve linked nodes/structures, return inline dict

### 9.3 Pipe Network Service (`swe2d/workbench/services/pipe_network_service.py`)

- Add coupling distribute line reading from GPKG layer in `build_pipe_network_config()` (~line 64)

## 10. Not In Scope

- Internal flow sources/polygons (separate feature, already implemented)
- Free outfall BC (implemented in previous session)
- Changing the pipe-end BC kernel or pipe1d solver
- HEC-22 inlet grate coupling (separate feature)
