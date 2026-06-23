# Gmsh Meshing Backend — User Guide

**Document version**: 2.0  
**Applies to**: HYDRA SWE2D plugin, GPU solver path  
**Last updated**: 2026-06-14

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Conceptual Model: Topology-First Meshing](#3-conceptual-model-topology-first-meshing)
4. [GUI Workflow](#4-gui-workflow)
5. [Cell Types](#5-cell-types)
6. [Arc Modes](#6-arc-modes)
7. [Interface Controls](#7-interface-controls)
8. [Quality Controls](#8-quality-controls)
9. [Gmsh Algorithm Controls](#9-gmsh-algorithm-controls)
10. [Standalone CLI](#10-standalone-cli)
11. [Programmatic API](#11-programmatic-api)
12. [Environment Variables](#12-environment-variables)
13. [Troubleshooting](#13-troubleshooting)
14. [Full Option Reference](#14-full-option-reference)

---

## 1. Overview

The Gmsh meshing backend generates 2D unstructured computational meshes for the SWE2D solver using the open-source [Gmsh](https://gmsh.info) library (version 4.x). It is the **default and recommended backend** and supports:

- **Triangular cells** via Frontal-Delaunay or Delaunay algorithms
- **Quadrilateral cells** via Blossom recombination of Delaunay triangles
- **Cartesian (structured) cells** via Transfinite surface + Recombine
- **Flow-aligned quads** for channel regions
- **Breakline embedding** — linear features (bank lines, channel centerlines) that constrain the mesh
- **Constraint refinement zones** — polygon-driven local size fields
- **Quad-edge boundary layers** — smooth graded transitions at mesh boundaries
- **Inter-region interface conformance** — shared edges between neighboring zones guarantee matching nodes (no hanging nodes)
- **Iterative quality loop** — automatic retry with tuned parameters to meet element quality thresholds
- **Multi-threading** — parallel mesh generation

---

## 2. Prerequisites

### Required

- **`gmsh` Python package ≥ 4.12**

  ```bash
  pip install gmsh
  ```

  Or install via Conda:

  ```bash
  conda install -c conda-forge gmsh
  ```

The plugin checks availability at startup and falls back to the "Structured" backend if `gmsh` is not found.

### Optional

- **`hydra_meshing_native`** — a bundled C++ extension that accelerates polyline distance/overlap computations. If not present, pure-Python fallbacks are used.

---

## 3. Conceptual Model: Topology-First Meshing

The Gmsh backend uses a **topology-first** approach. Instead of drawing elements directly, you define a conceptual model composed of:

### 3.1 Regions (Required)

Polygons that define the mesh domain. Each region has:

| Field | Type | Description |
|---|---|---|
| `region_id` | Integer | Unique identifier |
| `target_size` | Float | Target element edge length (model units) |
| `cell_type` | String | `triangular`, `quadrilateral`, `cartesian`, `channel_generator`, or `empty` |

A region can have **interior rings** to create holes (excluded from the mesh).

Each polygon part of a multi-part feature becomes a separate region in the conceptual model.

### 3.2 Arcs (Optional)

Linear features (breaklines) embedded in the mesh to align element edges. Used for channel centerlines, bank lines, or any linear constraint.

| Field | Type | Description |
|---|---|---|
| `arc_id` | Integer | Unique identifier |
| `region_id` | Integer | Region to associate with (optional) |
| `node0`, `node1` | Integer | Optional endpoint node IDs (fallback) |
| `arc_role` | String | `centerline`, `left_bank`, `right_bank`, or `breakline` |
| `use_global_arc_ctrl` | Boolean | Use global arc mode (1) or per-arc override (0) |
| `arc_mode_override` | String | Per-arc override: `hard_embed`, `soft_size_hint`, or `disabled` |
| `arc_soft_size_override` | Float | Per-arc soft size factor override |
| `arc_soft_dist_override` | Float | Per-arc soft distance factor override |

Arc geometry is read from the feature's polyline vertices. The **node0/node1** fields are optional fallback endpoints — the polyline geometry takes precedence.

### 3.3 Constraints (Optional)

Polygon zones that apply local mesh refinement (size field override).

| Field | Type | Description |
|---|---|---|
| `constraint_id` | Integer | Unique identifier |
| `target_size` | Float | Refined element edge length inside the zone |
| `cell_type` | String | Cell type for the zone |

Constraints are implemented as Gmsh **Distance + Threshold** fields. Points are sampled along the polygon boundary and interior (using an area-adaptive grid), and the size field transitions smoothly from the refined size inside the constraint to the region's default size within ~1.5× the target size.

### 3.4 Quad Edges (Optional)

Boundary-layer-like edge controls that define graded transition zones along specific edges.

| Field | Type | Description |
|---|---|---|
| `region_id` | Integer | Target region |
| `edge_id` | Integer | Edge identifier |
| `target_size` | Float | Fine size at the edge |
| `n_layers` | Integer | Number of graded layers |
| `first_height` | Float | First layer height |
| `growth_rate` | Float | Layer growth rate |

### 3.5 Nodes (Optional)

Explicit control points. Must have `node_id`, `x`, `y`.

---

## 4. GUI Workflow

### 4.1 Create Model Database

1. Open **Plugins → HYDRA → Open Workbench**
2. **Map tab → Model files and mesh actions → Create 2D Model GeoPackage**
3. Choose save location

This creates a GeoPackage with all template layers including:
- `swe2d_topo_regions`
- `swe2d_topo_arcs`
- `swe2d_topo_nodes`
- `swe2d_topo_constraints`
- `swe2d_topo_quad_edges`

### 4.2 Topology Tab

1. Switch to the **Topology** tab
2. Click **Create Topology Template Layers** (fills empty layers with schema)
3. Edit each layer in QGIS:
   - **Regions**: Digitize domain polygon(s). Set `target_size` (e.g., 10.0 for fine, 50.0 for coarse) and `cell_type`.
   - **Arcs**: Digitize breaklines along channels or other linear features.
   - **Constraints**: Digitize refinement polygons around structures, bridges, or areas of interest.
   - **Quad Edges**: Digitize lines along boundaries where graded cell sizing is desired.
4. Select the layers in the respective combo boxes in the Topology tab
5. Set **Default target size** and **Default cell type** (fallback values for features that lack explicit values)
6. Set **Meshing backend** to "gmsh"

### 4.3 Generate Mesh

Click **Generate Mesh From Topology Layers**. The plugin:

1. Reads all features from selected layers
2. Builds a `ConceptualModel`
3. Runs the Gmsh backend with configured options
4. Repairs and optimizes the result
5. Loads the mesh into the solver

Progress is shown in the status label. Any Gmsh warnings or errors are logged.

---

## 5. Cell Types

| Type | Gmsh Method | Best For |
|---|---|---|
| `triangular` | Frontal-Delaunay (algorithm 6) | General domains, complex geometry, fast generation |
| `quadrilateral` | Blossom recombination on Delaunay triangulation | Structured domains, GPU performance (~30% faster than triangles) |
| `cartesian` | Transfinite Surface + Recombine | Simple rectangular or trapezoidal zones, very fast |
| `channel_generator` | Flow-aligned structured quads | Channel reaches, river corridors |
| `empty` | Surface excluded from mesh | Holes, voids, excluded areas |

### Triangular

Uses Gmsh's Frontal-Delaunay algorithm (code 6) by default. Falls back to Delaunay (code 5) when the quality loop is active and Frontal-Delaunay fails.

### Quadrilateral

Generates Delaunay triangles then applies Blossom recombination into quads. The recombination algorithm defaults to standard Blossom (code 1). The `gmsh_quad_algorithm` controls the underlying triangulation, and `gmsh_recombination_algorithm` controls the recombination method.

### Cartesian (Transfinite)

Assigns Transfinite curve constraints to all four sides of a region, creating a structured grid. The number of nodes on each edge is determined by the region size and `target_size`. Recombination produces pure quads. This requires that the region has exactly 4 boundary segments (or that transfinite harmonization can match shared edges with neighbors).

### Channel Generator

Similar to Cartesian but with flow-aligned longitudinal curves. Uses the region's flow direction heuristic to orient the structured mesh.

---

## 6. Arc Modes

Arcs (breaklines) can be embedded in the mesh in three modes, controlled globally or per-arc:

### Hard Embed (default)

Arcs are embedded as Gmsh **embedded curves** (via `gmsh.model.mesh.embed(1, curves, 2, surface)`). Elements are forced to align with the arc — their edges follow the breakline. No additional size field is generated.

**Use when**: The arc must appear exactly in the mesh (channel banks, property lines, wall alignments).

### Soft Size Hint

Arcs are converted into **Distance + Threshold** size fields. The region's `target_size` is scaled by `arc_soft_size_factor` (default 0.5) near the arc, creating a smooth graded transition over `arc_soft_dist_factor` × refined size (default 2.0×). No forced edge alignment.

**Use when**: You want elements to be smaller near a feature but not strictly aligned to it (e.g., a rough thalweg line).

### Disabled

The arc is ignored — not embedded and not used for sizing.

### Per-Arc Override

Each arc feature can set `use_global_arc_ctrl = 0` and provide its own `arc_mode_override`, `arc_soft_size_override`, and `arc_soft_dist_override` to override the global settings.

---

## 7. Interface Controls

Interface controls manage shared boundaries between adjacent regions with different target sizes.

| Control | Default | Description |
|---|---|---|
| **Interface transition grading** | Enabled | Apply size-based grading at region interfaces |
| **Transition distance factor** | 2.5 | Distance over which size transitions (× coarser size) |
| **Transition minimum ratio** | 1.25 | Skip grading if size ratio is below this threshold |
| **Transverse interface conformance** | Disabled | Post-process to snap transverse interface nodes |
| **Centroid merge** | Disabled | Merge centroids of matched transverse interface nodes |
| **Interface snap tolerance** | 1.0 | Tolerance for snapping interface nodes |
| **Reject near-unshared** | Enabled | Reject mixed interfaces with near-coincident unshared nodes |
| **Reject tolerance** | 1.0e-3 | Tolerance for near-coincident node rejection |

These controls ensure that elements on shared edges of neighboring regions match node-for-node (conforming interfaces), preventing hanging nodes.

### Transfinite Harmonization

For `cartesian` and `channel_generator` regions with shared edges, the backend automatically harmonizes transfinite curve node counts so that adjacent structured zones have matching segmentation. This is critical for conforming quad boundaries.

---

## 8. Quality Controls

The iterative quality loop automatically retries mesh generation with tuned parameters when element quality thresholds are not met.

### Enable / Disable

Disabled by default. Enable via the **Enable iterative quality loop** checkbox or the `BACKWATER_GMSH_QUALITY_ENABLE` environment variable.

### Quality Thresholds

| Threshold | Default | Description |
|---|---|---|
| Minimum angle | 18.0° | Elements below this angle are failed |
| Maximum aspect ratio | 12.0 | Elements above this ratio are failed |
| Maximum non-orthogonality | 82.0° | Elements above this non-orthogonality are failed |
| Minimum area (relative to bbox) | 1.0e-11 | Elements below this area are failed |

### Retry Ladder

On failure, the loop tries combinations of:

| Parameter | Default ladder | Effect |
|---|---|---|
| Size scale | 1.0, 0.9, 0.8, 0.7 | Reduces target element size |
| Smoothing increments | 0, 3, 6 | Extra smoothing passes per retry |
| Recombine topology passes | 5, 12, 20 | Recombination topology optimization passes |
| Recombine minimum quality | 0.01, 0.03, 0.06 | Minimum quality for recombination |
| Random factors | 1e-9, 1e-7, 1e-6 | Random perturbation for diversification |
| Algorithm switch | Disabled | Alternate between Frontal-Delaunay and Delaunay |
| Optimize methods | Laplace2D, Relocate2D | Gmsh optimize methods applied per attempt |

### Controls

| Control | Default | Description |
|---|---|---|
| Max iterations | 2 | Maximum retry attempts |
| Time limit | 55.0 s | Total time budget for all attempts |
| Strict mode | Disabled | Accept only the first passing candidate |
| Recombine node repositioning | Enabled | Allow Gmsh to reposition nodes during recombination optimization |

If no passing candidate is found within the budget, the backend returns the **best available candidate** (the one with the highest composite score) with a warning. If all attempts fail to produce any mesh, a **best-effort baseline** fallback is attempted with default settings.

### Checkpoints

When `gmsh_quality_checkpoint_path` is set (not exposed in UI by default), each attempt's mesh is written as a `.npz` checkpoint file, allowing partial recovery.

---

## 9. Gmsh Algorithm Controls

### Triangle Algorithm

| Code | Name | Description |
|---|---|---|
| 6 (default) | Frontal-Delaunay | High-quality triangles; best for unstructured domains |
| 5 | Delaunay | Fast, good for large meshes; used as quality-loop fallback |
| 1 | MeshAdapt | Adaptive triangulation |
| 7 | BAMG | Bidimensional anisotropic mesh generator (requires MMG) |

### Quad Algorithm

Same codes as triangle algorithm (the triangulation that precedes Blossom recombination). Default is also 6 (Frontal-Delaunay).

### Recombination Algorithm

| Code | Name | Description |
|---|---|---|
| 1 (default) | Standard Blossom | High-quality quad recombination |
| 0 | Simple | Basic recombination (faster, lower quality) |
| 2 | Blossom with recombine unaligned | More aggressive recombination |

### Smoothing

Number of smoothing passes (default: 0). Higher values improve element shape but increase runtime.

### Optimization

- **Optimize iterations**: Number of passes (default: 0)
- **Netgen optimize**: Enable Netgen optimization (default: off) — adds computational cost but can improve quality for problematic meshes

### Threading

- **Number of threads**: Default 1 (Gmsh internal threads)
- **Max 2D threads**: Default 0 (auto-detect)

Set higher for large meshes on multi-core systems.

### Size Controls

- **Global min cell size**: Minimum element edge length (default: 0.0 = no lower bound)
- **Tolerance edge length**: Edges shorter than this are ignored (default: 0.0)
- **Mesh size from points**: Use point-based size fields (default: off)

### Global Recombine

When enabled, applies Blossom recombination to all regions regardless of per-region `cell_type`. Useful for converting a triangular mesh to quads globally.

### Flow-Aligned Quads

When **Full-region flow-aligned quads** is enabled, all quadrilateral/channel_generator regions are automatically modified to use flow-aligned structured meshing. If this fails on the first attempt, the backend automatically retries with per-region flow-align disabled.

---

## 10. Standalone CLI

The script `tools/gmsh_topology_mesher.py` provides a command-line interface for headless mesh generation from GeoPackage topology layers.

### Basic Usage

```bash
python tools/gmsh_topology_mesher.py \
    --source /path/to/model.gpkg \
    --regions-layer swe2d_topo_regions \
    --out-prefix /tmp/topo_mesh
```

### Full Options

```bash
python tools/gmsh_topology_mesher.py \
    --source /path/to/model.gpkg \
    --nodes-layer swe2d_topo_nodes \
    --arcs-layer swe2d_topo_arcs \
    --regions-layer swe2d_topo_regions \
    --constraints-layer swe2d_topo_constraints \
    --quad-edges-layer swe2d_topo_quad_edges \
    --default-size 20.0 \
    --default-cell-type triangular \
    --out-prefix /tmp/topo_mesh \
    --write-msh \
    --tolerance-edge-length 0.0 \
    --num-threads 1 \
    --verbosity 2
```

### Outputs

The tool writes three files:

| Extension | Format | Content |
|---|---|---|
| `.npz` | NumPy compressed | `node_x`, `node_y`, `node_z`, `cell_nodes`, `cell_face_offsets`, `cell_face_nodes`, `cell_type`, `region_id`, `target_size` |
| `.json` | JSON | Quality metrics, cell/node counts, timing |
| `.msh` (optional) | Gmsh v2.2 ASCII | Mesh in Gmsh format for external tools |

---

## 11. Programmatic API

### Minimal Example

```python
from swe2d.mesh.meshing import (
    ConceptualNode, ConceptualArc, ConceptualRegion,
    ConceptualModel, generate_face_centric_mesh,
)

# Build a simple rectangular domain
region = ConceptualRegion(
    region_id=0,
    ring_xy=[(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)],
    default_size=5.0,
    default_cell_type="triangular",
)

model = ConceptualModel(
    nodes=[], arcs=[], regions=[region], constraints=[], quad_edges=[],
)

# Generate mesh
mesh = generate_face_centric_mesh(model, backend="gmsh")

print(f"Nodes: {len(mesh.node_x)}")
print(f"Cells: {len(mesh.cell_face_offsets) - 1}")
```

### With Options

```python
mesh = generate_face_centric_mesh(
    model,
    backend="gmsh",
    options={
        "gmsh_tri_algorithm": 6,
        "gmsh_smoothing": 3,
        "gmsh_optimize_iters": 2,
        "gmsh_verbosity": 1,
        "gmsh_quality_enable": True,
        "gmsh_min_angle_deg": 20.0,
        "gmsh_max_aspect_ratio": 10.0,
        "gmsh_quality_max_iterations": 5,
        "gmsh_num_threads": 4,
    },
)
```

### Building from QGIS Layers

```python
from swe2d.mesh.meshing import conceptual_from_qgis_layers

# qgis_layer references from the QGIS project
model = conceptual_from_qgis_layers(
    nodes_layer=topo_nodes_layer,
    arcs_layer=topo_arcs_layer,
    regions_layer=topo_regions_layer,
    constraints_layer=topo_constraints_layer,
    quad_edges_layer=topo_quad_edges_layer,
    default_size=20.0,
    default_cell_type="triangular",
)

mesh = generate_face_centric_mesh(model, backend="gmsh")
```

### MeshResult Structure

```python
@dataclass
class MeshResult:
    node_x: np.ndarray          # Node X coordinates
    node_y: np.ndarray          # Node Y coordinates
    node_z: np.ndarray          # Node bed elevation (zeros until terrain assignment)
    cell_nodes: np.ndarray      # Triangulated connectivity (3 nodes per row)
    cell_face_offsets: np.ndarray  # Per-cell face ring start offsets (CSR format)
    cell_face_nodes: np.ndarray    # Face ring node indices (CSR format)
    cell_type: np.ndarray       # Cell type code per face
    region_id: np.ndarray       # Source region ID per cell
    target_size: np.ndarray     # Target size per cell
    quality_summary: dict       # Optional quality diagnostic dict
```

---

## 12. Environment Variables

All quality-related defaults can be overridden via environment variables prefixed with `BACKWATER_GMSH_`:

| Variable | Default | Description |
|---|---|---|
| `BACKWATER_GMSH_QUALITY_ENABLE` | `0` (False) | Enable iterative quality loop |
| `BACKWATER_GMSH_QUALITY_STRICT` | `0` (False) | Strict mode (first passing candidate only) |
| `BACKWATER_GMSH_MIN_ANGLE_DEG` | `18.0` | Minimum element angle threshold |
| `BACKWATER_GMSH_MAX_ASPECT_RATIO` | `12.0` | Maximum element aspect ratio |
| `BACKWATER_GMSH_MIN_AREA_REL_BBOX` | `1.0e-11` | Minimum element area relative to bounding box |
| `BACKWATER_GMSH_MAX_NON_ORTH_DEG` | `82.0` | Maximum non-orthogonality |
| `BACKWATER_GMSH_QUALITY_MAX_ITERATIONS` | `2` | Maximum retry attempts |
| `BACKWATER_GMSH_QUALITY_TIME_LIMIT_S` | `55.0` | Time budget for retry loop (seconds) |
| `BACKWATER_GMSH_QUALITY_SIZE_SCALES` | `1.0,0.9,0.8,0.7` | Retry size scale ladder (comma-separated) |
| `BACKWATER_GMSH_QUALITY_RECOMBINE_TOPOLOGY_PASSES` | `5,12,20` | Topology retry ladder |
| `BACKWATER_GMSH_QUALITY_RECOMBINE_MIN_QUALITY` | `0.01,0.03,0.06` | Min quality retry ladder |
| `BACKWATER_GMSH_QUALITY_RANDOM_FACTORS` | `1e-9,1e-7,1e-6` | Random perturbation retry ladder |
| `BACKWATER_GMSH_QUALITY_OPTIMIZE_METHODS` | `Laplace2D,Relocate2D` | Optimize methods (comma-separated) |
| `BACKWATER_GMSH_ALGO_SWITCH_ON_FAILURE` | `0` (False) | Switch algorithms on retry |
| `BACKWATER_GMSH_RECOMBINE_NODE_REPOSITIONING` | `1` (True) | Allow node repositioning during recombination |
| `BACKWATER_GMSH_NUM_THREADS` | (not set) | Gmsh thread count |
| `BACKWATER_GMSH_QUAD_FULL_REGION_FLOW_ALIGN` | `0` (False) | Enable full-region flow-aligned quads |

Example:

```bash
export BACKWATER_GMSH_QUALITY_ENABLE=1
export BACKWATER_GMSH_MIN_ANGLE_DEG=25.0
export BACKWATER_GMSH_MAX_ASPECT_RATIO=8.0
export BACKWATER_GMSH_QUALITY_MAX_ITERATIONS=10
export BACKWATER_GMSH_QUALITY_TIME_LIMIT_S=120.0
```

---

## 13. Troubleshooting

### Mesh Generation Fails

| Symptom | Likely Cause | Solution |
|---|---|---|
| "Gmsh Python package is not installed" | Missing `gmsh` dependency | `pip install gmsh` |
| "No valid regions in topology layer" | Regions layer empty or missing geometry | Check region polygons are valid and not null |
| Single-pass build fails | Complex geometry, overlapping arcs, degenerate boundaries | Enable the **quality loop** for automatic retry with tuned parameters |
| Flow-aligned quads fail | Channel geometry incompatible with flow-align | The backend automatically retries with flow-align disabled; check the logs |
| Quality loop produces no valid candidate | Very thin or complex geometry; stringent thresholds | Relax thresholds (reduce `min_angle_deg`, increase `max_aspect_ratio`); increase `max_iterations` |
| Hanging nodes at region interfaces | Interface controls not enabled | Enable **interface transition grading** and **transverse interface conformance** |

### Mesh Quality Issues

| Symptom | Likely Cause | Solution |
|---|---|---|
| Very small (degenerate) elements | Arcs or constraints with conflicting size fields | Disable arcs or increase `tolerance_edge_length` |
| Elements too large | `target_size` too large | Reduce per-region `target_size` |
| Elements too small | Constraints producing overly refined zones | Increase constraint `target_size` or check `arc_soft_size_factor` |
| Bad quads (warped or inverted) | Recombination struggles with complex geometry | Lower `gmsh_recombination_algorithm` to 0 (simple); increase smoothing |
| Asymmetric mesh near arcs | Arc soft size hint is too diffuse | Reduce `arc_soft_dist_factor` from 2.0 to 1.0 |

### Performance

| Issue | Solution |
|---|---|
| Mesh generation too slow | Increase `gmsh_num_threads`; use Delaunay (code 5) instead of Frontal-Delaunay (code 6) |
| Quality loop takes too long | Reduce `gmsh_quality_max_iterations`; lower `gmsh_quality_time_limit_s` |
| Very large mesh memory issues | Increase per-region `target_size`; add coarse zones away from areas of interest |

---

## 14. Full Option Reference

All options that can be passed via the `options` dict to `generate_face_centric_mesh` or set via the GUI.

### General

| Option Key | UI Widget | Type | Default |
|---|---|---|---|
| `gmsh_tri_algorithm` | Gmsh triangle algorithm combo | int | 6 (Frontal-Delaunay) |
| `gmsh_quad_algorithm` | Gmsh quad algorithm combo | int | 6 (Frontal-Delaunay) |
| `gmsh_recombination_algorithm` | Recombine algorithm combo | int | 1 (Blossom) |
| `gmsh_smoothing` | Smoothing passes spin | int | 0 |
| `gmsh_optimize_iters` | Optimize iterations spin | int | 0 |
| `gmsh_optimize_netgen` | Netgen optimize checkbox | bool | False |
| `gmsh_verbosity` | Gmsh verbosity spin | int | 2 |
| `gmsh_num_threads` | Number of threads spin | int | 1 |
| `gmsh_max_num_threads_2d` | Max 2D threads spin | int | 0 (auto) |
| `gmsh_mesh_size_min` | Global min cell size spin | float | 0.0 |
| `gmsh_tolerance_edge_length` | Tolerance edge length spin | float | 0.0 |
| `gmsh_mesh_size_from_points` | Mesh size from points checkbox | bool | False |
| `gmsh_global_recombine` | Global recombine checkbox | bool | False |
| `gmsh_quad_full_region_flow_align` | Full-region flow align checkbox | bool | False |

### Arc Controls

| Option Key | UI Widget | Type | Default |
|---|---|---|---|
| `gmsh_arc_mode` | Arc mode combo | str | `hard_embed` |
| `gmsh_arc_soft_size_factor` | Arc soft size factor spin | float | 0.5 |
| `gmsh_arc_soft_dist_factor` | Arc soft dist factor spin | float | 2.0 |

### Interface Controls

| Option Key | UI Widget | Type | Default |
|---|---|---|---|
| `gmsh_interface_transition_enable` | Transition grading checkbox | bool | True |
| `gmsh_interface_transition_dist_factor` | Transition dist factor spin | float | 2.5 |
| `gmsh_interface_transition_min_ratio` | Transition min ratio spin | float | 1.25 |
| `gmsh_interface_conformance` | Transverse conformance checkbox | bool | False |
| `gmsh_transverse_interface_centroid_merge` | Centroid merge checkbox | bool | False |
| `gmsh_interface_snap_tol` | Interface snap tolerance spin | float | 1.0 |
| `gmsh_interface_reject_near_unshared` | Reject near-unshared checkbox | bool | True |
| `gmsh_interface_reject_tol` | Reject tolerance spin | float | 1.0e-3 |

### Quality Loop

| Option Key | UI Widget | Type | Default |
|---|---|---|---|
| `gmsh_quality_enable` | Enable quality loop checkbox | bool | False |
| `gmsh_quality_strict` | Strict mode checkbox | bool | False |
| `gmsh_quality_max_iterations` | Max iterations spin | int | 2 |
| `gmsh_quality_time_limit_s` | Time limit spin | float | 55.0 |
| `gmsh_min_angle_deg` | Min angle spin | float | 18.0 |
| `gmsh_max_aspect_ratio` | Max aspect ratio spin | float | 12.0 |
| `gmsh_max_non_orth_deg` | Max non-orthogonality spin | float | 82.0 |
| `gmsh_min_area_rel_bbox` | Min area rel bbox spin | float | 1.0e-11 |
| `gmsh_quality_size_scales` | (advanced) comma-separated | tuple | 1.0,0.9,0.8,0.7 |
| `gmsh_quality_smooth_increments` | (advanced) comma-separated | tuple | 0,3,6 |
| `gmsh_quality_recombine_topology_passes` | (advanced) comma-separated | tuple | 5,12,20 |
| `gmsh_quality_recombine_minimum_quality` | (advanced) comma-separated | tuple | 0.01,0.03,0.06 |
| `gmsh_quality_random_factors` | (advanced) comma-separated | tuple | 1e-9,1e-7,1e-6 |
| `gmsh_quality_optimize_methods` | (advanced) comma-separated | tuple | Laplace2D,Relocate2D |
| `gmsh_algorithm_switch_on_failure` | (advanced) checkbox | bool | False |
| `gmsh_quality_recombine_node_repositioning` | (advanced) checkbox | bool | True |

### Internal / Diagnostics

| Option Key | Type | Default | Description |
|---|---|---|---|
| `gmsh_quality_checkpoint_path` | str | `""` | File path for per-attempt `.npz` checkpoints |
| `gmsh_progress_path` | str | `""` | File path for JSON progress emission |
| `gmsh_progress_emit_interval_s` | float | 0.75 | Minimum interval between progress updates |
