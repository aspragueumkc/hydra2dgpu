# HYDRA — GPU-Accelerated 2D Shallow Water Equation Plugin for QGIS

**Version**: 2.0 (GPU-Only)
**Last Updated**: 2026-06-22

<style>
:root { --fig-bg: #1e1e2e; --fig-fg: #cdd6f4; --fig-border: #45475a; --fig-input: #313244; --fig-label: #a6adc8; --fig-btn: #89b4fa; --fig-heading: #f5c2e7; }
.fig-frame { display:inline-block; min-width:480px; max-width:560px; background:var(--fig-bg); border:1px solid var(--fig-border); border-radius:8px; font-family:'Segoe UI',system-ui,sans-serif; color:var(--fig-fg); overflow:hidden; box-shadow:0 4px 16px rgba(0,0,0,0.3); margin:8px 0; }
.fig-header { background:#181825; padding:8px 14px; font-size:13px; font-weight:600; color:var(--fig-heading); border-bottom:1px solid var(--fig-border); }
.fig-body { padding:8px 14px; }
.fig-row { display:flex; align-items:center; gap:8px; margin-bottom:4px; }
.fig-row label { flex:0 0 150px; font-size:12px; color:var(--fig-label); text-align:right; }
.fig-row input { flex:1; background:var(--fig-input); color:var(--fig-fg); border:1px solid var(--fig-border); border-radius:4px; padding:4px 8px; font-size:12px; }
.fig-sep { border:none; border-top:1px solid #585b70; margin:6px 0; }
.fig-btn-row { display:flex; gap:4px; margin-top:4px; flex-wrap:wrap; }
.fig-btn { background:var(--fig-btn); color:#1e1e2e; border:none; border-radius:4px; padding:5px 10px; font-size:11px; font-weight:600; cursor:pointer; text-align:center; }
.fig-btn.secondary { background:#45475a; color:var(--fig-fg); }
.fig-btn.danger { background:#f38ba8; }
.fig-section-label { font-size:11px; font-weight:600; color:var(--fig-heading); margin:4px 0; }
.fig-progress { background:#313244; border-radius:3px; height:6px; margin:6px 0; overflow:hidden; }
.fig-progress-fill { background:#89b4fa; width:0%; height:100%; border-radius:3px; }
</style>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Installation & Requirements](#2-installation--requirements)
3. [The Studio Interface](#3-the-studio-interface)
4. [Layers Tab](#4-layers-tab)
5. [Mesh Tab](#5-mesh-tab)
6. [Parameters Tab](#6-parameters-tab)
7. [Running the Solver](#7-running-the-solver)
8. [Results & Postprocessing](#8-results--postprocessing)
9. [Troubleshooting](#9-troubleshooting)
10. [References](#10-references)

---

## 1. Overview

HYDRA is a QGIS-integrated plugin for 2D shallow water equation (SWE) modeling, powered by a CUDA-accelerated finite-volume solver. It couples:

- **2D surface hydrodynamics** — unstructured-mesh FVM with Godunov-type fluxes
- **1D urban drainage networks** — SWMM-style pipe network coupling (EGL, Diffusion, Dynamic wave)
- **Hydraulic structures** — weirs, culverts (FHWA HDS-5), gates, bridges, pumps
- **Rainfall & infiltration** — rain-on-grid with SCS Curve Number method

### Primary Use Cases

| Application | Description |
|---|---|
| **Flood inundation** | Dam breaks, urban flooding, overbank flow |
| **Storm drain surcharge** | Surface–network interaction, manhole flooding |
| **Drainage design** | Real-time what-if with GPU performance |
| **Rainfall event simulation** | Extreme event runoff and infiltration |
| **Culvert/weir analysis** | HDS-5 culvert rating, weir discharge, structure sizing |

---

## 2. Installation & Requirements

### System Requirements

| Component | Requirement |
|---|---|
| **QGIS** | 3.28+ (Linux primary; Windows/macOS secondary) |
| **Python** | 3.12+ (within QGIS environment) |
| **CUDA Toolkit** | 11.x or 12.x |
| **NVIDIA GPU** | Compute Capability ≥ 7.5 (RTX 3060+; A100/H100 recommended) |
| **VRAM** | 4 GB minimum; 8+ GB for 100k+ cell meshes |
| **C++ Compiler** | GCC 10+ or Clang 12+ (C++17) |
| **CMake** | 3.16+ |

### Python Dependencies

| Package | Required | Purpose |
|---|---|---|
| `numpy` | ✅ | Array operations, mesh data |
| `scipy` | ❌ | Optional 1D solver backend |
| `matplotlib` | ❌ | In-plugin plotting (timeseries, profiles) |
| `shapely` | ❌ | Geometry operations for BC polyline sampling |

### C++ Dependencies (bundled)

| Component | Purpose |
|---|---|
| **pybind11** (2.13.6+) | Python ↔ C++ bindings (auto-fetched by CMake) |
| **GMsh 4.x** | Optional mesh generation backend |
| **TQMesh** | Optional quadrilateral mesh generation |

### Build the Native Module

```bash
# Clone the repository
git clone https://github.com/aspragueumkc/hydra2dgpu.git
cd hydra2dgpu

# Create build directory
mkdir build && cd build

# Configure with CUDA (requires CUDA toolkit on PATH)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)
```

> **Mixed precision (experimental):** Add `-DSWE2D_STATE_FP32=ON` to the cmake command to store solver state arrays as `float` instead of `double`. This reduces GPU memory traffic by ~35% with a small accuracy trade-off in very shallow flows. Only recommended for GPU-bound simulations on memory-constrained cards. The precompiled binaries use full `double` precision.

The build produces:
- `hydra_swe2d.cpython-312-x86_64-linux-gnu.so` — GPU solver module
- `hydra_native.so` — 1D backwater solver module
- `hydra_meshing_native.so` — Mesh generation kernels
- `hydra_overlay.so` — High-performance rendering overlay

### Install as QGIS Plugin

```bash
# From QGIS Plugin Manager:
#   1. Open QGIS → Plugins → Manage and Install Plugins
#   2. Click "Install from ZIP"
#   3. Select the plugin archive or point to the repository root

# Or symlink into QGIS plugin directory:
ln -s /path/to/hydra2dgpu \
  ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/hydra2dgpu
```

### Verify Installation

```python
from swe2d.runtime.backend import swe2d_gpu_available
print(f"GPU available: {swe2d_gpu_available()}")
```

---

## 3. The Studio Interface

The workbench opens as a **dock-integrated Studio** inside the QGIS main window. The left dock contains three tabs that follow the simulation workflow:

<div class="fig-frame"><div class="fig-header">Left Dock: HYDRA2D Model Setup</div><div class="fig-body">
<div style="display:flex;gap:4px;margin-bottom:8px;">
<div class="fig-btn" style="flex:1;font-size:11px;padding:4px">Layers</div>
<div class="fig-btn secondary" style="flex:1;font-size:11px;padding:4px">Mesh</div>
<div class="fig-btn secondary" style="flex:1;font-size:11px;padding:4px">Parameters</div>
</div>
<div style="background:#313244;border-radius:4px;padding:20px;text-align:center;font-size:12px;color:var(--fig-label);">Active tab content (varies by tab)</div>
</div></div>

| Tab | Workflow Step | Section |
|-----|---------------|---------|
| **Layers** | Load input data (nodes, cells, terrain, Manning, BC layers) | [§4](#4-layers-tab) |
| **Mesh** | Define topology regions, generate mesh with Gmsh | [§5](#5-mesh-tab) |
| **Parameters** | Configure solver, rain, stability, structures, and run | [§6](#6-parameters-tab) |

**Additional docks** (right and bottom):

| Panel | Location | Purpose |
|-------|----------|---------|
| **HYDRA2D View** | Right dock | Mesh display, depth/velocity/result visualization |
| **HYDRA2D CFD Inspector** | Right dock (tabbed) | Model Settings, Mesh Settings inspector trees + Help |
| **HYDRA2D Results** | Right dock (tabbed) | Map overlay controls, output storage toggles |
| **HYDRA2D Temporal** | Bottom dock | Animation bar for timestep scrubbing |
| **HYDRA2D Log** | Bottom dock | Live log output with [ERROR]/[WARNING] color coding |

---


## 4. Layers Tab

The **Layers** tab is the first step in the workflow. It contains three pages in a QToolBox:

1. **Load Layers** — select input data layers
2. **Mesh Setup** — GeoPackage management, terrain assignment, BC configuration
3. **Utilities** — GeoPackage explorer, log viewer, status display

### 4.1 Load Layers Page

The Load Layers page provides a grid of QGIS layer selectors. Select the layers that define your model geometry, terrain, roughness, and drainage configuration.

<div class="fig-frame"><div class="fig-header">Load Layers</div><div class="fig-body">
<div class="fig-row"><label>Nodes layer:</label><input placeholder="swe2d_mesh_nodes..." value="swe2d_mesh_nodes"></div>
<div class="fig-row"><label>Cells layer:</label><input placeholder="swe2d_mesh_cells..." value="swe2d_mesh_cells"></div>
<div class="fig-row"><label>Terrain raster:</label><input placeholder="dem_10m"></div>
<div class="fig-row"><label>Manning polygons:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>CN polygons:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>Rain gages:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>Hyetographs:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>Sample lines:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>Drainage nodes:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>Drainage links:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>Drainage inlets:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>Drainage node-inlets:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>Structures:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>BC lines:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>Layer group:</label><input placeholder="(no group)"></div>
<div class="fig-sep"></div>
<div class="fig-btn-row"><div class="fig-btn">Autopopulate From Group</div><div class="fig-btn secondary">Refresh Layers</div></div>
<div class="fig-btn-row"><div class="fig-btn">Create 2D Model GeoPackage</div></div>
</div></div>

#### Widget Reference

| Widget | Purpose | Valid Values | When to Use |
|--------|---------|-------------|-------------|
| **Nodes layer** | QGIS point layer containing mesh node coordinates. Field `node_id` must be present. | Any point layer | Always — required for mesh construction |
| **Cells layer** | QGIS polygon/multipolygon layer defining mesh cell geometry. Each cell has a `cell_id` referencing `node_id`. | Any polygon layer | Always — required for mesh construction |
| **Terrain raster** | Digital elevation model (DEM) raster used to assign node bed elevations via **Assign Node Z From Terrain** in the Mesh Setup page. | DEM raster layer | Always — assign elevations after mesh is built |
| **Manning polygons** | Polygon layer with Manning's n values for spatially varying roughness. Leave empty for uniform n (set in Parameters tab). | Polygon layer with numeric roughness field | Spatially varying bed roughness |
| **CN polygons** | Polygon layer containing SCS Curve Number values for runoff computation. | Polygon layer with CN field | When infiltration method is SCS Curve Number |
| **Rain gages (points)** | Point layer defining rain gauge locations. Each gauge needs an ID matching entries in the hyetograph table. | Point layer with gauge IDs | Rainfall-on-grid simulations |
| **Rain hyetographs (table)** | Table layer containing precipitation hyetographs. Columns: time (hours) and rainfall intensity. | Table layer with time/intensity columns | Spatial rainfall with Thiessen interpolation |
| **Sample lines layer** | Line layer for sampling flow results along cross-sections during simulation. | Line layer | When cross-section output is needed |
| **Drainage nodes layer** | Point layer for drainage network nodes (manholes, junctions). | Point layer | Coupled 1D-2D drainage simulations |
| **Drainage links layer** | Line layer for drainage network links (pipes, channels). | Line layer | Coupled 1D-2D drainage simulations |
| **Drainage inlet types (table)** | Table layer defining inlet types (grate, curb, combination) and their hydraulic capture curves. | Table layer | Inlet-specific hydraulics |
| **Drainage node-inlets (table)** | Table layer mapping drain nodes to inlet types. | Table layer | Advanced inlet configuration |
| **Hydraulic structures layer** | Line layer for structures (weirs, culverts, gates, bridges, pumps). | Line layer with structure type field | Structure modeling |
| **BC lines layer** | Line layer for boundary condition segments. Each segment defines BC type (inflow, stage, normal depth, etc.). | Line layer with BC type/value fields | Non-uniform BC assignment |
| **Layer group** | QGIS layer group containing all input layers for this model. | Existing layer group | Batch auto-population of combos |

**Action buttons**:

| Button | Purpose |
|--------|---------|
| **Autopopulate From Group** | Walk the selected layer group and auto-fill all layer combos by matching layer names against known keywords. |
| **Refresh Layers** | Refresh all layer combos to reflect current QGIS project layers. Use after adding or renaming layers. |
| **Create 2D Model GeoPackage** | Create a new GeoPackage to store model geometry, boundary conditions, and results. Must be done once before running a model. |

### 4.2 Mesh Setup Page

The Mesh Setup page manages mesh I/O, terrain assignment, and boundary condition configuration.

<div class="fig-frame"><div class="fig-header">Mesh Setup</div><div class="fig-body">
<div class="fig-btn-row"><div class="fig-btn" style="flex:1;min-width:220px">Load 2D Model GeoPackage</div></div>
<div class="fig-btn-row"><div class="fig-btn secondary" style="flex:1">Export Mesh To Map Layers</div><div class="fig-btn secondary" style="flex:1">Load Mesh From Layers</div></div>
<div class="fig-btn-row"><div class="fig-btn secondary" style="flex:1">Assign Node Z From Terrain</div><div class="fig-btn secondary" style="flex:1">Pull Node Z From Layer</div></div>
<div class="fig-sep"></div>
<div style="font-size:11px;font-weight:600;color:var(--fig-label);margin-bottom:4px;">Boundary Conditions</div>
<div class="fig-row"><label>Default BC type:</label><input value="Normal Depth ▾"></div>
<div class="fig-row" style="margin-left:168px;gap:16px;">
<label style="flex:none;width:auto;font-size:11px;"><input type="checkbox" checked> Inflow progressive</label>
<label style="flex:none;width:auto;font-size:11px;"><input type="checkbox"> Uniform inflow velocity</label>
</div>
</div></div>

#### Widget Reference

| Widget | Purpose | Valid Values | When to Use |
|--------|---------|-------------|-------------|
| **Load 2D Model GeoPackage** | Load an existing model GeoPackage. All layer combos and BC settings are populated from package metadata. | — | Opening an existing project |
| **Export Mesh To Map Layers** | Export the in-memory mesh (nodes + cells) as QGIS map layers for inspection. | — | After mesh generation or loading |
| **Load Mesh From Selected Layers** | Build an in-memory mesh from currently selected nodes and cells layers. | — | After editing layer geometry or node elevations externally |
| **Assign Mesh Node Z From Terrain** | Sample the selected terrain raster at in-memory mesh nodes and update node_z. | — | After mesh is built and terrain raster is loaded |
| **Pull Mesh Node Z From Nodes Layer** | Legacy workflow: read bed_z from the selected nodes layer into in-memory mesh node_z. | — | When using a pre-existing nodes layer with bed_z |
| **Default BC type** | Default boundary condition type for all BC line segments. Per-segment overrides via BC layer attributes. | Wall (1), Inflow Q (2), Stage (3), Normal Depth (6/7), Timeseries Flow/Stage (102/103), Open (4), Reflecting (5) | Always — set before running |
| **Inflow progressive** | Ramp inflow gradually at simulation start to avoid numerical shock. | Checkbox | Inflow boundaries with sudden discharge onset |
| **Uniform inflow velocity** | Apply uniform velocity profile across inflow boundary cells. Unchecked for parabolic (shear) distribution. | Checkbox | When inflow velocity profile matters |

### 4.3 Utilities Page

<div class="fig-frame"><div class="fig-header">Utilities</div><div class="fig-body">
<div class="fig-btn-row"><div class="fig-btn secondary" style="flex:1;min-width:200px">Open Model GeoPackage Explorer</div></div>
<div class="fig-btn-row"><div class="fig-btn secondary" style="flex:1;min-width:200px">Open Run Log Viewer</div></div>
<div class="fig-sep"></div>
<div style="font-size:11px;color:var(--fig-label);font-style:italic;padding:4px 0;">No layer-linked mesh yet</div>
</div></div>

#### Widget Reference

| Widget | Purpose |
|--------|---------|
| **Open Model GeoPackage Explorer** | Browse model GeoPackage tables and open matching viewers; rename/delete model result tables. |
| **Open Run Log Viewer** | View, search, and export the current model run log. Shows solver output, timestep diagnostics, and error messages. |
| **Layer status label** | Displays current model status (e.g. "No layer-linked mesh yet", or mesh statistics after loading). |

---

## 5. Mesh Tab

The **Mesh** tab is the second step in the workflow. It provides topology-based mesh generation using Gmsh (recommended) or a built-in structured fallback.

Two pages in the QToolBox:

1. **Layer Setup** — select topology layers
2. **Controls** — meshing backend, generation, validation

<div class="fig-frame"><div class="fig-header">Layer Setup</div><div class="fig-body">
<div class="fig-row"><label>Topology nodes layer:</label><input placeholder="Select layer..."></div>
<div class="fig-row"><label>Topology arcs layer:</label><input placeholder="Select layer..."></div>
<div class="fig-row"><label>Topology regions layer:</label><input placeholder="Select layer..."></div>
<div class="fig-row"><label>Constraints layer:</label><input placeholder="Select layer..."></div>
<div class="fig-row"><label>Quad edges / transition:</label><input placeholder="Select layer..."></div>
<div class="fig-btn-row"><div class="fig-btn" style="flex:1">Create Topology Template Layers</div></div>
</div></div>

<div class="fig-frame"><div class="fig-header">Controls</div><div class="fig-body">
<div class="fig-row"><label>Meshing backend:</label><input value="Gmsh (recommended) ▾"></div>
<div class="fig-row"><label>Default target size:</label><input value="20.0"></div>
<div class="fig-row"><label>Default cell type:</label><input value="triangular ▾"></div>
<div class="fig-sep"></div>
<div class="fig-section-label">Gmsh Controls</div>
<div class="fig-row"><label>Triangle algorithm:</label><input value="Frontal-Delaunay ▾"></div>
<div class="fig-row"><label>Quad algorithm:</label><input value="Frontal+Blossom ▾"></div>
<div class="fig-row"><label>Recombine algorithm:</label><input value="Blossom ▾"></div>
<div class="fig-row"><label>Num threads:</label><input value="1"></div>
<div class="fig-row"><label>Smoothing passes:</label><input value="0"></div>
<div class="fig-sep"></div>
<div class="fig-btn-row"><div class="fig-btn">Validate &amp; Summarize</div></div>
<div class="fig-btn-row"><div class="fig-btn secondary">Edit Region Attributes</div></div>
<div class="fig-btn-row"><div class="fig-btn secondary">Edit Quad/Transition Edges</div></div>
<div class="fig-btn-row"><div class="fig-btn">Generate Mesh</div></div>
<div class="fig-btn-row"><div class="fig-btn danger">Terminate</div></div>
</div></div>

### Mesh Generation Workflow

1. **Create topology layers** — click **Create Topology Template Layers** in the Layer Setup page to create the `swe2d_topo_*` template layers in your GeoPackage.

2. **Define regions** — edit the `topo_regions` layer to add region polygons:
   - One polygon per mesh block
   - Set `target_size` (element edge length) per region
   - Set `cell_type`: `triangular`, `quadrilateral`, `cartesian`, or `empty` (hole)
   - Interior rings create hole cutouts
   
3. **Add arcs and constraints** (optional):
   - **Arcs** for boundary-aligned meshing
   - **Constraints** for local refinement (polygon = size field)
   - **Quad Edges** for Gmsh transition spacing

4. **Configure meshing backend** — Gmsh (recommended) or Structured fallback

5. **Validate & Generate** — click **Validate & Summarize** to check your topology, then **Generate Mesh**

For a comprehensive guide to Gmsh meshing, see [GMSH_MESHING_GUIDE.md](GMSH_MESHING_GUIDE.md).

### Topology Layer Reference

| Widget | Purpose | Valid Values | When to Use |
|--------|---------|-------------|-------------|
| **Topology nodes layer** | Point layer from topology template containing node coordinates. | Topology nodes layer | Always |
| **Topology arcs layer** | Line layer for boundary-aligned mesh edges. Controls node spacing. | Topology arcs layer | Aligned meshing near breaklines |
| **Topology regions layer** | Polygon layer defining mesh blocks. Each region has `target_size`, `cell_type`. | Topology regions layer | Always — primary mesh control |
| **Constraints layer** | Polygon layer for local refinement size fields. | Topology constraints layer | Local mesh refinement |
| **Quad edges / transition** | Line layer for quad transition spacing at region interfaces. | Quad edges layer | Structured quad-channel transitions |

### Meshing Backend Widgets

| Widget | Purpose | Valid Values | Default |
|--------|---------|-------------|---------|
| **Meshing backend** | Select mesh generation engine. | `Gmsh (recommended)`, `Structured (built-in fallback)` | Gmsh |
| **Default target size** | Default element edge length for regions without explicit `target_size`. | 0.01–1e6 | 20.0 |
| **Default cell type** | Default cell shape. | `triangular`, `quadrilateral`, `cartesian`, `empty` | triangular |

### Gmsh Algorithm Reference (Controls > Gmsh section)

| Widget | Purpose | Valid Values | Default |
|--------|---------|-------------|---------|
| **Triangle algorithm** | 2D meshing algorithm for triangular cells. | `Frontal-Delaunay (6)` (quality), `Delaunay (5)` (faster) | 6 (Frontal-Delaunay) |
| **Quadrilateral algorithm** | 2D meshing algorithm for quadrilateral cells. | `Frontal+Blossom (6)`, `Delaunay+Blossom (5)`, `Packing of Parallelograms (9)` | 6 |
| **Recombine algorithm** | Triangle-to-quad recombination method. | `Simple (0)`, `Blossom (1)`, `Full-quad (2)` | 1 (Blossom) |
| **Apply global recombine** | Run `gmsh.model.mesh.recombine()` globally after mesh generation. | Checkbox | Off |
| **Flow-aligned quads** | Apply transfinite surfaces for edge-aligned quad spacing across full region. | Checkbox | On |
| **Smoothing passes** | Number of mesh smoothing passes. | 0–100 | 0 |
| **Optimize iterations** | Number of mesh optimization iterations. | 0–100 | 0 |
| **Num threads** | `General.NumThreads` for Gmsh. 0 = auto. | 0–256 | 1 |
| **Max 2D threads** | `Mesh.MaxNumThreads2D` cap. 0 = auto. | 0–256 | 0 |
| **Arc mode** | How topology arcs influence the mesh. | `hard_embed` (strict), `soft_size_hint`, `disabled` | hard_embed |
| **Arc soft size factor** | Target-size factor near arcs in soft mode. Lower = finer. | 0.05–1.0 | 0.5 |
| **Arc soft distance factor** | Arc-influence distance multiplier in soft mode. | 0.1–10.0 | 2.0 |
| **Interface transition grading** | Apply Distance/Threshold grading near shared interfaces. | Checkbox | On |
| **Interface grading distance** | Distance multiplier for interface influence width. | 0.25–20.0 | 2.5 |
| **Interface grading min ratio** | Only apply grading when adjacent target sizes differ by this ratio. | 1.0–10.0 | 1.25 |
| **Global min cell size** | Minimum allowed element edge length. | 0–1e6 | 0.0 |
| **Ignore edges shorter than** | Tolerance for ignoring short edges during meshing. | 0–1e6 | 0.0 |
| **Use region target_size** | Use region polygon `target_size` for mesh sizing. | Checkbox | On |
| **Enable Gmsh quality loop** | Iterative quality improvement loop. | Checkbox | Off |
| **Quality max iterations** | Maximum quality loop iterations. | 1–50 | 2 |
| **Quality time limit** | Quality loop time budget in seconds. | 1–3600 | 55.0 |

---

## 6. Parameters Tab

The **Parameters** tab is the third step in the workflow. It contains five pages in a QToolBox:

1. **Solver Parameters** — core solver configuration
2. **Rain / Hydrology** — rainfall, infiltration, source terms
3. **Stability Controls** — wet/dry front handling, capping, damping
4. **Structures & Drainage** — 1D-2D coupling and hydraulic structures
5. **Run / Output** — simulation execution and results output

### 6.1 Solver Parameters Page

<div class="fig-frame"><div class="fig-header">Solver Parameters</div><div class="fig-body">
<div class="fig-row"><label>Manning n:</label><input value="0.020"></div>
<div class="fig-row"><label>CFL:</label><input value="0.45"></div>
<div class="fig-row"><label>h_min:</label><input value="0.000001"></div>
<div class="fig-row"><label>Initial condition:</label><input value="Dry start ▾"></div>
<div class="fig-row"><label>Initial depth:</label><input value="0.0"></div>
<div class="fig-row"><label>Initial WSE:</label><input value="0.0"></div>
<div class="fig-row"><label>Variable timestep:</label><input type="checkbox" checked style="flex:none;width:auto;"></div>
<div class="fig-row"><label>dt:</label><input value="5.0"></div>
<div class="fig-row"><label>Initial dt:</label><input value="0.0"></div>
<div class="fig-row"><label>GPU diag sync:</label><input value="0"></div>
<div class="fig-row"><label>Tiny mode:</label><input value="Off (0) ▾"></div>
<div class="fig-row"><label>Wet cell threshold:</label><input value="0.003"></div>
<div class="fig-row"><label>CUDA graph replay:</label><input type="checkbox" style="flex:none;width:auto;"></div>
<div class="fig-row"><label>SWE2D perf mode:</label><input type="checkbox" style="flex:none;width:auto;"></div>
<div class="fig-row"><label>Run time:</label><input value="01:00" style="width:80px;flex:none;"></div>
<div class="fig-sep"></div>
<div class="fig-row"><label>Reconstruction:</label><input value="Van Leer (4) ▾"></div>
<div class="fig-row"><label>Temporal order:</label><input value="RK4 (4) ▾"></div>
<div class="fig-row"><label>Internal flow layer:</label><input value="(none) ▾"></div>
</div></div>

#### Widget Reference

| Widget | Purpose | Valid Values | Default | When to Use |
|--------|---------|-------------|---------|-------------|
| **Manning n** | Manning's roughness coefficient. | 0.0–1.0 | 0.020 | Always — controls bed friction |
| **CFL** | Courant-Friedrichs-Lewy number for explicit timestep control. Lower = more stable but smaller timesteps. | 0.01–0.99 | 0.45 | Always — stability vs. performance trade-off |
| **h_min** | Minimum water depth threshold. Cells below this are treated as dry. | 1e-9–1.0 | 1e-6 | Always — wet/dry threshold |
| **Initial condition** | Starting condition for the entire domain. | `Dry start`, `Uniform depth`, `Uniform WSE` | Dry start | Always |
| **Initial depth** | Constant initial depth when using `Uniform depth`. | 0–1e6 | 0.0 | Uniform depth start |
| **Initial WSE** | Constant water surface elevation when using `Uniform WSE`. Depth = WSE - bed. | -1e6–1e6 | 0.0 | Uniform WSE start |
| **Variable timestep** | When checked, dt is computed adaptively from CFL condition. | Checkbox | Off | Adaptive timestepping |
| **dt (fixed or dt_max)** | Fixed timestep (variable off) or dt_max upper bound (variable on). | 1e-4–1e6 | 0.05 | Always |
| **Initial dt (0 = auto)** | First-step timestep before adaptive CFL adjusts. 0 = automatic. | 0–1e6 | 0.0 | First step control |
| **GPU diag sync (steps)** | Number of solver steps between GPU diagnostics sync. Higher = less overhead. | 1–1,000,000 | 10 | Performance tuning |
| **Tiny mode** | Handling strategy for wet/dry cells near h_min. | `Off (0)`, `Auto (1)`, `Fused (2)`, `Persistent (3)` | Persistent (3) | Stability near wet/dry fronts |
| **Tiny active/wet threshold** | Max wet cells before tiny-mode optimization engages. | 1–10,000,000 | 2000 | Small-scale simulations |
| **CUDA graph replay** | Enable CUDA graph capture/replay for kernel launches. Reduces overhead. | Checkbox | Off | Stable kernel topology with small dt |

> **CUDA Graphs — Quick Guide:** Graph replay captures a sequence of GPU
> kernels and replays them as a single unit, reducing CPU launch overhead.
> Gives 10–20% speedup for small, fixed timesteps. **Disable** when using:
> - RK4 or higher temporal schemes (graph-incompatible staging)
> - Adaptive timestepping (kernel arguments change each step)
> - Structures coupling with face-flux mode (kernel topology changes)
> If you see "CUDA graph replay failed" in the log, disable graphs — the
> solver automatically falls back to non-graph execution.
| **SWE2D perf mode** | High-performance mode with aggressive optimizations (kernel fusion, reduced sync). | Checkbox | Off | Maximum GPU throughput |
| **Internal flow layer** | Polygon layer defining internal source/sink flow regions. | Layer combo | (none) | Internal source/sink flows |
| **Internal flow field** | Field name in the internal flow layer containing discharge values. Positive = source, negative = sink. | Text | q_cms | Internal flow configuration |
| **Run duration** | Total simulation duration. | Text (decimal hours or HH:MM) | 1:00 | Always |
| **Reconstruction** | Spatial scheme for cell-face value extrapolation. | `First-order (0)`, `MUSCL Fast (1)`, `MUSCL MinMod (2)`, `MUSCL MC (3)`, `MUSCL Van Leer (4)`, `WENO3-like (5)`, `WENO5 (6)` | First-order (0) | Accuracy vs. speed trade-off |
| **Temporal discretization** | Time integration method (ODE solver). | `Euler RK1 (1)`, `RK2 Heun (2)`, `RK4 (4)`, `Graph-safe RK4 (5)`, `Graph-safe RK5 (6)` | RK2 (2) | Accuracy vs. stability trade-off |

### 6.2 Rain / Hydrology Page

<div class="fig-frame"><div class="fig-header">Rain / Hydrology</div><div class="fig-body">
<div class="fig-row"><label>Max rel depth increase:</label><input value="2.0"></div>
<div class="fig-row"><label>Max source dh/step:</label><input value="0.0"></div>
<div class="fig-row"><label>Max source rate:</label><input value="0.0"></div>
<div class="fig-row"><label>Extreme rain mode:</label><input type="checkbox" style="flex:none;width:auto;"></div>
<div class="fig-row"><label>Source CFL beta:</label><input value="0.25"></div>
<div class="fig-row"><label>Source max substeps:</label><input value="16"></div>
<div class="fig-row"><label>True source subcycling:</label><input type="checkbox" style="flex:none;width:auto;"></div>
<div class="fig-row"><label>IMEX split:</label><input type="checkbox" style="flex:none;width:auto;"></div>
<div class="fig-row"><label>Stage-coupled IMEX RK2:</label><input type="checkbox" style="flex:none;width:auto;"></div>
<div class="fig-sep"></div>
<div class="fig-row"><label>Rain rate (mm/hr):</label><input value="0.0"></div>
<div class="fig-row"><label>CN default:</label><input value="75"></div>
<div class="fig-row"><label>Ia ratio:</label><input value="0.05"></div>
<div class="fig-row"><label>Use spatial rain/CN:</label><input type="checkbox" style="flex:none;width:auto;"></div>
<div class="fig-row"><label>Infiltration method:</label><input value="SCS CN ▾"></div>
<div class="fig-row"><label>Storm area layer:</label><input placeholder="(none)"></div>
<div class="fig-row"><label>Rain buffer rings:</label><input value="0"></div>
</div></div>

#### Widget Reference

| Widget | Purpose | Valid Values | Default | When to Use |
|--------|---------|-------------|---------|-------------|
| **Max rel depth increase** | Maximum relative water depth increase per timestep from source terms. 0 = unlimited. | 0–1000 | 2.0 | Rainfall/source simulations |
| **Max source dh/step** | Maximum absolute depth change per step from sources. 0 = unlimited. | 0–10 | 0.0 | Intense rainfall stability |
| **Max source rate** | Maximum source rate (rainfall intensity) cap. Values above this are clamped. 0 = no cap. | 0–100 | 0.0 | Extreme events |
| **Extreme rain mode** | Enhanced source stabilization for high-intensity storms. | Checkbox | Off | Extreme rainfall |
| **Source CFL beta** | CFL factor for source term sub-stepping. Lower = smaller substeps = more stability. | 0.01–2.0 | 0.25 | Source stability tuning |
| **Source max substeps** | Maximum substeps for source term integration. | 1–512 | 16 | Source sub-cycling |
| **True source subcycling** | Integrate sources with multiple substeps per hydrodynamic step. | Checkbox | Off | Stiff source terms |
| **IMEX source split** | Split source terms into implicit (stiff) and explicit (non-stiff) components. | Checkbox | Off | Mixed source stiffness |
| **Stage-coupled IMEX-RK2** | Tie source evaluation to intermediate RK stages. | Checkbox | Off | Tighter source coupling |
| **Rain rate** | Uniform rainfall rate applied to the entire domain. | 0–2000 mm/hr | 0.0 | Uniform rainfall events |
| **Rain update interval (s)** | Re-evaluate SCS-CN rate every N seconds. 0 = per-step evaluation. | 0–3600 | 60 | Performance tuning for long simulations |
| **Default CN** | Default SCS Curve Number. Overridden by CN polygon layer. | 1–100 | 75.0 | SCS infiltration |
| **SCS Ia/S ratio** | Initial abstraction ratio. Standard SCS value = 0.20. Lower = more runoff. | 0–1.0 | 0.2 | SCS infiltration |
| **Spatial rainfall** | Use Thiessen polygon interpolation when rain gage + hyetograph layers are configured. | Checkbox | On | Spatially variable rainfall |
| **Infiltration method** | Infiltration model for rainfall-runoff. | `SCS Curve Number`, `None` | SCS Curve Number | Rainfall simulations |
| **Storm area layer** | Optional polygon layer defining storm extent. Only cells within receive rain. | Layer combo | (none) | Sub-domain rainfall |
| **Rain boundary buffer rings** | Buffer rings where rainfall is applied outside the storm area boundary. Prevents dry artifacts. | 0–10 | 1 | Storm area edge smoothing |

### 6.3 Stability Controls Page

<div class="fig-frame"><div class="fig-header">Stability Controls</div><div class="fig-body">
<div class="fig-row"><label>Shallow damping depth:</label><input value="0.0001"></div>
<div class="fig-row"><label>Shallow-front fallback:</label><input type="checkbox" checked style="flex:none;width:auto;"></div>
<div class="fig-row"><label>Front flux damping:</label><input value="0.5"></div>
<div class="fig-row"><label>Active-set hysteresis:</label><input type="checkbox" checked style="flex:none;width:auto;"></div>
<div class="fig-row"><label>Depth cap:</label><input value="1e6"></div>
<div class="fig-row"><label>Momentum cap min speed:</label><input value="50.0"></div>
<div class="fig-row"><label>Momentum cap celerity mult:</label><input value="20.0"></div>
<div class="fig-row"><label>Max inv area:</label><input value="1e6"></div>
<div class="fig-row"><label>CFL lambda cap:</label><input value="1e6"></div>
</div></div>

#### Widget Reference

| Widget | Purpose | Valid Values | Default | When to Use |
|--------|---------|-------------|---------|-------------|
| **Shallow damping depth** | Depth threshold below which velocity damping is applied to stabilize wetting/drying fronts. | 1e-8–10 | 1e-4 | Wet/dry front stability |
| **Shallow-front recon fallback** | Fall back to first-order reconstruction at shallow wet/dry fronts to prevent overshoot. | Checkbox | On | Recommended: always enabled |
| **Front flux damping** | Damping factor applied to fluxes at wet/dry fronts. Higher = more damping. | 0–1 | 0.5 | Front oscillations |
| **Active-set hysteresis** | Prevent wet/dry cells from flipping every timestep. Improves front stability. | Checkbox | On | Recommended: always enabled |
| **Depth cap** | Maximum allowable water depth. Depths exceeding this are clamped. | 0.001–1e7 | 1e6 | Unphysical depth spikes |
| **Momentum cap min speed** | Minimum flow speed below which momentum capping is inactive. | 0.1–1e4 | 50.0 | Preventing low-velocity capping |
| **Momentum cap celerity mult** | Multiplier on wave celerity to determine the momentum cap. | 0.1–1000 | 20.0 | Momentum limiting |
| **Max inv area** | Maximum cell area for cell inversion risk detection. | 1–1e12 | 1e6 | Large cells with steep gradients |
| **CFL lambda cap** | Maximum eigenvalue (wave speed) in CFL calculation. Prevents tiny timesteps from high wave speeds. | 1–1e12 | 1e6 | Anomalously high wave speeds |

### 6.4 Structures & Drainage Page

<div class="fig-frame"><div class="fig-header">Structures & Drainage</div><div class="fig-body">
<div class="fig-row"><label>Coupling loop:</label><input value="cuda ▾"></div>
<div class="fig-row"><label>Culvert solver mode:</label><input value="Secant (0) ▾"></div>
<div class="fig-row"><label>Culvert face flux:</label><input type="checkbox" style="flex:none;width:auto;"></div>
<div class="fig-row"><label>Bridge coupling mode:</label><input value="phase3_spatial ▾"></div>
<div class="fig-sep"></div>
<div class="fig-row"><label>Drainage solver mode:</label><input value="EGL (0) ▾"></div>
<div class="fig-row"><label>Drainage GPU method:</label><input value="step ▾"></div>
<div class="fig-row"><label>Coupling substeps:</label><input value="1"></div>
<div class="fig-row"><label>Max coupling substeps:</label><input value="10"></div>
<div class="fig-row"><label>Head deadband:</label><input value="0.001"></div>
<div class="fig-row"><label>Dynamic relaxation:</label><input value="0.5"></div>
<div class="fig-row"><label>Adaptive depth fraction:</label><input value="0.1"></div>
<div class="fig-row"><label>Adaptive wave Courant:</label><input value="0.5"></div>
<div class="fig-row"><label>Implicit iters:</label><input value="50"></div>
<div class="fig-row"><label>Implicit relax:</label><input value="0.5"></div>
<div class="fig-row"><label>Use redistribution:</label><input type="checkbox" style="flex:none;width:auto;"></div>
</div></div>

#### Widget Reference

| Widget | Purpose | Valid Values | Default | When to Use |
|--------|---------|-------------|---------|-------------|
| **Coupling loop** | Backend for drainage/structure-2D interaction. | `CUDA coupling loop (GPU)` | CUDA | Structures or drainage enabled |
| **Culvert solver mode** | Culvert hydraulics method. | `Direct (Newton/secant) (0)`, `Precomputed lookup (1)` | 0 | Culvert structures |
| **Culvert coupling mode** | Face-based flux coupling distributes culvert discharge across the 2D cell face. | Checkbox | Off | Better spatial resolution for culverts |
| **Enable redistribution override** | Read per-structure redistribution parameters from GeoPackage. | Checkbox | On | Per-structure redistribution control |
| **Bridge stacked coupling mode** | Spatial redistribution method for bridge stacked coupling. | `Phase 3 spatial`, `Legacy scalar weighting` | Phase 3 | Bridge structures |

> **⚠ Bridge Stacked Coupling — Not Production Ready.** The bridge stacked
> mesh feature is experimental and does not produce correct results. Do not
> use bridge structures in production simulations. Use standard culvert or
> weir structures instead.
| **Drainage equation set** | Governing equations for 1D drainage network flow. | `EGL (0)`, `Diffusion wave (1)`, `Dynamic Saint-Venant (2)` | EGL | Drainage coupling |
| **Drainage GPU method** | GPU execution strategy for drainage coupling. | `Per-step (step)`, `Native iterative (iterative)` | step | Drainage coupling |
| **Drainage substeps** | Number of drainage substeps per SWE2D timestep. | 1–256 | 1 | Stiff drainage systems |
| **Drainage max adaptive substeps** | Maximum adaptive substeps for drainage coupling. | 1–1024 | 64 | Adaptive drainage |
| **Drainage head deadband** | Head difference below which no drainage flow is computed. Prevents oscillation near zero flow. | 0–10 | 0.001 | Zero-flow oscillations |
| **Drainage dynamic relaxation** | Relaxation factor for drainage coupling iteration. Lower = more stability. | 0–1 | 1.0 | Stiff coupling |
| **Drainage adaptive depth fraction** | Fraction of cell water depth allowed to be drained per step. | 0.001–1.0 | 0.2 | Adaptive drainage |
| **Drainage adaptive wave Courant** | Courant target for adaptive drainage timestep control. | 0.001–10 | 0.5 | Adaptive drainage |
| **Drainage implicit iterations (GPU)** | Implicit solver iterations for GPU drainage. | 1–8 | 2 | GPU drainage convergence |
| **Drainage implicit relaxation (GPU)** | Relaxation factor for implicit drainage on GPU. | 0.1–1.0 | 0.5 | GPU drainage stability |

### 6.5 Run / Output Page

<div class="fig-frame"><div class="fig-header">Run / Output</div><div class="fig-body">
<div class="fig-btn-row"><div class="fig-btn" style="flex:1">Run 2D Model</div><div class="fig-btn danger" style="flex:1">Cancel</div></div>
<div class="fig-progress"><div class="fig-progress-fill" style="width:0%"></div></div>
<div class="fig-sep"></div>
<div class="fig-section-label">Output</div>
<div class="fig-row"><label>Output interval:</label><input value="00:30"></div>
<div class="fig-row"><label>Line output interval:</label><input value="00:05"></div>
<div class="fig-sep"></div>
<div class="fig-section-label">Debugging</div>
<div class="fig-btn-row">
<div class="fig-btn secondary" style="flex:1">Preview Overrides</div>
<div class="fig-btn secondary" style="flex:1">Preview Coupling</div>
<div class="fig-btn secondary" style="flex:1">Take Snapshot</div>
</div>
<div class="fig-sep"></div>
<div class="fig-section-label">Results Output</div>
<div class="fig-row"><label>Table prefix:</label><input placeholder="optional"></div>
<div class="fig-row"><label>GPKG path:</label><input placeholder="GeoPackage path (optional)"></div>
<div class="fig-btn-row">
<div class="fig-btn secondary" style="flex:0 0 auto;min-width:80px">Browse...</div>
<div class="fig-btn secondary" style="flex:0 0 auto;min-width:140px">Load Inputs From Results...</div>
</div>
</div></div>

#### Widget Reference

| Widget | Purpose | Valid Values | Default | When to Use |
|--------|---------|-------------|---------|-------------|
| **Run 2D Model** | Start the 2D shallow water simulation with current settings. | Button | — | Always — starts the solver |
| **Cancel** | Request cancellation at the next safe checkpoint. | Button | — | During an active run |
| **Progress bar** | Simulation progress indicator. Shows percentage and timestep info. | 0–100% | 0 | During execution |
| **Output interval** | Time between 2D mesh result writes to GeoPackage. Smaller = more data. | Text (decimal hr or HH:MM) | 00:30 | Always — controls result granularity |
| **Line output interval** | Time between sample line (cross-section) result writes. | Text (decimal hr or HH:MM) | 00:05 | Sample line output |
| **Preview Overrides** | Display summary of current parameter overrides. | Button | — | Pre-flight check |
| **Preview Drainage/Structure Coupling** | Preview 1D-2D coupling configuration. | Button | — | Drainage/structure runs |
| **Take Snapshot** | Save current model state snapshot during a running simulation. | Button | — | Debugging transient behavior |
| **Table prefix** | Optional prefix for GeoPackage result table names. | Text | — | Multiple runs in one GPKG |
| **Results GPKG** | Path to output GeoPackage. Leave empty for model GeoPackage. | File path | — | Separate results storage |
| **Browse...** | Browse for existing GeoPackage for results. | Button | — | Selecting results location |
| **Load Inputs From Results...** | Open a results GeoPackage and apply its widget settings. | Button | — | Re-running a previous setup |

---

## 7. Running the Solver

### 7.1 Starting a Simulation

1. Configure all parameters across the **Layers**, **Mesh**, and **Parameters** tabs
2. Navigate to the **Run / Output** page (in the Parameters tab)
3. Set **Run duration** (e.g., `01:00` for 1 hour)
4. Set **Output interval** (e.g., `00:30` for 30-minute snapshots)
5. Optionally set **Line output interval** for sample-line results
6. Click **Run 2D Model**

### 7.2 Monitoring Progress

Progress is displayed in two places:

- **Progress bar** — Run / Output page — shows completion percentage
- **HYDRA2D Log** — bottom dock — shows live solver output with timestep diagnostics, CFL number, wet cell count, and error messages

The **HYDRA2D CFD Inspector** (right dock) shows real-time solver parameter snapshots.

### 7.3 Advanced Options

| Control | Location | Description |
|---|---|---|
| Adaptive CFL | Solver Parameters | Auto-adjusts dt based on CFL number |
| CUDA Graphs | Solver Parameters | Cache kernel graphs for repeated small dt |
| Rain-on-grid | Rain / Hydrology | Enable rainfall + CN infiltration |
| Drainage coupling | Structures & Drainage | 1D–2D surface coupling |
| Take Snapshot | Run / Output | Save current model state during simulation |

### 7.4 Cancelling a Run

Click **Cancel** in the Run / Output page. The solver completes the current timestep, writes a partial snapshot, and exits gracefully.

### 7.5 Running Headless (No QGIS)

The same GPU solver can run from a terminal or a CI/CD pipeline without
launching QGIS. This is useful for batch sweeps, automated regression
tests, and running long simulations on a headless GPU server.

```bash
mamba activate qgis_stable
python -m swe2d.cli run mesh.gpkg params.json --results out.gpkg --progress
```

The mesh GPKG must be pre-baked (generated via `tools/gmsh_topology_mesher.py`
or the Studio UI). The params file is JSON — the same shape the Studio UI
persists to the project's `workbench_state_json`, minus widget types.

**Batch runs** with concurrent GPU execution (via NVIDIA MPS):

```bash
python -m swe2d.cli batch batch.json mesh.gpkg --results out.gpkg -w 4
```

The CLI writes results to a separate GPKG with the same schema as the
Studio UI's output. Optional `--status-file-path` writes a JSON status
file every few seconds so a parent process (typically the QGIS batch
dialog) can show progress without parsing stdout.

See **[CLI Guide](CLI_GUIDE.md)** for the full command reference, params
JSON schema, status file format, and programmatic API
(`from swe2d.cli.headless_runner import execute_run`).

---

## 8. Results & Postprocessing

### 8.1 Sample Lines Setup

1. Draw a sample line layer on the map using the QGIS digitizing tools
2. Select the sample lines layer in the **Load Layers** page (Layers tab)
3. After a run, results at each line are automatically computed at the line output interval
4. Use the **HYDRA2D Results** dock (right) to view timeseries and profiles

### 8.2 Results Panel

The right-side **HYDRA2D Results** dock provides run selection, timestep navigation, and overlay controls:

<div class="fig-frame"><div class="fig-header">HYDRA2D Results</div><div class="fig-body">
<div class="fig-row"><label>Run selector:</label><input value="swe2d_20260622_120000 ▾"></div>
<div class="fig-row"><label>Variable:</label><input value="Depth (h) ▾"></div>
<div class="fig-row"><label>Timestep:</label><input value="12 / 48 ▾"></div>
<div class="fig-row" style="margin-left:158px;gap:16px;">
<label style="flex:none;width:auto;font-size:11px;"><input type="checkbox" checked> Show overlay</label>
<label style="flex:none;width:auto;font-size:11px;"><input type="checkbox"> Show arrows</label>
</div>
<div class="fig-sep"></div>
<div class="fig-section-label">Sample Lines</div>
<div class="fig-row"><label>Line:</label><input value="sample_line_01 ▾"></div>
<div class="fig-btn-row"><div class="fig-btn secondary" style="flex:1">Plot Profile</div><div class="fig-btn secondary" style="flex:1">Plot Timeseries</div></div>
<div class="fig-btn-row"><div class="fig-btn secondary" style="flex:1">Export CSV</div></div>
<div class="fig-sep"></div>
<div class="fig-section-label">Coupling Diagnostics</div>
<div class="fig-row"><label>Metric:</label><input value="Drainage link flow ▾"></div>
<div class="fig-row"><label>Element:</label><input value="link_0042 ▾"></div>
</div></div>

- Scrub through timesteps using the **HYDRA2D Temporal** dock (bottom):

<div class="fig-frame" style="max-width:100%"><div class="fig-header">HYDRA2D Temporal</div><div class="fig-body">
<div style="display:flex;align-items:center;gap:6px;font-size:12px;">
<span style="color:var(--fig-label);">◀◀</span><span style="color:var(--fig-btn);">▶</span><span style="color:var(--fig-label);">▶▶</span>
<div class="fig-progress" style="flex:1"><div class="fig-progress-fill" style="width:25%"></div></div>
<span>t = 300 s / 3600 s</span>
<span style="color:var(--fig-label);">▸</span>
</div>
</div></div>

- Plot depth, velocity, and WSE along sample lines
- View coupling diagnostics for drainage and structure flows

### 8.3 Results Overlay on Map

The **HYDRA2D Results** dock provides high-performance canvas overlay:

<div class="fig-frame"><div class="fig-header">Map Overlay Controls</div><div class="fig-body">
<div class="fig-row" style="margin-left:0;"><label style="flex:none;text-align:left;width:auto;"><input type="checkbox" checked> Show overlay</label></div>
<div class="fig-row"><label>Style:</label><input value="Depth (h) ▾"></div>
<div class="fig-row"><label>Color map:</label><input value="Viridis ▾"></div>
<div class="fig-row"><label>Opacity:</label><input value="0.7" style="width:60px;flex:none;"></div>
<div class="fig-row" style="margin-left:0;gap:16px;">
<label style="flex:none;width:auto;font-size:11px;"><input type="checkbox"> Velocity arrows</label>
<label style="flex:none;width:auto;font-size:11px;"><input type="checkbox"> Streamlines</label>
</div>
<div class="fig-sep"></div>
<div class="fig-btn-row"><div class="fig-btn secondary" style="flex:1">Export Overlay to GeoTIFF</div></div>
</div></div>

### 8.4 Export Workflows

| Action | Location |
|---|---|
| Export Overlay to GeoTIFF | Results dock → overlay controls |
| Export Line CSV | Results dock → Line viewer → **Export Table CSV** |
| Take Snapshot | Run / Output page → **Take Snapshot** (saves current state to GPKG) |

### 8.5 Table Prefix and Custom GPKG Path

- **Table prefix**: When running multiple simulations, set a unique prefix in the Run / Output page to keep result tables separate in the same GeoPackage
- **Custom GPKG path**: By default, results are written to the model GeoPackage. Specify an alternate path for large result sets or portable data packages

### 8.6 GeoPackage Explorer

Open the **Model GeoPackage Explorer** from the Layers tab → Utilities page to:
- Browse all result tables and input layers
- Preview table contents
- Rename or delete tables
- Open matching viewers

---

## 9. Troubleshooting

| Problem | Symptom | Solution |
|---|---|---|
| GPU not detected | `swe2d_gpu_available()` returns False | Verify CUDA toolkit on PATH; rebuild with `cmake .. -DCUDAToolkit_ROOT=/path/to/cuda` |
| Build fails with pybind11 | `pybind11 not found` | CMake auto-fetches pybind11; check internet access or install `pybind11-dev` |
| Segfault on startup | QGIS crashes | Run `fc-cache -r`; check fontconfig cache corruption |
| NaN in results | Solver diverges | Lower CFL (0.3), switch to First-order or MUSCL-MinMod, increase h_min |
| Wet/dry chatter | Oscillating depth at fronts | Enable `active_set_hysteresis`, increase `front_flux_damping` |
| Culvert flow too low | Near-zero culvert discharge | Check FHWA code matches inlet type; verify invert elevations and slope |
| Drainage coupling slow | Coupling dominates runtime | Reduce `coupling_substeps`, increase `head_deadband_m` |
| CUDA graph errors | "CUDA graph replay failed" | Disable CUDA graphs for RK4+; reduce graph window |
| Memory exhaustion | Kernel launch failure | Reduce mesh size; increase output interval; reduce VRAM usage |
| Mesh generation fails | Gmsh returns error | Check region geometry validity; try lower target size; switch triangle algorithm |

### Performance Tips

- Use **quadrilateral cells** for structured domains — they're ~30% faster than triangles on GPU
- **CUDA Graphs** give 10–20% speedup for small fixed timesteps (disable for RK4+)
- Keep **wet domain** compact — GPU parallelism is limited by the wet cell count
- Set **output interval** to be ≥10× larger than dt to avoid snapshot overhead
- Use **streamline backend = CUDA** for fast velocity visualization

---

## 10. Layer Styles (QML)

SWE2D automatically applies styled layer definitions (QML) to every
GeoPackage layer when it is loaded. The styles configure editor widgets,
field aliases, constraints, and default values — not visual symbology.

### How Styles Are Applied

1. When a **new model GPKG is created**, all QML files from the `QML/`
   directory are embedded into the GPKG's `layer_styles` table.
2. When a **model GPKG is loaded**, each layer gets its QML style applied
   from the embedded `layer_styles` table.
3. For **topology template layers** (created fresh, not from GPKG), QML
   is loaded directly from the `QML/` directory on disk.

### Customizing Styles

You can customize layer styles in two ways:

#### Option 1: Save to the GeoPackage (recommended)

1. In QGIS, right-click the layer → **Properties → Symbology**
2. Make your changes (colors, labels, rendering order, etc.)
3. Click **Apply**, then **OK**
4. Right-click the layer → **Properties → Styles → Save Style**
5. Choose **In GeoPackage (database)** and select the `default` style
6. Click **Save**

The style is now embedded in the GPKG and will be applied every time the
layer is loaded from this GPKG.

#### Option 2: Save as a QML file

1. In QGIS, right-click the layer → **Properties → Styles → Save Style**
2. Choose **In QML file**
3. Save to the `QML/` directory inside your HYDRA plugin installation:
   ```
   <plugin_dir>/QML/<layer_name>.qml
   ```
4. The plugin will use your modified QML the next time a GPKG is created
   or a topology template layer is loaded.

> **Note:** QML files in the `QML/` directory are the source of truth for
> new GPKG creation. If you save a custom style only to the GPKG (Option 1),
> it will not carry over to new GPKGs. For permanent style changes, use
> Option 2.

### Available QML Files

| File | Layer |
|------|-------|
| `swe2d_bc_lines.qml` | Boundary condition lines |
| `swe2d_cn_zones.qml` | CN zones |
| `swe2d_drainage_inlets.qml` | Drainage inlets |
| `swe2d_drainage_links.qml` | Drainage links |
| `swe2d_drainage_node_inlets.qml` | Drainage node-inlets |
| `swe2d_drainage_nodes.qml` | Drainage nodes |
| `swe2d_hydrographs.qml` | Hydrographs |
| `swe2d_hyetographs.qml` | Hyetographs |
| `swe2d_manning_zones.qml` | Manning's n zones |
| `swe2d_rain_gages.qml` | Rain gages |
| `swe2d_sample_lines.qml` | Sample lines |
| `swe2d_storm_areas.qml` | Storm areas |
| `swe2d_structures.qml` | Structures |
| `swe2d_topo_arcs.qml` | Topology arcs |
| `swe2d_topo_constraints.qml` | Topology constraints |
| `swe2d_topo_nodes.qml` | Topology nodes |
| `swe2d_topo_quad_edges.qml` | Topology quad edges |
| `swe2d_topo_regions.qml` | Topology regions |

## 11. References

- Toro, E. F. *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.
- FHWA. *Hydraulic Design of Highway Culverts* (HDS-5). FHWA-HIF-05-012.
- Akan, A. O. *Urban Stormwater Hydrology*. Technomic Publishing.
- QGIS Documentation: https://docs.qgis.org
- HYDRA GPU Architecture: [SWE2D_GPU_ARCHITECTURE_REPORT.md](SWE2D_GPU_ARCHITECTURE_REPORT.md)
- Headless CLI Guide: [CLI_GUIDE.md](CLI_GUIDE.md)
- Gmsh Meshing Guide: [GMSH_MESHING_GUIDE.md](GMSH_MESHING_GUIDE.md)
- Results Path Guide: [RESULTS_PATH_GUIDE.md](RESULTS_PATH_GUIDE.md)
- GeoPackage Explorer Guide: [GPKG_EXPLORER_GUIDE.md](GPKG_EXPLORER_GUIDE.md)
- Drainage Solver Mode Guide: [DRAINAGE_SOLVER_MODE_GUIDE.md](DRAINAGE_SOLVER_MODE_GUIDE.md)
- Rainfall CN Guide: [RAINFALL_CN_GUIDE.md](RAINFALL_CN_GUIDE.md)

---

**[HYDRA2DGPU GitHub](https://github.com/aspragueumkc/qgis-hydra-plugin)** | **[C++ API Reference](dochub://open/C++%20API%20Reference)**

*Document generated from HYDRA repository state on 2026-06-22. For the latest API details, see source code and inline docstrings.*

---

## Related Documentation

- **[Documentation Index](INDEX.md)** — All guides by audience
- **[CLI Guide](CLI_GUIDE.md)** — Headless runs, batch sweeps, CI/CD
- **[Developer Guide](DEVELOPER_GUIDE.md)** — Architecture, test suite, contribution workflow
- **[Model GeoPackage Schema](MODEL_GEOPACKAGE_SCHEMA.md)** — Input GPKG tables
- **[Results GeoPackage Schema](RESULTS_GEOPACKAGE_SCHEMA.md)** — Output GPKG tables
- **[Gmsh Meshing Guide](GMSH_MESHING_GUIDE.md)** — Mesh generation workflow
- **[Repository Knowledge Graph](../graphify-out/GRAPH_REPORT.md)** — Codebase structure
