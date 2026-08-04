# HYDRA — GPU-Accelerated 2D Shallow Water Equation Plugin for QGIS

**Version**: 2.0 (GPU-Only)
**Last Updated**: 2026-07-29

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
10. [Agent-Assisted Modeling (MCP)](#10-agent-assisted-modeling-mcp)
11. [Layer Styles (QML)](#11-layer-styles-qml)
12. [Graph Editor](#12-graph-editor)
13. [CLI Quickstart](#13-cli-quickstart)
14. [Batch Runner Workflow](#14-batch-runner-workflow)
15. [References](#15-references)

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

![HYDRA2D Studio workbench — full QGIS window with all docks visible](images/studio_overview.jpg)
*Live screenshot of the HYDRA2D Studio workbench docked inside QGIS. Visible from this single view: the **HYDRA2D Model Setup** dock (left) with its Mesh Generation toolbox, the **HYDRA2D Run** dock (top) with Run / Cancel / Snapshot controls, the **HYDRA2D Temporal** dock (top) with playback controls, the **HYDRA2D View** dock (top right) with Mesh / Time Series / Profile tabs, the **HYDRA2D Results** dock (right) with the Overlay toolbox, the **HYDRA2D CFD Inspector / HYDRA2D Results** tab strip, and the **HYDRA2D Log** dock (bottom) with startup messages. The Studio workbench opens via **HYDRA2DGPU → Open HYDRA2DGPU Workbench** once a model GeoPackage has been created or loaded.*

The **HYDRA2D Model Setup** dock (left) has two top-level tabs — **Mesh Generation** and **Simulation** — that switch the entire dock between the model-loading workflow and the solver configuration workflow. Each tab contains its own internal pages.

![HYDRA2D Model Setup dock — Mesh Generation tab](images/hydra2d_setup.jpg)
*Live screenshot of the HYDRA2D Model Setup dock with the **Mesh Generation** tab active (Filter parameters, Show advanced parameters, Import/Export section). When no model GeoPackage is loaded the dock is largely empty; once a model is opened the Layer Setup / Mesh Setup / Utilities pages populate. The accordion at the bottom of the screenshot is the empty-state rendering of those sub-pages — only the active page is populated at runtime.*

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

### 4.1 Import / Export Page (Layers tab — first page)

The Import / Export page loads the model GeoPackage, exchanges mesh data with the QGIS map canvas, and exposes the import / export / assign-Z operations that were previously split across separate "Load Layers" and "Mesh Setup" pages. Select the layers that define your model geometry, terrain, roughness, and drainage configuration.

![Import / Export page (Mesh Generation tab)](images/tab_mg_0_import_export.jpg)
*Live screenshot of the **Import / Export** page (first page of the **Mesh Generation** tab in the HYDRA2D Model Setup dock). The page exposes the load / export / assign-Z operations and the default boundary-condition controls. Combo boxes default to `(none)` until a model GeoPackage is opened; the dock's other QToolBox pages — Layer Setup, General, Algorithm, Mesh Definition, Quality Loop — collapse below.*

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

### 4.2 Layer Setup Page (Layers tab — second page)

The Layer Setup page provides a grid of QGIS layer selectors for the topology layers used by Gmsh-based mesh generation.

![Layer Setup page (Mesh Generation tab)](images/tab_mg_1_layer_setup.jpg)
*Live screenshot of the **Layer Setup** page (second page of the **Mesh Generation** tab). Visible widgets: Topology nodes layer, Topology arcs layer, Topology regions layer, Constraints layer, Quad edges / transition layers, Elevation source. Once a model GeoPackage is opened, the combos populate with the matching QGIS layers.*

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

> **Per-edge BC relaxation:** Each BC line segment can optionally supply a
> `bc_relax` field (0.0–1.0) to override the global **Open BC relax** value
> for that edge. When set, the per-edge value takes precedence over the
> global spinbox. Leave the field empty or set to `NULL` to use the global
> default. WALL, INFLOW_Q, and STAGE boundary types ignore the `bc_relax`
> field entirely.

### 4.3 General Page (Mesh tab — Controls / General)

The General page sets the meshing backend, default target size, and default cell type. The standalone "Utilities" subpage from earlier guides has been folded into the menu bar (`HYDRA2DGPU → Open Model GeoPackage Explorer`, `Open Run Log Viewer`).

![General page (Mesh Generation tab)](images/tab_mg_2_general.jpg)
*Live screenshot of the **General** page (third page of the **Mesh Generation** tab). Visible widgets: Meshing backend (Gmsh recommended, Structured fallback), Default target size, Default cell type, plus the **Generate Mesh** and **Terminate** action buttons. The Mesh Generation tab's other QToolBox pages (Import/Export, Layer Setup, Algorithm, Mesh Definition, Quality Loop) collapse below the active page.*

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

### 4.4 Algorithm Page (Mesh Generation tab)

![Algorithm page (Mesh Generation tab)](images/tab_mg_3_algorithm.jpg)
*Live screenshot of the **Algorithm** page (fourth page of the **Mesh Generation** tab). Contains the Gmsh algorithm selector (Triangle, Quad, Recombine, smoothing passes, num threads) plus the **Validate & Summarize**, **Edit Region Attributes**, **Edit Quad/Transition Edges** action buttons. See the full reference table below for the Gmsh algorithm parameters.*

### 4.5 Mesh Definition Page

![Mesh Definition page (Mesh Generation tab)](images/tab_mg_4_mesh_definition.jpg)
*Live screenshot of the **Mesh Definition** page (fifth page of the **Mesh Generation** tab). Defines arc handling, interface transition grading, min cell size, edge tolerance, and region-target-size usage.*

### 4.6 Quality Loop Page

![Quality Loop page (Mesh Generation tab)](images/tab_mg_5_quality_loop.jpg)
*Live screenshot of the **Quality Loop** page (sixth page of the **Mesh Generation** tab). Enables the iterative Gmsh quality loop with max iterations and time limit.

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

![Solver Parameters page (Simulation tab)](images/tab_sim_0_solver_parameters.jpg)
*Live screenshot of the **Solver Parameters** page (first page of the **Simulation** tab in the HYDRA2D Model Setup dock). Visible sections: **Time Stepping** (CFL, dt, initial dt, variable-timestep toggle), **Boundary Conditions** (default BC type, BC lines layer, inflow progressive, uniform inflow velocity), **Physics_Friction** (Manning polygons, Manning n, h_min, internal flow layer), **Numerics** (Reconstruction, Temporal discretization), and the collapsed **Initial Conditions** group. Other QToolBox pages (Rain / Hydrology, Stability Controls, Structures & Drainage, Output) collapse below the active page.*

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
| **Reconstruction** | Spatial scheme for cell-face value extrapolation. | `First-order (0)`, `MUSCL Fast (1)`, `MUSCL MinMod (2)`, `MUSCL MC (3)`, `MUSCL Van Leer (4)`, `Barth-Jespersen (5)`, `WENO3 (6)`, `WENO5 (7)`, `MP5 (8)` | First-order (0) | Accuracy vs. speed trade-off — see scheme guidance below |
| **Temporal discretization** | Time integration method (ODE solver). | `Euler RK1 (1)`, `RK2 Heun (2)`, `RK4 (4)`, `Graph-safe RK4 (5)`, `Graph-safe RK5 (6)` | RK2 (2) | Accuracy vs. stability trade-off |

> **Choosing a reconstruction scheme:** See [SOLVER_ORDER_AND_STENCIL.md](SOLVER_ORDER_AND_STENCIL.md)
> for the complete spatial accuracy analysis and [ADVANCED_SPATIAL_SCHEMES.md](ADVANCED_SPATIAL_SCHEMES.md)
> for detailed guidance on the three schemes new in v2.0:
>
> - **Barth-Jespersen (5)**: Best when mesh quality is poor (sliver triangles, stretched quads, mixed
>   element types). Degrades gracefully to 1st-order isotropically. Good for urban drainage where
>   meshes are complex. Same CFL as MUSCL (0.8).
> - **WENO3 (6)**: 3rd-order accuracy with only 1-ring stencil memory. Good general-purpose upgrade
>   from 2nd-order MUSCL. Smoother results than TVD limiters on smooth flows.
> - **MP5 (8)**: Highest-order option (4th). Requires CFL ≤ 0.4 — doubles wall-clock time vs CFL 0.8.
>   Best for smooth flows where accuracy matters more than speed.
>
> **Note:** WENO5 moved from scheme 6 to scheme 7. If you have saved configurations with
> `spatial-scheme=6`, they will now select WENO3 instead of WENO5. The CLI emits a migration warning
> when the old value is detected.

### 6.2 Rain / Hydrology Page

![Rain / Hydrology page (Simulation tab)](images/tab_sim_1_rain___hydrology.jpg)
*Live screenshot of the **Rain / Hydrology** page (second page of the **Simulation** tab). Visible sections: **Rainfall Input** (Rain gages, Rain hyetographs, Rain rate, Spatial rainfall toggle, Rain rate update interval, Storm area layer, Rain boundary buffer rings), **Infiltration** (method, CN polygons, default CN, SCS Ia/S ratio), and **Source Stability** (Max rel depth increase, Max source dh/step, Max source rate, Source CFL beta, Source max substeps).*

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

![Stability Controls page (Simulation tab)](images/tab_sim_2_stability_controls.jpg)
*Live screenshot of the **Stability Controls** page (third page of the **Simulation** tab). Sets shallow damping, wet/dry front handling, depth cap, momentum cap (min speed / celerity multiplier), max inverse area, and CFL lambda cap.*

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
| **Open BC relax** | Relaxation factor applied to OPEN, REFLECT, NORMAL_DEPTH, and NORMAL_DEPTH_SLOPE boundaries to damp reflections. 0 = no relaxation (fully reflective damping), 1 = maximum relaxation. WALL, INFLOW_Q, and STAGE boundaries are unaffected. | 0.0–1.0 | 0.0 | Higher-order spatial schemes (WENO5, MUSCL Van Leer) with boundary instabilities |

### 6.4 Structures & Drainage Page

![Structures & Drainage page (Simulation tab)](images/tab_sim_3_structures_and_drainage.jpg)
*Live screenshot of the **Structures & Drainage** page (fourth page of the **Simulation** tab). Configures the 1D-2D coupling (loop backend, culvert solver, culvert face-flux toggle, bridge coupling), the drainage solver (mode, GPU method, substeps, head deadband, dynamic relaxation), and adaptive coupling (depth fraction, wave Courant, implicit iters, implicit relax, redistribution).*

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

### 6.6 Drainage Configuration Details

This section provides detailed guidance for configuring drainage network parameters in the GeoPackage layers.

#### Node Types and Semantics

The drainage network supports several node types, each with specific behavior:

| Node Type | Description | Required Fields | Typical Use |
|-----------|-------------|-----------------|-------------|
| **junction** | Standard pipe junction where multiple pipes meet | `invert_elev`, `max_depth`, `surface_area` | Interior network connections |
| **inlet** | Node with surface inlet capture ( grate, curb, etc.) | `invert_elev`, `max_depth`, `surface_area`, plus inlet configuration | Surface → pipe water entry |
| **outfall** | Network discharge point with boundary condition | `invert_elev`, `max_depth`, `surface_area`, `outfall_mode` | Pipe → surface water exit |
| **storage** | Large storage node (detention pond, wetland) | `invert_elev`, `max_depth`, `surface_area` | Flood attenuation |
| **pipe_end** | Pipe that terminates at a 2D surface cell | `invert_elev`, `max_depth`, `surface_area`, pipe-end geometry | Direct pipe-surface coupling |

**Node behavior:**
- **Junction nodes** provide storage volume when pipes are surcharged. Water can pool in the node when inflow exceeds pipe capacity.
- **Inlet nodes** capture surface water and route it into the pipe network. Configure inlet type in the `swe2d_drainage_inlets` table.
- **Outfall nodes** discharge water back to the surface. Configure boundary condition via `outfall_mode` field.

#### Surcharge Behavior

When pipe flow exceeds capacity, the network becomes **pressurized** and water can:

1. **Pool at junction nodes** — if node depth exceeds `max_depth`, water is stored in the node (limited by `surface_area`).
2. **Overflow to surface** — if node depth exceeds rim elevation (`invert_elev + max_depth`), water flows back to 2D surface cells (via SURFACE_2D_JUNCTION_OVERFLOW faces).

**Surcharge configuration:**
- Set `rim_elev` to control when overflow starts (default: `invert_elev + max_depth`)
- Set `surface_area` to define node storage capacity (larger = more buffering)
- Enable `enable_overflow` and set `overflow_elevation` for controlled overflow
- Set `max_overflow_rate` to limit discharge during surcharge (prevents numerical instability)

**Manhole storage example:**
```
# 5m × 5m manhole, 2m max depth
invert_elev = 10.0    # Pipe invert elevation
max_depth = 2.0      # Water can rise 2m above invert
surface_area = 25.0  # 25 m² storage area
rim_elev = 12.0      # Overflow starts at 12m elevation

# Can store up to 50 m³ before overflowing
```

#### Loss Coefficients

Loss coefficients account for energy losses at pipe entrances and exits due to:
- Pipe geometry changes (contractions, expansions)
- Fittings (bends, tees, crosses)
- Appurtenances (valves, meters)

**Coefficient hierarchy** (highest priority first):
1. **Node-level overrides** — `inlet_loss_k`, `outlet_loss_k` on DrainageNode
2. **Link-level values** — `entrance_loss_k`, `exit_loss_k` on DrainageLink
3. **Default values** — 0.5 (entrance), 1.0 (exit)

**Typical values** (from FHWA HDS-5):
- **Pipe entrance**: 0.2–0.5 (well-designed transitions), 0.5–1.0 (sharp transitions)
- **Pipe exit**: 0.5–1.0 (gradual expansion), 1.0–2.0 (sudden expansion)
- **Manhole**: 0.0–0.2 (minimal loss in large manholes)

**Loss coefficient semantics:**
```python
# Node-level override (takes precedence)
node = DrainageNode(
    node_id="MH-1",
    inlet_loss_k=0.3,    # Override for all pipes entering this manhole
    outlet_loss_k=0.8,   # Override for all pipes leaving this manhole
)

# Link-level value (fallback if node-level not set)
link = DrainageLink(
    link_id="P-1",
    entrance_loss_k=0.5,  # FHWA standard entrance loss
    exit_loss_k=1.0,      # FHWA standard exit loss
)
```

**When to adjust loss coefficients:**
- **Increase** (e.g., 0.5 → 1.0) if model overestimates flow (too much inflow from upstream)
- **Decrease** (e.g., 0.5 → 0.2) if model underestimates flow (not enough inflow)
- **Set to 0** for direct connections (no fitting losses)

#### Outfall Boundary Conditions

Configure outfall behavior via the `outfall_mode` field on drainage nodes:

| Mode | Description | Required Fields | Use Case |
|------|-------------|-----------------|----------|
| **free** (default) | Node drains freely; depth reset to 0 each step unless backwatered by surface cell | None | Most outfalls |
| **fixed_wse** | Tailwater clamped to fixed water surface elevation | `outfall_fixed_wse` | Controlled tailwater conditions |
| **stage_discharge** | Outflow rate from rating table | `outfall_rating_table` (list of [wse, Q] pairs) | Complex tailwater curves |
| **tabular** | Time-varying rating table (stage, Q vs. time) | `outfall_rating_wse`, `outfall_rating_q`, `outfall_tabular_time` | Time-varying tailwater |

**Rating table format** (for `stage_discharge`):
```python
outfall_rating_table = [
    [wse_1, Q_1],   # [meters, m³/s]
    [wse_2, Q_2],
    # ... linear interpolation between points
]
```

**Outfall best practices:**
- Use **free** mode for most cases — allows proper backwater interaction with 2D surface
- Use **fixed_wse** when modeling a downstream control structure with known water level
- Use **stage_discharge** when you have field-measured rating curves
- For tidal outfalls, use **free** mode and let the 2D surface provide tidal forcing

#### Interpreting Drainage Results

Drainage results are available in the results GeoPackage tables:

**Per-cell results** (`swe2d_cell_results` table):
| Field | Meaning | Units | Typical Range |
|-------|---------|-------|---------------|
| `cell_h` | Flow depth (water surface - invert) | m | 0–pipe_diameter |
| `cell_y` | Water surface elevation | m | invert_elev to rim_elev |
| `cell_Q` | Flow rate | m³/s | -max_flow to +max_flow |
| `cell_velocity` | Flow velocity (Q/A) | m/s | 0–10 (pressurized) |
| `cell_A` | Flow area | m² | 0 to full_pipe_area |

**Per-link results** (`swe2d_link_results` table):
| Field | Meaning | Units | Typical Range |
|-------|---------|-------|---------------|
| `link_q` | Mean flow magnitude | m³/s | 0–max_flow |
| `link_velocity` | Mean velocity | m/s | 0–10 |
| `link_froude` | Froude number | dimensionless | 0–1 (subcritical), >1 (supercritical) |
| `link_is_surcharged` | Pressurized flow flag | boolean | 0 or 1 |

**Common result patterns:**

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| **Node depth ≈ max_depth consistently** | Node is at capacity, may be surcharging | Check for surface overflow, consider increasing `surface_area` |
| **Link is_surcharged = 1 but link_Q ≈ 0** | Pipe is full but not flowing — potential blockage | Check downstream conditions, verify loss coefficients |
| **High inlet capture but low pipe flow** | Bottleneck downstream | Increase pipe diameter or reduce loss coefficients |
| **Oscillating node depth** | Numerical instability | Reduce `dt`, increase `coupling_substeps`, switch to First-order reconstruction |
| **Mass balance drift over time** | Accumulating error | Check CFL, verify loss coefficients, reduce `adaptive_depth_fraction` |

**Visualization tips:**
- Use `link_is_surcharged` to identify pressurized pipes (color them red in results overlay)
- Use `node_depth` vs `node_max_depth` ratio to see which nodes are near capacity
- Use `cell_velocity` to identify high-velocity sections (potential erosion concerns)
- Compare initial and final mass to verify conservation (should be within 0.1% for well-behaved simulations)

### 6.5 Output Page (Simulation tab — Run / Output)

![Output page (Simulation tab)](images/tab_sim_4_output.jpg)
*Live screenshot of the **Output** page (fifth page of the **Simulation** tab). The actual Run / Cancel / Snapshot / Batch controls are in the top **HYDRA Run** dock; this page collects the output / debugging / results-output controls (output interval, line output interval, preview overrides, preview coupling, take snapshot, table prefix, results GPKG path).*

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

![HYDRA2D Results dock — Overlay toolbox page open](images/hydra2d_results.jpg)
*Live screenshot of the HYDRA2D Results dock with the **Overlay** toolbox page open. Visible sections: **Field_Colormap** (Field, WSE render, Colormap, Resolution), **Color Range** (Opacity, Auto contrast, Min depth threshold, Color min/max, Reset), **Overlay Style** (Lock canvas extent, Visible cells only). The **Runs** toolbox tab below selects the run id, variable, timestep, sample lines, and coupling diagnostics.*

- Scrub through timesteps using the **HYDRA2D Temporal** dock (bottom):

![HYDRA2D Temporal dock — playback controls](images/hydra2d_temporal.jpg)
*Live screenshot of the HYDRA2D Temporal dock. The animation controls (◀◀ / ▶ / ▶▶ buttons, timeline scrubber, playback-speed selector) become active once the first timestep of a finished run is loaded into the Results dock.*

The **HYDRA2D Log** dock (bottom of the QGIS window) streams solver output live:

![HYDRA2D Log dock — startup messages](images/hydra2d_log.jpg)
*Live screenshot of the HYDRA2D Log dock at startup. After a run starts the dock fills with per-timestep progress lines, [WARNING] / [ERROR] entries highlighted in red, and the final solver summary.*

- Plot depth, velocity, and WSE along sample lines
- View coupling diagnostics for drainage and structure flows

### 8.3 Results Overlay on Map

The **HYDRA2D Results** dock provides high-performance canvas overlay:

![HYDRA2D Results dock — overlay section](images/hydra2d_results.jpg)
*Live screenshot of the HYDRA2D Results dock showing the Overlay toolbox section (Field, WSE render, Colormap, Resolution, Opacity, Auto contrast, Min/Max depth thresholds, Lock canvas extent, Visible cells only). Style, colormap, and overlay toggles appear here when a finished run is loaded; **Export Overlay to GeoTIFF** is in the same section.*

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

### ANUGA Validation Suite

The full hydraulic validation suite is in `tests/test_anuga_suite.py` and
imports ANUGA's analytical solutions directly from
`reference/anuga_validation_tests/`. Run it to verify SWE2D matches ANUGA
across ~20 classical dam-break, lake-at-rest, subcritical, supercritical,
transcritical, and 2D radial test cases:

```bash
mamba run -n qgis_stable python -m unittest tests.test_anuga_suite -v
```

Each test compares the GPU solution against ANUGA's closed-form or numerical
ground truth with documented L1/L∞ tolerances.

---

## 10. Agent-Assisted Modeling (MCP)

HYDRA ships with an [MCP](https://modelcontextprotocol.io/) server
(`tools/hydra_mcp/`) that lets an AI agent assist you live inside QGIS or
operate headless model/results data.

### Enabling the live GUI bridge

There are two ways to expose a running QGIS session to an agent:

1. **Before launching QGIS:** set the environment variable so the plugin
   auto-starts the bridge:
   ```bash
   HYDRA_MCP_BRIDGE=1 qgis
   ```
   The plugin writes a per-session token file (`hydra_mcp_bridge_*.json`)
   under `$XDG_RUNTIME_DIR` or `/tmp`.

2. **If QGIS is already running:** open the QGIS Python Console
   (**Plugins ▸ Python Console**) and paste:
   ```python
   import os
   if not os.environ.get("HYDRA_MCP_BRIDGE"):
       os.environ["HYDRA_MCP_BRIDGE"] = "1"
       from tools.hydra_mcp.qgis_bridge import bootstrap_bridge_if_needed
       bootstrap_bridge_if_needed()
   ```
   Watch the log for the `HYDRA_MCP_BRIDGE_READY <socket> <token_path>` line.

### Attaching an agent

With the bridge running, an MCP client calls:

```json
{"mode": "display"}
```

`gui_launch(mode="display")` waits up to its `timeout` for the token file,
connects, and returns session metadata.  The agent can then drive the live
Studio workbench (`gui_widget_tree`, `gui_get_value`, `gui_set_value`,
`gui_screenshot`) and query model/results GeoPackages (`model_inspect`,
`run_list`, `results_query`).

> **Security note:** the bridge is same-machine only (QLocalSocket) and is
> opt-in.  Normal plugin use is unaffected when `HYDRA_MCP_BRIDGE` is unset.
> See `tools/hydra_mcp/README.md` for full client configuration.

The MCP server must be started with the **same Python environment that QGIS
itself uses** — the one that has `qgis`, `osgeo`, and the HYDRA plugin on
`sys.path`.  If the server is launched from a different environment, imports
such as `osgeo` and the native solver extensions will fail.  See
`tools/hydra_mcp/README.md` for example client configurations.

---

## 11. Layer Styles (QML)

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

## 12. Graph Editor (Hydrographs & Hyetographs)

Use the **Graph Editor** to author and edit time-series graphs —
hydrographs (discharge vs. time) and hyetographs (rainfall intensity vs.
time) — for the active model GeoPackage. The dialog writes back to the
`swe2d_hydrographs` and `swe2d_hyetographs` tables in the same GPKG, so
the forcing curves you author here flow directly into the **Simulation →
Run / Output** page.

### When to use it

- Authoring or editing a **discharge hydrograph** for an inflow boundary
  without exporting to CSV or hand-editing tables.
- Authoring or editing a **rainfall hyetograph** for a uniform-rainfall
  simulation or for one of the rain gages in a Thiessen-weighted setup.
- Visually inspecting an existing time-series before running a
  simulation — the editor plots the curve as you type.

### Open the Graph Editor

```text
HYDRA2DGPU → Graph Editor…
```

(Equivalent menu action is `HYDRA2DMenuGraphEditorAction`.) If the active
project has no model GeoPackage loaded, the dialog shows an error and
exits.

### Workflow

1. **Pick a series type** (hydrograph vs. hyetograph) from the dropdown.
2. **Pick or create a series id** (matches a BC line id for hydrographs or
   a rain gage id for hyetographs).
3. **Edit rows in the table.** Columns are `time` (hours) and `value`
   (m³/s for hydrographs, mm/hr for hyetographs). Add / remove rows with
   the table buttons.
4. **The plot updates live** as you type, showing the curve.
5. **Save the graph.** Writes the new / edited rows into the model
   GeoPackage's `swe2d_hydrographs` / `swe2d_hyetographs` tables.
6. **Use the graph** in the **Simulation → Run / Output** page — the
   `Build Run Spec` step reads the same tables and validates monotonic
   time, units, and series-id presence before the solver starts.

> The dialog keep-alive list is owned by the controller, so opening the
> editor multiple times is safe — each opens a separate window.

### Cross-references

- BC / rainfall semantics: [§4.2 Layer Setup](#42-layer-setup-page-layers-tab--second-page)
  and [§6.2 Rain / Hydrology Page](#62-rain--hydrology-page).
- Schema reference: [MODEL_GEOPACKAGE_SCHEMA.md](MODEL_GEOPACKAGE_SCHEMA.md).
- CLI equivalent: `bc_configure(...)` / `rainfall_configure(...)` — see
  [CLI Guide](CLI_GUIDE.md).

---

## 13. CLI Quickstart

The same GPU solver runs from a terminal or CI pipeline without QGIS via
the `swe2d-cli` entry point. The CLI is the supported path for batch
sweeps, regression tests, and headless GPU servers.

### Single run

```bash
mamba activate qgis_stable
python -m swe2d.cli run mesh.gpkg params.json --results out.gpkg --progress
```

| Arg | Meaning |
|---|---|
| `mesh.gpkg` | Baked model GeoPackage (created via Studio or `tools/gmsh_topology_mesher.py`). |
| `params.json` | Simulation parameters (same shape the Studio UI persists to `workbench_widget_state_json`, minus widget types). |
| `--results out.gpkg` | Where to write run results. Defaults to the model GPKG if omitted. |
| `--progress` | Print a percent-progress line every few steps; required for parent processes that tail stdout. |
| `--status-file-path <file>` | Write a JSON status record to `<file>` every few seconds — consumed by the Studio batch dialog. |

**Headless output schema matches the Studio:** `swe2d_cell_results`,
`swe2d_link_results`, `swe2d_run_logs`. Inspect with
`results_query(gpkg_path, run_id, field)` (programmatic) or the Studio
Results dock (GUI).

### Multi-run batch

```bash
python -m swe2d.cli batch batch.json mesh.gpkg --results results_root.gpkg -w 4
```

See [§14 Batch Runner Workflow](#14-batch-runner-workflow) for the
`batch.json` schema and the multi-worker / MPS contract.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Success — all runs finished, results written. |
| `1` | A run failed; details in `--status-file-path` or stdout. |
| `2` | Bad arguments / missing GPKG. |
| `130` | SIGINT (Ctrl-C) — the solver completes the current timestep and exits. |

### Cross-references

- Full argument reference: [CLI_GUIDE.md](CLI_GUIDE.md).
- Params shape: [RUN_SPEC_SCHEMA.md](RUN_SPEC_SCHEMA.md).

---

## 14. Batch Runner Workflow

The batch runner (`python -m swe2d.cli batch …`) drives many runs in
parallel against a single baked mesh. Use it for parameter sweeps,
calibration studies, or Monte Carlo / sensitivity work.

### `batch.json` shape

```json
{
  "mesh": "model.gpkg",
  "results_root": "results_root.gpkg",
  "runs": [
    {"name": "baseline", "params": {"run_duration": "1:00", "manning_n": 0.025}},
    {"name": "dry-manning-035", "params": {"manning_n": 0.035}},
    {"name": "dry-manning-045", "params": {"manning_n": 0.045}}
  ]
}
```

| Key | Required | Meaning |
|---|---|---|
| `mesh` | ✅ | Path to the baked model GeoPackage. |
| `results_root` | ✅ | Output GeoPackage path. Created if missing. |
| `runs` | ✅ | List of run specs. Each entry's `name` becomes the run_id (sanitized); each entry's `params` is merged into the base run spec. |
| `runs[i].name` | optional | Defaults to `run_<idx>_<unix_ts>`. |
| `runs[i].params` | optional | Run-parameter overrides (any field accepted by `spec_build`). |

Determinism: run ids are derived from the spec (not the wall clock), so
re-running the same `batch.json` produces the same ids and overwrites the
previous results.

### Multi-worker semantics

```bash
python -m swe2d.cli batch batch.json mesh.gpkg --results results_root.gpkg -w 4
```

- `-w N` launches `N` worker subprocesses.
- Each worker is a separate `swe2d.cli run` invocation; they share the
  GPU via **NVIDIA MPS** (CUDA Multi-Process Service). Set up MPS
  before launching the batch — see NVIDIA's MPS guide. Without MPS,
  multi-worker runs will contend for the GPU and likely run slower than
  serial.
- Workers do not coordinate beyond the results GeoPackage; results
  appear in `results_root` as each worker finishes.

### Status file

```bash
python -m swe2d.cli batch batch.json mesh.gpkg --results results_root.gpkg \
    --status-file-path status.json
```

The CLI writes a JSON snapshot to `status.json` every few seconds:

```json
{
  "state": "running",
  "completed": ["baseline"],
  "in_progress": "dry-manning-035",
  "pending": ["dry-manning-045"],
  "t_s": 412.7
}
```

- The Studio batch dialog tails this file and renders progress.
- A `state: "done"` line means every run finished; a `state: "failed"`
  line includes the failing run id.

### Cancellation

- **Ctrl-C in the parent** sends SIGINT to every worker; each worker
  finishes its current timestep and exits cleanly.
- **`run_cancel(job_id)` from the MCP server** aborts a single running
  job inside an active `swe2d-cli run` (not a batch). The batch runner
  waits for currently-running workers, then exits.

### When to prefer batch over per-run

| Use case | Recommended path |
|---|---|
| Single one-off simulation, want a clear log | `swe2d.cli run … --progress` |
| Parameter sweep with deterministic IDs | `swe2d.cli batch …` |
| Live progress in Studio UI | `swe2d.cli batch … --status-file-path …` |
| Long-running unattended Monte Carlo | `nohup swe2d.cli batch … &` with MPS |

### Cross-references

- Full batch runner reference: [CLI_GUIDE.md §4](CLI_GUIDE.md).
- Source: [`tools/swe2d/cli/batch_runner.py`](../../tools/swe2d/cli/batch_runner.py).
- Batch dialog source: [`swe2d/workbench/dialogs/batch_simulation_dialog.py`](../../swe2d/workbench/dialogs/batch_simulation_dialog.py).

---

## 15. References

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
