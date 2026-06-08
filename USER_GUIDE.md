# HYDRA — QGIS Hydrodynamics & Runoff Application User Guide

This guide covers day-to-day use of the HYDRA plugin inside QGIS.

For the broader documentation index and lifecycle buckets, see [docs/README.md](docs/README.md).

## 1. What the Plugin Does

The plugin performs steady-flow 1D backwater calculations using a GeoPackage model and shows:

- Tabular result output
- Longitudinal and section plotting views
- QGIS layer-based editing workflow for model data

## 2. Prerequisites

- QGIS 3.x with Python plugin support
- HYDRA plugin installed and enabled
- A writable location for GeoPackage model files (`*.gpkg`)

Optional:

- SciPy for the `scipy` solver option
- matplotlib for in-plugin plots

## 3. Open the Plugin

From the QGIS menu bar:

- Open **HYDRA > Open HYDRA Panel**

The dock widget opens on the right side of QGIS.

## 4. Create a New Model

1. Select **HYDRA > Create New Model GeoPackage...**
2. Choose output path and file name
3. Choose CRS when prompted
4. The plugin creates a starter model and loads it into the panel

A model GeoPackage includes these key layers:

- `cross_sections`
- `centerline`
- `boundary_conditions`
- `model_results` (written after a successful run)

## 5. Load an Existing Model

1. Select **HYDRA > Load Model GeoPackage...**
2. Pick a `*.gpkg` model
3. The panel loads boundary settings, sections, and persisted results (if present)

## 6. Edit Model Data

The plugin is designed for layer/form editing in QGIS.

1. Select **HYDRA > Enable/Disable Layer Editing** (to enable)
2. Edit features in `cross_sections`, `centerline`, and `boundary_conditions`
3. Use configured layer forms and actions
4. Select **HYDRA > Save Layer Edits** to commit

Important:

- If editing is enabled and unsaved changes exist, running is blocked.
- Save layer edits before running.

### 6.1 Cross-Section Form Actions

The cross-section form includes actions such as:

- Select terrain raster (stores selected raster id in project variable)
- Update feature Z values from terrain

Culvert fields are shown/hidden based on `culvert_code`.

## 7. Set Boundary and Solver Options

In the panel Boundary tab, you can set:

- DS BC: `known_wse` or `normal_depth`
- DS value (WSE or slope input)
- Flow (cfs)

From the QGIS menu:

- **HYDRA > Options > Solver > Python (py)**
- **HYDRA > Options > Solver > SciPy (scipy)**
- **HYDRA > Options > Alpha Method > Conveyance**
- **HYDRA > Options > Alpha Method > Area**

These menu choices mirror the panel selector state.

## 8. Run the Model

1. Ensure edits are saved
2. Select **HYDRA > Run Model**
3. Review status and outputs in the panel

The run persists results to `model_results` in the same GeoPackage.

## 9. View Results

Use menu actions:

- **HYDRA > Open Results Plot**
- **HYDRA > Open Results Table**

Results can also reload automatically when opening a model with existing `model_results` rows.

## 10. SWE2D Workbench

The plugin also includes a 2D SWE workbench for topology-driven meshing and native solver runs.

Typical workflow:

1. Open the 2D/SWE2D workbench in the plugin UI.
2. Load or select the topology layers for nodes, arcs, regions, and constraints.
3. Choose a meshing backend.
4. Use Gmsh when you need constraint-driven local refinement inside a larger region.
5. Use TQMesh when you want a quad-oriented layout for explicit side-based regions.
6. Use the structured backend only for simple coarse tiling; it does not perform true local refinement.

Notes:

- Constraint polygons are now interpreted as real local sizing controls in the Gmsh path.
- Multipart QGIS topology layers are supported.
- The CUDA solver path is the primary runtime target for SWE2D.

Recommended SWE2D wet/dry stability controls in the workbench:

- `Shallow-front recon fallback`: optional localized first-order fallback for
	shallow edge pairs near wetting fronts.
- `Front flux damping`: momentum-flux damping at wet/dry interfaces.
- `Active-set hysteresis`: reduces rapid wet/dry activation toggling.
- `Shallow damping depth` and `Max rel depth increase`: additional robustness
	controls for thin advancing fronts.

Experimental 3D patch projection controls:

- `3D projection residual sample stride`: Jacobi iterations between residual
	checks (maps to `BACKWATER_SWE3D_PROJECTION_RESIDUAL_SAMPLE_ITERS`).
- `3D projection divergence gate`: enables divergence-quality gating during
	projection retry flow (maps to `BACKWATER_SWE3D_PROJECTION_DIVERGENCE_GATE_ENABLE`).
- `3D projection divergence ratio target`: maximum allowed divergence RMS ratio
	for accepting a projection attempt (maps to `BACKWATER_SWE3D_PROJECTION_DIVERGENCE_RATIO_TARGET`).

When Experimental 3D mode is active, run logs include a `3D projection controls`
line so the active projection gate/target settings are visible in the run record.

## 11. Drainage Network & Culvert Coupling

The 2D SWE workbench supports a 1D drainage network (pipe network) that
exchanges flow with the 2D surface via inlets, outfalls, and pipe-end
exchange objects. Structure culverts (HDS-5 FHWA hydraulics) are coupled
through a face-flux mechanism.

### Drainage Link Types

Each drainage link has a `link_type` field set via a value-map dropdown:

| Type | Value | Description |
|---|---|---|
| Conduit | `conduit` | Circular pipe flow (orifice + Manning capacity) |
| Short lateral | `lateral_simple` | Simplified algebraic flow (estimate only) |
| Pump | `pump` | Fixed-flow pump link |
| Weir | `weir` | Broad-crested weir link |
| Orifice | `orifice` | Orifice link |
| **Culvert (HDS-5)** | `culvert` | FHWA HEC-5 inlet/outlet control hydraulics |

### Culvert Drainage Links (`link_type = "culvert"`)

When a drainage link is set to `"culvert"`, the network solver computes
flow using the same FHWA HDS-5 methodology as the HydraulicStructure
culvert path: inlet control, outlet control (secant solve), orifice
capacity, and Manning full-flow capacity — the minimum of all four
controls determines the link flow.

Required fields for culvert links:

- `link_id`, `from_node`, `to_node` — connectivity
- `length` — barrel length (model units)
- `diameter` or `culvert_rise`/`culvert_span` — cross-section dimensions
- `roughness_n` — Manning's n
- `culvert_shape` — `circular`, `box`, `rectangular`, `pipe_arch`

Optional fields:

- `inlet_invert_elev`, `outlet_invert_elev` — invert elevations (defaults
  to connected-node invert)
- `entrance_loss_k` — Ke entrance loss coefficient (default 0.5)
- `exit_loss_k` — Kx exit loss coefficient (default 1.0)
- `culvert_barrels` — number of barrels (default 1)
- `culvert_code` — FHWA culvert code (default 1)
- `max_flow` — total flow cap

### Pipe-End Exchange

Culvert links that connect to a `pipe_end`-type drainage node
automatically generate a `PipeEndExchange` object, enabling two-way
flow exchange between the 1D network and the co-located 2D surface
cell. This is the same mechanism used for daylighted pipe ends.

### Structure Culverts (Face-Flux)

HydraulicStructure culverts use face-flux coupling by default
(`culvert_face_flux_mode = "face_flux"`). In this mode, culvert mass
is applied through face-based flux arrays (`d_ext_struct_flux_h/hu/hv`)
rather than the cell-center source kernel, providing FVM-consistent
mass+momentum transfer at the culvert inlet/outlet faces.

When the drainage network is also active, the coupling controller
automatically falls back to the fused CUDA path (or CPU path) that
handles both drainage source terms and structure flows together,
preventing the drainage contribution from being silently dropped.

### Coexistence of Drainage + Face-Flux Culverts

Running a model with both the 1D drainage network and face-flux
structure culverts is fully supported:

| Code Path | Drainage Handled? | Structures Handled? |
|---|---|---|
| CPU | ✓ surface exchange | ✓ Python HDS-5 |
| CUDA (fused) | ✓ inlet_cell/inlet_flow | ✓ GPU source kernel |
| CUDA (persistent, drainage=None) | N/A | ✓ face-flux kernel |

The persistent GPU path is automatically disabled when drainage is
present, ensuring the fused path (which accepts drainage arrays) is
used.

## 11a. Save a Copy of the Model

Use:

- **HYDRA > Save Model GeoPackage As...**

This writes the current model to a selected GeoPackage path.

## 12. Troubleshooting

### Run blocked by unsaved edits

- Save with **HYDRA > Save Layer Edits**
- Re-run the model

### "GeoPackage required" warnings

- The plugin workflow supports `*.gpkg` models
- Load or create a GeoPackage model first

### Missing plots / matplotlib warnings

- Install matplotlib in the QGIS Python environment
- Restart QGIS and rerun

### SciPy option selected but unavailable

- Install SciPy in the QGIS Python environment
- If unavailable, use solver option `py`

### Centerline-related load errors

- Confirm `centerline` exists and has at least one valid feature

## 13. Recommended Workflow

1. Create or load a GeoPackage model
2. Enable layer editing
3. Update cross sections, centerline, and boundary conditions
4. Save layer edits
5. Choose Solver and Alpha Method
6. Run model
7. Review plot and table
8. Save model copy if needed

For 2D runs, add the topology layers first, then choose Gmsh when local refinement matters.
