# HYDRA — GPU-Accelerated 2D Shallow Water Equation Plugin for QGIS

**Version**: 2.0 (GPU-Only)
**Last Updated**: 2026-06-09

---

## Table of Contents

1. [Overview](#1-overview)
2. [Requirements & Dependencies](#2-requirements--dependencies)
3. [Installation](#3-installation)
4. [Plugin Capabilities](#4-plugin-capabilities)
5. [Model Setup & Preprocessing](#5-model-setup--preprocessing)
6. [Running the Solver](#6-running-the-solver)
7. [Postprocessing & Results](#7-postprocessing--results)
8. [Technical Reference: Hydraulic Theory](#8-technical-reference-hydraulic-theory)
9. [API Reference](#9-api-reference)
10. [Troubleshooting](#10-troubleshooting)
11. [References](#11-references)

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

## 2. Requirements & Dependencies

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
| `h5py` | ❌ | HDF5 result export |
| `netCDF4` | ❌ | UGRID NetCDF export |
| `shapely` | ❌ | Geometry operations for BC polyline sampling |

### C++ Dependencies (bundled)

| Component | Purpose |
|---|---|
| **pybind11** (2.13.6+) | Python ↔ C++ bindings (auto-fetched by CMake) |
| **GMsh 4.x** | Optional mesh generation backend |
| **TQMesh** | Optional quadrilateral mesh generation |

---

## 3. Installation

### 3.1 Build the Native Module

```bash
# Clone the repository
git clone https://github.com/user/qgis-backwater-plugin-GPU_ONLY.git
cd qgis-backwater-plugin-GPU_ONLY

# Create build directory
mkdir build && cd build

# Configure with CUDA (requires CUDA toolkit on PATH)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)
```

The build produces:
- `hydra_swe2d.cpython-312-x86_64-linux-gnu.so` — GPU solver module
- `hydra_native.so` — 1D backwater solver module
- `hydra_meshing_native.so` — Mesh generation kernels
- `hydra_overlay.so` — High-performance rendering overlay

### 3.2 Install as QGIS Plugin

```bash
# From QGIS Plugin Manager:
#   1. Open QGIS → Plugins → Manage and Install Plugins
#   2. Click "Install from ZIP"
# 3. Select the plugin archive or point to the repository root

# Or symlink into QGIS plugin directory:
ln -s /path/to/qgis-backwater-plugin-GPU_ONLY \
  ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/backwater_2d
```

### 3.3 Verify Installation

```python
from swe2d.runtime.backend import swe2d_gpu_available
print(f"GPU available: {swe2d_gpu_available()}")
```

---

## 4. Plugin Capabilities

### 4.1 Solver Features

| Feature | Options |
|---|---|
| **Spatial schemes** | First-order, MUSCL (Fast/MinMod/MC/Van Leer), WENO5 |
| **Temporal schemes** | Euler (RK1), RK2 (Heun), RK4, Graph-safe RK4, RK5 (Cash-Karp) |
| **Bed friction** | Manning, Chezy, Darcy-Weisbach, Nikuradse |
| **Turbulence** | None (laminar); Smagorinsky/K-ε/K-ω SST (skeleton) |
| **Mesh support** | Triangles, quads, general polygons (CSR storage) |
| **GPU acceleration** | Full CUDA path; CUDA graph caching for small timesteps |
| **Adaptive timestepping** | CFL-based with per-step safety factor |
| **Stability hardening** | Momentum cap, depth cap, front flux damping, shallow damping, active-set hysteresis |
| **Degenerate cell handling** | Skip, repair (neighbor average), merge (flux redirect) |

### 4.2 Meshing Capabilities

| Backend | Cell Types | Features |
|---|---|---|
| **Gmsh** (default) | Triangles, quads | Transfinite zones, breakline embedding, constraint polygons, quality loop |
| **TQMesh** | Quads (via tri→quad) | 4-side polylines, quad-oriented splitting |
| **Structured** | Triangles | Deterministic Cartesian grid, fast fallback |

### 4.3 Boundary Conditions

| Code | Type | Description |
|---|---|---|
| 1 | Wall | Zero normal flux (reflective) |
| 2 | Inflow Q | Prescribed discharge per edge [L²/T] |
| 3 | Stage | Prescribed water surface elevation [L] |
| 4 | Open | Characteristic/Riemann extrapolation |
| 5 | Reflecting | Slip boundary (symmetry) |
| 6 | Normal Depth | Manning equilibrium depth (Sf-based) |
| 102 | Timeseries Flow Q | Hydrograph-driven discharge |
| 103 | Timeseries Stage | Hydrograph-driven WSE |

### 4.4 Drainage Network

**Node types**: Junction, Outfall, Storage, Inlet, Pipe end
**Link types**: Conduit, Short lateral, Pump, Weir, Orifice, Culvert (HDS-5)
**Solver modes**: EGL (energy grade line), Diffusion wave, Dynamic wave (Saint-Venant)

### 4.5 Hydraulic Structures

| Type | Model | Key Parameters |
|---|---|---|
| **Weir** | Broad-crested $Q = C w h^{3/2}$ | crest_elev, width, coeff |
| **Culvert** | FHWA HDS-5 (inlet/outlet/orifice control) | shape, code, rise, span, length, barrels |
| **Gate** | Orifice equation | crest_elev, opening, coeff |
| **Bridge** | Deck-submerged flow limiting | deck_soffit_elev, influence_width, embankment |
| **Pump** | Constant or rated discharge | q_pump, enabled |

### 4.6 Rainfall & Infiltration

- Rain-on-grid with Thiessen polygon or raster-based distribution
- SCS Curve Number (CN) infiltration: $P_e = \max(0, P - I_a)^2 / (P - I_a + S)$
- Configurable initial abstraction ratio ($I_a = 0.2 \times S$)
- Internal flow sources (subsurface or point injection)

### 4.7 Results & Export

| Format | Content | Access |
|---|---|---|
| **GeoPackage** | Run logs, mesh results, sample lines, coupling diagnostics | QGIS attribute tables |
| **HEC-RAS HDF5** | Mesh + results for cross-software compatibility | QGIS Mesh layer |
| **UGRID NetCDF** | Standard mesh/results format | QGIS Mesh layer (MDAL) |
| **GeoTIFF** | Rasterized overlay fields | QGIS Raster layer |
| **CSV** | Sample line timeseries/profile export | Spreadsheet tools |

---

## 5. Model Setup & Preprocessing

### 5.1 Creating a Model GeoPackage

1. Open the **SWE2D Workbench** (Plugins → HYDRA → Open Workbench)
2. Navigate to the **Map** tab → **Model files and mesh actions** group
3. Click **Create 2D Model GeoPackage**
4. Select save location; this creates a `.gpkg` with all model table templates:
   - `swe2d_topo_nodes`, `swe2d_topo_arcs`, `swe2d_topo_regions`
   - `swe2d_topo_constraints`, `swe2d_topo_quad_edges`
   - `swe2d_manning_zones`, `swe2d_bc_lines`, `swe2d_sample_lines`
   - `swe2d_drainage_nodes`, `swe2d_drainage_links`, `swe2d_structures`

### 5.2 Topology Layer Setup

1. Navigate to the **Topology** tab
2. Create topology template layers (click **Create Topology Template Layers**)
3. Edit **Regions layer** — define domain boundary polygons:
   - One polygon per mesh block
   - Set `target_size` (element edge length) per region
   - Set `cell_type`: `triangular`, `quadrilateral`, `cartesian`, or `empty` (hole)
   - Interior rings create hole cutouts
4. Optionally add **Arcs** for boundary-aligned meshing
5. Optionally add **Constraints** for local refinement (polygon = size field)
6. Optionally add **Quad Edges** for Gmsh transition spacing

### 5.3 Mesh Generation

1. Select your regions layer in the Topology tab
2. Set **Meshing backend**: Gmsh (recommended) or structured
3. Configure **Default target size** and **Default cell type**
4. Click **Generate Mesh From Topology Layers**
5. Monitor progress in the status label and runtime log

### 5.4 Boundary Condition Assignment

1. Navigate to the **Boundary** tab
2. For each side (left/right/bottom/top):
   - Select **BC type** from dropdown
   - Enter **BC value** (e.g., flow rate for Inflow Q, elevation for Stage)
   - Enter **Hydrograph** text if using timeseries BCs
3. Optionally create a **BC polyline layer** for per-edge BC overrides
4. Click **Preview Overrides** to verify BC assignment

### 5.5 Terrain Assignment

**Option A — Direct mesh sampling (recommended)**:
1. Select a terrain raster in the **Map** tab
2. Click **Assign Mesh Node Z From Terrain**
3. The raster is sampled directly onto mesh node coordinates

**Option B — Nodes layer workflow**:
1. Export mesh to **SWE2D_Mesh_Nodes** layer
2. Edit `bed_z` values in QGIS
3. Click **Pull Mesh Node Z From Nodes Layer**

### 5.6 Model Parameters

Navigate to the **Model** tab → **Solver Parameters**:

| Parameter | Default | Description |
|---|---|---|
| Manning's n | 0.03 | Global bed roughness |
| CFL | 0.45 | Courant–Friedrichs–Lewy safety factor |
| h_min | 1e-6 | Wet/dry threshold depth |
| Spatial scheme | First-order | Reconstruction/limiter choice |
| Temporal scheme | RK2 | Time integration method |
| dt_fixed | -1 | Fixed timestep (-1 = adaptive) |
| Initial condition | Dry | Dry bed or uniform depth/WSE |

### 5.7 Drainage Network Setup

1. Create/edit drainage layers:
   - `swe2d_drainage_nodes` — point features (node_id, invert_elev, node_type)
   - `swe2d_drainage_links` — line features (link_id, from_node, to_node, link_type)
   - `swe2d_drainage_inlets` — table (inlet_type_id, weir_length, coeff_weir)
   - `swe2d_drainage_node_inlets` — table (node_id, inlet_type_id)
2. Select layers in the **Map** tab
3. Configure solver mode (EGL/Diffusion/Dynamic) in Model → Drainage

### 5.8 Hydraulic Structures Setup

1. Create/edit `swe2d_structures` layer:
   - `structure_type`: 1=Weir, 2=Culvert, 3=Gate, 4=Bridge, 5=Pump
   - `crest_elev`, `width`/`height`/`diameter`, `enabled`
   - For culverts: `culvert_code` (FHWA code), `culvert_rise`, `culvert_span`, `length`
2. Select layer in Map tab

---

## 6. Running the Solver

### 6.1 Quick Run

1. Configure parameters in **Model** tab
2. Navigate to **Run** tab
3. Set **Run duration** (e.g., `01:00` for 1 hour)
4. Set **Output interval** (e.g., `00:30` for 30-minute snapshots)
5. Click **Run 2D Model**
6. Monitor progress in the **Runtime Log** panel (right side)

### 6.2 Advanced Options

| Control | Location | Description |
|---|---|---|
| Adaptive CFL | Model → Solver | Auto-adjusts dt based on CFL number |
| CUDA Graphs | Model → Solver | Cache kernel graphs for repeated small dt |
| Rain-on-grid | Model → Rain/Hydrology | Enable rainfall + CN infiltration |
| Drainage coupling | Model → Structures & Drainage | 1D–2D surface coupling |
| Extended outputs | Run → Outputs | Save mesh results, sample lines, coupling results to GeoPackage |

### 6.3 Cancelling a Run

Click **Cancel** in the Run tab. The solver completes the current timestep, writes a partial snapshot, and exits gracefully.

---

## 7. Postprocessing & Results

### 7.1 Sample Lines

1. Draw a sample line on the map (Map tab → **Draw Sample Line On Map**)
2. After a run, results at the line are computed automatically
3. Click **Open Line Results Viewer** to view timeseries and profiles

### 7.2 Results Panel

- Click **Show Results Panel** (Run tab) for a dockable multi-timestep viewer
- Scrub through timesteps, plot depth/velocity/WSE along sample lines

### 7.3 High-Performance Overlay

- Enable the **High-Perf Canvas Overlay** in the Results tab for real-time depth/velocity colormaps rendered directly on the QGIS map canvas
- Supports velocity arrows and streamline visualization

### 7.4 GeoPackage Explorer

- Click **Open Model GeoPackage Explorer** (Map tab) to browse all result tables
- Preview, rename, or delete model tables

### 7.5 Exporting Results

| Action | Location |
|---|---|
| Save Mesh to HEC-RAS HDF5 | Map tab → **Save Mesh To HEC-RAS HDF5** |
| Save Results to HDF5 | Map tab → **Save Results To HEC-RAS HDF5** |
| Save Results to UGRID NetCDF | Map tab → **Save Results To UGRID NetCDF** |
| Export Overlay to GeoTIFF | Results overlay controls → **Export GeoTIFF** |
| Export Line CSV | Line Results Viewer → **Export Table CSV** |
| Take Snapshot | Run tab → **Take Snapshot** (writes HDF + GeoPackage results) |

---

## 8. Technical Reference: Hydraulic Theory

### 8.1 Shallow Water Equations

The 2D SWE in conservative form:

$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{U}) = \mathbf{S}$$

where $\mathbf{U} = (h, hu, hv)^T$ is the state vector of depth and momentum, and:

$$\mathbf{F} = \begin{pmatrix} hu & hu^2 + \frac{1}{2}gh^2 & hvu \\ hv & huv & hv^2 + \frac{1}{2}gh^2 \end{pmatrix}$$

Source terms $\mathbf{S}$ include bed slope, friction, rainfall, and external forcing.

### 8.2 Spatial Discretization

The solver uses a cell-centered finite-volume method on unstructured meshes:

$$\frac{d\mathbf{U}_i}{dt} = -\frac{1}{|V_i|} \sum_{j \in \mathcal{F}(i)} \mathbf{F}_{ij} \cdot \mathbf{n}_{ij} \, \Delta l_{ij} + \mathbf{S}_i$$

**Reconstruction schemes**:

| Scheme | Method | Limiter |
|---|---|---|
| First-order (0) | Piecewise constant | None |
| MUSCL Fast (1) | Linear gradient | Superbee TVD |
| MUSCL MinMod (2) | Linear gradient | Minmod (most diffusive) |
| MUSCL MC (3) | Linear gradient | Monotonized-Central |
| MUSCL Van Leer (4) | Linear gradient | Van Leer smooth TVD |
| WENO5 (6) | Nonlinear WENO weights | WENO5 + 2-ring LSQ gradient |

### 8.3 Numerical Flux

The normal flux at each face is computed via an approximate Riemann solver. For SWE, the flux uses the Rusanov (scalar dissipation) or HLL/HLLC family:

$$\mathbf{F}_{ij} = \frac{1}{2}\left[\mathbf{F}(\mathbf{U}_L) + \mathbf{F}(\mathbf{U}_R) - |\hat{a}|(\mathbf{U}_R - \mathbf{U}_L)\right]$$

where $\hat{a} = \max(|u_L| + \sqrt{gh_L},\; |u_R| + \sqrt{gh_R})$ is the maximum wave speed estimate.

### 8.4 Temporal Integration

| Method | Stages | Order | Stability Region |
|---|---|---|---|
| Euler (RK1) | 1 | 1 | Small |
| Heun (RK2) | 2 | 2 | Moderate |
| RK4 | 4 | 4 | Large |
| Graph-safe RK4 | 4 | 4 | Graph-compatible staging |
| Cash-Karp RK5 | 5 | 5 | Very large |

SSP (Strong Stability Preserving) property is maintained for RK2 and RK4 variants.

### 8.5 Bed Friction

The Manning equation provides the friction source term:

$$S_f = -g \frac{n^2}{h^{1/3}} |V| V$$

where $n$ is Manning's roughness, $h$ is water depth, and $V = (u, v)$ is the velocity vector. The friction source is treated implicitly via sub-stepping to maintain stability at high roughness values.

For USC units, the Manning multiplier $k_m = 1.486$ is applied: $V = (k_m / n) R_h^{2/3} S^{1/2}$.

### 8.6 Wet/Dry Front Treatment

The solver implements several stability mechanisms near wet/dry interfaces:

- **Active-set hysteresis**: Cells that become wet are kept active for 1 extra step
- **Front flux damping**: Momentum flux is damped as depth approaches $h_{\min}$
- **Shallow front reconstruction fallback**: Forces first-order reconstruction when $h < h_{\text{threshold}}$
- **Depth cap**: Hard upper bound on depth per timestep to prevent unphysical spikes

### 8.7 Rainfall Infiltration (SCS Curve Number)

The SCS CN method computes excess rainfall:

$$S = \frac{25400}{CN} - 254 \quad [\text{mm}]$$

$$I_a = \alpha \cdot S \quad (\alpha = 0.2 \text{ standard})$$

$$P_e = \frac{(P - I_a)^2}{P - I_a + S} \quad \text{for } P > I_a$$

where $CN$ is the curve number (0–100), $S$ is the potential maximum retention, and $P$ is cumulative precipitation.

### 8.8 Drainage Network Equations

#### Energy Grade Line (EGL) — Default

$$H_1 = H_2 + h_L$$

where $H = z + d + V^2/(2g)$ is the total head and $h_L$ includes friction + entrance/exit losses.

#### Diffusion Wave

$$\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = q_\ell$$

$$Q = -\frac{k}{n} A R_h^{2/3} \sqrt{S_f}$$

Friction slope $S_f$ is computed from the water surface gradient; no inertial terms.

#### Dynamic Wave (Saint-Venant 1D)

$$\frac{\partial A}{\partial t} + \frac{\partial Q}{\partial x} = q_\ell$$

$$\frac{\partial Q}{\partial t} + \frac{\partial}{\partial x}\left(\frac{Q^2}{A}\right) + gA\frac{\partial d}{\partial x} = -gA S_f + gA S_0$$

Solved semi-implicitly with predictor-corrector coupling iterations.

### 8.9 Culvert Hydraulics (FHWA HDS-5)

The culvert solver implements five control modes, selecting the minimum flow:

1. **Inlet control**: $Q = C_d A \sqrt{2g \Delta H}$ where $C_d$ depends on inlet geometry (FHWA nomograph)
2. **Outlet control**: Bernoulli equation with entrance loss $K_e$, friction (Manning), and exit loss $K_x$
3. **Orifice control**: Submerged orifice: $Q = C_d A_o \sqrt{2g(H_{us} - H_{ds})}$
4. **Manning capacity**: Friction-limited flow: $Q_{\max} = (k_m/n) A R_h^{2/3} S_0^{1/2}$
5. **Max flow cap**: User-specified upper bound

All dimensions are converted to feet internally (HDS-5 tables are in USC), computed, then converted back to model units.

### 8.10 Weir Discharge

Broad-crested weir equation:

$$Q = C_w \cdot L \cdot H^{3/2}$$

where $C_w$ is the weir coefficient (default 1.7 for SI), $L$ is effective weir length, and $H$ is head above crest.

### 8.11 Unit System

All calculations are performed in **model units** (derived from the QGIS project CRS):

| Quantity | SI | USC (feet) |
|---|---|---|
| Gravity | 9.81 m/s² | 32.17 ft/s² |
| Manning multiplier | 1.0 | 1.486 |
| Length scale | 1.0 m/m | 0.3048 m/ft |
| Model→feet (culvert) | 3.28084 | 1.0 |

---

## 9. API Reference

### 9.1 Core Classes

#### `SWE2DBackend`

```python
class SWE2DBackend:
    """GPU-accelerated 2D SWE solver backend."""

    def __init__(self, use_gpu: bool = True):
        """Initialize backend.

        Args:
            use_gpu: If True, use CUDA GPU path (required for GPU-only build).
        """

    def build_mesh(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        node_z: np.ndarray,
        cell_nodes: np.ndarray,
        cell_face_offsets: Optional[np.ndarray] = None,
        cell_face_nodes: Optional[np.ndarray] = None,
    ) -> None:
        """Build mesh and upload to GPU.

        Args:
            node_x, node_y: Node coordinates [L].
            node_z: Node bed elevations [L].
            cell_nodes: Connectivity array (flat, 3 nodes per tri).
            cell_face_offsets: Per-cell face ring offsets (quads/polygons).
            cell_face_nodes: Face ring node indices.
        """

    def initialize(
        self,
        h0: np.ndarray,
        n_mann: float = 0.03,
        cfl: float = 0.45,
        g: float = 9.81,
        k_mann: float = 1.0,
        h_min: float = 1.0e-6,
        dt_max: float = 10.0,
        dt_fixed: float = -1.0,
        temporal_order: int = 2,
        spatial_scheme: int = 0,
        ...
    ) -> None:
        """Initialize solver state on GPU.

        Args:
            h0: Initial depth array [L].
            n_mann: Global Manning's n.
            cfl: CFL safety factor.
            g: Gravitational acceleration [L/T²].
            k_mann: Manning multiplier (1.0 SI, 1.486 USC).
            h_min: Wet/dry threshold depth [L].
            dt_max: Maximum timestep [s].
            dt_fixed: Fixed timestep (-1 = adaptive).
            temporal_order: Time integration order (1/2/4/5/6).
            spatial_scheme: Spatial reconstruction (0-6).
        """

    def step(self, dt_request: float = -1.0) -> SWE2DStepDiag:
        """Advance one timestep.

        Args:
            dt_request: Requested dt (-1 = CFL adaptive, >0 = fixed).

        Returns:
            SWE2DStepDiag with dt, max_courant, wet_cells, etc.
        """

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current state arrays.

        Returns:
            (h, hu, hv) — depth and momentum components.
        """

    def set_boundary_conditions(
        self,
        edge_n0: np.ndarray,
        edge_n1: np.ndarray,
        bc_type: np.ndarray,
        bc_val: np.ndarray,
    ) -> None:
        """Set static boundary conditions on edges.

        Args:
            edge_n0, edge_n1: Edge node indices.
            bc_type: BC type codes per edge (1=wall, 2=inflow, 3=stage, etc.).
            bc_val: BC values per edge (discharge [L²/T] or elevation [L]).
        """

    def set_boundary_hydrographs_native(
        self,
        edge_ids: np.ndarray,
        hydrograph_times: np.ndarray,
        hydrograph_values: np.ndarray,
        bc_codes: np.ndarray,
    ) -> None:
        """Upload hydrograph time-series for boundary edges to GPU.

        Args:
            edge_ids: Edge indices with hydrographs.
            hydrograph_times: Time values [s].
            hydrograph_values: Corresponding values (Q [L²/T] or WSE [L]).
            bc_codes: BC type for each edge (102=flow, 103=stage).
        """

    def set_rain_cn_forcing_native(
        self,
        cell_gage_idx: np.ndarray,
        gage_offsets: np.ndarray,
        hg_time_s: np.ndarray,
        hg_cum_mm: np.ndarray,
        cn_values: np.ndarray,
    ) -> None:
        """Upload rainfall/CN forcing to GPU.

        Args:
            cell_gage_idx: Gage index per cell.
            gage_offsets: Cumulative offset per gage.
            hg_time_s: Hyetograph time [s].
            hg_cum_mm: Cumulative rainfall [mm].
            cn_values: Curve number per cell.
        """

    def set_external_sources_native(
        self,
        source_mps: np.ndarray,
    ) -> None:
        """Set per-cell external source terms [m/s].

        Args:
            source_mps: Source rate per cell (positive = addition).
        """
```

#### `SWE2DCouplingController`

```python
class SWE2DCouplingController:
    """Surface–drainage–structure coupling controller."""

    def __init__(
        self,
        cell_area: Sequence[float],
        cell_bed: Sequence[float],
        drainage: Optional[SWE2DUrbanDrainageModule] = None,
        structures: Optional[SWE2DStructureModule] = None,
        coupling_loop: str = "cuda",
        drainage_solver_backend: str = "gpu",
        culvert_solver_mode: int = 0,
        length_scale_si_to_model: float = 1.0,
        ...
    ):
        """Initialize coupling controller.

        Args:
            cell_area: Cell areas [L²].
            cell_bed: Cell bed elevations [L].
            drainage: 1D drainage module (or None for no coupling).
            structures: Hydraulic structures module (or None).
            coupling_loop: "cuda" for GPU coupling path.
            drainage_solver_backend: "gpu" for GPU drainage step.
            length_scale_si_to_model: SI meters per model unit.
        """

    def set_cell_centroids(
        self,
        cx: np.ndarray,
        cy: np.ndarray,
    ) -> None:
        """Set cell centroids for structure influence-width weighting."""

    def compute_source_rates(
        self,
        t_s: float,
        dt_s: float,
        h: np.ndarray,
        hu: np.ndarray,
        hv: np.ndarray,
        depth_sources: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute coupled source terms.

        Args:
            t_s: Current time [s].
            dt_s: Timestep [s].
            h, hu, hv: Current state arrays.
            depth_sources: Optional depth source [L] per cell.

        Returns:
            source_mps: Depth source rate per cell [L/T].
        """

    @property
    def last_diag(self) -> SWE2DCouplingDiagnostics:
        """Last step diagnostics (drainage flows, structure flows, etc.)."""
```

#### `SWE2DUrbanDrainageModule`

```python
class SWE2DUrbanDrainageModule:
    """1D pipe network solver (SWMM-style)."""

    def __init__(self, cfg: PipeNetworkConfig):
        """Initialize with network configuration."""

    def initialize(self) -> None:
        """Compute geometry tables and set initial state."""

    def step(
        self,
        dt_s: float,
        cell_wse_2d: Optional[np.ndarray] = None,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Advance 1D network one timestep.

        Args:
            dt_s: Timestep [s].
            cell_wse_2d: WSE per 2D cell for surface coupling.

        Returns:
            (sinks, sources): List of (cell_id, flow [m³/s]) pairs.
        """
```

#### `SWE2DStructureModule`

```python
class SWE2DStructureModule:
    """Hydraulic structure evaluator (HDS-5 culverts, weirs, etc.)."""

    def __init__(
        self,
        cfg: HydraulicStructureConfig,
        model_to_ft: float = 1.0,
    ):
        """Initialize with structure configuration.

        Args:
            cfg: Structure configuration.
            model_to_ft: Model units to feet conversion (for HDS-5).
        """

    def structure_flows(
        self,
        cell_wse: np.ndarray,
        dt_s: float,
    ) -> List[float]:
        """Compute flow through each structure.

        Args:
            cell_wse: Water surface elevation per cell [L].
            dt_s: Timestep [s].

        Returns:
            flows: Flow rate per structure [model L³/T].
        """

    def structure_details(
        self,
        cell_wse: np.ndarray,
    ) -> List[Dict[str, object]]:
        """Get detailed structure diagnostics per structure."""

    @property
    def cfg(self) -> HydraulicStructureConfig:
        """Structure configuration."""
```

### 9.2 Configuration Dataclasses

#### `SolverModelOptions`

```python
@dataclass
class SolverModelOptions:
    temporal_scheme: TemporalScheme = TemporalScheme.SSP_RK2
    spatial_discretization: SpatialDiscretization = SpatialDiscretization.FV_FIRST_ORDER
    turbulence_model: TurbulenceModel = TurbulenceModel.NONE
    bed_friction_model: BedFrictionModel = BedFrictionModel.MANNING
    equation_set: SWE2DEquationSet = SWE2DEquationSet.HYDROSTATIC_2D
    godunov_mode: GodunovSolverMode = GodunovSolverMode.RUSANOV
    rain: Optional[RainFieldConfig] = None
    pipe_network: Optional[PipeNetworkConfig] = None
    hydraulic_structures: Optional[HydraulicStructureConfig] = None
```

#### `PipeNetworkConfig`

```python
@dataclass
class PipeNetworkConfig:
    enabled: bool = False
    nodes: List[DrainageNode] = field(default_factory=list)
    links: List[DrainageLink] = field(default_factory=list)
    inlets: List[InletExchange] = field(default_factory=list)
    outfalls: List[OutfallExchange] = field(default_factory=list)
    pipe_ends: List[PipeEndExchange] = field(default_factory=list)
    solver_mode: DrainageSolverMode = DrainageSolverMode.EGL
    coupling_substeps: int = 1
    max_coupling_substeps: int = 64
    head_deadband_m: float = 1.0e-3
    adaptive_depth_fraction: float = 0.2
    adaptive_wave_courant: float = 0.5
    implicit_coupling_iterations: int = 2
    implicit_coupling_relaxation: float = 0.5
    dynamic_flow_relaxation: float = 1.0
```

#### `HydraulicStructureConfig`

```python
@dataclass
class HydraulicStructureConfig:
    enabled: bool = False
    structures: List[HydraulicStructure] = field(default_factory=list)
    control_interval_s: float = 1.0
    gravity: float = 9.81
```

#### `HydraulicStructure`

```python
@dataclass
class HydraulicStructure:
    structure_id: str
    structure_type: StructureType
    upstream_cell: int
    downstream_cell: int
    crest_elev: float
    enabled: bool = True
    metadata: Dict[str, float] = field(default_factory=dict)
```

#### `DrainageNode`

```python
@dataclass
class DrainageNode:
    node_id: str
    node_type: str  # "junction", "outfall", "storage", "inlet", "pipe_end"
    invert_elev: float
    max_depth: float = 0.0
    rim_elev: float = 0.0
    cell_id: int = -1
    surface_area: float = 0.0
```

#### `DrainageLink`

```python
@dataclass
class DrainageLink:
    link_id: str
    from_node_id: str
    to_node_id: str
    link_type: str  # "conduit", "lateral_simple", "pump", "weir", "orifice", "culvert"
    link_shape: str = "circular"
    length: float = 0.0
    roughness_n: float = 0.013
    diameter: float = 0.0
    span: float = 0.0
    rise: float = 0.0
    culvert_code: int = 1
    culvert_barrels: int = 1
    inlet_invert_elev: float = 0.0
    outlet_invert_elev: float = 0.0
```

### 9.3 Enumerations

| Enum | Values |
|---|---|
| `SpatialDiscretization` | `FV_FIRST_ORDER` (0), `FV_MUSCL_FAST` (1), `FV_MUSCL_MINMOD` (2), `FV_MUSCL_MC` (3), `FV_MUSCL_VAN_LEER` (4), `FV_WENO5` (6) |
| `TemporalScheme` | `EULER_1ST` (1), `SSP_RK2` (2), `GRAPH_SAFE_RK4` (5), `GRAPH_SAFE_RK5` (6) |
| `BedFrictionModel` | `MANNING` (0), `CHEZY` (1), `DARCY_WEISBACH` (2), `NIKURADSE` (3) |
| `TurbulenceModel` | `NONE` (0), `SMAGORINSKY` (1), `K_EPSILON` (2), `K_OMEGA_SST` (3) |
| `StructureType` | `WEIR` (1), `CULVERT` (2), `GATE` (3), `BRIDGE` (4), `PUMP` (5) |
| `DrainageSolverMode` | `EGL` (0), `DIFFUSION` (1), `DYNAMIC` (2) |
| `SWE2DEquationSet` | `HYDROSTATIC_2D` (0) |
| `GodunovSolverMode` | `RUSANOV` (0), `HLL` (1), `HLLC` (2) |

### 9.4 Utility Functions

```python
def swe2d_gpu_available() -> bool:
    """Check if CUDA GPU solver is available."""
```

---

## 10. Troubleshooting

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

### Performance Tips

- Use **quadrilateral cells** for structured domains — they're ~30% faster than triangles on GPU
- **CUDA Graphs** give 10–20% speedup for small fixed timesteps (disable for RK4+)
- Keep **wet domain** compact — GPU parallelism is limited by the wet cell count
- Set **output interval** to be ≥10× larger than dt to avoid snapshot overhead
- Use **streamline backend = CUDA** for fast velocity visualization

---

## 11. References

- Toro, E. F. *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.
- FHWA. *Hydraulic Design of Highway Culverts* (HDS-5). FHWA-HIF-05-012.
- Akan, A. O. *Urban Stormwater Hydrology*. Technomic Publishing.
- QGIS Documentation: https://docs.qgis.org
- HYDRA GPU Architecture: [docs/SWE2D_GPU_ARCHITECTURE_REPORT.md](SWE2D_GPU_ARCHITECTURE_REPORT.md)
- Godunov FVM Implementation: [docs/GODUNOV_2D_GPU_IMPLEMENTATION_GUIDE.md](GODUNOV_2D_GPU_IMPLEMENTATION_GUIDE.md)

---

*Document generated from HYDRA repository state on 2026-06-09. For the latest API details, see source code and inline docstrings.*
