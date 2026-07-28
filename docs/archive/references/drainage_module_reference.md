---
type: reference
status: complete
created: 2026-07-23
completed: 2026-07-25
---

# Drainage Module Reference

Comprehensive technical reference for the SWE2D urban drainage module, covering architecture, data model, GPU pipeline, and integration with the 2D surface solver.

## Architecture Overview

The drainage module provides 1D pipe network modeling coupled to the 2D shallow-water solver through a unified face mesh. The system follows a **Model-View-Presenter** architecture:

- **Model Layer** (`swe2d/extensions/`, `swe2d/runtime/non_gui_runtime_service.py`): Pure Python data structures and GPU-agnostic business logic
- **Controller Layer** (`swe2d/runtime/coupling.py`): Orchestration of coupling, state management, and source term computation
- **View/Service Layer** (`swe2d/workbench/services/`): QGIS integration, GKG schema, and user interaction

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `DrainageNode` | `extension_models.py` | Node dataclass (junctions, inlets, outfalls, pipe ends) |
| `DrainageLink` | `extension_models.py` | Link dataclass (conduits, culverts, weirs, orifices) |
| `PipeNetworkConfig` | `extension_models.py` | Top-level configuration container |
| `SWE2DDrainageSoA` | `coupling.py` | Structure-of-arrays layout for GPU upload |
| `SWE2DCouplingController` | `coupling.py` | Main coupling orchestration class |
| `pipe1d.cu/h` | `cpp/src/` | CUDA 1D solver implementation |

## Data Model

### DrainageNode

```python
@dataclass
class DrainageNode:
    node_id: str
    x: float, y: float                 # Coordinates
    invert_elev: float                  # Pipe invert elevation
    max_depth: float                    # Maximum storage depth
    surface_area: float = 50.0          # Node surface area [m²]
    node_type: str = "junction"         # junction, outfall, storage, inlet, pipe_end
    rim_elev: Optional[float] = None    # Manhole rim elevation (surcharge overflow)
    # Loss coefficients (override per-link values)
    inlet_loss_k: Optional[float] = None
    outlet_loss_k: Optional[float] = None
    # Outfall BC configuration
    outfall_mode: str = "free"          # free, fixed_wse, stage_discharge
    outfall_fixed_wse: float = 0.0
    outfall_rating_table: list = field(default_factory=list)
```

### DrainageLink

```python
@dataclass
class DrainageLink:
    link_id: str
    from_node_id: str, to_node_id: str   # Connected nodes
    length: float                        # Link length [ft | m]
    roughness_n: float = 0.013           # Manning's n
    diameter: Optional[float] = None     # Circular pipe diameter [ft | m]
    link_shape: str = "circular"          # circular, rectangular, elliptical
    width: Optional[float] = None        # Rectangular width / elliptical span [ft | m]
    height: Optional[float] = None       # Rectangular height / elliptical rise [ft | m]
    max_cell_length: float = 0.0         # Max sub-cell length for mesh refinement
    entrance_loss_k: float = 0.5         # FHWA entrance loss (alias for inlet_loss_k)
    exit_loss_k: float = 1.0             # FHWA exit loss (alias for outlet_loss_k)
```

### Structure-of-Arrays (SoA) Layout

The GPU solver uses a SoA layout for cache-coefficient memory access:

```python
@dataclass
class SWE2DDrainageSoA:
    # Node arrays [n_nodes]
    node_x, node_y                      # Coordinates [ft | m]
    node_invert_elev                    # Invert elevations [ft | m]
    node_max_depth                      # Maximum depth [ft | m]
    node_surface_area                   # Surface area for storage [ft² | m²]
    node_rim_elev                       # Manhole rim elevation [ft | m]
    node_inlet_loss_k, node_outlet_loss_k  # Node-level loss overrides
    node_is_inlet                       # Inlet flag (deprecated, derived from inlet_node)

    # Link arrays [n_links]
    link_from, link_to                  # Node connectivity
    link_length                         # Link length [ft | m]
    link_roughness_n                    # Manning's roughness
    link_diameter                       # Equivalent diameter [ft | m]
    link_inlet_loss_k, link_outlet_loss_k  # Link-level loss coefficients

    # Inlet arrays [n_inlets]
    inlet_cell                          # 2D surface cell index
    inlet_node                          # Drainage node index
    inlet_crest_elev                    # Inlet crest elevation [ft | m]
    inlet_width, inlet_coefficient      # Inlet geometry [ft | m]
    inlet_type                          # HEC-22 inlet type code

    # Pipe-end arrays [n_pipe_ends]
    pipe_end_cell, pipe_end_node        # Surface cell and drainage node
    pipe_end_invert_elev, pipe_end_diameter  # Geometry [ft | m]
```

## GPU Pipeline

### Pipeline Stages

```
1. Mesh Build (once)
   ├─ swe2d_build_unified_mesh()
   ├─ swe2d_pipe1d_upload_pipe_end_surface_faces()
   └─ swe2d_pipe1d_init_cell_area()

2. Per Timestep
   ├─ Stage 0: Predict (explicit RK2)
   │  ├─ swe2d_pipe1d_compute_AQ_slopes()  [MUSCL reconstruction]
   │  ├─ Unified face flux kernel
   │  └─ swe2d_pipe1d_step() [half-step]
   │
   ├─ Stage 1: Correct (explicit RK2)
   │  ├─ swe2d_pipe1d_compute_AQ_slopes()  [recompute slopes]
   │  ├─ Unified face flux kernel
   │  └─ swe2d_pipe1d_step() [full-step]
   │
   └─ Readback
      └─ swe2d_pipe1d_readback_cell_state()
```

### Memory Layout

**Device State** (`Pipe1DDeviceState` in `pipe1d.cuh`):

```cpp
// Per-cell state [n_cells_all]
double* d_A;              // Flow area [ft² | m²]
double* d_Q;              // Flow rate [cfs | m³/s]
double* d_cell_y;         // Water surface elevation [ft | m]
double* d_cell_h;         // Flow depth (invert-relative) [ft | m]
double* d_cell_q;         // Flow velocity (Q/A) [fps | m/s]
double* d_cell_fr;        // Froude number

// Per-cell geometry [n_cells_all]
double* d_cell_length;    // Sub-cell length [ft | m]
double* d_cell_area;      // Cross-sectional area [ft² | m²]
double* d_cell_invert;    // Invert elevation [ft | m]
double* d_cell_width;     // Cross-section width [ft | m]
double* d_cell_height;    // Cross-section height (alias: cell_rise) [ft | m]
double* d_cell_crown;     // Pipe crown elevation (manhole cells) [ft | m]
double* d_cell_rim;       // Manhole rim elevation [ft | m]
double* d_cell_surface_area;  // Storage surface area [ft² | m²]

// MUSCL reconstruction buffers
double* d_slope_A;        // Area gradient
double* d_slope_Q;        // Flow gradient
// Note: d_slope_H removed as dead code (reconstructed from A/T)
```

## Face Classes

The unified face mesh uses integer class codes for dispatch:

| Class | Name | Direction | Use Case |
|-------|------|-----------|----------|
| 0 | PIPE_INTERIOR | Bidirectional | Interior pipe faces |
| 1 | PIPE_UPSTREAM | Forward → Backward | Link upstream end |
| 2 | PIPE_DOWNSTREAM | Forward → Backward | Link downstream end |
| 3 | STORAGE_PIPE | Bidirectional | Manhole → pipe (storage cell) |
| 4 | SURFACE_2D_INLET | Surface → Pipe | HEC-22 inlet capture (one-directional) |
| 5 | SURFACE_2D_JUNCTION_OVERFLOW | Pipe → Surface | Manhole surcharge overflow relief |
| 6 | SURFACE_2D_PIPE_END | Bidirectional | Pipe-end coupling |
| 7 | SURFACE_2D_OUTFALL | Pipe → Surface | Outfall discharge |
| 8 | STORAGE_PIPE (legacy) | Bidirectional | Storage cell (deprecated alias for class 3) |

**Inlet Node Bidirectional Architecture**: Inlet nodes achieve bidirectional behavior through two separate face classes:
- **Class 4** (SURFACE_2D_INLET): Captures surface runoff into the pipe network. Flow is one-directional (surface → pipe only). When surface water surface elevation (WSE) is below the inlet crest elevation, no capture occurs.
- **Class 5** (SURFACE_2D_JUNCTION_OVERFLOW): Provides overflow relief from the pipe network to the surface. When pipe water surface elevation exceeds the manhole rim elevation, excess flow is discharged to the surface.

Inlet nodes are assigned both face class 4 and class 5 faces, enabling the capture → conveyance → overflow cycle.

### Face Flux Computation

Each face computes flux using a Riemann solver adapted for the 1D network:

```
For interior faces (class 0):
    Q_face = minmod_flux(A_left, Q_left, A_right, Q_right, d_slope_A, d_slope_Q)

For boundary faces (classes 1-7):
    Q_face = boundary_flux(class, A, Q, BC_params)
```

## Coupling Protocol

### Source Term Integration

The 2D solver calls the coupling controller each substep:

```python
# Called by 2D solver for each surface cell
source_h, source_hu, source_hv = controller.apply_native_device_sources(
    t, dt, backend
)
```

The controller:
1. Reads 2D cell state (h, hu, hv)
2. Computes drainage exchange (inlets, pipe ends, outfalls)
3. Computes structure exchange (culverts, bridges, weirs)
4. Returns net source terms

### Data Flow

```
2D Surface Solver
    ↓ (cell state)
Coupling Controller
    ├→ Drainage Module (SWE2DCouplingController)
    │   ├→ Pack SoA from DrainageNetwork
    │   ├→ Upload to GPU
    │   ├→ Run pipe1d_step() (CUDA)
    │   └→ Readback cell state
    └→ Structure Module (SWE2DStructureModule)
    ├→ Pack structure geometry
    ├→ Upload to GPU
    └→ Run structure kernels (CUDA)
    ↓ (source terms)
2D Surface Solver (apply sources)
```

### Time Integration

The drainage module supports multiple coupling schemes:

```python
# Configurable via PipeNetworkConfig
coupling_substeps: int = 1              # 1D substeps per 2D call
implicit_coupling_iterations: int = 2  # Predictor-corrector iterations
implicit_coupling_relaxation: float = 0.5  # Under-relaxation factor
```

## Conservation Properties

### Mass Conservation

Mass is conserved through:
- **Flux-form update**: `dA/dt = -dQ/dx` (continuity equation)
- **Face flux matching**: Flux out of cell i equals flux into cell i+1
- **Source term accounting**: All 2D→1D and 1D→2D exchanges are tracked

### Mass Balance Check

```python
# Post-simulation mass balance
initial_mass = sum(A * length)  # Initial pipe volume [ft³ | m³]
final_mass = sum(A * length)    # Final pipe volume [ft³ | m³]
surface_exchange = sum(inlet_flux) - sum(outfall_flux)  # [cfs | m³/s]
expected_change = surface_exchange * dt  # [ft³ | m³]

assert abs(final_mass - initial_mass - expected_change) < tolerance
```

## Results Schema

### Readback State

The `swe2d_pipe1d_readback_cell_state()` binding returns:

```python
{
    # Per-cell state
    "cell_A": np.array,        # Flow area [ft² | m²]
    "cell_Q": np.array,        # Flow rate [cfs | m³/s]
    "cell_velocity": np.array, # Velocity [fps | m/s]
    "cell_depth": np.array,    # Flow depth [ft | m]
    "cell_q": np.array,        # Unit discharge [ft²/s | m²/s]
    "cell_h": np.array,        # Water surface elevation [ft | m]
    "cell_y": np.array,        # Same as cell_h (alias)
    "cell_invert": np.array,   # Invert elevation [ft | m]
    "cell_width": np.array,    # Cross-section width [ft | m]
    "cell_height": np.array,   # Cross-section height [ft | m] (alias: cell_rise)
    "cell_surface_area": np.array,  # Storage surface area [ft² | m²]
    "cell_crown": np.array,    # Pipe crown elevation [ft | m]
    "cell_rim": np.array,      # Manhole rim elevation [ft | m]
    "cell_max_depth": np.array,    # Maximum depth [ft | m]
    "cell_shape_type": np.array,    # Shape type code
    "cell_owner_link": np.array,    # Owning link index
    "cell_sub_idx": np.array,       # Sub-cell index within link
    "cell_class": np.array,         # Cell class (0=pipe, 1=manhole, 2=inlet)

    # Per-link aggregates
    "link_q": np.array,        # Mean |Q| per link [cfs | m³/s]

    # Metadata
    "n_manhole_cells": int,   # Manhole cell count
    "n_inlet_cells": int,     # Inlet cell count
}
```

### Aggregation Services

The `non_gui_runtime_service.py` provides aggregation:

```python
# Sample drainage state at simulation time
results = runtime_service.sample_drainage_state(
    coupling_state,
    time_index
)

# Results include per-link and per-node aggregates
results["link_flow"]      # Mean flow per link
results["node_depth"]     # Depth at each node
results["node_wse"]       # Water surface elevation
```

## Naming Conventions

### Loss Coefficients

The drainage module supports multiple naming conventions:

| Context | Field Names | Notes |
|---------|-------------|-------|
| GPKG Schema | `inlet_loss_k`, `outlet_loss_k` (primary) | Also supports `entrance_loss_k`, `exit_loss_k` (FHWA aliases) |
| Python Model | `inlet_loss_k`, `outlet_loss_k` (DrainageNode) | `entrance_loss_k`, `exit_loss_k` (DrainageLink - FHWA) |
| C++ Kernel | `face_k_in`, `face_k_out` | Internal GPU array names |
| C++ SoA | `link_inlet_loss_k`, `link_outlet_loss_k` | Per-link fallback values |

**Priority**: Node-level overrides → Link-level values → Defaults (0.5, 1.0)

### Geometry Fields

| Field | Meaning | Aliases |
|-------|---------|---------|
| `cell_height` | Cross-section height (vertical dimension) | `cell_rise` |
| `cell_h` | Flow depth (water surface - invert) | - |
| `cell_y` | Water surface elevation (absolute) | Same as `cell_h` + invert |

## Solver Modes

### Drainage Solver Modes

```python
class DrainageSolverMode(IntEnum):
    EGL = 0        # Energy-grade-line (pressure pipe flow)
    DIFFUSION = 1  # Diffusion-wave (Manning gravity flow)
    DYNAMIC = 2    # Full Saint-Venant (momentum equation)
```

- **EGL**: Default for storm drains, models pressure flow using Bernoulli + Manning
- **DIFFUSION**: Good for partially-full gravity sewers and open channels
- **DYNAMIC**: Captures surge, bore propagation, backwater transients

### Spatial Reconstruction

```python
class SpatialDiscretization(IntEnum):
    FV_FIRST_ORDER = 0     # No reconstruction (piecewise constant)
    FV_MUSCL_MINMOD = 2    # MUSCL with minmod limiter (default)
    FV_WENO3 = 6           # Third-order WENO
```

## Performance Considerations

### Memory Footprint

Per-cell memory usage (double precision):
- State arrays: ~16 bytes/cell × 6 arrays = 96 bytes/cell
- Geometry arrays: ~16 bytes/cell × 15 arrays = 240 bytes/cell
- **Total**: ~336 bytes/cell (~3.5 KB per 10 cells)

For a 10,000-cell network: ~3.4 MB GPU memory (~170 KB per 500 cells)

### Computational Cost

Per-timestep cost (rough estimates):
- MUSCL slope computation: O(n_cells) kernel launch
- Face flux computation: O(n_faces) kernel launch
- RK2 update: O(n_cells) kernel launch

Typical throughput: ~1M cells/sec on modern GPU

## Extension Points

### Adding New Face Classes

1. Add class code to `pipe1d.cuh` enum
2. Implement boundary flux function in `pipe1d.cu`
3. Add dispatch logic in unified face kernel
4. Update documentation

### Adding New Link Types

1. Add `link_type` string to `DrainageLink`
2. Implement geometry lookup in `pipe1d_lookup_geometry()`
3. Add loss coefficient logic in face flux kernel
4. Update GPKG schema

### Adding New Inlet Types

1. Add HEC-22 type code to `InletType` enum
2. Implement capture equation in inlet flux kernel
3. Add parameters to `InletExchange` dataclass
4. Update validation tests

## Troubleshooting

### Common Issues

| Issue | Symptoms | Fix |
|-------|----------|-----|
| Mass not conserved | Draining network loses/gains mass | Check face flux boundary conditions, verify `n_manhole_cells`/`n_inlet_cells` stored correctly |
| Instability | Oscillations, NaN values | Reduce `dt`, check CFL condition, verify pipe geometry |
| Wrong depth at nodes | Readback depth doesn't match expectations | Check invert elevations, verify `cell_height` vs `cell_h` confusion |
| Coupling not working | No exchange between 2D and 1D | Verify inlet/pipe-end cell indices, check face class dispatch |

### Debug Mode

Enable verbose logging:

```python
import logging
logging.getLogger("swe2d.runtime.coupling").setLevel(logging.DEBUG)
```

GPU kernel debugging:

```bash
export CUDA_LAUNCH_BLOCKING=1  # Synchronous kernel launches for better error messages
```

## References

- **FHWA HDS-5**: Hydraulic Design of Highway Culverts
- **HEC-22**: Urban Drainage Design Manual
- **SWMM 5.2** Reference Manual (for engine comparisons)
- **Godunov (1959)**: Finite difference method for conservation laws

## Appendix: Field Mapping Tables

### GPKG → Python Model

| GPKG Field | Python Field | Type |
|------------|--------------|------|
| `node_id` | `DrainageNode.node_id` | str |
| `invert_elev` | `DrainageNode.invert_elev` | float |
| `max_depth` | `DrainageNode.max_depth` | float |
| `surface_area` | `DrainageNode.surface_area` | float |
| `inlet_loss_k` | `DrainageNode.inlet_loss_k` | float |
| `outlet_loss_k` | `DrainageNode.outlet_loss_k` | float |
| `entrance_loss_k` | `DrainageLink.entrance_loss_k` | float (alias) |
| `exit_loss_k` | `DrainageLink.exit_loss_k` | float (alias) |

### Python Model → GPU SoA

| Python Field | GPU Array | Shape |
|--------------|-----------|-------|
| `nodes[i].x` | `d_cell_x` (packed) | [n_nodes] |
| `nodes[i].invert_elev` | `d_cell_invert` (packed) | [n_nodes] |
| `links[i].from_node_id` | `link_from` (packed) | [n_links] |
| `links[i].entrance_loss_k` | `d_cell_link_k_in` (broadcast) | [n_pipe_cells] |
| `inlets[i].cell_id` | `inlet_cell` | [n_inlets] |

### GPU SoA → Readback State

| GPU Array | Readback Key | Notes |
|-----------|--------------|-------|
| `d_A` | `cell_A` | Flow area |
| `d_Q` | `cell_Q` | Flow rate |
| `d_cell_y` | `cell_h`, `cell_y` | Water surface elevation (two keys for compatibility) |
| `d_cell_height` | `cell_height` | Cross-section height, NOT flow depth |

---

**Document Version**: 1.0  
**Last Updated**: 2026-07-23  
**Maintainer**: Drainage Module Development Team