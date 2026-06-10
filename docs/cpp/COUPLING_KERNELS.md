# Coupling Kernels (Surface ↔ Drainage ↔ Structures)

## Overview

The coupling layer orchestrates three-way exchange between the 2D SWE surface solver, the 1D drainage network solver, and discrete hydraulic structures (culverts, weirs, gates, bridges, pumps). The GPU coupling kernels handle the device-side computation of source terms that transfer volume between these systems.

## Data Flow

```
SWE2DCouplingController (Python)
    ↓ pack_*_soa()          ← SoA packing from Python config objects
SWE2DCouplingSoA (Python dataclass)
    ↓ swe2d_gpu_upload_*   ← pybind11: upload SoA to GPU
CUDA device arrays
    ↓ swe2d_gpu_compute_coupling_sources()  ← kernel launch
Source rates (m³/s per cell)
    ↓ swe2d_solver_set_external_sources()   ← inject into next SWE step
```

## CUDA Coupling Kernels

### `swe2d_gpu_compute_coupling_sources`

Launch: one block per cell with sources.

For each coupling cell, computes:

```
source_cell = inflow_drainage + outflow_drainage + structure_flow
              + rainfall + infiltration + internal_source
```

Where:
- **inflow/outflow drainage**: From `SWE2DDrainageSoA` — inlet exchange rate computed by the 1D drainage solver, mapped to 2D cells
- **structure flow**: From `SWE2DStructuresSoA` — culvert/weir/gate/bridge/pump discharge through upstream/downstream cell pair
- **rainfall**: From rain gage data (hyetograph) + SCS Curve Number infiltration
- **internal source**: Point/polygon source terms from `internal_flow` forcing layer

### `swe2d_gpu_compute_bridge_coupling_sources`

Separate kernel for bridge structures using the stacked geometry approach. Handles:

- **Deck coupling**: Vertical exchange between bridge deck cells and main channel
- **Under-deck coupling**: Constricted flow beneath bridge deck
- **Phase 3 spatial redistribution**: Influence-width weighted distribution of bridge coupling sources across multiple cells

### `swe2d_gpu_redistribute_structure_sources`

Redistributes point-source structure flows (from culverts, weirs, gates, pumps) across multiple cells using influence-width weighting:

```
cell_weight = influence_width(cell) / sum(influence_width(all cells))
source_cell = total_structure_flow * cell_weight
```

### `swe2d_gpu_drainage_step`

Advances the 1D drainage network on GPU (pipe network solve for EGL/Diffusion/Dynamic wave modes). Evaluated after each 2D timestep:

1. Update node water surface elevations from coupled 2D cells
2. Solve node-to-node link equations (conduit, weir, orifice, pump, culvert)
3. Compute net exchange volume per node
4. Return sink/source arrays for 2D coupling

## SoA Structures (GPU Memory Layout)

### `SWE2DDrainageSoA`

| Field | Device Array | Description |
|---|---|---|
| `node_x/y` | `double*` | Drainage node coordinates |
| `link_from/to` | `int32_t*` | Link connectivity |
| `link_type` | `int32_t*` | Conduit / weir / orifice / pump / culvert |
| `link_geom` | `double*` | Length, diameter, roughness, inlet/outlet elev |
| `coupling_cell` | `int32_t*` | 2D cell index per drainage node |
| `exchange_rate` | `double*` | Computed m³/s exchange per node |

### `SWE2DStructuresSoA`

| Field | Device Array | Description |
|---|---|---|
| `structure_type` | `int32_t*` | Weir(1)/Culvert(2)/Gate(3)/Bridge(4)/Pump(5) |
| `upstream_cell` | `int32_t*` | 2D cell upstream |
| `downstream_cell` | `int32_t*` | 2D cell downstream |
| `crest_elev` | `double*` | Structure crest/invert elevation |
| `geom_params` | `double*` | Width, height, diameter, length, coefficient |
| `flow_rate` | `double*` | Computed m³/s through structure |

### `SWE2DCulvertFaceFluxSoA`

Face-flux culvert coupling stores per-face geometry for culverts that exchange flow along an edge rather than as a point source:

| Field | Description |
|---|---|
| `n_culvert_faces` | Number of culvert coupling faces |
| `d_face_edge_idx` | Edge index for each culvert face |
| `d_face_culvert_id` | Culvert structure index per face |
| `d_face_q` | Pre-computed unit discharge per face [m²/s] |

## Structure Computation on Device

Structure flows are computed on-device for all types except culvert table-based mode (which uses pre-computed rating tables uploaded from Python). The device-side evaluators mirror the Python implementations exactly (`swe2d/extensions/structures.py`).

### Weir Flow (Device)

```
Q = Cw * L * H^1.5
```

Where `H = max(0, max(WSE_us, WSE_ds) - crest_elev)` and `Cw` is the weir coefficient, `L` is effective weir length.

### Culvert Flow (Device — HDS-5 Nomograph)

The device path for HDS-5 culverts mirrors the FHWA nomograph approach:

1. Compute inlet control Q from inlet geometry, headwater depth, and inlet coefficient
2. Compute outlet control Q from full-flow Manning + entrance/exit losses
3. Select minimum of inlet, outlet, orifice, and Manning capacity
4. Apply tailwater submergence correction

### Gate Flow (Device)

```
Q = Cd * A_gate * sqrt(2 * g * H)
```

Where `A_gate` is the gate opening area and `H` is the head differential.

### Pump Flow (Device)

```
Q = q_pump  (constant, if enabled)
```

### Bridge Flow (Device)

Bridge coupling uses the stacked plan (under-deck, deck, over-deck layers) computed by `bridge_stacked_mesh.py`. The device kernel:
1. Identifies cells in each bridge zone
2. Applies loss-coefficient-weighted flow attenuation
3. Distributes sources per the `bridge_stacked_coupling_mode` (legacy_scalar or phase3_spatial)
