# HDS-5 Culvert Implementation

## Overview

The FHWA HDS-5 (Hydraulic Design of Highway Culverts) method computes culvert discharge under inlet or outlet control, selecting the minimum of five possible flow regimes:

1. **Inlet control** — Flow limited by the inlet geometry (entrance shape, headwater depth)
2. **Outlet control** — Flow limited by friction, entrance/exit losses, and tailwater
3. **Orifice control** — Submerged inlet and/or outlet
4. **Manning capacity** — Full-flow friction capacity
5. **Max flow cap** — User-specified upper bound

## USC Conversion

HDS-5 nomograph coefficients are tabulated in US Customary units (feet, CFS). The culvert path is the **only code path** that converts geometry to feet internally:

```
model_geometry → model_to_ft → feet
    ↓
HDS-5 computation in USC (feet, CFS)
    ↓
CFS → model units (÷ model_to_ft³)
```

The `model_to_ft` factor comes from `swe2d/units.py::model_to_ft()`:
- SI CRS (meters): 3.28084 ft/m
- USC CRS (feet): 1.0 ft/ft

## Computation Modes

### Mode 0: Full HDS-5 Nomograph (Device or Python)

Evaluates all five control regimes and selects the minimum.

**Inlet control** — Uses FHWA inlet control coefficients (`culvert_code` selects the nomograph):
- Code 1: Circular concrete, square edge with headwall
- Code 2: Circular concrete, grooved end
- Code 3: Circular CMP, projecting
- Code 4: Rectangular box, 30-45° wingwalls
- Code 5: Rectangular box, headwall parallel to embankment
- (Additional codes follow the HDS-5 nomograph numbering)

**Outlet control** — Solves the Bernoulli equation with:
- Entrance loss: `h_e = K_e * V²/(2g)`
- Friction loss: Manning's equation over full pipe length
- Exit loss: `h_x = K_x * V²/(2g)`
- Tailwater elevation from downstream WSE

**Orifice control** — `Q = Cd * A * sqrt(2*g*H)` with:
- Unsubmerged inlet: weir-type approach
- Submerged inlet: full orifice
- Submerged outlet: full pipe flow

### Mode 1: Table-Based (Python Pre-compute → GPU Upload)

Rating tables are pre-computed in Python by `swe2d_gpu_build_culvert_tables()` and uploaded to the GPU. The device kernel evaluates the table at the current head difference via linear interpolation. This is faster but requires table regeneration when geometry changes.

## Device Helper Functions

Defined in `swe2d_bindings.cpp` (flat `bw2d_*` functions) and mirrored in `swe2d/extensions/structures.py`:

| Function | Purpose |
|---|---|
| `bw2d_circular_area(d)` | Cross-sectional area of circular pipe at given depth |
| `bw2d_circular_wet_perimeter(d)` | Wetted perimeter at given depth |
| `bw2d_circular_hydraulic_radius(d)` | Hydraulic radius = Area / Perimeter |
| `bw2d_manning_capacity_full(d, slope, n)` | Full-pipe Manning capacity |
| `bw2d_culvert_inlet_control(hw, d, code)` | Inlet control Q from HDS-5 nomograph |
| `bw2d_culvert_outlet_control(hw, tw, ...)` | Outlet control Q from Bernoulli solve |
| `bw2d_culvert_orifice(hw, tw, crest, ...)` | Orifice control Q |
| `bw2d_culvert_flow(...)` | Top-level: min of all regimes |

## Python ↔ C++ Parity

The Python culvert module (`swe2d/extensions/structures.py:SWE2DStructureModule`) and the C++ device culvert path MUST produce identical results for the same input. This is verified by:
- `test_swe2d_drainage_structures.py` — compares Python structure flows against the standalone `culvert_routine` library
- GPU integration tests — compare total volumes with device-computed structure flows

## Culvert Face-Flux Mode

An alternative coupling method where culvert exchange is distributed along a 2D mesh edge rather than applied as a point source to a single cell. The face-flux mode:
1. Identifies the mesh edge(s) closest to the culvert inlet/outlet
2. Computes equivalent unit discharge `q = Q / edge_length`
3. Applies q to the edge flux computation directly (like a boundary condition)

This avoids the influence-width redistribution step and provides smoother coupling for wide culverts or embankment crossings.
