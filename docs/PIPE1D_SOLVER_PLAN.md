# 1D Pipe Network Solver — Design & Implementation Plan

## Goal
Replace the existing per-link drainage solver modes (0/1/2) with a proper network-scale
1D pipe solver that runs on the CSR graph topology. Two options: **Diffusion Wave**
(explicit, no inertia) and **Fully Dynamic** (semi-implicit, captures backwater).
Modeled on HEC-RAS's approach: same semi-implicit FVM infrastructure, adapted for
closed-conduit flow.

---

## What HEC-RAS Does (Reference)

From HEC-RAS documentation (`modeling-pipe-networks`):

> "The new implementation of pipe flow in HEC-RAS utilizes the same semi-implicit
> computational methods that drive the 2D and 1D finite volume shallow water solvers."

Key points:
- **Preissmann slot** (older versions) or **volume decomposition** (v7.0+) for pressurized flow
- **Semi-implicit** pressure term: bypasses Courant constraint for pressure waves
- No water hammer (inertia terms handled implicitly, not absent)
- Volume-conservative, handles wetting/drying, uses subgrid bathymetry

From `hydraulic-equations-pipe-flow`:

```
Continuity:  ∂A/∂t + ∂Q/∂x = q
Momentum:    ∂V/∂t + V·∂V/∂x = -g·∂H/∂x - (τ_b/ρR) - (F_L/ρA)
Boundary:    τ_b = ρ·C_D·|V|·V,  C_D = n²g/R^(1/3)
Minor losses: F_L/ρA = g·H_L/L
```

Diffusion wave (user's specified simplification, HEC-RAS default):
```
g·∂H/∂x = -(n²g/R^(4/3) + K_L/2L)·|V|·V
```

---

## Architecture Decision

### Option A — Force the 2D SWE Kernel ❌

Attempting to use `swe2d_update_kernel` with reinterpreted state fails because
the HLLE/HLLC flux function uses 2D gravity wave eigenvalues `λ = u ± √(gh)`.
Pipe eigenvalues are `±√(gH)` (pressure waves), ~100× faster. The CFL constraint
would be catastrophic and the physics are wrong.

### Option B — New 1D Pipe Solver on the CSR Graph ✅

Build `swe2d_pipe1d_step` as a **separate kernel** that:
1. Reuses the **same CSR infrastructure** (`cell_owned_offsets`, `cell_peer_offsets`)
2. Solves on the **1D pipe network graph** — not the 2D mesh
3. Stores `A` (cross-sectional area) and `Q` (discharge) as state
4. Uses a **1D HLLE flux** appropriate for the pipe momentum equation
5. Adds **Manning's friction + minor losses** as source terms
6. Couples to the 2D surface via the existing `swe2d_drainage_pipe_end_exchange_kernel`

The old mode 0/1/2 kernels are **deleted** and replaced entirely.

---

## Two Solver Modes

### Mode: Diffusion Wave (default)

Explicit, no local inertia terms:

```
Q_new = Q_old + dt · (−g·A·n²·|Q|·Q·P / (A²·R^(4/3)))
A_new = A_old − dt · ∂Q/∂x
```

Head gradient `∂H/∂x` is not a driving term — friction slope `S_f` balances
the bed slope `S`. Flow is **Manning's normal flow** driven by pipe geometry,
independent of the transient head field. No backwater propagation.

Flux: 1D HLLE with wave speed `c = √(g·H/L)` where `H` is piezometric head.

### Mode: Fully Dynamic

Semi-implicit, captures backwater and network coupling:

```
Q_new = Q_old + dt · (−g·A·∂H/∂x − g·A·n²·|Q|·Q·P / (A²·R^(4/3)) − g·A·K_L·|Q|·Q·P / (2·A³))
A_new = A_old − dt · ∂Q/∂x
```

The pressure gradient term `g·A·∂H/∂x` is treated **semi-implicitly**: it uses the
new head (unknown at the start of the step), which couples all pipe segments into
a system solved iteratively. This captures backwater, surcharge, and wave
propagation without the water hammer CFL restriction.

Flux: 1D HLLE with `c = √(g·H/L)` for the pressure wave.

---

## Frontend — No New Layers

Existing drainage layers are unchanged:

| Layer | Role | Changes |
|-------|------|---------|
| `swe2d_drainage_nodes` (Point) | Network junctions / manholes | None |
| `swe2d_drainage_links` (LineString) | One pipe = one 1D cell | None |
| `swe2d_drainage_inlets` | Surface-to-network inlet BC | None |
| `swe2d_drainage_node_inlets` | Inlet-type assignments | None |
| `swe2d_drainage_pipe_ends` | Network-to-2D coupling | None |

**Mesh cell length**: `DrainageLink.length` IS the cell length. Each link = one
pipe cell. No new field required unless subdivision is wanted.

### Optional: `max_cell_length` for Automatic Subdivision

Add `max_cell_length` to `DrainageLink`. When `link.length > max_cell_length > 0`,
subdivide into `N = ceil(link.length / max_cell_length)` sub-cells:

```
sub-cell i:
  length  = link.length / N
  midpoint = linear interpolation along link from_node → to_node
  area    = same as parent link
  perim   = same as parent link
  invert  = linearly interpolated between inlet/outlet inverts
```

The CSR mesh builder performs this subdivision. No kernel changes — the builder
produces more cells from the same link. `max_cell_length = 0` (default) = no
subdivision.

### Node Representation

A `DrainageNode` is a **mesh node in the 1D CSR graph** (peer neighbor), not a
2D mesh entity. It is NOT part of the 2D surface mesh.

For a junction connecting N pipes:
- Each pipe cell lists the junction as a **peer neighbor** in `cell_peer_ids`
- All N pipes share the same node index
- `node_depth[n]` is the junction water depth, shared across all connected pipes
- `node_surface_area[n]` is used in the node continuity equation

The `PipeEndExchange` bridges the two index spaces:
```
pipe_end_cell[c] → 2D surface cell index  (lives in d_h, d_cell_area, …)
pipe_end_node[n]  → 1D network node index (lives in d_node_depth, d_node_net_q, …)
```

### Solver Mode Selector

Replace the existing DrainageSolverMode labels:

| Old | New |
|-----|-----|
| Mode 0 — Energy/Bernoulli | ~~deleted~~ |
| Mode 1 — Diffusion Wave | ~~deleted~~ |
| Mode 2 — Dynamic Wave | ~~deleted~~ |
| — | **Diffusion Wave** (explicit Manning normal flow) |
| — | **Fully Dynamic** (semi-implicit, network-coupled) |

The existing radio/dropdown UI maps to two named options instead of numbered modes.

---

## C++ Kernel Design

### `swe2d_build_pipe1d_mesh`

Signature:
```cpp
void swe2d_build_pipe1d_mesh(
    int32_t               n_links,
    const int32_t*        link_from_node,    // [n_links] node index in DrainageNode array
    const int32_t*        link_to_node,      // [n_links]
    const double*         link_length,       // [n_links] geometric length (m)
    const double*         link_diameter,      // [n_links] pipe diameter (m)
    const double*         link_roughness_n,  // [n_links] Manning's n
    const double*         link_inlet_loss_k, // [n_links] minor loss K at inlet
    const double*         link_outlet_loss_k,// [n_links] minor loss K at outlet
    const double*         node_invert_elev,   // [n_nodes] invert elevation
    const double*         node_surface_area,  // [n_nodes] junction surface area
    const double*         node_max_depth,    // [n_nodes] max depth
    const double*         link_invert_in,     // [n_links] inlet invert (for subdivision interp)
    const double*         link_invert_out,    // [n_links] outlet invert
    int32_t               max_cell_length,    // 0 = no subdivision, >0 = max sub-cell length (m)
    Pipe1DDeviceState*   dev);               // output: CSR arrays + pipe geometry buffers
```

Output device buffers in `Pipe1DDeviceState`:
```cpp
struct Pipe1DDeviceState {
    // CSR topology (same format as 2D mesh)
    int32_t*  d_owned_offsets;  // [n_pipe_cells + 1]
    int32_t*  d_owned_ids;      // [n_owned_faces]  face = neighbor cell
    int32_t*  d_peer_offsets;   // [n_pipe_cells + 1]
    int32_t*  d_peer_ids;       // [n_peers]  peer = DrainageNode index

    // Pipe cell geometry
    double*   d_cell_length;    // [n_pipe_cells] actual cell length (after subdivision)
    double*   d_cell_area;      // [n_pipe_cells] cross-sectional area
    double*   d_cell_perim;     // [n_pipe_cells] wetted perimeter
    double*   d_cell_invert;    // [n_pipe_cells] invert at cell midpoint
    double*   d_cell_n;        // [n_pipe_cells] Manning's n
    double*   d_cell_k_loss;    // [n_pipe_cells] minor loss K

    // Node state
    double*   d_node_depth;     // [n_nodes] current node depth
    double*   d_node_net_q;     // [n_nodes] net flow accumulator

    // Pipe cell state
    double*   d_A;              // [n_pipe_cells] current cross-sectional area
    double*   d_Q;              // [n_pipe_cells] current discharge
    double*   d_A_prev;         // [n_pipe_cells] area from previous step (for dynamic wave)
    double*   d_Q_iter;         // [n_pipe_cells] Q from previous Picard iteration

    int32_t   n_pipe_cells;
    int32_t   n_nodes;
};
```

### `swe2d_pipe1d_flux_kernel`

One thread per pipe cell. Accumulates discharge at each cell face:

```cuda
__global__ void swe2d_pipe1d_flux_kernel(
    int32_t          n_cells,
    const int32_t*   owned_offsets,   // CSR
    const int32_t*   owned_ids,
    const int32_t*   peer_offsets,
    const int32_t*   peer_ids,
    const double*    cell_length,
    const double*    cell_wse,       // WSE at each cell centroid (from node depths)
    const double*    node_invert,     // node invert elevations
    const double*    cell_A,          // current area
    const double*    cell_Q,          // current discharge
    double*          flux_Q,           // output: accumulated Q
    double           gravity)
{
    // For each owned face of cell c:
    //   look up neighbor cell n (or node if peer)
    //   HLLE wave speed: c = sqrt(g * |H_c - H_n| / cell_length)
    //   HLLE flux: F = 0.5 * (Q_c + Q_n - c * (A_n - A_c))
    //   accumulate into flux_Q[c] and flux_Q[n]
}
```

### `swe2d_pipe1d_update_kernel`

One thread per pipe cell. Two paths:

**Diffusion Wave path**:
```cuda
// Q_new = Q_old + dt * S_Q
// S_Q = -g * A * n² * |Q| * Q * P / (A² * R^(4/3))  // Manning friction
//      - g * A * K_loss * |Q| * Q * P / (2 * A³)    // minor losses
// A_new = A_old - dt * (flux_Q_out - flux_Q_in) / cell_length
```

**Fully Dynamic path**:
```cuda
// Q_new = Q_old + dt * ( -g*A*∂H/∂x - g*A*n²*|Q|*Q*P/(A²*R^(4/3)) - g*A*K_loss*|Q|*Q*P/(2*A³) )
// ∂H/∂x = (H_right - H_left) / cell_length  // head gradient from WSE
// A_new = A_old - dt * (flux_Q_out - flux_Q_in) / cell_length

// Semi-implicit: the system (Q_new for all cells) is coupled through ∂H/∂x.
// Solve via Picard iteration: iterate Q until ∂H/∂x converges.
// Each iteration: recompute Q from current Q and current ∂H/∂x.
```

### `swe2d_drainage_node_update_kernel` (existing, unchanged)

Already exists. Updates `node_depth` from accumulated `node_net_q`:
```
d_new = d_old + dt * ΣQ_net / node_surface_area
```

---

## 2D ↔ 1D Coupling — Unchanged

`swe2d_drainage_pipe_end_exchange_kernel` stays exactly as written.
It already bridges the two index spaces:
- `pipe_end_cell` → 2D mesh index (reads `cell_wse`, writes `q_cell`)
- `pipe_end_node` → 1D network node index (reads `node_depth`, writes `node_depth_write`)

The 1D pipe solver writes to `d_Q` (discharge per pipe cell). The exchange kernel
reads node depth and computes the volume exchange between the pipe network and the
2D surface. The 1D solver does not need to know about the 2D mesh directly.

---

## Data Flow

```
User edits swe2d_drainage_links (QGIS)
         ↓
DrainageLink.length              → pipe_cell.length
DrainageLink.diameter            → pipe_cell.area, pipe_cell.perim
DrainageLink.roughness_n         → pipe_cell.n
DrainageLink.entrance_loss_k     → pipe_cell.k_loss (at inlet)
DrainageLink.exit_loss_k         → pipe_cell.k_loss (at outlet)
DrainageNode (junction x, y)    → peer neighbor index (no new entity)
         ↓
swe2d_build_pipe1d_mesh()  →  CSR + geometry buffers on GPU
         ↓
swe2d_pipe1d_step(
    Diffusion Wave | Fully Dynamic,
    dt, gravity,
    coupling_substeps,
    implicit_coupling_iterations,
    implicit_coupling_relaxation)
         ↓
  iterates swe2d_pipe1d_flux_kernel + swe2d_pipe1d_update_kernel
  per coupling sub-step
         ↓
swe2d_drainage_pipe_end_exchange_kernel()  ← unchanged
         ↓
2D surface cell update (swe2d_update_kernel) ← unchanged
```

---

## Python Bindings

```cpp
// New functions
m.def("swe2d_build_pipe1d_mesh", &build_pipe1d_mesh, ...);
m.def("swe2d_pipe1d_step", &pipe1d_step,
    py::arg("dev"),
    py::arg("dt"),
    py::arg("solver_mode"),        // "diffusion_wave" | "fully_dynamic"
    py::arg("coupling_substeps"), // int
    py::arg("implicit_iters"),     // int (Picard iterations for dynamic)
    py::arg("relaxation"),        // double
    py::arg("gravity"),
    "...");
```

`solver_mode` is a string: `"diffusion_wave"` or `"fully_dynamic"`.
Replaces the old integer `DrainageSolverMode` enum.

---

## Frontend Changes

### `swe2d/extensions/extension_models.py`
- `DrainageLink`: add `max_cell_length: float = 0.0`
- `PipeNetworkConfig`: replace `solver_mode: DrainageSolverMode` with
  `pipe_solver_mode: str = "diffusion_wave"` (`"diffusion_wave"` | `"fully_dynamic"`)

### `swe2d/workbench/services/schema_definitions.py`
- `swe2d_drainage_links`: add `max_cell_length: float` field

### `swe2d/runtime/coupling.py`
- Delete all `solver_mode` integer dispatch
- Add `pipe_solver_mode` string dispatch to new C++ functions
- Update `apply_native_device_sources` to call `swe2d_pipe1d_step`

### UI — Drainage Solver Mode Selector
Replace the three-mode radio buttons with two named options:
- **Diffusion Wave** — fast, Manning normal flow, no backwater
- **Fully Dynamic** — semi-implicit, captures backwater and surcharge

---

## Effort Estimate

| Phase | Hours |
|--------|-------|
| C++ mesh builder + bindings | 2–3 |
| 1D flux kernel (HLLE) | 2–3 |
| Diffusion wave update kernel | 1–2 |
| Fully dynamic update kernel + Picard loop | 3–4 |
| Python bindings + coupling.py wiring | 1–2 |
| Frontend schema + UI | 2 |
| Delete old mode 0/1/2 kernels | 1 |
| Tests (diffusion wave, fully dynamic, mass conservation, backwater) | 3–4 |
| **Total** | **14–20** |

---

## What the Old Modes Were Missing

| Problem | Impact |
|---------|--------|
| Mode 0 claimed Bernoulli but mixed Manning's + orifice-style C_minor with extra 1/A² | Flow magnitude wildly wrong |
| Mode 1 used variable head gradient as driving slope | Not diffusion wave (should use fixed bed slope) |
| Mode 2 missing local acceleration ∂Q/∂t | Not dynamic wave — just Manning with relaxation |
| No network coupling | Each link computed in isolation, no backwater propagation |
| No node continuity coupling | Junctions don't share head/flow |

The new solver fixes all of these. Diffusion Wave matches the user's specified
`g·∂H/∂x + g(S_f − S) = 0` simplification. Fully Dynamic includes the full
pressure gradient term solved semi-implicitly.

---

## Files Changed

| File | Change |
|------|--------|
| `cpp/src/swe2d_gpu.cuh` | Add `Pipe1DDeviceState` struct |
| `cpp/src/swe2d_gpu.cu` | Add builder + flux + update kernels; **delete** `swe2d_drainage_link_kernel` |
| `cpp/src/swe2d_bindings.cpp` | Add bindings for new functions |
| `swe2d/extensions/extension_models.py` | Add `max_cell_length`; replace `solver_mode` enum with `pipe_solver_mode` str |
| `swe2d/workbench/services/schema_definitions.py` | Add `max_cell_length` field |
| `swe2d/runtime/coupling.py` | Wire new solver; remove old mode dispatch |
| `swe2d/runtime/backend.py` | Expose new functions |
| `tests/test_swe2d_drainage_structures.py` | Update tests for new mode names |
