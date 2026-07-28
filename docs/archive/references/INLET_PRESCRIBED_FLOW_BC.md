---
type: reference
status: complete
created: 2026-07-12
completed: 2026-07-25
---

# Inlet Prescribed-Flow BC + Node Storage Fix

**Status:** ✅ IMPLEMENTED
**Date:** 2026-07-12
**Reference:** SWMM (Rossman & Huber, 2016)

---

## Problem Statement

When a node has an inlet assignment (e.g., drop inlet from street to pipe network), the current architecture has two physical incorrectnesses:

### Bug 1 — Phantom flow from head-driven flux at inlet nodes

The `swe2d_pipe1d_flux_kernel` computes head `H_n = node_invert + node_depth[n]` at ALL nodes, including inlet nodes. It then uses the HLLE flux formula to drive flow between the inlet node and adjacent pipe cells.

But the inlet node has NO pipe cell — it's a boundary. The head at the inlet node is set by the BC (currently from 2D WSE), NOT by the pipe network state.

This means the HLLE flux is effectively using a head-based BC at the inlet to drive flow into the network, which is wrong: **the inlet should PRESCRIBE the flow (from the inlet exchange), and the network should TRANSPORT it to other pipes**.

### Bug 2 — Inlet node has no real storage, making head meaningless

Even if Bug 1 were fixed, the `node_depth` at an inlet node comes from the BC (which sets it to 0 or to 2D WSE-based value). For a drop inlet, the "node" is really just a connection point — it doesn't have a real manhole / storage structure. Using `node_depth` to compute driving head for the network is physically wrong.

---

## SWMM Reference Approach

In SWMM (Rossman & Huber, 2016), inlet nodes are **flow-prescribed boundary nodes**:

1. The inlet (catch basin / drop inlet) has an assigned rated curve or weir/orifice equation that gives `Q_inlet = f(wse_surface)`
2. This `Q_inlet` is added directly to the node's inflow (not computed from head difference)
3. The pipe network solves continuity: `ΣQ_in - ΣQ_out = ΔV/Δt` at each node
4. The node depth `h` comes from `ΔV = A_node * h` where `A_node` is the surface area of the manhole structure
5. **The node head `H_node = invert + h` is computed FROM the 1D mass balance, NOT imposed from the 2D**

---

## Proposed Fix Architecture

### Phase 1: Treat Inlet Nodes as Prescribed-Flow BCs

#### Data Layer (`swe2d/runtime/coupling.py`)

Add `node_is_inlet: np.ndarray` to `SWE2DDrainageSoA` — int32 array, 1 if node has inlet assignment. Build it by marking nodes that appear in `inlet_node[]`.

#### GPU Kernel (`cpp/src/pipe1d.cu`)

Modify `swe2d_pipe1d_flux_kernel` — in the interior neighbor branch, skip HLLE head-based flux when the shared node is an inlet node:

```cpp
// Interior neighbor: head at shared node
const int32_t from_n = cell_from_node[nbr];
const int32_t to_n   = cell_to_node[nbr];
const int32_t shared_node = (dir > 0.0) ? cell_to_node[c] : cell_from_node[c];

// If the shared node is an inlet node, do NOT drive head-based flux.
// The inlet prescribes flow to the node via the inlet exchange kernel.
// Flow to/from the inlet is handled by node_net_q accumulation.
if (node_is_inlet && node_is_inlet[shared_node]) {
    // Treat as neutral neighbor: zero driving term
    H_n = H_c;   // zero head difference
    A_n = cell_A[c];
    Q_n = 0.0;   // no reverse flux from inlet node
} else {
    // Normal head-driven HLLE flux
    H_n = node_invert[shared_node] + node_depth[shared_node];
    ...
}
```

This zeroes the HLLE driving term without skipping the interface entirely, preserving the pipe cell's own continuity.

#### Host Wrapper Changes

- Add `node_is_inlet` parameter to `swe2d_pipe1d_flux_kernel`
- Add `d_node_is_inlet` device pointer to `SWE2DPipe1DState` struct
- Upload in `swe2d_gpu_upload_drainage_exchange_params()`
- Pass to flux kernel call
- Update all call sites of `swe2d_pipe1d_flux_kernel`

---

### Phase 2: Node Storage (SWMM-style)

The inlet exchange currently does:

```cpp
atomicAdd(&node_depth_delta[n], dt_s * (q_capture - q_relief) / node_area);
```

This means the inlet sets `node_depth` through `node_depth_delta`. This is actually physically correct for a drop inlet structure that has real surface area. The water enters the node via the inlet exchange, increasing node depth. The pipe network then sees the node head `H_node = invert + node_depth` and transports the water.

**Changes needed:**

1. **`swe2d/runtime/coupling.py` — `build_drainage_soa()`:**
   - `node_surface_area` is read from node metadata: `surface_area` or `surface_area_m2`
   - If node is an inlet node AND `node_surface_area` is very small (< 0.1 m²), set to a minimum realistic value (e.g., `0.5 m²` for a small catch basin)

2. **Existing fix:** `node_surface_area` floor already changed from `1.0` to `1e-6` in `swe2d_pipe1d_update_node_depth_kernel`. Minimum floor of `1e-6 m²` ensures tiny inlet structures still get realistic depth changes.

---

### Phase 3: Flow Direction Convention Verification

Currently:
- `swe2d_drainage_inlet_exchange_kernel` does `atomicAdd(&node_depth_delta[n], dt_s * (q_capture - q_relief) / node_area)` — positive delta = water entering node = node depth increases ✓
- `swe2d_pipe1d_update_node_depth_kernel` does `node_depth[n] += dt * node_net_q / area` — `node_net_q > 0` means net flow INTO node from pipes, increases depth ✓
- `accumulate_node_flux_kernel`: `atomicAdd(&node_net_q[fn], -Q_eff)` (outflow from fn), `atomicAdd(&node_net_q[tn], Q_eff)` (inflow to tn) ✓

**Verification:** When `q_capture > 0` (water entering node from surface), node depth increases. This water can flow through the pipe network. The flux kernel should NOT add additional head-driven flow at the inlet node — the water already entered via the inlet exchange.

---

### Phase 4: Already-Fixed Bug — Boundary Flux (Bug B)

The boundary flux branch in `swe2d_pipe1d_flux_kernel` was computing `F = dH * c_face` (m/s velocity) instead of `F = dH * c_face * cell_area_full[c]` (m³/s). **Status: FIXED.**

---

## Implementation Checklist

### Python changes (`swe2d/runtime/coupling.py`)

- [x] Add `node_is_inlet: np.ndarray` field to `SWE2DDrainageSoA` dataclass
- [x] Build `node_is_inlet` array in `pack_pipe_network_soa()`: mark nodes that appear in `inlet_node[]` as 1
- [x] Pass `node_is_inlet` through `static_args` in GPU upload

### C++ GPU changes (`cpp/src/swe2d_gpu.cu`)

- [x] Add `d_node_is_inlet` device pointer to `SWE2DPipe1DState` struct (in `pipe1d.cuh`)
- [x] Add `d_node_is_inlet` device pointer allocation in `swe2d_build_pipe1d_mesh()`
- [x] Upload `node_is_inlet` array in `swe2d_gpu_upload_drainage_exchange_params()` via `swe2d_mark_inlet_nodes_kernel`
- [x] Zero `d_node_is_inlet` on allocation

### C++ GPU changes (`cpp/src/pipe1d.cu`)

- [x] Add `node_is_inlet` parameter to `swe2d_pipe1d_flux_kernel`
- [x] Modify interior flux branch: skip HLLE head-based flux when `node_is_inlet[shared_node] == 1`
- [x] Update host wrapper `swe2d_pipe1d_flux_kernel_host()` to pass `node_is_inlet`
- [x] Update all call sites of `swe2d_pipe1d_flux_kernel`

### Build

- [x] Reconfigure: `cd build && cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release`
- [x] Build: `cmake --build . -j$(nproc)`
- [x] Verify no nvcc errors

---

## Key Design Decisions

1. **Skip HLLE driving term, not skip the whole interface:** When `node_is_inlet[shared_node] == 1`, we set `H_n = H_c` and `Q_n = 0` to zero the driving term, but we still compute the HLLE flux so the pipe cell's own continuity is conserved. This is cleaner than skipping the whole interface.

2. **No change to the inlet exchange kernel:** The `swe2d_drainage_inlet_exchange_kernel` already correctly computes `q_capture` (weir/orifice from surface) and adds it to `node_depth_delta`. No change needed there.

3. **Node surface area:** The minimum `node_surface_area = 1e-6` floor (already fixed) ensures tiny inlet structures still get realistic depth changes. If the user sets `surface_area_m2` in metadata, that value is used.

4. **Multiple inlets at same node:** If two inlets share a node, `node_is_inlet[n] = 1` for that node. Both inlets' `q_capture` contributions are summed in `node_depth_delta`. This is physically correct — multiple inlets discharge to the same catch basin.

5. **Boundary flux branch already fixed:** Bug B (boundary flux missing `cell_area_full` scaling) was fixed in a previous session. The boundary branch now correctly computes volumetric flux in m³/s.
