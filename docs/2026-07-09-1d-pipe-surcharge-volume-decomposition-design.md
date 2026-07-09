# 1D Pipe Network Surcharge / Pressurized Flow — Volume Decomposition Design

**Date:** 2026-07-09  
**Scope:** Extend the `pipe1d` fully-dynamic solver to accurately represent surcharge (pressurized) conditions and optionally exchange surcharge water with the 2D surface domain.  
**Out of scope:** Air-pocket modeling, water hammer, transient compressible effects.

## 1. Goal

When a closed conduit fills to capacity, additional inflow must be stored as surcharge above the crown. The current fully-dynamic solver already clamps the pipe cell area at `A_full` and stores excess volume in the node head, but the pipe-end exchange kernel caps the node depth at `node_max_depth`, undoing the surcharge. In diffusion-wave mode, the flux kernel also computes the cell head from the capped area, producing incorrect head gradients in full pipes. This design makes surcharge behavior explicit, accurate, and two-way-coupled with the 2D surface.

## 2. Why Volume Decomposition Fits Our Solver

Volume decomposition is not tied to HEC-RAS v7’s Eulerian-Lagrangian SWE formulation. It is a geometric convention:

- **Pipe cells** carry only the free-surface portion of the cross-section, capped at `A_full`.
- **Node heads** can rise above the crown; the volume above the crown is stored in the node’s real surface area (`node_surface_area`).
- The **pressure gradient** is driven by the piezometric head at the nodes, which is already the dominant term in our fully-dynamic kernel.

Our current `pipe1d.cu` already implements most of this unintentionally in the fully-dynamic path:
- `A_new` is clamped to `A_full` in `swe2d_pipe1d_fully_dynamic_kernel` (line 730).
- The pressure term uses `H_from = node_invert[fn] + node_depth[fn]`.
- `node_depth` is not capped by the pipe update kernel.

The problems are at the **boundaries**:
- The pipe-end exchange kernel, `swe2d_drainage_pipe_end_exchange_kernel`, reconciles the node depth and caps it at `node_max_depth[n]`, undoing any surcharge.
- In the diffusion-wave mode, the flux kernel still computes `H_c = cell_invert + A / T`, which is invalid once the cell is full.

Correcting those two places is the core of this work.

## 3. Design

### 3.1 State Model

- `node_head[n] = node_invert[n] + node_depth[n]` (piezometric head above datum).
- `node_crown[n] = node_invert[n] + node_max_depth[n]`.
- `cell_crown[c] = cell_invert[c] + cell_height[c]` (pipe crown at the cell midpoint).
- A pipe cell `c` is **full** when both end-node heads are at or above the cell crown:
  ```
  is_full[c] = (node_head[from_node[c]] >= cell_crown[c]) &&
               (node_head[to_node[c]]   >= cell_crown[c])
  ```
- Full cells use `A = A_full` and `P = P_full`.
- Surcharge volume is stored entirely in node heads, not in pipe cells.

### 3.2 Flux Kernel Changes (`swe2d_pipe1d_flux_kernel`)

Current cell-head computation for the HLLE flux is:
```
H_c = cell_invert[c] + cell_A[c] / T_c
```

Replace it with a full-aware value:

```
if (is_full[c]) {
    // For a full cell, the piezometric head is the node head field, not the local A/T.
    // Use a simple representative head (e.g., average of the two end-node heads).
    H_c = 0.5 * (node_head[from_node[c]] + node_head[to_node[c]]);
} else {
    // Free surface: keep the existing A/T based head.
    H_c = cell_invert[c] + cell_A[c] / fmax(T_c, 1e-6);
}
```

For interior faces, the neighbor head already uses the shared node head, so this change makes the head gradient consistent across full cells. The wave speed in the HLLE flux remains based on `sqrt(g * |H_c - H_n| / L)` and will naturally reflect the pressure head when the pipe is full.

### 3.3 Fully-Dynamic Update Kernel Changes (`swe2d_pipe1d_fully_dynamic_kernel`)

The current implementation already does most of what is needed:
- Clamp `A_new` at `A_full`.
- Compute the pressure gradient from node heads (`H_from`, `H_to`).
- Store new discharge in `cell_Q_iter` and area in `cell_A_iter`.

No change is required here unless Picard iteration becomes unstable with very small node surface areas. If that happens, the relaxation factor or the number of iterations can be tuned; switching to a global linear solver is left as a future fallback.

However, the **pipe-end exchange kernel** that runs after the step must stop capping `node_depth` at `node_max_depth`. The reconciled node depth should be written as-is, so surcharge can persist across coupling steps.

### 3.4 Node Continuity

`node_depth` is updated by net pipe flux divided by `node_surface_area`. With the volume decomposition approach, the depth is allowed to exceed `node_max_depth` without any artificial cap. This represents surcharge storage in the manhole or junction.

### 3.5 Optional Two-Way Surface Coupling

The pipe-end exchange kernel already has the right shape for two-way flow: it computes the change in node depth over the coupling step and translates that into a source/sink for the 2D cell. Two changes are needed:

1. **Stop capping `node_depth_write` at `node_max_depth`.** This cap currently undoes surcharge. The reconciled depth should be written as-is when volume decomposition is enabled.
2. **Gate positive (network → surface) flow by an overflow condition.** If the node head is above the surface WSE (or an explicit overflow elevation) and the inlet is enabled for overflow, allow the node to discharge. If overflow is disabled, the node head may still rise above the crown but no water leaves the network at this coupling point.

### 3.6 Data-Model / Schema Additions

Add to `swe2d_drainage_inlets` (or `swe2d_drainage_nodes`):
- `enable_overflow`: boolean, default `false`.
- `overflow_elevation`: optional elevation above datum; if omitted, use the node crown (`node_invert + node_max_depth`).
- `max_overflow_rate`: optional capacity limit (m³/s), default unlimited.

These fields are optional; existing models continue to work with one-way coupling.

### 3.7 Device-State Additions

- `Pipe1DDeviceState`: `d_node_max_depth` is already available via `node_invert + node_max_depth` host computation. If `d_node_max_depth` is not on device, add it or compute `d_node_crown` once at build time.
- `d_cell_height` already exists on device; use it to compute `cell_crown` in the flux kernel.
- Pass overflow parameters and a flag for volume-decomposition mode to the pipe-end exchange kernel.

## 4. Implementation Steps

1. **Compute and upload `d_node_max_depth`** (or `d_node_crown`) to the device.
2. **Modify `swe2d_pipe1d_flux_kernel`** to detect full cells and use a node-head-based representative head instead of `A/T` for the diffusion-wave mode.
3. **Verify `swe2d_pipe1d_fully_dynamic_kernel`** already clamps area and uses node heads for pressure gradient; tune Picard iterations if needed.
4. **Modify `swe2d_drainage_pipe_end_exchange_kernel`**:
   - Remove the `fmin(node_max_depth[n], ...)` cap on `node_depth_write` when volume decomposition is enabled.
   - Gate positive (network → surface) flow by the overflow condition and `max_overflow_rate`.
5. **Update Python extension models and schema** for the overflow fields.
6. **Add tests** (Section 5).
7. **Update user docs** with guidance on when to enable overflow and how to set `node_surface_area` for realistic manhole storage.

## 5. Testing

- **Single pipe surcharge**  
  Constant inflow into a level pipe with a closed downstream end. Verify that `A` caps at `A_full`, node head rises monotonically, and total volume is conserved (inflow × dt ≈ `A_full * L + node_head_above_crown * node_surface_area`).

- **Surcharge propagation**  
  Two-pipe network with different invert elevations. Verify that a surcharge at one junction propagates upstream/downstream via node heads and that flow direction follows the head gradient even when both pipes are full.

- **Two-way overflow**  
  Surcharge a node connected to a 2D surface cell. Verify that water leaves the node and enters the 2D cell when the node head exceeds the overflow elevation, and that the 2D WSE rises accordingly.

- **Mass conservation across coupling**  
  Run a closed system (no external boundaries) with surcharge and overflow and verify total water volume in pipe + node + 2D remains constant.

## 6. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Picard iteration fails when node storage is very small and pressure waves are very fast. | Increase `implicit_iters` or reduce `dt`; consider a global linear pressure solver only if observed. |
| Flux kernel becomes too diffusive with averaged full-cell head. | Evaluate upwind-based full-cell head (`max(H_from, H_to)` in direction of flow) if averaging proves too dissipative. |
| Two-way overflow causes 2D instabilities. | Apply the overflow flux as a source rate limited by the inlet capacity and the available surcharge head. |
| Users forget to set `node_surface_area`, producing unrealistic surcharge heads. | Add validation warning in the workbench when drainage nodes have zero or default surface area. |

## 7. Acceptance Criteria

- Full cells in the fully-dynamic mode produce head gradients consistent with the node heads, not the capped area.
- A pipe with sustained inflow reaches `A_full` and then stores additional volume as rising node head.
- Optional two-way overflow moves water from a surcharged node back to the 2D surface when enabled.
- Mass is conserved in the pipe + node + 2D system for all tested surcharge scenarios.
- Existing free-surface pipe tests continue to pass unchanged.
