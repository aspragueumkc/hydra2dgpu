---
type: plan
status: complete
created: 2026-07-17
completed: 2026-07-25
---

# Pipe1D Godunov Rewrite — Fully-Dynamic Solver

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace the SWMM-style hybrid fully-dynamic solver with a true 1D Godunov FVM (MUSCL + HLLE + RK2) to eliminate the mass drift. Diffusion-wave mode unchanged.

**Architecture:** Three kernels per RK stage: reconstruct (MUSCL minmod slopes), flux (HLLE interior + characteristic BC at boundaries + source terms), update (explicit RK2). Node mass balance accumulates `stage_coeff × dir·F` per stage. Boundary ghost states from Riemann invariants, not the relaxation law.

---

## Key Design Points

| Component | Implementation |
|-----------|---------------|
| Reconstruction | MUSCL minmod on `A` and `Q`: `dU_i = minmod(U_i−U_{i−1}, U_{i+1}−U_i)` |
| Interior flux | HLLE (existing, unchanged) |
| Boundary BC | Ghost cell: `H_ghost = node_invert + node_depth`; `A_ghost = A(H_ghost)`; `Q_ghost` from outgoing Riemann invariant |
| Friction | Explicit source: `−g·A·Sf` where `Sf = n²·|Q|·Q/(k²·A²·R^4/3)` |
| Bed slope | Explicit source from cell invert: `g·A·(S0_cell)` |
| Time integration | RK2 (Heun's): two stages, per-stage flux + source → cumulative |
| Node debit | `atomicAdd(&node_net_q[n], stage_coeff · dir · F)` per boundary face, per stage |
| Coupling layer | Pipe-end / outfall / junction kernels unchanged |

## What gets deleted from `swe2d_pipe1d_fully_dynamic_kernel`

Everything. The old kernel has: Picard half-loop, A_new_cont per iteration, sigma damping, dq1/dq2/dq4 momentum, semi-implicit friction denominator, regime override. All gone.

## What replaced it

**New global kernels (all in `cpp/src/pipe1d.cu`):**

1. `swe2d_pipe1d_reconstruct_slopes_kernel` — 1 thread per cell. Reads `cell_A`, `cell_Q`, `cell_length`, `cell_to_node`, `cell_from_node`. Writes `dA_dx`, `dQ_dx` (slopes times dx: `minmod(A_i−A_{i-1}, A_{i+1}−A_i)`). Zero-gradient at link ends (virtual-node faces: slope = 0 to neighbor).

2. `swe2d_pipe1d_godunov_flux_kernel` — based on existing flux kernel. Added per-face: reconstruct left/right states `(A_L, Q_L)` and `(A_R, Q_R)` using MUSCL: `U_L = U_own + 0.5 × minmod_slope`; `U_R = U_nbr − 0.5 × minmod_slope_nbr`. For boundary faces (nbr < 0): set `(A_R, Q_R)` from a ghost cell using the Riemann-invariant BC against the node head.

   **Boundary BC (replaces relaxation law):**
   ```
   H_ghost = node_invert[shared_node] + node_depth[shared_node]
   A_ghost = A(H_ghost) via xsect_getAofY (or linear approx for speed first pass)
   c_cell  = sqrt(g * A_cell / T_cell)
   if Q_cell > 0 and Fr < 1:   // subcritical outflow
       Q_ghost = Q_cell + c_cell * (A_ghost - A_cell)
   elif Q_cell < 0 and Fr < 1: // subcritical inflow (node pushing into pipe)
       Q_ghost = Q_cell - c_cell * (A_ghost - A_cell)
   else:                        // supercritical or dry
       Q_ghost = Q_cell
       A_ghost = A_cell
   ```
   Then HLLE flux between cell and ghost state. Also `atomicAdd(&node_net_q[shared_node], stage_coeff · dir · F)`.

   Source terms evaluated per cell: `S_A = 0`, `S_Q = −g · A_eff · Sf + g · A_eff · S0_cell` where `Sf` from Manning's and `S0_cell = cell_S0[c]`.

   Writes `flux_Q_out[c]` (HLLE flux + source contributions for the stage).

   Parameters: adds `stage_coeff` (double), `node_surface_area` (for dry-node cap), `node_is_boundary`.

3. `swe2d_pipe1d_rk2_update_kernel` — 1 thread per cell. Reads `cell_A`, `cell_Q`, `flux_Q_out`, `A_prev`, `Q_prev` (start-of-step). Writes `cell_A_new` and `cell_Q_new`:
   ```
   if stage == 0:
       A_star  = A_prev - dt * flux_A / L
       Q_star  = Q_prev - dt * flux_Q / L
       cell_A_new[c] = A_star;  cell_Q_new[c] = Q_star
       A_prev_save[c] = A_prev;  Q_prev_save[c] = Q_prev
   else:
       A_final = 0.5 * (A_prev_save + A_star - dt * flux_A / L)
       Q_final = 0.5 * (Q_prev_save + Q_star - dt * flux_Q / L)
   ```
   Actually simpler: the host wrapper manages two arrays `d_A_tmp`, `d_Q_tmp` (stage buffer) + `d_A_start`, `d_Q_start` (RK2 start-of-step save). The update kernel writes:
   - Stage 0: `new = start − dt · flux,  start_copy = start`
   - Stage 1: `new = 0.5 · (start_copy + tmp − dt · flux)`

## Host wrapper: `swe2d_pipe1d_fully_dynamic_host` (replaces old fully_dynamic host wrapper)

Called from `swe2d_pipe1d_step` when mode is `"fully_dynamic"`.

```
stage 0:
    reconstruct_godunov_slopes(...)
    godunov_flux_kernel(stage_coeff=1.0, ...)
    rk2_update_kernel(stage=0, dt=local_dt, ...)
    // stage buffer now holds A*, Q*
stage 1:
    reconstruct_godunov_slopes(...)   // re-read A*, Q* for slopes
    godunov_flux_kernel(stage_coeff=1.0, ...)
    rk2_update_kernel(stage=1, dt=0.5*local_dt, ...)
    // final A, Q in cell_A_new, cell_Q_new
```

Node mass balance: `node_net_q` was zeroed before the substep loop (F1 fix) and accumulated across both stages via `atomicAdd(&node_net_q[n], stage_coeff · dir · F)`. For RK2, the correct cumulative flux into the node = `0.5 · (F_stage0 + F_stage1)`? No — actually the node mass balance is simpler: the `node_net_q` accumulates `dir · F` per stage without stage_coeff weighting (each stage's flux is the actual mass flux rate at that stage), and the net volume transfer to the node is `local_dt · (coeff0 · F0 + coeff1 · F1)` / ... hmm, this needs precision.

Actually, for conservation: the total volume transferred from cell to node over RK2 is:
- Stage 0: ΔV0 = local_dt · Dir·F0 · cell_exit_face_area_factor
- Stage 1: ΔV1 = local_dt · Dir·F1

The node mass balance should accumulate: `node_net_q += stage_coeff · Dir·F` where stage_coeff for RK2 (Heun's) is:
- Stage 0: coeff = 1.0 (the full-step tentative flux)
- Stage 1: coeff = 0.5 (half-weight in the final correction)

Hmm but Heun's: `U* = U^n + Δt·R(U^n)`; `U^{n+1} = U^n + 0.5·Δt·R(U^n) + 0.5·Δt·R(U*)`. So the total flux contribution is `0.5·F(U^n) + 0.5·F(U*)` = `0.5·(F0 + F1)`. So node_net_q accumulation: `node_net_q += 0.5 · Dir·F` for each stage? Wait that's cleaner: `node_net_q += Dir·F / n_stages` per stage, then dh = local_dt · node_net_q / area.

Simplest correct approach: each stage's flux kernel contributes `atomicAdd(&node_net_q[n], dir·F / n_stages)` (where n_stages=2). Then `dh = local_dt · node_net_q / area` at step end.

OR: each stage contributes `atomicAdd(&node_net_q[n], dir·F)`, then `dh = (local_dt / n_stages) · node_net_q / area` at step end. Same thing.

I'll use the latter: simpler kernel code (no stage_coeff param), adjust dh calc.

## What to do with old tests

- `test_closed_system_conserves_mass_fully_dynamic` — should now PASS (mass drift eliminated).
- `test_node_outflow_limited_by_storage (fully_dynamic subtest)` — should PASS.
- Pre-existing failing tests in `test_swe2d_pipe1d` that test the fully-dynamic kernel — some may pass, some may need assertion adjustments. Review and report.
- Diffusion tests unaffected.

## Build

Standard inc build: `cd build && cmake --build . -j$(nproc)`

---

Files: `cpp/src/pipe1d.cu` (new kernels + modified host), `cpp/src/pipe1d.cuh` (declarations).
