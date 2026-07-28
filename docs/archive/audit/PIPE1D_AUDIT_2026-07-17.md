---
type: audit
status: complete
created: 2026-07-17
completed: 2026-07-25
---

# Pipe1D Solver & Coupling Audit — 2026-07-17

Audit of `cpp/src/pipe1d.cu`, `cpp/src/pipe1d.cuh`, the coupling kernels in
`cpp/src/swe2d_gpu.cu` (inlet / outfall / pipe-end exchange), and the Python
orchestration in `swe2d/runtime/coupling.py`. Trigger: non-physical results in
coupled 1D/2D runs. Findings are ranked by severity. Probes were run on the GPU
against the built `hydra_swe2d` module (scripts in `/tmp/opencode/probe*.py`);
no production code was modified.

## Executive summary

The pipe solver is **not mass-conservative at network-node boundaries**. Pipe
cells are advanced by one quantity (HLLE / boundary-relaxation *face fluxes*),
while node storage is advanced by a different quantity (the momentum-equation
`cell_Q`). The two are never equal per face, so mass is created or destroyed at
every node every timestep — catastrophically during transients (probe: **−91 %
of total system mass in 200 steps** in a closed two-node network). Layered on
top: the fully-dynamic kernel applies its continuity update once *per Picard
iteration* (up to 8× per substep), inlet nodes are infinite sources in
`fully_dynamic` mode, the diffusion-wave mode exhibits a persistent
checkerboard oscillation, and several exchange paths (junction overflow,
2D-coupled outfalls) leak or sink mass outright.

---

## P0 — Mass-conservation defects (numerically confirmed)

### 1. Node mass balance uses `cell_Q`, cell continuity uses face flux `F`

- Cell area update: `A_new = A − dt·flux_Q_out[c]/L` where `flux_Q_out` comes
  from the flux kernel (HLLE interior, boundary relaxation) —
  `pipe1d.cu:1600` (diffusion), `pipe1d.cu:1875` (fully-dynamic).
- Boundary-face flux formula: `F = (H_end − H_n_eff)·c_wave·A_eff·dir_sign`
  — `pipe1d.cu:1370`.
- Node depth update: `node_net_q` accumulated from **`cell_Q`** (the
  Manning/dynamic-wave discharge), not from the face fluxes —
  `pipe1d.cu:2279-2280` (`atomicAdd(&node_net_q[fn], −Q)` /
  `atomicAdd(&node_net_q[tn], Q)`).

`F` (relaxation law) and `cell_Q` (Manning / dynamic wave) are computed by
unrelated formulas and are equal only at exact steady state. Every transient
therefore leaks mass at every node–cell interface. The regime override
(`pipe1d.cu:1644-1668, 2007-2031`) makes it worse: it caps `Q` but not `F`.

**Probe (closed 2-node system, 1 link, 10 sub-cells, SLOT surcharge, d0=1 m):**
diffusion_wave lost **90.9 %** of total mass in 200 steps; fully_dynamic lost
**73.3 %**. Most of the loss occurs in the first 2 steps (surcharge collapse
drains cells via `F` while nodes are credited ≈0 via `cell_Q`).

**Fix direction:** accumulate the *boundary-face fluxes* into `node_net_q`
from inside the flux kernel (atomicAdd per boundary face), so node and cell
see the same mass crossing the interface; or make the boundary-face flux in
continuity equal to `cell_Q` of the end cell (SWMM convention: one link flow
serves both sides).

### 2. Fully-dynamic kernel applies continuity once per Picard iteration

`pipe1d.cu:1875`: `A_new_cont = A_curr − local_dt·flux_Q[c]/L` with
`A_curr = A_new_cont` at `pipe1d.cu:1932`, inside the `it = 0..7` Picard loop.
The comment above it says the update "anchors on the start-of-substep area so
the Picard loop does not advance A multiple times per substep" — but the code
anchors on the **iterate** `A_curr`, not `A_orig`. Consequences:

- A advances `k·dt·flux/L` per substep, `k` = iterations executed (1–8).
- The convergence test `dA < tol_A` can never trigger while `flux_Q ≠ 0`,
  because `dA = dt·flux/L` is constant every iteration.
- The stall detector then forces dt-halving retries whose accepted iterate
  over- or under-advances A by a step-varying factor.

Fix direction: `A_new_cont = A_orig − local_dt·flux_Q[c]/L` (compute once);
Picard should iterate on `Q` only.

### 3. First-substep boundary flux uses `cell_y == 0` as WSE

`pipe1d.cu:1364`: `H_end = cell_y ? cell_y[c] : H_c;` — the boundary branch
reads `cell_y[c]` unconditionally. `cell_y` is memset to 0 at mesh build
(`pipe1d.cu:1042`), so on the very first substep `H_end = 0` (absolute
elevation zero), producing a huge spurious `c_wave` and boundary flux into
cells from any wet node. The interior branch has the correct guard
(`(cell_y && cell_y[c] != 0.0) ? cell_y[c] : H_c_geom`, `pipe1d.cu:1213`);
the boundary branch does not. Also note the `cell_y[c] != 0.0` test treats a
legitimate WSE of exactly 0 as "unavailable" (minor).

**Probe:** single wet node (0.3 m) + dry pipe: step 0 creates **+0.22 m³**
(fully_dynamic) / **+0.56 m³** (diffusion) out of nothing.

### 4. `fully_dynamic` inlet override makes inlet nodes infinite sources

`pipe1d.cu:2052-2060`: cells touching an inlet node get `cell_Q_new[c] = 0.0`.
The node mass balance (`pipe1d.cu:2279-2280`) therefore accumulates 0 for the
inlet node, so it is **never debited** — while the boundary-face flux
(`pipe1d.cu:1322-1396`, which has *no* inlet override) keeps transferring
water from the node into the pipe based on head difference.

**Probe (inlet node with 0.3 m, dry pipe, fully_dynamic):** node depth pinned
at exactly `0.30000` for 10 steps while pipe volume grows 0.81 → 0.84 m³ —
water from nowhere, forever. In a real run the inlet exchange then stops
capturing once `rem_node_storage = 0` (capture is limited by node storage,
`swe2d_gpu.cu:4671-4679`), so the coupled behaviour is: capture a small fixed
volume, stop capturing, then leak phantom water into the pipe indefinitely.

### 5. Node mass balance does not limit outflow by available storage

`pipe1d.cu:2307-2310`: `dh = dt·node_net_q/area; d = max(0, d + dh)`. If a node
cannot supply `|node_net_q|·dt`, the depth clamps at 0 but the receiving node
is still credited the full amount → mass created. Probe (single cell):
**+0.019 m³** created the step the upstream node emptied.

---

## P1 — Physics / formulation defects

### 6. "Wave speed" `sqrt(g·ΔH/L)` is dimensionally wrong

`pipe1d.cu:1265` (interior) and `pipe1d.cu:1365` (boundary):
`c_wave = sqrt(g·|ΔH|/L)`. Units are `m^0.5/s`, not `m/s`; it depends on the
head *difference* and sub-cell length rather than local depth. Correct HLLE
celerity is `sqrt(g·A/T)` (hydraulic depth). Consequences:

- At mild gradients / long cells it is 10–100× too small → almost no
  numerical diffusion → persistent odd-even checkerboard. **Probe
  (diffusion, 10 sub-cells):** node depth and Q oscillate ±0.022–0.076 m³/s
  with alternating sign every step and never converge; total mass sloshes
  ±0.038 m³/step in and out of existence. The oscillation also blocks net
  conveyance (V1 stayed ~0 over 8 steps).
- At steep gradients it is too large → the CFL clamp does the real work.

The boundary-branch flux `F = ΔH·c_wave·A·dir` is likewise a fabricated
relaxation law (`∝ ΔH^1.5·A·sqrt(g/L)`), which is why it never matches
`cell_Q` (finding 1).

### 7. Regime override (normal-flow cap) triggers in wrong regimes and uses wrong Froude depth

- Trigger `Sf_HGL < S0 − 1e-6` (`pipe1d.cu:1647, 2010`) also fires for
  **backwater** (`Sf < 0 < S0`) and for **pressurised** flow, capping `|Q|` at
  open-channel Manning normal flow `Q_n` — wrong physics for surcharge and
  for adverse flow (SWMM's `checkNormalFlow` is for downhill supercritical
  flow only).
- Froude number uses `R_h = A/P` instead of hydraulic depth `A/T`
  (`pipe1d.cu:1633, 1890, 1996`). For a circular pipe `R_h ≈ D/4` vs
  `A/T ≈ 0.39·D` near half-full → `Fr` overestimated by ~25 % → override
  fires too early; same wrong `Fr` feeds the `sigma` inertial damping
  (`pipe1d.cu:1889-1893`).
- The override caps `Q` but not the flux-kernel `F` (see finding 1).

### 8. Pipe-end weir/orifice exchange mixes invert datums (head bias ±sub_len/2·S0)

`pipe1d.cu:3356` sets `node_depth[n] = cell_h[c_pipe]` (depth above the
sub-cell **midpoint** invert); `pipe1d.cu:3185-3186` then computes
`node_head = node_invert[n] + node_depth[n]` (node datum). True pipe WSE is
`cell_invert[c_pipe] + cell_h[c_pipe]`. Bias = `node_invert − cell_invert`:

- FROM-node (upstream) pipe-end: **+sub_len/2·S0** overestimate → outflow
  overestimated / capture underestimated; sign of `dH` can flip.
- TO-node (downstream) pipe-end: **−sub_len/2·S0** underestimate → outflow
  suppressed.

At the default `max_cell_length = 25` m and `S0 = 2 %`, the bias is **0.25 m**
on a weir/orifice equation sensitive to centimetres. Fix: convert
`cell_h` to the node datum (`node_depth = cell_h + cell_invert −
node_invert`) or use `cell_y[c_pipe]` directly as the head.

### 9. Junction surcharge path destroys mass three ways

- `swe2d_junction_bc_kernel` clamps junction depth to `[0, node_max_depth]`
  (`pipe1d.cu:3579-3583`). Since `junction_node` = *all* interior storage
  nodes (`coupling.py:532-535`), any surcharge volume above `node_max_depth`
  is silently deleted every step. (The mass-balance kernel itself
  deliberately does *not* clamp — `pipe1d.cu:2283-2291` — so the BC kernel is
  the sole destroyer.)
- The clamp runs **before** `swe2d_pipe1d_junction_overflow_kernel`
  (order at `pipe1d.cu:2657-2665`), capping WSE at ≤ rim, so the
  rim-triggered overflow can never fire.
- Even if it fired, the overflow kernel writes to `dev->drain_ws.d_q_cell`
  (`pipe1d.cu:4073`), which is **never folded** into
  `d_external_source_mps` (the fold at `swe2d_gpu.cu:8282-8286` reads
  `coupling_ws.d_drainage_q`) → node is debited, surface never credited.
- Additionally `d_node_crown` is allocated but never populated
  (`pipe1d.cu:892-894`), so the crown-clamp branch is dead code.

### 10. 2D-coupled outfall nodes are mass sinks

Outfall nodes from `cfg.outfalls` are marked `is_boundary`
(`swe2d_gpu.cu:9664-9674`) → the mass-balance kernel skips them
(`pipe1d.cu:2304`) → pipe water arriving at the outfall node is removed from
the pipe cell (via boundary flux) but never recorded anywhere. The legacy
outfall exchange (`swe2d_gpu.cu:4710-4818`) can only discharge *pre-existing*
node storage to the surface, and for `zero_storage` outfalls it forces
`node_depth = 0` (`swe2d_gpu.cu:4762`), so a pipe discharging to a
2D-coupled outfall simply vanishes instead of wetting the surface cell.

---

## P2 — Code defects

### 11. Out-of-bounds device read: `node_is_inlet[vnode]`

`pipe1d.cu:1283-1285`: in the interior-face branch,
`node_is_inlet[cell_to_node[c]]` / `[cell_from_node[c]]` is read without a
`< n_nodes` guard. For interior sub-cell faces those indices are virtual-node
indices `≥ n_nodes`, but `d_node_is_inlet` is sized `n_nodes`
(`pipe1d.cu:890`). The garbage flag randomly zeroes interior fluxes
(`F = 0.0` at `pipe1d.cu:1286-1288`). The fully-dynamic kernel's inlet check
correctly guards (`pipe1d.cu:2053-2056`); the flux kernel's does not.

### 12. Elliptical `A_open` table formula inverted (lower half)

`pipe1d.cu:559-577` (`pipe1d_compute_pipe_end_A_open_table`, host lambda):
for `yRel ≤ 1` it returns `πab − segArea` (the *complement* of the filled
area). `A_open(0) ≈ A_full` (full area at zero submergence), `A_open(D/2) ≈
2.86ab` instead of `πab/2`. Only affects elliptical pipe-ends, but there the
exchange orifice area is grossly overestimated. (The device-side elliptical
geometry `xsect_getAofY_elliptical` is correct.)

### 13. `max_cell_length` truncated to `int32`

`pipe1d.cuh:264` declares `int32_t max_cell_length`; `coupling.py:2003` passes
`int(dsoa.max_cell_length)`. `0.5 m → 0` silently disables subdivision;
`2.5 m → 2`.

### 14. Latent double-fold of `d_pipe_end_q_2d`

The pipe-end exchange Q is folded into `d_external_source_mps` twice per
timestep: inside `swe2d_pipe1d_step` (`pipe1d.cu:2802-2806`) and again in
`swe2d_gpu_compute_coupling_full_on_device` (`swe2d_gpu.cu:8299-8304`).
Currently masked only because the latter zeroes `d_external_source_mps` first
(`swe2d_gpu.cu:8009`). Any reordering (or a second coupling call without a
zero) double-counts the exchange. Pick one fold site.

### 15. Smaller items

- `swe2d_drainage_apply_delta_kernel` clamps node depth at `node_max_depth`
  (`swe2d_gpu.cu:4831-4834`) — destroys backwater volume at inlet nodes
  (diffusion mode).
- `swe2d_pipe1d_step` cudaMalloc/cudaFree three work arrays every call
  (`pipe1d.cu:2562-2568`) — performance only.
- Host wrappers launch on the default stream while surrounding work uses
  `dev->d_stream`; currently safe because `d_stream` is a plain blocking
  `cudaStreamCreate` (`swe2d_gpu.cu:5309`), but it serialises everything.
- `volume_decomposition` is hardcoded to 1 at `pipe1d.cu:2592`.

---

## Probe evidence (GPU, this build)

| Probe | Setup | Result |
|---|---|---|
| 1 | Closed 2-node, 1 link, 10 sub-cells, SLOT, d0=1 m | diffusion −90.9 % mass / 200 steps; fully_dynamic −73.3 % |
| 2 | Single cell, full pipe, surcharge off | Conserved (A pinned at clamp) — mismatch hides when A can't move; +0.019 m³ created when node empties (finding 5) |
| 3a | Single cell, partial fill | Small leak −2e-4 m³/step (near-steady mismatch, finding 1) |
| 3b | 10 sub-cells, partial fill | diffusion: persistent checkerboard, ±0.022 m³/step slosh (findings 1, 6); fully_dynamic: growing monotone leak (findings 1, 2) |
| 4 | Inlet node 0.3 m + dry pipe | fully_dynamic: V0 pinned 0.30000, pipe fills from nowhere (finding 4); diffusion: ±0.076 m³/s Q oscillation (finding 6); step-0 creation +0.22/+0.56 m³ (finding 3) |

## Recommended fix order

1. Make node mass balance consume the same boundary-face fluxes the cells use
   (finding 1) — this is the core conservation defect.
2. Anchor fully-dynamic continuity on `A_orig`, once per substep (finding 2).
3. Guard the boundary branch against `cell_y == 0` first-substep (finding 3).
4. Remove the `cell_Q = 0` inlet override and instead debit inlet nodes by
   the boundary-face flux (finding 4).
5. Fix `c_wave` to `sqrt(g·A/T)` (finding 6) — should also kill the
   checkerboard.
6. Fix pipe-end datum bias (finding 8), junction ordering/fold (finding 9),
   outfall storage handling (finding 10).
7. Memory safety (finding 11), elliptical table (finding 12), int truncation
   (finding 13), single fold site (finding 14).
