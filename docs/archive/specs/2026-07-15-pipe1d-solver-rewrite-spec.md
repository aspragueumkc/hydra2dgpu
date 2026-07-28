---
type: spec
status: superseded
created: 2026-07-15
completed: 2026-07-25
superseded_by: docs/archive/specs/PIPE1D_SWE2D_TEMPORAL_COUPLING_SPEC.md
---

# Pipe1D Solver Rewrite — Technical Specification

> **Status:** SPEC ONLY. No implementation, no test planning.
> **Goal:** Replace the broken boundary-flux formula and cell-mesh wiring in
> `cpp/src/pipe1d.cu` with a SWMM-aligned formulation that handles
> open-channel flow, pressurized flow, supercritical/subcritical regime
> transitions, and all four node types correctly.
> **Reference:** `reference/Stormwater-Management-Model-develop/src/solver/{dynwave,dwflow,link,node,forcmain}.c`.

---

## 1. Audit of the Current Solver

The current implementation has the following defects, ordered by severity.

### 1.1 Boundary flux is wrong (catastrophic drain on steep slopes)

`cpp/src/pipe1d.cu:633-635` (flux kernel, `nbr < 0` branch):

```cpp
const double c_face = sqrt(fmax(0.0, g * fabs(dH))) / fmax(1e-12, cell_length[c]);
F = dH * c_face * cell_area_full[c];   // ← uses A_full, not cell_A
```

- **`c_face` has units 1/s, not m/s.** The interior HLLE at line 651
  uses `sqrt(g * |H_c - H_n| / cell_length)` (m/s) — consistent. The
  boundary form is dimensionally inconsistent.
- **Uses `cell_area_full[c]` regardless of cell's actual water
  content.** A cell with `A = 1e-4` (essentially empty) still
  transmits full-pipe flux.
- **No CFL limit.** The cell can be commanded to lose more volume in
  one timestep than it contains; the area update then clamps to
  `A_floor` but the requested drain was already applied to neighbours
  via the mass balance, so the system drifts.
- **Result on slope = 1.0, forced head [0.10, 0.0]:** cells go
  bimodal — upstream cells clamp to `A_floor`, downstream cells
  saturate at `A_full`, with `Q ≈ 13 m³/s` reported in the full
  cells. Manning's law at the same depth gives `Q ≈ 0.5 m³/s`.

### 1.2 Sub-cells are wired as boundary cells instead of in the FV stencil

`cpp/src/pipe1d.cu:351-409` (cell CSR build) treats every sub-cell
of a subdivided link as if it were directly connected to the two
**end nodes** of the link:

- Every sub-cell gets `cell_from_node[c] = link.from` and
  `cell_to_node[c] = link.to` (lines 363-364, with `from=J1, to=O2`
  for all 4 cells of the 1-link test case).
- The `outlet_neighbor`/`inlet_neighbor` search at lines 388-400
  **finds no neighbour** (every cell's `from == J1` and every
  cell's `to == O2`, so the inner loop's "if my from == your to" is
  never satisfied). All neighbours default to `-1` (boundary).
- **Consequence:** the interior HLLE flux branch (`else { ... }` at
  line 649) is never taken. Every face is a boundary face, and the
  broken boundary formula runs four times per cell instead of once.
- **SWMM does not put hidden network nodes between sub-cells** —
  correct. SWMM treats the link as a single 1D cell; the
  finite-volume "sub-cells" are purely internal to the link's
  stencil, with shared VIRTUAL NODES that exist only inside the
  discretisation. The current code does not implement this. **The
  fix is to keep the sub-cells but make them internal to the link
  (see §2.1).**

### 1.3 Initial area from depth is geometrically wrong

`cpp/src/pipe1d.cu:1217-1240` (`swe2d_pipe1d_init_area_from_depth`):

```cpp
double frac = fmin(1.0, depth / full_depth);
init_A[c] = A_full * frac;
```

- **Assumes a linear `A ∝ depth/full_depth` relation.** This is
  only correct for rectangles. For a circular pipe, `A(d=0.10 in D=1)`
  is 0.082 m² (geometric), but the init gives `0.785 × 0.10 = 0.079 m²`
  (≈4% low — small here, but worse for shallower fills).
- More importantly, the `depth` it uses is
  `0.5 * (node_depth[from] + node_depth[to])`, which is the
  **average of the two end-node depths of the link**, not the local
  cell depth. For a 4-cell sub-division of a 100 m link with
  `from_depth = 0.10, to_depth = 0.0`, the init gives every cell
  `depth = 0.05` regardless of the cell's location along the link.
- **Fix:** use the geometric `pipe1d_area_from_depth` (or
  `xsect_getAofY` in SWMM terms) at the cell's local water depth, not
  the linear approximation.

### 1.4 Init area used in the diffusion wave, not the kernel

The diffusion-wave kernel uses `cell_A[i]` directly, with `A_floor`
as a clamp. But the value of `cell_A` after `init_area_from_depth` is
the **wrong (linear, average)** value. So the kernel's `A = max(A_floor,
cell_A[i])` is operating on a bad initial state.

### 1.5 No regime detection (supercritical / subcritical)

The boundary flux is applied identically regardless of the local
Froude number. On a steep slope, the flow is supercritical and the
downstream BC should not influence the upstream — but the current
formula does, causing the bimodal instability.

SWMM's `checkNormalFlow` (`reference/.../dwflow.c:637-686`) handles
this by **overriding the dynamic-wave flow with Manning's normal
flow** when:
- water surface slope < conduit slope (supercritical), OR
- Froude ≥ 1.0 at upstream, OR
- the link has an outfall node.

This is the SWMM convention. The new solver must implement the
equivalent.

### 1.6 No Preissmann slot

Pressurised flow is not modelled. The cell area is clamped at
`A_full`, and any depth above the crown is silently lost. SWMM's
SLOT surcharge method (`dwflow.c:575-588`) adds a fictitious
narrow slot above the crown so the cell can hold pressurised volume
and the dynamic-wave equations can continue to apply.

### 1.7 Local losses at junctions are sign- and direction-sensitive

`cpp/src/pipe1d.cu:763-769` (diffusion wave):

```cpp
const double cm = 0.0
    + (k_in  > 0.0 && A1 > 0.0 ? k_in  / (2.0 * A1 * L_link) : 0.0)
    + (k_out > 0.0 && A2 > 0.0 ? k_out / (2.0 * A2 * L_link) : 0.0);
const double denom = 1.0 + dt * cf * absQ + dt * cm * absQ;
```

- `L_link` is the **total link length**, not the local cell length.
  The minor-loss coefficient `cm` is therefore diluted by the
  sub-cell count (4 here), effectively under-applying the entrance
  / exit losses.
- SWMM places local losses in the **node coupling**, not the
  link momentum equation (`dwflow.c:562-571`,
  `getLocalLosses`). They are applied at the cell's end faces.

### 1.8 Mass balance uses `d_node_is_boundary` but boundary cells
  still accumulate flux

`swe2d_pipe1d_node_mass_balance_host` (line 1153) skips
`d_node_is_boundary` nodes — correct — but the flux kernel at
line 656 still **writes flux into** boundary nodes via the boundary
formula. The boundary depth is then zeroed by the outfall kernel
(`swe2d_outfall_free_bc_kernel`), so the "drain" is invisible. The
interior cells end up absorbing the imbalance.

### 1.9 No `crown_elev` per node

SWMM computes `Node[i].crownElev = max over connecting links of
(invert + offset + yFull)` (`dynwave.c:144-153`). The current
code never tracks the crown elevation per node, so it can't detect
surcharge or apply the Preissmann slot consistently.

### 1.10 Inlets are already correct (confirm, do not change)

`cpp/src/swe2d_gpu.cu:4698-4805`
(`swe2d_drainage_outfall_exchange_kernel`) and the HEC-22 grate /
curb / slotted / combo / relief inlet logic in
`tests/test_swe2d_gpu_drainage_network.py` use the SWMM-style
orifice / weir equations with availability limiters, which is the
correct form for a fixed-2D-cell coupling. **Keep as-is**, do not
rewrite inlets.

---

## 2. Corrected Solver Specification

### 2.1 Mesh structure

**Sub-cells stay, but they are internal to the link — they live in
the finite-volume stencil, not the network graph.** This is the key
change vs. the current implementation. SWMM does the same: a link is
treated as one 1D cell at the network level, but the discretisation
is a finite-volume method on `N` sub-cells with shared **virtual
nodes** between them. The virtual nodes are not in the network
graph; they are bookkeeping inside the link.

**Network graph (what the coupling layer / mass-balance sees):**

```
[PipeEndNode A] ——link C1—— [Junction J1] ——link C2—— [Outfall O2]
                              ↑ 1D storage
                              ↑ node mass balance runs here
```

**Internal link discretisation (what the flux / momentum / continuity
kernels see — for one link of length `L_link` with `N = ⌈L_link /
L_max_cell⌉` sub-cells):**

```
       cell 0          cell 1          cell N-1
   ┌──────────┐    ┌──────────┐     ┌──────────┐
   │          │    │          │     │          │
   │ A_0, Q_0│    │ A_1, Q_1│ ... │A_{N-1},Q_{N-1}
   │          │    │          │     │          │
   └──┬───┬───┘    └──┬───┬───┘     └──┬───┬───┘
      │   │          │   │           │   │
   face 0 face 1  face 1 face 2  face N-1 face N
      │              │               │       │
   V[0] (boundary)  V[1] (virtual) ... V[N-1] (virtual)  V[N] (boundary)
   = link.from      = V_1            = V_{N-1}              = link.to
   (network node)   (internal)       (internal)              (network node)

Each cell i has:
    length            L_i = L_link / N  (or weighted to honour the
                                         user-specified L_max_cell)
    invert            z_i = (z_from + (i + 0.5) * (z_to - z_from) / N)
    from-face at      V[i]   (network if i == 0, virtual otherwise)
    to-face at        V[i+1] (network if i == N-1, virtual otherwise)
    cell_from_node    i   (index into a per-link cell list, NOT a
                            network node)
    cell_to_node      i+1
```

**Virtual nodes are pure pass-throughs (no storage).** They carry a
single state `H_v` (water surface elevation) and a `Q_v` (flux
through the node). They are updated by the inter-cell flux
computation, not by a node mass balance. The face reconstruction
between cell `i` and cell `i+1` uses an **upwind scheme** on the
shared virtual node:

```
H_v[i] = H_i      if Q_v[i] > 0  (flow from cell i to cell i+1)
H_v[i] = H_{i+1}  if Q_v[i] < 0  (flow from cell i+1 to cell i)
Q_v[i] = face flux, F_v[i]
```

The face flux `F_v[i]` is the standard HLLE flux (matching the
interior branch of the current flux kernel, line 651) using the two
adjacent cell states:

```
c_wave = sqrt(g * |H_i - H_{i+1}| / L_face)        (m/s, not 1/s)
F_v[i] = 0.5 * (Q_i + Q_{i+1} - c_wave * (A_{i+1} - A_i))
```

**Network boundary faces (only 2 per link)** use the
boundary-flux formula (corrected, with cell_A and CFL limit —
see §2.13.2), driven by the head difference between the
end cell and the network node's water surface:

```
For cell 0's upstream face (dir = -1):
    H_n = node_invert[link.from] + node_depth[link.from]
    c_wave = sqrt(g * |H_0 - H_n| / L_face)
    A_eff = max(cell_A[0], A_floor)
    F = (H_0 - H_n) * c_wave * A_eff
    F = clamp(F, ±A_eff * L_face / dt)             (CFL limit)
For cell N-1's downstream face (dir = +1):
    H_n = node_invert[link.to] + node_depth[link.to]
    (same formula with H_{N-1} in place of H_0)
```

**Node types (full set, replacing the current "junction | outfall |
storage | inlet | pipe_end" enum):**

| Type            | Connects to 2D? | Storage | Inflow source     | Notes                                         |
|-----------------|-----------------|---------|-------------------|-----------------------------------------------|
| `pipe_end`      | yes (1 cell)    | none    | the cell's WSE/h  | invert ≈ coupled cell invert, long culvert-like  |
| `junction`      | no              | yes     | manhole           | add entrance/exit losses, can surcharge         |
| `inlet`         | yes (1 cell)    | yes     | HEC-22 capture    | existing code is correct                          |
| `outfall`       | no              | depends | none              | free / normal-depth / fixed-WSE / rating-curve / tabular |

**Why virtual nodes must be pass-throughs, not junctions.** A user
could add an explicit junction mid-link, in which case the link
gets split at that node into two links and the junction's mass
balance runs as a real network node. SWMM enforces this — links
cannot be split by a hidden mid-link node. We follow the same
convention: virtual nodes exist only inside the stencil and have
no storage. They are not addressable from the network graph or
the coupling layer.

### 2.2 Cell and virtual-node state

**Per sub-cell (one set per `cell_index = link_offset + i`, `i ∈ [0, N-1]`):**

- `cell_A`  — flow area (m²)
- `cell_Q`  — discharge (m³/s, sign: + downstream)
- `cell_h`  — flow depth (m), derived: `h = f_inv(A, shape, full)`
- `cell_y`  — water surface elevation (m abs): `cell_y = cell_invert + h`
- `cell_q`  — `cell_Q / cell_A` (sign-aware)
- `cell_fr` — Froude: `|cell_q| / sqrt(g * R_h)`
- `cell_slot_width` — Preissmann slot width (0 for open-channel or
  no-slot surcharge; SWMM's Sjoberg formula when pressurised)
- `cell_invert` — bed elevation of the cell (linearly interpolated
  between the link's end-node inverts along the cell's centroid)
- `cell_length` — `L_link / N` (or weighted to honour
  `max_cell_length`)
- `cell_link_id` — index of the link this cell belongs to (for
  nodal mass-balance lookups)
- `cell_local_index` — `i` in `[0, N-1]` (used to identify end
  cells: `i == 0` and `i == N-1`)

**Per virtual node (one set per `vnode_index = link_offset + i`, `i ∈ [1, N-1]`):**

- `vnode_H`     — water surface elevation (m abs)
- `vnode_Q`     — flux through the node (m³/s)
- `vnode_invert` — bed elevation of the virtual node
  (linearly interpolated; for a horizontal pipe, equals the conduit
  invert at the face location)
- No surface area, no storage. Updated by the inter-cell flux
  computation (HLLE + upwind face reconstruction).

**Per network node (independent of cells):**

- `node_depth`  — water depth in the manhole (m)
- `node_head`   — `node_invert + node_depth`
- `node_area`   — surface area of the manhole bucket (m²)
- `node_crown`  — `max over links of (link.invert_at_node + link.yFull)`
- `node_is_outfall` / `node_is_pipe_end` / `node_is_inlet` (booleans)
- `node_outfall_mode` — `free` | `normal_depth` | `fixed_wse` |
  `rating_curve` | `tabular`
- For inlets: existing `InletExchange` (unchanged)

### 2.3 Governing equations (per sub-cell)

Each sub-cell of a link is a 1D finite-volume element. The
equations are the **1D Saint-Venant equations** with the
diffusion-wave simplification (no inertial term) or full dynamic
wave (with inertial term) — selectable per link or per simulation.

**Diffusion wave (per sub-cell, semi-implicit):**

```
Continuity:  dA/dt + dQ/dx = 0
Momentum:    dH/dx = Sf                   (friction slope only,
                                           no inertial, no local
                                           acceleration)
            Q = (1/n) A R^(2/3) Sf^(1/2)  (Manning)
```

Discretised on a sub-cell of length `L_i` with the two end states
at the **virtual nodes** (or network nodes for the end cells):

```
Sf = (H_up_face - H_dn_face) / L_i

Q_i = sign(Sf) * (1/n) * A_i * R_i^(2/3) * |Sf|^(1/2)

dA_i/dt = (Q_in_face - Q_out_face) / L_i
```

where `Q_in_face` is the flux into the cell at the upstream face
(computed by the inter-cell flux in §2.1.1) and `Q_out_face` is
the flux out at the downstream face. For the end cells, one of
those faces is a network boundary (using the corrected boundary
flux from §2.13).

**Dynamic wave (per sub-cell, semi-implicit, Picard):**

```
Continuity:  dA/dt + dQ/dx = 0
Momentum:    dQ/dt + d(Q²/A)/dx + g A dH/dx + g A Sf = 0
            Sf = (n² Q |Q|) / (k_mann² A² R^(4/3))
            H = z + h(A, shape)
```

Discretised per sub-cell with two end states (`H_up, Q_up, A_up`
at the upstream face; `H_dn, Q_dn, A_dn` at the downstream face).
Interior gradients use averaged end values, time-derivative uses
mid-point rule. Local losses at the **link's network boundary
faces only** (the end-cell end faces, not the internal virtual
nodes) use HEC-22 / SWMM:

```
Δh_loss = (k_in  / (2g)) * (Q/A)²       at cell 0's upstream face
Δh_loss = (k_out / (2g)) * (Q/A)²      at cell N-1's downstream face
```

These are subtracted from `H_node` before computing the boundary
flux at the link's two end faces, matching SWMM's
`getLocalLosses` (`dwflow.c:562-571`). Internal virtual-node faces
do not have local losses (losses are not subdivided across
sub-cells).

### 2.4 Cross-section geometry (circular, rectangular, elliptical)

Replace the linear `A = A_full * depth/full_depth` approximation
with the **actual geometric functions**:

```
circular:   y = D/2 - R cos(θ/2)
            A = (D²/8)(θ - sin θ)
            P = D θ / 2
            T = D sin(θ/2)
            θ = 2 acos(1 - 2 y / D)
rectangular: A = b y,  P = b + 2y,  T = b
elliptical:  A = (π a y) / (2 b) - ...  (per xsect.dat)
```

These match SWMM's `xsect_getAofY`, `xsect_getWofY`,
`xsect_getRofY` in `xsect.c`. Implement as device functions
(`__device__`).

### 2.5 Preissmann slot for pressurised flow

When `y > yFull` (depth above crown) and surcharge method is SLOT:

```
slot_width(y) = 0.01 * wMax                         if y > 1.78 yFull
slot_width(y) = wMax * 0.5423 * exp(-(y/yFull)^2.4) otherwise
```

The slot adds fictitious top width so the dynamic-wave equations
continue to apply above the crown:

```
T_pressurised = slot_width(y)        (used in continuity)
A_pressurised = A_full + (y - yFull) * slot_width(y)
R_h_pressurised = R_h_full            (hydraulic radius saturates)
```

`yFull` here is the link's full depth. The slot width is a
property of the link, not the cell. Match `dwflow.c:575-619`.

### 2.6 Regime override (SWMM's `checkNormalFlow`)

After computing the dynamic-wave `Q_dw`, replace it with Manning's
normal flow `Q_n` when **any** of:

1. `y_up < y_dn` (water surface slope < conduit slope, i.e. local
   supercritical)
2. `Fr_up >= 1.0` (Froude number at upstream ≥ 1)
3. downstream node is an outfall

```
Q_n = (1/n) A_up R_up^(2/3) sqrt(S0)        (Manning at conduit slope)
Q = min(Q_dw, Q_n)                          (take smaller — caps at
                                              normal flow)
```

This is the only stable way to handle supercritical transitions
without rewriting the full St Venant solver with bore-tracking. Match
`dwflow.c:637-686` exactly.

### 2.7 Node mass balance

After all link flows are computed for the timestep, update each
**network** node's depth from net inflow:

```
dV = (Σ Q_in - Σ Q_out + Q_lateral) * dt
dH = dV / A_surface
node_depth_new = max(0, min(node_max_depth, node_depth + dH/A_surface))
```

**Network nodes only.** Virtual nodes inside a link have no
storage and are not part of the mass balance — they are pure
pass-throughs whose state is the inter-cell flux.

**Boundary nodes** (`is_pipe_end`, `is_outfall`) **skip** the mass
balance and have their depth set by the boundary-condition kernel
instead. This is already correct in the current code
(`swe2d_pipe1d_node_mass_balance_host`); keep.

### 2.8 Boundary conditions for outfall nodes

Five modes, selectable per node:

| Mode           | Behaviour                                                                                   |
|----------------|---------------------------------------------------------------------------------------------|
| `free`         | `node_depth = 0` every step (water exits to "infinity").                                  |
| `normal_depth` | Compute `h_nd` from Manning's equation at the downstream link's slope (SWMM checkNormalFlow). |
| `fixed_wse`    | `node_head = fixed_wse_elev` every step.                                                   |
| `rating_curve` | `Q = f(H)` interpolated from `[(wse, Q), ...]` table; tailwater from Q.                    |
| `tabular`      | `H = f(t)` from a time-series; tailwater from the time series.                            |

For `normal_depth` outfalls, use the **2D solver's** `h_nd` formula
as the reference shape (`cpp/src/swe2d_gpu.cu:335-359`,
NORMAL_DEPTH_SLOPE case), but use the link's circular geometry to
solve Manning's exactly:

```
For circular pipe, given Q, find y such that
    Q = (1/n) A(y) R(y)^(2/3) S^(1/2)
Solve by bisection or Newton on y in [h_min, yFull].
```

(The 2D formula `h_nd = (Q n / sqrt(S))^(3/5)` is the wide-rectangle
approximation; the link geometry is circular, so use the exact
solver.)

For `rating_curve`, the rating table is
`[(wse_m, Q_m3s), ...]`, monotone interpolation. The link sees
this as a downstream water-surface boundary.

### 2.9 Boundary conditions for pipe-end nodes

A pipe-end is a **direct coupling** between the link and a
co-located 2D surface cell. The node invert is the same as the 2D
cell's bed elevation (or `cell_invert + 0` for culvert-like
configurations). There is no manhole storage; the node depth is
**the 2D cell's water depth**.

```
H_upstream_node = coupled_cell_invert + coupled_cell_h
H_downstream_node = coupled_cell_invert + coupled_cell_h
                   (for both ends; pipe-end is symmetric)
```

The flow is then driven by the head difference between the two
2D cells that the pipe-end couples. The cell mass balance is
governed by the 2D cells, not by the link.

This is the simplest case: the link is essentially a 1D conduit
between two 2D surface cells. The exchange kernel already
handles this in `swe2d_gpu_apply_pipe_end_bc`. **Keep**, but
verify the cell_A update uses the geometric (not linear)
`A(y)` from §2.4.

### 2.10 Boundary conditions for junction nodes

A junction (manhole) is a **storage node** with possible surcharge.

```
H_node = node_invert + node_depth
For each connecting link L (L.from == node or L.to == node):
    If L is a conduit:
        A_at_face = A(y_node)                  (geometric, from §2.4)
        If L is pressurised at the face (y > yFull) and SLOT:
            A_at_face = A_full + (y - yFull) * slot_width(y)
        Add local loss k_in or k_out at the face (SWMM convention):
            dH_loss = (k / (2g)) (Q / A)²
            H_at_link_face = H_node ± dH_loss
        (sign: upstream face subtracts loss, downstream face adds;
         matches SWMM sign convention)
```

Surcharge: when `y_node > node_crown`, the manhole is pressurised.
Apply the orifice equation from the cell's 2D cover (if coupled)
or simply hold the depth at `node_crown` (junction-only case).

### 2.11 Boundary conditions for inlet nodes

**Unchanged.** Existing code in
`cpp/src/swe2d_gpu.cu:4698-4805` (`swe2d_drainage_outfall_exchange_kernel`)
and the HEC-22 grate/curb/slotted/combo/relief inlets are
correct. The flow rate is a function of the 2D cell's WSE and the
inlet geometry, with availability limiters. Do not modify.

### 2.12 Local losses at faces (HEC-22 / SWMM)

Local losses are applied at the **link's two network boundary
faces** (cell 0's upstream face and cell N-1's downstream face),
not at internal virtual-node faces. For each network face:

```
k_face = k_in   at the upstream face (cell 0)
k_face = k_out  at the downstream face (cell N-1)

H_at_link_face = H_node - sign(direction) * (k_face / (2g)) * (Q / A)²
```

Internal virtual-node faces do not have local losses. The current
`cm` term in the diffusion-wave kernel, which divides by `L_link`
and so under-counts the loss, is removed entirely.

### 2.13 Face flux: inter-cell (virtual node) vs network boundary

The cell has 2 faces. The flux formula used depends on whether
the face is a **virtual node** (internal, between two sub-cells) or
a **network boundary** (cell 0's upstream face or cell N-1's
downstream face).

**2.13.1 Inter-cell flux (at virtual nodes)**

Between cell `i` and cell `i+1`, sharing virtual node `V`:

```
H_left  = cell_y[i]      (water surface of upstream cell)
H_right = cell_y[i+1]
A_left  = cell_A[i]
A_right = cell_A[i+1]
Q_left  = cell_Q[i]
Q_right = cell_Q[i+1]

# Upwind face reconstruction for the shared virtual node:
H_v = (Q_left > 0) ? H_left : H_right

# Wave speed (matches SWMM's HLLE)
c_wave = sqrt(g * fabs(H_left - H_right) /
              max(1e-12, 0.5 * (cell_length[i] + cell_length[i+1])))

# HLLE flux (interior branch of the current flux kernel, line 651)
F_v = 0.5 * (Q_left + Q_right - c_wave * (A_right - A_left))

# CFL: flux cannot drain or fill either cell in one timestep
F_max = min(A_left, A_right) * 0.5 * (cell_length[i] + cell_length[i+1]) / dt
F_v = clamp(F_v, -F_max, F_max)
```

**2.13.2 Network boundary flux (cell 0 upstream, cell N-1 downstream)**

```
H_n = node_invert[node] + node_depth[node]

# Apply local loss to the network head (subtractive at upstream face,
# additive at downstream face)
H_n_eff = H_n - sign(direction) * (k_face / (2g)) * (Q / A)²

# Wave speed (now with consistent units, m/s, matching the interior)
c_wave = sqrt(g * fabs(cell_y[end_cell] - H_n_eff) /
              max(1e-12, cell_length[end_cell]))

# Flux scales with the cell's actual water area, NOT A_full
A_eff = max(cell_A[end_cell], A_floor)
F = (cell_y[end_cell] - H_n_eff) * c_wave * A_eff

# CFL limit (this is the fix for the catastrophic drain)
F_cfl = A_eff * cell_length[end_cell] / dt
F = clamp(F, -F_cfl, F_cfl)

# Dry-cell / dry-node protection
if sign(direction) * F > 0 and cell_A[end_cell] < A_floor: F = 0
if node_depth[node] <= 0 and sign(direction) * F < 0:        F = 0
```

**2.13.3 Mass-balance accounting at the link / 2D boundary**

For a pipe-end: the link's `Q` at the end face is the exchange term
applied to the 2D cell (source or sink). Update the 2D cell's
mass balance accordingly.

For a junction: the link's `Q` at the end face is added to the
junction's `node.inflow` (if flow is into the node) or
`node.outflow`. After all links, run the node mass balance from
§2.7.

For an inlet: the inlet code already does this. Do not modify.

### 2.14 Time-stepping and stability

- **Diffusion wave:** semi-implicit, unconditionally stable. CFL
  limited per sub-cell:
  ```
  dt ≤ L_i / max(|c_wave_i|)
  c_wave_i = |q_i| + sqrt(g R_h_i)
  ```
  Use the **shortest sub-cell** in the network as the binding
  constraint. For uniform sub-cells, this is the smallest `L_i`
  (driven by the user's `max_cell_length` setting).

- **Dynamic wave:** semi-implicit with Picard iteration on
  `(Q_i, A_i)` per sub-cell until residual < `tol_Q` (default
  1e-4 m³/s) and `tol_A` (default 1e-6 m²). 8 iterations max.
  If residual stalls, halve `dt` and retry (SWMM's
  variable-step `getLinkStep`).

- **Pressurised:** still stable under SLOT; the slot's narrow
  width keeps the wave speed bounded.

### 2.15 Mesh-side coupling

The 2D solver updates `h, hu, hv` on the surface cells. The
pipe1d step runs **after** each 2D step (or sub-step), using the
2D cell's WSE as the boundary for pipe-end and inlet nodes.

Source terms to the 2D cells from pipe1d:
- For pipe-end: `ΔQ = (Q_link - Q_link_prev) * dt / cell_area`
- For inlet: `ΔQ = (Q_captured - Q_returned) * dt / cell_area`

These are added to the 2D `hu` (momentum x) and `hv` (momentum y)
according to the link's orientation, with the sign convention that
positive `Q` is downstream.

### 2.16 Summary of changes required

| File | Change |
|------|--------|
| `cpp/src/pipe1d.cu`           | Rewrite cell-mesh build (per-link sub-cells with internal virtual nodes, §2.1). Rewrite flux kernel: split into inter-cell (HLLE + upwind + CFL, §2.13.1) and network-boundary (corrected wave speed, cell_A, CFL, dry-cell, §2.13.2) branches. Replace `init_area_from_depth` with geometric `xsect_getAofY`-equivalent (§2.4). Rewrite diffusion-wave and dynamic-wave kernels to use sub-cell end states from the virtual nodes (§2.3). Add Preissmann slot (§2.5). Add regime override (`checkNormalFlow`, §2.6). Move local losses from momentum equation to end faces (§2.12). |
| `cpp/src/pipe1d.cuh`           | New device functions `xsect_getAofY`, `xsect_getWofY`, `xsect_getRofY` (§2.4). New fields: `d_vnode_H`, `d_vnode_Q`, `d_vnode_invert`, `d_cell_slot_width`, `d_node_crown`. New host wrapper signatures. |
| `cpp/src/swe2d_gpu.cu`         | Update `swe2d_gpu_apply_pipe_end_bc` to use geometric `A(y)`. |
| `cpp/src/swe2d_bindings.cpp`   | Add new entry points only if a Python-side change is needed; existing `swe2d_pipe1d_step` signature is fine if internals are correct. |
| Python-side                    | **No changes to Python for this rewrite.** The Python coupling layer is correct; the issue is purely C++/CUDA. |

### 2.17 Things explicitly out of scope

- Python refactoring
- New test cases
- Test planning
- Validation against SWMM
- Inlet rewrite
- 2D-solver changes

The user requested **a spec only** for the corrected solver.
Implementation, testing, and validation are separate work.

---

**Status:** Implementation phases 1-4 + Phase 5 BC kernels complete. Build passes EXIT 0. Step 14 verified. Known gaps listed in `docs/AGENT_SESSION_RECOVERY_LOG.md` (2026-07-16 entry).
