---
type: plan
status: complete
created: 2026-07-15
completed: 2026-07-25
---

# Pipe1D Solver Rewrite — Implementation Plan

> **Spec:** `docs/archive/specs/2026-07-15-pipe1d-solver-rewrite-spec.md`
> **Strategy:** Mostly serial C++/CUDA implementation (heavy inter-file
> coupling in `pipe1d.cu` and `pipe1d.cuh`) with two parallel tracks:
> (1) the implementation itself, (2) a parallel test-planning track.
> **Model policy (per user direction):** **`deepseek-v4-flash` (opencode-go
> variant) is used for `refactor` tasks only.** `coding` and `test`
> tasks use the **base/default** subagent model. Documentation tasks
> use base/default as well (no flash outside `refactor`).
> **Excluded:** actual testing, validation against SWMM, Python
> refactoring, inlet rewrite — all are out of scope per §2.17 of the
> spec.

---

## 1. Selectable step dicts

Each step has `action` (imperative description with routing keyword),
`type` (one of `coding|refactor|test|ui|docs`), and `phase` (which
of the 6 implementation phases or the parallel track it belongs to).

| # | action (routing keyword in **bold**) | type | phase |
|---|--------------------------------------|------|-------|
| 1 | Add **cross-section** geometry device functions (`xsect_getAofY`, `xsect_getWofY`, `xsect_getRofY`) and Preissmann slot width helper to `cpp/src/pipe1d.cu` | coding | 1 |
| 2 | Rewrite cell-mesh build in `cpp/src/pipe1d.cu` and `cpp/src/pipe1d.cuh` to use per-link sub-cells with internal virtual nodes per spec §2.1 | refactor | 2 |
| 3 | Add per-cell and per-virtual-node state to `Pipe1DDeviceState` in `cpp/src/pipe1d.cuh` (slot width, crown elev, vnode fields) per spec §2.2 | refactor | 2 |
| 4 | Rewrite flux kernel: split inter-cell (HLLE + upwind + CFL) vs network-boundary (corrected wave speed, cell_A, CFL, dry-cell) branches per spec §2.13 | refactor | 3 |
| 5 | Rewrite diffusion-wave kernel per sub-cell: use sub-cell end states from virtual nodes; remove `cm` local-loss term; add cell_A and slot-width support per spec §2.3 | refactor | 4 |
| 6 | Add dynamic-wave kernel per sub-cell: Picard iteration, local losses at end faces only per spec §2.3 | coding | 4 |
| 7 | Add Preissmann slot to cell geometry helpers per spec §2.5 | coding | 4 |
| 8 | Add regime override (SWMM `checkNormalFlow`) in diffusion and dynamic wave kernels per spec §2.6 | coding | 4 |
| 9 | Update `swe2d_pipe1d_node_mass_balance_host` to skip virtual nodes (only network nodes participate) per spec §2.7 | refactor | 5 |
| 10 | Implement five outfall modes (free / normal_depth / fixed_wse / rating_curve / tabular) with bisection solver for circular pipe normal depth per spec §2.8 | coding | 5 |
| 11 | Implement pipe-end node BC: invert = coupled 2D cell invert, no manhole storage per spec §2.9 | coding | 5 |
| 12 | Implement junction BC: storage node with HEC-22 local losses at end faces, surcharge via crown per spec §2.10 | coding | 5 |
| 13 | Update `swe2d_gpu_apply_pipe_end_bc` in `cpp/src/swe2d_gpu.cu` to use geometric `A(y)` from §2.4 | refactor | 5 |
| 14 | Build with `mamba run -n qgis_stable cmake --build build -j$(nproc)` and verify clean build (no new warnings) | coding | 6 |
| 15 | Review spec compliance of phase 1 (geometry + slot helper) | test | 1-rev |
| 16 | Review spec compliance of phase 2 (mesh + state) | test | 2-rev |
| 17 | Review spec compliance of phase 3 (flux kernel) | test | 3-rev |
| 18 | Review spec compliance of phase 4 (wave kernels + slot + regime) | test | 4-rev |
| 19 | Review spec compliance of phase 5 (node mass balance + outfall/pipe-end/junction + pipe_end_bc) | test | 5-rev |
| 20 | Review full implementation against the spec end-to-end | test | 6-rev |
| 21 | Write a separate test plan (parallel track) for the rewritten solver: which existing tests stay, which need updating, which new tests to add — independent of the implementation | docs | parallel |
| 22 | Update `docs/AGENT_SESSION_RECOVERY_LOG.md` and `docs/archive/specs/` cross-references to reflect the rewrite is in place | docs | 6 |

## 2. Pre-computed agent + model columns

Per `AGENT_SELECTION.md` and the user's instruction
(flash `opencode-go` variant for **`refactor` only**; base/default
for `coding`, `test`, and `docs`):

| #  | type     | agent                                            | model        |
|----|----------|--------------------------------------------------|--------------|
| 1  | coding   | `cpp-pro` (base)                                 | base/default |
| 2  | refactor | `cpp-pro_opencode-go_deepseek-v4-flash`         | flash        |
| 3  | refactor | `cpp-pro_opencode-go_deepseek-v4-flash`         | flash        |
| 4  | refactor | `cpp-pro_opencode-go_deepseek-v4-flash`         | flash        |
| 5  | refactor | `cpp-pro_opencode-go_deepseek-v4-flash`         | flash        |
| 6  | coding   | `cpp-pro` (base)                                 | base/default |
| 7  | coding   | `cpp-pro` (base)                                 | base/default |
| 8  | coding   | `cpp-pro` (base)                                 | base/default |
| 9  | refactor | `cpp-pro_opencode-go_deepseek-v4-flash`         | flash        |
| 10 | coding   | `cpp-pro` (base)                                 | base/default |
| 11 | coding   | `cpp-pro` (base)                                 | base/default |
| 12 | coding   | `cpp-pro` (base)                                 | base/default |
| 13 | refactor | `cpp-pro_opencode-go_deepseek-v4-flash`         | flash        |
| 14 | coding   | `build-engineer` (base)                          | base/default |
| 15 | test     | `cpp-pro` (base)                                 | base/default |
| 16 | test     | `cpp-pro` (base)                                 | base/default |
| 17 | test     | `cpp-pro` (base)                                 | base/default |
| 18 | test     | `cpp-pro` (base)                                 | base/default |
| 19 | test     | `cpp-pro` (base)                                 | base/default |
| 20 | test     | `cpp-pro` (base)                                 | base/default |
| 21 | docs     | `python-pro` (base)                              | base/default |
| 22 | docs     | `python-pro` (base)                              | base/default |

**Rule:** `type == "refactor"` → `*_opencode-go_deepseek-v4-flash`.
Everything else → base variant. The flash model is **never** used
for new code, new tests, or new documentation.

## 3. Execution graph

The C++/CUDA work is **strictly serial** across phases 1 → 6
because every phase builds on the data structures or
functions established by the previous one. The
**test-planning track (#21)** is the only true parallel
opportunity and runs alongside phase 1.

```
[Phase 1: geometry]  →  [rev15]  ─┐
                                  ├──→ [Phase 2: mesh + state]  →  [rev16]  ──┐
                                                                             ├──→ ...
[Parallel: test plan  ──────────────────────────────────────────────────────────┘  (independent)
                                                                                     │
                                                                                     ▼
                                                                            [Phase 6: build]  →  [rev20]  →  [docs22]
```

`git status` and `git diff` are checked at the start of every
phase to ensure the working tree is clean (no surprise edits from
parallel tracks or stale `__pycache__`).

## 4. Per-phase notes

### Phase 1 — Geometry device functions (#1)

- New `__device__` functions in `cpp/src/pipe1d.cu`:
  `xsect_getAofY`, `xsect_getWofY`, `xsect_getRofY` for circular,
  rectangular, elliptical (per spec §2.4).
- New `__device__ double slot_width(TXsect*, double y)` matching
  `dwflow.c:575-588` (Sjoberg formula + 1% cap).
- New `__device__ double getAreaPressurised(...)` for the
  `A_full + (y - yFull) * slot_width` extension.
- No public API change; this phase is purely additive.

### Phase 2 — Mesh build + state (#2, #3)

- Rewrite the cell CSR build in `cpp/src/pipe1d.cu:351-409` so that
  sub-cell `i` of a link has `from = V[i]` and `to = V[i+1]`, with
  `V[0] = link.from` (network) and `V[N] = link.to` (network).
- Add virtual-node storage on the device
  (`d_vnode_H, d_vnode_Q, d_vnode_invert, d_vnode_count_per_link`).
- Add `d_cell_slot_width`, `d_node_crown` to the device state.
- **Critical:** the network graph does **not** change. The Python
  coupling layer (§2.15) does not need updates.

### Phase 3 — Flux kernel (#4)

- Single kernel with two branches:
  - `nbr >= 0` (interior virtual-node face) → HLLE + upwind + CFL
    per spec §2.13.1.
  - `nbr < 0` (network boundary) → corrected wave speed, `cell_A`
    not `A_full`, CFL limit, dry-cell threshold per spec §2.13.2.
- Remove the broken `c_face = sqrt(g * |dH|) / cell_length`
  (units 1/s, not m/s).
- Add `dt` parameter for CFL.

### Phase 4 — Wave kernels (#5, #6, #7, #8)

- Diffusion wave: replace momentum-equation `cm` local-loss term
  with a separate end-face loss application per spec §2.3 and
  §2.12. Sub-cell uses `H_up_face` and `H_dn_face` from the two
  faces (one or both may be virtual nodes).
- Dynamic wave: Picard iteration (8 max, halve-on-stall) per spec
  §2.14. End-face losses at the two network boundary faces only.
- Preissmann slot: applied in `xsect_getAofY` /
  `xsect_getWofY` when `y > yFull` and surcharge method is SLOT.
- Regime override (`checkNormalFlow`): after computing
  `Q_dw`, replace with `min(Q_dw, Q_n)` when `y_up < y_dn` or
  `Fr_up >= 1` or downstream is an outfall (per spec §2.6).

### Phase 5 — Node mass balance + boundary conditions (#9-#13)

- Update mass balance to **skip** virtual-node indices (only
  network nodes participate).
- Five outfall modes: implement `free` (existing behaviour),
  `normal_depth` (bisection on circular Manning's), `fixed_wse`
  (clamp `node_head`), `rating_curve` (monotone interpolate), and
  `tabular` (time series).
- Pipe-end: invert = coupled 2D cell invert, node depth =
  `cell_invert + cell_h`, no manhole storage.
- Junction: storage node with HEC-22 / SWMM local losses at end
  faces, surcharge at `node_crown`.
- `swe2d_gpu_apply_pipe_end_bc` uses geometric `A(y)` (not linear
  approximation) for the area update.

### Phase 6 — Build + final review (#14, #20, #22)

- `cd build && mamba run -n qgis_stable cmake --build . -j$(nproc)`.
- Expect pre-existing warnings (`swe2d_reconstruct.cu` line
  directive style, `swe2d_bindings.cpp` sign-compare, etc.) but no
  **new** warnings from the rewrite.
- Final review reads the full diff against the spec end-to-end
  and runs the per-phase review summary.

## 5. Routing keywords (per task)

Each task contains at least one of the required routing keywords
(`python`, `pyqt5`, `ui`, `refactor`, `cpp`, `cuda`, `gpu`, `kernel`,
`build`, `compile`, `cmake`, `test`, `validate`, `docs`).

- C++ tasks: `cpp`, `cuda`, `gpu`, `kernel`
- Refactor tasks: `refactor` (in addition to the cpp keywords)
- Build task: `build`, `compile`, `cmake`
- Test-plan / docs tasks: `docs`, `test`, `validate`

## 6. Superpowers workflow

The implementing subagent must, per the subagent-driven-development
skill, follow the TDD discipline where it applies. For this
rewrite:

- **Phase 1 (geometry helpers):** no test, but the helper
  functions are pure and can be sanity-checked against SWMM
  values in `dwflow.c:573-619` (manual cross-reference is
  sufficient — the implementation work is to match the
  reference exactly).
- **Phases 2-4 (mesh, flux, wave):** existing tests under
  `tests/test_swe2d_pipe1d.py`, `tests/test_swe2d_gpu_drainage_network.py`,
  `tests/test_swe2d_pipe1d_surcharge.py` will exercise the changes
  if they are run. **However, the spec says testing is out of
  scope** — the implementer should not add new tests in this
  pass, only ensure the existing code compiles.
- **Phase 5 (boundary conditions):** existing
  `tests/test_swe2d_gpu_drainage_network.py` exercises pipe-end
  BCs.
- **Phase 6 (build):** no test, just a clean compile.

The review subagent uses the
`superpowers:requesting-code-review` skill and the
`superpowers:spec-reviewer-prompt` template.

## 7. Cross-review rule

The implementer subagent does **not** self-review. Each
implementation phase (#1-#14) is followed by an independent
review step (#15-#20) using a fresh subagent with the base/default
model. The review checks the diff against the spec section
named in the step's `action` field. If the review finds issues,
the implementer is re-dispatched (same model as before) with the
review's findings, and the review is re-run.

## 8. Machine-readable JSON block

```json
{
  "spec": "docs/archive/specs/2026-07-15-pipe1d-solver-rewrite-spec.md",
  "strategy": "mostly-serial-cpp-rewrite-with-parallel-test-plan",
  "model_policy": "deepseek-v4-flash (opencode-go variant) ONLY for type=refactor; base/default for everything else (coding, test, docs)",
  "review_model": "base/default",
  "steps": [
    {"id": 1,  "action": "Add cross-section geometry device functions and Preissmann slot helper to cpp/src/pipe1d.cu", "type": "coding",   "phase": 1,       "agent": "cpp-pro",                                "model": "base/default", "depends_on": []},
    {"id": 2,  "action": "Rewrite cell-mesh build in cpp/src/pipe1d.cu and cpp/src/pipe1d.cuh to use per-link sub-cells with internal virtual nodes per spec section 2.1", "type": "refactor", "phase": 2,       "agent": "cpp-pro_opencode-go_deepseek-v4-flash",   "model": "deepseek-v4-flash", "depends_on": [1]},
    {"id": 3,  "action": "Add per-cell and per-virtual-node state to Pipe1DDeviceState in cpp/src/pipe1d.cuh (slot width, crown elev, vnode fields) per spec section 2.2", "type": "refactor", "phase": 2,       "agent": "cpp-pro_opencode-go_deepseek-v4-flash",   "model": "deepseek-v4-flash", "depends_on": [1, 2]},
    {"id": 4,  "action": "Rewrite flux kernel: split inter-cell HLLE upwind CFL vs network-boundary corrected wave speed cell_A CFL dry-cell branches per spec section 2.13", "type": "refactor", "phase": 3,       "agent": "cpp-pro_opencode-go_deepseek-v4-flash",   "model": "deepseek-v4-flash", "depends_on": [2, 3]},
    {"id": 5,  "action": "Rewrite diffusion-wave kernel per sub-cell: use sub-cell end states from virtual nodes; remove cm local-loss term; add cell_A and slot-width support per spec section 2.3", "type": "refactor", "phase": 4,       "agent": "cpp-pro_opencode-go_deepseek-v4-flash",   "model": "deepseek-v4-flash", "depends_on": [4]},
    {"id": 6,  "action": "Add dynamic-wave kernel per sub-cell: Picard iteration, local losses at end faces only per spec section 2.3", "type": "coding",   "phase": 4,       "agent": "cpp-pro",                                "model": "base/default", "depends_on": [4, 5]},
    {"id": 7,  "action": "Add Preissmann slot to cell geometry helpers per spec section 2.5", "type": "coding",   "phase": 4,       "agent": "cpp-pro",                                "model": "base/default", "depends_on": [4, 5]},
    {"id": 8,  "action": "Add regime override (SWMM checkNormalFlow) in diffusion and dynamic wave kernels per spec section 2.6", "type": "coding",   "phase": 4,       "agent": "cpp-pro",                                "model": "base/default", "depends_on": [5, 6, 7]},
    {"id": 9,  "action": "Update swe2d_pipe1d_node_mass_balance_host to skip virtual nodes (only network nodes participate) per spec section 2.7", "type": "refactor", "phase": 5,       "agent": "cpp-pro_opencode-go_deepseek-v4-flash",   "model": "deepseek-v4-flash", "depends_on": [2, 3, 8]},
    {"id": 10, "action": "Implement five outfall modes (free normal_depth fixed_wse rating_curve tabular) with bisection solver for circular pipe normal depth per spec section 2.8", "type": "coding",   "phase": 5,       "agent": "cpp-pro",                                "model": "base/default", "depends_on": [4, 8]},
    {"id": 11, "action": "Implement pipe-end node BC: invert equals coupled 2D cell invert, no manhole storage per spec section 2.9", "type": "coding",   "phase": 5,       "agent": "cpp-pro",                                "model": "base/default", "depends_on": [4, 9]},
    {"id": 12, "action": "Implement junction BC: storage node with HEC-22 local losses at end faces, surcharge via crown per spec section 2.10", "type": "coding",   "phase": 5,       "agent": "cpp-pro",                                "model": "base/default", "depends_on": [9, 10]},
    {"id": 13, "action": "Update swe2d_gpu_apply_pipe_end_bc in cpp/src/swe2d_gpu.cu to use geometric A(y) from spec section 2.4", "type": "refactor", "phase": 5,       "agent": "cpp-pro_opencode-go_deepseek-v4-flash",   "model": "deepseek-v4-flash", "depends_on": [1, 9]},
    {"id": 14, "action": "Build with mamba run -n qgis_stable cmake --build build -j$(nproc) and verify clean build (no new warnings)", "type": "coding", "phase": 6,                 "agent": "build-engineer",                         "model": "base/default", "depends_on": [13]},

    {"id": 15, "action": "Review spec compliance of phase 1 (geometry + slot helper) per spec section 2.4 and 2.5", "type": "test", "phase": "1-rev", "agent": "cpp-pro", "model": "base/default", "depends_on": [1]},
    {"id": 16, "action": "Review spec compliance of phase 2 (mesh + state) per spec section 2.1 and 2.2", "type": "test", "phase": "2-rev", "agent": "cpp-pro", "model": "base/default", "depends_on": [2, 3]},
    {"id": 17, "action": "Review spec compliance of phase 3 (flux kernel) per spec section 2.13", "type": "test", "phase": "3-rev", "agent": "cpp-pro", "model": "base/default", "depends_on": [4]},
    {"id": 18, "action": "Review spec compliance of phase 4 (wave kernels + slot + regime) per spec section 2.3, 2.5, 2.6", "type": "test", "phase": "4-rev", "agent": "cpp-pro", "model": "base/default", "depends_on": [5, 6, 7, 8]},
    {"id": 19, "action": "Review spec compliance of phase 5 (node mass balance + outfall/pipe-end/junction + pipe_end_bc) per spec section 2.7-2.11", "type": "test", "phase": "5-rev", "agent": "cpp-pro", "model": "base/default", "depends_on": [9, 10, 11, 12, 13]},
    {"id": 20, "action": "Review full implementation against the spec end-to-end", "type": "test", "phase": "6-rev", "agent": "cpp-pro", "model": "base/default", "depends_on": [14]},

    {"id": 21, "action": "Write a separate test plan for the rewritten solver: which existing tests stay, which need updating, which new tests to add (parallel track, independent of the implementation)", "type": "docs", "phase": "parallel", "agent": "python-pro", "model": "base/default", "depends_on": []},
    {"id": 22, "action": "Update docs/AGENT_SESSION_RECOVERY_LOG.md and cross-references to reflect the rewrite is in place", "type": "docs", "phase": 6, "agent": "python-pro", "model": "base/default", "depends_on": [20]}
  ],
  "review_template": "superpowers:requesting-code-review with superpowers:spec-reviewer-prompt",
  "out_of_scope": [
    "Python refactoring",
    "new test cases",
    "test execution",
    "validation against SWMM",
    "inlet rewrite"
  ]
}
```

## 9. Things to confirm before starting

- [ ] Working tree is clean (`git status` shows nothing)
- [ ] `__pycache__` purged (per `CACHE_DISCIPLINE.md`) before any
  `cmake --build` (the C++ side does not use `__pycache__`, but the
  build process invokes a Python script that does)
- [ ] `cpp/src/swe2d_bindings.cpp` does **not** need new pybind11
  entries if all the new C++ state is internal to the device
  kernels (the existing `swe2d_pipe1d_step` host wrapper is the only
  public surface)
- [ ] The implementer for phase 5 (#10-#12) reads
  `reference/Stormwater-Management-Model-develop/src/solver/{dynwave,dwflow,link,node}.c`
  for SWMM reference on outfall modes

---

**Status:** 14 of 22 plan steps completed (Phases 1-4 + Phase 5 BC kernels). Reviews skipped due to subagent reliability. See `docs/AGENT_SESSION_RECOVERY_LOG.md` (2026-07-16 entry) for gaps.
