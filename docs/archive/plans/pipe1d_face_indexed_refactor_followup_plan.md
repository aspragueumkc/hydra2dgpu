---
type: plan
status: complete
created: 2026-07-20
completed: 2026-07-25
---

# Pipe1D Face-Indexed Refactor — Gap-Closure Follow-Up Plan (Parallelized)

**Status:** Proposed (2026-07-20). Closes the gaps remaining after
`docs/pipe1d_face_indexed_refactor_plan.md` (the original Phase 0–4 plan).
**Owner:** Aaron Sprague.
**Strategy:** Maximize parallel subagent dispatch. The nine gaps have a
shallow dependency graph; most can run concurrently. Three independent
waves + one convergence gate.

---

## 1. Goal

Drive the pipe1D face-indexed FVM refactor from "struct + kernel
scaffolded, many legacy paths still live" to "single unified face kernel
handles every face class, legacy node/pipe-end/junction abstractions are
deleted, all Phase-1 tests and all regression suites are green."

## 2. Gap Inventory (from direct code audit, 2026-07-20)

| ID | Gap | File:line |
|---|---|---|
| G1 | `Pipe1DDeviceState` still carries all legacy fields | `cpp/src/pipe1d.cuh:18–172` |
| G2 | Preissmann `d_cell_slot_width` allocated but memset to 0 → slot never expands A above A_full | `cpp/src/pipe1d.cu:1221` |
| G3 | Mesh build only emits face classes {0,1,2,3,5}; class 4 (SURFACE_2D_INLET) and 6 (CULVERT) never built → dead branches in unified kernel | `cpp/src/pipe1d.cu:38,53,65,76,104` |
| G4 | `swe2d_pipe1d_step` still launches junction_bc / junction_overflow / update_node_depth / fold kernels / mark_inlet_nodes / scale_double / pipe_end_clamp_area | `cpp/src/pipe1d.cu:4182–4263, 2336, 2379, 3930, 4173, 4461, 5173` |
| G5 | `swe2d_gpu.cu` still launches `swe2d_culvert_face_flux_kernel` and `swe2d_fold_drainage_q_kernel` | `cpp/src/swe2d_gpu.cu:3016, 3156` |
| G6 | Junction-overflow upload `cudaMemcpy` sized by host `n`, device allocated to `d_n_junctions` → "invalid argument" when host array oversized | `cpp/src/pipe1d.cu:4977–4980` |
| G7 | Two readback bindings (`readback_node_state`, `readback_cell_state`) with two schemas; partial aliasing | `cpp/src/swe2d_bindings.cpp:1918, 2222`; `swe2d/runtime/coupling.py:1490–1559` |
| G8/G9 | `apply_native_device_sources` still calls `swe2d_gpu_compute_coupling_full_on_device` and `swe2d_pipe1d_upload_junction_overflow_state` | `swe2d/runtime/coupling.py:1782, 2067` |

## 3. Dependency Graph

```
                ┌──────────┐
                │  F1      │  strip legacy struct
                │ (G1)     │
                └────┬─────┘
                     │
       ┌─────────────┼─────────────┐
       │             │             │
   ┌───▼───┐    ┌────▼────┐   ┌────▼────┐
   │  F4   │    │   F7    │   │   F2    │   (F2 is independent of F1;
   │ (G4)  │    │  (G7)   │   │  (G2)   │    draw here for layout only)
   └───┬───┘    └────┬────┘   └─────────┘
       │             │
       │       ┌─────▼──────┐
       │       │    F8      │  coupling.py unified wiring
       │       │  (G8/G9)   │
       │       └─────┬──────┘
       │             │
       │   ┌─────────┼──────────┐
       │   │         │          │
   ┌───▼───▼──┐  ┌───▼────┐ ┌───▼───┐
   │   F5     │  │  F3    │ │  F6   │   (all three independent of F1)
   │  (G5)    │  │ (G3)   │ │ (G6)  │
   └────┬─────┘  └───┬────┘ └───┬───┘
        │            │          │
        └────────────┼──────────┘
                     │
              ┌──────▼──────┐
              │     F8      │  depends on F3 + F5 + F6
              │   (G8/G9)   │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │     F9      │  full regression + ASan/UBSan + nsys
              └─────────────┘
```

**Independent leaves (no blockers, can start in wave A):**
F2 (slot init), F3 (face classes 4/6), F5 (delete legacy wrappers),
F6 (upload guard), F7 (readback schema collapse).

**Single-blocker phases:**
- F1 (struct cleanup) — only blocks F4.
- F4 (delete step kernels) — depends on F1 only.
- F8 (coupling wiring) — depends on F3 + F5 + F6 (and F4 finishing).
- F9 (gate) — depends on everything.

## 4. Wave Structure — Maximize Parallel Dispatch

### Wave A — five independent agents in parallel

No inter-agent blockers. All five can be dispatched simultaneously with
`dispatching-parallel-agents`. Each writes to a disjoint slice of the
codebase. Cache discipline + per-agent rebuild + targeted test gate.

| Slot | Phase | Gap | Agent | Files (disjoint) |
|---|---|---|---|---|
| A1 | F1 strip legacy struct | G1 | `cpp-pro` | `cpp/src/pipe1d.cuh` |
| A2 | F2 init slot width | G2 | `cpp-pro` | `cpp/src/pipe1d.cu` mesh-build region (~lines 805–900, 1060, 1221) |
| A3 | F3 build face classes 4 + 6 | G3 | `cpp-pro` | `cpp/src/pipe1d.cu` mesh-build face loop (~lines 1340–1510), `cpp/src/swe2d_bindings.cpp` binding signature, `swe2d/runtime/coupling.py` `_build_pipe1d_mesh_on_device` |
| A4 | F5 delete legacy wrappers | G5 | `cpp-pro` | `cpp/src/swe2d_gpu.cu` (lines 3016, 3156, ~8473–8809) |
| A5 | F6 fix upload guard | G6 | `debugger` | `cpp/src/pipe1d.cu:4952–5040` |
| A6 | F7 collapse readback schema | G7 | `cpp-pro` + `python-pro` (pair) | `cpp/src/swe2d_bindings.cpp:1918, 2222`; `swe2d/runtime/coupling.py:1490–1559`; test callers |

A6 is a `python-pro` + `cpp-pro` pair — the C++ binding change and the
Python consumer change must land atomically. Dispatch as a paired task.

### Wave B — depends on Wave A

| Slot | Phase | Gap | Agent | Depends on |
|---|---|---|---|---|
| B1 | F4 delete step kernels | G4 | `cpp-pro` | A1 (F1) |

B1 reviews what the kernels being deleted reference. If F1 already
deleted those struct fields, B1 deletes the kernels. If any field is
still referenced from outside (unlikely after F1), B1 also rewires.

### Wave C — depends on Wave B + parts of Wave A

| Slot | Phase | Gap | Agent | Depends on |
|---|---|---|---|---|
| C1 | F8 wire unified path | G8/G9 | `python-pro` | A3 (F3), A4 (F5), A5 (F6), B1 (F4) |

C1 finishes the Python-side wiring now that every legacy wrapper
destination is gone. Drops `apply_coupling_drainage` /
`compute_coupling_full_on_device` / `upload_junction_overflow_state`
calls. Verifies only `swe2d_pipe1d_step` + the unified face kernel
remain in the per-step path.

### Wave D — final gate

| Slot | Phase | Gap | Agent | Depends on |
|---|---|---|---|---|
| D1 | F9 full regression | all | `test-automator` | A1–A6, B1, C1 |

ASan/UBSan rebuild + full regression suite + nsys profile comparison vs
the original plan's baseline.

## 5. Per-Phase Detail (condensed — the parallelized view)

The full per-phase implementation steps live in the appendix below.
Each phase is small enough that one focused subagent lands it cleanly.

### F1 — strip legacy struct (A1)

Delete every legacy field listed in plan §3.6 of the predecessor. Verify
with `grep -rn "d_node_\|d_vnode_\|d_pipe_end_\|d_junction_node\|d_cell_from_node\|d_junction_2d_cell" cpp/src/` — only comments remain.

### F2 — init slot width (A2)

Replace `cudaMemset(dev->d_cell_slot_width, 0, ...)` at `pipe1d.cu:1221`
with per-cell init: pipe cells → `xsect_wMax(shape, params)`; manhole /
inlet cells → `h_cell_width[c]`. Verify with
`tests/test_swe2d_pipe1d_surcharge.py` — 3 slot tests pass.

### F3 — build face classes 4 + 6 (A3)

Extend `swe2d_build_pipe1d_mesh` to accept inlet-capture face arrays
(per plan §3.5 SURFACE_2D_INLET SoA) and culvert face arrays (per plan
§3.5 CULVERT). Loop over inlet nodes with coupled 2D cell and emit
class-4 faces; loop over culvert structures and emit class-6 faces.
Allocate `d_face_inlet_*` SoA + `d_ghost_culvert_struct_idx`. Update
Python `_build_pipe1d_mesh_on_device` to assemble the new arrays.

### F4 — delete step kernels (B1)

Remove the eight legacy kernel calls listed in G4 from
`swe2d_pipe1d_step`. If F1 already deleted the struct fields the
kernels read, the kernels are pure dead code; otherwise wire them to
read face fluxes directly. Verify godunov update reads
`face_F_h`/`face_F_Q` directly at line 3219 (rewrite to per-cell
accumulators if not).

### F5 — delete legacy wrappers (A4)

Remove `swe2d_culvert_face_flux_kernel` (swe2d_gpu.cu:3156) +
`swe2d_fold_drainage_q_kernel` (swe2d_gpu.cu:3016). CULVERT flows
through class 6 (F3); source-sink coupling writes
`d_ext_struct_flux_h` directly. Audit
`swe2d_gpu_compute_coupling_full_on_device` and drop culvert/drain-q
launches + inner `cudaStreamSynchronize` (line ~8807).

### F6 — fix upload guard (A5)

At `pipe1d.cu:4977–4980`, change every `cudaMemcpy` size from host `n`
to `min(n, n_junc)` where `n_junc = p.d_n_junctions`. Mirror the
face-patch block's guard at line ~5003. Add a regression test that
calls `swe2d_pipe1d_upload_junction_overflow_state` with host arrays
sized larger than `d_n_junctions` and expects no CUDA error.

### F7 — single readback schema (A6)

Delete `swe2d_pipe1d_readback_node_state` (swe2d_bindings.cpp:1918).
Extend `swe2d_pipe1d_readback_cell_state` (line 2222) to populate
`cell_velocity = cell_Q / cell_A` and keep `cell_depth` as documented
alias. Update `coupling.py:1490–1559` to consume only the cell schema.
Update tests reading `state["node_depth"]`.

### F8 — wire unified path (C1)

In `coupling.py` `apply_native_device_sources`:
- Drop `swe2d_gpu_compute_coupling_full_on_device` call (line 1782);
  replace with direct `swe2d_gpu_apply_unified_face_flux` call.
- Drop `swe2d_pipe1d_upload_junction_overflow_state` call (line 2067)
  IF class-5 face in unified kernel handles it (after F4); otherwise
  keep but ensure F6 fix is in place.
- Confirm only `swe2d_pipe1d_step` + unified face kernel remain.

### F9 — full gate (D1)

Original plan §4 gate + ASan/UBSan + nsys profile. Compare against the
predecessor plan's profile baseline; no regression.

## 6. Superpowers Workflow

- **Wave dispatch**: `dispatching-parallel-agents` — Wave A dispatches
  five subagents in one message (A1, A2, A3, A4, A5, A6-paired). Each
  subagent has a disjoint file slice.
- **Per-wave convergence**: `subagent-driven-development` with a
  two-stage review between waves — each Wave A agent's output is
  reviewed by `debugger` before Wave B starts.
- **TDD**: `test-driven-development` — for each gap with a missing test
  (G2, G6, G7), write the failing test first inside the subagent's
  task description.
- **Cross-review** (repo rule): every C++ change reviewed by `debugger`
  before the wave is marked complete.
- **On unexpected failure**: `systematic-debugging`.
- **Before claiming completion**: `verification-before-completion` —
  rebuild + run mapped tests + paste actual output.
- **Cache discipline** (repo rule): `find . -type d -name __pycache__
  -exec rm -rf {} +` after every native rebuild before re-testing.

## 7. Selectable Step Dicts (machine-readable)

```python
[
  # ── Wave A — five independent parallel subagents ──
  {"action": "F1 strip legacy Pipe1DDeviceState fields", "type": "refactor", "phase": "F1",
   "agent": "cpp-pro", "model": "default", "wave": "A"},
  {"action": "F2 initialize Preissmann slot width on manhole and inlet cells",
   "type": "coding", "phase": "F2",
   "agent": "cpp-pro", "model": "default", "wave": "A"},
  {"action": "F3 build face classes 4 SURFACE_2D_INLET and 6 CULVERT with HEC-22 and structure SoA",
   "type": "coding", "phase": "F3",
   "agent": "cpp-pro", "model": "default", "wave": "A"},
  {"action": "F5 delete swe2d_culvert_face_flux_kernel and swe2d_fold_drainage_q_kernel and rewire compute_coupling_full_on_device",
   "type": "refactor", "phase": "F5",
   "agent": "cpp-pro", "model": "default", "wave": "A"},
  {"action": "F6 fix swe2d_pipe1d_upload_junction_overflow_state invalid argument cudaMemcpy host size guard",
   "type": "debugging", "phase": "F6",
   "agent": "debugger", "model": "default", "wave": "A"},
  {"action": "F7 collapse readback bindings to single cell schema with cell_velocity and cell_depth aliases",
   "type": "refactor", "phase": "F7",
   "agent": "python-pro", "model": "default",
   "pair": "cpp-pro", "wave": "A"},

  # ── Wave B — single subagent after F1 ──
  {"action": "F4 delete legacy kernels launched from swe2d_pipe1d_step including junction_bc junction_overflow update_node_depth fold kernels mark_inlet scale_double pipe_end_clamp",
   "type": "refactor", "phase": "F4",
   "agent": "cpp-pro", "model": "default", "wave": "B",
   "depends_on": ["F1"]},

  # ── Wave C — single subagent after F3+F5+F6+F4 ──
  {"action": "F8 wire apply_native_device_sources onto unified face kernel path drop legacy coupling wrappers",
   "type": "python", "phase": "F8",
   "agent": "python-pro", "model": "default", "wave": "C",
   "depends_on": ["F3", "F5", "F6", "F4"]},

  # ── Wave D — final gate ──
  {"action": "F9 full regression gate plus ASan UBSan plus nsys profile comparison vs predecessor baseline",
   "type": "test", "phase": "F9",
   "agent": "test-automator", "model": "default", "wave": "D",
   "depends_on": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]},
]
```

## 8. Routing Keywords

| Step | Routing keywords |
|---|---|
| F1 | `c++`, `refactor` |
| F2 | `c++`, `coding`, `debugging` |
| F3 | `c++`, `python`, `coding` |
| F4 | `c++`, `refactor` |
| F5 | `c++`, `refactor` |
| F6 | `c++`, `python`, `debugging` |
| F7 | `c++`, `python`, `refactor` |
| F8 | `python`, `refactor` |
| F9 | `test`, `validate` |

## 9. Wave Dispatch Protocol

### Wave A — single message, six parallel subagents

Each subagent gets a self-contained prompt with:
1. Exact files it owns (disjoint slice).
2. The gap ID and acceptance criterion.
3. The TDD test to write first if one doesn't exist.
4. The verification gate command to run before claiming done.
5. Instruction to commit with a descriptive message and report file:line
   evidence.

Dispatch order (no dependencies, so all simultaneous):
- A1 `cpp-pro` — F1 strip struct
- A2 `cpp-pro` — F2 slot width init
- A3 `cpp-pro` — F3 face classes 4/6
- A4 `cpp-pro` — F5 delete legacy wrappers
- A5 `debugger` — F6 upload guard
- A6-paired `python-pro` + `cpp-pro` — F7 readback collapse

After all six complete: `debugger` agent cross-reviews each diff.

### Wave B — single subagent after A1

B1 `cpp-pro` — F4 delete step kernels. Waits for A1 to land.

### Wave C — single subagent after A3+A4+A5+B1

C1 `python-pro` — F8 wire unified path. Waits for the four upstream
phases.

### Wave D — single gate after everything

D1 `test-automator` — F9 full gate.

## 10. Risk Register (top 5)

1. **Wave A agents collide on `pipe1d.cu`** (MEDIUM) — F1 deletes fields;
   F2 reads/writes mesh-build region; F3 extends mesh build; F6 edits
   upload host at line ~4977. File:line disjointness holds IF F1, F2,
   F3 each touch only their line ranges and F6 stays in lines 4952–5040.
   The mesh-build region (F2, F3) is disjoint from the upload region
   (F6). F1's struct edit is in `pipe1d.cuh`, not `pipe1d.cu`. **No
   collision in steady state, but a hasty F3 edit could drift into F6's
   region.** Mitigate: each Wave A agent's prompt lists forbidden
   line ranges.
2. **F7 readback collapse breaks external callers / docs** (LOW) —
   Mitigate by keeping `cell_depth` as a documented alias and updating
   docs.
3. **F3 face-class-4 build reveals missing HEC-22 implementation**
   (MEDIUM) — kernel has class-4 branch but per-face HEC-22 SoA never
   allocated. F3 allocates and asserts non-null.
4. **F4 godunov update still reads from legacy accumulators** (MEDIUM) —
   F4 verifies godunov source arrays (line 3219) and rewires if needed.
5. **F6 silent truncation if host truly intends oversized upload**
   (LOW) — Mitigate by raising Python-side assertion that
   `len(host_junction_*) <= d_n_junctions` before calling binding.

## 11. Verification Gate

### Per-Wave gate (after each wave completes)

```bash
cd /home/aaron/QGIS_Plugins_dev/private-repo-hydra2dgpu/build
mamba run -n qgis_stable cmake --build . -j$(nproc)
cd /home/aaron/QGIS_Plugins_dev/private-repo-hydra2dgpu
find . -type d -name __pycache__ -exec rm -rf {} +
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v \
    tests.test_pipe1d_face_indexed_mesh \
    tests.test_swe2d_pipe1d \
    tests.test_swe2d_pipe1d_surcharge
```

### Final gate (F9, Wave D)

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

# ASan/UBSan build + zero errors
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1" ..
mamba run -n qgis_stable cmake --build . -j$(nproc)
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    python3 -m unittest -v tests.test_pipe1d_face_indexed_mesh

# nsys profile
mamba run -n qgis_stable env PYTHONPATH="$PWD:$PWD/build" \
    nsys profile -o /tmp/opencode/phase_d_followup.qdstrm --force-overwrite=true \
    python3 -m unittest tests.test_swe2d_gpu_drainage_network
```

## 12. Reference

- Predecessor plan: `docs/pipe1d_face_indexed_refactor_plan.md`
- Original F1–F15 audit: `docs/PIPE1D_AUDIT_2026-07-17.md` (historical)
- Codebase audit: `docs/CODEBASE_AUDIT_2026-07-19.md`