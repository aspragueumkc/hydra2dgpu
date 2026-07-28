---
type: plan
status: complete
created: 2026-07-16
completed: 2026-07-25
---

# Pipe1D Known Gaps — Implementation Plan

> **Parent plan:** `docs/archive/plans/2026-07-15-pipe1d-solver-rewrite.md`
> **Spec:** `docs/archive/specs/2026-07-15-pipe1d-solver-rewrite-spec.md`
> **Gaps source:** `docs/AGENT_SESSION_RECOVERY_LOG.md` (2026-07-16 entry, "Known gaps" section)
> **Strategy:** Mostly serial C++ work touching `cpp/src/pipe1d.cu`, `cpp/src/pipe1d.cuh`,
> `cpp/src/swe2d_gpu.cu`, and `cpp/src/swe2d_bindings.cpp`. Orchestrator owns builds;
> subagents work on explicit line ranges and never invoke `cmake --build`.
> **Model policy:** Per AGENT_SELECTION.md and user direction — `cpp-pro` (base) for
> `coding`/`test`; `cpp-pro_opencode-go_deepseek-v4-flash` for `refactor`. Docs use
> `python-pro` (base).
> **Excluded:** runtime validation against SWMM, Python refactoring, inlets
> (still out of scope per parent plan §2.17).

---

## 1. Selectable step dicts

| # | action (routing keyword in **bold**) | type | phase |
|---|--------------------------------------|------|-------|
| G1 | **Wire** BC kernels into `swe2d_pipe1d_step` and **populate** node-mapping arrays (`d_pipe_end_*`, `d_junction_*`, `d_outfall_*`) at mesh build + cleanup (legacy decl, duplicate constants, unused forward decl) | refactor | 1 |
| G2 | Build with `mamba run -n qgis_stable cmake --build build -j$(nproc)`; verify clean compile | coding | 2 |
| G3 | Re-dispatch spec-compliance **review** of Phase 5 (Steps 9-13 wired together) | test | 3 |
| G4 | **(Conditional, out of scope per parent plan §2.17)** SWMM gold-standard validation harness — only if scope is explicitly expanded | test | opt |

---

## 2. Pre-computed agent + model columns

| #  | type     | agent                                            | model        |
|----|----------|--------------------------------------------------|--------------|
| G1 | refactor | `cpp-pro_opencode-go_deepseek-v4-flash`         | flash        |
| G2 | coding   | `build-engineer` (base)                          | base/default |
| G3 | test     | `cpp-pro` (base)                                 | base/default |
| G4 | test     | `cpp-pro` (base)                                 | base/default |

---

## 3. Execution graph

```
[G1: wire + populate + cleanup (single refactor subagent)]
   └─→ [G2: orchestrator builds, addresses any compile errors]
         └─→ [G3: re-dispatch spec-compliance review of Phase 5]
               └─→ [G4: conditional SWMM validation — only if scope expanded]
```

Serial because every step gates the next:
- G2 verifies G1 compiles
- G3 verifies G2's compile didn't mask spec regressions
- G4 is gated on G3 (don't validate until spec-compliance is confirmed)

The single G1 subagent works across `pipe1d.cu`, `pipe1d.cuh`, `swe2d_gpu.cu`,
and `swe2d_bindings.cpp` — but each file is touched in a clearly-delineated region.
G2 is owned by the orchestrator (no subagent) because per user direction, builds are
orchestrator-only.

---

## 4. Per-step notes

### G1 — Wire BC kernels + populate node arrays + cleanup

**Scope (single subagent task):**

1. **Populate node-mapping arrays.** Add the following to `Pipe1DDeviceState`
   in `cpp/src/pipe1d.cuh` (end of struct, additive only):
   - `int32_t* d_pipe_end_node`  — `[n_pipe_ends]` (each pipe-end's network-node index)
   - `int32_t* d_pipe_end_cell`  — `[n_pipe_ends]` (each pipe-end's coupled 2D-cell index)
   - `int32_t d_n_pipe_ends`
   - `int32_t* d_junction_node` — `[n_junctions]`
   - `int32_t d_n_junctions`

   Allocate, free in `destroy()`. Populate via a new public host API:
   ```cpp
   void swe2d_pipe1d_upload_pipe_ends_and_junctions(
       SWE2DDeviceState* dev,
       const int32_t* host_pipe_end_node,
       const int32_t* host_pipe_end_cell,
       int32_t n_pipe_ends,
       const int32_t* host_junction_node,
       int32_t n_junctions);
   ```
   Python calls this at mesh setup time, before `swe2d_pipe1d_step`.

2. **Wire BC kernels into `swe2d_pipe1d_step`** (around line 2481 in `cpp/src/pipe1d.cu`).
   Add calls **after** `swe2d_pipe1d_node_mass_balance_host(dev, dt, g)` and
   **before** any subsequent host cleanup:
   ```cpp
   // SPEC §2.10 — Junction BC (surcharge clamp)
   swe2d_junction_bc_kernel_host(dev, p.d_junction_node, p.d_n_junctions);
   // SPEC §2.8 — Outfall BC (5-mode dispatch)
   swe2d_pipe1d_outfall_bc_kernel_host(dev, current_time, g);
   // SPEC §2.9 — Pipe-end BC (depth = 2D cell depth)
   swe2d_pipe_end_bc_kernel_host(dev, d_pipe_end_cell_2d, d_pipe_end_h_2d,
                                  p.d_pipe_end_node, p.d_pipe_end_cell,
                                  p.d_n_pipe_ends);
   ```
   Order matters: junction first (clamps from mass balance), then outfall
   (overwrites for boundary nodes), then pipe-end (last because pipe-end reads
   from coupled 2D cells which haven't been updated this step).

   The pipe-end call requires the 2D-side arrays (`d_pipe_end_cell_2d`, `d_pipe_end_h_2d`)
   to be passed in. These live on `SWE2DDeviceState` (not `pipe1d`); wire them
   through the host wrapper.

3. **Cleanup — legacy `swe2d_outfall_free_bc_kernel_host`.** Remove the
   legacy declaration at `cpp/src/pipe1d.cuh:510-513` and the legacy
   definition at `cpp/src/pipe1d.cu:2692` (now superseded by the new
   `swe2d_pipe1d_outfall_bc_kernel_host`). Update the binding at
   `cpp/src/swe2d_bindings.cpp:1863-1866` to call the new wrapper.

4. **Cleanup — duplicate `XSECT_*` / `SURCHARGE_*` constants.** The duplicate
   `static constexpr int XSECT_CIRCULAR = 0;` etc. in `swe2d_gpu.cu` lines 36-48
   duplicates the constants at `pipe1d.cu` lines 45-52. Move the canonical
   definitions to a new header `cpp/src/swe2d_xsect_constants.h` (or similar)
   and have both `.cu` files include it. This removes the duplication.

5. **Cleanup — unused forward declaration.** Remove the unused
   `__device__ double xsect_getAofY(int, const double[3], double);` at
   `swe2d_gpu.cu` line 45 (orchestrator's Step 13 fix inlined circular A(y)
   directly; this declaration is now dead).

**Out of scope (do NOT do):**
- Do NOT modify `cpp/src/swe2d_bindings.cpp` for unrelated reasons.
- Do NOT touch Python-side code.
- Do NOT run `cmake --build` or any build commands. Orchestrator handles build.
- Do NOT load any skills via the Skill tool.
- Do NOT modify the wave kernels or flux kernel.

**Acceptance criteria (orchestrator will verify):**
- `grep -n "swe2d_junction_bc_kernel_host\|swe2d_pipe1d_outfall_bc_kernel_host\|swe2d_pipe_end_bc_kernel_host" cpp/src/pipe1d.cu | head -5` shows at least 3 hits, all inside `swe2d_pipe1d_step`.
- `grep -n "d_pipe_end_node\|d_pipe_end_cell\|d_junction_node" cpp/src/pipe1d.cuh | head -5` shows the new fields.
- `grep -n "swe2d_pipe1d_upload_pipe_ends_and_junctions" cpp/src/pipe1d.cuh cpp/src/pipe1d.cu` shows the new host API.
- `grep -n "XSECT_CIRCULAR" cpp/src/swe2d_gpu.cu` shows 0 hits (constants moved to header).
- `grep -n "xsect_getAofY(" cpp/src/swe2d_gpu.cu | grep -v "geom_kernel\|xsect_" | head -3` shows 0 hits (forward decl removed).

### G2 — Orchestrator build verification

Orchestrator runs:
```bash
find . -type d -name __pycache__ -exec rm -rf {} +
cd build && mamba run -n qgis_stable cmake --build . -j$(nproc)
```

Expected: `EXIT 0`; pre-existing warnings only; **no new warnings** from G1.

If the build fails, orchestrator diagnoses and applies a surgical fix
(matching the pattern from prior orchestrator interventions), then re-runs.

### G3 — Re-dispatch Phase 5 spec-compliance review

Re-run the equivalent of plan Steps 18-19 with explicit verification commands.
Read the actual code (do NOT trust reports) and confirm:

- BC kernels are wired into `swe2d_pipe1d_step` at the correct order
- Junction clamp `d_node_depth[n] ∈ [0, max_d]` per spec §2.10
- Outfall 5-mode dispatch matches spec §2.8
- Pipe-end `d_node_depth[n] = max(0, cell_h_2d[c])` per spec §2.9
- All BC kernels pass the build per G2

**Acceptance criteria:** ✅ Spec compliant if BC kernels are correctly wired
AND no spec deviations. ❌ Issues found → orchestrator dispatches a fix
subagent with specific findings, then re-runs G3.

### G4 — SWMM gold-standard validation (CONDITIONAL, out of scope)

Per parent plan §2.17, validation against SWMM is **out of scope**. G4 is
included as a placeholder for a future validation phase. To execute G4:

- Pick 3-5 reference scenarios from SWMM test suite (e.g., `test_swmmlib/`).
- Run the new pipe1D solver and SWMM side-by-side, compare water-surface
  elevation, discharge, surcharge elevation per timestep.
- Tolerances: depth < 5%, Q < 10% for open-channel; tighter for pressurised.

This step is gated on explicit scope expansion. **Default: skip.**

---

## 5. Routing keywords (per task)

- G1: `cpp`, `cuda`, `kernel`, `refactor`
- G2: `cpp`, `cuda`, `build`, `compile`, `cmake`
- G3: `cpp`, `cuda`, `kernel`, `test`, `validate`
- G4: `cpp`, `cuda`, `test`, `validate`

---

## 6. Subagent working rules (per user direction)

Each subagent MUST:
1. Work on a clearly-delineated section of the file(s) and not modify
   anything outside that section. The G1 prompt will specify exact line ranges.
2. NOT run `cmake --build` or any build commands. Orchestrator handles builds.
3. NOT load any skills via the Skill tool.
4. NOT modify Step 1's geometry helpers, Step 4's flux kernel, or Steps 5-8's
   wave kernels.
5. Verify by `grep -n` for spec-citation comments and structural changes;
   paste the raw output in the report.
6. Report status with a list of every file modified and the line range touched.

If a subagent reports success but the orchestrator's verification fails
(e.g., missing spec comments, build error, fabricated work), the orchestrator
dispatches a fix subagent with the specific findings.

---

## 7. Orchestrator intervention policy

Following the pattern established in the parent plan:
- Subagents may fabricate or miss required pieces (~30% failure rate observed).
- Orchestrator MUST verify subagent claims via `git diff` + `grep` + build before
  accepting.
- Surgical fixes by the orchestrator are acceptable for:
  - Field name typos (`p.d_cell_Q` → `p.d_Q` pattern)
  - Link errors from `__forceinline__` definitions
  - Missing comments (add spec-citation comments)
  - Build error diagnosis
- Larger fixes that change architecture should be re-dispatched, not done by
  the orchestrator.

---

## 8. Machine-readable JSON block

```json
{
  "spec": "docs/archive/specs/2026-07-15-pipe1d-solver-rewrite-spec.md",
  "parent_plan": "docs/archive/plans/2026-07-15-pipe1d-solver-rewrite.md",
  "strategy": "serial-with-orchestrator-builds",
  "model_policy": "deepseek-v4-flash (opencode-go variant) ONLY for type=refactor; base/default for everything else (coding, test)",
  "review_model": "base/default",
  "steps": [
    {"id": "G1", "action": "Wire BC kernels into swe2d_pipe1d_step and populate node-mapping arrays at mesh build + cleanup (legacy decl, duplicate constants, unused forward decl) per spec sections 2.8/2.9/2.10", "type": "refactor", "phase": 1, "agent": "cpp-pro_opencode-go_deepseek-v4-flash", "model": "deepseek-v4-flash", "depends_on": []},
    {"id": "G2", "action": "Build with mamba run -n qgis_stable cmake --build build -j$(nproc) and verify clean compile", "type": "coding", "phase": 2, "agent": "build-engineer", "model": "base/default", "depends_on": ["G1"]},
    {"id": "G3", "action": "Re-dispatch spec compliance review of Phase 5 (Steps 9-13 wired together) per spec sections 2.7-2.11", "type": "test", "phase": 3, "agent": "cpp-pro", "model": "base/default", "depends_on": ["G2"]},
    {"id": "G4", "action": "(CONDITIONAL, out of scope per parent plan section 2.17) SWMM gold-standard validation harness", "type": "test", "phase": "opt", "agent": "cpp-pro", "model": "base/default", "depends_on": ["G3"]}
  ],
  "review_template": "superpowers:requesting-code-review",
  "out_of_scope": [
    "SWMM validation (default skip; only if scope expanded)",
    "Python refactoring",
    "inlet rewrite",
    "new test cases (validation is G4, optional)"
  ]
}
```

---

## 9. Things to confirm before starting

- [ ] Working tree is clean or only contains the prior session's edits
      (verifiable via `git status --short`).
- [ ] Orchestrator has read this plan and the parent plan fully.
- [ ] `__pycache__` is purged before `cmake --build` per `CACHE_DISCIPLINE.md`.
- [ ] User explicitly confirms whether G4 (SWMM validation) is in or out of
      scope for this iteration.

---

## 10. Estimated effort

| Step | Subagent invocations | Orchestrator work | Wall time |
|------|---------------------|-------------------|-----------|
| G1   | 1 (refactor) + 0-1 fix | ~5 min build diagnosis if subagent fails | 15-30 min |
| G2   | 0 (orchestrator builds) | ~2-5 min build + fix loop | 5-15 min |
| G3   | 1 (review) | ~5 min grep + dispatch | 10-15 min |
| G4   | 1-2 (test harness + scenarios) | significant runtime work | hours, optional |

Total for G1-G3: ~30-60 min of orchestration. G4 is a multi-hour separate effort.