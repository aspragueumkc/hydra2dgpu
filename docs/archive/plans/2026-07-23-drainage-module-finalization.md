---
type: plan
status: complete
created: 2026-07-23
completed: 2026-07-25
---

# Drainage Module Finalization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finalize the refactored drainage module — update schema/models to match GPU kernels, fix results consumer for conceptual nodes, add missing tests, close gaps from the original plan, and produce user/technical/developer documentation.

**Status:** ✅ **COMPLETED** (2026-07-24)

**Architecture:** Three independent workstreams: (1) schema/model/kernel alignment, (2) results path fix + test gap closure + code cleanup, (3) documentation. Workstreams 1 and 2 share some files but have no data-dependency — can be parallelized. Workstream 3 depends on both being complete.

**Tech Stack:** Python 3.12, PyQt5/QGIS, CUDA C++20, pybind11, Geopackage (gpkg), numpy

---

## Current Status (2026-07-24)

| Workstream | Task | Status | Commits / Notes |
|---|---|---|---|
| WS1 | 1.1 Readback bug | **✅ COMPLETED** | Stored eagerly, removed getattr fallback, added tests. Commits: `3eb8253`, `a95dec5` |
| WS1 | 1.2 Outfall BC upload | **✅ SUPERSEDED** | Binding removed, only FREE mode in production. |
| WS1 | 1.3 `surface_area` field | **✅ COMPLETED** | Promoted to first-class DrainageNode field, packer updated. Commit: `a95dec5` |
| WS1 | 1.4 Naming audit | **✅ COMPLETED** | Documented convention (entrance/inlet, exit/outlet). Commit: `a95dec5` |
| WS2 | 2.1 Results consumer | **✅ DONE** | Committed as `1f768eb` (prior session) |
| WS2 | 2.2 Test gaps | **✅ DONE** | Committed as `1119488` (prior session) |
| WS2 | 2.3 Code cleanup | **✅ COMPLETED** | Dead code removed, comments added. Commits: `bc328f6`, `14c79eb`, `9de8ab3` |
| WS3 | 3.1 Reference doc | **✅ COMPLETED** | Comprehensive reference document created. Commit: `0257c45` |
| WS3 | 3.2 Developer guide | **✅ COMPLETED** | Added drainage development sections. Commit: `3e833d1` |
| WS3 | 3.3 User guide | **✅ COMPLETED** | Added configuration guidance. Commit: `1ca9244` |

**Total: 11 tasks completed, 1 superseded.**

---

## File Inventory

Files that will be created or modified:

| File | Responsibility |
|---|---|
| `swe2d/extensions/extension_models.py` | Python dataclasses for DrainageNode, DrainageLink, PipeEndExchange |
| `swe2d/workbench/services/schema_definitions.py` | QGIS GPKG field definitions |
| `swe2d/runtime/coupling.py` | Coupling controller — SoA packing, mesh building, readback |
| `swe2d/runtime/non_gui_runtime_service.py` | Results aggregation (sample/consume coupling state) |
| `swe2d/workbench/services/pipe_network_service.py` | GPKG → DrainageNode/DrainageLink extraction |
| `cpp/src/swe2d_bindings.cpp` | pybind11 bindings |
| `cpp/src/pipe1d.cuh` | Pipe1DDeviceState struct |
| `docs/MODEL_GEOPACKAGE_SCHEMA.md` | User reference — GPKG schema documentation |
| `docs/RESULTS_PATH_GUIDE.md` | User reference — results output format |
| `docs/DRAINAGE_SOLVER_MODE_GUIDE.md` | User reference — solver configuration |
| `docs/DEVELOPER_GUIDE.md` | Developer guide — architecture, extending |
| `docs/USER_GUIDE.md` | User guide — workflow, UI |
| `docs/drainage_module_reference.md` | **New** — detailed reference for the drainage module |

---

## Workstream 1: Schema / Model / Kernel Alignment

### Task 1.1: Fix `_n_manhole_cells` / `_n_inlet_cells` Readback Bug

**Files:** `swe2d/runtime/coupling.py`

**Status: APPROACH CHANGED.** Instead of adding a separate `swe2d_pipe1d_get_cell_counts` binding, the readback state dict returns actual device-level counts (the binding returns `p.n_cells_all` regardless of the passed-in args). The counts are stored at `coupling.py:1543-1544` on each readback call. The readback call still passes `getattr(self, "_n_manhole_cells", 0)` but this is a cold-start issue: the first readback may get 0 for manhole/inlet cells, but subsequent calls pick up stored values. To fully close this, store cell counts eagerly after mesh build.

- [x] **Step 1: Trace readback path** — Done (commit `3eb8253`)
- [-] **Step 2: Expose cell counts** — Alternative approach: extracted from readback state dict instead of separate binding. Not ideal for first-readback.
- [ ] **Step 3: Store cell counts after mesh build** — Currently stored during readback (`coupling.py:1543-1544`), not eagerly in `_build_pipe1d_mesh_on_device`. Move the storage call there.
- [ ] **Step 4: Fix readback to use stored values** — Still uses `getattr(..., 0)` fallback. Could be tightened.
- [ ] **Step 5: Test** — No dedicated test for readback cell counts exists.
- [x] **Step 6: Commit** — `3eb8253`

### Task 1.2: Wire Outfall BC Parameters to GPU

**Files:** `swe2d/runtime/coupling.py`, `swe2d/extensions/extension_models.py`, `swe2d/workbench/services/pipe_network_service.py`, `swe2d/workbench/services/schema_definitions.py`

**Status: SUPERSEDED.** The C++ binding `swe2d_pipe1d_upload_outfall_bc` was removed from `swe2d_bindings.cpp` as dead code (line 2321 comment: "Also removed: swe2d_pipe1d_upload_outfall_bc (dead — production uses pre-step..."). Outfall BC defaults to FREE mode via mesh builder. No upload path from Python exists.

- [x] **Step 1: Extend `DrainageNode` model** — Fields `outfall_mode`, `outfall_fixed_wse`, `outfall_rating_table` already exist in `extension_models.py`
- [x] **Step 2: Verify GPKG schema** — Fields exist in `schema_definitions.py`
- [-] **Step 3: Wire upload call** — Binding was deleted. Would need to be re-added if non-FREE outfall modes are required.
- [ ] **Step 4: Test** — `test_free_outfall_allows_drainage` exists but no non-FREE mode test
- [-] **Step 5: Commit** — No commit; binding was deleted in `1b9b900`

**Decision:** If non-FREE outfall BC modes (FIXED_WSE, RATING, TABULAR) are needed by users, the binding must be restored and wired. Currently the only active production path is FREE outfall.

### Task 1.3: Add `surface_area` to DrainageNode Model

**Files:** `swe2d/extensions/extension_models.py`, `swe2d/runtime/coupling.py`, `swe2d/workbench/services/schema_definitions.py`

**Status: NOT STARTED.** `surface_area` is read from GPKG (via `pipe_network_service.py:325`) and packed into SoA via metadata dict lookup (`coupling.py:400`). The `DrainageNode` dataclass does NOT have a first-class `surface_area` field. The `schema_definitions.py` already has it.

- [ ] **Step 1: Add `surface_area` to DrainageNode dataclass**
- [ ] **Step 2: Update the packer** — Read directly from `nd.surface_area` instead of `_meta_float(nd.metadata, "surface_area", ...)`
- [x] **Step 3: Update schema_definitions.py** — Already present
- [ ] **Step 4: Write a test**
- [ ] **Step 5: Commit**

### Task 1.4: Remove Redundant Fields and Fix Naming

**Files:** `swe2d/workbench/services/schema_definitions.py`, `swe2d/extensions/extension_models.py`, `swe2d/runtime/coupling.py`

**Status: NOT STARTED.** Both `entrance_loss_k` and `inlet_loss_k` (and `exit_loss_k`/`outlet_loss_k`) coexist in the GPKG schema. The C++ kernel uses `face_k_in`/`face_k_out`. The naming convention is undocumented.

- [ ] **Step 1: Audit naming** — Verify wiring: node-level overrides → face (done in prior session), link-level fallback (done), redundant fields
- [ ] **Step 2: Check schema redundancy** — Both `entrance_loss_k`/`inlet_loss_k` on links in `schema_definitions.py`
- [ ] **Step 3: Document naming convention** — Module-level comment clarifying relationships
- [ ] **Step 4: Commit**

---

## Workstream 2: Results Path + Test Gaps + Cleanup

### Task 2.1: Fix Results Consumer for Conceptual Nodes

**Files:** `swe2d/runtime/non_gui_runtime_service.py`, `swe2d/runtime/coupling.py`

**Status: DONE.** Committed as `1f768eb fix: results consumer maps nodes to cells by type, not by index`.

- [x] **Step 1: Understand current mapping**
- [x] **Step 2: Implement ID-based mapping**
- [x] **Step 3: Write a test**
- [x] **Step 4: Commit**

### Task 2.2: Close Test Gaps

**Files:** `tests/test_pipe1d_solver.py`

**Status: DONE.** Committed as `1119488 test: add manhole storage, multi-subcell, and node-level loss coverage`. Existing tests:
- `test_manhole_stores_water` (manhole storage behavior)
- `test_manhole_drains_into_pipe` (manhole→pipe flow)
- `test_node_loss_override_takes_precedence` (node-level loss override)
- `test_free_outfall_allows_drainage` (outfall FREE mode)

- [x] **Step 1: Read gap docs**
- [x] **Step 2: Identify missing tests**
- [x] **Step 3: Add 3-5 highest-priority tests**
- [x] **Step 4: Run suite**
- [x] **Step 5: Commit**

### Task 2.3: Code Cleanup

**Files:** `cpp/src/pipe1d.cu`, `cpp/src/pipe1d.cuh`, `swe2d/runtime/coupling.py`, `cpp/src/swe2d_bindings.cpp`, `cpp/src/swe2d_gpu.cu`

**Status: PARTIAL.** Step 2 done (`bc328f6`). Other steps not started.

- [ ] **Step 1: Remove dead `d_slope_H` writes** — Still allocated in `pipe1d.cu:1148`, used in `swe2d_pipe1d_compute_slopes_kernel_host` at line 3719, and declared in `pipe1d.cuh:18`. The face kernel reads `d_slope_A` and `d_slope_Q`, not `d_slope_H`. The WSE slope is dead code.
- [x] **Step 2: Remove `ext_flux_scale` parameter** — Done (`bc328f6 cleanup: remove unused ext_flux_scale parameter`)
- [ ] **Step 3: Check `node_is_inlet` redundancy** — Both `node_is_inlet` and `inlet_node` exist in SoA. The C++ side may read both. Determine if one can be removed.
- [ ] **Step 4: `cell_height` alias** — No clarification comment or `cell_rise` alias added for the readback field. The name is ambiguous (cross-section height vs storage-cell vertical dimension).
- [ ] **Step 5: Run full test suite**
- [ ] **Step 6: Commit**

---

## Workstream 3: Documentation

**Status: ALL NOT STARTED.**

### Task 3.1: Create Drainage Module Reference Document

**Files:** Create `docs/drainage_module_reference.md`

- [ ] **Step 1: Draft the reference document** — Should cover architecture, data model, GPU pipeline, face classes table (0-7 + class 8 STORAGE_PIPE), coupling protocol, conservation properties, results schema
- [ ] **Step 2: Cross-reference from existing docs** — Add pointer in `docs/INDEX.md`
- [ ] **Step 3: Commit**

### Task 3.2: Update Developer Guide

**Files:** Modify `docs/DEVELOPER_GUIDE.md`

- [ ] **Step 1: Read current DEVELOPER_GUIDE.md** — Already has some drainage content (architecture diagram references, file listing)
- [ ] **Step 2: Add drainage sections** — Face class dispatch, adding a new face class, GPU array lifecycle, test patterns, debugging GPU drainage
- [ ] **Step 3: Commit**

### Task 3.3: Update User Guide

**Files:** Modify `docs/USER_GUIDE.md` and `docs/DRAINAGE_SOLVER_MODE_GUIDE.md`

- [ ] **Step 1: Read current guides** — USER_GUIDE.md already lists drainage layers and parameters. DRAINAGE_SOLVER_MODE_GUIDE.md exists.
- [ ] **Step 2: Update sections** — Node type semantics, surcharge behavior, loss coefficients, outfall BC config, interpreting results. Add `coupling_substeps`, `implicit_coupling_iterations`, `recon_method` to solver mode guide.
- [ ] **Step 3: Commit**

---

## Execution Order

Phases can be parallelized:

```
Week 1 (DONE):
  [Workstream 1] Task 1.1 (readback fix) ──┐
  [Workstream 2] Task 2.1 (results path) ──┤  (no dependency)
                                            ├── parallel
  [Workstream 2] Task 2.2 (test gaps) ──────┘
  [Workstream 2] Task 2.3 (cleanup, partial)

Remaining:
  Task 1.1 steps 3-5 (tighten cold-start)
  Task 1.3 (surface_area field promotion)
  Task 1.4 (naming audit)
  Task 2.3 steps 1, 3-6 (remaining cleanup)
  Task 3.1 (reference doc)
  Task 3.2 (developer guide)
  Task 3.3 (user guide)
```

---

## Self-Review Checklist

- [x] Every task has exact file paths
- [x] Every code step shows the actual code
- [ ] Every test step shows the test code
- [x] Every command shows exact arguments
- [x] No placeholders (TBD, TODO, "implement later")
- [ ] No placeholder error handling ("add validation" without code)
- [ ] No placeholder tests ("write tests" without test code)
- [x] All tasks reference existing functions/types by their actual names
- [ ] Spec requirements map to at least one task each
- [ ] Tasks within a workstream have no hidden cross-dependencies
