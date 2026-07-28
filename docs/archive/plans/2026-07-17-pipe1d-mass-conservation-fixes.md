---
type: plan
status: superseded
created: 2026-07-17
completed: 2026-07-25
superseded_by: docs/pipe1d_face_indexed_refactor_plan.md
---

# Pipe1D Mass-Conservation & Physics Fixes Implementation Plan

> **STATUS (2026-07-19): SUPERSEDED.** This plan is mostly moot.
>
> Further investigation of the `test_closed_system_conserves_mass_diffusion_wave` drift revealed that the pipe1D solver's hybrid "cells + network-nodes + virtual-nodes" discretization has no clean FVM interpretation — the boundary relaxation law `F = dH·c·A·dir` at the cell↔network-node interface is not a finite-volume flux, and the `d_vnode_*` array family is a face-state workaround indexed as if it were a node. Patching this with F1–F15 leaves the architecture in place and the test still doesn't reach machine precision.
>
> **The plan has been replaced by a structural refactor:** `docs/pipe1d_face_indexed_refactor_plan.md`. That refactor eliminates the `d_node_*` and `d_vnode_*` abstractions entirely; manholes/junctions/inlets become full FV cells; pipe-ends become direct HLLC face couplings to 2D SWE2D cells; one unified face-flux kernel handles every face class. Mass conservation becomes automatic by FV construction, and F1–F15 dissolve.
>
> This document is retained as historical record of the F1–F15 finding-by-finding analysis. **Do not execute the tasks below.** If you are looking for current pipe1D work, see `docs/pipe1d_face_indexed_refactor_plan.md`.

---

# Pipe1D Mass-Conservation & Physics Fixes Implementation Plan (HISTORICAL)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 15 defects from `docs/PIPE1D_AUDIT_2026-07-17.md`, centered on making the pipe1d solver mass-conservative at network-node boundaries.

**Architecture:** The core defect is that pipe cells advance continuity via face fluxes `F` while node storage advances via momentum-equation `cell_Q`. The fix unifies both sides on the **boundary-face flux**: the flux kernel atomicAdds `dir·F` into `d_node_net_q` per substep, and the node mass-balance consumes that accumulator. All other fixes (Picard anchoring, first-substep WSE guard, inlet override removal, wave-speed correction, regime-override guards, datum fixes, junction/outfall routing, code defects) layer on top.

**Tech Stack:** CUDA C++ (`cpp/src/pipe1d.cu`, `pipe1d.cuh`, `swe2d_gpu.cu`, `swe2d_bindings.cpp`), Python orchestration (`swe2d/runtime/coupling.py`), GPU-gated `unittest` suite (`tests/`), built module `hydra_swe2d` in `build/`.

**Source spec:** `docs/PIPE1D_AUDIT_2026-07-17.md` (findings referenced as F1–F15).

---

## Build & Test Commands (used by every task)

```bash
# Incremental rebuild (build/ is already configured; do NOT re-run cmake configure)
cd /home/aaron/QGIS_Plugins_dev/private-repo-hydra2dgpu/build
mamba run -n qgis_stable cmake --build . -j$(nproc)
cd /home/aaron/QGIS_Plugins_dev/private-repo-hydra2dgpu

# After ANY structural change to a Python module or native rebuild:
find . -type d -name __pycache__ -exec rm -rf {} +

# Targeted tests (GPU required)
mamba run -n qgis_stable python3 -m unittest -v tests.test_pipe1d_mass_conservation

# Phase gate (full pipe1d suite)
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation \
    tests.test_swe2d_pipe1d \
    tests.test_swe2d_pipe1d_surcharge \
    tests.test_pipe1d_accumulation \
    tests.test_swe2d_gpu_drainage_network \
    tests.test_pipe_cell_coupling_output \
    tests.test_drainage_inlet_outfall_vs_swmm \
    tests.test_swmm_validation_pipe_end \
    tests.test_pipe1d_vs_swmm \
    tests.test_workbench_imports
```

## Key Facts Established During Planning (do not re-derive)

- `swe2d_pipe1d_step` signature: `cpp/src/pipe1d.cuh:541-551`; binding `cpp/src/swe2d_bindings.cpp:1810-1842`; Python call `swe2d/runtime/coupling.py:1697`.
- Per-timestep call order in `swe2d/runtime/coupling.py::apply_native_device_sources`: `swe2d_pipe1d_step` (line 1697) runs **before** `swe2d_gpu_compute_coupling_full_on_device` (line 1741). The latter zeroes `d_external_source_mps` at `swe2d_gpu.cu:8009`, so the in-step pipe-end fold at `pipe1d.cu:2802-2806` is **always wiped** and re-added at `swe2d_gpu.cu:8299-8304`. Single-fold decision: **delete the in-step fold, keep `swe2d_gpu.cu:8299-8304`**.
- `d_node_net_q` is sized `[n_nodes + n_vnodes]` (`pipe1d.cu:885`), zeroed in the mass-balance host (`pipe1d.cu:2357`). After F1 it is zeroed in `swe2d_pipe1d_step` **before** the substep loop.
- Existing consumer of `node_net_q` with the same sign convention (positive = entering node): legacy pipe-end exchange `pipe1d.cu:3037-3090` — F1 makes its input exact.
- `d_node_surface_area` is sized `[n_nodes]` (`pipe1d.cu:886`); `d_node_is_boundary` exists (`pipe1d.cu:2304`) and marks outfalls + pipe-ends (`swe2d_gpu.cu:9664-9674`).
- Outfall exchange kernel launch site (`swe2d_gpu.cu:8236-8245`) has `p.d_node_net_q` in scope (`p = dev->pipe1d` at `swe2d_gpu.cu:8213`).
- Readback binding `swe2d_pipe1d_readback_node_state` returns dict incl. `node_depth`, `cell_A`, `cell_Q`, `cell_invert` (`swe2d_bindings.cpp:2056-2110`).
- 2D coupling source readback: `swe2d_gpu_readback_coupling_sources` (`swe2d_gpu.cu:8325`).
- Probe scripts (reference only, do not commit): `/tmp/opencode/probe_mass_conservation.py`, `probe2_single_cell.py`, `probe3_partial.py`, `probe4_inlet.py`.
- `tests/pipe1d_runner.py` (`Pipe1DRunner`) is the canonical standalone driver; the new test file uses the direct pattern from `tests/test_swe2d_pipe1d.py` instead (finer control over init state).
- `max_cell_length` subdivision logic (`pipe1d.cu:625-631`) already computes in double; only the signature/binding/Python cast truncate.
- Device-side elliptical geometry `xsect_getAofY_elliptical` (in `pipe1d.cu` device helpers / `swe2d_xsect_constants.h`) is correct — mirror it host-side.

## Agent + Model Pre-Computation (per `.opencode/rules/AGENT_SELECTION.md`)

| Task | Phase | Type | Routing keywords | Agent | Model |
|---|---|---|---|---|---|
| 1. Failing conservation tests | A | test | test, validate, gpu | test-automator | kimi-for-coding/kimi-for-coding-highspeed |
| 2. F1+F5 flux-kernel node balance | B | coding | c++, cuda, kernel, gpu | cpp-pro | kimi-for-coding/k3 |
| 3. F3 first-substep WSE guard | B | coding | c++, cuda, kernel | cpp-pro | kimi-for-coding/k3 |
| 4. F4+F11 inlet override removal | B | coding | c++, cuda, kernel | cpp-pro | kimi-for-coding/k3 |
| 5. F2 Picard A_orig anchor | B | coding | c++, cuda, kernel | cpp-pro | kimi-for-coding/k3 |
| 6. F6 wave-speed correction | C | coding | c++, cuda, kernel | cpp-pro | kimi-for-coding/k3 |
| 7. F7 regime-override guards | C | coding | c++, cuda, kernel | cpp-pro | kimi-for-coding/k3 |
| 8. F8 pipe-end datum bias | C | coding | c++, cuda, kernel | cpp-pro | kimi-for-coding/k3 |
| 9. F9+F14 junction path + single fold | C | coding | c++, cuda, kernel | cpp-pro | kimi-for-coding/k3 |
| 10. F10 outfall mass routing | C | coding | c++, cuda, kernel | cpp-pro | kimi-for-coding/k3 |
| 11. F12 elliptical A_open table | D | coding | c++, cuda, python binding | cpp-pro | kimi-for-coding/k3 |
| 12. F13 max_cell_length double | D | coding | c++, python, refactor | cpp-pro | kimi-for-coding/k3 |
| 13. F15 smaller items + sanitizer | D | debugging | debug, validate, cuda | debugger | kimi-for-coding/k3 |
| 14. Full validation + docs | E | docs | docs, validate | python-pro | commandcode/mimo-v2.5 |
| Cross-review (per C++ task) | B–D | debugging | debug, review, cuda | debugger | kimi-for-coding/k3 |

Dispatches use the **base agent names** (`cpp-pro`, `test-automator`, `debugger`, `python-pro`, `performance-engineer`, `build-engineer`). The per-step `model` field in the JSON block documents the preferred model for that step (Kimi K3 for code/debug/python, Kimi highspeed for test) but is informational — the Task tool dispatches by base name and the harness default model applies unless overridden.

## Superpowers Workflow

- **Execution:** `subagent-driven-development` — fresh subagent per task, two-stage review between tasks. Tasks are **sequential** (nearly all edit `cpp/src/pipe1d.cu`).
- **TDD:** `test-driven-development` — Task 1 lands all failing tests first; each fix task flips its mapped test(s) to green. Do not write fix code before the failing test exists.
- **Cross-review (repo rule):** every C++ change by `cpp-pro` is reviewed by `debugger` (different subagent) before the task is marked complete: dispatch with the task's diff (`git diff`) and the audit finding; reviewer verifies correctness against the finding and checks for regressions in neighboring kernels.
- **On unexpected failure:** `systematic-debugging` — reproduce, isolate to kernel, form hypothesis, then fix. Do not patch blindly.
- **Before claiming completion of any task:** `verification-before-completion` — rebuild + run the mapped tests + phase gate, paste actual output.
- **Cache discipline (repo rule):** `find . -type d -name __pycache__ -exec rm -rf {} +` after every native rebuild before re-testing.

## Test → Fix Mapping

| Test (in `tests/test_pipe1d_mass_conservation.py`) | Finding | Fixed by |
|---|---|---|
| `test_closed_system_conserves_mass_diffusion_wave` | F1 | Task 2 |
| `test_closed_system_conserves_mass_fully_dynamic` | F1+F2 | Tasks 2, 5 |
| `test_node_outflow_limited_by_storage` | F5 | Task 2 |
| `test_first_substep_creates_no_water_*` | F3 | Task 3 |
| `test_inlet_node_drains_and_conserves` | F4 | Task 4 |
| `test_diffusion_checkerboard_decays` | F6 | Task 6 |
| `test_pipe_end_datum_outflow_direction` | F8 | Task 8 |
| `test_junction_overflow_reaches_surface` | F9 | Task 9 |
| `test_outfall_pipe_inflow_reaches_surface` | F10 | Task 10 |
| `test_elliptical_a_open_table` | F12 | Task 11 |
| `test_fractional_max_cell_length` | F13 | Task 12 |

---

## Phase A — Test Scaffolding

### Task 1: Failing mass-conservation regression tests

{"action": "write failing gpu validate tests for pipe1d mass conservation", "type": "test", "phase": "A"}
Agent: test-automator / kimi-for-coding/kimi-for-coding-highspeed. Cross-review: python-pro / kimi-for-coding/k3.

**Files:**
- Create: `tests/test_pipe1d_mass_conservation.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_pipe1d_mass_conservation.py` with exactly this content (coupled-setup tests for F8/F9/F10 reference upload patterns from `tests/test_swe2d_gpu_drainage_network.py:1089-1240` and `tests/test_swmm_validation_pipe_end.py:328` — copy their `swe2d_gpu_upload_drainage_exchange_params` / `swe2d_pipe1d_upload_pipe_ends_and_junctions` argument order):

```python
"""Mass-conservation regression tests for the pipe1d solver.

Each test maps to a finding in docs/PIPE1D_AUDIT_2026-07-17.md.
GPU-gated. Run:
    mamba run -n qgis_stable python3 -m unittest -v tests.test_pipe1d_mass_conservation
"""
from __future__ import annotations

import unittest

import numpy as np


def _load_module():
    try:
        import hydra_swe2d as m
        return m
    except Exception:
        return None


_MOD = _load_module()


def _gpu_available():
    if _MOD is None:
        return False
    try:
        return bool(_MOD.swe2d_gpu_available())
    except Exception:
        return False


G, K_MANN, H_MIN = 9.81, 1.0, 1.0e-4


@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestPipe1DMassConservation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from swe2d.runtime.backend import SWE2DBackend
        cls._backend = SWE2DBackend()
        cls._backend.build_mesh(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0, 1, 2], dtype=np.int32),
        )
        cls._backend.initialize(
            h0=np.asarray([0.1], dtype=np.float64),
            hu0=np.zeros(1, dtype=np.float64),
            hv0=np.zeros(1, dtype=np.float64),
            dt_fixed=0.05,
            dt_max=0.05,
        )
        cls._dev = int(_MOD.swe2d_get_coupling_dev_ptr())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls._backend, "destroy"):
            cls._backend.destroy()

    # ── helpers ──────────────────────────────────────────────────────────
    def _build_two_node_link(self, mcl=10.0, d0=1.0, d1=0.0,
                             node_area=(1.0, 1.0), init_from_depth=True):
        """Closed 2-node system, 1 link (audit probe 1).

        N0 invert 10 (depth d0) -- 100 m, D=0.5, n=0.013 --> N1 invert 9 (d1).
        Returns (n_cells, sub_len).
        """
        _MOD.swe2d_build_pipe1d_mesh(
            1,
            np.array([0], dtype=np.int32), np.array([1], dtype=np.int32),
            np.array([100.0]), np.array([0.5]), np.array([0.013]),
            np.array([0.0]), np.array([0.0]),
            np.array([10.0, 9.0]), np.array(list(node_area)),
            np.array([5.0, 5.0]),
            np.array([10.0]), np.array([9.0]),
            mcl, self._dev,
            np.zeros(1, dtype=np.int32),
            np.array([0.5]), np.array([0.5]),
        )
        n_sub = max(1, int(np.ceil(100.0 / mcl)))
        _MOD.swe2d_pipe1d_upload_node_depth(self._dev, np.array([d0, d1]))
        if init_from_depth:
            _MOD.swe2d_pipe1d_init_area_from_depth(self._dev, H_MIN)
        return n_sub, 100.0 / n_sub

    def _mass(self, n_cells, sub_len, node_area=(1.0, 1.0)):
        st = _MOD.swe2d_pipe1d_readback_node_state(self._dev, 2, n_cells)
        nd = np.asarray(st["node_depth"], dtype=np.float64)
        A = np.asarray(st.get("cell_A", np.zeros(n_cells)), dtype=np.float64)
        q = np.asarray(st.get("cell_Q", np.zeros(n_cells)), dtype=np.float64)
        total = float(np.sum(A) * sub_len + np.sum(nd * np.asarray(node_area)))
        return total, nd, A, q

    def _step(self, mode, dt=0.5):
        _MOD.swe2d_pipe1d_step(self._dev, dt, mode, 1, 2, 0.5, G, K_MANN, H_MIN)

    # ── F1: node balance must consume the same fluxes as cell continuity ──
    def _run_closed(self, mode, n_steps=200):
        n_sub, sub_len = self._build_two_node_link()
        m0, _, _, _ = self._mass(n_sub, sub_len)
        for _ in range(n_steps):
            self._step(mode)
        m1, _, _, _ = self._mass(n_sub, sub_len)
        return m0, m1

    def test_closed_system_conserves_mass_diffusion_wave(self):
        m0, m1 = self._run_closed("diffusion_wave")
        self.assertGreater(m0, 0.0)
        self.assertAlmostEqual(
            m1 / m0, 1.0, places=4,
            msg=f"diffusion_wave drift {(m1 - m0) / m0:+.4%} over 200 steps (F1)")

    def test_closed_system_conserves_mass_fully_dynamic(self):
        m0, m1 = self._run_closed("fully_dynamic")
        self.assertGreater(m0, 0.0)
        self.assertAlmostEqual(
            m1 / m0, 1.0, places=4,
            msg=f"fully_dynamic drift {(m1 - m0) / m0:+.4%} over 200 steps (F1+F2)")

    # ── F5: node outflow limited by available storage ─────────────────────
    def test_node_outflow_limited_by_storage(self):
        for mode in ("diffusion_wave", "fully_dynamic"):
            with self.subTest(mode=mode):
                n_sub, sub_len = self._build_two_node_link(
                    d0=0.05, d1=0.0, node_area=(0.05, 0.05))
                m0, _, _, _ = self._mass(n_sub, sub_len, (0.05, 0.05))
                for _ in range(20):
                    self._step(mode, dt=2.0)
                m1, nd, _, _ = self._mass(n_sub, sub_len, (0.05, 0.05))
                self.assertLessEqual(
                    m1, m0 * (1.0 + 1e-9),
                    f"{mode}: mass created {m1 - m0:+.3e} m3 when node emptied (F5)")
                self.assertGreaterEqual(
                    m1, m0 * (1.0 - 1e-4),
                    f"{mode}: mass destroyed {m1 - m0:+.3e} m3 (F5)")

    # ── F3: first substep must not read cell_y == 0 as absolute WSE ───────
    def test_first_substep_creates_no_water(self):
        for mode in ("diffusion_wave", "fully_dynamic"):
            with self.subTest(mode=mode):
                # Dry pipe: skip init_area_from_depth so d_A stays ~0.
                n_sub, sub_len = self._build_two_node_link(
                    d0=0.3, d1=0.0, init_from_depth=False)
                m0, _, _, _ = self._mass(n_sub, sub_len)
                self._step(mode, dt=0.5)
                m1, _, _, _ = self._mass(n_sub, sub_len)
                self.assertLess(
                    m1 - m0, 1e-6 * m0 + 1e-9,
                    f"{mode}: step 0 created {m1 - m0:+.4e} m3 from nothing (F3)")

    # ── F4: inlet nodes must be debited by the boundary-face flux ─────────
    def test_inlet_node_drains_and_conserves(self):
        # Mark node 0 as inlet via the drainage exchange upload; mirror the
        # argument order used in tests/test_swe2d_gpu_drainage_network.py
        # (swe2d_gpu_upload_drainage_exchange_params call near line 1089).
        n_sub, sub_len = self._build_two_node_link(d0=0.3, d1=0.0)
        self._mark_inlet_node0()
        m0, _, _, _ = self._mass(n_sub, sub_len)
        depths = []
        for _ in range(10):
            self._step("fully_dynamic")
            _, nd, _, _ = self._mass(n_sub, sub_len)
            depths.append(float(nd[0]))
        m1, _, _, _ = self._mass(n_sub, sub_len)
        self.assertLess(
            depths[-1], 0.3 - 1e-6,
            f"inlet node depth pinned at {depths[-1]:.5f} — infinite source (F4)")
        self.assertAlmostEqual(
            m1 / m0, 1.0, places=4,
            msg=f"inlet system drift {(m1 - m0) / m0:+.4%} (F4)")

    def _mark_inlet_node0(self):
        """Upload one inlet assignment at node 0 (no 2D coupling cells)."""
        # Copy the exact swe2d_gpu_upload_drainage_exchange_params call from
        # tests/test_swe2d_gpu_drainage_network.py (~line 1089) with:
        #   n_inlets=1, inlet_node=[0], inlet cell = -1 (no capture source),
        #   n_outfalls=0, n_pipe_ends=0.
        raise NotImplementedError(
            "mirror the upload call from tests/test_swe2d_gpu_drainage_network.py")

    # ── F6: correct wave speed kills the odd-even checkerboard ────────────
    def test_diffusion_checkerboard_decays(self):
        n_sub, sub_len = self._build_two_node_link(
            mcl=10.0, d0=0.25, d1=0.25)  # 10 sub-cells, half-full
        amps = []
        for _ in range(30):
            self._step("diffusion_wave", dt=0.5)
            _, _, _, q = self._mass(n_sub, sub_len)
            amps.append(float(np.max(np.abs(q))))
        early = max(amps[:5])
        late = max(amps[-5:])
        self.assertLess(
            late, 0.5 * early + 1e-9,
            f"checkerboard not decaying: early={early:.4f} late={late:.4f} (F6)")

    # ── F13: fractional max_cell_length must subdivide ────────────────────
    def test_fractional_max_cell_length(self):
        n_sub, _ = self._build_two_node_link(mcl=0.3, d0=0.0, d1=0.0)
        # L=100, mcl=0.3 -> ceil(100/0.3) = 334 sub-cells (was 100 with int cast)
        self.assertEqual(n_sub, 334)
        n_cells_native = _MOD.swe2d_pipe1d_get_cell_count(self._dev)
        self.assertEqual(n_cells_native, 334)
```

- [ ] **Step 2: Implement `_mark_inlet_node0` and the coupled-setup tests**

Read `tests/test_swe2d_gpu_drainage_network.py:1050-1250` and `tests/test_swmm_validation_pipe_end.py:280-360`. Using their exact binding argument patterns, add to the same test file:

1. `_mark_inlet_node0` — call `swe2d_gpu_upload_drainage_exchange_params` with `n_inlets=1`, `inlet_node=[0]`, all other arrays empty (sizes 0), mirroring the drainage-network test's call exactly.
2. `test_pipe_end_datum_outflow_direction` (F8) — sloped pipe (node inverts 10→9, L=100, D=0.5, mcl=50 → 2 sub-cells; downstream cell invert ≈ 9.25). Half-fill pipe (d0=d1=0.25 → true end-cell WSE ≈ 9.5). Rebuild the 2D backend with `node_z = [9.35, 9.35, 9.35]` and `h0 = [0.0]` so surface WSE = 9.35 (below true head 9.5, above buggy head 9.25). Upload one pipe-end (node 1, cell 0, invert 9.0, D=0.5) via `swe2d_pipe1d_upload_pipe_ends_and_junctions` (pattern at `test_swmm_validation_pipe_end.py:328`). Run one `swe2d_pipe1d_step`. Read `swe2d_gpu_readback_coupling_sources(1)` — assert `source[0] > 0` (buggy datum: ≤ 0). Note: pipe1d_step folds into `d_external_source_mps` in-step at `pipe1d.cu:2802-2806`; if Task 9 has already removed that fold, call `swe2d_gpu_compute_coupling_full_on_device` before readback instead — pick per current code state.
3. `test_junction_overflow_reaches_surface` (F9) — junction at node 1 with rim = invert + 0.4, overflow diam 0.3, coeff 0.6, coupled to 2D cell 0. Set d0=d1=0.8 (WSE 9.8 > rim 9.4). Run `swe2d_pipe1d_step` then `swe2d_gpu_compute_coupling_full_on_device`; assert coupling source at cell 0 > 0 and node 1 depth decreased versus pre-overflow value. Verify the junction upload wiring for `d_node_rim` / overflow arrays in `swe2d_pipe1d_upload_pipe_ends_and_junctions` (`pipe1d.cu:3627`); if rim is not wired, note it for Task 9.
4. `test_outfall_pipe_inflow_reaches_surface` (F10) — 2-node link draining to an outfall at node 1 coupled to 2D cell 0 (pattern from `test_swe2d_gpu_drainage_network.py`). d0=1.0, d1=0.0. Run `swe2d_pipe1d_step` then `swe2d_gpu_compute_coupling_full_on_device`; assert coupling source at cell 0 > 0 (buggy: 0 — mass vanished).

- [ ] **Step 3: Run to verify the tests FAIL for the right reasons**

```bash
mamba run -n qgis_stable python3 -m unittest -v tests.test_pipe1d_mass_conservation
```
Expected: FAILURES — closed-system drift ≫ 0.01 % (audit: −91 %), first-substep creation > 0.1 m³, inlet depth pinned at 0.30000, checkerboard not decaying, coupled tests at 0 or NotImplementedError remnants. `test_fractional_max_cell_length` errors on missing `swe2d_pipe1d_get_cell_count` (added in Task 12) — mark it with `@unittest.skipUnless(hasattr(_MOD, "swe2d_pipe1d_get_cell_count"), "binding not yet added")` so it skips instead of erroring.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipe1d_mass_conservation.py
git commit -m "test: pipe1d mass-conservation regression tests (audit 2026-07-17)"
```

---

## Phase B — P0: Mass-Conservation Fixes

### Task 2: F1+F5 — flux kernel accumulates boundary-face flux into `node_net_q`; supply cap

{"action": "rewrite c++ cuda flux kernel and node mass balance so nodes consume boundary-face fluxes", "type": "coding", "phase": "B"}
Agent: cpp-pro / kimi-for-coding/k3. Cross-review: debugger / kimi-for-coding/k3.

**Files:**
- Modify: `cpp/src/pipe1d.cu` (flux kernel 1129-1406, host wrapper 2073+, mass-balance kernels 2245-2378, step 2535-2655)
- Modify: `cpp/src/pipe1d.cuh` (flux kernel + host wrapper declarations ~313, accumulate-kernel decl if present)

**Design (binding decisions — implement exactly):**

1. `swe2d_pipe1d_flux_kernel` gains three appended parameters (after `vnode_idx`):
   `double* node_net_q, const double* __restrict__ node_surface_area, const int32_t* __restrict__ node_is_boundary`.
2. In the boundary branch (`nbr < 0`), after all existing zeroing rules (dry-cell, dry-node, pipe-end), insert the supply cap and accumulation:

```cpp
            // ── Node mass balance coupling (audit F1 + F5) ─────────────
            // The node must see the SAME mass the cell continuity sees:
            // accumulate dir*F (positive = entering the node) for every
            // boundary face. Storage nodes are supply-limited (F5); boundary
            // (prescribed-head) nodes are exempt — their exchange is settled
            // with the 2D surface by the outfall/pipe-end exchange kernels.
            if (shared_node >= 0 && shared_node < n_nodes && node_net_q) {
                const bool is_bnd = node_is_boundary && node_is_boundary[shared_node];
                if (!is_bnd && dir * F < 0.0) {
                    const double area_n = (node_surface_area)
                        ? fmax(node_surface_area[shared_node], 1.0e-6) : 1.0e-6;
                    // node_net_q < 0: flux drafted from this node in earlier
                    // substeps of THIS step (node_depth is constant across
                    // substeps). Rate-space cap: remaining volume / local_dt.
                    const double drafted = (node_net_q[shared_node] < 0.0)
                        ? -node_net_q[shared_node] : 0.0;
                    const double supply_rate =
                        node_depth[shared_node] * area_n / fmax(dt, 1.0e-12) - drafted;
                    double r = -dir * F;                 // > 0 = node supplying cell
                    if (r > supply_rate) {
                        r = fmax(0.0, supply_rate);
                        F = -dir * r;                    // rescale, keep sign convention
                    }
                }
                atomicAdd(&node_net_q[shared_node], dir * F);
            }
```

   This block goes immediately before the shared `total_flux += dir * F;` at `pipe1d.cu:1399` — but it must only run for boundary faces. Simplest correct placement: inside the `else` (boundary) branch, right before its closing brace at `pipe1d.cu:1396`, after the pipe-end zeroing at 1392-1395 (so pipe-end nodes contribute 0 — their exchange path is the weir/orifice kernel).
3. `swe2d_pipe1d_step` (`pipe1d.cu:2570-2572`): zero the accumulator BEFORE the substep loop:

```cpp
    const double local_dt = dt / static_cast<double>(coupling_substeps);

    // F1: node_net_q is accumulated by the flux kernel across substeps.
    CUDA_CHECK(cudaMemsetAsync(p.d_node_net_q, 0,
        static_cast<size_t>(p.n_nodes + p.n_vnodes) * sizeof(double), dev->d_stream));
```

4. Pass the new args at the flux-kernel host call (`pipe1d.cu:2577-2601`): append `p.d_node_net_q, p.d_node_surface_area, p.d_node_is_boundary`.
5. `swe2d_pipe1d_node_mass_balance_host` (`pipe1d.cu:2333-2378`): delete the memset (moved to step) and the entire `swe2d_pipe1d_accumulate_node_flux_kernel` launch block (2360-2368). Keep only the depth-update launch. Its `dt` parameter now receives `local_dt`.
6. Update the step's mass-balance call (`pipe1d.cu:2655`): `swe2d_pipe1d_node_mass_balance_host(dev, local_dt, g);`
7. Delete `swe2d_pipe1d_accumulate_node_flux_kernel` (`pipe1d.cu:2254-2281`) and any declaration of it in `pipe1d.cuh` (grep to confirm). Also delete the now-dead comment about `cell_Q`-based accumulation.
8. Update `swe2d_pipe1d_flux_kernel_host` signature (def ~2073 + decl `pipe1d.cuh:313`) with the three new parameters.
9. Update the kernel doc header (`pipe1d.cu:2283-2291`) — the depth-update kernel behavior is unchanged; the accumulation source is now the flux kernel.

**Invariant to preserve:** vnode slots in `node_net_q` are never written by the flux kernel (guard `shared_node < n_nodes`), so the depth-update kernel semantics are unchanged.

- [ ] **Step 1: Apply the edits above** (read each region first; match existing style; no new comments beyond the ones shown).

- [ ] **Step 2: Rebuild + purge cache**

```bash
mamba run -n qgis_stable cmake --build . -j$(nproc)   # workdir: build/
find . -type d -name __pycache__ -exec rm -rf {} +     # workdir: repo root
```

- [ ] **Step 3: Run mapped tests**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation.TestPipe1DMassConservation.test_closed_system_conserves_mass_diffusion_wave \
    tests.test_pipe1d_mass_conservation.TestPipe1DMassConservation.test_node_outflow_limited_by_storage
```
Expected: PASS (drift < 0.01 %). `test_closed_system_conserves_mass_fully_dynamic` may still fail — it also needs Task 5.

- [ ] **Step 4: Run the pre-existing kernel suite for regressions**

```bash
mamba run -n qgis_stable python3 -m unittest -v tests.test_swe2d_pipe1d tests.test_swe2d_pipe1d_surcharge
```
Expected: PASS. If a pre-existing test asserted defect-dependent behavior, record it for the cross-review; do not edit it in this task.

- [ ] **Step 5: Cross-review** — dispatch `debugger` with `git diff` + audit findings F1/F5. Address or explicitly rebut findings.

- [ ] **Step 6: Commit**

```bash
git add cpp/src/pipe1d.cu cpp/src/pipe1d.cuh
git commit -m "fix(pipe1d): node mass balance consumes boundary-face fluxes (audit F1+F5)"
```

### Task 3: F3 — first-substep boundary WSE guard

{"action": "guard c++ cuda boundary flux branch against cell_y zero on first substep", "type": "coding", "phase": "B"}
Agent: cpp-pro / kimi-for-coding/k3. Cross-review: debugger.

**Files:**
- Modify: `cpp/src/pipe1d.cu:1364`

- [ ] **Step 1: Apply the one-line fix** — mirror the interior branch guard (`pipe1d.cu:1213`):

```cpp
// before:
const double H_end = cell_y ? cell_y[c] : H_c;
// after:
const double H_end = (cell_y && cell_y[c] != 0.0) ? cell_y[c] : H_c;
```

- [ ] **Step 2: Rebuild + purge, run:**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation.TestPipe1DMassConservation.test_first_substep_creates_no_water \
    tests.test_swe2d_pipe1d
```
Expected: PASS.

- [ ] **Step 3: Cross-review, then commit**

```bash
git add cpp/src/pipe1d.cu
git commit -m "fix(pipe1d): guard boundary flux against zero cell_y on first substep (audit F3)"
```

### Task 4: F4+F11 — remove inlet overrides (infinite source + OOB read)

{"action": "remove c++ cuda inlet flux overrides so inlet nodes are debited by boundary flux", "type": "coding", "phase": "B"}
Agent: cpp-pro / kimi-for-coding/k3. Cross-review: debugger.

**Files:**
- Modify: `cpp/src/pipe1d.cu` (flux kernel 1280-1288; fully-dynamic kernel 2048-2060)

**Rationale:** With Task 2, inlet nodes (not marked `is_boundary`) are debited/credited by boundary-face fluxes through `node_net_q`, and the inlet exchange kernel credits node storage via `d_node_delta`. The two overrides are therefore both wrong (F4 infinite source) and memory-unsafe (F11: `node_is_inlet[cell_to_node[c]]` reads vnode indices ≥ `n_nodes` on an array sized `n_nodes`, `pipe1d.cu:890`).

- [ ] **Step 1: Delete the interior-branch inlet override** (`pipe1d.cu:1280-1288`):

```cpp
// DELETE this entire block:
            // For inlet nodes, override with a "neutral" neighbor so the
            // prescribed-flow BC is honoured (inlet exchange kernel supplies
            // the flow) — same convention as the previous implementation.
            const bool shared_is_inlet = node_is_inlet && (
                (dir > 0.0) ? (node_is_inlet[cell_to_node[c]])
                            : (node_is_inlet[cell_from_node[c]]));
            if (shared_is_inlet) {
                F = 0.0;
            }
```

- [ ] **Step 2: Delete the fully-dynamic inlet override** (`pipe1d.cu:2048-2060`):

```cpp
// DELETE this entire block:
    // Inlet boundary-condition override (prescribed flow BC).
    ...
    if (node_is_inlet != nullptr) {
        const bool fn_inlet = ...
        ...
        if (fn_inlet || tn_inlet) {
            cell_Q_new[c] = 0.0;
        }
    }
```

`node_is_inlet` remains a legitimate kernel parameter (used in the flux-kernel preamble head-averaging skip at `pipe1d.cu:1201-1206`, which is already guarded by `< n_nodes` at line 1195 — verify that guard is still intact).

- [ ] **Step 3: Rebuild + purge, run:**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation.TestPipe1DMassConservation.test_inlet_node_drains_and_conserves \
    tests.test_swe2d_gpu_drainage_network
```
Expected: inlet test PASSES (depth drops below 0.3, mass conserved); drainage-network suite PASS.

- [ ] **Step 4: Cross-review, then commit**

```bash
git add cpp/src/pipe1d.cu
git commit -m "fix(pipe1d): remove inlet flux/Q overrides; inlet nodes debited by face flux (audit F4+F11)"
```

### Task 5: F2 — anchor fully-dynamic continuity on `A_orig`, once per substep

{"action": "fix c++ cuda fully-dynamic kernel to anchor continuity on A_orig outside picard loop", "type": "coding", "phase": "B"}
Agent: cpp-pro / kimi-for-coding/k3. Cross-review: debugger.

**Files:**
- Modify: `cpp/src/pipe1d.cu` (fully-dynamic kernel 1836-1974)

- [ ] **Step 1: Hoist continuity out of the Picard loop.** In the `half` loop, after the restart block (`pipe1d.cu:1848-1854`), insert:

```cpp
        // SPEC §2.14 — Continuity is explicit: computed ONCE per substep
        // attempt from the start-of-substep area and the fixed face flux.
        // The Picard loop below iterates on Q only (audit F2).
        double A_cont = A_orig - local_dt * flux_Q[c] / L;
        if (surcharge_method == SURCHARGE_SLOT) {
            A_cont = fmax(A_floor, A_cont);
        } else {
            A_cont = fmax(A_floor, fmin(A_full, A_cont));
        }
        A_curr = A_cont;
```

- [ ] **Step 2: Delete the per-iteration continuity block** (`pipe1d.cu:1870-1881`, the `A_new_cont` computation and its clamps) and delete `A_curr = A_new_cont;` at `pipe1d.cu:1932` (keep `Q_curr = Q_new;`).

- [ ] **Step 3: Convergence/stall on Q only** (`pipe1d.cu:1935-1952`): replace the block with:

```cpp
            // Convergence check (SPEC §2.14) — Q only; A is now explicit.
            const double dQ = fabs(Q_new - Q_iter_prev);
            if (dQ < tol_Q) {
                half_converged = true;
                break;
            }

            // Detect stall for halving decision (≥2 non-improving iters).
            const bool not_improving = (dQ >= 0.95 * last_dQ);
            stall_count = not_improving ? (stall_count + 1) : 0;
            last_dQ = dQ;

            if (stall_count >= 2) break;
```

Delete `A_iter_prev`, `tol_A`, `last_dA` declarations/usages that become unused.

- [ ] **Step 4: Rebuild + purge, run:**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation.TestPipe1DMassConservation.test_closed_system_conserves_mass_fully_dynamic \
    tests.test_swe2d_pipe1d tests.test_swe2d_pipe1d_surcharge tests.test_pipe1d_vs_swmm
```
Expected: PASS. (The diffusion checkerboard test may still fail — that is Task 6.)

- [ ] **Step 5: Cross-review, then commit**

```bash
git add cpp/src/pipe1d.cu
git commit -m "fix(pipe1d): anchor fully-dynamic continuity on A_orig once per substep (audit F2)"
```

---

## Phase C — P1: Physics / Formulation Fixes

### Task 6: F6 — correct wave speed `sqrt(g·A/T)`

{"action": "fix c++ cuda wave speed to hydraulic-depth celerity sqrt(g*A/T)", "type": "coding", "phase": "C"}
Agent: cpp-pro / kimi-for-coding/k3. Cross-review: debugger.

**Files:**
- Modify: `cpp/src/pipe1d.cu:1259-1265` (interior), `cpp/src/pipe1d.cu:1362-1365` (boundary)

- [ ] **Step 1: Interior face** — replace the dimensionally wrong `sqrt(g·|ΔH|/L)`:

```cpp
            // HLLE wave speed (m/s): hydraulic-depth celerity sqrt(g·A/T),
            // averaged across the face (audit F6). L_avg is still used by
            // the CFL clamp below.
            const double L_c  = fmax(cell_length[c],  1.0e-12);
            const double L_n  = fmax(cell_length[nbr], 1.0e-12);
            const double L_avg = 0.5 * (L_c + L_n);
            const double hd_c = A_c_safe / T_c_safe;
            const double hd_n = fmax(A_n, 1.0e-12) / fmax(T_n, 1.0e-10);
            const double c_wave = sqrt(g * 0.5 * (hd_c + hd_n));
```

(`A_c_safe`, `T_c_safe` already exist at `pipe1d.cu:1184-1185`; `T_n` from the geometry lookup at 1248-1251.)

- [ ] **Step 2: Boundary face** — replace:

```cpp
            // Wave speed (m/s): hydraulic-depth celerity of the end cell
            // (audit F6). Replaces the dimensionally wrong sqrt(g·|ΔH|/L).
            const double H_end = (cell_y && cell_y[c] != 0.0) ? cell_y[c] : H_c;
            const double c_wave = sqrt(g * (fmax(A_eff, 1.0e-12) / T_c_safe));
```

- [ ] **Step 3: Rebuild + purge, run:**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation \
    tests.test_swe2d_pipe1d tests.test_pipe1d_vs_swmm
```
Expected: `test_diffusion_checkerboard_decays` PASS; all conservation tests still PASS (the CFL clamp bounds the larger celerity).

- [ ] **Step 4: Cross-review, then commit**

```bash
git add cpp/src/pipe1d.cu
git commit -m "fix(pipe1d): wave speed from hydraulic depth sqrt(g*A/T) (audit F6)"
```

### Task 7: F7 — regime-override guards + hydraulic-depth Froude

{"action": "fix c++ cuda regime override to fire only downhill open-channel and use A/T froude depth", "type": "coding", "phase": "C"}
Agent: cpp-pro / kimi-for-coding/k3. Cross-review: debugger.

**Files:**
- Modify: `cpp/src/pipe1d.cu:1627-1668` (diffusion), `cpp/src/pipe1d.cu:1883-1894` (sigma), `cpp/src/pipe1d.cu:1991-2031` (fully-dynamic)

- [ ] **Step 1: Hydraulic-depth Froude everywhere.** Replace `sqrt(g * R_h)` / `sqrt(g * R_h_new)` / `sqrt(g * R_h_local)` with `sqrt(g * A / fmax(T, 1.0e-10))` at:
  - `pipe1d.cu:1633` (diffusion `fr`): `double fr = fabs(q_new) / fmax(1.0e-12, sqrt(g * A_new / fmax(T_new, 1.0e-10)));`
  - `pipe1d.cu:1889-1891` (sigma `fr`): `abs_v / sqrt(g * A_eff / fmax(T_c, 1.0e-10))`
  - `pipe1d.cu:1996` and `2029` (fully-dynamic `fr`): `sqrt(g * A_final / fmax(T_new, 1.0e-10))`

- [ ] **Step 2: Trigger guards** — in BOTH override blocks (`pipe1d.cu:1644-1655` and `2007-2018`), replace the trigger computation with downhill-only, non-pressurised logic (SWMM `checkNormalFlow` semantics):

```cpp
        const double S0_cell = (cell_S0 != nullptr) ? cell_S0[c] : 0.0;
        const double Sf_HGL = (H_up - H_dn) / L;
        // SWMM checkNormalFlow: only for downhill supercritical flow.
        // Never fires for backwater (Sf·S0 < 0), adverse flow, or
        // pressurised (SLOT surcharge) cells (audit F7).
        const bool pressurised = (surcharge_method == SURCHARGE_SLOT)
                                 && (A_new > A_full);   // use A_final in the fully-dynamic kernel
        const bool downhill = (S0_cell > 0.0 && Sf_HGL > 0.0 && Q_new > 0.0)
                           || (S0_cell < 0.0 && Sf_HGL < 0.0 && Q_new < 0.0);
        const bool supercritical_by_slope = downhill
            && (fabs(Sf_HGL) < fabs(S0_cell) - 1.0e-6);
        const bool fr_supercritical = !pressurised && (fr >= 1.0);
        const bool is_end_cell_downstream = (cell_is_end != nullptr)
            && (cell_is_end[c] != 0)
            && (to_node >= 0) && (to_node < n_nodes);
        const bool downstream_is_outfall = !pressurised && is_end_cell_downstream
            && (node_is_outfall != nullptr) && (node_is_outfall[to_node] != 0);

        if (!pressurised && (supercritical_by_slope || fr_supercritical || downstream_is_outfall)) {
```

(In the diffusion kernel use `A_new`/`Q_new`; in the fully-dynamic kernel use `A_final`/`Q_curr` — adjust names to each kernel's locals.)

- [ ] **Step 3: Rebuild + purge, run the full Phase C gate:**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation tests.test_swe2d_pipe1d \
    tests.test_swe2d_pipe1d_surcharge tests.test_pipe1d_vs_swmm \
    tests.test_swmm_validation_pipe_end
```
Expected: PASS. Pay attention to SWMM-comparison tests — the override now fires less often (backwater/surcharge no longer capped).

- [ ] **Step 4: Cross-review, then commit**

```bash
git add cpp/src/pipe1d.cu
git commit -m "fix(pipe1d): regime override downhill-only, hydraulic-depth Froude (audit F7)"
```

### Task 8: F8 — pipe-end head datum bias

{"action": "fix c++ cuda pipe-end node head to convert cell depth to node datum", "type": "coding", "phase": "C"}
Agent: cpp-pro / kimi-for-coding/k3. Cross-review: debugger.

**Files:**
- Modify: `cpp/src/pipe1d.cu:3273-3317` (`swe2d_pipe_end_clamp_area_kernel`), `3335-3372` (`swe2d_pipe_end_update_node_head_kernel` + host), `2667-2676` (call site), `3374-3421` (weir/orifice host wrapper), `pipe1d.cuh` (decls)

**Defect:** `node_depth[n] = cell_h[c_pipe]` stores depth above the sub-cell midpoint invert into a field interpreted as depth above the node invert — bias `±sub_len/2·S0` on the weir/orifice head.

- [ ] **Step 1: `swe2d_pipe_end_update_node_head_kernel`** — switch to WSE-based conversion. New signature adds `const double* cell_y` and `const double* node_invert`, drops `cell_h`:

```cpp
__global__ __launch_bounds__(256, 4) void swe2d_pipe_end_update_node_head_kernel(
    int32_t n_pipe_ends,
    const int32_t* __restrict__ pipe_end_node,
    const int32_t* __restrict__ pipe_end_pipe_cell,
    const double*  __restrict__ cell_y,        // WSE per pipe cell (node datum)
    const double*  __restrict__ node_invert,
    double* __restrict__ node_depth)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pipe_ends) return;
    const int32_t n = pipe_end_node[i];
    const int32_t c_pipe = pipe_end_pipe_cell[i];
    if (n < 0 || c_pipe < 0) return;

    // Audit F8: node_depth is depth above the NODE invert; convert the
    // open-end cell WSE to the node datum.
    node_depth[n] = fmax(0.0, cell_y[c_pipe] - node_invert[n]);
}
```

Update host wrapper + `pipe1d.cuh` decl + call site (`pipe1d.cu:2673-2675`) to pass `p.d_cell_y, p.d_node_invert, p.d_node_depth`.

- [ ] **Step 2: `swe2d_pipe_end_clamp_area_kernel`** (`pipe1d.cu:3316`) — same datum bug after the exchange updates `A`. Add `const double* cell_invert` and `const double* node_invert` params; replace:

```cpp
    if (n >= 0 && node_depth) node_depth[n] = fmax(0.0, h_new);
// with:
    if (n >= 0 && node_depth && cell_invert && node_invert)
        node_depth[n] = fmax(0.0, h_new + cell_invert[c_pipe] - node_invert[n]);
```

Thread `cell_invert` through: kernel launch at `pipe1d.cu:3418-3421`, host wrapper `swe2d_pipe_end_weir_orifice_kernel_host` (new param, e.g. after `pipe_cell_length`), `pipe1d.cuh` decl, and the step call site (`pipe1d.cu:2769-2793`) passing `p.d_cell_invert`.

- [ ] **Step 3: Rebuild + purge, run:**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation.TestPipe1DMassConservation.test_pipe_end_datum_outflow_direction \
    tests.test_swmm_validation_pipe_end tests.test_swe2d_pipe1d
```
Expected: PASS.

- [ ] **Step 4: Cross-review, then commit**

```bash
git add cpp/src/pipe1d.cu cpp/src/pipe1d.cuh
git commit -m "fix(pipe1d): pipe-end node head converted to node datum (audit F8)"
```

### Task 9: F9+F14 — junction surcharge path + single pipe-end fold site

{"action": "rework c++ cuda junction overflow path to conserve mass and use single fold site", "type": "coding", "phase": "C"}
Agent: cpp-pro / kimi-for-coding/k3. Cross-review: debugger.

**Files:**
- Modify: `cpp/src/pipe1d.cu` (junction BC 3565-3595; step ordering 2654-2807; mesh build crown population ~589-1086; overflow host 4060-4075)
- Modify: `cpp/src/swe2d_gpu.cu:8292-8304` (keep — the single fold site; verify only)

**Facts (verified during planning):** `swe2d_pipe1d_step` runs before `compute_coupling_full_on_device` each timestep; the latter zeroes `d_external_source_mps` (`swe2d_gpu.cu:8009`) then folds `d_pipe_end_q_2d` (`swe2d_gpu.cu:8299-8304`). The in-step fold (`pipe1d.cu:2802-2806`) is therefore always wiped — it is the site to delete.

- [ ] **Step 1: Junction BC floor-only clamp** (`pipe1d.cu:3578-3583`) — delete the `max_d` upper clamp (surcharge volume must not be deleted; the mass-balance kernel deliberately does not clamp, `pipe1d.cu:2283-2291`):

```cpp
    double d = d_node_depth[n];
    if (d < 0.0) d = 0.0;
    d_node_depth[n] = d;
```

Keep the crown branch (3585-3594) — it becomes live once crown is populated (Step 2).

- [ ] **Step 2: Populate `d_node_crown` at mesh build.** In `swe2d_build_pipe1d_mesh` (host, after the link loop that has `link_from_node`/`link_to_node` and per-link geometry, near `pipe1d.cu:892-894` where `d_node_crown` is allocated), compute on host and upload:

```cpp
    // Audit F9: crown = node invert + max height of connected links.
    std::vector<double> node_crown(n_nodes, 0.0);
    for (int32_t i = 0; i < n_links; ++i) {
        const double h_link = (link_shape_type && link_shape_type[i] != 0)
            ? (link_height ? link_height[i] : link_diameter[i])
            : link_diameter[i];
        const int32_t fn = link_from_node[i], tn = link_to_node[i];
        if (fn >= 0 && fn < n_nodes)
            node_crown[fn] = std::max(node_crown[fn], node_invert_elev[fn] + h_link);
        if (tn >= 0 && tn < n_nodes)
            node_crown[tn] = std::max(node_crown[tn], node_invert_elev[tn] + h_link);
    }
    copy_h2d_d(dev->d_node_crown, node_crown.data(), static_cast<size_t>(n_nodes));
```

Also grep `d_node_rim` population; if it is never uploaded, populate it in `swe2d_pipe1d_upload_pipe_ends_and_junctions` from the junction max-depth (rim = invert + node_max_depth) so the overflow kernel's rim check (`pipe1d.cu:4022`) can fire. Note the chosen wiring in the commit message.

- [ ] **Step 3: Retarget junction overflow into `d_pipe_end_q_2d` and reorder.** In `swe2d_pipe1d_step`:
  1. Hoist the exchange-buffer alloc/zero (currently inside the pipe-end-only block at `pipe1d.cu:2699-2731`) so it runs when `p.d_n_pipe_ends > 0 || p.d_n_junctions > 0`, immediately after the junction BC call.
  2. Move the junction overflow call (`pipe1d.cu:2662-2665`) to AFTER that zero.
  3. Change the overflow host wrapper's output buffer (`pipe1d.cu:4073`) from `dev->drain_ws.d_q_cell` to `p.d_pipe_end_q_2d`.
  4. Delete the in-step fold (`pipe1d.cu:2796-2806`, the `swe2d_fold_pipe_end_q_to_source_kernel` launch inside the step) — the single fold site is `swe2d_gpu.cu:8299-8304`.

- [ ] **Step 4: Verify the surviving fold site** (`swe2d_gpu.cu:8292-8304`) is unchanged and runs every timestep in both graph and non-graph paths (it is inside `compute_coupling_full_on_device`, called from `apply_native_device_sources`, `coupling.py:1741`, and from `swe2d_recompute_coupling_for_stage`, `swe2d_gpu.cu:8316-8323`). No edit expected — confirm by inspection and state the confirmation in the commit message.

- [ ] **Step 5: Rebuild + purge, run:**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation.TestPipe1DMassConservation.test_junction_overflow_reaches_surface \
    tests.test_swe2d_gpu_drainage_network tests.test_pipe_cell_coupling_output
```
Expected: PASS.

- [ ] **Step 6: Cross-review, then commit**

```bash
git add cpp/src/pipe1d.cu cpp/src/swe2d_gpu.cu
git commit -m "fix(pipe1d): junction overflow conserved + single pipe-end fold site (audit F9+F14)"
```

### Task 10: F10 — route pipe inflow at 2D-coupled outfalls to the surface

{"action": "route c++ cuda outfall pipe inflow to coupled surface cell via node_net_q", "type": "coding", "phase": "C"}
Agent: cpp-pro / kimi-for-coding/k3. Cross-review: debugger.

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu:4710-4818` (outfall exchange kernel), `8236-8245` (launch), and the kernel's host wrapper/declaration if separate
- Modify: `cpp/src/swe2d_gpu.cuh` if the kernel is declared there

**Scope decision (audit-limited):** implement ONLY the pipe→surface direction (the audit's finding). Backflow from fixed-head outfall BCs into the pipe is external boundary inflow, not surface-sourced — leave unchanged and note as a known limitation in the commit message.

- [ ] **Step 1: Add `node_net_q` to `swe2d_drainage_outfall_exchange_kernel`** — append parameter `const double* __restrict__ node_net_q`; at the end of the kernel body (after the existing surcharge/backwater logic), add:

```cpp
    // Audit F10: pipe water arriving at this (is_boundary) outfall node was
    // removed from the pipe cell by the boundary-face flux and accumulated
    // into node_net_q by the pipe1d flux kernel. The mass-balance depth
    // update skips boundary nodes, so route the net pipe inflow straight to
    // the coupled surface cell.
    const double q_pipe = node_net_q ? node_net_q[n] : 0.0;
    if (q_pipe > 0.0) {
        atomicAdd(&q_cell[c], q_pipe);
    }
```

- [ ] **Step 2: Pass `p.d_node_net_q` at the launch** (`swe2d_gpu.cu:8236-8245`, append after `p.d_node_surface_area` position per the new signature) and update any host wrapper + header declaration.

- [ ] **Step 3: Rebuild + purge, run:**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation.TestPipe1DMassConservation.test_outfall_pipe_inflow_reaches_surface \
    tests.test_swe2d_gpu_drainage_network tests.test_drainage_inlet_outfall_vs_swmm
```
Expected: PASS.

- [ ] **Step 4: Cross-review, then commit**

```bash
git add cpp/src/swe2d_gpu.cu cpp/src/swe2d_gpu.cuh
git commit -m "fix(drainage): route pipe inflow at 2D-coupled outfalls to surface (audit F10)"
```

---

## Phase D — P2: Code Defects

### Task 11: F12 — elliptical `A_open` table formula + readback binding

{"action": "fix c++ host elliptical a_open table formula and add python readback binding", "type": "coding", "phase": "D"}
Agent: cpp-pro / kimi-for-coding/k3. Cross-review: debugger.

**Files:**
- Modify: `cpp/src/pipe1d.cu:559-577` (`pipe1d_compute_pipe_end_A_open_table` host lambda)
- Modify: `cpp/src/swe2d_bindings.cpp` (new readback binding), `cpp/src/pipe1d.cuh` (decl)

- [ ] **Step 1: Replace the inverted lower-half formula** with the single-branch ellipse segment formula (verify it matches device `xsect_getAofY_elliptical` — grep it and mirror exactly):

```cpp
        // Elliptical: filled area below depth y (0..2b), a = W/2, b = H/2.
        // A(y) = a·b·(acos(t) − t·sqrt(1−t²)), t = 1 − y/b.
        // A(0)=0, A(b)=πab/2, A(2b)=πab. Mirrors xsect_getAofY_elliptical.
        const double a = 0.5 * width;
        const double b = 0.5 * height;
        const double y_clamped = fmax(0.0, fmin(y, 2.0 * b));
        const double t = fmax(-1.0, fmin(1.0, 1.0 - y_clamped / b));
        return a * b * (acos(t) - t * sqrt(fmax(0.0, 1.0 - t * t)));
```

- [ ] **Step 2: Add a debug readback binding** `swe2d_pipe1d_readback_pipe_end_A_open_table(dev_ptr) -> np.ndarray [n_pipe_ends * PIPE1D_TABLE_N]` — D2H copy of `dev->pipe1d.d_pipe_end_A_open_table` (guard null → empty array). Follow the existing readback binding pattern at `swe2d_bindings.cpp:2056-2110`.

- [ ] **Step 3: Enable `test_elliptical_a_open_table`** in `tests/test_pipe1d_mass_conservation.py`: build one elliptical pipe-end (shape_type=2, W=0.6, H=0.4 → a=0.3, b=0.2), upload, read the table; assert monotonic non-decreasing, `tbl[-1] ≈ π·a·b` (±1 %), midpoint ≈ `π·a·b/2` (±2 %), `tbl[0] < 0.05·π·a·b`.

- [ ] **Step 4: Rebuild + purge, run test; cross-review; commit**

```bash
git add cpp/src/pipe1d.cu cpp/src/pipe1d.cuh cpp/src/swe2d_bindings.cpp tests/test_pipe1d_mass_conservation.py
git commit -m "fix(pipe1d): elliptical A_open table formula + readback binding (audit F12)"
```

### Task 12: F13 — `max_cell_length` as double end-to-end

{"action": "change c++ and python max_cell_length from int32 to double to stop truncation", "type": "coding", "phase": "D"}
Agent: cpp-pro / kimi-for-coding/k3 (C++ binding) — the Python one-liners may be folded into the same subagent task. Cross-review: debugger.

**Files:**
- Modify: `cpp/src/pipe1d.cuh:264` (`int32_t max_cell_length` → `double max_cell_length`)
- Modify: `cpp/src/pipe1d.cu:603` (definition)
- Modify: `cpp/src/swe2d_bindings.cpp:1772` (lambda arg type)
- Modify: `swe2d/runtime/coupling.py:2003` (`int(dsoa.max_cell_length)` → `float(dsoa.max_cell_length)`)
- Modify: `tests/pipe1d_runner.py:29` (`max_cell_length: int = 25` → `float = 25.0`)

- [ ] **Step 1: Apply the type changes.** The subdivision logic (`pipe1d.cu:625-631`) already computes `ceil(L / (double)max_cell_length)` — only signatures change.

- [ ] **Step 2: Add `swe2d_pipe1d_get_cell_count(dev_ptr) -> int`** binding returning `dev->pipe1d.n_pipe_cells` (needed by `test_fractional_max_cell_length`).

- [ ] **Step 3: Rebuild + purge, run:**

```bash
mamba run -n qgis_stable python3 -m unittest -v \
    tests.test_pipe1d_mass_conservation.TestPipe1DMassConservation.test_fractional_max_cell_length \
    tests.test_swe2d_pipe1d tests.test_pipe1d_vs_swmm tests.test_workbench_imports
```

- [ ] **Step 4: Cross-review, then commit**

```bash
git add cpp/src/pipe1d.cuh cpp/src/pipe1d.cu cpp/src/swe2d_bindings.cpp \
    swe2d/runtime/coupling.py tests/pipe1d_runner.py
git commit -m "fix(pipe1d): max_cell_length as double end-to-end (audit F13)"
```

### Task 13: F15 — smaller items + memory-safety verification

{"action": "remove drainage delta clamp, persist cuda work buffers, validate with compute-sanitizer", "type": "debugging", "phase": "D"}
Agent: debugger / kimi-for-coding/k3. Cross-review: cpp-pro / kimi-for-coding/k3.

**Files:**
- Modify: `cpp/src/swe2d_gpu.cu:4831-4834` (apply_delta clamp)
- Modify: `cpp/src/pipe1d.cu:2562-2568, 2809-2811` (per-step cudaMalloc/Free), `cpp/src/pipe1d.cuh` (persistent buffers)

- [ ] **Step 1: Remove the backwater-destroying clamp** — delete `if (node_max_depth) d = fmin(d, node_max_depth[i]);` at `swe2d_gpu.cu:4833` (keep the floor at 0). Rationale: the pipe1d mass-balance kernel deliberately does not clamp (`pipe1d.cu:2283-2291`); surcharge volume must be conserved.

- [ ] **Step 2: Persistent step work buffers (perf, audit F15).** Move `d_flux_Q`, `d_A_new`, `d_Q_new` into `Pipe1DDeviceState` as `d_step_flux_q/d_step_a_new/d_step_q_new` with an `n_step_capacity`; lazy-allocate/reallocate in `swe2d_pipe1d_step` following the existing pattern at `pipe1d.cu:2699-2708`; free in the state destructor (`pipe1d.cuh:184` `_P_FREE` block). Remove the per-call `cudaMalloc/cudaFree`.

- [ ] **Step 3: Memory-safety validation (F11 regression proof).** Run compute-sanitizer on the closed-system probe:

```bash
mamba run -n qgis_stable compute-sanitizer --tool memcheck \
    python3 /tmp/opencode/probe_mass_conservation.py 2>&1 | tail -20
```
(Or `tools/run_compute_sanitizer.py` if it wraps this.) Expected: `ERROR SUMMARY: 0 errors`. Any OOB at `pipe1d.cu` → fix before proceeding.

- [ ] **Step 4: Rebuild + purge, run the FULL Phase gate** (all suites from the header). Expected: PASS.

- [ ] **Step 5: Cross-review, then commit**

```bash
git add cpp/src/swe2d_gpu.cu cpp/src/pipe1d.cu cpp/src/pipe1d.cuh
git commit -m "fix(pipe1d): remove delta clamp, persist step buffers; sanitizer clean (audit F15)"
```

---

## Phase E — Validation & Documentation

### Task 14: Full validation, SWMM harness, audit resolution doc

{"action": "validate full pipe1d suite and write docs resolution report", "type": "docs", "phase": "E"}
Agent: python-pro / mimo-v2.5 (docs) with the validation runs executed first. Cross-review: debugger.

- [ ] **Step 1: Full gate** — every suite in the header command block, plus:

```bash
mamba run -n qgis_stable python3 -m unittest -v tests.test_batch_runner_orchestrator tests.test_cli
```

- [ ] **Step 2: SWMM validation harness** — run `tests/swmm_validation/` per its README/compare.py; confirm reports stay within `tolerances.py`. If a tolerance legitimately shifts because the old behavior was the defect, document the before/after in the resolution doc and get cross-review sign-off before adjusting any tolerance.

- [ ] **Step 3: Mass-conservation evidence** — re-run the four probes (`/tmp/opencode/probe*.py`) and capture before/after numbers (audit baseline: diffusion −90.9 %, fully_dynamic −73.3 % over 200 steps; target: |drift| < 0.01 %).

- [ ] **Step 4: Write `docs/PIPE1D_AUDIT_RESOLUTION_2026-07-17.md`** — table of F1–F15 → status (fixed/test name/commit hash), the probe evidence table, known limitations (outfall backflow from fixed-head BCs untreated; `volume_decomposition` still hardcoded at `pipe1d.cu:2592`; host wrappers still launch on the default stream while surrounding work uses `dev->d_stream` — currently safe but serialising; boundary-face flux formula remains a relaxation law — now mass-consistent by construction). Append a session entry to `docs/AGENT_SESSION_RECOVERY_LOG.md` per `.opencode/rules/SESSION_DOCUMENTATION.md`.

- [ ] **Step 5: Commit**

```bash
git add docs/PIPE1D_AUDIT_RESOLUTION_2026-07-17.md docs/AGENT_SESSION_RECOVERY_LOG.md
git commit -m "docs: pipe1d audit resolution report (2026-07-17)"
```

---

## Machine-Readable Plan (auto_agent_selector §8)

```json
{
  "plan": "2026-07-17-pipe1d-mass-conservation-fixes",
  "steps": [
    {"action": "write failing gpu validate tests for pipe1d mass conservation", "type": "test", "phase": "A", "agent": "test-automator", "model": "kimi-for-coding/kimi-for-coding-highspeed"},
    {"action": "rewrite c++ cuda flux kernel and node mass balance so nodes consume boundary-face fluxes", "type": "coding", "phase": "B", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "guard c++ cuda boundary flux branch against cell_y zero on first substep", "type": "coding", "phase": "B", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "remove c++ cuda inlet flux overrides so inlet nodes are debited by boundary flux", "type": "coding", "phase": "B", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "fix c++ cuda fully-dynamic kernel to anchor continuity on A_orig outside picard loop", "type": "coding", "phase": "B", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "fix c++ cuda wave speed to hydraulic-depth celerity sqrt(g*A/T)", "type": "coding", "phase": "C", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "fix c++ cuda regime override to fire only downhill open-channel and use A/T froude depth", "type": "coding", "phase": "C", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "fix c++ cuda pipe-end node head to convert cell depth to node datum", "type": "coding", "phase": "C", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "rework c++ cuda junction overflow path to conserve mass and use single fold site", "type": "coding", "phase": "C", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "route c++ cuda outfall pipe inflow to coupled surface cell via node_net_q", "type": "coding", "phase": "C", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "fix c++ host elliptical a_open table formula and add python readback binding", "type": "coding", "phase": "D", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "change c++ and python max_cell_length from int32 to double to stop truncation", "type": "coding", "phase": "D", "agent": "cpp-pro", "model": "kimi-for-coding/k3"},
    {"action": "remove drainage delta clamp, persist cuda work buffers, validate with compute-sanitizer", "type": "debugging", "phase": "D", "agent": "debugger", "model": "kimi-for-coding/k3"},
    {"action": "validate full pipe1d suite and write docs resolution report", "type": "docs", "phase": "E", "agent": "python-pro", "model": "commandcode/mimo-v2.5"}
  ]
}
```

## Risks & Notes

- **Intermediate states are intentionally inconsistent:** between Task 2 and Tasks 3–6 some tests stay red by design (TDD). Only commit per-task as instructed; the branch is green at each phase gate.
- **Pre-existing tests may encode defect behavior.** If `tests/test_swe2d_pipe1d.py` or the SWMM harness fails after a fix, the default assumption is the test encoded the defect — the cross-review subagent must confirm before any test edit.
- **Boundary-face flux remains a relaxation law** (`F = ΔH·c·A·dir`). After Task 2, conservation no longer depends on `F` matching `cell_Q` (both sides see `F`), and Task 6 gives `c` physical magnitude. A full characteristic-BC redesign is out of scope (YAGNI) and noted in the resolution doc.
- **Within-launch atomic race in the supply cap** (Task 2): two boundary faces of the same node in one launch may both pass the cap check. Bounded by one face's flux per substep; documented in code. Exact for the common single-link-per-node case.
- **`swe2d_pipe1d_step` signature is unchanged** — no Python call-site churn except Task 12's `float()` cast.
- **Commit cadence:** commits happen only during plan execution as listed per task; confirm with the user before the first commit if execution was not explicitly approved.
