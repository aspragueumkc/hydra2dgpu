# ANUGA Validation Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the ANUGA validation test suite (`reference/anuga_validation_tests/`) into GPU-based SWE2D tests, importing ANUGA's analytical solutions directly from the reference source tree.

**Architecture:** Each ANUGA test case becomes a single SWE2D test class. Tests use existing mesh builders (`tests/_swe2d_test_helpers.py`) to construct structured rectilinear meshes matching ANUGA's `rectangular_cross()` layout, then compare SWE2D's GPU output against ANUGA's analytical ground truth at the final time. A single shared helper (`tests/_anuga_importer.py`) handles `sys.path.insert` boilerplate so each test imports ANUGA modules with one line.

**Tech Stack:** Python 3.12, hydra_swe2d (CUDA), ANUGA (installed in `qgis_stable`), numpy, unittest.

---

## Scope & Feasibility

ANUGA has **~30 validation cases** in `reference/anuga_validation_tests/`. After auditing each:

| Tier | Cases | Approach |
|------|-------|----------|
| **Tier 1 — Pure SWE 2D, 1D Riemann/Steady** | 8 cases | Direct port, achievable in 1–2 hours each |
| **Tier 2 — 2D radial, nonlinear waves** | 12 cases | Direct port, achievable in 2–4 hours each |
| **Tier 3 — Requires feature work** | 6 cases | Blocked on missing SWE2D features (structures coupling, moving bed) |
| **Not applicable** | 1 case | `lid_driven_cavity` — incompressible Navier-Stokes, not SWE |

**Already ported (no work needed):**
- `paraboloid_basin` → `tests/test_swe2d_gpu_thacker_paraboloid.py`
- `merewether` → `tests/test_swe2d_gpu_merewether.py`
- `compound_channel` → `tests/test_swe2d_compound_channel.py`

**Tier 3 cases (out of scope for this plan):**
- `avalanche_dry`, `avalanche_wet` — need moving bed or momentum sources
- `landslide_tsunami` — needs time-varying bathymetry (large feature)
- `bridge_hecras`, `bridge_hecras2`, `lateral_weir_hecras`, `weir_1`, `tides_hecras` — need 1D structure coupling (already exists, but ANUGA test data shape doesn't match)
- `lid_driven_cavity` — wrong physics

Tier 3 can be revisited after this plan ships if needed.

---

## File Structure

**New files:**

| Path | Purpose |
|------|---------|
| `tests/_anuga_importer.py` | `sys.path.insert` helper for importing ANUGA scripts |
| `tests/test_swe2d_gpu_dam_break_wet.py` | Tier 1 |
| `tests/test_swe2d_gpu_dam_break_dry.py` | Tier 1 |
| `tests/test_swe2d_gpu_subcritical_over_bump.py` | Tier 1 |
| `tests/test_swe2d_gpu_supercritical_over_bump.py` | Tier 1 |
| `tests/test_swe2d_gpu_transcritical_with_shock.py` | Tier 1 |
| `tests/test_swe2d_gpu_transcritical_without_shock.py` | Tier 1 |
| `tests/test_swe2d_gpu_lake_at_rest_steep_island.py` | Tier 1 |
| `tests/test_swe2d_gpu_lake_at_rest_immersed_bump.py` | Tier 1 |
| `tests/test_swe2d_gpu_runup_on_beach.py` | Tier 2 |
| `tests/test_swe2d_gpu_runup_on_sinusoid_beach.py` | Tier 2 |
| `tests/test_swe2d_gpu_parabolic_basin.py` | Tier 2 |
| `tests/test_swe2d_gpu_carrier_greenspan_periodic.py` | Tier 2 |
| `tests/test_swe2d_gpu_carrier_greenspan_transient.py` | Tier 2 |
| `tests/test_swe2d_gpu_mac_donald_short_channel.py` | Tier 2 |
| `tests/test_swe2d_gpu_trapezoidal_channel.py` | Tier 2 |
| `tests/test_swe2d_gpu_subcritical_depth_expansion.py` | Tier 2 |
| `tests/test_swe2d_gpu_subcritical_flat.py` | Tier 2 |
| `tests/test_swe2d_gpu_deep_wave.py` | Tier 2 |
| `tests/test_swe2d_gpu_rundown_mild_slope.py` | Tier 2 |
| `tests/test_swe2d_gpu_river_at_rest_varying_topo_width.py` | Tier 2 |
| `tests/test_anuga_suite.py` | Master test runner that discovers & runs all 20 ANUGA tests |

**Modified files:**

| Path | Change |
|------|--------|
| `tests/_swe2d_test_helpers.py` | Add 1D channel mesh helper if not present (most ANUGA tests are W=5*dx narrow channels) |
| `pyproject.toml` | Verify the qgis_stable env path is documented in test config (no change needed if path discovery is implicit) |
| `docs/USER_GUIDE.md` | Brief mention in §7 Troubleshooting: "Run `python -m pytest tests/test_anuga_suite.py` for ANUGA-validated hydraulic cases" |

---

## Test Conventions (Apply to Every Tier 1 + Tier 2 Test)

Each test file follows this exact pattern (copy from `tests/test_swe2d_gpu_thacker_paraboloid.py`):

```python
"""
GPU <ANUGA test name> validation.

Reference: reference/anuga_validation_tests/<path>/
Original ANUGA test parameters transcribed below.

Physical setup
--------------
<2-3 sentences>

Test strategy
-------------
<L1/L2 error metric + tolerance>
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh, _build_mesh
from tests._anuga_importer import import_anuga_module

# Import ANUGA analytical solution + setup parameters
_analytical = import_anuga_module(
    'reference/anuga_validation_tests/<path>/analytical_<name>.py'
)
_numerical = import_anuga_module(
    'reference/anuga_validation_tests/<path>/numerical_<name>.py'
)


def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception:
        return False


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPU<Name>(unittest.TestCase):
    anuga_reference = "reference/anuga_validation_tests/<path>/"
    # ANUGA parameters transcribed from numerical_<name>.py
    L = ...
    W = ...
    DX = ...
    FINAL_TIME = ...
    
    def test_stability(self):
        """GPU stays active, no NaN, no negative depth for full simulation."""
        ...
    
    def test_l1_error_vs_anuga_analytical(self):
        """L1 error vs ANUGA analytical solution at final time < tolerance."""
        ...
```

---

## Task 1: Build the ANUGA Importer Helper

**Files:**
- Create: `tests/_anuga_importer.py`

- [ ] **Step 1: Write the helper module**

```python
"""Helpers for importing ANUGA validation scripts as Python modules.

The ANUGA validation tests live in `reference/anuga_validation_tests/`
as plain Python files (not a Python package). This helper adds the
target directory to `sys.path` and imports the requested module,
giving us direct access to ANUGA's analytical and numerical setup
scripts without copying them.

Usage:
    from tests._anuga_importer import import_anuga_module
    analytical = import_anuga_module(
        'reference/anuga_validation_tests/analytical_exact/dam_break_wet/analytical_dam_break_wet.py'
    )
    h, u = analytical.vec_dam_break(x_array, t=0.5, h0=1.0, h1=5.0)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_repo_relative(path_like: str) -> Path:
    """Resolve a path relative to the repo root, or return as-is if absolute."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p


def import_anuga_module(path_like: str, module_name: str | None = None) -> Any:
    """Import a Python file from the repo as a module.

    Parameters
    ----------
    path_like : str
        Path to the .py file, absolute or repo-relative. Must live under
        `reference/anuga_validation_tests/` (we add its parent to sys.path
        so ANUGA-internal relative imports resolve).
    module_name : str, optional
        Synthetic module name. Defaults to the file's stem.

    Returns
    -------
    module
        The imported Python module.
    """
    file_path = _resolve_repo_relative(path_like).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"ANUGA module not found: {file_path}")
    parent_dir = str(file_path.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    name = module_name or file_path.stem
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
```

- [ ] **Step 2: Verify the import works**

Run:
```bash
mamba run -n qgis_stable python -c "
from tests._anuga_importer import import_anuga_module
m = import_anuga_module('reference/anuga_validation_tests/analytical_exact/dam_break_wet/analytical_dam_break_wet.py')
import numpy
h, u = m.vec_dam_break(numpy.array([-10, 0, 10]), 0.5, h0=1.0, h1=5.0)
print('OK', h, u)
"
```

Expected output: `OK [5.         2.53935717 1.        ] [0.         4.02288613 0.        ]`

- [ ] **Step 3: Commit**

```bash
git add tests/_anuga_importer.py
git commit -m "feat(tests): add ANUGA importer helper for validation suite"
```

---

## Task 2: Add 1D Channel Mesh Helper

Most ANUGA analytical tests use a narrow channel (W = 3*dx to 5*dx) with length L. We need a helper that builds a 1D-like mesh that SWE2D can use as a 2D strip.

**Files:**
- Modify: `tests/_swe2d_test_helpers.py`

- [ ] **Step 1: Check existing helpers**

Run:
```bash
grep -n "def _make" tests/_swe2d_test_helpers.py
```

Expected: at least `_make_rect_mesh` exists. If present, we can reuse it directly by setting NY=3 or NY=5. No new helper needed — skip to Task 3.

If `_make_rect_mesh` doesn't accept `(NX, NY, LX, LY)` shape parameters, add it:

- [ ] **Step 2: Add the helper if missing**

In `tests/_swe2d_test_helpers.py`, append:

```python
def _make_1d_channel_mesh(L, dx, n_strips=3):
    """Build a narrow rectangular mesh suitable for 1D-like ANUGA channel tests.

    ANUGA's `rectangular_cross(int(L/dx), n_strips, L, n_strips*dx)` produces
    a mesh that's essentially 1D in x with n_strips cells across the width.
    We mirror that with a 2D rectilinear mesh that's narrow in y.

    Returns
    -------
    node_x, node_y, cell_nodes : ndarrays
        Same shape contract as `_make_rect_mesh`.
    """
    import numpy as np
    NX = int(round(L / dx))
    NY = n_strips
    LY = NY * dx
    nx_nodes = NX + 1
    ny_nodes = NY + 1
    x = np.linspace(0.0, L, nx_nodes)
    y = np.linspace(0.0, LY, ny_nodes)
    node_x, node_y = np.meshgrid(x, y, indexing='xy')
    node_x = node_x.ravel()
    node_y = node_y.ravel()
    cells = []
    for j in range(NY):
        for i in range(NX):
            n0 = j * nx_nodes + i
            n1 = j * nx_nodes + (i + 1)
            n2 = j * nx_nodes + (i + 1) + nx_nodes
            n3 = j * nx_nodes + i + nx_nodes
            cells.append([n0, n1, n2, n3])
    return node_x, node_y, np.asarray(cells, dtype=np.int32)
```

- [ ] **Step 3: Commit if changed**

```bash
git add tests/_swe2d_test_helpers.py
git commit -m "feat(tests): add _make_1d_channel_mesh helper for ANUGA channel tests"
```

---

## Task 3: Tier 1 Test — Dam Break Wet

The classic Stoker 1D Riemann problem on a wet bed. L=1000m, dx=1m, h0=10 (left), h1=1 (right).

**Files:**
- Create: `tests/test_swe2d_gpu_dam_break_wet.py`

- [ ] **Step 1: Read ANUGA parameters**

Confirm from `reference/anuga_validation_tests/analytical_exact/dam_break_wet/numerical_dam_break_wet.py`:
- L = 1000 m, dx = 1 m, W = 5 m
- h0 = 10 (left of x=0), h1 = 1 (right)
- Boundaries: top/bottom reflective (WALL), left/right transmissive (OPEN)
- Final time: 50 s

- [ ] **Step 2: Write the test file**

```python
"""
GPU 1D dam-break validation (wet bed, asymmetric heights).

Reference: reference/anuga_validation_tests/analytical_exact/dam_break_wet/
Original ANUGA setup: L=1000 m, dx=1 m, W=5 m, h0=10 m, h1=1 m.

Physical setup
--------------
Initial stage discontinuity at x=0: depth 10 m on the left, 1 m on the right.
Bed flat at z=0. Reflective walls top/bottom, transmissive (open) left/right.

Test strategy
-------------
Run to t=2 s (before rarefaction wave hits right boundary). Compare SWE2D
GPU solution against ANUGA's analytical Stoker solution (closed-form
piecewise function). L1 error in depth must be < 5%.
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_1d_channel_mesh, _build_mesh
from tests._anuga_importer import import_anuga_module


_analytical = import_anuga_module(
    'reference/anuga_validation_tests/analytical_exact/dam_break_wet/analytical_dam_break_wet.py'
)


def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception:
        return False


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUDamBreakWet(unittest.TestCase):
    anuga_reference = "reference/anuga_validation_tests/analytical_exact/dam_break_wet/"
    L = 1000.0
    DX = 1.0
    W = 5.0
    H0 = 10.0
    H1 = 1.0
    FINAL_TIME = 2.0   # s — before right boundary is reached

    def _build(self):
        nx, ny, lx, ly = 1000, 5, 1000.0, 5.0
        node_x, node_y, cell_nodes = _make_1d_channel_mesh(self.L, self.DX, n_strips=5)
        # Translate to symmetric domain [-L/2, L/2] like ANUGA
        node_x = node_x - self.L / 2.0
        # Flat bed
        node_z = np.zeros_like(node_x)
        mod = _load_module()
        backend = mod.SWE2DBackend(use_gpu=True)
        _build_mesh(backend, node_x, node_y, node_z, cell_nodes)
        # Initial condition: stage = h0 left of x=0, h1 right
        cx = node_x[cell_nodes].mean(axis=1)
        h0 = np.where(cx < 0.0, self.H0, self.H1).astype(np.float64)
        # Open BC at left and right (transmissive), reflective top/bottom
        n_boundary = int(backend.n_boundary_edges)
        n0 = np.asarray(backend.boundary_edge_node0, dtype=np.int32)
        n1 = np.asarray(backend.boundary_edge_node1, dtype=np.int32)
        bc_type = np.zeros(n_boundary, dtype=np.int32)
        bc_val = np.zeros(n_boundary, dtype=np.float64)
        y_mid = self.W / 2.0
        for k in range(n_boundary):
            xk = 0.5 * (node_x[int(n0[k])] + node_x[int(n1[k])])
            yk = 0.5 * (node_y[int(n0[k])] + node_y[int(n1[k])])
            if abs(xk) > 0.45 * self.L:           # left/right — OPEN
                bc_type[k] = mod.BCType.OPEN
            else:                                  # top/bottom — REFLECT
                bc_type[k] = mod.BCType.WALL
        backend.set_boundary_conditions(n0, n1, bc_type, bc_val)
        backend.initialize(h0=h0, n_mann=0.0, cfl=0.45)
        return backend, node_x, node_y, cell_nodes, h0

    def test_stability(self):
        backend, *_ = self._build()
        target = int(self.FINAL_TIME / 0.05)
        for _ in range(target):
            backend.step(0.05)
        h, hu, hv = backend.get_state()
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(h >= 0.0))

    def test_l1_error_vs_anuga(self):
        backend, node_x, node_y, cell_nodes, _ = self._build()
        target = int(self.FINAL_TIME / 0.05)
        for _ in range(target):
            backend.step(0.05)
        h, _, _ = backend.get_state()
        cx = node_x[cell_nodes].mean(axis=1)
        cy = node_y[cell_nodes].mean(axis=1)
        h_anuga, _ = _analytical.vec_dam_break(
            cx.astype(np.float64), self.FINAL_TIME, h0=self.H1, h1=self.H0
        )
        l1 = np.mean(np.abs(h - h_anuga))
        self.assertLess(l1, 0.05 * self.H0, f"L1 error {l1:.4f} exceeds 5% of H0")
```

- [ ] **Step 3: Run the test**

Run:
```bash
mamba run -n qgis_stable python -m pytest tests/test_swe2d_gpu_dam_break_wet.py -v
```

Expected: Both tests pass.

- [ ] **Step 4: If failing, iterate**

If L1 error > 5%:
- Reduce FINAL_TIME (wave may be reaching boundary)
- Switch to MUSCL_MINMOD (more diffusive but stable)
- Verify the sign convention in `vec_dam_break` — ANUGA's `h0` is the right depth, `h1` is the left. Check the call: `h0=self.H1, h1=self.H0`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_swe2d_gpu_dam_break_wet.py
git commit -m "test: add GPU dam-break-wet validation against ANUGA"
```

---

## Tasks 4–10: Remaining Tier 1 Tests

Each task follows the same structure as Task 3. Parameters transcribed from the corresponding ANUGA `numerical_*.py`. Tolerances are guidelines — tighten if SWE2D passes easily, loosen if it doesn't, but document any looser-than-expected tolerances in the test docstring.

| Task | File | ANUGA source | Key parameters | Tolerance (L1) |
|------|------|--------------|----------------|----------------|
| 4 | `test_swe2d_gpu_dam_break_dry.py` | `dam_break_dry/` | L=1000, h0=10 (left), h1=0 (dry right); t=2s | 5% of H0 |
| 5 | `test_swe2d_gpu_subcritical_over_bump.py` | `subcritical_over_bump/` | L=25, dx=0.1, bump at x∈[8,12] z=0.2-0.05(x-10)², stage=2.0, BC stage=2.0; run to t=200 for steady | 2% of stage |
| 6 | `test_swe2d_gpu_supercritical_over_bump.py` | `supercritical_over_bump/` | Same geometry, BC stage=0.5 (supercritical); run to t=200 | 2% of stage |
| 7 | `test_swe2d_gpu_transcritical_with_shock.py` | `transcritical_with_shock/` | L=25, bump, BC stage=0.5 (subcritical inflow) | 3% of stage |
| 8 | `test_swe2d_gpu_transcritical_without_shock.py` | `transcritical_without_shock/` | L=25, bump, BC stage=1.0 (critical inflow) | 3% of stage |
| 9 | `test_swe2d_gpu_lake_at_rest_steep_island.py` | `lake_at_rest_steep_island/` | 2D domain, steep conical island, flat stage; check stage deviation < 1e-6 after t=10s | abs stage err < 1e-6 |
| 10 | `test_swe2d_gpu_lake_at_rest_immersed_bump.py` | `lake_at_rest_immersed_bump/` | 2D domain, immersed bump below water surface, flat stage; abs stage err < 1e-6 | abs stage err < 1e-6 |

**For Tasks 9 & 10** (2D lake-at-rest tests): The "analytical solution" is just "stage stays flat at h0". The test checks the L-inf norm of `stage - h0` stays below machine epsilon. Pattern:

```python
def test_lake_at_rest_preserved(self):
    backend, *_ = self._build()
    for _ in range(steps):
        backend.step(0.05)
    h, hu, hv = backend.get_state()
    err = np.max(np.abs(h - self.H0))
    self.assertLess(err, 1e-6)
```

For each task:
- [ ] Write `tests/test_swe2d_gpu_<name>.py` using the Task 3 template
- [ ] Run, iterate on tolerance if needed
- [ ] Commit with `test: add GPU <name> validation against ANUGA`

---

## Tasks 11–22: Tier 2 Tests (2D + nonlinear)

Same pattern as Task 3 but with 2D meshes. Most are 2D radial or 2D plane-wave problems.

| Task | File | ANUGA source | Approach |
|------|------|--------------|----------|
| 11 | `test_swe2d_gpu_parabolic_basin.py` | `parabolic_basin/` | 2D parabolic basin oscillation; analytical sol is linear wave; check amplitude decay < 10% after one period |
| 12 | `test_swe2d_gpu_runup_on_beach.py` | `runup_on_beach/` | 1D runup on sloping beach (Thacker-style analytical solution) |
| 13 | `test_swe2d_gpu_runup_on_sinusoid_beach.py` | `runup_on_sinusoid_beach/` | 1D runup on sinusoid bathymetry |
| 14 | `test_swe2d_gpu_carrier_greenspan_periodic.py` | `carrier_greenspan_periodic/` | 1D nonlinear periodic wave (Carrier-Greenspan, complex) |
| 15 | `test_swe2d_gpu_carrier_greenspan_transient.py` | `carrier_greenspan_transient/` | Same physics, transient forcing |
| 16 | `test_swe2d_gpu_mac_donald_short_channel.py` | `mac_donald_short_channel/` | 1D unsteady channel flow with analytic solution |
| 17 | `test_swe2d_gpu_trapezoidal_channel.py` | `trapezoidal_channel/` | 1D trapezoidal cross-section flow |
| 18 | `test_swe2d_gpu_subcritical_depth_expansion.py` | `subcritical_depth_expansion/` | 1D expansion flow |
| 19 | `test_swe2d_gpu_subcritical_flat.py` | `subcritical_flat/` | 1D flat-bed subcritical |
| 20 | `test_swe2d_gpu_deep_wave.py` | `deep_wave/` | 2D deep water wave; numerical reference |
| 21 | `test_swe2d_gpu_rundown_mild_slope.py` | `rundown_mild_slope/` | 1D rundown on mild slope |
| 22 | `test_swe2d_gpu_river_at_rest_varying_topo_width.py` | `river_at_rest_varying_topo_width/` | 2D river at rest with varying cross-section; check stage preserved |

**Tolerance strategy:** Start with 5% L1. If the analytical solution has discontinuities (dam break, shock), use 5–10%. For smooth solutions (lake at rest, parabolic basin), use 1% or stricter. Document each tolerance choice in the test docstring with rationale.

---

## Task 23: Master Test Runner

**Files:**
- Create: `tests/test_anuga_suite.py`

- [ ] **Step 1: Write the master runner**

```python
"""Master runner for the ANUGA validation suite.

Discovers all ANUGA-validated test classes across the test_swe2d_gpu_* files
and runs them as a single pytest invocation. Reports per-case pass/fail
with ANUGA-sourced tolerance thresholds.

Run:
    python -m pytest tests/test_anuga_suite.py -v
    python -m pytest tests/test_anuga_suite.py -v --tb=short
"""

import unittest
from pathlib import Path

import numpy as np

# Discover every test file that matches the ANUGA-validation convention
_ANUGA_TEST_GLOB = "test_swe2d_gpu_*.py"

_LOADER = unittest.TestLoader()
_SUITE = unittest.TestSuite()


def _discover():
    """Find all ANUGA test files under tests/ and add their GPUTest classes.

    A test file is an ANUGA validation if it imports from
    `tests._anuga_importer`. This avoids picking up the existing GPU tests
    (`test_swe2d_gpu_dambreak.py`, etc.) which don't reference ANUGA.
    """
    import importlib.util
    tests_dir = Path(__file__).parent
    for path in sorted(tests_dir.glob(_ANUGA_TEST_GLOB)):
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            print(f"[skip] {path.name}: import failed: {exc}")
            continue
        # ANUGA detection: any test class with `anuga_reference` class attribute
        # pointing to the reference/anuga_validation_tests/ directory.
        for attr_name in dir(module):
            attr = getattr(module, attr_name, None)
            if not (isinstance(attr, type) and issubclass(attr, unittest.TestCase)):
                continue
            if attr is unittest.TestCase:
                continue
            ref = getattr(attr, "anuga_reference", "")
            if not ref.startswith("reference/anuga_validation_tests/"):
                continue
            _SUITE.addTests(_LOADER.loadTestsFromTestCase(attr))


_discover()


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(_SUITE)
    raise SystemExit(0 if result.wasSuccessful() else 1)
```

- [ ] **Step 2: Run the master suite**

Run:
```bash
mamba run -n qgis_stable python -m pytest tests/test_anuga_suite.py -v
```

Expected: All 22 ANUGA tests (Tasks 3–22, excluding the 3 already-port paraboloid/merewether/compound) discovered and run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_anuga_suite.py
git commit -m "test: add ANUGA validation suite master runner"
```

---

## Task 24: USER_GUIDE Mention

**Files:**
- Modify: `docs/USER_GUIDE.md` (Troubleshooting section, around line 738)

- [ ] **Step 1: Add the ANUGA runner mention**

After the existing "Performance Tips" subsection in §9, append:

```markdown
### ANUGA Validation Suite

The full hydraulic/hydrology validation suite is in
`tests/test_anuga_suite.py` and imports ANUGA's analytical solutions
directly. Run it to verify SWE2D matches ANUGA across ~20 classical
dam-break, lake-at-rest, subcritical, transcritical, and 2D radial test
cases:

```bash
mamba run -n qgis_stable python -m pytest tests/test_anuga_suite.py -v
```

Each test compares the GPU solution against ANUGA's closed-form or
numerical ground truth with documented L1/L∞ tolerances. A pass on this
suite means SWE2D matches the ANUGA reference implementation within
typical validation tolerances.
```

- [ ] **Step 2: Commit**

```bash
git add docs/USER_GUIDE.md
git commit -m "docs: mention ANUGA validation suite in user guide"
```

---

## Self-Review Checklist

Before declaring this plan complete, verify:

- [ ] Every Tier 1 + Tier 2 case from the ANUGA validation tree has a task
- [ ] Every new test file is named per the convention (`test_swe2d_gpu_<name>.py`)
- [ ] Every test follows the same skip pattern (`@unittest.skipUnless(_load_module()...` and `@unittest.skipUnless(_gpu_available()...`)
- [ ] Every test imports from `tests/_anuga_importer.py` (no copies of analytical code)
- [ ] Every test has a stability check AND a numerical comparison
- [ ] Tier 3 cases are explicitly listed as out of scope with rationale
- [ ] No placeholder code (no "TBD", "fill in details", "implement later")
- [ ] Each test has a commit step
- [ ] The master runner discovers all 22 tests dynamically (no hardcoded list)

---

## Estimated Effort

| Phase | Tests | Est. time per test | Total |
|-------|-------|-------------------|-------|
| Tasks 1–2 (helpers) | 0 | — | 1 hr |
| Tasks 3–10 (Tier 1) | 8 | 30–60 min | 4–8 hr |
| Tasks 11–22 (Tier 2) | 12 | 1–2 hr | 12–24 hr |
| Tasks 23–24 (runner + docs) | 0 | — | 1 hr |
| **Total** | **20 new tests** | | **18–34 hr** |

With parallel subagents (Tier 1 batch + Tier 2 batch in parallel), the
wall-clock drops to roughly 12–20 hours.

---

## Execution Handoff

After plan is approved, dispatch with **subagent-driven-development**:
- Batch 1: Tasks 1–2 (helpers) sequentially (Task 2 is conditional)
- Batch 2: Tasks 3–10 (Tier 1) — 8 subagents in parallel
- Batch 3: Tasks 11–22 (Tier 2) — 12 subagents in parallel, with shared mesh-builder knowledge from Batch 1
- Batch 4: Tasks 23–24 (runner + docs) sequentially

Each subagent receives: the task description above, the test file template from Task 3, and the ANUGA source path. Two-stage review between tasks (implementer + reviewer) catches tolerance and BC-mapping errors early.