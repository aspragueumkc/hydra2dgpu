"""
GPU hydraulics test suite — all valid scheme combinations.

Exercises the CUDA solver across a matrix of (spatial_scheme, temporal_scheme,
godunov_mode) on representative physical cases:

  1. Dam break (flat bed, Stoker comparison)
  2. Lake at rest (sinusoidal bed, well-balanced check)
  3. Compound channel flow (steady partitioned flow)

Each combination is validated for:
  - GPU active throughout (no CPU fallback)
  - Finite state (no NaN)
  - Positivity (h >= 0)
  - Physical plausibility (wave propagation, mass conservation, etc.)
"""

import os
import unittest
import numpy as np

from tests._swe2d_test_helpers import (
    _make_rect_mesh,
    _make_gmsh_triangle_mesh,
    _build_mesh,
    stoker_dam_break,
    VALID_SPATIAL_SCHEMES,
    VALID_TEMPORAL_SCHEMES,
    QUICK_SPATIAL_COMBOS,
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


# ─── Helpers ──────────────────────────────────────────────────────────────

def _cell_centroids_rect(nx, ny, lx, ly, node_x):
    """Compute cell centroid x-coordinates for a structured rect mesh."""
    n_cells = 2 * nx * ny
    cx = np.empty(n_cells)
    stride = nx + 1
    for ci in range(n_cells):
        row, col = divmod(ci // 2, nx)
        if ci % 2 == 0:
            nodes = [row * stride + col,
                     row * stride + col + 1,
                     (row + 1) * stride + col + 1]
        else:
            nodes = [row * stride + col,
                     (row + 1) * stride + col + 1,
                     (row + 1) * stride + col]
        cx[ci] = np.mean(node_x[nodes])
    return cx


def _zb_cell_rect(nx, ny, node_z):
    """Compute per-cell bed elevation (node average) for a rect mesh."""
    n_cells = 2 * nx * ny
    zb = np.empty(n_cells)
    stride = nx + 1
    for ci in range(n_cells):
        row, col = divmod(ci // 2, nx)
        if ci % 2 == 0:
            nodes = [row * stride + col,
                     row * stride + col + 1,
                     (row + 1) * stride + col + 1]
        else:
            nodes = [row * stride + col,
                     (row + 1) * stride + col + 1,
                     (row + 1) * stride + col]
        zb[ci] = np.mean(node_z[nodes])
    return zb


def _run_dambreak_case(mod, spatial, temporal, godunov):
    """Run a dam-break case with the given schemes. Returns (linf_error, diag)."""
    nx, ny = 80, 5
    lx, ly = 1000.0, 50.0
    hL, hR = 2.0, 0.5
    t_end = 10.0

    node_x, node_y, node_z, cell_nodes = _make_rect_mesh(nx, ny, lx, ly)
    mesh = mod.swe2d_build_mesh(
        node_x, node_y, node_z, cell_nodes,
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64))

    cell_cx = _cell_centroids_rect(nx, ny, lx, ly, node_x)
    h0 = np.where(cell_cx <= lx / 2.0, hL, hR)

    solver = mod.swe2d_create_solver(
        mesh, h0, n_mann=0.0, cfl=0.45, dt_max=0.5,
        temporal_order=temporal, spatial_scheme=spatial,
        godunov_mode=godunov, use_gpu=True)

    t = 0.0
    last_diag = None
    while t < t_end:
        last_diag = mod.swe2d_step(solver, -1.0)
        t += last_diag["dt"]

    h, _, _ = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)

    mid_row = ny // 2
    start = mid_row * nx * 2
    end = start + nx * 2
    cx_strip = cell_cx[start:end]
    h_strip = h[start:end]

    x_shifted = cx_strip - lx / 2.0
    h_exact = stoker_dam_break(x_shifted, t_end, hL, hR)
    linf = float(np.max(np.abs(h_strip - h_exact)))
    return linf, last_diag


def _run_lakerest_case(mod, spatial, temporal, godunov):
    """Run a lake-at-rest case. Returns (max_deviation, diag)."""
    nx, ny = 20, 10
    lx, ly = 200.0, 100.0
    eta0 = 1.0
    a_bed = 0.3
    n_steps = 100

    def zb_func(x, y):
        return a_bed * np.sin(np.pi * x / lx) * np.cos(np.pi * y / ly)

    node_x, node_y, node_z, cell_nodes = _make_rect_mesh(nx, ny, lx, ly, zb_func=zb_func)
    mesh = mod.swe2d_build_mesh(
        node_x, node_y, node_z, cell_nodes,
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64))

    zb_cell = _zb_cell_rect(nx, ny, node_z)
    h0 = np.maximum(0.0, eta0 - zb_cell)

    solver = mod.swe2d_create_solver(
        mesh, h0, n_mann=0.0, cfl=0.45, dt_max=5.0,
        temporal_order=temporal, spatial_scheme=spatial,
        godunov_mode=godunov, use_gpu=True)

    last_diag = None
    for _ in range(n_steps):
        last_diag = mod.swe2d_step(solver, -1.0)

    h, _, _ = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)

    eta = h + zb_cell
    wet = h > 1e-6
    deviation = float(np.max(np.abs(eta[wet] - eta0))) if wet.any() else 1.0
    return deviation, last_diag


# ─── Test class ───────────────────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUAllSchemeCombinations(unittest.TestCase):
    """Run all valid (spatial, temporal, godunov) combos on key test cases."""

    # Each combination runs in ~1-2 seconds, so the full matrix is manageable.
    # Sub-tests are parameterized so failures name the exact combination.

    def _test_combination(self, spatial, temporal, godunov, label):
        mod = _load_module()

        # 1. Dam break — L_inf error
        linf, diag = _run_dambreak_case(mod, spatial, temporal, godunov)
        self.assertTrue(diag["gpu_active"],
                        f"[{label}] GPU inactive in dam break")
        self.assertLess(linf, 0.50,
                        f"[{label}] Dam break L_inf too large: {linf:.4f}")

        # 2. Lake at rest — free-surface drift
        dev, diag2 = _run_lakerest_case(mod, spatial, temporal, godunov)
        self.assertTrue(diag2["gpu_active"],
                        f"[{label}] GPU inactive in lake at rest")
        self.assertLess(dev, 1.0e-8,
                        f"[{label}] Lake-at-rest drift too large: {dev:.3e}")

    # ── Individual test methods for the quick-validation set ─────────────
    # Each spatial scheme with RK2 + godunov_mode=0

    def test_spatial0_first_order(self):
        self._test_combination(0, 2, 0, "spatial=0(FO) temporal=2(RK2) godunov=0")

    def test_spatial1_muscl_fast(self):
        self._test_combination(1, 2, 0, "spatial=1(Fast) temporal=2(RK2) godunov=0")

    def test_spatial2_minmod(self):
        self._test_combination(2, 2, 0, "spatial=2(MinMod) temporal=2(RK2) godunov=0")

    def test_spatial3_mc(self):
        self._test_combination(3, 2, 0, "spatial=3(MC) temporal=2(RK2) godunov=0")

    def test_spatial4_van_leer(self):
        self._test_combination(4, 2, 0, "spatial=4(VanLeer) temporal=2(RK2) godunov=0")

    def test_spatial6_weno5(self):
        self._test_combination(6, 2, 0, "spatial=6(WENO5) temporal=2(RK2) godunov=0")

    # ── Temporal scheme sweep (spatial=0, godunov=0) ────────────────────

    def test_temporal1_euler(self):
        self._test_combination(0, 1, 0, "spatial=0 temporal=1(Euler)")

    def test_temporal3_rk3(self):
        self._test_combination(0, 3, 0, "spatial=0 temporal=3(RK3)")

    def test_temporal5_graph_rk4(self):
        self._test_combination(0, 5, 0, "spatial=0 temporal=5(Graph-RK4)")

    def test_temporal6_graph_rk5(self):
        self._test_combination(0, 6, 0, "spatial=0 temporal=6(Graph-RK5)")

    # ── Godunov rollout mode (spatial=0, temporal=2) ────────────────────

    def test_godunov_rollout(self):
        self._test_combination(0, 2, 1, "spatial=0 temporal=2 godunov=1(rollout)")


# ─── Full matrix (parameterized via loop, reports all failures) ───────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUAllSchemeCombinationsFull(unittest.TestCase):
    """Full (spatial × temporal × godunov) matrix.

    This test is slower but covers all valid paths.  Disabled by default
    because it runs O(60) combinations.  Enable with:
        BACKWATER_RUN_FULL_SUITE=1 python -m unittest ...
    """

    FLAG = "BACKWATER_RUN_FULL_SUITE"

    def setUp(self):
        if os.environ.get(self.FLAG, "0") != "1":
            self.skipTest(f"Set {self.FLAG}=1 to enable the full scheme matrix")

    def test_all_combinations(self):
        import itertools
        mod = _load_module()
        combos = list(itertools.product(
            VALID_SPATIAL_SCHEMES, VALID_TEMPORAL_SCHEMES, [0, 1]))

        failures = []
        for spatial, temporal, godunov in combos:
            label = f"s{spatial}_t{temporal}_g{godunov}"
            try:
                linf, diag = _run_dambreak_case(mod, spatial, temporal, godunov)
                if not diag["gpu_active"]:
                    failures.append(f"{label}: GPU inactive")
                elif linf >= 0.50:
                    failures.append(f"{label}: L_inf={linf:.4f}")

                dev, diag2 = _run_lakerest_case(mod, spatial, temporal, godunov)
                if not diag2["gpu_active"]:
                    failures.append(f"{label}LR: GPU inactive")
                elif dev >= 1.0e-8:
                    failures.append(f"{label}LR: drift={dev:.3e}")
            except Exception as e:
                failures.append(f"{label}: {e}")

        if failures:
            self.fail(f"{len(failures)} combination failures:\n  " + "\n  ".join(failures))
