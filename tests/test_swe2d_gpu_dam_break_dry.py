"""
GPU 1D dam-break validation (dry bed).

Reference: reference/anuga_validation_tests/analytical_exact/dam_break_dry/
Original ANUGA setup: L=1000 m, dx=1 m, W=5 m, h_L=10 m, h_R=0 m (dry).

Physical setup
--------------
Initial stage discontinuity at x=0: depth 10 m on the left, dry bed (0 m) on
the right. Bed flat at z=0. Transmissive (OPEN) left/right boundaries allow
the wave to pass; the wave front is ~60 m from the dam at t=2 s and does not
reach the domain boundaries (500 m away).

Test strategy
--------------
Run to t=2 s. Compare SWE2D GPU solution against the ANUGA analytical
dry-dam-break (Ritter) solution. The GPU solver applies an RCMK cell
permutation, so we use ``swe2d_get_cell_perm`` to align coordinates with
the returned state.

Tolerance: L1 error < 5 % of H_L (0.50 m). This is a classic dry-bed
dam-break; the shock wave / wet-dry front is captured by first-order FVM
with no special wetting-drying treatment, so a 5 % tolerance on the
10 m left-side depth is appropriate.
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh
from tests._anuga_importer import import_anuga_module


_analytical = import_anuga_module(
    "reference/anuga_validation_tests/analytical_exact/dam_break_dry/"
    "analytical_dam_break_dry.py"
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


def _channel_bc_open_sides(nx: int, ny: int):
    """Build BC arrays: left/right OPEN (4), top/bottom WALL (1)."""
    stride = nx + 1
    n0, n1, tp, val = [], [], [], []
    # Left boundary — OPEN (transmissive)
    for j in range(ny):
        n0.append(j * stride)
        n1.append((j + 1) * stride)
        tp.append(4)
        val.append(0.0)
    # Right boundary — OPEN (transmissive)
    for j in range(ny):
        n0.append(j * stride + nx)
        n1.append((j + 1) * stride + nx)
        tp.append(4)
        val.append(0.0)
    # Bottom — WALL
    for i in range(nx):
        n0.append(i)
        n1.append(i + 1)
        tp.append(1)
        val.append(0.0)
    # Top — WALL
    top0 = ny * stride
    for i in range(nx):
        n0.append(top0 + i)
        n1.append(top0 + i + 1)
        tp.append(1)
        val.append(0.0)
    return (
        np.array(n0, dtype=np.int32),
        np.array(n1, dtype=np.int32),
        np.array(tp, dtype=np.int32),
        np.array(val, dtype=np.float64),
    )


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUDamBreakDry(unittest.TestCase):
    anuga_reference = "reference/anuga_validation_tests/analytical_exact/dam_break_dry/"
    NX = 1000
    NY = 5
    LX = 1000.0
    LY = 5.0
    H_L = 10.0
    H_R = 0.0
    T_END = 2.0

    def _build(self, spatial_scheme: int = 0):
        mod = _load_module()
        node_x, node_y, _, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY
        )
        node_x = node_x - self.LX / 2.0  # centre dam at x = 0

        # Boundary conditions: OPEN left/right, WALL top/bottom
        bc_n0, bc_n1, bc_type, bc_val = _channel_bc_open_sides(self.NX, self.NY)

        mesh = mod.swe2d_build_mesh(
            node_x,
            node_y,
            np.zeros_like(node_x),
            cell_nodes,
            bc_n0,
            bc_n1,
            bc_type,
            bc_val,
        )
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]
        nx_p1 = self.NX + 1

        # Cell centroids in original (unpermuted) order via divmod
        cell_cx = np.empty(n_cells)
        cell_cy = np.empty(n_cells)
        for ci in range(n_cells):
            row, col = divmod(ci // 2, self.NX)
            if ci % 2 == 0:
                n = [row * nx_p1 + col, row * nx_p1 + col + 1, (row + 1) * nx_p1 + col + 1]
            else:
                n = [row * nx_p1 + col, (row + 1) * nx_p1 + col + 1, (row + 1) * nx_p1 + col]
            cell_cx[ci] = float(np.mean(node_x[n]))
            cell_cy[ci] = float(np.mean(node_y[n]))

        # Initial condition — piecewise with dam at x = 0
        h0 = np.where(cell_cx < 0.0, self.H_L, self.H_R).astype(np.float64)

        cfl = 0.4 if spatial_scheme == 8 else 0.45
        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.0, cfl=cfl, dt_max=0.5, use_gpu=True, g=9.8,
            spatial_scheme=spatial_scheme,
        )

        # Permutation for aligning GPU state with centroids
        perm = mod.swe2d_get_cell_perm(mesh)
        cx_p = cell_cx[perm]
        cy_p = cell_cy[perm]

        return mod, mesh, solver, cx_p, cy_p

    def _run_to_end(self, spatial_scheme: int = 0):
        mod, mesh, solver, cx_p, cy_p = self._build(spatial_scheme)
        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = mod.swe2d_step(solver, -1.0)
            t += last_diag["dt"]
        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        return h, cx_p, cy_p, last_diag

    def test_stability(self):
        h, _, _, last_diag = self._run_to_end()
        self.assertTrue(last_diag["gpu_active"])
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(h >= -1e-12))

    def test_l1_error_vs_anuga(self):
        h, cx_p, cy_p, _ = self._run_to_end()
        # Strip along centreline (middle y rows)
        strip_tol = self.LY * 0.15
        mask = np.abs(cy_p - self.LY / 2.0) < strip_tol
        order = np.argsort(cx_p[mask])
        cx_strip = cx_p[mask][order]
        h_strip = h[mask][order]
        # ANUGA analytical: h0=0 (dry right), h1=10 (wet left)
        h_exact, _ = _analytical.vec_dam_break(
            cx_strip, self.T_END, h0=self.H_R, h1=self.H_L
        )
        l1 = float(np.mean(np.abs(h_strip - h_exact)))
        limit = 0.05 * self.H_L
        self.assertLess(
            l1,
            limit,
            msg=f"GPU dry dam-break L1 error {l1:.6f} m exceeds limit ({limit:.4f} m)",
        )

    def test_new_schemes_stability(self):
        """Sweep schemes 5, 6, 8 — must remain stable (no NaN, no negative depth)."""
        for scheme, name in [(5, "Barth-Jespersen"), (6, "WENO3"), (8, "MP5")]:
            h, _, _, last_diag = self._run_to_end(spatial_scheme=scheme)
            self.assertTrue(last_diag["gpu_active"], f"GPU inactive for {name}")
            self.assertTrue(np.all(np.isfinite(h)), f"NaN/Inf depth for {name}")
            self.assertTrue(np.all(h >= -1e-10), f"Negative depth for {name}: min={h.min():.4e}")
