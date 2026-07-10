"""
GPU numerical-only validation — deep water wave propagation.

Reference: reference/anuga_validation_tests/analytical_exact/deep_wave/

No analytical solution used. Tests stable wave propagation from a Gaussian
hump initial condition in a deep, flat-bottomed channel.

Physical setup
--------------
10 000 m × 500 m channel on a flat bed at z = −100 m.
Initial condition: Gaussian hump at the centre of the domain:
    h0 = 100.0 + 0.5 · exp(−(x − 5000)² / 500²)
      ≈ 100 m background depth with a 0.5 m perturbation.

Because the bed is at z = −100 m, the stage is approximately −99 m at the
hump crest. The perturbation splits into left- and right-propagating waves
that exit through the open boundaries.

Boundary conditions: OPEN (transmissive) on left and right, WALL on top/bottom.
T_END = 100.0 s — enough time for the wave to propagate away from the source.

Checks
------
1. GPU active, all depths finite and non-negative (stability).
2. The standard deviation of depth changes from its initial value, indicating
   wave propagation.
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_rect_mesh


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


def _channel_bc_arrays(nx, ny, left_type, left_val, right_type, right_val):
    """BC arrays for a channel: custom left/right, walls top/bottom."""
    stride = nx + 1
    n0, n1, tp, vl = [], [], [], []
    for j in range(ny):
        n0.append(j * stride)
        n1.append((j + 1) * stride)
        tp.append(left_type)
        vl.append(float(left_val))
    for j in range(ny):
        n0.append(j * stride + nx)
        n1.append((j + 1) * stride + nx)
        tp.append(right_type)
        vl.append(float(right_val))
    for i in range(nx):
        n0.append(i)
        n1.append(i + 1)
        tp.append(1)
        vl.append(0.0)
    top0 = ny * stride
    for i in range(nx):
        n0.append(top0 + i)
        n1.append(top0 + i + 1)
        tp.append(1)
        vl.append(0.0)
    return (
        np.array(n0, dtype=np.int32),
        np.array(n1, dtype=np.int32),
        np.array(tp, dtype=np.int32),
        np.array(vl, dtype=np.float64),
    )


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUDeepWave(unittest.TestCase):
    anuga_reference = (
        "reference/anuga_validation_tests/analytical_exact/deep_wave/"
    )
    NX = 200
    NY = 10
    LX = 10000.0
    LY = 500.0
    T_END = 100.0

    def _build(self, spatial_scheme: int = 0):
        mod = _load_module()

        # Flat bed at z = -100 m
        def bed_func(x, y):
            return np.full_like(x, -100.0)

        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY, zb_func=bed_func
        )

        # BC: left OPEN(4), right OPEN(4), walls top/bottom
        bc = _channel_bc_arrays(self.NX, self.NY, 4, 0.0, 4, 0.0)

        mesh = mod.swe2d_build_mesh(node_x, node_y, node_z, cell_nodes, *bc)
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]
        nx_p1 = self.NX + 1

        # Centroids (original order)
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

        # Gaussian hump initial condition on flat deep water
        background_depth = 100.0
        hump_amplitude = 0.5
        hump_centre = self.LX / 2.0  # 5000 m
        hump_width = 500.0
        h0 = background_depth + hump_amplitude * np.exp(
            -((cell_cx - hump_centre) ** 2) / (hump_width ** 2)
        )
        h0 = h0.astype(np.float64)

        cfl = 0.4 if spatial_scheme == 8 else 0.45
        solver = mod.swe2d_create_solver(
            mesh, h0, n_mann=0.0, cfl=cfl, dt_max=0.5, use_gpu=True, g=9.8,
            spatial_scheme=spatial_scheme,
        )

        perm = mod.swe2d_get_cell_perm(mesh)
        cx_p = cell_cx[perm]
        cy_p = cell_cy[perm]
        # Store initial h0 for comparison (permuted order)
        h0_p = h0[perm]
        return mod, mesh, solver, cx_p, cy_p, h0_p

    def _run_to_end(self, spatial_scheme: int = 0):
        mod, mesh, solver, cx_p, cy_p, h0_p = self._build(spatial_scheme)
        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = mod.swe2d_step(solver, -1.0)
            t += last_diag["dt"]
        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        return h, cx_p, cy_p, h0_p, last_diag

    def test_stability(self):
        """GPU active, all depths finite and non-negative."""
        h, _, _, _, last_diag = self._run_to_end()
        self.assertTrue(last_diag["gpu_active"])
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertTrue(np.all(h >= -1e-12))

    def test_wave_propagates(self):
        """Depth std changes from initial (wave energy redistributes)."""
        h, _, _, h0_p, _ = self._run_to_end()
        initial_std = float(np.std(h0_p))
        final_std = float(np.std(h))
        # The wave spreads out, so std should decrease from the initial hump
        # (exact change depends on numerics, but it must differ measurably)
        self.assertNotAlmostEqual(
            initial_std, final_std, delta=1e-6,
            msg=f"Depth std unchanged: initial={initial_std:.6f}, final={final_std:.6f}"
        )

    def test_new_schemes_stability(self):
        """Sweep schemes 5, 6, 8 — must remain stable (no NaN, no negative depth)."""
        for scheme, name in [(5, "Barth-Jespersen"), (6, "WENO3"), (8, "MP5")]:
            h, _, _, _, last_diag = self._run_to_end(spatial_scheme=scheme)
            self.assertTrue(last_diag["gpu_active"], f"GPU inactive for {name}")
            self.assertTrue(np.all(np.isfinite(h)), f"NaN/Inf depth for {name}")
            self.assertTrue(np.all(h >= -1e-10), f"Negative depth for {name}: min={h.min():.4e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
