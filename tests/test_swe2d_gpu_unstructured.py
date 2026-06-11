"""
GPU-only unstructured SWE2D validation (gmsh triangle meshes).

This suite exercises the CUDA solver on the same unstructured physical cases
used for CPU debugging so higher-order GPU regressions are caught directly.
"""

import os
import sys
import unittest
import numpy as np



from tests._swe2d_test_helpers import (
    _make_gmsh_triangle_mesh,
    _build_mesh,
    stoker_dam_break,
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


def _gmsh_available():
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestGPUUnstructuredDamBreak(unittest.TestCase):
    LX, LY = 1000.0, 50.0
    SIZE_STABILITY = 50.0
    SIZE_ACCURACY = 25.0
    H_L, H_R = 2.0, 0.5
    T_END = 10.0

    def test_godunov_rollout_mode_smoke(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, cell_cx, _ = _make_gmsh_triangle_mesh(
            self.LX, self.LY, self.SIZE_STABILITY
        )
        mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
        h0 = np.where(cell_cx <= self.LX / 2.0, self.H_L, self.H_R)

        solver = mod.swe2d_create_solver(
            mesh,
            h0.copy(),
            n_mann=0.0,
            cfl=0.45,
            dt_max=0.5,
            temporal_order=2,
            spatial_scheme=0,
            godunov_mode=1,
            use_gpu=True,
        )

        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = mod.swe2d_step(solver, -1.0)
            t += last_diag["dt"]

        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        self.assertTrue(last_diag["gpu_active"], "GPU inactive in Godunov rollout mode")
        self.assertTrue(np.isfinite(h).all() and np.isfinite(hu).all() and np.isfinite(hv).all())
        self.assertGreaterEqual(float(np.min(h)), 0.0)
        self.assertLess(float(np.max(h)), 1.0e6)

    def test_stability_all_schemes(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, cell_cx, _ = _make_gmsh_triangle_mesh(
            self.LX, self.LY, self.SIZE_STABILITY
        )
        mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
        h0 = np.where(cell_cx <= self.LX / 2.0, self.H_L, self.H_R)

        for scheme_id in range(7):
            solver = mod.swe2d_create_solver(
                mesh,
                h0.copy(),
                n_mann=0.0,
                cfl=0.45,
                dt_max=0.5,
                spatial_scheme=scheme_id,
                use_gpu=True,
            )

            t = 0.0
            step = 0
            last_diag = None
            while t < self.T_END:
                last_diag = mod.swe2d_step(solver, -1.0)
                t += last_diag["dt"]
                step += 1
                h, hu, hv = mod.swe2d_get_state(solver)
                if (not np.isfinite(h).all()) or (np.max(h) > 1.0e6):
                    mod.swe2d_destroy(solver)
                    self.fail(
                        f"GPU scheme {scheme_id} diverged at step {step}, "
                        f"t={t:.3f}s, hmax={np.max(np.abs(h)):.3e}"
                    )

            h, hu, hv = mod.swe2d_get_state(solver)
            mod.swe2d_destroy(solver)
            self.assertTrue(last_diag["gpu_active"], f"GPU inactive for scheme {scheme_id}")
            self.assertTrue(np.isfinite(h).all(), f"Non-finite depth for scheme {scheme_id}")

    def test_accuracy_scheme0(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = _make_gmsh_triangle_mesh(
            self.LX, self.LY, self.SIZE_ACCURACY
        )
        mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
        h0 = np.where(cell_cx <= self.LX / 2.0, self.H_L, self.H_R)

        solver = mod.swe2d_create_solver(
            mesh,
            h0.copy(),
            n_mann=0.0,
            cfl=0.45,
            dt_max=0.5,
            spatial_scheme=0,
            use_gpu=True,
        )

        t = 0.0
        while t < self.T_END:
            diag = mod.swe2d_step(solver, -1.0)
            t += diag["dt"]

        h, _, _ = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        self.assertTrue(diag["gpu_active"])

        mid_y = self.LY / 2.0
        strip_tol = self.LY * 0.15
        mask = np.abs(cell_cy - mid_y) < strip_tol
        cx_strip = cell_cx[mask]
        h_strip = h[mask]

        order = np.argsort(cx_strip)
        cx_strip = cx_strip[order]
        h_strip = h_strip[order]

        x_shifted = cx_strip - self.LX / 2.0
        h_exact = stoker_dam_break(x_shifted, self.T_END, self.H_L, self.H_R)
        linf = np.max(np.abs(h_strip - h_exact))

        self.assertLess(linf, 0.50, f"GPU unstructured scheme0 L_inf too large: {linf:.4f}")


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestGPUUnstructuredLakeAtRest(unittest.TestCase):
    LX, LY = 200.0, 100.0
    SIZE = 15.0
    ETA0 = 1.0
    A_BED = 0.3
    N_STEPS = 100

    @classmethod
    def _zb_func(cls, x, y):
        return cls.A_BED * np.sin(np.pi * x / cls.LX) * np.cos(np.pi * y / cls.LY)

    def test_well_balanced_all_schemes(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, _, _ = _make_gmsh_triangle_mesh(
            self.LX, self.LY, self.SIZE, zb_func=self._zb_func
        )
        mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)

        cn = cell_nodes.reshape(-1, 3)
        zb_cell = (node_z[cn[:, 0]] + node_z[cn[:, 1]] + node_z[cn[:, 2]]) / 3.0
        h0 = np.maximum(0.0, self.ETA0 - zb_cell)

        for scheme_id in range(7):
            solver = mod.swe2d_create_solver(
                mesh,
                h0.copy(),
                n_mann=0.0,
                cfl=0.45,
                dt_max=5.0,
                spatial_scheme=scheme_id,
                use_gpu=True,
            )

            last_diag = None
            for _ in range(self.N_STEPS):
                last_diag = mod.swe2d_step(solver, -1.0)

            h, _, _ = mod.swe2d_get_state(solver)
            mod.swe2d_destroy(solver)

            self.assertTrue(last_diag["gpu_active"], f"GPU inactive for scheme {scheme_id}")

            eta = h + zb_cell
            wet = h > 1.0e-6
            self.assertTrue(wet.any(), f"GPU scheme {scheme_id}: all cells dry")
            self.assertTrue(np.isfinite(eta[wet]).all(), f"GPU scheme {scheme_id}: non-finite eta")

            deviation = np.max(np.abs(eta[wet] - self.ETA0))
            self.assertLess(
                deviation,
                1.0e-8,
                f"GPU scheme {scheme_id}: lake-at-rest drift {deviation:.3e}",
            )


def _make_tiny_triangle_pair_mesh():
    """Build a two-triangle mesh with one highly skinny cell.

    The first triangle has area ~5e-10 m^2, which makes it a good regression
    target for degenerate-cell handling: if the solver keeps treating that cell
    as active, the CFL timestep collapses to the tiny-cell scale.
    """
    node_x = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 1.0e-9, 1.0], dtype=np.float64)
    node_z = np.zeros(4, dtype=np.float64)
    cell_nodes = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
    return node_x, node_y, node_z, cell_nodes


@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestGPUDegenerateCellHandling(unittest.TestCase):
    """Regression test for tiny / nearly collinear cells.

    The solver should force the skinny cell quiescent instead of letting it
    dominate the CFL timestep or retain a non-physical wet state.
    """

    def test_tiny_cell_is_quiesced_and_dt_recovers(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes = _make_tiny_triangle_pair_mesh()
        mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)

        # Both cells start wet so the test checks whether the tiny cell is
        # actively suppressed rather than merely left untouched.
        h0 = np.array([1.0, 1.0], dtype=np.float64)
        solver = mod.swe2d_create_solver(
            mesh,
            h0.copy(),
            n_mann=0.0,
            cfl=0.45,
            dt_max=0.5,
            use_gpu=True,
            gpu_diag_sync_interval_steps=1,
            degen_mode=1,  # Skip mode: permanently exclude degenerate cells
        )

        diag0 = mod.swe2d_step(solver, -1.0)
        h1, hu1, hv1 = mod.swe2d_get_state(solver)
        diag1 = mod.swe2d_step(solver, -1.0)
        h2, hu2, hv2 = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        self.assertTrue(diag0["gpu_active"])
        self.assertTrue(diag1["gpu_active"])
        self.assertTrue(np.isfinite(h1).all() and np.isfinite(hu1).all() and np.isfinite(hv1).all())
        self.assertTrue(np.isfinite(h2).all() and np.isfinite(hu2).all() and np.isfinite(hv2).all())

        # The skinny cell is cell 0. A good degenerate-cell policy should force
        # it dry/quiescent after the first step instead of leaving it wet.
        self.assertLess(
            h1[0],
            1.0e-12,
            f"Tiny cell remained wet after step 1: h={h1[0]:.3e}",
        )

        # Once the tiny cell is quenched, the CFL timestep should recover to a
        # normal-scale value driven by the regular triangle.
        self.assertGreater(
            diag1["dt"],
            1.0e-4,
            f"CFL timestep stayed tiny after quenching the skinny cell: dt={diag1['dt']:.3e}",
        )

        # The regular triangle should stay finite and positive.
        self.assertGreaterEqual(h2[1], 0.0)
        self.assertTrue(np.isfinite(h2[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
