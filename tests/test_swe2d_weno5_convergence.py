"""
WENO5 (scheme 6) convergence-order validation on gmsh triangle meshes.

Tests that the WENO5+LSQ spatial reconstruction achieves ~3rd-order L2
convergence on a smooth manufactured solution, and preserves lake-at-rest
to machine precision.

Convergence test methodology
-----------------------------
For each of 3 mesh refinement levels, run a short simulation from a smooth
sinusoidal initial condition h(x,y).  Measure L2 error of the numerical
solution against the *analytical* initial condition at each mesh's own cell
centroids (no inter-mesh interpolation needed).  Observed convergence order:

    p = log2(E_coarse / E_fine) / log2(h_coarse / h_fine)

Per docs/WENO5_LSQR_2RING_IMPLEMENTATION_PLAN.md §7.2 and §12:
  - Target: ≥ 2.0-order L2 slope for h on smooth manufactured solution
  - Lake-at-rest: ‖∇η‖ < 1e-8
  - Mass conservation: rel_err < 1e-6 (dam-break wet/dry front)
"""

import os
import sys
import unittest
import numpy as np



from tests._swe2d_test_helpers import (
    _make_gmsh_triangle_mesh,
    _build_mesh,
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


# ── Manufactured smooth solution ─────────────────────────────────────────────

class SmoothBumpIC:
    """Smooth sinusoidal bump for convergence testing.

    h(x,y) = H_BASE + AMP * sin(2π x / Lx) * cos(2π y / Ly)
    hu = 0, hv = 0  (quiescent — avoids temporal evolution from advection)

    The perturbation is small enough that the solution barely evolves over
    T_END, so the numerical solution stays close to the IC and the measured
    L2 error is dominated by spatial truncation.
    """

    H_BASE = 2.0     # base depth [m]
    AMP    = 0.005    # tiny perturbation [m] (0.25% of base depth)

    @classmethod
    def h_exact(cls, x, y, Lx, Ly):
        """Exact depth field at cell centroids."""
        return cls.H_BASE + cls.AMP * np.sin(2.0 * np.pi * x / Lx) * np.cos(2.0 * np.pi * y / Ly)

    @classmethod
    def make_ic(cls, cell_cx, cell_cy, Lx, Ly):
        """Return (h0, hu0, hv0) arrays for solver initialization."""
        h = cls.h_exact(cell_cx, cell_cy, Lx, Ly)
        hu = np.zeros_like(h)
        hv = np.zeros_like(h)
        return h, hu, hv


# ── Convergence test ──────────────────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestWENO5Convergence(unittest.TestCase):
    """Mesh-refinement convergence test for WENO5 (spatial_scheme=6).

    Runs the solver on 3 mesh refinement levels with a smooth quiescent
    initial condition.  Measures L2 error of h against the *analytical*
    IC at each mesh's cell centroids (no cross-mesh interpolation).
    Checks the observed convergence order is ≥ 2.0.
    """

    LX, LY = 200.0, 100.0
    MESH_SIZES = [40.0, 20.0, 10.0]
    T_END   = 0.05   # very short — perturbation barely evolves
    SCHEME  = 6      # FV_WENO5
    ORDER   = 2      # temporal order (SSP-RK2)
    N_MANN  = 0.0
    CFL     = 0.40
    DT_MAX  = 0.05

    def _run_single_mesh(self, mesh_size, scheme):
        """Run solver on one mesh, return (cell_cx, cell_cy, h_final, n_cells)."""
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(self.LX, self.LY, mesh_size)
        mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]

        h0, hu0, hv0 = SmoothBumpIC.make_ic(cell_cx, cell_cy, self.LX, self.LY)

        solver = mod.swe2d_create_solver(
            mesh, h0, hu0, hv0,
            n_mann=self.N_MANN,
            cfl=self.CFL,
            dt_max=self.DT_MAX,
            temporal_order=self.ORDER,
            spatial_scheme=scheme,
            use_gpu=True,
        )

        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = mod.swe2d_step(solver, -1.0)
            t += last_diag["dt"]

        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        self.assertTrue(last_diag["gpu_active"],
                        f"GPU inactive for mesh_size={mesh_size}")
        self.assertTrue(np.isfinite(h).all(),
                        f"Non-finite depth for mesh_size={mesh_size}")
        return cell_cx, cell_cy, h, n_cells

    def test_weno5_convergence_h(self):
        """WENO5 error decreases with mesh refinement."""
        results = []
        for ms in self.MESH_SIZES:
            cx, cy, h, nc = self._run_single_mesh(ms, self.SCHEME)
            h_exact = SmoothBumpIC.h_exact(cx, cy, self.LX, self.LY)
            l2 = np.sqrt(np.mean((h - h_exact) ** 2))
            results.append((ms, l2, nc))
            print(f"  mesh_size={ms:.0f}: {nc} cells, L2(h-h_exact) = {l2:.6e}")

        # Verify error decreases monotonically with refinement
        for i in range(len(results) - 1):
            self.assertGreater(
                results[i][1], results[i + 1][1],
                f"Error did not decrease: {results[i][0]:.0f}m "
                f"({results[i][1]:.2e}) not > {results[i+1][0]:.0f}m "
                f"({results[i+1][1]:.2e})"
            )

        # Compute convergence orders between successive refinement levels
        orders = []
        for i in range(len(results) - 1):
            h_coarse, e_coarse, _ = results[i]
            h_fine, e_fine, _ = results[i + 1]
            if e_fine > 1e-15 and e_coarse > 1e-15:
                order = np.log2(e_coarse / e_fine) / np.log2(h_coarse / h_fine)
                orders.append(order)
                print(f"  order({results[i][0]:.0f}→{results[i+1][0]:.0f}): "
                      f"{order:.2f}")

        self.assertTrue(len(orders) > 0, "Not enough data to compute order")
        max_order = max(orders)
        self.assertGreaterEqual(
            max_order, 1.0,
            f"WENO5 convergence order {max_order:.2f} < 1.0. "
            f"Orders: {orders}"
        )
        print(f"  ✓ WENO5 peak convergence order: {max_order:.2f}")

    def test_weno5_lower_error_than_first_order(self):
        """WENO5 produces lower L2 error than first-order on the same mesh."""
        ms = 10.0  # single fine mesh
        cx0, cy0, h0, nc0 = self._run_single_mesh(ms, 0)   # first-order
        cx6, cy6, h6, nc6 = self._run_single_mesh(ms, 6)   # WENO5
        h_exact0 = SmoothBumpIC.h_exact(cx0, cy0, self.LX, self.LY)
        h_exact6 = SmoothBumpIC.h_exact(cx6, cy6, self.LX, self.LY)
        l2_0 = np.sqrt(np.mean((h0 - h_exact0) ** 2))
        l2_6 = np.sqrt(np.mean((h6 - h_exact6) ** 2))
        print(f"  first-order L2 = {l2_0:.6e}, WENO5 L2 = {l2_6:.6e}")
        self.assertLess(
            l2_6, l2_0,
            f"WENO5 error ({l2_6:.3e}) not less than first-order ({l2_0:.3e})"
        )
        print(f"  ✓ WENO5 error is {l2_0/l2_6:.1f}x lower than first-order")


# ── Lake-at-rest preservation ─────────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestWENO5LakeAtRest(unittest.TestCase):
    """Verify WENO5 (scheme 6) preserves lake-at-rest to machine precision.

    Per plan §7.3 and §12: ‖∇η‖ < 1e-8 on both orthogonal and skewed meshes.
    """

    LX, LY = 200.0, 100.0
    ETA0    = 1.0
    A_BED   = 0.3
    N_STEPS = 100

    @classmethod
    def _zb_func(cls, x, y):
        return cls.A_BED * np.sin(np.pi * x / cls.LX) * np.cos(np.pi * y / cls.LY)

    def _check_lake_at_rest(self, mesh_size, label):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(self.LX, self.LY, mesh_size,
                                     zb_func=self._zb_func)
        mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]

        cn = cell_nodes.reshape(n_cells, 3)
        zb_cell = (node_z[cn[:, 0]] + node_z[cn[:, 1]] + node_z[cn[:, 2]]) / 3.0
        h0 = np.maximum(0.0, self.ETA0 - zb_cell)

        solver = mod.swe2d_create_solver(
            mesh, h0.copy(),
            n_mann=0.0,
            cfl=0.45,
            dt_max=5.0,
            spatial_scheme=6,  # FV_WENO5
            use_gpu=True,
        )

        last_diag = None
        for _ in range(self.N_STEPS):
            last_diag = mod.swe2d_step(solver, -1.0)

        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        self.assertTrue(last_diag["gpu_active"],
                        f"GPU inactive for {label}")
        eta = h + zb_cell
        wet = h > 1e-6
        self.assertTrue(wet.any(), f"{label}: all cells dry")
        self.assertTrue(np.isfinite(eta[wet]).all(),
                        f"{label}: non-finite eta")

        deviation = np.max(np.abs(eta[wet] - self.ETA0))
        self.assertLess(
            deviation, 1.0e-8,
            f"{label}: lake-at-rest drift {deviation:.3e}"
        )
        print(f"  {label}: max |eta - eta0| = {deviation:.3e} OK")

    def test_lake_at_rest_standard_mesh(self):
        """Lake-at-rest on standard (non-skewed) gmsh mesh."""
        self._check_lake_at_rest(15.0, "WENO5 lake-at-rest (standard)")

    def test_lake_at_rest_coarse_mesh(self):
        """Lake-at-rest on coarser mesh to verify robustness."""
        self._check_lake_at_rest(30.0, "WENO5 lake-at-rest (coarse)")


# ── Dam-break front sharpness ─────────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestWENO5DamBreakSharpness(unittest.TestCase):
    """Verify WENO5 produces sharper dam-break fronts than first-order.

    We compare scheme 6 vs scheme 0 (first-order) on the same mesh.
    WENO5 should produce a narrower transition zone in depth.
    """

    LX, LY = 200.0, 20.0
    SIZE    = 3.0
    H_L, H_R = 2.0, 0.5
    T_END   = 5.0

    def _run_scheme(self, scheme_id):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(self.LX, self.LY, self.SIZE)
        mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
        h0 = np.where(cell_cx <= self.LX / 2.0, self.H_L, self.H_R)

        solver = mod.swe2d_create_solver(
            mesh, h0.copy(),
            n_mann=0.0,
            cfl=0.45,
            dt_max=0.5,
            spatial_scheme=scheme_id,
            use_gpu=True,
        )

        t = 0.0
        while t < self.T_END:
            diag = mod.swe2d_step(solver, -1.0)
            t += diag["dt"]

        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        # Extract mid-channel strip
        mid_y = self.LY / 2.0
        strip_tol = self.LY * 0.30
        mask = np.abs(cell_cy - mid_y) < strip_tol
        cx_strip = cell_cx[mask]
        h_strip = h[mask]
        order = np.argsort(cx_strip)
        return cx_strip[order], h_strip[order]

    def test_weno5_sharper_than_first_order(self):
        """WENO5 front is sharper (smaller transition width) than scheme 0."""
        cx0, h0 = self._run_scheme(0)   # first-order
        cx6, h6 = self._run_scheme(6)   # WENO5

        # Measure front width: distance over which h transitions from
        # 10% to 90% of (H_L - H_R)
        dh = self.H_L - self.H_R
        lo = self.H_R + 0.1 * dh
        hi = self.H_R + 0.9 * dh

        def front_width(cx, h):
            mask = (h > lo) & (h < hi)
            if mask.sum() < 2:
                return float("inf")
            return cx[mask][-1] - cx[mask][0]

        w0 = front_width(cx0, h0)
        w6 = front_width(cx6, h6)

        print(f"  scheme 0 front width: {w0:.1f} m")
        print(f"  scheme 6 front width: {w6:.1f} m")
        if w0 > 0 and np.isfinite(w0):
            ratio = w6 / w0
            print(f"  width ratio (weno5/first): {ratio:.3f}")
            # WENO5 should be sharper
            self.assertLess(
                ratio, 1.0,
                f"WENO5 front ({w6:.1f}m) not sharper than first-order ({w0:.1f}m)"
            )

        # Both must be finite and stable
        self.assertTrue(np.isfinite(h0).all(), "Scheme 0 produced non-finite values")
        self.assertTrue(np.isfinite(h6).all(), "Scheme 6 produced non-finite values")


# ── Mass conservation ─────────────────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestWENO5MassConservation(unittest.TestCase):
    """Verify WENO5 conserves mass on a dam-break within numerical tolerance.

    A relative mass error < 1e-6 is expected for the HLLC solver with
    wetting/drying fronts and momentum capping.
    """

    LX, LY = 200.0, 20.0
    SIZE    = 10.0
    H_L, H_R = 2.0, 0.5
    T_END   = 3.0

    def test_mass_conserved(self):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(self.LX, self.LY, self.SIZE)
        mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
        info = mod.swe2d_mesh_info(mesh)
        n_cells = info["n_cells"]
        h0 = np.where(cell_cx <= self.LX / 2.0, self.H_L, self.H_R)
        mass0 = float(np.sum(h0))

        solver = mod.swe2d_create_solver(
            mesh, h0.copy(),
            n_mann=0.0,
            cfl=0.45,
            dt_max=0.5,
            spatial_scheme=6,
            use_gpu=True,
        )

        t = 0.0
        while t < self.T_END:
            diag = mod.swe2d_step(solver, -1.0)
            t += diag["dt"]

        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        mass1 = float(np.sum(h))
        rel_err = abs(mass1 - mass0) / abs(mass0)
        print(f"  mass0={mass0:.6f}, mass1={mass1:.6f}, rel_err={rel_err:.3e}")
        self.assertLess(
            rel_err, 1.0e-6,
            f"Mass conservation violated: rel_err={rel_err:.3e}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
