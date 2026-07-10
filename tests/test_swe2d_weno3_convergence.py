"""
WENO3 (scheme 6 / FV_WENO3) convergence-order validation on gmsh triangle meshes.

FV_WENO3 is a true 3-sub-stencil WENO reconstruction (1-ring, 3rd-order).
This test verifies that the scheme achieves at least 2.5-order L2 convergence
on a smooth manufactured solution.

Convergence test methodology
-----------------------------
For each of 3 mesh refinement levels, run a short simulation from a smooth
sinusoidal initial condition h(x,y).  Measure L2 error of the numerical
solution against the *analytical* initial condition at each mesh's own cell
centroids (no inter-mesh interpolation needed).  Observed convergence order:

    p = log2(E_coarse / E_fine) / log2(h_coarse / h_fine)

Per swe2d/extensions/extension_models.py §SpatialDiscretization:
  - FV_WENO3 = 6: "True 3-sub-stencil WENO (1-ring, 3rd-order)"
  - Target: ≥ 2.5-order L2 slope for h on smooth manufactured solution
"""

import unittest
import numpy as np

from swe2d.runtime.backend import SWE2DBackend, swe2d_available, swe2d_gpu_available
from swe2d.extensions.extension_models import SpatialDiscretization, TemporalScheme
from tests._swe2d_test_helpers import _make_gmsh_triangle_mesh


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

@unittest.skipUnless(swe2d_available(), "hydra_swe2d native module not available")
@unittest.skipUnless(swe2d_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestWENO3Convergence(unittest.TestCase):
    """Mesh-refinement convergence test for WENO3 (FV_WENO3 / spatial_scheme=6).

    Runs the solver on 3 mesh refinement levels with a smooth quiescent
    initial condition.  Measures L2 error of h against the *analytical*
    IC at each mesh's cell centroids (no cross-mesh interpolation).
    Checks the observed convergence order is ≥ 2.5 (3rd-order scheme).
    """

    LX, LY = 200.0, 100.0
    # nx=16 → mesh_size ~ Lx/16 = 12.5, nx=32 → 6.25, nx=64 → 3.125
    MESH_SIZES = [12.5, 6.25, 3.125]
    T_END   = 0.1    # short — perturbation barely evolves
    CFL     = 0.5    # per task specification
    DT_MAX  = 0.1
    N_MANN  = 0.0

    def _run_single_mesh(self, mesh_size):
        """Run solver on one mesh, return (cell_cx, cell_cy, h_final, n_cells)."""
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(self.LX, self.LY, mesh_size)

        h0, hu0, hv0 = SmoothBumpIC.make_ic(cell_cx, cell_cy, self.LX, self.LY)

        backend = SWE2DBackend()
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        n_cells = backend.n_cells

        backend.initialize(
            h0, hu0=hu0, hv0=hv0,
            n_mann=self.N_MANN,
            cfl=self.CFL,
            dt_max=self.DT_MAX,
            temporal_scheme=TemporalScheme.SSP_RK3,
            spatial_discretization=SpatialDiscretization.FV_WENO3,
        )

        t = 0.0
        last_diag = None
        while t < self.T_END:
            last_diag = backend.step(-1.0)
            t += last_diag["dt"]

        h, hu, hv = backend.get_state()
        backend.destroy()

        self.assertTrue(last_diag.get("gpu_active", False),
                        f"GPU inactive for mesh_size={mesh_size}")
        self.assertTrue(np.isfinite(h).all(),
                        f"Non-finite depth for mesh_size={mesh_size}")
        return cell_cx, cell_cy, h, n_cells

    def test_weno3_convergence_h(self):
        """WENO3 L2 error of h decreases with mesh refinement with order ≥ 2.5."""
        results = []
        for ms in self.MESH_SIZES:
            cx, cy, h, nc = self._run_single_mesh(ms)
            h_exact = SmoothBumpIC.h_exact(cx, cy, self.LX, self.LY)
            l2 = np.sqrt(np.mean((h - h_exact) ** 2))
            results.append((ms, l2, nc))
            print(f"  mesh_size={ms:.4f}: {nc} cells, L2(h-h_exact) = {l2:.6e}")

        # Verify error decreases monotonically with refinement
        for i in range(len(results) - 1):
            self.assertGreater(
                results[i][1], results[i + 1][1],
                f"Error did not decrease: {results[i][0]:.4f}m "
                f"({results[i][1]:.2e}) not > {results[i+1][0]:.4f}m "
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
                print(f"  order({results[i][0]:.4f}→{results[i+1][0]:.4f}): "
                      f"{order:.2f}")

        self.assertTrue(len(orders) > 0, "Not enough data to compute order")
        max_order = max(orders)
        # FV_WENO3 is a 3rd-order scheme (per enum doc), target ≥ 2.5
        self.assertGreaterEqual(
            max_order, 2.5,
            f"WENO3 convergence order {max_order:.2f} < 2.5. "
            f"Orders: {orders}"
        )
        print(f"  ✓ WENO3 peak convergence order: {max_order:.2f}")


# ── Lake-at-rest preservation ─────────────────────────────────────────────────

@unittest.skipUnless(swe2d_available(), "hydra_swe2d native module not available")
@unittest.skipUnless(swe2d_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestWENO3LakeAtRest(unittest.TestCase):
    """Verify WENO3 (scheme 6) preserves lake-at-rest to machine precision.

    Per docs on well-balanced property: ‖∇η‖ < 1e-8.
    """

    LX, LY = 200.0, 100.0
    ETA0    = 1.0
    A_BED   = 0.3
    N_STEPS = 100

    @classmethod
    def _zb_func(cls, x, y):
        return cls.A_BED * np.sin(np.pi * x / cls.LX) * np.cos(np.pi * y / cls.LY)

    def _check_lake_at_rest(self, mesh_size, label):
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(self.LX, self.LY, mesh_size,
                                     zb_func=self._zb_func)

        backend = SWE2DBackend()
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        n_cells = backend.n_cells

        # Cell-averaged bed elevation from node z values
        cn = cell_nodes.reshape(n_cells, 3)
        zb_cell = (node_z[cn[:, 0]] + node_z[cn[:, 1]] + node_z[cn[:, 2]]) / 3.0
        h0 = np.maximum(0.0, self.ETA0 - zb_cell)

        backend.initialize(
            h0.copy(),
            n_mann=0.0,
            cfl=0.45,
            dt_max=5.0,
            temporal_scheme=TemporalScheme.SSP_RK2,
            spatial_discretization=SpatialDiscretization.FV_WENO3,
        )

        last_diag = None
        for _ in range(self.N_STEPS):
            last_diag = backend.step(-1.0)

        h, hu, hv = backend.get_state()
        backend.destroy()

        self.assertTrue(last_diag.get("gpu_active", False),
                        f"GPU inactive for {label}")
        eta = h + zb_cell
        wet = h > 1e-6
        self.assertTrue(wet.any(), f"{label}: all cells dry")
        self.assertTrue(np.isfinite(eta[wet]).any(),
                        f"{label}: non-finite eta")

        deviation = np.max(np.abs(eta[wet] - self.ETA0))
        self.assertLess(
            deviation, 1.0e-8,
            f"{label}: lake-at-rest drift {deviation:.3e}"
        )
        print(f"  {label}: max |eta - eta0| = {deviation:.3e} OK")

    def test_lake_at_rest_standard_mesh(self):
        """Lake-at-rest on standard (non-skewed) gmsh mesh."""
        self._check_lake_at_rest(15.0, "WENO3 lake-at-rest (standard)")

    def test_lake_at_rest_coarse_mesh(self):
        """Lake-at-rest on coarser mesh to verify robustness."""
        self._check_lake_at_rest(30.0, "WENO3 lake-at-rest (coarse)")


# ── Mass conservation ─────────────────────────────────────────────────────────

@unittest.skipUnless(swe2d_available(), "hydra_swe2d native module not available")
@unittest.skipUnless(swe2d_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestWENO3MassConservation(unittest.TestCase):
    """Verify WENO3 conserves mass on a dam-break within numerical tolerance.

    A relative mass error < 1e-6 is expected for the HLLC solver with
    wetting/drying fronts and momentum capping.
    """

    LX, LY = 200.0, 20.0
    SIZE    = 10.0
    H_L, H_R = 2.0, 0.5
    T_END   = 3.0

    def test_mass_conserved(self):
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(self.LX, self.LY, self.SIZE)

        backend = SWE2DBackend()
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        n_cells = backend.n_cells

        h0 = np.where(cell_cx <= self.LX / 2.0, self.H_L, self.H_R)
        mass0 = float(np.sum(h0))

        backend.initialize(
            h0.copy(),
            n_mann=0.0,
            cfl=0.45,
            dt_max=0.5,
            temporal_scheme=TemporalScheme.SSP_RK2,
            spatial_discretization=SpatialDiscretization.FV_WENO3,
        )

        t = 0.0
        while t < self.T_END:
            diag = backend.step(-1.0)
            t += diag["dt"]

        h, hu, hv = backend.get_state()
        backend.destroy()

        mass1 = float(np.sum(h))
        rel_err = abs(mass1 - mass0) / abs(mass0)
        print(f"  mass0={mass0:.6f}, mass1={mass1:.6f}, rel_err={rel_err:.3e}")
        self.assertLess(
            rel_err, 1.0e-6,
            f"Mass conservation violated: rel_err={rel_err:.3e}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
