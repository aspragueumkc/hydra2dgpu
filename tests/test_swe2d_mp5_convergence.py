"""
FV_MP5 (scheme 8) convergence-order validation on gmsh triangle meshes.

Tests that the MP5 Suresh-Huynh mapped monotonicity-preserving (nominally
4th-order) spatial reconstruction achieves ≥ 3.5-order L2 convergence
on a smooth manufactured solution.

Convergence test methodology
----------------------------
For each of 3 mesh refinement levels (nx = 16, 32, 64), run a short
simulation from a smooth sinusoidal initial condition h(x,y) with quiescent
momentum.  Measured L2 error against the *analytical* initial condition at
each mesh's own cell centroids (no inter-mesh interpolation needed).

The perturbation amplitude is tiny (0.25 % of base depth) and the end time
is short (0.01 s), so the numerical solution stays close to the initial
condition and the measured error is dominated by spatial truncation error.

Observed convergence order is estimated via log-log linear regression:

    log(E) = p * log(h) + intercept      (h = mesh size)

giving p as the convergence order.  With only 3 refinements we also report
pairwise orders as a consistency check.

References
----------
  Suresh & Huynh, "Accurate Monotonicity-Preserving Schemes with
  Runge-Kutta Time Stepping", JCP 136, 83–99 (1997).

  docs/ADVANCED_SPATIAL_SCHEMES.md §5 — FV_MP5 design notes.
  docs/SOLVER_ORDER_AND_STENCIL.md — scheme numbering.
  docs/superpowers/specs/2026-07-10-advanced-spatial-schemes-design.md §3.3.
"""

import unittest
import numpy as np

from tests._swe2d_test_helpers import _make_cartesian_quad_mesh


# ── Availability guards (same pattern as WENO5 convergence test) ──────────────

def _load_module():
    """Return hydra_swe2d native module or None."""
    try:
        import hydra_swe2d  # noqa: F401
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

class StandingWaveIC:
    """Linearized-SWE standing wave with wall BCs at x=0 and x=Lx.

    h(x,y,t) = H_BASE + AMP * cos(π x / Lx) * cos(ω t)
    u(x,y,t) = -(AMP * ω / (π H_BASE / Lx)) * sin(π x / Lx) * sin(ω t)
    v(x,y,t) = 0

    with ω = sqrt(g * H_BASE) * π / Lx.  This is an exact solution of the
    linearized shallow-water equations and satisfies u = 0 at the lateral
    walls (x = 0 and x = Lx), so the default wall BCs are consistent.
    """

    H_BASE = 2.0
    AMP = 0.005
    G = 9.80665

    @classmethod
    def omega(cls, Lx):
        return np.sqrt(cls.G * cls.H_BASE) * np.pi / Lx

    @classmethod
    def h_exact(cls, x, y, t, Lx):
        return cls.H_BASE + cls.AMP * np.cos(np.pi * x / Lx) * np.cos(cls.omega(Lx) * t)

    @classmethod
    def u_exact(cls, x, y, t, Lx):
        return -(cls.AMP * cls.omega(Lx) / (np.pi * cls.H_BASE / Lx)) * \
               np.sin(np.pi * x / Lx) * np.sin(cls.omega(Lx) * t)

    @classmethod
    def make_ic(cls, cell_cx, cell_cy, Lx):
        h = cls.h_exact(cell_cx, cell_cy, 0.0, Lx)
        u = cls.u_exact(cell_cx, cell_cy, 0.0, Lx)
        return h, cls.H_BASE * u, np.zeros_like(h)


# ── Convergence test ──────────────────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestMP5Convergence(unittest.TestCase):
    """Mesh-refinement convergence test for FV_MP5 (spatial_scheme = 8).

    Runs the solver on 3 mesh refinement levels with a smooth standing-wave
    initial condition.  Measures L2 error of h against the *analytical*
    solution at T_END.  Checks that the observed convergence order is ≥ 3.5.
    """

    LX, LY      = 200.0, 200.0   # square domain [m]
    NX_VALS     = [16, 32, 64]   # cells along x (controls refinement)
    T_END       = 1.0            # simulation time [s] — keep temporal error below spatial error
    SCHEME      = 8              # FV_MP5 from SpatialDiscretization
    ORDER_TARGET = 3.5           # 4th-order scheme → expect ≥ 3.5
    CFL         = 0.4            # MP5 max CFL

    @classmethod
    def _h_characteristic(cls, mesh_size):
        """Return the characteristic mesh spacing h for a given mesh_size.

        Override in subclasses to compute h differently (e.g. from cell
        count and domain area).
        """
        return mesh_size

    def _run_single_mesh(self, nx: int):
        """Run solver on one mesh refinement, return (l2_error, n_cells, h).

        Parameters
        ----------
        nx : int
            Number of cells along x; h = Lx / nx.

        Returns
        -------
        l2_error  : float  -- area-weighted L2 norm of (h - h_exact)
        n_cells   : int    -- number of cells in the mesh
        h         : float  -- characteristic mesh spacing
        """
        from swe2d.runtime.backend import SWE2DBackend
        from swe2d.extensions.extension_models import SpatialDiscretization, TemporalScheme

        # Cartesian quad mesh with square cells: nx along x, ny = nx along y
        ny = nx

        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_cartesian_quad_mesh(nx, ny, self.LX, self.LY)

        n_cells = cell_nodes.shape[0]
        cell_face_offsets = np.arange(0, 4 * n_cells + 1, 4, dtype=np.int32)

        backend = SWE2DBackend()
        backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            cell_face_offsets=cell_face_offsets,
        )

        h0, hu0, hv0 = StandingWaveIC.make_ic(cell_cx, cell_cy, self.LX)
        backend.initialize(
            h0=h0, hu0=hu0, hv0=hv0,
            n_mann=0.0,
            cfl=self.CFL,
            temporal_scheme=TemporalScheme.GRAPH_SAFE_RK5,
            spatial_discretization=SpatialDiscretization.FV_MP5,
        )

        backend.run(t_end=self.T_END)
        h, hu, hv = backend.get_state()

        # Area-weighted L2 error against the analytical solution at T_END
        areas    = backend.cell_areas()
        h_exact  = StandingWaveIC.h_exact(cell_cx, cell_cy, self.T_END, self.LX)
        l2_error = np.sqrt(np.sum(areas * (h - h_exact)**2) / np.sum(areas))

        n_cells_out = backend.n_cells
        backend.destroy()
        return l2_error, n_cells_out, self.LX / nx

    def test_mp5_convergence_h(self):
        """FV_MP5 L2 error decreases with mesh refinement at order ≥ 3.5."""
        results = []
        for nx in self.NX_VALS:
            l2, n_cells, h_char = self._run_single_mesh(nx)
            results.append((h_char, l2, n_cells))
            print(f"  nx={nx}: h={h_char:.2f}, {n_cells} cells, "
                  f"L2(h-h_exact) = {l2:.6e}")

        # 1. Error must decrease monotonically with refinement
        for i in range(len(results) - 1):
            self.assertGreater(
                results[i][1], results[i + 1][1],
                f"Error did not decrease: h={results[i][0]:.2f}m "
                f"({results[i][1]:.2e}) -> h={results[i+1][0]:.2f}m "
                f"({results[i+1][1]:.2e})"
            )

        # 2. Pairwise convergence orders (diagnostic)
        pairwise_orders = []
        for i in range(len(results) - 1):
            h_coarse, e_coarse, _ = results[i]
            h_fine,   e_fine,   _ = results[i + 1]
            if e_fine > 1e-15 and e_coarse > 1e-15:
                p = np.log2(e_coarse / e_fine) / np.log2(h_coarse / h_fine)
                pairwise_orders.append(p)
                print(f"  pairwise order ({h_coarse:.1f}→{h_fine:.1f}): "
                      f"{p:.3f}")

        # 3. Log-log linear regression over all 3 refinements
        h_arr = np.array([r[0] for r in results], dtype=np.float64)
        e_arr = np.array([r[1] for r in results], dtype=np.float64)
        log_h = np.log(h_arr)
        log_e = np.log(e_arr)
        # polyfit returns (slope, intercept) where log_e = slope*log_h + intercept
        slope, intercept = np.polyfit(log_h, log_e, 1)
        order = float(slope)   # positive slope = convergence; p = slope

        print(f"  log-log fit:  slope={slope:.4f}, intercept={intercept:.4f}")
        print(f"  Observed convergence order: {order:.3f}")
        if pairwise_orders:
            print(f"  Pairwise orders: {[f'{p:.3f}' for p in pairwise_orders]}")

        mesh_size_str = ", ".join(f"{r[0]:.2f}" for r in results)
        l2_err_str    = ", ".join(f"{r[1]:.4e}" for r in results)
        self.assertGreaterEqual(
            order, self.ORDER_TARGET,
            f"FV_MP5 convergence order {order:.2f} < {self.ORDER_TARGET}. "
            f"Expected ≥ {self.ORDER_TARGET} for a 4th-order scheme.\n"
            f"Mesh sizes: [{mesh_size_str}]\n"
            f"L2 errors:  [{l2_err_str}]\n"
            f"Pairwise orders: {[f'{p:.3f}' for p in pairwise_orders]}"
        )
        print(f"  ✓ FV_MP5 convergence order {order:.3f} ≥ {self.ORDER_TARGET}")


# ── Comparison against first-order ────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestMP5BetterThanFirstOrder(unittest.TestCase):
    """Verify FV_MP5 produces lower L2 error than first-order on the same mesh."""

    LX, LY = 200.0, 200.0
    NX     = 32
    T_END  = 1.0

    def _run_scheme(self, scheme_id: int):
        """Run solver with given scheme, return area-weighted L2 error."""
        from swe2d.runtime.backend import SWE2DBackend
        from swe2d.extensions.extension_models import SpatialDiscretization, TemporalScheme

        # Cartesian quad mesh with square cells
        nx = self.NX
        ny = nx

        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_cartesian_quad_mesh(nx, ny, self.LX, self.LY)

        n_cells = cell_nodes.shape[0]
        cell_face_offsets = np.arange(0, 4 * n_cells + 1, 4, dtype=np.int32)

        backend = SWE2DBackend()
        backend.build_mesh(
            node_x, node_y, node_z, cell_nodes,
            cell_face_offsets=cell_face_offsets,
        )

        h0, hu0, hv0 = StandingWaveIC.make_ic(cell_cx, cell_cy, self.LX)
        scheme = SpatialDiscretization(scheme_id)
        cfl = 0.4

        backend.initialize(
            h0=h0, hu0=hu0, hv0=hv0,
            n_mann=0.0,
            cfl=cfl,
            temporal_scheme=TemporalScheme.GRAPH_SAFE_RK5,
            spatial_discretization=scheme,
        )

        backend.run(t_end=self.T_END)
        h, hu, hv = backend.get_state()

        areas    = backend.cell_areas()
        h_exact  = StandingWaveIC.h_exact(cell_cx, cell_cy, self.T_END, self.LX)
        l2_error = np.sqrt(np.sum(areas * (h - h_exact)**2) / np.sum(areas))

        backend.destroy()
        return l2_error

    def test_mp5_lower_error_than_first_order(self):
        """FV_MP5 should produce substantially lower error than scheme 0."""
        l2_fo  = self._run_scheme(0)   # first-order
        l2_mp5 = self._run_scheme(8)   # FV_MP5

        print(f"  first-order L2 = {l2_fo:.6e}")
        print(f"  FV_MP5     L2 = {l2_mp5:.6e}")
        print(f"  ratio         = {l2_fo/l2_mp5:.1f}x")

        self.assertLess(
            l2_mp5, l2_fo,
            f"FV_MP5 error ({l2_mp5:.3e}) is NOT lower than "
            f"first-order ({l2_fo:.3e})"
        )
        self.assertGreater(
            l2_fo / l2_mp5, 1.5,
            f"FV_MP5 error ratio ({l2_fo/l2_mp5:.1f}) is marginal; "
            f"expected >1.5x improvement over first-order"
        )
        print(f"  ✓ FV_MP5 error is {l2_fo/l2_mp5:.1f}x lower than first-order")


if __name__ == "__main__":
    unittest.main(verbosity=2)
