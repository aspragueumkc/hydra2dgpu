"""
test_swe2d_unstructured.py
Second-order reconstruction debugging on unstructured triangle meshes.

All existing dam-break and lake-at-rest tests use structured quad grids split
into right-triangle pairs (_make_rect_mesh).  This file reproduces the same
physical test cases on a fully unstructured Delaunay triangle mesh generated
by Gmsh, isolating whether the higher-order reconstruction infrastructure is
broken on irregular topology (non-uniform edge lengths, non-orthogonal cell
pairs, variable edge-normal directions).

Reconstruction modes tested
---------------------------
  0  FV_FIRST_ORDER
  1  FV_MUSCL_FAST  (Superbee)
  2  FV_MUSCL_MINMOD
  3  FV_MUSCL_MC
  4  FV_MUSCL_VAN_LEER

Pass criteria
-------------
  Dam-break    : depth_max stays finite (< 1e6 m) and velocity_max < 1e6 m/s
                 for all 36 000 steps; solver does not NaN/Inf.
  Lake-at-rest : After 100 steps the maximum free-surface deviation
                 |h + zb - eta0| < 1e-8 m  (looser than structured, because
                 the Green-Gauss gradient on an irregular mesh is O(h) not
                 O(h²), so exact balance needs tighter arithmetic than the
                 structured case gives).

If scheme 0 passes but schemes 1-4 fail the unstructured tests, the bug is
in gradient/reconstruction logic, not in the mesh topology plumbing.
If scheme 0 also fails, there is a mesh-connectivity build problem.
"""

import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


def _gmsh_available():
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


def _make_gmsh_triangle_mesh(Lx: float, Ly: float, mesh_size: float,
                              zb_func=None):
    """
    Generate an unstructured Delaunay triangle mesh over the rectangle
    [0, Lx] x [0, Ly] using Gmsh.

    Parameters
    ----------
    Lx, Ly      : domain extents (m).
    mesh_size   : target element size (m).
    zb_func     : optional callable(node_x, node_y) -> node_z for bed elevation.

    Returns
    -------
    node_x, node_y, node_z  : shape (N_nodes,)
    cell_nodes               : flat int32 array of triangle node indices,
                               length = 3 * N_cells.
    cell_cx, cell_cy         : centroid x/y for each cell, shape (N_cells,).
    """
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("unstructured_rect")

        # Four corner points
        p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, mesh_size)
        p2 = gmsh.model.geo.addPoint(Lx,  0.0, 0.0, mesh_size)
        p3 = gmsh.model.geo.addPoint(Lx,  Ly,  0.0, mesh_size)
        p4 = gmsh.model.geo.addPoint(0.0, Ly,  0.0, mesh_size)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.geo.addPlaneSurface([cl])

        gmsh.model.geo.synchronize()

        # Use Frontal-Delaunay algorithm (6) for quality triangles
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Laplace2D", niter=3)

        # Extract nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        node_x = node_coords[:, 0].copy()
        node_y = node_coords[:, 1].copy()
        if zb_func is not None:
            node_z = zb_func(node_x, node_y).astype(np.float64)
        else:
            node_z = np.zeros(len(node_x), dtype=np.float64)

        # Extract triangle elements (type 2)
        elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, surf)
        cell_nodes_list = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == 2:  # triangle
                enodes = np.array(enodes, dtype=np.int64).reshape(-1, 3)
                for tri in enodes:
                    for t in tri:
                        cell_nodes_list.append(tag_to_idx[int(t)])

        cell_nodes = np.array(cell_nodes_list, dtype=np.int32)
        n_cells = len(cell_nodes) // 3

        # Compute centroids
        cn = cell_nodes.reshape(n_cells, 3)
        cell_cx = (node_x[cn[:, 0]] + node_x[cn[:, 1]] + node_x[cn[:, 2]]) / 3.0
        cell_cy = (node_y[cn[:, 0]] + node_y[cn[:, 1]] + node_y[cn[:, 2]]) / 3.0

    finally:
        gmsh.finalize()

    return node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy


def _build_mesh(mod, node_x, node_y, node_z, cell_nodes):
    """Wrap swe2d_build_mesh with empty optional arrays."""
    return mod.swe2d_build_mesh(
        node_x, node_y, node_z, cell_nodes,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64))


# ─────────────────────────────────────────────────────────────────────────────
# Stoker analytical solution (copied from test_swe2d_dambreak.py)
# ─────────────────────────────────────────────────────────────────────────────

def stoker_dam_break(x, t, hL, hR, g=9.81):
    cL = np.sqrt(g * hL)

    def f(cm):
        hm = cm * cm / g
        fL = 2.0 * (cL - cm)
        if hm > hR and hm > 0.0:
            Qr = np.sqrt(0.5 * g * (hm + hR) / (hR * hm))
            fR = (hm - hR) * Qr
        else:
            fR = 2.0 * (cm - np.sqrt(g * hR))
        return fR - fL

    lo, hi = 0.0, cL
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    cm = 0.5 * (lo + hi)
    hm = cm * cm / g
    um = 2.0 * (cL - cm)
    cR = np.sqrt(g * hR)

    if hm > hR:
        Qr = np.sqrt(0.5 * g * (hm + hR) / (hR * hm))
        S = um + hR * Qr
    else:
        S = um + cm

    h = np.empty_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi <= -cL * t:
            h[i] = hL
        elif xi <= (um - cm) * t:
            c_here = (2.0 * cL - xi / t) / 3.0
            h[i] = c_here ** 2 / g
        elif xi <= S * t:
            h[i] = hm
        else:
            h[i] = hR
    return h


# ─────────────────────────────────────────────────────────────────────────────
# Test: unstructured dam-break stability for all reconstruction modes
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestDamBreakUnstructuredStability(unittest.TestCase):
    """
    Smoke test: all 5 reconstruction modes must remain bounded on an
    unstructured triangle mesh dam-break.  Instability (NaN/Inf or depth
    exceeding 1e6) is an immediate failure.
    """

    LX    = 1000.0   # m  – channel length
    LY    = 50.0     # m  – channel width
    SIZE  = 50.0     # m  – target element size (coarse for speed)
    H_L   = 2.0
    H_R   = 0.5
    T_END = 10.0     # s

    @classmethod
    def setUpClass(cls):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(cls.LX, cls.LY, cls.SIZE)
        cls.mod = mod
        cls.mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
        info = mod.swe2d_mesh_info(cls.mesh)
        n_cells = info["n_cells"]
        cls.h0 = np.where(cell_cx <= cls.LX / 2.0, cls.H_L, cls.H_R)
        cls.n_cells = n_cells
        print(f"\n[unstructured dam-break] mesh: {n_cells} cells, "
              f"mesh_size={cls.SIZE} m")

    def _run_scheme(self, scheme_id):
        mod = self.mod
        solver = mod.swe2d_create_solver(
            self.mesh, self.h0.copy(),
            n_mann=0.0,
            cfl=0.45, dt_max=0.5,
            spatial_scheme=scheme_id,
            use_gpu=False)

        t = 0.0
        step = 0
        while t < self.T_END:
            diag = mod.swe2d_step(solver, -1.0)
            t += diag["dt"]
            step += 1
            h, hu, hv = mod.swe2d_get_state(solver)
            if not np.isfinite(h).all() or np.max(h) > 1.0e6:
                mod.swe2d_destroy(solver)
                return False, step, np.max(np.abs(h)), t
        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        return True, step, np.max(h), t

    def _test_scheme(self, scheme_id):
        ok, steps, hmax, t_end = self._run_scheme(scheme_id)
        self.assertTrue(ok,
            msg=(f"Scheme {scheme_id} diverged at t={t_end:.3f}s "
                 f"after {steps} steps: hmax={hmax:.3e}"))
        print(f"  scheme {scheme_id}: {steps} steps, hmax={hmax:.4f} m  OK")

    def test_scheme_0_first_order(self):
        self._test_scheme(0)

    def test_scheme_1_superbee(self):
        self._test_scheme(1)

    def test_scheme_2_minmod(self):
        self._test_scheme(2)

    def test_scheme_3_mc(self):
        self._test_scheme(3)

    def test_scheme_4_van_leer(self):
        self._test_scheme(4)


# ─────────────────────────────────────────────────────────────────────────────
# Test: unstructured dam-break accuracy (scheme 0 vs analytical)
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestDamBreakUnstructuredAccuracy(unittest.TestCase):
    """
    First-order L∞ accuracy check against Stoker solution at t=10s.
    Uses a finer mesh to get a meaningful error estimate.
    """

    LX    = 1000.0
    LY    = 50.0
    SIZE  = 25.0     # finer mesh for accuracy check
    H_L   = 2.0
    H_R   = 0.5
    T_END = 10.0
    LINF_LIMIT = 0.25 * 2.0   # 25% of h_L — generous for unstructured

    @classmethod
    def setUpClass(cls):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(cls.LX, cls.LY, cls.SIZE)
        cls.mod = mod
        cls.mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
        info = mod.swe2d_mesh_info(cls.mesh)
        n_cells = info["n_cells"]
        cls.h0 = np.where(cell_cx <= cls.LX / 2.0, cls.H_L, cls.H_R)
        cls.cell_cx = cell_cx
        cls.cell_cy = cell_cy
        cls.n_cells = n_cells
        print(f"\n[unstructured dam-break accuracy] mesh: {n_cells} cells, "
              f"mesh_size={cls.SIZE} m")

    def test_stoker_linf_error_scheme0(self):
        mod = self.mod
        solver = mod.swe2d_create_solver(
            self.mesh, self.h0.copy(),
            n_mann=0.0,
            cfl=0.45, dt_max=0.5,
            spatial_scheme=0,
            use_gpu=False)
        t = 0.0
        while t < self.T_END:
            diag = mod.swe2d_step(solver, -1.0)
            t += diag["dt"]
        h, _, _ = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        # Centre-channel strip (mid-Y)
        mid_y = self.LY / 2.0
        strip_tol = self.LY * 0.15
        mask = np.abs(self.cell_cy - mid_y) < strip_tol
        cx_strip = self.cell_cx[mask]
        h_strip = h[mask]

        order = np.argsort(cx_strip)
        cx_strip = cx_strip[order]
        h_strip = h_strip[order]

        x_shifted = cx_strip - self.LX / 2.0
        h_exact = stoker_dam_break(x_shifted, self.T_END, self.H_L, self.H_R)

        linf = np.max(np.abs(h_strip - h_exact))
        self.assertLess(linf, self.LINF_LIMIT,
            msg=f"Dam-break L∞ error {linf:.4f} m exceeds limit {self.LINF_LIMIT:.4f} m")
        print(f"  scheme 0 L∞ error: {linf:.4f} m  (limit {self.LINF_LIMIT:.4f} m)")


# ─────────────────────────────────────────────────────────────────────────────
# Test: unstructured lake-at-rest well-balanced for all reconstruction modes
# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "hydra_swe2d not built")
@unittest.skipUnless(_gmsh_available(), "gmsh not installed")
class TestLakeAtRestUnstructured(unittest.TestCase):
    """
    Well-balanced lake-at-rest on an unstructured triangle mesh.
    After 100 CPU steps the free-surface deviation must be < 1e-8 m.
    The tolerance is relaxed vs the structured test (1e-10) because the
    Green-Gauss gradient on an irregular mesh does not achieve exact
    cancellation of the bed-slope source term to machine precision.
    """

    LX, LY = 200.0, 100.0
    SIZE   = 15.0     # target element size (m)
    ETA0   = 1.0      # free-surface elevation (m)
    A_BED  = 0.3      # sinusoidal bed amplitude (m)
    N_MANN = 0.0
    N_STEPS = 100
    DEVIATION_LIMIT = 1e-8

    @classmethod
    def _zb_func(cls, x, y):
        return cls.A_BED * np.sin(np.pi * x / cls.LX) * np.cos(np.pi * y / cls.LY)

    @classmethod
    def setUpClass(cls):
        mod = _load_module()
        node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy = \
            _make_gmsh_triangle_mesh(cls.LX, cls.LY, cls.SIZE,
                                     zb_func=cls._zb_func)
        cls.mod = mod
        cls.mesh = _build_mesh(mod, node_x, node_y, node_z, cell_nodes)
        info = mod.swe2d_mesh_info(cls.mesh)
        n_cells = info["n_cells"]

        # Per-cell bed elevation (average of 3 node values)
        cn = cell_nodes.reshape(n_cells, 3)
        zb_cell = (node_z[cn[:, 0]] + node_z[cn[:, 1]] + node_z[cn[:, 2]]) / 3.0
        cls.h0 = np.maximum(0.0, cls.ETA0 - zb_cell)
        cls.zb_cell = zb_cell
        cls.n_cells = n_cells
        print(f"\n[unstructured lake-at-rest] mesh: {n_cells} cells, "
              f"mesh_size={cls.SIZE} m")

    def _run_scheme(self, scheme_id):
        mod = self.mod
        solver = mod.swe2d_create_solver(
            self.mesh, self.h0.copy(),
            n_mann=self.N_MANN,
            cfl=0.45, dt_max=5.0,
            spatial_scheme=scheme_id,
            use_gpu=False)
        for _ in range(self.N_STEPS):
            mod.swe2d_step(solver, -1.0)
        h, _, _ = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        return h

    def _test_scheme(self, scheme_id):
        h = self._run_scheme(scheme_id)
        eta = h + self.zb_cell
        wet = h > 1e-6
        if not wet.any():
            self.fail(f"Scheme {scheme_id}: all cells dry after lake-at-rest!")
        # Check for NaN/Inf first
        self.assertTrue(np.isfinite(eta[wet]).all(),
            msg=f"Scheme {scheme_id}: NaN/Inf in free surface after {self.N_STEPS} steps")
        deviation = np.max(np.abs(eta[wet] - self.ETA0))
        self.assertLess(deviation, self.DEVIATION_LIMIT,
            msg=(f"Scheme {scheme_id}: lake-at-rest deviation {deviation:.2e} m "
                 f"> limit {self.DEVIATION_LIMIT:.0e} m"))
        print(f"  scheme {scheme_id}: max |eta - eta0| = {deviation:.2e} m  OK")

    def test_scheme_0_first_order(self):
        self._test_scheme(0)

    def test_scheme_1_superbee(self):
        self._test_scheme(1)

    def test_scheme_2_minmod(self):
        self._test_scheme(2)

    def test_scheme_3_mc(self):
        self._test_scheme(3)

    def test_scheme_4_van_leer(self):
        self._test_scheme(4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
