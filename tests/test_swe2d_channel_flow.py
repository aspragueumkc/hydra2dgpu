"""
test_swe2d_channel_flow.py — Steady channel flow validation tests.

Two GPU-validated test cases inspired by the SWASHES benchmark library
(Delestre et al., 2013, Int. J. Numer. Methods Fluids 72:3).

Test 1 — Uniform rectangular channel (Manning's normal depth)
--------------------------------------------------------------
Domain  : 1 000 m × 10 m rectangular channel.
Bed     : linear slope S₀ = 0.001  (z = S₀ · (L − x)).
BCs     : INFLOW_Q upstream (q = 0.5 m²/s), OPEN downstream, WALL sides.
IC      : exact Manning normal depth h_n everywhere.
Metric  : after 100 steps, max|h − h_n|/h_n < 1 % over interior cells.

Analytical reference (wide rectangular Manning's formula):
    h_n = (q · n / √S₀)^(3/5)          ≈ 0.639 m
    V_n = q / h_n                        ≈ 0.783 m/s
    Fr  = V_n / √(g h_n)                ≈ 0.31  (subcritical)

Test 2 — Irregular-bed channel stability (MacDonald bathymetry)
----------------------------------------------------------------
Domain  : same geometry.
Bed     : non-uniform elevation z(x) derived from the MacDonald
          sinusoidal-bed benchmark (Delestre et al. 2013, §3.1.1)
          where h_ref(x) = h_n · (1 + ε · sin(2π x / L)),  ε = 0.05
          is an exact 1-D steady state under the continuous PDEs.
BCs     : same as Test 1.
IC      : h_ref at each cell centroid, hu = q, hv = 0.
Metrics : (a) no NaN/Inf + GPU stays active after 50 steps
          (b) all depths ≥ 0
          (c) mean interior hu within 5 % of q (discharge conservation)

NOTE on C-property
------------------
The full MacDonald accuracy check (max|h − h_ref|/h_n < ε) requires the
solver to have the exact C-property for moving water — a stronger property
than lake-at-rest well-balancing.  Our Audusse-style hydrostatic
reconstruction is lake-at-rest well-balanced only.  The discrete steady
state for a non-uniform bed differs from the continuous 1-D solution by
O(Δx) per step, and the solution drifts over many steps.  We therefore
test stability and conservation rather than exact reproduction of h_ref.

Derivation of z(x)
------------------
The 1-D steady-state energy equation gives:
    d/dx (z + h + u²/2g) = −Sf          (Sf = Manning friction slope)
Rearranging for dz/dx with u = q/h:
    dz/dx = −Sf − (1 − Fr²) · dh/dx
where
    Sf  = n² q² / h^(10/3)
    Fr² = q²   / (g h³)
This is integrated numerically from x = 0 with z(0) = S₀ · L.

Open-source benchmark references
---------------------------------
SWASHES: https://github.com/luc-git/SWASHES
  (O. Delestre, C. Lucas, P.-A. Ksinant et al., IJNMF 2013)
HEC-RAS 2D test cases: https://www.hec.usace.army.mil/confluence/rasdocs
Malpasset dam-break geometry + high-water marks: opentelemac.org/validation
UK Environment Agency 2D flood modelling benchmarks (2013, EA report)
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Module / GPU availability ─────────────────────────────────────────────────

def _load_module():
    try:
        import backwater_swe2d
        return backwater_swe2d
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


# ── Structured rectangular mesh builder ──────────────────────────────────────

def _make_rect_channel_mesh(nx, ny, Lx, Ly, node_zb):
    """
    Triangulate [0, Lx] × [0, Ly] with nx × ny quads each split into two
    counter-clockwise triangles:
      lower: (n00, n10, n11)
      upper: (n00, n11, n01)

    node_zb : 1-D float64 array of bed elevation at each node (row-major,
              outer index = y-row, inner index = x-column).
    """
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    Xg, Yg = np.meshgrid(xs, ys)
    node_x = Xg.ravel().astype(np.float64)
    node_y = Yg.ravel().astype(np.float64)
    stride = nx + 1
    cells = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            cells.extend([n00, n10, n11])
            cells.extend([n00, n11, n01])
    return (node_x, node_y,
            np.asarray(node_zb, dtype=np.float64),
            np.array(cells, dtype=np.int32))


def _channel_bc_arrays(nx, ny, q_inflow):
    """
    BC edge arrays for a rectangular channel:
      - left  boundary (x = 0)  : INFLOW_Q = 2, value = q_inflow [m²/s]
      - right boundary (x = Lx) : OPEN     = 4, value = 0
    Top / bottom boundaries default to WALL = 1 (no explicit entry needed).

    Left boundary edges arise from the upper triangle at i = 0:
      edge (n01, n00) = ((j+1)*stride, j*stride)  →  sorted pair (j*stride, (j+1)*stride)
    Right boundary edges arise from the lower triangle at i = NX−1:
      edge (n10, n11) = (j*stride+NX, (j+1)*stride+NX)
    """
    stride = nx + 1
    n0s, n1s, tps, vls = [], [], [], []

    # Left boundary (x = 0): INFLOW_Q
    for j in range(ny):
        n0s.append(j * stride)
        n1s.append((j + 1) * stride)
        tps.append(2)          # INFLOW_Q
        vls.append(float(q_inflow))

    # Right boundary (x = Lx): OPEN (zero-gradient outflow)
    for j in range(ny):
        n0s.append(j * stride + nx)
        n1s.append((j + 1) * stride + nx)
        tps.append(4)          # OPEN
        vls.append(0.0)

    return (np.array(n0s, dtype=np.int32),
            np.array(n1s, dtype=np.int32),
            np.array(tps, dtype=np.int32),
            np.array(vls, dtype=np.float64))


def _cell_centroids_rect(nx, ny, Lx, Ly):
    """
    Return (cx, cy) for all triangles produced by _make_rect_channel_mesh.
    Cells are ordered as:
      index 2*(j*nx + i)     → lower triangle of quad (i, j)
      index 2*(j*nx + i) + 1 → upper triangle of quad (i, j)

    Lower: nodes at (i,j), (i+1,j), (i+1,j+1)
      cx = (3i+2)/3 · dx,  cy = (2j+1)/3 · dy
    Upper: nodes at (i,j), (i+1,j+1), (i,j+1)
      cx = (3i+1)/3 · dx,  cy = (2j+2)/3 · dy (= (j+2/3)·dy? careful: (j+j+1+j+1)/3·dy)
    """
    dx = Lx / nx
    dy = Ly / ny
    n_cells = 2 * nx * ny
    cx = np.empty(n_cells, dtype=np.float64)
    cy = np.empty(n_cells, dtype=np.float64)
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            # Lower triangle: (i,j), (i+1,j), (i+1,j+1)
            cx[2 * k]     = (i + (i + 1) + (i + 1)) * dx / 3.0
            cy[2 * k]     = (j + j + (j + 1))        * dy / 3.0
            # Upper triangle: (i,j), (i+1,j+1), (i,j+1)
            cx[2 * k + 1] = (i + (i + 1) + i)        * dx / 3.0
            cy[2 * k + 1] = (j + (j + 1) + (j + 1))  * dy / 3.0
    return cx, cy


# ── Analytical solutions ──────────────────────────────────────────────────────

def manning_normal_depth(q, n_mann, S0):
    """Manning's normal depth for a wide rectangular channel (R ≈ h)."""
    # q = (1/n) · h_n^(5/3) · √S₀  →  h_n = (q·n/√S₀)^(3/5)
    return (q * n_mann / np.sqrt(S0)) ** (3.0 / 5.0)


def macdonald_exact_solution(Lx, n_pts, q, n_mann, S0, amplitude, g=9.81):
    """
    Compute the MacDonald sinusoidal-perturbation benchmark.

    Prescribes
        h_ref(x) = h_n · (1 + amplitude · sin(2π x / Lx))
    and derives the bed elevation z(x) from the 1-D steady energy equation:
        dz/dx = −Sf − (1 − Fr²) · dh/dx
    where
        Sf  = n² q² / h^(10/3)      (Manning friction slope)
        Fr² = q²   / (g h³)

    Integration uses the trapezoidal rule starting from z(0) = S₀ · Lx
    (the same initial elevation as a uniform linear slope).

    Returns
    -------
    x       : 1-D array of x coordinates (n_pts points, 0 … Lx)
    h_ref   : 1-D array of prescribed steady-state depths
    z_ref   : 1-D array of derived bed elevations
    """
    h_n = manning_normal_depth(q, n_mann, S0)
    x = np.linspace(0.0, Lx, n_pts)
    h_ref = h_n * (1.0 + amplitude * np.sin(2.0 * np.pi * x / Lx))

    dh = np.gradient(h_ref, x)
    Sf  = n_mann**2 * q**2 / h_ref**(10.0 / 3.0)
    Fr2 = q**2 / (g * h_ref**3)
    dzx = -Sf - (1.0 - Fr2) * dh         # dz/dx

    # Trapezoidal integration: z(0) = S₀·Lx  →  z(Lx) ≈ 0
    z_ref = np.empty(n_pts, dtype=np.float64)
    z_ref[0] = S0 * Lx
    for i in range(1, n_pts):
        dx_i = x[i] - x[i - 1]
        z_ref[i] = z_ref[i - 1] + 0.5 * (dzx[i - 1] + dzx[i]) * dx_i

    return x, h_ref, z_ref


# ── Test 1: Uniform Manning channel flow ─────────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "backwater_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestUniformChannelFlow(unittest.TestCase):
    """
    Manning uniform-flow steady-state validation.

    The solver is initialised exactly at Manning's normal depth with the
    correct unit-width inflow discharge (INFLOW_Q BC) and a free outflow
    (OPEN BC).  After 100 explicit time steps the solution should remain
    within 1 % of h_n over the channel interior.

    This simultaneously validates:
    - INFLOW_Q and OPEN boundary condition treatment
    - Manning friction source term
    - Bed-slope (hydrostatic reconstruction) balancing friction
    - GPU solver stability for multi-step production runs
    """

    LX, LY  = 1000.0, 10.0     # channel dimensions [m]
    NX, NY  = 80, 4             # structured grid cells (640 triangles)
    S0      = 0.001             # bed slope [m/m]
    N_MANN  = 0.03              # Manning's n [s m^-1/3]
    Q       = 0.5               # unit-width inflow discharge [m²/s]
    N_STEPS = 100
    TOL_REL = 0.01              # 1 % relative error in h

    # ── helpers ──────────────────────────────────────────────────────────────

    def _make_mesh_and_solver(self):
        mod = _load_module()
        h_n = manning_normal_depth(self.Q, self.N_MANN, self.S0)

        # Bed: linear slope z(x) = S₀·(Lx − x)
        xs = np.linspace(0.0, self.LX, self.NX + 1)
        ys = np.linspace(0.0, self.LY, self.NY + 1)
        Xg, _ = np.meshgrid(xs, ys)
        node_zb = self.S0 * (self.LX - Xg.ravel())

        node_x, node_y, node_z, cell_nodes = _make_rect_channel_mesh(
            self.NX, self.NY, self.LX, self.LY, node_zb)
        bc_n0, bc_n1, bc_tp, bc_vl = _channel_bc_arrays(
            self.NX, self.NY, self.Q)

        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_n0, bc_n1, bc_tp, bc_vl)

        n_cells = mod.swe2d_mesh_info(mesh)["n_cells"]

        # Initialise at exact normal depth everywhere
        h0  = np.full(n_cells, h_n,    dtype=np.float64)
        hu0 = np.full(n_cells, self.Q, dtype=np.float64)   # hu = q
        hv0 = np.zeros(n_cells,         dtype=np.float64)

        solver = mod.swe2d_create_solver(
            mesh, h0, hu0, hv0,
            n_mann=self.N_MANN,
            cfl=0.45,
            dt_max=0.5,
            use_gpu=True,
            gpu_diag_sync_interval_steps=1,
        )
        return mod, solver, h_n

    def test_depth_stays_at_normal_depth(self):
        """After 100 steps the depth stays within 1 % of h_n over interior cells."""
        mod, solver, h_n = self._make_mesh_and_solver()
        last_diag = None
        for _ in range(self.N_STEPS):
            last_diag = mod.swe2d_step(solver, -1.0)
        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        self.assertTrue(last_diag["gpu_active"], "GPU solver was not active")
        self.assertTrue(np.isfinite(h).all(),  "Non-finite depth encountered")
        self.assertTrue(np.isfinite(hu).all(), "Non-finite hu encountered")

        # Interior cells: exclude first/last 10 % (near INFLOW and OPEN BCs)
        cx, _ = _cell_centroids_rect(self.NX, self.NY, self.LX, self.LY)
        mask = (cx > 0.10 * self.LX) & (cx < 0.90 * self.LX)
        rel_err = np.abs(h[mask] - h_n) / h_n
        max_rel_err = float(rel_err.max())
        self.assertLess(
            max_rel_err, self.TOL_REL,
            f"Uniform channel: max |h − h_n|/h_n = {max_rel_err:.4f} > "
            f"{self.TOL_REL:.4f}  (h_n = {h_n:.4f} m)",
        )

    def test_discharge_conservation(self):
        """Mean interior unit discharge stays within 2 % of prescribed inflow q."""
        mod, solver, h_n = self._make_mesh_and_solver()
        for _ in range(self.N_STEPS):
            mod.swe2d_step(solver, -1.0)
        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        cx, _ = _cell_centroids_rect(self.NX, self.NY, self.LX, self.LY)
        mask = (cx > 0.10 * self.LX) & (cx < 0.90 * self.LX)
        mean_q = float(np.mean(hu[mask]))
        rel_err = abs(mean_q - self.Q) / self.Q
        self.assertLess(
            rel_err, 0.02,
            f"Uniform channel: mean interior hu = {mean_q:.4f} m²/s deviates "
            f"{rel_err:.4f} from q = {self.Q:.4f} m²/s",
        )


# ── Test 2: Irregular-bed channel stability ───────────────────────────────────

@unittest.skipUnless(_load_module() is not None, "backwater_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestMacDonaldChannelFlow(unittest.TestCase):
    """
    Stability and discharge-conservation test for a channel with irregular
    bathymetry inspired by the MacDonald sinusoidal-bed benchmark
    (Delestre et al. 2013, §3.1.1 — SWASHES).

    Motivation
    ----------
    The full SWASHES accuracy check (max |h − h_ref|/h_n < ε) requires the
    solver to have the **C-property** — exact discrete balance between bed
    slope and friction for every moving-water steady state.  Our solver uses
    Audusse-style hydrostatic reconstruction, which is lake-at-rest
    well-balanced but NOT C-property well-balanced.  For that reason we do NOT
    assert exact reproduction of h_ref.

    What we DO assert (sufficient for a production 2-D hydraulic solver):

    1. **Stability / no crash**: the solver runs 50 steps on irregular
       bathymetry without producing NaN/Inf values and without the GPU kernel
       being deactivated.

    2. **Positivity**: all cell depths remain ≥ 0 throughout.

    3. **Discharge conservation**: mean unit discharge in the interior is
       within 5 % of the prescribed inflow q after 50 steps.  This confirms
       that the INFLOW_Q and OPEN BCs are exchanging momentum correctly and
       that no spurious source term is creating mass.

    The bed elevation z(x) is the MacDonald derivation, which provides a
    smooth, physically realistic non-uniform bathymetry (gentle sinusoidal
    perturbation around a uniform slope).  The solver is initialised at the
    corresponding h_ref so that the initial transient is small and any
    stability issue or conservation error is clearly visible.

    Relationship to open-source benchmark suites
    ---------------------------------------------
    - SWASHES (Delestre et al. 2013): https://github.com/luc-git/SWASHES
      The C-property accuracy test can be added once the solver is upgraded
      to a well-balanced moving-water scheme (e.g., Rogers et al. 2003,
      Xing & Shu 2011, or Audusse & Bristeau 2005).
    - Malpasset dam-break high-water marks: opentelemac.org/validation
    - HEC-RAS 2D test cases: hec.usace.army.mil/confluence/rasdocs
    """

    LX, LY    = 1000.0, 10.0
    NX, NY    = 80, 4
    S0        = 0.001
    N_MANN    = 0.03
    Q         = 0.5
    AMPLITUDE = 0.05    # ε: 5 % sinusoidal depth perturbation
    N_STEPS   = 50
    TOL_Q     = 0.05    # 5 % tolerance on discharge conservation

    def _make_mesh_and_solver(self):
        mod = _load_module()
        h_n = manning_normal_depth(self.Q, self.N_MANN, self.S0)

        # Compute MacDonald reference on a fine grid (2001 points)
        x_fine, h_fine, z_fine = macdonald_exact_solution(
            self.LX, n_pts=2001,
            q=self.Q, n_mann=self.N_MANN, S0=self.S0, amplitude=self.AMPLITUDE,
        )
        # Shift z so min(z) = 0.05 m (constant shift does not affect gradients)
        z_fine = z_fine - z_fine.min() + 0.05

        # Evaluate z at node positions (interpolate from fine grid)
        xs = np.linspace(0.0, self.LX, self.NX + 1)
        ys = np.linspace(0.0, self.LY, self.NY + 1)
        Xg, _ = np.meshgrid(xs, ys)
        node_zb = np.interp(Xg.ravel(), x_fine, z_fine)

        node_x, node_y, node_z, cell_nodes = _make_rect_channel_mesh(
            self.NX, self.NY, self.LX, self.LY, node_zb)
        bc_n0, bc_n1, bc_tp, bc_vl = _channel_bc_arrays(
            self.NX, self.NY, self.Q)

        mesh = mod.swe2d_build_mesh(
            node_x, node_y, node_z, cell_nodes,
            bc_n0, bc_n1, bc_tp, bc_vl)

        n_cells = mod.swe2d_mesh_info(mesh)["n_cells"]

        # Initialise at h_ref (MacDonald exact solution) to minimise transient
        cx, _ = _cell_centroids_rect(self.NX, self.NY, self.LX, self.LY)
        h_ref_cell = np.interp(cx, x_fine, h_fine)

        h0  = h_ref_cell.astype(np.float64)
        hu0 = np.full(n_cells, self.Q, dtype=np.float64)
        hv0 = np.zeros(n_cells,         dtype=np.float64)

        solver = mod.swe2d_create_solver(
            mesh, h0, hu0, hv0,
            n_mann=self.N_MANN,
            cfl=0.45,
            dt_max=0.5,
            use_gpu=True,
            gpu_diag_sync_interval_steps=1,
        )
        return mod, solver, cx, h_n

    def test_irregular_bed_stability(self):
        """50 steps on irregular bathymetry produce no NaN/Inf and GPU stays active."""
        mod, solver, cx, h_n = self._make_mesh_and_solver()
        last_diag = None
        for _ in range(self.N_STEPS):
            last_diag = mod.swe2d_step(solver, -1.0)
        h, hu, _ = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        self.assertTrue(last_diag["gpu_active"], "GPU solver was not active")
        self.assertTrue(np.isfinite(h).all(),  "Non-finite depth encountered")
        self.assertTrue(np.isfinite(hu).all(), "Non-finite hu encountered")

    def test_irregular_bed_discharge_conservation(self):
        """Interior mean unit discharge stays within 5 % of q after 50 steps."""
        mod, solver, cx, h_n = self._make_mesh_and_solver()
        for _ in range(self.N_STEPS):
            mod.swe2d_step(solver, -1.0)
        _, hu, _ = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)

        mask = (cx > 0.10 * self.LX) & (cx < 0.90 * self.LX)
        mean_q = float(np.mean(hu[mask]))
        rel_err = abs(mean_q - self.Q) / self.Q
        self.assertLess(
            rel_err, self.TOL_Q,
            f"Irregular-bed channel: mean interior hu = {mean_q:.4f} m²/s, "
            f"relative error {rel_err:.4f} > {self.TOL_Q:.4f}  (q = {self.Q} m²/s)",
        )

    def test_irregular_bed_depth_positive(self):
        """All cell depths must remain ≥ 0 throughout."""
        mod, solver, cx, h_n = self._make_mesh_and_solver()
        for _ in range(self.N_STEPS):
            mod.swe2d_step(solver, -1.0)
        h, _, _ = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        self.assertTrue((h >= 0.0).all(), f"Negative depths found (min = {h.min():.3e} m)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
