"""
Verify face-value monotonicity for all spatial schemes 0–8.

The suite checks that every finite-volume spatial reconstruction scheme
produces physically admissible (non-negative, finite) depth under a
standard dam-break scenario.  This is the weakest form of "monotonicity"
— a scheme that violates these bounds is unusable.

Schemes tested
--------------
  0  FV_FIRST_ORDER      — Godunov first-order upwind
  1  FV_MUSCL_FAST       — MUSCL + Superbee (fast limiter)
  2  FV_MUSCL_MINMOD     — MUSCL + MinMod
  3  FV_MUSCL_MC          — MUSCL + Monotonized-Central
  4  FV_MUSCL_VAN_LEER   — MUSCL + Van Leer
  5  FV_BARTH_JESPERSEN  — LSQ gradient + Barth-Jespersen limiter
  6  FV_WENO3            — True 3-sub-stencil WENO
  7  FV_WENO5            — WENO5 + 2-ring LSQ gradient
  8  FV_MP5              — Suresh-Huynh mapped monotonicity-preserving

References
----------
swe2d/extensions/extension_models.py — SpatialDiscretization enum
cpp/src/swe2d_solver.hpp — SWE2DSpatialScheme
"""

import numpy as np
import pytest

from tests._swe2d_test_helpers import _load_module

# ── Dependency checks ──────────────────────────────────────────────────────────

_hydra = _load_module()
_gpu_ok = False
if _hydra is not None:
    try:
        _gpu_ok = _hydra.swe2d_gpu_available()
    except Exception:
        pass


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except ImportError:
        return False


_gmsh_ok = _gmsh_available()
_have_all = _hydra is not None and _gpu_ok and _gmsh_ok

# ── All spatial schemes ────────────────────────────────────────────────────────

ALL_SCHEMES: list[int] = list(range(9))
SCHEME_NAMES: dict[int, str] = {
    0: "First-order",
    1: "MUSCL-Fast",
    2: "MUSCL-MinMod",
    3: "MUSCL-MC",
    4: "MUSCL-VanLeer",
    5: "Barth-Jespersen",
    6: "WENO3",
    7: "WENO5",
    8: "MP5",
}

# Conservative CFL per scheme (from _SCHEME_MAX_CFL in backend.py).
SCHEME_CFL: dict[int, float] = {
    0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50,
    5: 0.50, 6: 0.50, 7: 0.40, 8: 0.30,
}

DOMAIN_LX, DOMAIN_LY = 100.0, 10.0
MESH_SIZE = 8.0
H_LEFT, H_RIGHT = 2.0, 0.5
T_END = 2.0


def _build_mesh_and_solver(scheme: int):
    """Create a Gmsh mesh, build the solver, and return (mod, solver, n_cells).

    The solver is already initialised with a dam-break IC.
    Caller must call ``mod.swe2d_destroy(solver)`` after use.
    """
    from tests._swe2d_test_helpers import _make_gmsh_triangle_mesh

    mod = _hydra
    node_x, node_y, node_z, cell_nodes, cell_cx, _ = \
        _make_gmsh_triangle_mesh(DOMAIN_LX, DOMAIN_LY, MESH_SIZE)

    mesh = mod.swe2d_build_mesh(
        node_x, node_y, node_z, cell_nodes,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )

    info = mod.swe2d_mesh_info(mesh)
    n_cells = info["n_cells"]

    h0 = np.where(cell_cx <= DOMAIN_LX / 2.0, H_LEFT, H_RIGHT).astype(np.float64)

    solver = mod.swe2d_create_solver(
        mesh, h0,
        n_mann=0.0,
        cfl=SCHEME_CFL[scheme],
        dt_max=0.2,
        temporal_order=2,
        spatial_scheme=scheme,
        use_gpu=True,
    )

    return mod, solver, n_cells


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _have_all, reason="requires hydra_swe2d, CUDA GPU, and gmsh")
@pytest.mark.parametrize("scheme", ALL_SCHEMES, ids=[SCHEME_NAMES[s] for s in ALL_SCHEMES])
def test_solution_not_nan(scheme: int):
    """Each scheme must produce finite (non-NaN, non-Inf) depth and momentum."""
    mod, solver, _ = _build_mesh_and_solver(scheme)

    t = 0.0
    last_diag = None
    while t < T_END:
        last_diag = mod.swe2d_step(solver, -1.0)
        t += last_diag["dt"]

    h, hu, hv = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)

    assert last_diag is not None, "No diagnostic — solver did not step"
    assert last_diag.get("gpu_active", False), \
        f"GPU inactive for scheme {scheme} ({SCHEME_NAMES[scheme]})"
    assert np.isfinite(h).all(), \
        f"Scheme {scheme} ({SCHEME_NAMES[scheme]}) produced NaN/Inf depth"
    assert np.isfinite(hu).all(), \
        f"Scheme {scheme} ({SCHEME_NAMES[scheme]}) produced NaN/Inf x-momentum"
    assert np.isfinite(hv).all(), \
        f"Scheme {scheme} ({SCHEME_NAMES[scheme]}) produced NaN/Inf y-momentum"
    assert np.all(h >= 0), \
        f"Scheme {scheme} ({SCHEME_NAMES[scheme]}) produced negative depth"


@pytest.mark.skipif(not _have_all, reason="requires hydra_swe2d, CUDA GPU, and gmsh")
@pytest.mark.parametrize("scheme", ALL_SCHEMES, ids=[SCHEME_NAMES[s] for s in ALL_SCHEMES])
def test_depth_non_negative(scheme: int):
    """Depth must remain non-negative across all cells after short run."""
    mod, solver, _ = _build_mesh_and_solver(scheme)

    t = 0.0
    while t < T_END:
        diag = mod.swe2d_step(solver, -1.0)
        t += diag["dt"]

    h, hu, hv = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)

    name = SCHEME_NAMES[scheme]
    assert np.all(h >= 0), \
        f"Scheme {scheme} ({name}): min depth = {float(h.min()):.4e} m (negative)"

    # Verify basic physical bounds: h should not exceed initial max + 20 %
    # of the jump.  All schemes should pass this generous bound.
    jump = H_LEFT - H_RIGHT
    upper = H_LEFT + 0.20 * jump
    assert h.max() <= upper, \
        f"Scheme {scheme} ({name}): max depth = {float(h.max()):.4e} m " \
        f"exceeds bound {upper:.4f} m"
