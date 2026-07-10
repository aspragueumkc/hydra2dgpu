"""
Robustness tests for the three new spatial schemes (5, 6, 8) on
poor-quality (highly-graded / low-element-quality) unstructured meshes.

Validates that each scheme:
  - Completes a simulation without crashing.
  - Produces only finite (non-NaN, non-Inf) depth and momentum.
  - Maintains depth within physically admissible bounds (no excessive
    overshoot / undershoot that would indicate catastrophic instability).

References
----------
docs/ADVANCED_SPATIAL_SCHEMES.md §3, §4, §5
cpp/src/swe2d_solver.hpp — SWE2DSpatialScheme enum
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

# ── Scheme definitions ─────────────────────────────────────────────────────────

SCHEMES: list[int] = [5, 6, 8]
SCHEME_NAMES: dict[int, str] = {
    5: "Barth-Jespersen",
    6: "WENO3",
    8: "MP5",
}
# Conservative CFL values for stability on poor-quality meshes.
SCHEME_CFL: dict[int, float] = {
    5: 0.50,
    6: 0.50,
    8: 0.30,
}

DOMAIN_LX, DOMAIN_LY = 100.0, 5.0      # High aspect ratio → poor elements
MESH_SIZE = 12.0                        # Very coarse → poor element quality
H_LEFT, H_RIGHT = 2.0, 0.5             # Dam-break initial condition
T_END = 1.0                             # Short simulation


def _make_poor_mesh():
    """Build a deliberately poor-quality unstructured triangle mesh via Gmsh.

    A coarse mesh on a high-aspect-ratio rectangle produces elongated,
    low-quality triangles.  This stresses the gradient / reconstruction
    kernels far more than a well-shaped mesh would.

    Returns
    -------
    (node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy)
    """
    from tests._swe2d_test_helpers import _make_gmsh_triangle_mesh

    return _make_gmsh_triangle_mesh(DOMAIN_LX, DOMAIN_LY, MESH_SIZE)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _have_all, reason="requires hydra_swe2d, CUDA GPU, and gmsh")
@pytest.mark.parametrize("scheme", SCHEMES, ids=[SCHEME_NAMES[s] for s in SCHEMES])
def test_no_nan_on_poor_mesh(scheme: int):
    """Each new scheme must produce only finite values on a poor-quality mesh.

    Fails if any depth or momentum value is NaN or Inf after a short
    dam-break simulation on a highly coarsed, high-aspect-ratio mesh.
    """
    mod = _hydra
    node_x, node_y, node_z, cell_nodes, cell_cx, _ = _make_poor_mesh()

    # Build mesh with empty boundary conditions (all wall)
    mesh = mod.swe2d_build_mesh(
        node_x, node_y, node_z, cell_nodes,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )

    # Dam-break initial condition
    h0 = np.where(cell_cx <= DOMAIN_LX / 2.0, H_LEFT, H_RIGHT).astype(np.float64)

    solver = mod.swe2d_create_solver(
        mesh, h0,
        n_mann=0.0,
        cfl=SCHEME_CFL[scheme],
        dt_max=0.1,
        temporal_order=2,
        spatial_scheme=scheme,
        use_gpu=True,
    )

    t = 0.0
    last_diag = None
    while t < T_END:
        last_diag = mod.swe2d_step(solver, -1.0)
        t += last_diag["dt"]

    h, hu, hv = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)

    # ── Assertions ────────────────────────────────────────────────────────
    assert last_diag is not None, "No diagnostic returned — solver failed to step"
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
@pytest.mark.parametrize("scheme", SCHEMES, ids=[SCHEME_NAMES[s] for s in SCHEMES])
def test_no_excessive_oscillations(scheme: int):
    """Scheme must keep depth within physically admissible bounds.

    On a dam-break IC (hL=2.0, hR=0.5), the numerical solution should
    not overshoot the initial range by more than 30 % of the jump
    (≈0.45 m).  This catches catastrophic oscillation modes that higher-
    order reconstructions can exhibit on poor-quality meshes.
    """
    mod = _hydra
    node_x, node_y, node_z, cell_nodes, cell_cx, _ = _make_poor_mesh()

    mesh = mod.swe2d_build_mesh(
        node_x, node_y, node_z, cell_nodes,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )

    h0 = np.where(cell_cx <= DOMAIN_LX / 2.0, H_LEFT, H_RIGHT).astype(np.float64)

    solver = mod.swe2d_create_solver(
        mesh, h0,
        n_mann=0.0,
        cfl=SCHEME_CFL[scheme],
        dt_max=0.1,
        temporal_order=2,
        spatial_scheme=scheme,
        use_gpu=True,
    )

    t = 0.0
    while t < T_END:
        diag = mod.swe2d_step(solver, -1.0)
        t += diag["dt"]

    h, hu, hv = mod.swe2d_get_state(solver)
    mod.swe2d_destroy(solver)

    # Physical bounds with 30 % overshoot tolerance
    jump = H_LEFT - H_RIGHT  # 1.5 m
    tol = 0.30 * jump         # 0.45 m
    lower_bound = H_RIGHT - tol   # 0.05 m
    upper_bound = H_LEFT + tol    # 2.45 m

    min_h = float(h.min())
    max_h = float(h.max())
    name = SCHEME_NAMES[scheme]

    assert min_h >= lower_bound, \
        f"Scheme {scheme} ({name}): h.min()={min_h:.4f} m below lower bound " \
        f"{lower_bound:.4f} m — excessive undershoot"
    assert max_h <= upper_bound, \
        f"Scheme {scheme} ({name}): h.max()={max_h:.4f} m above upper bound " \
        f"{upper_bound:.4f} m — excessive overshoot"
