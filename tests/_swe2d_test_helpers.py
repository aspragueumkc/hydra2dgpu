"""
Shared test utilities for SWE2D GPU test suites.

Provides mesh builders, analytical solutions, and other helpers reused across
multiple test files.  Extracted from the former test_swe2d_unstructured.py and
test_swe2d_dambreak.py after the CPU-only path was removed.
"""

from __future__ import annotations

import sys
import os
from typing import Callable, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Module loader
# ─────────────────────────────────────────────────────────────────────────────
def _load_module():
    """Import and return the hydra_swe2d native module, or None."""
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


def _gpu_available():
    """Return True if CUDA GPU solver is available."""
    mod = _load_module()
    if mod is None:
        return False
    try:
        return mod.swe2d_gpu_available()
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
def _make_cartesian_quad_mesh(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    zb_func: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a Cartesian quadrilateral mesh with nx × ny cells.

    Returns (node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy).
    cell_nodes has shape (nx*ny, 4) with CCW node indices for each quad.
    """
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    Xg, Yg = np.meshgrid(xs, ys)
    node_x = Xg.ravel().copy()
    node_y = Yg.ravel().copy()
    node_z = zb_func(node_x, node_y) if zb_func is not None else np.zeros_like(node_x)

    cells = []
    centroids_x = []
    centroids_y = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n11 = (j + 1) * stride + i + 1
            n01 = (j + 1) * stride + i
            cells.append([n00, n10, n11, n01])
            centroids_x.append(0.25 * (node_x[n00] + node_x[n10] + node_x[n11] + node_x[n01]))
            centroids_y.append(0.25 * (node_y[n00] + node_y[n10] + node_y[n11] + node_y[n01]))

    cell_nodes = np.array(cells, dtype=np.int32)
    cell_cx = np.array(centroids_x, dtype=np.float64)
    cell_cy = np.array(centroids_y, dtype=np.float64)
    return node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy


# ─────────────────────────────────────────────────────────────────────────────
# Structured rectangular mesh builder (triangulated quads)
# ─────────────────────────────────────────────────────────────────────────────
def _make_rect_mesh(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    zb_func: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a structured rectangular mesh with nx × ny quads (2 triangles each).

    Returns (node_x, node_y, node_z, cell_nodes).
    """
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    Xg, Yg = np.meshgrid(xs, ys)
    node_x = Xg.ravel().copy()
    node_y = Yg.ravel().copy()
    node_z = zb_func(node_x, node_y) if zb_func is not None else np.zeros_like(node_x)

    cells = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            cells.extend([n00, n10, n11])
            cells.extend([n00, n11, n01])

    return node_x, node_y, node_z, np.array(cells, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Gmsh unstructured triangle mesh builder
# ─────────────────────────────────────────────────────────────────────────────
def _make_gmsh_triangle_mesh(
    Lx: float,
    Ly: float,
    mesh_size: float,
    zb_func: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate an unstructured triangular mesh with Gmsh.

    Returns (node_x, node_y, node_z, cell_nodes, cell_cx, cell_cy).
    """
    import gmsh
    gmsh.initialize()
    gmsh.model.add("swe2d_test_mesh")

    # Create rectangle
    tag = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
    gmsh.model.occ.synchronize()

    # Set uniform mesh size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Extract nodes
    _, node_coords, _ = gmsh.model.mesh.getNodes()
    node_xyz = node_coords.reshape(-1, 3)
    node_x = node_xyz[:, 0].copy()
    node_y = node_xyz[:, 1].copy()

    if zb_func is not None:
        node_z = zb_func(node_x, node_y).astype(np.float64)
    else:
        node_z = np.zeros_like(node_x)

    # Extract triangle elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    tri_type = 2  # 3-node triangle in Gmsh
    cell_nodes_list = []
    for etype, nodes in zip(elem_types, elem_node_tags):
        if etype == tri_type:
            cell_nodes_list.append(nodes.reshape(-1, 3))
    if not cell_nodes_list:
        gmsh.finalize()
        raise RuntimeError("No triangle elements found in gmsh mesh")
    cell_nodes = np.vstack(cell_nodes_list).astype(np.int32) - 1  # 1-based → 0-based

    gmsh.finalize()

    # Compute cell centroids
    cx = np.mean(node_x[cell_nodes], axis=1)
    cy = np.mean(node_y[cell_nodes], axis=1)

    return node_x, node_y, node_z, cell_nodes, cx, cy


# ─────────────────────────────────────────────────────────────────────────────
# Mesh wrapper (build mesh with empty BCs for testing)
# ─────────────────────────────────────────────────────────────────────────────
def _build_mesh(
    mod,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
    cell_nodes: np.ndarray,
) -> object:
    """Construct a mesh handle with no boundary conditions (all interior)."""
    return mod.swe2d_build_mesh(
        node_x, node_y, node_z, cell_nodes,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stoker (1957) analytical dam-break solution (wet-bed)
# ─────────────────────────────────────────────────────────────────────────────
def stoker_dam_break(
    x: np.ndarray,
    t: float,
    hL: float,
    hR: float,
    g: float = 9.81,
) -> np.ndarray:
    """Return h(x, t) for the Stoker wet-bed dam-break solution."""
    cL = np.sqrt(g * hL)
    cR = np.sqrt(g * hR)

    def f(cm):
        hm = cm * cm / g
        fL = 2.0 * (cL - cm)
        hm_ = max(hm, 1e-12)
        if hm > hR and hm > 0.0:
            Qr = np.sqrt(0.5 * g * (hm + hR) / (hR * hm_))
            fR = (hm - hR) * Qr
        else:
            fR = 2.0 * (cm - cR)
        return fR - fL

    # Bisection: f(0) < 0, f(cL) > 0
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

    # Right shock speed
    if hm > hR:
        Qr = np.sqrt(0.5 * g * (hm + hR) / (max(hR, 1e-12) * max(hm, 1e-12)))
        S = um + hR * Qr
    else:
        S = um + cm

    h = np.empty_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi <= -cL * t:
            h[i] = hL
        elif xi <= (um - cm) * t:
            c_here = (2.0 * cL - xi / t) / 3.0
            h[i] = c_here**2 / g
        elif xi <= S * t:
            h[i] = hm
        else:
            h[i] = hR
    return h


# ─────────────────────────────────────────────────────────────────────────────
# Channel mesh builder (orthogonal / non-orthogonal comparison)
# ─────────────────────────────────────────────────────────────────────────────
def _make_tri_channel_mesh(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    s0: float,
    skew_amp: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a triangular channel mesh over [0,lx]x[0,ly].

    Non-orthogonality is introduced by perturbing interior node x positions,
    while keeping all boundary nodes fixed so BC geometry remains identical.
    """
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    x_base, y_base = np.meshgrid(xs, ys)
    x = x_base.copy()
    y = y_base.copy()

    if skew_amp > 0.0:
        bump = np.sin(np.pi * x_base / lx) * np.sin(np.pi * y_base / ly)
        x += float(skew_amp) * bump

    node_x = x.ravel().astype(np.float64)
    node_y = y.ravel().astype(np.float64)
    node_z = (s0 * (lx - x_base)).ravel().astype(np.float64)

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

    return node_x, node_y, node_z, np.array(cells, dtype=np.int32)


def _channel_bc_edges(
    nx: int, ny: int, q_in: float, s0: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return BC arrays for a channel: inflow left, normal-depth right, walls top/bot."""
    stride = nx + 1
    n0, n1, tp, val = [], [], [], []

    # Left boundary: INFLOW_Q
    for j in range(ny):
        n0.append(j * stride)
        n1.append((j + 1) * stride)
        tp.append(2)
        val.append(float(q_in))

    # Right boundary: NORMAL_DEPTH_SLOPE
    for j in range(ny):
        n0.append(j * stride + nx)
        n1.append((j + 1) * stride + nx)
        tp.append(7)
        val.append(float(s0))

    # Bottom boundary: WALL
    for i in range(nx):
        n0.append(i)
        n1.append(i + 1)
        tp.append(1)
        val.append(0.0)

    # Top boundary: WALL
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


def _manning_normal_depth(q: float, n_mann: float, s0: float) -> float:
    """Wide-rectangular Manning normal depth."""
    return (q * n_mann / np.sqrt(s0)) ** (3.0 / 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# Valid scheme combinations for GPU hydraulics tests
# ─────────────────────────────────────────────────────────────────────────────
VALID_SPATIAL_SCHEMES = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # FO, Fast, MinMod, MC, VanLeer, BJ, WENO3, WENO5, MP5
VALID_TEMPORAL_SCHEMES = [1, 2, 3, 5, 6]    # Euler, RK2, RK3, Graph-RK4, Graph-RK5
GODUNOV_MODES = [0, 1]                       # standard, rollout

# Preferred combinations for quick validation (covers all spatial schemes with RK2)
QUICK_SPATIAL_COMBOS = [(s, 2, 0) for s in VALID_SPATIAL_SCHEMES]  # (spatial, temporal, godunov)

# Full cross-product (spatial × temporal × godunov) for thorough validation
FULL_COMBOS = [
    (s, t, g)
    for s in VALID_SPATIAL_SCHEMES
    for t in VALID_TEMPORAL_SCHEMES
    for g in GODUNOV_MODES
]


# ─────────────────────────────────────────────────────────────────────────────
# CLI runner (no GUI mocks — exercises swe2d.cli.headless_runner.execute_run)
# ─────────────────────────────────────────────────────────────────────────────
def _serialize_and_persist_mesh(
    gpkg_path: str,
    mesh_name: str,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
    cell_nodes: np.ndarray,
    bc_n0: np.ndarray,
    bc_n1: np.ndarray,
    bc_tp: np.ndarray,
    bc_vl: np.ndarray,
) -> None:
    """Build → serialize → persist a mesh into the GPKG.

    ``cell_nodes`` may be either a flat triangulated array (length 3*N)
    or a flat polygon array with companion ``cell_face_offsets`` of length
    N+1 supplied alongside via ``cell_nodes`` paired with ``cell_face_offsets``.

    The CLI reads meshes back via query_mesh_from_gpkg → load_baked_mesh →
    swe2d_deserialize_mesh, so the BLOB must include the BC edges.
    """
    from hydra_swe2d import (
        swe2d_build_mesh, swe2d_build_mesh_poly, swe2d_serialize_mesh, swe2d_mesh_info,
    )
    if cell_nodes.ndim == 2 and cell_nodes.shape[1] == 4:
        # Polygon cells: cell_nodes shape (N, 4) — derive offsets
        n_cells = cell_nodes.shape[0]
        cell_face_offsets = np.arange(0, (n_cells + 1) * 4, 4, dtype=np.int32)
        pm = swe2d_build_mesh_poly(
            node_x, node_y, node_z,
            cell_face_offsets,
            cell_nodes.astype(np.int32).ravel(),
            bc_n0, bc_n1, bc_tp, bc_vl,
        )
    elif cell_nodes.shape[0] % 3 == 0:
        # Triangulated flat array (3*N entries for N triangles)
        pm = swe2d_build_mesh(node_x, node_y, node_z, cell_nodes,
                               bc_n0, bc_n1, bc_tp, bc_vl)
    else:
        # Flat polygon (N*4 entries for N quad cells).  Derive offsets
        # assuming 4 corners per cell.
        n_cells = cell_nodes.shape[0] // 4
        cell_face_offsets = np.arange(0, (n_cells + 1) * 4, 4, dtype=np.int32)
        pm = swe2d_build_mesh_poly(
            node_x, node_y, node_z,
            cell_face_offsets,
            cell_nodes.astype(np.int32),
            bc_n0, bc_n1, bc_tp, bc_vl,
        )
    blob = swe2d_serialize_mesh(pm)
    info = swe2d_mesh_info(pm)
    from swe2d.services.gpkg_persistence_service import persist_baked_mesh
    persist_baked_mesh(
        gpkg_path, mesh_name, blob,
        info["n_nodes"], info["n_cells"], info["n_edges"],
    )


def _run_cli_coupling(
    gpkg_path: str,
    mesh_name: str,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
    cell_nodes: np.ndarray,
    bc_n0: np.ndarray,
    bc_n1: np.ndarray,
    bc_tp: np.ndarray,
    bc_vl: np.ndarray,
    params: dict,
    duration_s: float,
    q_in: float,
    structures_cfg: dict | None = None,
    h0: np.ndarray | None = None,
) -> None:
    """Invoke the headless CLI on a tiny mesh and persist baked results.

    Builds the mesh, bakes it into ``gpkg_path``, then calls
    ``swe2d.cli.headless_runner.execute_run`` with ``params`` overridden for
    ``duration_s``, ``q_in``, and the synthetic drainage network.

    Optional ``structures_cfg`` is added to params as ``structures`` so the
    CLI builds a HydraulicStructureConfig and wires up the structure
    coupling controller path.

    Optional ``h0`` sets the initial water depth on every cell (the CLI
    defaults to all-zeros which keeps the 2D solver dry and makes
    coupling-only assertions noisy).

    No mocks: the run is identical to what `python -m swe2d.cli run` would do.
    """
    # Serialize + persist the synthetic mesh (BC edges included)
    _serialize_and_persist_mesh(
        gpkg_path, mesh_name,
        node_x, node_y, node_z, cell_nodes,
        bc_n0, bc_n1, bc_tp, bc_vl,
    )

    # Determine cell count for drainage inlets.  Supports both triangulated
    # (flat int32) and polygon (N, 4) cell_nodes representations.
    if cell_nodes.ndim == 1:
        ncells = int(cell_nodes.size // 3)
    else:
        ncells = int(cell_nodes.shape[0])
    nodes_x_mean = float(node_x.mean())
    nodes_y_mean = float(node_y.mean())
    inlet_cell = 0
    outlet_cell = max(ncells - 1, 0)

    drainage_cfg = {
        "nodes": [
            {
                "id": "n_in", "type": "inlet",
                "invert": 9.5, "y_max": 12.0, "area": 5.0,
                "surcharge_depth": 1.0, "initial_depth": 0.5,
                "x": nodes_x_mean - 5.0, "y": nodes_y_mean,
            },
            {
                "id": "n_out", "type": "outfall",
                "invert": 5.0, "y_max": 12.0, "area": 5.0,
                "surcharge_depth": 1.0, "initial_depth": 0.0,
                "x": nodes_x_mean + 5.0, "y": nodes_y_mean,
            },
        ],
        "links": [
            {
                "from": "n_in", "to": "n_out",
                "length": 10.0, "diameter": 1.0,
                "roughness": 0.013, "max_flow": -1.0,
            },
        ],
        "inlets": [
            {
                "node_id": "n_in",
                "inlet_cell": inlet_cell,
                "flow_rate": float(q_in),
            },
        ],
        "outfalls": [
            {
                "node_id": "n_out",
                "invert": 5.0,
            },
        ],
    }

    # Override mesh name + run length + add drainage to params.
    # Output/snap intervals default to t_end (1 sample), so force smaller
    # intervals to get a meaningful coupling time series.
    p = dict(params)
    p["mesh"] = mesh_name
    p["params"] = dict(p.get("params", {}))
    p["params"]["duration_s"] = float(duration_s)
    p["params"]["output_interval_s"] = float(p["params"].get("output_interval_s", 1.0))
    p["drainage"] = drainage_cfg
    if structures_cfg is not None:
        p["structures"] = structures_cfg
    if h0 is not None:
        p["params"]["h0"] = np.asarray(h0, dtype=np.float64).tolist()

    # Force SI units so gravity = 9.81 (default) and 1 cell area ≈ 1 m².
    # Avoids USC conversion path in this synthetic test.
    from swe2d import units as _u
    _u.configure(1.0)

    from swe2d.cli.headless_runner import execute_run
    execute_run(
        mesh_gpkg=gpkg_path,
        params=p,
        results_gpkg=gpkg_path,
    )


def _read_coupling_rows(gpkg_path: str, run_id: str | None = None) -> dict:
    """Read swe2d_baked_coupling as {key: (times, values)} decoded from BLOB.

    ``key`` is a (component, object_id, metric) tuple. If ``run_id`` is given,
    restrict to that run; otherwise return all runs' rows keyed by
    ``(run_id, component, object_id, metric)``.
    """
    import sqlite3
    conn = sqlite3.connect(gpkg_path)
    try:
        if run_id is None:
            rows = conn.execute(
                "SELECT run_id, component, object_id, metric, times_blob, values_blob "
                "FROM swe2d_baked_coupling"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT run_id, component, object_id, metric, times_blob, values_blob "
                "FROM swe2d_baked_coupling WHERE run_id=?",
                (run_id,),
            ).fetchall()
    finally:
        conn.close()

    out: dict = {}
    for rid, comp, oid, metric, tb, vb in rows:
        out[(rid, comp, oid, metric)] = (
            np.frombuffer(tb, dtype=np.float64),
            np.frombuffer(vb, dtype=np.float64),
        )
    return out
