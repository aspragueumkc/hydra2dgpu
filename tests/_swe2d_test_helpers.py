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
VALID_SPATIAL_SCHEMES = [0, 1, 2, 3, 4, 6]  # FO, Fast, MinMod, MC, VanLeer, WENO5
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
