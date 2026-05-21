#!/usr/bin/env python3
"""Utilities for converting static 3D geometry into SWE3D patch tensors.

The helper functions in this module are intentionally dependency-light:
- OBJ parsing supports common vertex/face records.
- Geometry is reconstructed on a sub-cell stencil for fractional cut-cell
    porosity and directional open-area tensors.
- Outputs are flat arrays ordered as x-fastest, then y, then z.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


_CUTCELL_SAMPLES_PER_AXIS = 3
_OBSTACLE_METHOD_FRACTIONAL_CUTCELL = "fractional_cutcell"
_OBSTACLE_METHOD_FAVOR1981_POROSITY = "favor1981_porosity"


@dataclass(frozen=True)
class PatchGridSpec:
    """Structured Cartesian patch descriptor."""

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    origin_x: float
    origin_y: float
    origin_z: float


def patch_cell_centers(spec: PatchGridSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x/y/z patch cell-center coordinates."""
    x = spec.origin_x + (np.arange(spec.nx, dtype=np.float64) + 0.5) * float(spec.dx)
    y = spec.origin_y + (np.arange(spec.ny, dtype=np.float64) + 0.5) * float(spec.dy)
    z = spec.origin_z + (np.arange(spec.nz, dtype=np.float64) + 0.5) * float(spec.dz)
    return x, y, z


def _normalize_obstacle_method(method: Optional[str]) -> str:
    raw = str(method or "").strip().lower()
    if not raw:
        return _OBSTACLE_METHOD_FRACTIONAL_CUTCELL

    aliases = {
        _OBSTACLE_METHOD_FRACTIONAL_CUTCELL,
        "cutcell",
        "fractional",
        "fractional_cut_cell",
    }
    if raw in aliases:
        return _OBSTACLE_METHOD_FRACTIONAL_CUTCELL

    porosity_aliases = {
        _OBSTACLE_METHOD_FAVOR1981_POROSITY,
        "favor",
        "favor1981",
        "hirt_nichols_1981",
        "hirt_nichols",
        "porosity",
    }
    if raw in porosity_aliases:
        return _OBSTACLE_METHOD_FAVOR1981_POROSITY

    return _OBSTACLE_METHOD_FRACTIONAL_CUTCELL


def load_obj_mesh(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a triangular mesh from a Wavefront OBJ file.

    Supports:
    - `v x y z`
    - `f i j k ...` (fan triangulated)
    - OBJ index tokens with slash syntax (`i/j/k`)
    - negative face indices
    """
    vertices = []
    faces = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0].lower()

            if tag == "v" and len(parts) >= 4:
                try:
                    vx = float(parts[1])
                    vy = float(parts[2])
                    vz = float(parts[3])
                except Exception:
                    continue
                vertices.append((vx, vy, vz))
                continue

            if tag != "f" or len(parts) < 4:
                continue

            idx = []
            for token in parts[1:]:
                head = token.split("/")[0].strip()
                if not head:
                    continue
                try:
                    vidx = int(head)
                except Exception:
                    continue
                if vidx < 0:
                    vidx = len(vertices) + vidx + 1
                if vidx <= 0:
                    continue
                idx.append(vidx - 1)

            if len(idx) < 3:
                continue

            root = idx[0]
            for i in range(1, len(idx) - 1):
                faces.append((root, idx[i], idx[i + 1]))

    if not vertices:
        raise ValueError(f"OBJ has no vertices: {file_path}")
    if not faces:
        raise ValueError(f"OBJ has no faces: {file_path}")

    vertices_arr = np.asarray(vertices, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int32)

    max_index = int(np.max(faces_arr)) if faces_arr.size > 0 else -1
    if max_index >= vertices_arr.shape[0]:
        raise ValueError(f"OBJ face index out of bounds: {file_path}")

    return vertices_arr, faces_arr


def apply_instance_transform(
    vertices: np.ndarray,
    translation_xyz: Sequence[float],
    yaw_deg: float = 0.0,
    scale_xyz: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Apply scale -> yaw(Z) -> translation transform to mesh vertices."""
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")

    scale = np.asarray(scale_xyz, dtype=np.float64).ravel()
    if scale.size == 1:
        scale = np.repeat(scale[0], 3)
    if scale.size != 3:
        raise ValueError("scale_xyz must have 1 or 3 elements")

    trans = np.asarray(translation_xyz, dtype=np.float64).ravel()
    if trans.size != 3:
        raise ValueError("translation_xyz must have 3 elements")

    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)
    rot = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return (verts * scale[None, :]) @ rot.T + trans[None, :]


def _center_index_bounds(coords: np.ndarray, lo: float, hi: float) -> Optional[Tuple[int, int]]:
    i0 = int(np.searchsorted(coords, float(lo), side="left"))
    i1 = int(np.searchsorted(coords, float(hi), side="right")) - 1
    i0 = max(0, min(i0, int(coords.size) - 1))
    i1 = max(0, min(i1, int(coords.size) - 1))
    if i1 < i0:
        return None
    return i0, i1


def _points_inside_closed_mesh_once(
    points: np.ndarray,
    triangles: np.ndarray,
    eps: float,
    y_jitter: float,
    z_jitter: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    tris = np.asarray(triangles, dtype=np.float64)

    n_points = int(pts.shape[0]) if pts.ndim == 2 else 0
    if n_points <= 0:
        return np.zeros(0, dtype=bool)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or tris.shape[0] <= 0:
        return np.zeros(n_points, dtype=bool)

    px = pts[:, 0]
    py = pts[:, 1] + float(y_jitter)
    pz = pts[:, 2] + float(z_jitter)

    py_min = float(np.min(py))
    py_max = float(np.max(py))
    pz_min = float(np.min(pz))
    pz_max = float(np.max(pz))

    counts = np.zeros(n_points, dtype=np.int32)

    for tri in tris:
        v0 = tri[0]
        v1 = tri[1]
        v2 = tri[2]

        tri_ymin = float(min(v0[1], v1[1], v2[1]))
        tri_ymax = float(max(v0[1], v1[1], v2[1]))
        tri_zmin = float(min(v0[2], v1[2], v2[2]))
        tri_zmax = float(max(v0[2], v1[2], v2[2]))
        if tri_ymax < py_min or tri_ymin > py_max or tri_zmax < pz_min or tri_zmin > pz_max:
            continue

        e1x = float(v1[0] - v0[0])
        e1y = float(v1[1] - v0[1])
        e1z = float(v1[2] - v0[2])
        e2x = float(v2[0] - v0[0])
        e2y = float(v2[1] - v0[1])
        e2z = float(v2[2] - v0[2])

        # dot(e1, cross([1,0,0], e2))
        a = e1z * e2y - e1y * e2z
        if abs(a) <= eps:
            continue

        inv_a = 1.0 / a

        sx = px - float(v0[0])
        sy = py - float(v0[1])
        sz = pz - float(v0[2])

        u = inv_a * (sy * (-e2z) + sz * e2y)

        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x

        v = inv_a * qx
        t = inv_a * (e2x * qx + e2y * qy + e2z * qz)

        # Strict barycentric bounds avoid double-counting shared triangle edges.
        hit = (u > eps) & (v > eps) & ((u + v) < (1.0 - eps)) & (t > eps)
        counts += hit.astype(np.int32)

    return (counts & 1) == 1


def _points_inside_closed_mesh(points: np.ndarray, triangles: np.ndarray, eps: float = 1.0e-10) -> np.ndarray:
    """Return a boolean mask for points inside a closed triangle mesh.

    Uses parity ray casting along +x with a Moller-Trumbore style intersection
    test vectorized over points and looped over triangles. The test is evaluated
    with small jitter variants and majority-voted to reduce edge-hit artifacts.
    """
    pts = np.asarray(points, dtype=np.float64)
    n_points = int(pts.shape[0]) if pts.ndim == 2 else 0
    if n_points <= 0:
        return np.zeros(0, dtype=bool)

    jitters = (
        (1.0e-6, 2.0e-6),
        (3.0e-6, -2.0e-6),
        (-2.0e-6, 4.0e-6),
    )
    votes = np.zeros(n_points, dtype=np.int32)
    for jy, jz in jitters:
        inside = _points_inside_closed_mesh_once(pts, triangles, eps=eps, y_jitter=jy, z_jitter=jz)
        votes += inside.astype(np.int32)
    return votes >= 2


def _normalize_mesh_item(item: object) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Normalize mesh item payloads.

    Supported item formats:
    - (vertices, faces)
    - (vertices, faces, outside_point)
    - {"vertices": ..., "faces": ..., "outside_point": ...}

    Backward compatibility:
    - mapping key `inside_point` is still accepted and treated as outside-point
      seed input.
    """
    seed_point = None
    if isinstance(item, Mapping):
        verts = item.get("vertices")
        faces = item.get("faces")
        seed_point = item.get("outside_point", item.get("inside_point"))
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
        verts = item[0]
        faces = item[1]
        if len(item) >= 3:
            seed_point = item[2]
    else:
        raise ValueError("mesh item must be tuple/list or mapping")

    verts_arr = np.asarray(verts, dtype=np.float64)
    faces_arr = np.asarray(faces, dtype=np.int32)

    seed_arr: Optional[np.ndarray] = None
    if seed_point is not None:
        arr = np.asarray(seed_point, dtype=np.float64).ravel()
        if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
            seed_arr = np.asarray(arr[:3], dtype=np.float64)

    return verts_arr, faces_arr, seed_arr


def _nearest_center_index(coords: np.ndarray, value: float) -> int:
    idx = int(np.searchsorted(coords, float(value), side="left"))
    idx = max(0, min(idx, int(coords.size) - 1))
    if idx > 0 and abs(coords[idx - 1] - value) <= abs(coords[idx] - value):
        idx -= 1
    return int(idx)


def _surface_voxel_mask_from_triangles(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_centers: np.ndarray,
    triangles: np.ndarray,
    shell_thickness: float,
) -> np.ndarray:
    """Return a shell mask of voxels that intersect triangle surfaces."""
    mask = np.zeros((z_centers.size, y_centers.size, x_centers.size), dtype=bool)
    tris = np.asarray(triangles, dtype=np.float64)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or tris.shape[0] <= 0:
        return mask

    tol = 1.0e-8
    for tri in tris:
        v0 = tri[0]
        v1 = tri[1]
        v2 = tri[2]

        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        nrm = float(np.linalg.norm(normal))
        if nrm <= 1.0e-14:
            continue
        n_hat = normal / nrm

        bmin = np.min(tri, axis=0) - shell_thickness
        bmax = np.max(tri, axis=0) + shell_thickness

        xb = _center_index_bounds(x_centers, float(bmin[0]), float(bmax[0]))
        yb = _center_index_bounds(y_centers, float(bmin[1]), float(bmax[1]))
        zb = _center_index_bounds(z_centers, float(bmin[2]), float(bmax[2]))
        if xb is None or yb is None or zb is None:
            continue

        i0, i1 = xb
        j0, j1 = yb
        k0, k1 = zb
        xs = x_centers[i0 : i1 + 1]
        ys = y_centers[j0 : j1 + 1]
        zs = z_centers[k0 : k1 + 1]
        if xs.size <= 0 or ys.size <= 0 or zs.size <= 0:
            continue

        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        points = np.empty((xx.size, 3), dtype=np.float64)
        points[:, 0] = xx.ravel(order="C")
        points[:, 1] = yy.ravel(order="C")
        points[:, 2] = zz.ravel(order="C")

        rel = points - v0[None, :]
        signed_dist = rel @ n_hat
        near = np.abs(signed_dist) <= shell_thickness
        if not np.any(near):
            continue

        proj = points[near] - signed_dist[near, None] * n_hat[None, :]
        c = proj - v0[None, :]

        dot00 = float(np.dot(e1, e1))
        dot01 = float(np.dot(e1, e2))
        dot11 = float(np.dot(e2, e2))
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) <= 1.0e-14:
            continue
        inv_denom = 1.0 / denom

        dot20 = c @ e1
        dot21 = c @ e2
        u = (dot11 * dot20 - dot01 * dot21) * inv_denom
        v = (dot00 * dot21 - dot01 * dot20) * inv_denom
        in_tri = (u >= -tol) & (v >= -tol) & ((u + v) <= (1.0 + tol))
        if not np.any(in_tri):
            continue

        near_flat = np.nonzero(near)[0][in_tri]
        nxy = int(ys.size * xs.size)
        kk = near_flat // nxy
        rem = near_flat % nxy
        jj = rem // int(xs.size)
        ii = rem % int(xs.size)
        mask[k0 + kk, j0 + jj, i0 + ii] = True

    return mask


def _dilate_mask_6(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        expanded = out.copy()
        expanded[1:, :, :] |= out[:-1, :, :]
        expanded[:-1, :, :] |= out[1:, :, :]
        expanded[:, 1:, :] |= out[:, :-1, :]
        expanded[:, :-1, :] |= out[:, 1:, :]
        expanded[:, :, 1:] |= out[:, :, :-1]
        expanded[:, :, :-1] |= out[:, :, 1:]
        out = expanded
    return out


def _flood_fill_6(blocked: np.ndarray, seed_kji: Tuple[int, int, int]) -> np.ndarray:
    blocked_arr = np.asarray(blocked, dtype=bool)
    visited = np.zeros_like(blocked_arr, dtype=bool)

    nk, nj, ni = blocked_arr.shape
    sk, sj, si = [int(v) for v in seed_kji]
    if sk < 0 or sk >= nk or sj < 0 or sj >= nj or si < 0 or si >= ni:
        return visited
    if blocked_arr[sk, sj, si]:
        return visited

    q = deque([(sk, sj, si)])
    visited[sk, sj, si] = True
    while q:
        k, j, i = q.popleft()
        for dk, dj, di in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            kk = k + dk
            jj = j + dj
            ii = i + di
            if kk < 0 or kk >= nk or jj < 0 or jj >= nj or ii < 0 or ii >= ni:
                continue
            if blocked_arr[kk, jj, ii] or visited[kk, jj, ii]:
                continue
            visited[kk, jj, ii] = True
            q.append((kk, jj, ii))
    return visited


def _inside_mask_from_outside_seed_fill(
    sub_phi: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_centers: np.ndarray,
    triangles: np.ndarray,
    outside_point_xyz: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> Tuple[np.ndarray, bool]:
    """Return inside-cell mask using an outside-fluid seed point.

    Returns `(inside_mask, leak_detected)` where `inside_mask` has the same
    shape as `sub_phi` and marks cells to set solid.
    """
    open_mask = np.asarray(sub_phi, dtype=np.float64) > 0.5
    inside_mask = np.zeros_like(open_mask, dtype=bool)
    if not np.any(open_mask):
        return inside_mask, False

    shell_thickness = max(0.20 * min(float(dx), float(dy), float(dz)), 1.0e-9)
    surface = _surface_voxel_mask_from_triangles(
        x_centers=x_centers,
        y_centers=y_centers,
        z_centers=z_centers,
        triangles=triangles,
        shell_thickness=shell_thickness,
    )

    def _pick_seed(blocked_arr: np.ndarray, seed_guess: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
        sk0, sj0, si0 = seed_guess
        if not blocked_arr[sk0, sj0, si0]:
            return int(sk0), int(sj0), int(si0)
        candidates = np.argwhere(~blocked_arr)
        if candidates.size <= 0:
            return None
        cand_xyz = np.empty((candidates.shape[0], 3), dtype=np.float64)
        cand_xyz[:, 0] = x_centers[candidates[:, 2]]
        cand_xyz[:, 1] = y_centers[candidates[:, 1]]
        cand_xyz[:, 2] = z_centers[candidates[:, 0]]
        d2 = np.sum((cand_xyz - outside_point_xyz[None, :]) ** 2, axis=1)
        best = candidates[int(np.argmin(d2))]
        return int(best[0]), int(best[1]), int(best[2])

    def _touches_boundary(flooded_arr: np.ndarray) -> bool:
        return bool(
            np.any(flooded_arr[0, :, :])
            or np.any(flooded_arr[-1, :, :])
            or np.any(flooded_arr[:, 0, :])
            or np.any(flooded_arr[:, -1, :])
            or np.any(flooded_arr[:, :, 0])
            or np.any(flooded_arr[:, :, -1])
        )

    def _inside_from_outside_flood(flooded_arr: np.ndarray) -> Tuple[np.ndarray, bool]:
        inside_arr = open_mask & (~flooded_arr)
        return inside_arr, _touches_boundary(flooded_arr)

    blocked = surface | (~open_mask)

    si = _nearest_center_index(x_centers, float(outside_point_xyz[0]))
    sj = _nearest_center_index(y_centers, float(outside_point_xyz[1]))
    sk = _nearest_center_index(z_centers, float(outside_point_xyz[2]))

    seed = _pick_seed(blocked, (sk, sj, si))
    if seed is None:
        return inside_mask, False

    flooded = _flood_fill_6(blocked, seed)
    if not np.any(flooded):
        return inside_mask, False

    inside_mask, touches_boundary = _inside_from_outside_flood(flooded)
    outside_seed_invalid = not touches_boundary

    if (outside_seed_invalid or (not np.any(inside_mask))) and np.any(surface):
        blocked_dilated = _dilate_mask_6(surface, iterations=1) | (~open_mask)
        seed_dilated = _pick_seed(blocked_dilated, seed)
        if seed_dilated is not None:
            flooded_dilated = _flood_fill_6(blocked_dilated, seed_dilated)
            if np.any(flooded_dilated):
                inside_dilated, touches_boundary_dilated = _inside_from_outside_flood(flooded_dilated)
                if touches_boundary_dilated and np.any(inside_dilated):
                    inside_mask = inside_dilated
                    outside_seed_invalid = False

    if outside_seed_invalid:
        return inside_mask, True

    return inside_mask, False


def _subcell_centers_from_coarse_centers(
    centers: np.ndarray,
    dxyz: float,
    samples_per_axis: int,
) -> np.ndarray:
    centers_arr = np.asarray(centers, dtype=np.float64)
    r = max(1, int(samples_per_axis))
    offsets = ((np.arange(r, dtype=np.float64) + 0.5) / float(r) - 0.5) * float(dxyz)
    sub = centers_arr[:, None] + offsets[None, :]
    return sub.reshape(-1)


def _coarse_mean_from_fine(
    fine: np.ndarray,
    samples_per_axis: int,
) -> np.ndarray:
    r = max(1, int(samples_per_axis))
    arr = np.asarray(fine, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("fine array must be 3D")
    nzf, nyf, nxf = arr.shape
    if (nzf % r) != 0 or (nyf % r) != 0 or (nxf % r) != 0:
        raise ValueError("fine array shape must be divisible by samples_per_axis")
    nz = nzf // r
    ny = nyf // r
    nx = nxf // r
    reshaped = arr.reshape(nz, r, ny, r, nx, r)
    return np.mean(reshaped, axis=(1, 3, 5))


def _coarse_face_open_from_fine_fluid(
    fluid_fine: np.ndarray,
    samples_per_axis: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate coarse face-open fractions from fine fluid occupancy.

    This is a FAVOR-like porosity approximation: per coarse cell, area-open
    fractions are reconstructed from fluid occupancy sampled near opposing face
    planes and averaged per face-normal direction.
    """
    r = max(1, int(samples_per_axis))
    arr = np.asarray(fluid_fine, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("fluid_fine must be 3D")

    nzf, nyf, nxf = arr.shape
    if (nzf % r) != 0 or (nyf % r) != 0 or (nxf % r) != 0:
        raise ValueError("fine array shape must be divisible by samples_per_axis")

    nz = nzf // r
    ny = nyf // r
    nx = nxf // r
    reshaped = arr.reshape(nz, r, ny, r, nx, r)

    x_minus = np.mean(reshaped[..., 0], axis=(1, 3))
    x_plus = np.mean(reshaped[..., r - 1], axis=(1, 3))
    ax = 0.5 * (x_minus + x_plus)

    y_minus = np.mean(reshaped[:, :, :, 0, :, :], axis=(1, 4))
    y_plus = np.mean(reshaped[:, :, :, r - 1, :, :], axis=(1, 4))
    ay = 0.5 * (y_minus + y_plus)

    z_minus = np.mean(reshaped[:, 0, :, :, :, :], axis=(2, 4))
    z_plus = np.mean(reshaped[:, r - 1, :, :, :, :], axis=(2, 4))
    az = 0.5 * (z_minus + z_plus)

    np.clip(ax, 0.0, 1.0, out=ax)
    np.clip(ay, 0.0, 1.0, out=ay)
    np.clip(az, 0.0, 1.0, out=az)
    return ax, ay, az


def _fractional_mesh_fluid_volume(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    z_centers: np.ndarray,
    triangles: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    samples_per_axis: int,
    outside_point_xyz: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, bool]:
    """Return per-cell fluid fraction for one mesh item on a coarse subdomain.

    Returns `(phi_mesh, ax_mesh, ay_mesh, az_mesh, used_seed_fill, seed_leak)`
    where `phi_mesh` is coarse fluid volume fraction and `ax/ay/az` are
    per-cell directional face-open fractions in [0,1].
    """
    r = max(1, int(samples_per_axis))

    if x_centers.size <= 0 or y_centers.size <= 0 or z_centers.size <= 0:
        shape = (z_centers.size, y_centers.size, x_centers.size)
        ones = np.ones(shape, dtype=np.float64)
        return ones, ones.copy(), ones.copy(), ones.copy(), False, False

    x_sub = _subcell_centers_from_coarse_centers(x_centers, dx, r)
    y_sub = _subcell_centers_from_coarse_centers(y_centers, dy, r)
    z_sub = _subcell_centers_from_coarse_centers(z_centers, dz, r)

    nzf = int(z_sub.size)
    nyf = int(y_sub.size)
    nxf = int(x_sub.size)
    n_fine = nzf * nyf * nxf

    if n_fine <= 0:
        shape = (z_centers.size, y_centers.size, x_centers.size)
        ones = np.ones(shape, dtype=np.float64)
        return ones, ones.copy(), ones.copy(), ones.copy(), False, False

    used_seed_fill = False
    seed_leak = False

    inside_fine = None
    if outside_point_xyz is not None and np.all(np.isfinite(outside_point_xyz)):
        seed_inside, leak = _inside_mask_from_outside_seed_fill(
            sub_phi=np.ones((nzf, nyf, nxf), dtype=np.float64),
            x_centers=x_sub,
            y_centers=y_sub,
            z_centers=z_sub,
            triangles=triangles,
            outside_point_xyz=np.asarray(outside_point_xyz, dtype=np.float64),
            dx=float(dx) / float(r),
            dy=float(dy) / float(r),
            dz=float(dz) / float(r),
        )
        seed_leak = bool(leak)
        if np.any(seed_inside):
            inside_fine = np.asarray(seed_inside, dtype=bool)
            used_seed_fill = True

    if inside_fine is None:
        zz, yy, xx = np.meshgrid(z_sub, y_sub, x_sub, indexing="ij")
        points = np.empty((n_fine, 3), dtype=np.float64)
        points[:, 0] = xx.ravel(order="C")
        points[:, 1] = yy.ravel(order="C")
        points[:, 2] = zz.ravel(order="C")
        inside_flat = _points_inside_closed_mesh(points, triangles)
        inside_fine = inside_flat.reshape((nzf, nyf, nxf), order="C")

    fluid_fine = (~inside_fine).astype(np.float64)
    phi_mesh = _coarse_mean_from_fine(fluid_fine, r)
    ax_mesh, ay_mesh, az_mesh = _coarse_face_open_from_fine_fluid(fluid_fine, r)
    np.clip(phi_mesh, 0.0, 1.0, out=phi_mesh)
    np.clip(ax_mesh, 0.0, 1.0, out=ax_mesh)
    np.clip(ay_mesh, 0.0, 1.0, out=ay_mesh)
    np.clip(az_mesh, 0.0, 1.0, out=az_mesh)
    return phi_mesh, ax_mesh, ay_mesh, az_mesh, used_seed_fill, seed_leak


def _compute_face_open_area_tensors(phi3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.asarray(phi3, dtype=np.float64)
    ax3 = np.clip(phi.copy(), 0.0, 1.0)
    ay3 = np.clip(phi.copy(), 0.0, 1.0)
    az3 = np.clip(phi.copy(), 0.0, 1.0)

    # Pairwise face openness is limited by the more obstructed adjacent cell,
    # then reflected back to each cell tensor slot. This keeps directional
    # openness continuous for fractional cut-cells.
    if phi.shape[2] > 1:
        face_x = np.minimum(phi[:, :, :-1], phi[:, :, 1:])
        ax3[:, :, :-1] = np.minimum(ax3[:, :, :-1], face_x)
        ax3[:, :, 1:] = np.minimum(ax3[:, :, 1:], face_x)

    if phi.shape[1] > 1:
        face_y = np.minimum(phi[:, :-1, :], phi[:, 1:, :])
        ay3[:, :-1, :] = np.minimum(ay3[:, :-1, :], face_y)
        ay3[:, 1:, :] = np.minimum(ay3[:, 1:, :], face_y)

    if phi.shape[0] > 1:
        face_z = np.minimum(phi[:-1, :, :], phi[1:, :, :])
        az3[:-1, :, :] = np.minimum(az3[:-1, :, :], face_z)
        az3[1:, :, :] = np.minimum(az3[1:, :, :], face_z)

    np.clip(ax3, 0.0, 1.0, out=ax3)
    np.clip(ay3, 0.0, 1.0, out=ay3)
    np.clip(az3, 0.0, 1.0, out=az3)
    return ax3, ay3, az3


def build_static_geometry_tensors(
    spec: PatchGridSpec,
    mesh_items: Sequence[object],
    terrain_elevation: Optional[np.ndarray] = None,
    cutcell_samples_per_axis: int = _CUTCELL_SAMPLES_PER_AXIS,
    obstacle_method: str = _OBSTACLE_METHOD_FRACTIONAL_CUTCELL,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Build phi/ax/ay/az arrays for static sub-grid solids.

    Args:
        spec: Patch grid descriptor.
        mesh_items: Sequence of `(vertices, faces)` tuples. Vertices are expected
            in world coordinates. Optional tuple third item / mapping key
            `outside_point` can be used to seed outside-fluid flood
            classification for non-airtight meshes (legacy `inside_point`
            mapping key is also accepted for compatibility).
        terrain_elevation: Optional terrain bed elevation grid `(ny, nx)` at each
            x/y patch column center.
        cutcell_samples_per_axis: Sub-cell resolution used for fractional
            cut-cell reconstruction (minimum 1).
                obstacle_method: Obstacle tensor reconstruction approach.
                        - `fractional_cutcell` (default): volume fraction + pair-min face
                            continuity from phi.
                        - `favor1981_porosity`: volume fraction + direct face-open sampling
                            (FAVOR-like approximation).

    Returns:
        phi, ax, ay, az (all flat float64 arrays of length `nx*ny*nz`) and
        summary diagnostics.
    """
    if spec.nx <= 0 or spec.ny <= 0 or spec.nz <= 0:
        raise ValueError("Patch dimensions must be positive")

    method = _normalize_obstacle_method(obstacle_method)
    cutcell_samples = max(1, int(cutcell_samples_per_axis))

    phi3 = np.ones((spec.nz, spec.ny, spec.nx), dtype=np.float64)
    x_centers, y_centers, z_centers = patch_cell_centers(spec)
    ax3 = np.ones_like(phi3)
    ay3 = np.ones_like(phi3)
    az3 = np.ones_like(phi3)

    terrain_valid_columns = 0
    if terrain_elevation is not None:
        terrain = np.asarray(terrain_elevation, dtype=np.float64)
        if terrain.ndim == 1 and terrain.size == spec.nx * spec.ny:
            terrain = terrain.reshape((spec.ny, spec.nx))
        if terrain.shape != (spec.ny, spec.nx):
            raise ValueError("terrain_elevation must have shape (ny, nx)")

        terrain_valid = np.isfinite(terrain)
        terrain_valid_columns = int(np.count_nonzero(terrain_valid))
        terrain_solid = terrain_valid[None, :, :] & (z_centers[:, None, None] <= terrain[None, :, :])
        phi3[terrain_solid] = 0.0

    if method == _OBSTACLE_METHOD_FAVOR1981_POROSITY:
        # Terrain solids are binary occupancy, so pair-min continuity is a
        # stable baseline; mesh solids below can further reduce these tensors.
        ax3, ay3, az3 = _compute_face_open_area_tensors(phi3)

    terrain_only_solid_cells = int(np.count_nonzero(phi3 < 0.5))

    mesh_count_used = 0
    mesh_seed_instances_requested = 0
    mesh_seed_instances_used = 0
    mesh_seed_leak_fallbacks = 0
    for item in (mesh_items or ()):  # type: ignore[assignment]
        try:
            verts, fcs, outside_point = _normalize_mesh_item(item)
        except Exception:
            continue

        if verts.ndim != 2 or verts.shape[1] != 3:
            continue
        if fcs.ndim != 2 or fcs.shape[1] != 3 or fcs.shape[0] <= 0:
            continue

        triangles = verts[fcs]
        tri_all = triangles.reshape(-1, 3)
        bmin = np.min(tri_all, axis=0)
        bmax = np.max(tri_all, axis=0)

        xb = _center_index_bounds(x_centers, float(bmin[0]), float(bmax[0]))
        yb = _center_index_bounds(y_centers, float(bmin[1]), float(bmax[1]))
        zb = _center_index_bounds(z_centers, float(bmin[2]), float(bmax[2]))
        if xb is None or yb is None or zb is None:
            continue

        i0, i1 = xb
        j0, j1 = yb
        k0, k1 = zb
        # Keep a one-cell halo so outside-seed classification can resolve
        # near-mesh exterior fluid cells.
        i0 = max(0, i0 - 1)
        j0 = max(0, j0 - 1)
        k0 = max(0, k0 - 1)
        i1 = min(spec.nx - 1, i1 + 1)
        j1 = min(spec.ny - 1, j1 + 1)
        k1 = min(spec.nz - 1, k1 + 1)
        sub_phi = phi3[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]
        open_mask = sub_phi > 0.5
        if not np.any(open_mask):
            continue

        used_seed_fill = False
        if outside_point is not None and np.all(np.isfinite(outside_point)):
            mesh_seed_instances_requested += 1
        x_sub = x_centers[i0 : i1 + 1]
        y_sub = y_centers[j0 : j1 + 1]
        z_sub = z_centers[k0 : k1 + 1]
        phi_mesh, ax_mesh, ay_mesh, az_mesh, used_seed_fill, seed_leak = _fractional_mesh_fluid_volume(
            x_centers=x_sub,
            y_centers=y_sub,
            z_centers=z_sub,
            triangles=triangles,
            dx=float(spec.dx),
            dy=float(spec.dy),
            dz=float(spec.dz),
            samples_per_axis=cutcell_samples,
            outside_point_xyz=(np.asarray(outside_point, dtype=np.float64) if outside_point is not None else None),
        )
        if seed_leak:
            mesh_seed_leak_fallbacks += 1
        if used_seed_fill:
            mesh_seed_instances_used += 1

        # Union solids conservatively: fluid fraction can only decrease.
        np.minimum(sub_phi, phi_mesh, out=sub_phi)
        if method == _OBSTACLE_METHOD_FAVOR1981_POROSITY:
            sub_ax = ax3[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]
            sub_ay = ay3[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]
            sub_az = az3[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]
            np.minimum(sub_ax, ax_mesh, out=sub_ax)
            np.minimum(sub_ay, ay_mesh, out=sub_ay)
            np.minimum(sub_az, az_mesh, out=sub_az)
        mesh_count_used += 1

    np.clip(phi3, 0.0, 1.0, out=phi3)

    if method != _OBSTACLE_METHOD_FAVOR1981_POROSITY:
        ax3, ay3, az3 = _compute_face_open_area_tensors(phi3)
    else:
        # Keep area tensors physically bounded by local open volume cap.
        np.minimum(ax3, np.clip(phi3, 0.0, 1.0), out=ax3)
        np.minimum(ay3, np.clip(phi3, 0.0, 1.0), out=ay3)
        np.minimum(az3, np.clip(phi3, 0.0, 1.0), out=az3)
        np.clip(ax3, 0.0, 1.0, out=ax3)
        np.clip(ay3, 0.0, 1.0, out=ay3)
        np.clip(az3, 0.0, 1.0, out=az3)

    phi = np.asarray(phi3, dtype=np.float64).ravel(order="C")
    ax = np.asarray(ax3, dtype=np.float64).ravel(order="C")
    ay = np.asarray(ay3, dtype=np.float64).ravel(order="C")
    az = np.asarray(az3, dtype=np.float64).ravel(order="C")

    solid_cells = int(np.count_nonzero(phi < 0.5))
    n_cells = int(phi.size)
    diagnostics = {
        "n_cells": float(n_cells),
        "solid_cells": float(solid_cells),
        "solid_fraction": float(solid_cells / max(n_cells, 1)),
        "terrain_solid_cells": float(terrain_only_solid_cells),
        "mesh_solid_cells": float(max(0, solid_cells - terrain_only_solid_cells)),
        "terrain_valid_columns": float(terrain_valid_columns),
        "mesh_instances_used": float(mesh_count_used),
        "mesh_seed_instances_requested": float(mesh_seed_instances_requested),
        "mesh_seed_instances_used": float(mesh_seed_instances_used),
        "mesh_seed_leak_fallbacks": float(mesh_seed_leak_fallbacks),
        "cutcell_samples_per_axis": float(cutcell_samples),
        "obstacle_method_favor1981": float(1.0 if method == _OBSTACLE_METHOD_FAVOR1981_POROSITY else 0.0),
        "phi_open_mean": float(np.mean(phi) if n_cells > 0 else 1.0),
        "ax_open_mean": float(np.mean(ax) if n_cells > 0 else 1.0),
        "ay_open_mean": float(np.mean(ay) if n_cells > 0 else 1.0),
        "az_open_mean": float(np.mean(az) if n_cells > 0 else 1.0),
    }

    return phi, ax, ay, az, diagnostics


def write_solid_voxels_obj(
    spec: PatchGridSpec,
    phi: np.ndarray,
    file_path: str,
    solid_threshold: float = 0.5,
) -> Dict[str, float]:
    """Export a voxelized solid representation from phi to OBJ.

    The exported mesh is the exposed-face shell of cells where `phi < solid_threshold`.
    """
    phi_arr = np.asarray(phi, dtype=np.float64).ravel(order="C")
    expected = int(spec.nx) * int(spec.ny) * int(spec.nz)
    if phi_arr.size != expected:
        raise ValueError(
            f"phi size mismatch: expected {expected}, got {phi_arr.size}"
        )

    phi3 = phi_arr.reshape((spec.nz, spec.ny, spec.nx), order="C")
    solid = phi3 < float(solid_threshold)

    out_dir = os.path.dirname(os.path.abspath(file_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not np.any(solid):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# No solid voxels for requested threshold\n")
        return {
            "solid_cells": 0.0,
            "vertices": 0.0,
            "faces": 0.0,
        }

    vertex_index: Dict[Tuple[float, float, float], int] = {}
    vertices: list[Tuple[float, float, float]] = []
    faces: list[Tuple[int, int, int, int]] = []

    def _add_vertex(v: Tuple[float, float, float]) -> int:
        key = (float(v[0]), float(v[1]), float(v[2]))
        idx = vertex_index.get(key)
        if idx is not None:
            return idx
        vertices.append(key)
        vid = len(vertices)
        vertex_index[key] = vid
        return vid

    def _add_face(corners: Sequence[Tuple[float, float, float]]) -> None:
        ids = [_add_vertex(tuple(c)) for c in corners]
        faces.append((ids[0], ids[1], ids[2], ids[3]))

    nx = int(spec.nx)
    ny = int(spec.ny)
    nz = int(spec.nz)
    dx = float(spec.dx)
    dy = float(spec.dy)
    dz = float(spec.dz)
    ox = float(spec.origin_x)
    oy = float(spec.origin_y)
    oz = float(spec.origin_z)

    for k in range(nz):
        z0 = oz + k * dz
        z1 = z0 + dz
        for j in range(ny):
            y0 = oy + j * dy
            y1 = y0 + dy
            for i in range(nx):
                if not solid[k, j, i]:
                    continue
                x0 = ox + i * dx
                x1 = x0 + dx

                # x- face
                if i == 0 or (not solid[k, j, i - 1]):
                    _add_face(((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)))
                # x+ face
                if i == (nx - 1) or (not solid[k, j, i + 1]):
                    _add_face(((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)))
                # y- face
                if j == 0 or (not solid[k, j - 1, i]):
                    _add_face(((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)))
                # y+ face
                if j == (ny - 1) or (not solid[k, j + 1, i]):
                    _add_face(((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)))
                # z- face
                if k == 0 or (not solid[k - 1, j, i]):
                    _add_face(((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)))
                # z+ face
                if k == (nz - 1) or (not solid[k + 1, j, i]):
                    _add_face(((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)))

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# SWE3D voxel solid export\n")
        f.write(f"# solid_threshold={float(solid_threshold):.6g}\n")
        for vx, vy, vz in vertices:
            f.write(f"v {vx:.17g} {vy:.17g} {vz:.17g}\n")
        for a, b, c, d in faces:
            f.write(f"f {a} {b} {c} {d}\n")

    return {
        "solid_cells": float(int(np.count_nonzero(solid))),
        "vertices": float(len(vertices)),
        "faces": float(len(faces)),
    }


def write_fluid_voxels_obj(
    spec: PatchGridSpec,
    phi: np.ndarray,
    file_path: str,
    fluid_threshold: float = 0.5,
) -> Dict[str, float]:
    """Export a voxelized fluid-domain shell from phi to OBJ.

    The exported mesh is the exposed-face shell of cells where
    ``phi >= fluid_threshold``.
    """
    phi_arr = np.asarray(phi, dtype=np.float64).ravel(order="C")
    expected = int(spec.nx) * int(spec.ny) * int(spec.nz)
    if phi_arr.size != expected:
        raise ValueError(
            f"phi size mismatch: expected {expected}, got {phi_arr.size}"
        )

    phi3 = phi_arr.reshape((spec.nz, spec.ny, spec.nx), order="C")
    fluid = phi3 >= float(fluid_threshold)

    out_dir = os.path.dirname(os.path.abspath(file_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not np.any(fluid):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# No fluid voxels for requested threshold\n")
        return {
            "fluid_cells": 0.0,
            "vertices": 0.0,
            "faces": 0.0,
        }

    vertex_index: Dict[Tuple[float, float, float], int] = {}
    vertices: list[Tuple[float, float, float]] = []
    faces: list[Tuple[int, int, int, int]] = []

    def _add_vertex(v: Tuple[float, float, float]) -> int:
        key = (float(v[0]), float(v[1]), float(v[2]))
        idx = vertex_index.get(key)
        if idx is not None:
            return idx
        vertices.append(key)
        vid = len(vertices)
        vertex_index[key] = vid
        return vid

    def _add_face(corners: Sequence[Tuple[float, float, float]]) -> None:
        ids = [_add_vertex(tuple(c)) for c in corners]
        faces.append((ids[0], ids[1], ids[2], ids[3]))

    nx = int(spec.nx)
    ny = int(spec.ny)
    nz = int(spec.nz)
    dx = float(spec.dx)
    dy = float(spec.dy)
    dz = float(spec.dz)
    ox = float(spec.origin_x)
    oy = float(spec.origin_y)
    oz = float(spec.origin_z)

    for k in range(nz):
        z0 = oz + k * dz
        z1 = z0 + dz
        for j in range(ny):
            y0 = oy + j * dy
            y1 = y0 + dy
            for i in range(nx):
                if not fluid[k, j, i]:
                    continue
                x0 = ox + i * dx
                x1 = x0 + dx

                if i == 0 or (not fluid[k, j, i - 1]):
                    _add_face(((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)))
                if i == (nx - 1) or (not fluid[k, j, i + 1]):
                    _add_face(((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)))
                if j == 0 or (not fluid[k, j - 1, i]):
                    _add_face(((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)))
                if j == (ny - 1) or (not fluid[k, j + 1, i]):
                    _add_face(((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)))
                if k == 0 or (not fluid[k - 1, j, i]):
                    _add_face(((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)))
                if k == (nz - 1) or (not fluid[k + 1, j, i]):
                    _add_face(((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)))

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# SWE3D voxel fluid export\n")
        f.write(f"# fluid_threshold={float(fluid_threshold):.6g}\n")
        for vx, vy, vz in vertices:
            f.write(f"v {vx:.17g} {vy:.17g} {vz:.17g}\n")
        for a, b, c, d in faces:
            f.write(f"f {a} {b} {c} {d}\n")

    return {
        "fluid_cells": float(int(np.count_nonzero(fluid))),
        "vertices": float(len(vertices)),
        "faces": float(len(faces)),
    }
