"""swe2d_velocity_layer.py

Sprint 3 helper utilities for velocity-vector overlays.

This module is intentionally Qt-free so it can be tested independently.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import sqlite3
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from swe2d.results.db_utils import (
        open_ro as _open_ro,
        table_columns as _table_columns,
        table_exists as _table_exists,
    )
except Exception:
    from .db_utils import (
        open_ro as _open_ro,
        table_columns as _table_columns,
        table_exists as _table_exists,
    )


@dataclass
class VelocitySnapshot:
    run_id: str
    t_s: float
    cell_id: np.ndarray
    h: np.ndarray
    hu: np.ndarray
    hv: np.ndarray
    source: str = "cell_momentum"


class VelocityVectorBuilder:
    """Load cached mesh snapshots and derive vector/glyph styling fields."""

    def __init__(self, max_cache_entries: int = 16):
        self._cache: "OrderedDict[Tuple[str, str, str, float], VelocitySnapshot]" = OrderedDict()
        self._max_cache_entries = max(1, int(max_cache_entries))

    def load_available_timesteps(self, gpkg_path: str, run_id: str, table_name: str = "swe2d_mesh_results") -> np.ndarray:
        conn = _open_ro(gpkg_path)
        if conn is None:
            return np.empty(0, dtype=np.float64)
        try:
            table_name = str(table_name or "swe2d_mesh_results")
            if not _table_exists(conn, table_name):
                return np.empty(0, dtype=np.float64)
            q_table = _quote_ident(table_name)
            cur = conn.execute(
                f"""
                SELECT DISTINCT t_s
                FROM {q_table}
                WHERE run_id = ?
                ORDER BY t_s
                """,
                (str(run_id),),
            )
            vals = [float(r[0]) for r in cur.fetchall()]
            return np.asarray(vals, dtype=np.float64)
        except Exception:
            return np.empty(0, dtype=np.float64)
        finally:
            conn.close()

    def load_snapshot(
        self,
        gpkg_path: str,
        run_id: str,
        t_s: float,
        t_tol: float = 0.5,
        table_name: str = "swe2d_mesh_results",
    ) -> VelocitySnapshot | None:
        table_name = str(table_name or "swe2d_mesh_results")
        key = (str(gpkg_path), table_name, str(run_id), float(t_s))
        snap = self._cache.get(key)
        if snap is not None:
            self._cache.move_to_end(key)
            return snap

        conn = _open_ro(gpkg_path)
        if conn is None:
            return None
        try:
            if not _table_exists(conn, table_name):
                return None
            q_table = _quote_ident(table_name)
            cur = conn.execute(
                f"""
                SELECT cell_id, h, hu, hv, t_s
                FROM {q_table}
                WHERE run_id = ? AND ABS(t_s - ?) < ?
                ORDER BY cell_id
                """,
                (str(run_id), float(t_s), float(t_tol)),
            )
            rows = cur.fetchall()
            if not rows:
                return None

            cell_id = np.asarray([int(r[0]) for r in rows], dtype=np.int32)
            h = np.asarray([float(r[1]) for r in rows], dtype=np.float64)
            hu_raw = np.asarray([float(r[2]) for r in rows], dtype=np.float64)
            hv_raw = np.asarray([float(r[3]) for r in rows], dtype=np.float64)
            t_used = float(rows[0][4])
            hu = hu_raw
            hv = hv_raw
            src = "cell_momentum"

            rec = _reconstruct_cell_momentum_from_face_flux(
                conn,
                run_id=str(run_id),
                t_s=t_used,
                t_tol=float(t_tol),
                cell_id=cell_id,
            )
            if rec is not None:
                rec_hu, rec_hv, _ = rec
                if rec_hu.size == hu_raw.size and rec_hv.size == hv_raw.size:
                    valid = np.isfinite(rec_hu) & np.isfinite(rec_hv)
                    hu = np.where(valid, rec_hu, hu_raw)
                    hv = np.where(valid, rec_hv, hv_raw)
                    if np.any(valid):
                        src = "face_flux_reconstruction"

            snap = VelocitySnapshot(
                run_id=str(run_id),
                t_s=t_used,
                cell_id=cell_id,
                h=h,
                hu=hu,
                hv=hv,
                source=src,
            )
            self._cache[key] = snap
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)
            return snap
        except Exception:
            return None
        finally:
            conn.close()

    def build_vectors(
        self,
        snapshot: VelocitySnapshot,
        cell_xy: Dict[int, Tuple[float, float]],
        stride: int = 1,
        min_depth: float = 1.0e-6,
        min_speed: float = 0.0,
    ) -> List[Dict[str, float]]:
        """Build vector records with speed and style-ready fields.

        Returns rows with: x, y, u, v, speed, angle_deg, cell_id.
        """
        stride = max(1, int(stride))
        h = np.maximum(snapshot.h, float(min_depth))
        u = snapshot.hu / h
        v = snapshot.hv / h
        speed = np.sqrt(u * u + v * v)

        out: List[Dict[str, float]] = []
        for i in range(0, snapshot.cell_id.size, stride):
            cid = int(snapshot.cell_id[i])
            xy = cell_xy.get(cid)
            if xy is None:
                continue
            s = float(speed[i])
            if s < float(min_speed):
                continue
            ui = float(u[i])
            vi = float(v[i])
            out.append(
                {
                    "cell_id": cid,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                    "u": ui,
                    "v": vi,
                    "speed": s,
                    "angle_deg": float(np.degrees(np.arctan2(vi, ui))),
                }
            )
        return out

    def build_streamline_traces(
        self,
        snapshot: VelocitySnapshot,
        cell_xy: Dict[int, Tuple[float, float]],
        seed_count: int = 48,
        max_steps: int = 28,
        step_len_factor: float = 0.85,
        min_depth: float = 1.0e-6,
        min_speed: float = 0.05,
        seed_stride: int = 1,
    ) -> List[Dict[str, object]]:
        """Build streamline-like traces from a single velocity snapshot.

        Traces are seeded from distributed active cells and integrated forward
        using nearest-neighbor (or KD-tree weighted) velocity samples.
        """
        seed_count = max(1, int(seed_count))
        max_steps = max(2, int(max_steps))
        step_len_factor = max(0.05, float(step_len_factor))
        seed_stride = max(1, int(seed_stride))

        h = np.maximum(snapshot.h, float(min_depth))
        u = snapshot.hu / h
        v = snapshot.hv / h
        speed = np.sqrt(u * u + v * v)

        pts_x: List[float] = []
        pts_y: List[float] = []
        vel_u: List[float] = []
        vel_v: List[float] = []
        vel_s: List[float] = []

        for i in range(snapshot.cell_id.size):
            cid = int(snapshot.cell_id[i])
            xy = cell_xy.get(cid)
            if xy is None:
                continue
            si = float(speed[i])
            if not np.isfinite(si):
                continue
            ui = float(u[i])
            vi = float(v[i])
            if not (np.isfinite(ui) and np.isfinite(vi)):
                continue
            pts_x.append(float(xy[0]))
            pts_y.append(float(xy[1]))
            vel_u.append(ui)
            vel_v.append(vi)
            vel_s.append(si)

        if len(pts_x) < 4:
            return []

        x_arr = np.asarray(pts_x, dtype=np.float64)
        y_arr = np.asarray(pts_y, dtype=np.float64)
        u_arr = np.asarray(vel_u, dtype=np.float64)
        v_arr = np.asarray(vel_v, dtype=np.float64)
        s_arr = np.asarray(vel_s, dtype=np.float64)

        finite = (
            np.isfinite(x_arr)
            & np.isfinite(y_arr)
            & np.isfinite(u_arr)
            & np.isfinite(v_arr)
            & np.isfinite(s_arr)
        )
        if not np.any(finite):
            return []

        x_arr = x_arr[finite]
        y_arr = y_arr[finite]
        u_arr = u_arr[finite]
        v_arr = v_arr[finite]
        s_arr = s_arr[finite]

        if x_arr.size < 4:
            return []

        x_min = float(np.min(x_arr))
        x_max = float(np.max(x_arr))
        y_min = float(np.min(y_arr))
        y_max = float(np.max(y_arr))
        span_x = max(1.0e-9, x_max - x_min)
        span_y = max(1.0e-9, y_max - y_min)
        base_len = max(1.0e-9, np.sqrt(span_x * span_y / float(max(1, x_arr.size))))
        step_len = step_len_factor * base_len

        tree = None
        try:
            from scipy.spatial import cKDTree  # type: ignore

            tree = cKDTree(np.column_stack((x_arr, y_arr)))
        except Exception:
            tree = None

        def _sample_velocity(px: float, py: float) -> Tuple[float, float, float]:
            if tree is not None:
                k = 4 if x_arr.size >= 4 else 1
                dist, idx = tree.query((px, py), k=k)
                idx_arr = np.atleast_1d(idx).astype(np.int64)
                dist_arr = np.atleast_1d(dist).astype(np.float64)
                if idx_arr.size <= 0:
                    return 0.0, 0.0, 0.0
                w = 1.0 / np.maximum(dist_arr, 1.0e-6)
                sw = float(np.sum(w))
                if sw <= 1.0e-12:
                    return 0.0, 0.0, 0.0
                us = float(np.sum(u_arr[idx_arr] * w) / sw)
                vs = float(np.sum(v_arr[idx_arr] * w) / sw)
                ss = float(np.hypot(us, vs))
                return us, vs, ss

            d2 = (x_arr - px) * (x_arr - px) + (y_arr - py) * (y_arr - py)
            if d2.size <= 0:
                return 0.0, 0.0, 0.0
            idx0 = int(np.argmin(d2))
            us = float(u_arr[idx0])
            vs = float(v_arr[idx0])
            ss = float(np.hypot(us, vs))
            return us, vs, ss

        order = np.lexsort((y_arr, x_arr))
        if seed_stride > 1 and order.size > 1:
            order = order[::seed_stride]
        if order.size <= 0:
            return []

        if order.size > seed_count:
            pick = np.linspace(0, order.size - 1, num=seed_count, dtype=np.int64)
            seed_idx = order[pick]
        else:
            seed_idx = order

        # Skip near-duplicate seeds by coarse bins to improve spatial coverage.
        bin_x = max(1.0e-9, span_x / max(4.0, np.sqrt(float(seed_count))))
        bin_y = max(1.0e-9, span_y / max(4.0, np.sqrt(float(seed_count))))
        seen_bins = set()
        seed_points: List[Tuple[float, float, float]] = []
        for idx in seed_idx:
            sx = float(x_arr[int(idx)])
            sy = float(y_arr[int(idx)])
            ss = float(s_arr[int(idx)])
            if ss < float(min_speed):
                continue
            bx = int((sx - x_min) / bin_x)
            by = int((sy - y_min) / bin_y)
            key = (bx, by)
            if key in seen_bins:
                continue
            seen_bins.add(key)
            seed_points.append((sx, sy, ss))
            if len(seed_points) >= seed_count:
                break

        if not seed_points:
            # Fallback: use top-speed cells as seeds.
            top = np.argsort(-s_arr)
            for idx in top[:seed_count]:
                ss = float(s_arr[int(idx)])
                if ss < float(min_speed):
                    break
                seed_points.append((float(x_arr[int(idx)]), float(y_arr[int(idx)]), ss))
            if not seed_points:
                return []

        traces: List[Dict[str, object]] = []
        margin = 2.0 * step_len
        for trace_id, (sx, sy, _) in enumerate(seed_points):
            px = float(sx)
            py = float(sy)
            pts: List[Tuple[float, float]] = [(px, py)]
            sampled_speeds: List[float] = []

            for _ in range(max_steps):
                us, vs, ss = _sample_velocity(px, py)
                if ss < float(min_speed) or not np.isfinite(ss):
                    break
                dx = us / ss
                dy = vs / ss
                local_step = step_len * min(2.5, max(0.75, 1.0 + 0.25 * ss))
                nx = px + dx * local_step
                ny = py + dy * local_step
                if (
                    nx < (x_min - margin)
                    or nx > (x_max + margin)
                    or ny < (y_min - margin)
                    or ny > (y_max + margin)
                ):
                    break
                pts.append((float(nx), float(ny)))
                sampled_speeds.append(float(ss))
                px, py = nx, ny

            if len(pts) < 3:
                continue

            seg_len = 0.0
            for i in range(1, len(pts)):
                seg_len += float(np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
            mean_speed = float(np.mean(np.asarray(sampled_speeds, dtype=np.float64))) if sampled_speeds else 0.0
            traces.append(
                {
                    "trace_id": int(trace_id),
                    "points": pts,
                    "mean_speed": mean_speed,
                    "length": float(seg_len),
                }
            )

        return traces

    @staticmethod
    def style_from_speed(speed: float) -> Dict[str, object]:
        """Map speed to color/width style fields for symbolization."""
        s = max(0.0, float(speed))
        if s < 0.25:
            return {"color": "#2c7bb6", "width": 0.4}
        if s < 0.75:
            return {"color": "#00a6ca", "width": 0.7}
        if s < 1.5:
            return {"color": "#fdae61", "width": 1.0}
        return {"color": "#d7191c", "width": 1.3}


def _quote_ident(name: str) -> str:
    # SQLite identifier quoting for dynamic table names.
    safe = str(name or "").replace('"', '""')
    return f'"{safe}"'


def _pick_column(available: Sequence[str], aliases: Sequence[str]) -> str:
    avail = {str(c).lower(): str(c) for c in available}
    for alias in aliases:
        col = avail.get(str(alias).lower())
        if col:
            return col
    return ""


def _reconstruct_cell_momentum_from_face_flux(
    conn: sqlite3.Connection,
    run_id: str,
    t_s: float,
    t_tol: float,
    cell_id: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, str] | None:
    table_candidates = (
        "swe2d_face_flux_results",
        "swe2d_face_results",
        "swe2d_flux_faces",
    )
    table = ""
    cols: List[str] = []
    for name in table_candidates:
        if _table_exists(conn, name):
            cand_cols = _table_columns(conn, name)
            if cand_cols:
                table = name
                cols = cand_cols
                break
    if not table or not cols:
        return None

    col_run = _pick_column(cols, ("run_id", "run", "result_id"))
    col_t = _pick_column(cols, ("t_s", "time_s", "time", "t"))
    col_cell = _pick_column(cols, ("cell_id", "cell", "cell_idx", "owner_cell"))
    col_nx = _pick_column(cols, ("nx", "normal_x", "face_nx"))
    col_ny = _pick_column(cols, ("ny", "normal_y", "face_ny"))
    col_qn = _pick_column(cols, ("flux_n", "qn", "normal_flux", "q_normal", "flux"))
    col_w = _pick_column(cols, ("face_length", "edge_length", "ds", "weight"))

    if not col_t or not col_cell or not col_nx or not col_ny or not col_qn:
        return None

    sel_cols = [col_cell, col_nx, col_ny, col_qn, col_t]
    if col_w:
        sel_cols.append(col_w)
    q = f'SELECT {", ".join(f"\"{c}\"" for c in sel_cols)} FROM "{table}" WHERE ABS("{col_t}" - ?) < ?'
    params: List[object] = [float(t_s), float(t_tol)]
    if col_run:
        q += f' AND "{col_run}" = ?'
        params.append(str(run_id))

    try:
        rows = conn.execute(q, tuple(params)).fetchall()
    except Exception:
        return None
    if not rows:
        return None

    cid_to_index = {int(c): i for i, c in enumerate(np.asarray(cell_id, dtype=np.int32).tolist())}
    if not cid_to_index:
        return None

    # Per-cell weighted normal equations for least-squares solve of [hu, hv]
    acc: Dict[int, List[float]] = {}
    for row in rows:
        try:
            cid = int(row[0])
            idx = cid_to_index.get(cid)
            if idx is None:
                continue
            nx = float(row[1])
            ny = float(row[2])
            qn = float(row[3])
            if not (np.isfinite(nx) and np.isfinite(ny) and np.isfinite(qn)):
                continue
            w = 1.0
            if col_w:
                w = float(row[5]) if len(row) > 5 else 1.0
            if not np.isfinite(w) or w <= 0.0:
                w = 1.0
            s = acc.setdefault(idx, [0.0, 0.0, 0.0, 0.0, 0.0])
            s[0] += w * nx * nx
            s[1] += w * nx * ny
            s[2] += w * ny * ny
            s[3] += w * nx * qn
            s[4] += w * ny * qn
        except Exception:
            continue

    if not acc:
        return None

    rec_hu = np.full(cell_id.size, np.nan, dtype=np.float64)
    rec_hv = np.full(cell_id.size, np.nan, dtype=np.float64)
    for idx, (sxx, sxy, syy, bx, by) in acc.items():
        det = (sxx * syy) - (sxy * sxy)
        if abs(det) <= 1.0e-14:
            continue
        rec_hu[idx] = (bx * syy - by * sxy) / det
        rec_hv[idx] = (by * sxx - bx * sxy) / det

    return rec_hu, rec_hv, table
