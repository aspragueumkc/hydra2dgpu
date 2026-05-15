"""swe2d_velocity_layer.py

Sprint 3 helper utilities for velocity-vector overlays.

This module is intentionally Qt-free so it can be tested independently.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import sqlite3
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


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


def _open_ro(gpkg_path: str) -> sqlite3.Connection | None:
    try:
        conn = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
        return conn
    except Exception:
        return None


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    try:
        cur = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None
    except Exception:
        return False


def _table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    try:
        cur = conn.execute(f'PRAGMA table_info("{table_name}")')
        return [str(r[1]) for r in cur.fetchall()]
    except Exception:
        return []


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
