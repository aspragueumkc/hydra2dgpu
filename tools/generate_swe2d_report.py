#!/usr/bin/env python3
"""
SWE2D Model Review Report Generator
=====================================

A reusable script that generates a comprehensive review report from a SWE2D
simulation stored in a GeoPackage. For use by engineers, reviewers, and AI
agents to produce standardized model-review documentation.

Usage:
    python tools/generate_swe2d_report.py <gpkg_path> [--run-id RUN_ID] [--out-dir DIR]

Examples:
    # Latest run in the specified gpkg
    python tools/generate_swe2d_report.py qgis_testing_project/swe2d_model_culvert_test.gpkg

    # Specific run
    python tools/generate_swe2d_report.py swe2d_model.gpkg --run-id swe2d_20260601T095937-0500

    # Custom output directory
    python tools/generate_swe2d_report.py model.gpkg --out-dir my_review_output

Requirements:
    - numpy
    - matplotlib
    - sqlite3 (stdlib)
    - The GeoPackage must contain SWE2D results tables
      (swe2d_run_logs, swe2d_mesh_results, swe2d_line_results_*, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional matplotlib import — we fail gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_G = 32.1740  # ft/s²  (standard gravity in US customary units)
_FT3_PER_M3 = 35.3147
_ACREFT_PER_FT3 = 1.0 / 43560.0
_FT_TO_M = 0.3048

# ===================================================================
#  GeoPackage helpers
# ===================================================================


def _table_exists(db: sqlite3.Connection, name: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _col_names(db: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in db.execute(f"PRAGMA table_info({table!r})").fetchall()]


def _discover_runs(db: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return all runs in descending created_utc order."""
    if not _table_exists(db, "swe2d_run_logs"):
        return []
    rows = db.execute(
        """
        SELECT run_id, created_utc, start_wallclock, end_wallclock, duration_s
        FROM swe2d_run_logs ORDER BY created_utc DESC
        """
    ).fetchall()
    return [
        {
            "run_id": r[0],
            "created_utc": r[1],
            "start_wallclock": r[2],
            "end_wallclock": r[3],
            "duration_s": r[4],
        }
        for r in rows
    ]


def _latest_run(db: sqlite3.Connection) -> Optional[str]:
    """Return the most-recent run_id (by created_utc) or None."""
    runs = _discover_runs(db)
    return runs[0]["run_id"] if runs else None


def _parse_log_metadata(db: sqlite3.Connection, run_id: str) -> Dict[str, Any]:
    """Return the JSON metadata blob from swe2d_run_logs as a dict."""
    row = db.execute(
        "SELECT metadata_json FROM swe2d_run_logs WHERE run_id=?", (run_id,)
    ).fetchone()
    if row and row[0]:
        return json.loads(row[0])
    return {}


def _get_log_tail(db: sqlite3.Connection, run_id: str, n_chars: int = 5000) -> str:
    """Return the last *n_chars* of the run log."""
    row = db.execute(
        f"SELECT SUBSTR(log_text, -{n_chars}) FROM swe2d_run_logs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    return row[0] if row and row[0] else ""


def _final_timestep(db: sqlite3.Connection, run_id: str, table: str) -> Optional[float]:
    """Return the largest t_s in *table* for the given run."""
    if not _table_exists(db, table):
        return None
    row = db.execute(
        f"SELECT MAX(t_s) FROM {table!r} WHERE run_id=?", (run_id,)
    ).fetchone()
    return row[0] if row else None


# ===================================================================
#  Data extraction
# ===================================================================


def extract_mesh_stats(
    db: sqlite3.Connection, run_id: str, t_s: float
) -> Dict[str, Any]:
    """Compute summary statistics of the mesh at a given timestep."""
    cur = db.execute(
        """
        SELECT COUNT(*)                                             AS n_total,
               SUM(CASE WHEN h > 0.001 THEN 1 ELSE 0 END)           AS n_wet,
               SUM(CASE WHEN h <= 0.001 THEN 1 ELSE 0 END)          AS n_dry,
               AVG(h)                                               AS h_avg_all,
               MIN(h)                                               AS h_min,
               MAX(h)                                               AS h_max,
               AVG(CASE WHEN h > 0.001 THEN h ELSE NULL END)        AS h_avg_wet,
               AVG(CASE WHEN h > 0.001 THEN hu/h ELSE NULL END)     AS u_avg_wet,
               AVG(CASE WHEN h > 0.001 THEN hv/h ELSE NULL END)     AS v_avg_wet,
               SQRT(AVG(CASE WHEN h > 0.001
                             THEN (hu/h)*(hu/h)+(hv/h)*(hv/h) ELSE 0 END))
                                                                     AS vel_mag_avg
        FROM swe2d_mesh_results
        WHERE run_id=? AND t_s=?
        """,
        (run_id, t_s),
    )
    row = cur.fetchone()
    if row is None:
        return {}
    return {
        "total_cells": int(row[0]),
        "wet_cells": int(row[1]),
        "dry_cells": int(row[2]),
        "h_avg_all": float(row[3]),
        "h_min": float(row[4]),
        "h_max": float(row[5]),
        "h_avg_wet": float(row[6]),
        "u_avg_wet": float(row[7]) if row[7] else 0.0,
        "v_avg_wet": float(row[8]) if row[8] else 0.0,
        "vel_mag_avg": float(row[9]),
    }


def extract_mesh_arrays(
    db: sqlite3.Connection, run_id: str, t_s: float
) -> Dict[str, np.ndarray]:
    """Return full arrays of h, hu, hv for the given timestep."""
    cur = db.execute(
        "SELECT h, hu, hv FROM swe2d_mesh_results WHERE run_id=? AND t_s=?",
        (run_id, t_s),
    )
    arr = np.array(cur.fetchall(), dtype=np.float64)
    return {"h": arr[:, 0], "hu": arr[:, 1], "hv": arr[:, 2]}


def extract_profile_data(
    db: sqlite3.Connection, run_id: str, t_s: float, line_id: int = 1
) -> Optional[Dict[str, np.ndarray]]:
    """Return the profile data for a sample line at the given time."""
    table = "swe2d_line_results_profile"
    if not _table_exists(db, table):
        return None
    cols = _col_names(db, table)
    if "station_m" not in cols:
        return None
    cur = db.execute(
        f"""
        SELECT station_m, depth_m, velocity_ms, wse_m, bed_m, flow_qn, wet, fr
        FROM {table!r}
        WHERE run_id=? AND t_s=? AND line_id=?
        ORDER BY station_m
        """,
        (run_id, t_s, line_id),
    )
    arr = np.array(cur.fetchall(), dtype=np.float64)
    if len(arr) == 0:
        return None
    return {
        "station": arr[:, 0],
        "depth": arr[:, 1],
        "velocity": arr[:, 2],
        "wse": arr[:, 3],
        "bed": arr[:, 4],
        "flow_qn": arr[:, 5],
        "wet": arr[:, 6],
        "fr": arr[:, 7],
    }


def extract_line_ts(
    db: sqlite3.Connection, run_id: str, line_id: int = 1
) -> Optional[Dict[str, np.ndarray]]:
    """Return time-series data for a sample line."""
    table = "swe2d_line_results_ts"
    if not _table_exists(db, table):
        return None
    cur = db.execute(
        f"""
        SELECT t_s, depth_m, velocity_ms, wse_m, bed_m, flow_cms, wet_frac, fr
        FROM {table!r}
        WHERE run_id=? AND line_id=?
        ORDER BY t_s
        """,
        (run_id, line_id),
    )
    arr = np.array(cur.fetchall(), dtype=np.float64)
    if len(arr) == 0:
        return None
    return {
        "t": arr[:, 0],
        "depth": arr[:, 1],
        "velocity": arr[:, 2],
        "wse": arr[:, 3],
        "bed": arr[:, 4],
        "flow": arr[:, 5],
        "wet_frac": arr[:, 6],
        "fr": arr[:, 7],
    }


def extract_structure_coupling(
    db: sqlite3.Connection, run_id: str
) -> Dict[str, Any]:
    """
    Return coupling data for all structures at the final timestep,
    plus time-series of flow, headwater, and tailwater.
    """
    table = "swe2d_coupling_results"
    if not _table_exists(db, table):
        return {}

    # -- Final-timestep summary --
    t_final = _final_timestep(db, run_id, table)
    if t_final is None:
        return {}
    cur = db.execute(
        """
        SELECT object_id, metric, value
        FROM swe2d_coupling_results
        WHERE run_id=? AND t_s=?
        ORDER BY object_id, metric
        """,
        (run_id, t_final),
    )
    final_data: Dict[str, Dict[str, float]] = {}
    for row in cur:
        oid = str(row[0])
        final_data.setdefault(oid, {})[str(row[1])] = float(row[2])

    # -- Time-series for flow, headwater, tailwater --
    def _ts(metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (time, struct_1, struct_2) arrays."""
        cur = db.execute(
            """
            SELECT t_s, object_id, value
            FROM swe2d_coupling_results
            WHERE run_id=? AND metric=?
            ORDER BY t_s, object_id
            """,
            (run_id, metric),
        )
        rows = cur.fetchall()
        if not rows:
            return np.array([]), np.array([]), np.array([])
        # Every two rows share the same t_s (one per object)
        times = np.array([r[0] for r in rows[::2]])
        vals_1 = np.array([rows[i][2] for i in range(0, len(rows), 2)])
        vals_2 = np.array([rows[i + 1][2] for i in range(0, len(rows), 2)])
        return times, vals_1, vals_2

    t_flow, s1_flow, s2_flow = _ts("flow")
    t_hw, s1_hw, s2_hw = _ts("available_head_up")
    t_tw, s1_tw, s2_tw = _ts("tailwater_depth")

    return {
        "final": final_data,
        "flow_t": t_flow,
        "flow_s1": s1_flow,
        "flow_s2": s2_flow,
        "head_t": t_hw,
        "head_s1": s1_hw,
        "head_s2": s2_hw,
        "tw_t": t_tw,
        "tw_s1": s1_tw,
        "tw_s2": s2_tw,
    }


def extract_structure_attributes(
    db: sqlite3.Connection,
    gpkg_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return attribute dictionaries for every structure."""
    table = "swe2d_structures"
    if not _table_exists(db, table):
        return []

    # Use OGR if available for geometry, otherwise just attributes
    try:
        from osgeo import ogr

        ds = ogr.Open(gpkg_path or ":memory:")
        if ds is None:
            raise RuntimeError("ogr.Open failed")
        lyr = ds.GetLayerByName(table)
        if lyr is None:
            ds = None
        else:
            lyr_defn = lyr.GetLayerDefn()
            field_names = [lyr_defn.GetFieldDefn(j).GetName() for j in range(lyr_defn.GetFieldCount())]
            lyr.ResetReading()
            structs = []
            for feat in lyr:
                row = {}
                for fn in field_names:
                    row[fn] = feat.GetField(fn)
                geom = feat.GetGeometryRef()
                if geom:
                    row["_geom_type"] = geom.GetGeometryName()
                    row["_length"] = geom.Length()
                    row["_coords"] = [
                        (geom.GetPoint(k)[0], geom.GetPoint(k)[1])
                        for k in range(geom.GetPointCount())
                    ]
                else:
                    row["_geom_type"] = None
                    row["_length"] = 0.0
                    row["_coords"] = []
                structs.append(row)
            ds = None
            return structs
    except (ImportError, RuntimeError):
        pass

    # Fallback: attributes only
    cols = _col_names(db, table)
    cur = db.execute(f"SELECT {','.join(cols)} FROM {table!r}")
    structs = []
    for row in cur.fetchall():
        s = dict(zip(cols, row))
        s["_geom_type"] = None
        s["_length"] = 0.0
        s["_coords"] = []
        structs.append(s)
    return structs


def extract_sample_line_attributes(db: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return attribute dictionaries for every sample line."""
    table = "swe2d_sample_lines"
    if not _table_exists(db, table):
        return []
    cols = _col_names(db, table)
    cur = db.execute(f"SELECT {','.join(cols)} FROM {table!r}")
    lines = []
    for row in cur.fetchall():
        lines.append(dict(zip(cols, row)))
    return lines


def extract_conservation_data(
    db: sqlite3.Connection, run_id: str
) -> Optional[Dict[str, Any]]:
    """Return conservation run summary and storage time-series."""
    # Run summary
    cur = db.execute(
        "SELECT * FROM swe2d_conservation_runs WHERE run_id=?", (run_id,)
    )
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    summary = dict(zip(cols, row)) if row else {}

    # Storage time-series
    cur = db.execute(
        """
        SELECT t_s, storage_model, storage_m3
        FROM swe2d_conservation_storage_ts
        WHERE run_id=? ORDER BY t_s
        """,
        (run_id,),
    )
    arr = np.array(cur.fetchall(), dtype=np.float64)
    ts = {}
    if len(arr) > 0:
        ts = {"t": arr[:, 0], "storage_model": arr[:, 1], "storage_m3": arr[:, 2]}
    return {"summary": summary, "ts": ts}


def extract_boundary_flux(
    db: sqlite3.Connection, run_id: str
) -> Optional[Dict[str, np.ndarray]]:
    """Return boundary-flux time-series for all BC groups."""
    table = "swe2d_boundary_flux_forensics_ts"
    if not _table_exists(db, table):
        return None
    cur = db.execute(
        f"""
        SELECT t_s, group_name, q_effective_cms, vol_effective_m3
        FROM {table!r}
        WHERE run_id=?
        ORDER BY t_s
        """,
        (run_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    # Group by group_name
    groups: Dict[str, Dict[str, list]] = {}
    for r in rows:
        grp = str(r[1])
        groups.setdefault(grp, {"t": [], "q": [], "vol": []})
        groups[grp]["t"].append(r[0])
        groups[grp]["q"].append(r[2])
        groups[grp]["vol"].append(r[3])
    result = {}
    for grp, data in groups.items():
        result[grp] = {
            "t": np.array(data["t"]),
            "q": np.array(data["q"]),
            "vol": np.array(data["vol"]),
        }
    return result


def extract_bc_lines(db: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Return BC line attributes."""
    table = "swe2d_bc_lines"
    if not _table_exists(db, table):
        return []
    cols = _col_names(db, table)
    # Exclude geometry blob from text output
    cur = db.execute(
        f"SELECT fid, bc_type, bc_value, priority, hydrograph_id FROM {table!r}"
    )
    return [dict(zip(["fid", "bc_type", "bc_value", "priority", "hydrograph_id"], r)) for r in cur.fetchall()]


# ===================================================================
#  Independent calculations
# ===================================================================


def mannings_full_flow(
    n: float, area: float, hyd_radius: float, slope: float, us_customary: bool = True
) -> float:
    """
    Manning's equation for full-pipe / open-channel flow.

    US customary:  Q = (1.486/n) * A * R^(2/3) * S^(1/2)   [ft³/s]
    SI:            Q = (1.0/n)   * A * R^(2/3) * S^(1/2)   [m³/s]
    """
    k = 1.486 if us_customary else 1.0
    if area <= 0 or hyd_radius <= 0 or slope <= 0:
        return 0.0
    return k / n * area * hyd_radius ** (2.0 / 3.0) * slope ** 0.5


def normal_depth_box_culvert(
    q: float, b: float, n: float, slope: float, us_customary: bool = True
) -> float:
    """
    Iteratively find the normal depth in a rectangular culvert.
    Returns depth *y* in ft (or m if SI).
    """
    k = 1.486 if us_customary else 1.0
    # Binary search between 0 and b (max reasonable depth = width)
    lo, hi = 0.001, b
    for _ in range(50):
        y = (lo + hi) / 2.0
        A = b * y
        P = b + 2.0 * y
        R = A / P
        Q_calc = k / n * A * R ** (2.0 / 3.0) * slope ** 0.5
        if Q_calc > q:
            hi = y
        else:
            lo = y
    return (lo + hi) / 2.0


def outlet_control_hw(
    q: float,
    area: float,
    hyd_radius: float,
    length: float,
    n: float,
    k_e: float,
    tw_depth: float,
    g: float = _G,
) -> float:
    """
    Simple outlet-control headwater depth (US customary).
        HW = TW + (1 + k_e + k_f * L) * V² / (2g)
    where k_f = 29 n² / R^(4/3).
    """
    if area <= 0:
        return 0.0
    k_f = 29.0 * n ** 2 / hyd_radius ** (4.0 / 3.0)
    v = q / area
    return tw_depth + (1.0 + k_e + k_f * length) * v ** 2 / (2.0 * g)


def critical_depth_rect(q: float, b: float, g: float = _G) -> float:
    """Critical depth in a rectangular channel: yc = (q²/g)^(1/3)."""
    return (q ** 2 / (g * b ** 2)) ** (1.0 / 3.0)


def integrate_profile_flow(
    station: np.ndarray, flow_qn: np.ndarray, wet: np.ndarray
) -> float:
    """Integrate unit-discharge (flow_qn) across the profile (trapezoidal)."""
    q_total = 0.0
    for i in range(len(station) - 1):
        ds = station[i + 1] - station[i]
        if wet[i] and wet[i + 1]:
            q_total += 0.5 * (flow_qn[i] + flow_qn[i + 1]) * ds
        elif wet[i]:
            q_total += flow_qn[i] * ds
    return q_total


def profile_manning_check(
    station: np.ndarray,
    depth: np.ndarray,
    bed: np.ndarray,
    wet: np.ndarray,
    n: float = 0.035,
) -> Dict[str, float]:
    """Rough Manning check across a profile cross-section."""
    wet_idx = wet > 0
    if np.sum(wet_idx) < 3:
        return {}
    s_wet = station[wet_idx]
    d_wet = depth[wet_idx]
    b_wet = bed[wet_idx]
    # Slope from linear fit
    A_mat = np.vstack([s_wet, np.ones_like(s_wet)]).T
    m, _ = np.linalg.lstsq(A_mat, b_wet, rcond=None)[0]
    slope = -m
    # Cross-section properties
    dx = np.diff(s_wet)
    d_mid = 0.5 * (d_wet[:-1] + d_wet[1:])
    area = float(np.sum(d_mid * dx))
    perim = float(np.sum(np.sqrt(dx ** 2 + np.diff(b_wet) ** 2)))
    r_h = area / perim if perim > 0 else 0.001
    q_man = mannings_full_flow(n, area, r_h, max(slope, 1e-6))
    return {"slope": slope, "area": area, "perimeter": perim, "hyd_radius": r_h, "q_manning": q_man}


# ===================================================================
#  Figure generation
# ===================================================================


def _make_figure_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stats_box(ax: plt.Axes, text: str, fontsize: int = 10) -> None:
    ax.axis("off")
    ax.text(
        0.05, 0.95, text, transform=ax.transAxes, fontsize=fontsize,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )


def figure_mesh_overview(
    mesh: Dict[str, np.ndarray], out: Path
) -> None:
    """Figure 1: Depth, velocity, Froude distributions + scatter plots."""
    h = mesh["h"]
    hu = mesh["hu"]
    hv = mesh["hv"]
    wet = h > 0.001
    h_w = h[wet]
    u_w = hu[wet] / h_w
    v_w = hv[wet] / h_w
    vel = np.sqrt(u_w ** 2 + v_w ** 2)
    fr = vel / np.sqrt(_G * h_w)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(h_w, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Water Depth (ft)" if np.median(h_w) < 30 else "Water Depth (m)")
    ax.set_ylabel("Cell Count")
    ax.set_title(f"Depth Distribution (Wet Cells)\nMean={np.mean(h_w):.2f}, Max={np.max(h_w):.2f}")
    ax.axvline(np.mean(h_w), color="red", linestyle="--", label=f"Mean={np.mean(h_w):.2f}")
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(vel[vel < np.percentile(vel, 99)], bins=50, color="coral", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Velocity Magnitude")
    ax.set_ylabel("Cell Count")
    ax.set_title(f"Velocity Distribution\nMean={np.mean(vel):.2f}, Max={np.max(vel):.2f}")
    ax.axvline(np.mean(vel), color="red", linestyle="--", label=f"Mean={np.mean(vel):.2f}")
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    sc = ax.scatter(h_w, vel, c=vel, s=1, cmap="viridis", alpha=0.5)
    ax.set_xlabel("Water Depth")
    ax.set_ylabel("Velocity")
    ax.set_title("Depth vs Velocity (Wet Cells)")
    plt.colorbar(sc, ax=ax, label="Velocity")

    ax = fig.add_subplot(gs[1, 0])
    sc = ax.scatter(u_w, v_w, c=h_w, s=1, cmap="Blues", alpha=0.5)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title("Velocity Components (u vs v)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label="Depth")

    ax = fig.add_subplot(gs[1, 1])
    fr_plot = fr[fr < np.percentile(fr[fr > 0], 99)]
    ax.hist(fr_plot, bins=50, color="mediumseagreen", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Froude Number")
    ax.set_ylabel("Cell Count")
    ax.set_title(f"Froude Number Distribution\nMean={np.mean(fr):.2f}")
    ax.axvline(1.0, color="red", linestyle="--", label="Fr=1 (critical)")
    ax.legend()

    ax = fig.add_subplot(gs[1, 2])
    _stats_box(
        ax,
        f"Mesh Summary (Final Timestep)\n"
        f"{'─'*30}\n"
        f"Total Cells:      {len(h)}\n"
        f"Wet Cells:        {np.sum(wet)} ({np.sum(wet)/len(h)*100:.1f}%)\n"
        f"Dry Cells:        {np.sum(~wet)}\n"
        f"{'─'*30}\n"
        f"Depth (wet):\n"
        f"  Mean:  {np.mean(h_w):.2f}\n"
        f"  Max:   {np.max(h_w):.2f}\n"
        f"  Min:   {np.min(h_w):.2f}\n"
        f"{'─'*30}\n"
        f"Velocity (wet):\n"
        f"  Mean:  {np.mean(vel):.2f}\n"
        f"  Max:   {np.max(vel):.2f}\n"
        f"  Mean u: {np.mean(u_w):.2f}\n"
        f"  Mean v: {np.mean(v_w):.2f}\n"
        f"{'─'*30}\n"
        f"Froude:\n"
        f"  Mean:  {np.mean(fr):.2f}\n"
        f"  Max:   {np.max(fr):.2f}",
    )

    plt.tight_layout()
    _save_fig(fig, out)


def figure_sample_line_profile(
    prof: Dict[str, np.ndarray], out: Path
) -> None:
    """Figure 2: Sample-line profile (bed, WSE, depth, velocity, Fr)."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    s, d, v, w, b, _, wet, fr = (
        prof["station"], prof["depth"], prof["velocity"],
        prof["wse"], prof["bed"], prof["flow_qn"], prof["wet"], prof["fr"],
    )
    wet_bool = wet > 0

    ax = axes[0]
    ax.fill_between(s, b, w, where=wet_bool, color="lightblue", alpha=0.5, label="Water")
    ax.plot(s, b, "brown", linewidth=2, label="Bed")
    ax.plot(s, w, "blue", linewidth=1.5, label="WSE")
    ax.set_ylabel("Elevation")
    ax.set_title("Sample Line Profile: Bed & Water Surface Elevation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(s, 0, d, where=wet_bool, color="steelblue", alpha=0.6)
    ax.plot(s, d, "blue", linewidth=1.5)
    ax.set_ylabel("Depth")
    ax.set_title(f"Water Depth (Max={np.max(d):.2f}, Mean Wet={np.mean(d[wet_bool]):.2f})")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(s, v, "green", linewidth=1.5)
    ax.set_ylabel("Velocity")
    ax.set_title(f"Velocity (Max={np.max(v):.2f})")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(s, fr, "red", linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Fr=1")
    ax.set_xlabel("Station")
    ax.set_ylabel("Froude Number")
    ax.set_title("Froude Number along Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, out)


def figure_line_ts(
    ts: Dict[str, np.ndarray], out: Path
) -> None:
    """Figure 3: Sample-line time-series."""
    t = ts["t"] / 3600.0  # convert to hours
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    pairs = [
        (0, 0, t, ts["flow"], "Flow", "Flow"),
        (0, 1, t, ts["depth"], "Depth", "Mean Depth"),
        (1, 0, t, ts["velocity"], "Velocity", "Mean Velocity"),
        (1, 1, t, ts["wse"], "WSE", "Water Surface Elevation", ts["bed"]),
        (2, 0, t, ts["wet_frac"] * 100, "Wet Fraction (%)", "Wetted Width Fraction"),
        (2, 1, t, ts["fr"], "Froude #", "Froude Number"),
    ]

    for (row, col, x, y, ylabel, title, *extra) in pairs:
        ax = axes[row, col]
        if extra:
            ax.plot(x, extra[0], "brown", linewidth=1.5, label="Mean Bed")
        ax.plot(x, y, "b-" if not extra else "orange", linewidth=1.5)
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if extra:
            ax.legend()
        if "Fraction" in title:
            ax.set_ylim(0, 100)
        if "Froude" in title:
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, out)


def figure_structures(
    struct_coupling: Dict[str, Any],
    struct_attrs: List[Dict[str, Any]],
    out: Path,
) -> None:
    """Figure 4: Structure performance."""
    if not struct_coupling:
        return
    ft = struct_coupling.get("flow_t", np.array([]))
    s1f = struct_coupling.get("flow_s1", np.array([]))
    s2f = struct_coupling.get("flow_s2", np.array([]))
    ht = struct_coupling.get("head_t", np.array([]))
    s1h = struct_coupling.get("head_s1", np.array([]))
    s2h = struct_coupling.get("head_s2", np.array([]))
    twt = struct_coupling.get("tw_t", np.array([]))
    s1tw = struct_coupling.get("tw_s1", np.array([]))
    s2tw = struct_coupling.get("tw_s2", np.array([]))

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Flow
    ax = axes[0, 0]
    if len(ft) > 0:
        ft_h = ft / 3600.0
        ax.plot(ft_h, s1f, "b-", lw=1.5, label="Structure 1")
        ax.plot(ft_h, s2f, "r-", lw=1.5, label="Structure 2")
        ax.plot(ft_h, s1f + s2f, "g--", lw=1, label="Combined")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Flow")
    ax.set_title("Flow through Structures")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Headwater
    ax = axes[0, 1]
    if len(ht) > 0:
        ht_h = ht / 3600.0
        ax.plot(ht_h, s1h, "b-", lw=1.5, label="Structure 1")
        ax.plot(ht_h, s2h, "r-", lw=1.5, label="Structure 2")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Headwater Depth")
    ax.set_title("Upstream Headwater Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Tailwater
    ax = axes[1, 0]
    if len(twt) > 0:
        twt_h = twt / 3600.0
        ax.plot(twt_h, s1tw, "b-", lw=1.5, label="Structure 1")
        ax.plot(twt_h, s2tw, "r-", lw=1.5, label="Structure 2")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Tailwater Depth")
    ax.set_title("Downstream Tailwater Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Head differential
    ax = axes[1, 1]
    if len(ht) > 0 and len(twt) > 0:
        min_len = min(len(s1h), len(s1tw))
        dh1 = np.array(s1h[:min_len]) - np.array(s1tw[:min_len])
        dh2 = np.array(s2h[:min_len]) - np.array(s2tw[:min_len])
        ax.plot(ht_h[:min_len], dh1, "b-", lw=1.5, label="Structure 1")
        ax.plot(ht_h[:min_len], dh2, "r-", lw=1.5, label="Structure 2")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("ΔH = HW − TW")
    ax.set_title("Head Differential")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative volume
    ax = axes[2, 0]
    if len(ft) > 1:
        dt = np.diff(np.append([0], ft))
        cum1 = np.cumsum(s1f * dt) * _ACREFT_PER_FT3
        cum2 = np.cumsum(s2f * dt) * _ACREFT_PER_FT3
        ax.plot(ft_h, cum1, "b-", lw=1.5, label="Structure 1")
        ax.plot(ft_h, cum2, "r-", lw=1.5, label="Structure 2")
        ax.plot(ft_h, cum1 + cum2, "g--", lw=1, label="Combined")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Cumulative Volume (acre-ft)")
    ax.set_title("Cumulative Volume through Structures")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[2, 1]
    cap1 = struct_coupling.get("final", {}).get("1", {}).get("manning_cap", "?")
    cap2 = struct_coupling.get("final", {}).get("2", {}).get("manning_cap", "?")
    flow1 = s1f[-1] if len(s1f) > 0 else 0
    flow2 = s2f[-1] if len(s2f) > 0 else 0
    hw1 = s1h[-1] if len(s1h) > 0 else 0
    hw2 = s2h[-1] if len(s2h) > 0 else 0
    tw1 = s1tw[-1] if len(s1tw) > 0 else 0
    tw2 = s2tw[-1] if len(s2tw) > 0 else 0
    total_flow = flow1 + flow2

    s1_pct = flow1 / cap1 * 100 if isinstance(cap1, (int, float)) and cap1 > 0 else 0
    s2_pct = flow2 / cap2 * 100 if isinstance(cap2, (int, float)) and cap2 > 0 else 0

    _stats_box(
        ax,
        f"Structure Summary (Final Timestep)\n"
        f"{'─'*35}\n"
        f"Structure 1:\n"
        f"  Flow:          {flow1:.1f}\n"
        f"  Manning Cap:   {cap1}\n"
        f"  Capacity Used: {s1_pct:.0f}%\n"
        f"  Headwater:     {hw1:.2f}\n"
        f"  Tailwater:     {tw1:.2f}\n"
        f"{'─'*35}\n"
        f"Structure 2:\n"
        f"  Flow:          {flow2:.1f}\n"
        f"  Manning Cap:   {cap2}\n"
        f"  Capacity Used: {s2_pct:.0f}%\n"
        f"  Headwater:     {hw2:.2f}\n"
        f"  Tailwater:     {tw2:.2f}\n"
        f"{'─'*35}\n"
        f"Combined Flow:   {total_flow:.1f}\n"
        f"# structures:    {len(struct_attrs)}\n"
        f"Total capacity used varies\n",
    )

    plt.tight_layout()
    _save_fig(fig, out)


def figure_conservation(
    cons: Dict[str, Any],
    bc_flux: Optional[Dict[str, Dict[str, np.ndarray]]],
    line_ts: Optional[Dict[str, np.ndarray]],
    struct_coupling: Dict[str, Any],
    run_id: str,
    out: Path,
) -> None:
    """Figure 5: Conservation & residuals."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    cons_ts = cons.get("ts", {})
    cons_sum = cons.get("summary", {})
    ct = cons_ts.get("t", np.array([])) / 3600.0
    cs = cons_ts.get("storage_m3", np.array([]))

    # Storage
    ax = axes[0, 0]
    if len(ct) > 0:
        ax.plot(ct, cs, "b-", lw=1.5)
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Storage (m³)")
    ax.set_title("Water Volume in Storage")
    ax.grid(True, alpha=0.3)

    # Storage delta
    ax = axes[0, 1]
    if len(cs) > 1:
        delta = np.diff(cs, prepend=cs[0])
        ax.plot(ct, delta, "orange", lw=1.5)
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("ΔStorage (m³)")
    ax.set_title("Cumulative Change in Storage")
    ax.grid(True, alpha=0.3)

    # BC inflow
    ax = axes[0, 2]
    if bc_flux:
        first_grp = next(iter(bc_flux.values()))
        bt = first_grp["t"] / 3600.0
        ax.plot(bt, first_grp["q"], "g-", lw=1.5)
        ax.set_title(f"Boundary Inflow (Q≈{np.mean(first_grp['q']):.1f})")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Q")
    ax.grid(True, alpha=0.3)

    # BC cumulative volume
    ax = axes[1, 0]
    if bc_flux:
        first_grp = next(iter(bc_flux.values()))
        bt = first_grp["t"] / 3600.0
        ax.plot(bt, first_grp["vol"], "purple", lw=1.5)
        ax.set_title("Boundary Inflow Cumulative Volume")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Volume (m³)")
    ax.grid(True, alpha=0.3)

    # Flow comparison
    ax = axes[1, 1]
    if line_ts is not None:
        lt = line_ts["t"] / 3600.0
        ax.plot(lt, line_ts["flow"], "b-", lw=1.5, label="Sample Line")
    if struct_coupling:
        ft = struct_coupling.get("flow_t", np.array([])) / 3600.0
        sf1 = struct_coupling.get("flow_s1", np.array([]))
        sf2 = struct_coupling.get("flow_s2", np.array([]))
        if len(ft) > 0:
            ax.plot(ft, sf1 + sf2, "r-", lw=1.5, label="Structures")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Flow")
    ax.set_title("Flow Comparison: BC vs Structures vs Sample Line")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mass balance summary
    ax = axes[1, 2]
    closure_m3 = cons_sum.get("closure_residual_m3", "N/A")
    source_m3 = cons_sum.get("source_total_m3", "N/A")
    dstorage_m3 = cons_sum.get("storage_delta_m3", "N/A")
    bc_vol = cons_sum.get("boundary_group_volume_sum_m3", "N/A")
    avg_q = cons_sum.get("avg_implied_boundary_q_cms", "N/A")
    _stats_box(
        ax,
        f"Mass Conservation\n"
        f"{'─'*35}\n"
        f"ΔStorage:       {_fmt(dstorage_m3)} m³\n"
        f"BC Inflow Vol:  {_fmt(bc_vol)} m³\n"
        f"Avg Net BC Q:   {_fmt(avg_q)} m³/s\n"
        f"{'─'*35}\n"
        f"Closure:        {_fmt(closure_m3)} m³\n"
        f"Source Total:   {_fmt(source_m3)} m³\n"
        f"{'─'*35}\n"
        f"Run ID: {run_id}\n",
    )

    plt.tight_layout()
    _save_fig(fig, out)


def figure_independent_verification(
    prof: Optional[Dict[str, np.ndarray]],
    line_ts: Optional[Dict[str, np.ndarray]],
    struct_coupling: Dict[str, Any],
    struct_attrs: List[Dict[str, Any]],
    out: Path,
) -> None:
    """Figure 6: Independent calculations vs model results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Left column: Sample Line verification ---
    ax = axes[0, 0]
    if prof is not None:
        wet = prof["wet"] > 0
        q_profile = integrate_profile_flow(prof["station"], prof["flow_qn"], prof["wet"])
        line_q = line_ts["flow"][-1] if line_ts is not None else 0
        ax.plot(prof["station"][wet], prof["flow_qn"][wet], "b.-", markersize=3)
        ax.set_title(f"Unit Flow along Sample Line\n"
                     f"Integrated Q = {q_profile:.1f}, Model Q = {line_q:.1f}")
    ax.set_xlabel("Station")
    ax.set_ylabel("Unit Flow Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if prof is not None:
        wet = prof["wet"] > 0
        ax.fill_between(prof["station"], 0, prof["depth"],
                        where=wet, color="lightblue", alpha=0.3, label="Depth")
        twin = ax.twinx()
        twin.plot(prof["station"], prof["velocity"], "r-", lw=1, label="Velocity")
        twin.set_ylabel("Velocity", color="r")
        twin.tick_params(axis="y", labelcolor="r")
        ax.set_title("Depth and Velocity along Profile")
    ax.set_xlabel("Station")
    ax.set_ylabel("Depth")
    ax.grid(True, alpha=0.3)

    # Manning check text
    ax = axes[1, 0]
    if prof is not None:
        mc = profile_manning_check(
            prof["station"], prof["depth"], prof["bed"], prof["wet"]
        )
        if mc:
            _stats_box(
                ax,
                f"Manning's Check (n=0.035)\n"
                f"{'─'*30}\n"
                f"Slope S ≈ {mc['slope']:.4f}\n"
                f"A = {mc['area']:.1f} ft²\n"
                f"P = {mc['perimeter']:.1f} ft\n"
                f"R = {mc['hyd_radius']:.2f} ft\n"
                f"{'─'*30}\n"
                f"Manning Q = {mc['q_manning']:.0f}\n"
                f"(Overpredicts: need energy\n"
                f" slope, not bed slope)\n"
                f"{'─'*30}\n"
                f"Profile-Integrated Q\n"
                f"= {integrate_profile_flow(prof['station'], prof['flow_qn'], prof['wet']):.1f}",
            )
        else:
            _stats_box(ax, "Insufficient wet points\nfor Manning check")
    else:
        _stats_box(ax, "No profile data available")

    # Culvert verification text
    ax = axes[1, 1]
    lines = ["Culvert Verification"]
    lines.append("─" * 35)
    for sa in struct_attrs:
        sid = sa.get("structure_id", "?")
        width = sa.get("width", sa.get("culvert_span", 0))
        height = sa.get("height", sa.get("culvert_rise", 0))
        slp = sa.get("culvert_slope", 0)
        n_val = sa.get("roughness_n", 0.02)
        # Try to get model results for this structure
        oid = str(sa.get("fid", sid))
        fin = struct_coupling.get("final", {}).get(oid, {})
        q_mod = fin.get("flow", 0)
        cap_mod = fin.get("manning_cap", 0)
        hw_mod = fin.get("available_head_up", 0)
        tw_mod = fin.get("tailwater_depth", 0)
        # Full-flow hydraulic radius for a box culvert: R = A/P = (w*h)/(2*(w+h))
        r_full = (width * height) / (2.0 * (width + height)) if width > 0 and height > 0 else 0.001
        q_full = mannings_full_flow(n_val, width * height, r_full, slp)
        if q_full > 0:
            lines.append(f"Structure {sid}:")
            lines.append(f"  Q_model={q_mod:.1f}, Full-flow={q_full:.0f}")
            lines.append(f"  ManningCap_model={cap_mod}")
            if q_mod > 0:
                y_n = normal_depth_box_culvert(q_mod, width, n_val, slp)
                lines.append(f"  Normal depth y_n ≈ {y_n:.2f}")
            if hw_mod > 0 and tw_mod > 0 and width * height > 0:
                hw_calc = outlet_control_hw(
                    q_mod, width * height,
                    (width * height) / (width + 2 * height),
                    sa.get("length", 80), n_val,
                    sa.get("entrance_loss_k", 0.5),
                    tw_mod,
                )
                lines.append(f"  HW_outlet_calc={hw_calc:.2f}")
                lines.append(f"  HW_model={hw_mod:.2f}, Δ={abs(hw_calc-hw_mod):.2f}")
            lines.append("")
    if len(lines) == 1:
        lines.append("No structure data available")
    _stats_box(ax, "\n".join(lines), fontsize=9)

    plt.tight_layout()
    _save_fig(fig, out)


# ===================================================================
#  Report generation (Markdown)
# ===================================================================


def _fmt(v: Any, decimals: int = 4) -> str:
    """Format a value nicely — if it's a float with small magnitude use scientific."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        if abs(v) < 0.001:
            return f"{v:.2e}"
        if decimals == 4:
            return f"{v:.4f}"
        return f"{v:.{decimals}f}"
    return str(v)


def generate_report(
    gpkg_path: str,
    run_id: str,
    mesh_dict: Dict[str, np.ndarray],
    mesh_stats: Dict[str, Any],
    prof: Optional[Dict[str, np.ndarray]],
    line_ts: Optional[Dict[str, np.ndarray]],
    struct_coupling: Dict[str, Any],
    struct_attrs: List[Dict[str, Any]],
    sample_lines: List[Dict[str, Any]],
    cons: Dict[str, Any],
    bc_flux: Optional[Dict[str, Dict[str, np.ndarray]]],
    bc_lines: List[Dict[str, Any]],
    log_metadata: Dict[str, Any],
    log_tail: str,
    out_dir: Path,
    t_final: float,
) -> str:
    """Assemble the Markdown report and return the full text."""
    lines: List[str] = []

    # ---- Header ----
    lines.append(f"# SWE2D Model Review Report")
    lines.append("")
    lines.append(f"**Run ID:** `{run_id}`")
    dur_s = log_metadata.get("duration_s") or 0
    lines.append(f"**Simulation Duration:** {dur_s:.0f} s ({dur_s/3600:.2f} hr)")
    lines.append(f"**GeoPackage:** `{gpkg_path}`")
    lines.append(f"**Report Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    # ---- 1. Overview ----
    lines.append("---")
    lines.append("## 1. Simulation Overview")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Mesh Cells | {mesh_stats.get('total_cells', '?')} |")
    lines.append(f"| Wet Cells (final) | {mesh_stats.get('wet_cells', '?')} ({mesh_stats.get('wet_cells', 0)/max(mesh_stats.get('total_cells',1),1)*100:.1f}%) |")
    lines.append(f"| Dry Cells (final) | {mesh_stats.get('dry_cells', '?')} |")
    lines.append(f"| Final Timestep | {t_final:.2f} s ({t_final/3600:.3f} hr) |")
    for bc in bc_lines:
        lines.append(f"| BC Line fid={bc['fid']} | type={bc['bc_type']} value={bc['bc_value']} |")
    lines.append(f"| Structures | {len(struct_attrs)} |")
    lines.append(f"| Sample Lines | {len(sample_lines)} |")
    lines.append("")

    # ---- 2. Mesh Results ----
    lines.append("---")
    lines.append("## 2. Mesh Results (Final Timestep)")
    lines.append("")
    lines.append(f"![Mesh Results Overview](fig1_mesh_results_overview.png)")
    lines.append("")
    if HAS_MPL:
        lines.append(f"*See `{out_dir / 'fig1_mesh_results_overview.png'}`*")
        lines.append("")

    lines.append("### 2.1 Depth Statistics")
    lines.append("")
    lines.append("| Statistic | Wet Cells Only | All Cells |")
    lines.append("|---|---|---|")
    lines.append(f"| Mean Depth | {mesh_stats.get('h_avg_wet', '?'):.2f} | {mesh_stats.get('h_avg_all', '?'):.2f} |")
    lines.append(f"| Max Depth | {mesh_stats.get('h_max', '?'):.2f} | {mesh_stats.get('h_max', '?'):.2f} |")
    lines.append(f"| Min Depth (>0.001) | — | {mesh_stats.get('h_min', '?'):.2f} |")
    lines.append("")

    lines.append("### 2.2 Velocity Statistics")
    lines.append("")
    h_w = mesh_dict["h"][mesh_dict["h"] > 0.001]
    vel = np.sqrt(mesh_dict["hu"][mesh_dict["h"] > 0.001] ** 2 +
                  mesh_dict["hv"][mesh_dict["h"] > 0.001] ** 2) / h_w
    lines.append("| Statistic | Value |")
    lines.append("|---|---|")
    lines.append(f"| Mean Velocity (wet) | {np.mean(vel):.2f} |")
    lines.append(f"| Max Velocity | {np.max(vel):.2f} |")
    lines.append(f"| Mean u | {mesh_stats.get('u_avg_wet', '?'):.2f} |")
    lines.append(f"| Mean v | {mesh_stats.get('v_avg_wet', '?'):.2f} |")
    fr_all = vel / np.sqrt(_G * h_w)
    lines.append(f"| Mean Froude | {np.mean(fr_all):.2f} |")
    lines.append(f"| Max Froude | {np.max(fr_all):.2f} |")
    lines.append("")

    # ---- 3. Sample Lines ----
    if prof is not None and line_ts is not None:
        lines.append("---")
        lines.append("## 3. Sample Line Results")
        lines.append("")
        lines.append(f"![Profile](fig2_sample_line_profile.png)")
        lines.append(f"![Time-Series](fig3_line_ts_results.png)")
        lines.append("")

        wet = prof["wet"] > 0
        q_int = integrate_profile_flow(prof["station"], prof["flow_qn"], prof["wet"])
        lines.append("### 3.1 Profile at Final Timestep")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|---|---|")
        lines.append(f"| Profile Length | {prof['station'][-1]:.2f} |")
        wet_width = np.sum(np.diff(prof['station']) * (wet[:-1].astype(bool) & wet[1:].astype(bool)))
        lines.append(f"| Wetted Width | {wet_width:.2f} |")
        lines.append(f"| Mean Depth (wet) | {np.mean(prof['depth'][wet]):.4f} |")
        lines.append(f"| Mean Velocity | {np.mean(prof['velocity'][wet]):.4f} |")
        lines.append(f"| Flow (line_ts) | {line_ts['flow'][-1]:.2f} |")
        lines.append(f"| Integrated Profile Q | {q_int:.2f} |")
        lines.append(f"| Froude (mean) | {np.mean(prof['fr'][wet]):.2f} |")
        lines.append("")
        lines.append(f"*Profile flow integration: Δ = {abs(q_int - line_ts['flow'][-1]):.2f}"
                     f" ({abs(q_int - line_ts['flow'][-1]) / max(line_ts['flow'][-1], 0.001) * 100:.1f}%)*")
        lines.append("")

    # ---- 4. Structures ----
    if struct_attrs:
        lines.append("---")
        lines.append("## 4. Structure Results")
        lines.append("")
        lines.append(f"![Structures](fig4_structure_results.png)")
        lines.append("")

        for sa in struct_attrs:
            sid = sa.get("structure_id", "?")
            lines.append(f"### 4.{struct_attrs.index(sa)+1} Structure {sid}")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|---|---|")
            for k, v in sa.items():
                if k.startswith("_"):
                    continue
                if v is not None:
                    lines.append(f"| {k} | {v} |")
            lines.append("")
            # Model results
            oid = str(sa.get("fid", sid))
            fin = struct_coupling.get("final", {}).get(oid, {})
            if fin:
                lines.append("**Model results at final timestep:**")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|---|---|")
                for mk, mv in fin.items():
                    lines.append(f"| {mk} | {_fmt(mv)} |")
                lines.append("")

    # ---- 5. Conservation & Residuals ----
    lines.append("---")
    lines.append("## 5. Mass Conservation & Residuals")
    lines.append("")
    lines.append(f"![Conservation](fig5_conservation_residuals.png)")
    lines.append("")

    cons_sum = cons.get("summary", {})
    lines.append("| Component | Volume (ft³) | Volume (m³) |")
    lines.append("|---|---|---|")
    lines.append(f"| ΔStorage | {_fmt(cons_sum.get('storage_delta_model'))} | {_fmt(cons_sum.get('storage_delta_m3'))} |")
    lines.append(f"| BC Inflow | {_fmt(cons_sum.get('boundary_group_volume_sum_model'))} | {_fmt(cons_sum.get('boundary_group_volume_sum_m3'))} |")
    lines.append(f"| Closure Residual | — | {_fmt(cons_sum.get('closure_residual_m3'))} |")
    lines.append(f"| Source Total | {_fmt(cons_sum.get('source_total_model'))} | {_fmt(cons_sum.get('source_total_m3'))} |")
    lines.append("")

    # ---- 6. Independent Verification ----
    lines.append("---")
    lines.append("## 6. Independent Verification")
    lines.append("")
    lines.append(f"![Verification](fig6_independent_verification.png)")
    lines.append("")

    # Sample line
    if prof is not None and line_ts is not None:
        q_int = integrate_profile_flow(prof["station"], prof["flow_qn"], prof["wet"])
        lines.append("### 6.1 Sample Line Flow Integration")
        lines.append("")
        lines.append("| Method | Flow | vs Reported |")
        lines.append("|---|---|---|")
        lines.append(f"| Model Reported (line_ts) | {line_ts['flow'][-1]:.2f} | — |")
        lines.append(f"| Integrated flow_qn × ds | {q_int:.2f} | {abs(q_int-line_ts['flow'][-1])/max(line_ts['flow'][-1],0.001)*100:.1f}% |")
        lines.append("")

    # Culvert checks
    if struct_attrs:
        lines.append("### 6.2 Culvert Hydraulics")
        lines.append("")
        for sa in struct_attrs:
            sid = sa.get("structure_id", "?")
            width = sa.get("width", sa.get("culvert_span", 0))
            height = sa.get("height", sa.get("culvert_rise", 0))
            slp = sa.get("culvert_slope", 0)
            n_val = sa.get("roughness_n", 0.02)
            oid = str(sa.get("fid", sid))
            fin = struct_coupling.get("final", {}).get(oid, {})
            q_mod = fin.get("flow", 0)
            cap_mod = fin.get("manning_cap", 0)
            hw_mod = fin.get("available_head_up", 0)
            tw_mod = fin.get("tailwater_depth", 0)

            if width and height and slp > 0:
                A = width * height
                # Full-flow wetted perimeter for a box culvert
                P = 2.0 * (width + height)
                R = A / P if P > 0 else 0.001
                q_full = mannings_full_flow(n_val, A, R, slp)

                lines.append(f"**Structure {sid}** (box {width}×{height}, S={slp}, n={n_val})")
                lines.append("")
                lines.append("| Check | Independent | Model | Δ |")
                lines.append("|---|---|---|---|")
                lines.append(f"| Manning Full-Flow | {q_full:.0f} | {cap_mod} | {abs(q_full-cap_mod)/max(cap_mod,1)*100:.0f}% |")
                if q_mod > 0:
                    yn = normal_depth_box_culvert(q_mod, width, n_val, slp)
                    lines.append(f"| Normal Depth | {yn:.2f} | — | — |")
                if hw_mod > 0 and tw_mod > 0 and A > 0:
                    hw_c = outlet_control_hw(q_mod, A, R, sa.get("length", 80), n_val,
                                              sa.get("entrance_loss_k", 0.5), tw_mod)
                    lines.append(f"| Outlet Control HW | {hw_c:.2f} | {hw_mod:.2f} | {abs(hw_c-hw_mod):.2f} |")
                lines.append("")

    # ---- 7. Residuals & Numerical Performance ----
    lines.append("---")
    lines.append("## 7. Residuals & Numerical Performance")
    lines.append("")

    # Parse WSE residual from last log lines
    wse_res = "see log"
    for line in log_tail.split("\n"):
        if "WSEres=" in line:
            parts = line.split("WSEres=")
            if len(parts) > 1:
                wse_res = parts[1].split()[0]
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| WSE Residual (final) | {wse_res} |")
    lines.append(f"| Closure Residual | {_fmt(cons_sum.get('closure_residual_m3'))} m³ |")
    lines.append(f"| Source Total | {_fmt(cons_sum.get('source_total_m3'))} m³ |")
    lines.append(f"| Total Steps | {len(log_tail.split(chr(10)))} (estimated) |")
    lines.append(f"| Wallclock Duration | {dur_s:.1f} s |")
    lines.append("")

    # ---- Footer ----
    lines.append("---")
    lines.append(f"*Report generated by `generate_swe2d_report.py`*")
    lines.append(f"*Run ID: {run_id}*")
    lines.append("")

    return "\n".join(lines)


# ===================================================================
#  Main entry point
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a SWE2D model-review report from a GeoPackage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s model.gpkg
              %(prog)s model.gpkg --run-id swe2d_20260601T095937-0500
              %(prog)s model.gpkg --out-dir my_report
        """),
    )
    parser.add_argument("gpkg", help="Path to the SWE2D GeoPackage")
    parser.add_argument("--run-id", "-r", default=None, help="Run ID (default: latest)")
    parser.add_argument(
        "--out-dir", "-o", default=None,
        help="Output directory (default: <gpkg_stem>_report/)",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    gpkg_path = Path(args.gpkg)
    if not gpkg_path.exists():
        print(f"ERROR: GeoPackage not found: {gpkg_path}", file=sys.stderr)
        sys.exit(1)

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = gpkg_path.parent / f"{gpkg_path.stem}_report"
    _make_figure_dir(out_dir)

    # Connect
    db = sqlite3.connect(str(gpkg_path))
    db.execute("PRAGMA journal_mode=WAL")  # safe for concurrent readers

    # Discover run
    run_id = args.run_id or _latest_run(db)
    if run_id is None:
        print("ERROR: No runs found in GeoPackage.", file=sys.stderr)
        db.close()
        sys.exit(1)
    print(f"Using run_id: {run_id}")

    # Metadata, duration, and log
    meta = _parse_log_metadata(db, run_id)
    dur_row = db.execute(
        "SELECT duration_s FROM swe2d_run_logs WHERE run_id=?", (run_id,)
    ).fetchone()
    dur_s = float(dur_row[0]) if dur_row and dur_row[0] else 0.0
    meta["duration_s"] = dur_s
    log_tail = _get_log_tail(db, run_id)

    # Timesteps
    t_final = _final_timestep(db, run_id, "swe2d_mesh_results")
    if t_final is None:
        print("ERROR: No mesh results found.", file=sys.stderr)
        db.close()
        sys.exit(1)
    print(f"Final timestep: {t_final:.4f} s")

    # Extract data
    print("Extracting mesh data ...")
    mesh_arrays = extract_mesh_arrays(db, run_id, t_final)
    mesh_stats = extract_mesh_stats(db, run_id, t_final)

    print("Extracting sample-line data ...")
    t_line = _final_timestep(db, run_id, "swe2d_line_results_profile")
    prof = None
    if t_line:
        prof = extract_profile_data(db, run_id, t_line)
    line_ts = extract_line_ts(db, run_id)

    print("Extracting structure data ...")
    struct_coupling = extract_structure_coupling(db, run_id)
    struct_attrs = extract_structure_attributes(db, str(gpkg_path))
    sample_lines = extract_sample_line_attributes(db)

    print("Extracting conservation / boundary data ...")
    cons = extract_conservation_data(db, run_id) or {"summary": {}, "ts": {}}
    bc_flux = extract_boundary_flux(db, run_id)
    bc_lines = extract_bc_lines(db)

    db.close()

    # ---- Figures ----
    if HAS_MPL and not args.no_figures:
        print("Generating figures ...")
        figure_mesh_overview(mesh_arrays, out_dir / "fig1_mesh_results_overview.png")
        if prof is not None:
            figure_sample_line_profile(prof, out_dir / "fig2_sample_line_profile.png")
        if line_ts is not None:
            figure_line_ts(line_ts, out_dir / "fig3_line_ts_results.png")
        if struct_coupling:
            figure_structures(struct_coupling, struct_attrs, out_dir / "fig4_structure_results.png")
        figure_conservation(cons, bc_flux, line_ts, struct_coupling, run_id,
                           out_dir / "fig5_conservation_residuals.png")
        figure_independent_verification(prof, line_ts, struct_coupling, struct_attrs,
                                       out_dir / "fig6_independent_verification.png")
        print("Figures saved.")
    else:
        print("Skipping figures (no matplotlib or --no-figures flag).")

    # ---- Report ----
    print("Assembling report ...")
    report_md = generate_report(
        str(gpkg_path), run_id,
        mesh_arrays, mesh_stats,
        prof, line_ts,
        struct_coupling, struct_attrs,
        sample_lines,
        cons, bc_flux, bc_lines,
        meta, log_tail,
        out_dir, t_final,
    )
    report_path = out_dir / "swe2d_model_review_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"Report complete!")
    print(f"  Report:  {report_path}")
    print(f"  Figures: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
