#!/usr/bin/env python3
"""Generate standard figures from a baked-results GeoPackage.

This is a headless, QGIS-free plotting utility intended for batch/post-run
figure generation.  It reads the baked mesh and result BLOBs directly from
``swe2d_baked_mesh``, ``swe2d_baked_results`` and ``swe2d_baked_line_ts`` and
writes PNG files to an output directory.

Usage:
    python tools/plot_baked_results.py path/to/results.gpkg
    python tools/plot_baked_results.py path/to/results.gpkg --outdir ./figures
    python tools/plot_baked_results.py path/to/results.gpkg --run-id <uuid>
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _require_module():
    from swe2d.runtime.backend import load_swe2d_native_module
    mod = load_swe2d_native_module()
    if mod is None:
        raise RuntimeError(
            "hydra_swe2d extension is required to deserialize baked meshes. "
            "Make sure the project is built/installed in the active environment."
        )
    return mod


def _load_mesh_arrays(gpkg_path: str, mesh_name: str) -> Dict[str, np.ndarray]:
    """Load node/cell geometry from the baked mesh BLOB."""
    mod = _require_module()
    conn = sqlite3.connect(gpkg_path)
    try:
        row = conn.execute(
            "SELECT baked_blob FROM swe2d_baked_mesh WHERE mesh_name=?",
            (mesh_name,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise FileNotFoundError(f"No baked mesh named {mesh_name!r} in {gpkg_path}")

    pm = mod.swe2d_deserialize_mesh(row[0])
    node_x = np.asarray(pm.node_x, dtype=np.float64)
    node_y = np.asarray(pm.node_y, dtype=np.float64)
    node_z = np.asarray(pm.node_z, dtype=np.float64)
    cell_nodes = np.asarray(pm.cell_face_nodes, dtype=np.int32)
    cell_face_offsets = np.asarray(pm.cell_face_offsets, dtype=np.int32)

    n_cells = cell_face_offsets.size - 1
    nv_per_cell = cell_face_offsets[1:] - cell_face_offsets[:-1]
    if not np.all(nv_per_cell == 3):
        raise NotImplementedError(
            "plot_baked_results currently only supports triangular cell meshes"
        )

    triangles = cell_nodes.reshape(n_cells, 3)

    # cell bed elevation as mean of corner node z
    bed_z = np.mean(node_z[triangles], axis=1)

    # cell centroids for optional scatter/checks
    cx = np.mean(node_x[triangles], axis=1)
    cy = np.mean(node_y[triangles], axis=1)

    return {
        "node_x": node_x,
        "node_y": node_y,
        "node_z": node_z,
        "triangles": triangles,
        "bed_z": bed_z,
        "cx": cx,
        "cy": cy,
        "n_cells": n_cells,
    }


def _load_run_results(gpkg_path: str, run_id: str) -> Dict[str, np.ndarray]:
    """Load snapshot arrays for a run."""
    conn = sqlite3.connect(gpkg_path)
    try:
        row = conn.execute(
            "SELECT n_timesteps, n_cells, times_blob, h_blob, hu_blob, hv_blob, "
            "max_h_blob, max_hu_blob, max_hv_blob "
            "FROM swe2d_baked_results WHERE run_id=?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise FileNotFoundError(f"No results for run_id={run_id!r} in {gpkg_path}")

    n_ts, n_cells = int(row[0]), int(row[1])
    times = np.frombuffer(row[2], dtype=np.float64)
    h_all = np.frombuffer(row[3], dtype=np.float64).reshape(n_ts, n_cells)
    hu_all = np.frombuffer(row[4], dtype=np.float64).reshape(n_ts, n_cells)
    hv_all = np.frombuffer(row[5], dtype=np.float64).reshape(n_ts, n_cells)

    if row[6] is not None:
        max_h = np.frombuffer(row[6], dtype=np.float64)
        max_hu = np.frombuffer(row[7], dtype=np.float64)
        max_hv = np.frombuffer(row[8], dtype=np.float64)
    else:
        max_h = np.max(h_all, axis=0)
        max_hu = np.max(hu_all, axis=0)
        max_hv = np.max(hv_all, axis=0)

    return {
        "times": times,
        "h_all": h_all,
        "hu_all": hu_all,
        "hv_all": hv_all,
        "max_h": max_h,
        "max_hu": max_hu,
        "max_hv": max_hv,
    }


def _velocity_magnitude(h: np.ndarray, hu: np.ndarray, hv: np.ndarray) -> np.ndarray:
    """Compute velocity magnitude (m/s in model units) from conserved variables."""
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.where(h > 1e-6, hu / h, 0.0)
        v = np.where(h > 1e-6, hv / h, 0.0)
    return np.sqrt(u * u + v * v)


def _plot_field(
    node_x: np.ndarray,
    node_y: np.ndarray,
    triangles: np.ndarray,
    facevalues: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    colorbar_label: str = "",
) -> None:
    """Render a cell-centered scalar field to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    triang = matplotlib.tri.Triangulation(node_x, node_y, triangles=triangles)
    fig, ax = plt.subplots(figsize=(10, 8))
    tpc = ax.tripcolor(triang, facecolors=facevalues, cmap=cmap, shading="flat")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    cb = fig.colorbar(tpc, ax=ax)
    if colorbar_label:
        cb.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_summary_timeseries(
    times: np.ndarray,
    h_all: np.ndarray,
    out_path: Path,
) -> None:
    """Plot summary global metrics over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_h_ts = np.max(h_all, axis=1)
    mean_h_ts = np.mean(h_all, axis=1)
    wet_cells_ts = np.sum(h_all > 1e-6, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(times, max_h_ts)
    axes[0].set_ylabel("Max depth")
    axes[0].set_title("Summary time series")

    axes[1].plot(times, mean_h_ts)
    axes[1].set_ylabel("Mean depth")

    axes[2].plot(times, wet_cells_ts)
    axes[2].set_ylabel("Wet cells")
    axes[2].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


def _plot_line_timeseries(
    gpkg_path: str,
    run_id: str,
    line_id: int,
    line_name: str,
    out_path: Path,
) -> bool:
    """Plot a single line time series if data exists."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conn = sqlite3.connect(gpkg_path)
    try:
        row = conn.execute(
            "SELECT n_timesteps, times_blob, depth_blob, vel_blob, wse_blob, "
            "bed_blob, flow_blob FROM swe2d_baked_line_ts "
            "WHERE run_id=? AND line_id=?",
            (run_id, line_id),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return False

    times = np.frombuffer(row[1], dtype=np.float64)
    depth = np.frombuffer(row[2], dtype=np.float64)
    vel = np.frombuffer(row[3], dtype=np.float64)
    wse = np.frombuffer(row[4], dtype=np.float64)
    bed = np.frombuffer(row[5], dtype=np.float64)
    flow = np.frombuffer(row[6], dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(times, depth)
    axes[0].set_ylabel("Depth")
    axes[0].set_title(f"Line {line_id}: {line_name}")

    axes[1].plot(times, vel)
    axes[1].set_ylabel("Velocity")

    axes[2].plot(times, wse, label="WSE")
    axes[2].plot(times, bed, label="Bed")
    axes[2].set_ylabel("Elevation")
    axes[2].legend()

    axes[3].plot(times, flow)
    axes[3].set_ylabel("Flow")
    axes[3].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_path)
    return True


def _list_run_ids(gpkg_path: str) -> List[Tuple[str, str]]:
    """Return list of (run_id, mesh_name) tuples."""
    conn = sqlite3.connect(gpkg_path)
    try:
        return list(conn.execute("SELECT run_id, mesh_name FROM swe2d_baked_results"))
    finally:
        conn.close()


def _list_lines(gpkg_path: str, run_id: str) -> List[Tuple[int, str]]:
    """Return list of (line_id, line_name) tuples for a run."""
    conn = sqlite3.connect(gpkg_path)
    try:
        return list(
            conn.execute(
                "SELECT line_id, line_name FROM swe2d_baked_line_ts WHERE run_id=?",
                (run_id,),
            )
        )
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate standard figures from a SWE2D baked-results GeoPackage."
    )
    parser.add_argument("gpkg", help="Path to the .gpkg results file")
    parser.add_argument(
        "--outdir",
        "-o",
        default=None,
        help="Output directory (default: <gpkg_stem>_figures)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run UUID to plot (default: first run in the file)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    gpkg_path = Path(args.gpkg).resolve()
    if not gpkg_path.exists():
        logger.error("GeoPackage not found: %s", gpkg_path)
        return 1

    runs = _list_run_ids(str(gpkg_path))
    if not runs:
        logger.error("No baked results found in %s", gpkg_path)
        return 1

    run_id = args.run_id
    if run_id is None:
        run_id = runs[0][0]
        logger.info("Using first run: %s", run_id)
    else:
        if not any(r[0] == run_id for r in runs):
            logger.error("run_id %s not found in %s", run_id, gpkg_path)
            return 1

    mesh_name = next(r[1] for r in runs if r[0] == run_id)
    outdir = Path(args.outdir) if args.outdir else gpkg_path.parent / f"{gpkg_path.stem}_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    mesh = _load_mesh_arrays(str(gpkg_path), mesh_name)
    results = _load_run_results(str(gpkg_path), run_id)

    n_ts = results["times"].size
    final_h = results["h_all"][-1]
    final_hu = results["hu_all"][-1]
    final_hv = results["hv_all"][-1]
    final_vel = _velocity_magnitude(final_h, final_hu, final_hv)

    _plot_field(
        mesh["node_x"],
        mesh["node_y"],
        mesh["triangles"],
        results["max_h"],
        f"Max water depth (run {run_id[:8]})",
        outdir / "max_h.png",
        cmap="Blues",
        colorbar_label="Depth",
    )
    _plot_field(
        mesh["node_x"],
        mesh["node_y"],
        mesh["triangles"],
        final_h,
        f"Final water depth at t={results['times'][-1]:.2f}s",
        outdir / "final_h.png",
        cmap="Blues",
        colorbar_label="Depth",
    )
    _plot_field(
        mesh["node_x"],
        mesh["node_y"],
        mesh["triangles"],
        final_vel,
        f"Final velocity magnitude at t={results['times'][-1]:.2f}s",
        outdir / "final_vel.png",
        cmap="turbo",
        colorbar_label="Velocity",
    )
    _plot_field(
        mesh["node_x"],
        mesh["node_y"],
        mesh["triangles"],
        mesh["bed_z"],
        "Bed elevation",
        outdir / "bed_z.png",
        cmap="terrain",
        colorbar_label="Elevation",
    )

    if n_ts > 1:
        _plot_summary_timeseries(
            results["times"], results["h_all"], outdir / "summary_ts.png"
        )

    for line_id, line_name in _list_lines(str(gpkg_path), run_id):
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in line_name)
        _plot_line_timeseries(
            str(gpkg_path),
            run_id,
            line_id,
            line_name,
            outdir / f"line_{line_id}_{safe_name}.png",
        )

    logger.info("Figures written to %s", outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
