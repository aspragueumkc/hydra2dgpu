"""CLI: headless GPU render of simulation frames to PNG.

Modes:
  single  — one frame from a baked results GPKG
  multi   — every stored timestep → PNG sequence
  live    — poll snapshot ring buffer → PNG as sim runs

Usage:
  hydra_viewer --mode single --gpkg results.gpkg --run-id RUN_ID \
               --field depth --width 1280 --height 720 \
               --output frame.png
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence, Tuple

import numpy as np


# ── Colormap helpers ───────────────────────────────────────────────────────
#
# Duplicated from swe2d.results.high_perf_viewer._COLOR_LUTS rather than
# importing — that module instantiates QGIS classes (QgsMapCanvasItem) at
# module load, which crashes outside a QGIS session.  Keep these LUTs in
# sync with _COLOR_LUTS in swe2d/results/high_perf_viewer.py.

def _build_color_lut(stops: Sequence[Tuple[float, Tuple[int, int, int]]]) -> np.ndarray:
    x = np.asarray([float(s[0]) for s in stops], dtype=np.float64)
    r = np.asarray([float(s[1][0]) for s in stops], dtype=np.float64)
    g = np.asarray([float(s[1][1]) for s in stops], dtype=np.float64)
    b = np.asarray([float(s[1][2]) for s in stops], dtype=np.float64)
    xi = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    lut = np.zeros((256, 3), dtype=np.uint8)
    lut[:, 0] = np.clip(np.interp(xi, x, r), 0.0, 255.0).astype(np.uint8)
    lut[:, 1] = np.clip(np.interp(xi, x, g), 0.0, 255.0).astype(np.uint8)
    lut[:, 2] = np.clip(np.interp(xi, x, b), 0.0, 255.0).astype(np.uint8)
    return lut


_COLOR_LUTS = {
    "turbo": _build_color_lut([
        (0.0,  (48, 18, 59)),
        (0.20, (50, 100, 220)),
        (0.40, (41, 187, 236)),
        (0.60, (124, 234, 87)),
        (0.80, (250, 205, 32)),
        (1.0,  (180, 4, 38)),
    ]),
    "viridis": _build_color_lut([
        (0.0,  (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0,  (253, 231, 37)),
    ]),
    "gray": _build_color_lut([(0.0, (0, 0, 0)), (1.0, (255, 255, 255))]),
}


def build_colormap_lut(key: str = "turbo") -> np.ndarray:
    """Build a 256-entry RGBA LUT (uint8, row-major)."""
    if key not in _COLOR_LUTS:
        raise ValueError(f"unknown colormap {key!r}")
    rgb = _COLOR_LUTS[key]
    out = np.zeros((256, 4), dtype=np.uint8)
    out[:, 0] = rgb[:, 0]
    out[:, 1] = rgb[:, 1]
    out[:, 2] = rgb[:, 2]
    out[:, 3] = 255
    return out


# ── GPKG loaders (Phase 2 MVP — minimal subset) ─────────────────────────────

def load_mesh_from_gpkg(gpkg_path: str, run_id: str) -> dict:
    """Load the baked mesh BLOB from a results GPKG.

    Returns dict with keys: node_x, node_y, node_z, cell_nodes, cell_x, cell_y.
    Raises RuntimeError if the GPKG / run_id is missing or malformed.
    """
    from swe2d.results.queries import load_baked_mesh
    mesh = load_baked_mesh(gpkg_path, run_id)
    required = {"node_x", "node_y", "cell_nodes", "cell_x", "cell_y"}
    missing = required - set(mesh.keys())
    if missing:
        raise RuntimeError(
            f"mesh BLOB for run_id={run_id!r} missing keys: {sorted(missing)}"
        )
    if "node_z" not in mesh:
        mesh["node_z"] = np.zeros_like(mesh["node_x"])
    return mesh


def load_field_from_gpkg(gpkg_path: str, run_id: str, field: str,
                          timestep: Optional[float]) -> np.ndarray:
    """Load a single field frame (h, hu, or hv) from a baked results GPKG.

    Returns a (n_cells,) ndarray.
    """
    from swe2d.results.queries import load_baked_field
    return np.asarray(load_baked_field(gpkg_path, run_id, field, timestep))


def load_timesteps_from_gpkg(gpkg_path: str, run_id: str, field: str) -> list:
    """Return the sorted list of stored timesteps for (run_id, field)."""
    from swe2d.results.queries import load_baked_timesteps
    return list(load_baked_timesteps(gpkg_path, run_id, field))


# ── Render + save ──────────────────────────────────────────────────────────

def render_field_rgba(solver, field_key: str, vmin: float, vmax: float,
                      width: int, height: int, cell_x: np.ndarray,
                      cell_y: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Call the Phase 2.1 binding; returns (H, W, 4) uint8 ndarray."""
    import hydra_swe2d as m
    result = m.swe2d_gpu_render_field_to_rgba(
        solver, field_key, vmin, vmax,
        width, height, lut,
        cell_x, cell_y,
        float(cell_x.min()), float(cell_x.max()),
        float(cell_y.min()), float(cell_y.max()),
    )
    return np.asarray(result["image"])


def save_png(rgba: np.ndarray, output_path: str) -> None:
    """Write RGBA ndarray as PNG to *output_path*."""
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(output_path)


# ── Command handlers ───────────────────────────────────────────────────────

def _build_solver_for_render(mesh: dict, field: np.ndarray):
    """Build a PySolver so the Phase 2 binding can pull device state.

    Phase 2 MVP: we don't actually run the solver — we just need the
    device-side field buffer (d_h / d_hu / d_hv) populated with *field*.
    The runtime reporter normally populates d_h after each step; here we
    upload *field* directly via the solver's existing ``swe2d_set_state``
    API if available, or fall back to running one no-op step.
    """
    import hydra_swe2d as m
    py_mesh = m.swe2d_build_mesh(
        mesh["node_x"], mesh["node_y"], mesh["node_z"],
        mesh["cell_nodes"],
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64),
    )
    solver = m.swe2d_create_solver(
        py_mesh, field.astype(np.float64, copy=False),
        n_mann=0.0, cfl=0.45, dt_max=0.5, use_gpu=True,
    )
    return solver


def cmd_single(args) -> int:
    """Render a single frame from a baked GPKG to PNG."""
    mesh = load_mesh_from_gpkg(args.gpkg, args.run_id)
    field = load_field_from_gpkg(args.gpkg, args.run_id,
                                  args.field, args.timestep)
    if args.vmin is None:
        args.vmin = float(np.nanmin(field))
    if args.vmax is None:
        args.vmax = float(np.nanmax(field))
    if args.vmax <= args.vmin:
        args.vmax = args.vmin + 1.0

    solver = _build_solver_for_render(mesh, field)
    try:
        lut = build_colormap_lut(args.cmap)
        rgba = render_field_rgba(
            solver, args.field, args.vmin, args.vmax,
            args.width, args.height,
            mesh["cell_x"], mesh["cell_y"], lut,
        )
        save_png(rgba, args.output)
    finally:
        import hydra_swe2d as m
        m.swe2d_destroy(solver)
    print(f"wrote {args.output} ({args.width}x{args.height})")
    return 0


def cmd_multi(args) -> int:
    """Render every stored timestep to a PNG sequence."""
    timesteps = load_timesteps_from_gpkg(args.gpkg, args.run_id, args.field)
    os.makedirs(args.output_dir, exist_ok=True)
    for i, ts in enumerate(timesteps):
        single_args = argparse.Namespace(**vars(args))
        single_args.mode = "single"
        single_args.timestep = ts
        single_args.output = os.path.join(
            args.output_dir, f"frame_{i:04d}.png"
        )
        cmd_single(single_args)
    print(f"wrote {len(timesteps)} frames to {args.output_dir}")
    return 0


def cmd_live(args) -> int:
    """Poll snapshot ring buffer, save PNG per new snapshot.

    Honors ``--max-frames`` (0 = unlimited) for batch/test runs.

    Snapshot readback is done inline via the ``swe2d_gpu_read_snapshots``
    / ``swe2d_gpu_snapshot_count`` bindings — no service-class wrapper.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    mesh = load_mesh_from_gpkg(args.gpkg, args.run_id) if args.gpkg else args.mesh
    lut = build_colormap_lut(args.cmap)
    i = 0
    last_t = -1.0
    import hydra_swe2d as _h
    while True:
        if args.max_frames and i >= args.max_frames:
            break
        if _h.swe2d_gpu_snapshot_count(args.solver) <= 0:
            import time
            time.sleep(0.5)
            continue
        snap = _h.swe2d_gpu_read_snapshots(args.solver)
        if not snap or "t_s" not in snap:
            import time
            time.sleep(0.5)
            continue
        t_s_arr = snap["t_s"]
        if t_s_arr.size == 0:
            import time
            time.sleep(0.5)
            continue
        last = int(t_s_arr.size) - 1
        t_s = float(t_s_arr[last])
        if t_s <= last_t:
            import time
            time.sleep(0.5)
            continue
        last_t = t_s
        if args.vmin is None or args.vmax is None:
            h_latest = snap["h"][last]
            args.vmin = float(np.nanmin(h_latest))
            args.vmax = float(np.nanmax(h_latest))
            if args.vmax <= args.vmin:
                args.vmax = args.vmin + 1.0
        rgba = render_field_rgba(
            args.solver, "h", args.vmin, args.vmax,
            args.width, args.height,
            mesh["cell_x"], mesh["cell_y"], lut,
        )
        save_png(rgba, os.path.join(args.output_dir, f"live_{i:04d}.png"))
        i += 1
    print(f"wrote {i} frames to {args.output_dir}")
    return 0


# ── argparse ───────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hydra_viewer",
        description="Headless GPU renderer for HYDRA simulation frames.",
    )
    p.add_argument("--mode", choices=["single", "multi", "live"], required=True)
    p.add_argument("--gpkg", help="results GPKG path (required for single/multi)")
    p.add_argument("--run-id", help="run id within the GPKG")
    p.add_argument("--field", choices=["depth", "hu", "hv"], default="depth")
    p.add_argument("--timestep", type=float, default=None,
                   help="simulation time to render; nearest snapshot is used")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--cmap", default="turbo")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--output", help="PNG path (single mode)")
    p.add_argument("--output-dir", help="output directory (multi/live modes)")
    p.add_argument("--solver", help="PySolver handle (live mode)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="live mode: stop after N frames (0 = unlimited)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "single":
        if not args.gpkg or not args.run_id or not args.output:
            print("single mode requires --gpkg, --run-id, --output",
                  file=sys.stderr)
            return 2
        return cmd_single(args)
    if args.mode == "multi":
        if not args.gpkg or not args.run_id or not args.output_dir:
            print("multi mode requires --gpkg, --run-id, --output-dir",
                  file=sys.stderr)
            return 2
        return cmd_multi(args)
    if args.mode == "live":
        if not args.output_dir or not args.solver:
            print("live mode requires --output-dir, --solver", file=sys.stderr)
            return 2
        return cmd_live(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())