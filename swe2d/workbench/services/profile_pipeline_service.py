"""swe2d/workbench/services/profile_pipeline_service.py

Pure-Python service for assembling longitudinal profile data across a chain
of drainage links. Owns ALL numpy computation. Zero Qt.

Reads:
  - swe2d_drainage_links (link_id, from_node, to_node, length, shape, dims)
  - swe2d_drainage_nodes (node_id, invert_elev, rim_elev, max_depth)
  - swe2d_baked_pipe_cell_ts (per-cell timeseries with geometry columns)

Writes: returns ProfileArrays dataclass for the View to render.

The cell_sub_idx ordering is upstream→downstream within a link:
  cell_sub_idx == 0   at from_node end
  cell_sub_idx == n-1 at to_node end
A chain may include links traversed in reverse (upstream-to-downstream).
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ChainSpec:
    """User-chosen ordered chain of links. Each entry is (link_id, reverse)."""
    link_specs: list[tuple[str, bool]] = field(default_factory=list)

    def cumulative_links(self) -> List[str]:
        return [lid for lid, _ in self.link_specs]

    def is_empty(self) -> bool:
        return len(self.link_specs) == 0


@dataclass
class ProfileArrays:
    """Plain numpy arrays + station bookkeeping for rendering.

    Every numeric field is a 1-D array of length N == number of
    cell-centers along the chain (cumulative, ordered upstream→downstream).

    `crown_offset_m[i]` is the crown's vertical offset ABOVE `invert_m[i]`
    for the i-th cell, so the crown elevation is simply
    ``invert_m[i] + crown_offset_m[i]``.  Computing it as an offset
    (rather than a hard-coded elevation) lets the View shade the
    interior of each cell using the actual invert + crown geometry.
    """
    station_m: np.ndarray
    invert_m: np.ndarray
    crown_offset_m: np.ndarray
    crown_m: np.ndarray           # invert_m + crown_offset_m (convenience)
    hgl_m: np.ndarray
    depth_m: np.ndarray
    velocity_ms: np.ndarray
    flow_cms: np.ndarray
    node_stations: List[float]
    node_ids: List[str]
    link_boundaries: List[Tuple[int, str]]
    crown_style: str             # 'circular' | 'rectangular' | 'mixed'


VALID_METRICS = ("depth", "velocity", "flow", "head")


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _open_conn(gpkg_path: str) -> sqlite3.Connection:
    return sqlite3.connect(gpkg_path)


def load_pipe_cell_records(
    gpkg_path: str,
    run_id: str,
    link_ids: List[str],
) -> Dict[Tuple[str, int, str], np.ndarray]:
    """Read pipe-cell timeseries keyed by (link_id, cell_sub_idx, metric)."""
    if not gpkg_path or not os.path.exists(gpkg_path):
        return {}
    if not link_ids:
        return {}

    placeholders = ",".join("?" for _ in link_ids)
    try:
        conn = _open_conn(gpkg_path)
    except sqlite3.Error:
        return {}
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_pipe_cell_ts'"
        )
        if cur.fetchone() is None:
            return {}
        cur.execute(
            f"SELECT link_id, cell_sub_idx, metric, values_blob FROM {_quote_ident('swe2d_baked_pipe_cell_ts')} "
            f"WHERE run_id = ? AND link_id IN ({placeholders})",
            (run_id, *link_ids),
        )
        out: Dict[Tuple[str, int, str], np.ndarray] = {}
        for row in cur.fetchall():
            key = (str(row[0]), int(row[1]), str(row[2]))
            blob = row[3]
            if blob is None:
                continue
            try:
                arr = np.frombuffer(blob, dtype=np.float64)
            except Exception:
                continue
            out[key] = arr
        return out
    finally:
        conn.close()


@dataclass(frozen=True)
class CellGeometry:
    """Per-cell geometry. Same value repeated on every metric-row of
    the same (link_id, sub_idx) in the GPKG, so we read it once."""
    invert: float
    width: float
    height: float
    shape_type: int              # 0 = circular, 1+ = rectangular / other


def load_pipe_cell_geometry(
    gpkg_path: str,
    run_id: str,
    link_ids: List[str],
) -> Dict[Tuple[str, int], CellGeometry]:
    """Read per-cell geometry (invert, width, height, shape) for a set of links.

    Returns an empty dict on missing file/table/rows.  Missing geometry
    columns on legacy tables yield width=height=0.0, invert=0.0 defaults
    so callers can still proceed (with a degenerate cell shape).
    """
    if not gpkg_path or not os.path.exists(gpkg_path):
        return {}
    if not link_ids:
        return {}
    placeholders = ",".join("?" for _ in link_ids)
    try:
        conn = _open_conn(gpkg_path)
    except sqlite3.Error:
        return {}
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_pipe_cell_ts'"
        )
        if cur.fetchone() is None:
            return {}

        # Inspect available columns.  Legacy tables have only the 7 blob
        # columns; later versions add the 4 geometry columns.  We
        # coalesce with sensible defaults so the pipeline never raises.
        col_rows = cur.execute(
            f"PRAGMA table_info({_quote_ident('swe2d_baked_pipe_cell_ts')})"
        ).fetchall()
        have = {str(r[1]) for r in col_rows}
        inv_expr = '"cell_invert"' if "cell_invert" in have else "0.0"
        wid_expr = '"cell_width"' if "cell_width" in have else "1.0"
        hgt_expr = '"cell_height"' if "cell_height" in have else "0.0"
        shp_expr = '"cell_shape_type"' if "cell_shape_type" in have else "0"

        # The geometry columns are duplicated on every metric-row of the
        # same (link_id, sub_idx); GROUP BY collapses to one row per
        # (link_id, sub_idx) with arbitrary metric value chosen.
        cur.execute(
            f"SELECT link_id, cell_sub_idx, {inv_expr}, {wid_expr}, {hgt_expr}, {shp_expr} "
            f"FROM {_quote_ident('swe2d_baked_pipe_cell_ts')} "
            f"WHERE run_id = ? AND link_id IN ({placeholders}) "
            f"GROUP BY link_id, cell_sub_idx",
            (run_id, *link_ids),
        )
        out: Dict[Tuple[str, int], CellGeometry] = {}
        for row in cur.fetchall():
            try:
                key = (str(row[0]), int(row[1]))
                geom = CellGeometry(
                    invert=float(row[2] or 0.0),
                    width=float(row[3] or 0.0),
                    height=float(row[4] or 0.0),
                    shape_type=int(row[5] or 0),
                )
                out[key] = geom
            except (TypeError, ValueError):
                continue
        return out
    finally:
        conn.close()


def _load_link_metadata(gpkg_path: str, link_id: str) -> dict:
    """Read length + invert + shape + dims for a single link."""
    if not gpkg_path or not os.path.exists(gpkg_path):
        return {}
    try:
        conn = _open_conn(gpkg_path)
    except sqlite3.Error:
        return {}
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT length, inlet_invert_elev, outlet_invert_elev, link_shape, diameter, rise, from_node, to_node "
            f"FROM {_quote_ident('swe2d_drainage_links')} WHERE link_id = ?",
            (link_id,),
        )
        row = cur.fetchone()
        if row is None:
            return {}
        return {
            "length": float(row[0]) if row[0] is not None else 0.0,
            "inlet_invert": float(row[1]) if row[1] is not None else 0.0,
            "outlet_invert": float(row[2]) if row[2] is not None else 0.0,
            "link_shape": str(row[3] or "circular"),
            "diameter": float(row[4]) if row[4] is not None else 0.0,
            "rise": float(row[5]) if row[5] is not None else 0.0,
            "from_node": str(row[6]) if row[6] is not None else "",
            "to_node": str(row[7]) if row[7] is not None else "",
        }
    finally:
        conn.close()


def _load_node_metadata(gpkg_path: str, node_id: str) -> dict:
    """Read invert_elev + rim_elev + max_depth for one node."""
    if not gpkg_path or not os.path.exists(gpkg_path) or not node_id:
        return {}
    try:
        conn = _open_conn(gpkg_path)
    except sqlite3.Error:
        return {}
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT invert_elev, rim_elev, max_depth FROM {_quote_ident('swe2d_drainage_nodes')} WHERE node_id = ?",
            (node_id,),
        )
        row = cur.fetchone()
        if row is None:
            return {"invert_elev": 0.0, "rim_elev": 0.0, "max_depth": 0.0}
        return {
            "invert_elev": float(row[0]) if row[0] is not None else 0.0,
            "rim_elev": float(row[1]) if row[1] is not None else 0.0,
            "max_depth": float(row[2]) if row[2] is not None else 0.0,
        }
    finally:
        conn.close()


def _crown_for_cell(cell_height: float, cell_width: float, shape: str) -> float:
    """Return crown = invert + height_offset.

    For circular conduits, crown = invert + width (diameter).
    For rectangular, crown = invert + height.
    """
    s = (shape or "circular").lower()
    if "rect" in s:
        return cell_height if cell_height > 0 else cell_width
    return cell_width if cell_width > 0 else cell_height


def assemble_chain_profile(
    gpkg_path: str,
    run_id: str,
    chain: ChainSpec,
    graph: "DrainageGraph",
    timestep_index: int,
    *,
    crown_offset_m: Optional[float] = None,
) -> ProfileArrays:
    """Compute full profile data for the chain at one timestep.

    All data comes from coupling results only (swe2d_baked_pipe_cell_ts
    for per-cell hydraulics + geometry, swe2d_baked_coupling for link
    lengths).  No model topology tables.

    Args:
        gpkg_path: Path to the results GPKG.
        run_id: Run id (matches swe2d_baked_pipe_cell_ts.run_id).
        chain: Ordered list of (link_id, reverse) tuples.
        graph: DrainageGraph (used for metadata like from_node/to_node).
        timestep_index: Index into the timeseries arrays.
        crown_offset_m: Optional override for crown elevation above invert.

    Returns:
        ProfileArrays with all 1D arrays length == sum of sub-cell counts.
    """
    if chain.is_empty():
        empty = np.zeros(0)
        return ProfileArrays(
            station_m=empty, invert_m=empty, crown_offset_m=empty, crown_m=empty,
            hgl_m=empty, depth_m=empty,
            velocity_ms=empty, flow_cms=empty,
            node_stations=[], node_ids=[], link_boundaries=[], crown_style="circular",
        )

    link_ids = chain.cumulative_links()
    records = load_pipe_cell_records(gpkg_path, run_id, link_ids)
    geometry = load_pipe_cell_geometry(gpkg_path, run_id, link_ids)

    # Pre-load link lengths from coupling results (swe2d_baked_coupling,
    # metric='length').  This is the ONLY source of link length — no model
    # topology tables are consulted.
    _coupling_lengths: dict[str, float] = {}
    try:
        _cl_conn = __import__('sqlite3').connect(f"file:{gpkg_path}?mode=ro", uri=True)
        _rows = _cl_conn.execute(
            "SELECT object_id, values_blob FROM swe2d_baked_coupling "
            "WHERE run_id=? AND component='drainage_link' AND metric='length'",
            (run_id,),
        ).fetchall()
        _cl_conn.close()
        for _oid, _blob in _rows:
            if _blob:
                _arr = np.frombuffer(_blob, dtype=np.float64)
                if _arr.size:
                    _coupling_lengths[str(_oid)] = float(_arr[0])
    except Exception:
        pass

    station_parts = []
    invert_parts = []
    depth_parts = []
    velocity_parts = []
    flow_parts = []
    head_parts = []
    width_parts = []
    height_parts = []
    shape_parts = []  # 0 = circular, 1+ = rectangular / other

    node_stations: List[float] = []
    node_ids: List[str] = []
    link_boundaries: List[Tuple[int, str]] = []

    cumulative_offset = 0.0
    cell_count = 0
    seen_any_rectangular = False
    seen_any_circular = False

    for link_id, reverse in chain.link_specs:
        length = _coupling_lengths.get(link_id, 0.0)
        fn = graph.from_node.get(link_id, "")
        tn = graph.to_node.get(link_id, "")

        sub_keys = sorted(
            [k for k in records.keys() if k[0] == link_id and k[2] == "depth"],
            key=lambda k: k[1],
        )
        if not sub_keys:
            continue
        n_sub = len(sub_keys)
        sub_step = length / max(1, n_sub)
        segment_start = len(station_parts)

        for (lid, sub_idx, _metric) in sub_keys:
            depth = records.get((lid, sub_idx, "depth"), np.zeros(0))
            velocity = records.get((lid, sub_idx, "velocity"), np.zeros(0))
            flow = records.get((lid, sub_idx, "flow"), np.zeros(0))
            head = records.get((lid, sub_idx, "head"), np.zeros(0))

            if depth.size == 0:
                continue

            t = min(int(timestep_index), depth.size - 1) if depth.size > 0 else 0
            t = max(0, t)

            # ── Per-cell geometry from the results GPKG ──────────────
            # cell_sub_idx 0 is at the from_node end, increasing downstream.
            cell_geom = geometry.get((lid, sub_idx))
            if cell_geom is not None:
                cell_invert = cell_geom.invert
                cell_width = cell_geom.width
                cell_height = cell_geom.height
                cell_shape = cell_geom.shape_type
            else:
                cell_invert = 0.0
                cell_width = 1.0
                cell_height = 1.0
                cell_shape = 0

            if cell_shape == 0:
                seen_any_circular = True
            else:
                seen_any_rectangular = True

            station_parts.append(cumulative_offset + (sub_idx + 0.5) * sub_step)
            invert_parts.append(cell_invert)
            depth_parts.append(float(depth[t]))
            velocity_parts.append(float(velocity[t]) if velocity.size > 0 else float("nan"))
            flow_parts.append(float(flow[t]) if flow.size > 0 else float("nan"))
            head_parts.append(float(head[t]) if head.size > 0 else float("nan"))
            width_parts.append(cell_width)
            height_parts.append(cell_height)
            shape_parts.append(cell_shape)

        segment_end = len(station_parts)
        if reverse:
            for parts in (
                invert_parts,
                depth_parts,
                velocity_parts,
                flow_parts,
                head_parts,
                width_parts,
                height_parts,
                shape_parts,
            ):
                parts[segment_start:segment_end] = reversed(parts[segment_start:segment_end])

        # Determine upstream + downstream nodes for this link
        upstream = fn
        downstream = tn
        if reverse:
            upstream, downstream = downstream, upstream

        # Record node endpoint at link start (chain start only)
        if not node_ids:
            node_stations.append(0.0)
            node_ids.append(upstream)
        # End of this link = downstream node
        cumulative_offset += length
        node_stations.append(cumulative_offset)
        node_ids.append(downstream)

        cells_added = segment_end - segment_start
        link_boundaries.append((cell_count, link_id))
        cell_count += cells_added

    if not station_parts:
        empty = np.zeros(0)
        return ProfileArrays(
            station_m=empty, invert_m=empty, crown_offset_m=empty, crown_m=empty,
            hgl_m=empty, depth_m=empty,
            velocity_ms=empty, flow_cms=empty,
            node_stations=[], node_ids=[], link_boundaries=[], crown_style="circular",
        )

    # Convert parts to arrays
    station_m = np.asarray(station_parts, dtype=np.float64)
    invert_m = np.asarray(invert_parts, dtype=np.float64)
    depth_m = np.asarray(depth_parts, dtype=np.float64)
    velocity_ms = np.asarray(velocity_parts, dtype=np.float64)
    flow_cms = np.asarray(flow_parts, dtype=np.float64)
    width_arr = np.asarray(width_parts, dtype=np.float64)
    height_arr = np.asarray(height_parts, dtype=np.float64)
    shape_arr = np.asarray(shape_parts, dtype=np.int64)

    # HGL = invert + depth (per cell). This is the authoritative water surface
    # elevation used for the water polygon and the HGL line — exactly the
    # approach the existing PG viewer takes (water_y[sub_idx] = invert_y[sub_idx] + depth).
    hgl_m = invert_m + depth_m

    # Crown offset per cell:
    #   shape_type == 0 (circular): invert + cell_width (diameter)
    #   shape_type != 0 (rect / elliptical): invert + cell_height
    # The optional `crown_offset_m` arg forces a uniform offset (e.g. for
    # debugging / explicit overrides).
    if crown_offset_m is not None:
        crown_offset = np.full(invert_m.size, float(crown_offset_m), dtype=np.float64)
    else:
        is_rect = shape_arr != 0
        crown_offset = np.where(is_rect, height_arr, width_arr)
    crown_m = invert_m + crown_offset

    if seen_any_rectangular and seen_any_circular:
        crown_style = "mixed"
    elif seen_any_rectangular:
        crown_style = "rectangular"
    else:
        crown_style = "circular"

    return ProfileArrays(
        station_m=station_m,
        invert_m=invert_m,
        crown_offset_m=crown_offset,
        crown_m=crown_m,
        hgl_m=hgl_m,
        depth_m=depth_m,
        velocity_ms=velocity_ms,
        flow_cms=flow_cms,
        node_stations=node_stations,
        node_ids=node_ids,
        link_boundaries=link_boundaries,
        crown_style=crown_style,
    )


def profile_at_variable(
    profile: ProfileArrays,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (values_per_station, station_m). metric ∈ {'depth','velocity','flow','head'}."""
    if metric == "depth":
        return profile.depth_m, profile.station_m
    if metric == "velocity":
        return profile.velocity_ms, profile.station_m
    if metric == "flow":
        return profile.flow_cms, profile.station_m
    if metric == "head":
        return profile.hgl_m, profile.station_m
    raise ValueError(f"Unknown metric: {metric!r}. Supported: {VALID_METRICS}")
