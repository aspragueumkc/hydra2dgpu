#!/usr/bin/env python3
"""Create a SWE2D validation GeoPackage for sample-line flow integration.

Case definition (imperial units):
- Channel length: 200 ft
- Trapezoid cross section: 10 ft bottom width, 3H:1V side slopes, 6 ft depth
- Target validation flow depth: 4 ft
- Bed slope (domain + downstream normal-depth BC): 0.005
- Manning n: 0.02
- Mesh target: 5 ft quads (via cartesian/quadrilateral topology controls)
- Sample line: x = 100 ft (mid-reach)

The script writes a GeoPackage with topology/BC/sample layers used by the GUI.
It also writes a JSON sidecar with the computed Manning inflow discharge.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from osgeo import ogr


def manning_trapezoid_q_cfs(
    depth_ft: float,
    bottom_width_ft: float,
    side_slope_h_to_v: float,
    slope: float,
    manning_n: float,
) -> float:
    """Return discharge in cfs using Manning's equation for a trapezoid.

    Uses US customary Manning coefficient 1.49.
    """
    y = float(depth_ft)
    b = float(bottom_width_ft)
    z = float(side_slope_h_to_v)
    s = max(float(slope), 1.0e-12)
    n = max(float(manning_n), 1.0e-12)

    area = y * (b + z * y)
    wetted_perimeter = b + 2.0 * y * math.sqrt(1.0 + z * z)
    hydraulic_radius = area / max(wetted_perimeter, 1.0e-12)
    return (1.49 / n) * area * (hydraulic_radius ** (2.0 / 3.0)) * math.sqrt(s)


def _add_fields(layer: ogr.Layer, fields: Sequence[Tuple[str, int, int, int]]) -> None:
    for name, ftype, width, precision in fields:
        fd = ogr.FieldDefn(name, ftype)
        if width > 0:
            fd.SetWidth(width)
        if precision > 0:
            fd.SetPrecision(precision)
        layer.CreateField(fd)


def _create_layer(ds: ogr.DataSource, name: str, geom_type: int, fields: Sequence[Tuple[str, int, int, int]]) -> ogr.Layer:
    layer = ds.CreateLayer(name, srs=None, geom_type=geom_type)
    _add_fields(layer, fields)
    return layer


def _polygon_wkt(ring_xy: Iterable[Tuple[float, float]]) -> str:
    pts = list(ring_xy)
    if not pts:
        raise ValueError("polygon ring is empty")
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    coords = ",".join(f"{x:.8f} {y:.8f}" for x, y in pts)
    return f"POLYGON (({coords}))"


def _linestring_wkt(points_xy: Iterable[Tuple[float, float]]) -> str:
    pts = list(points_xy)
    if len(pts) < 2:
        raise ValueError("line needs at least 2 points")
    coords = ",".join(f"{x:.8f} {y:.8f}" for x, y in pts)
    return f"LINESTRING ({coords})"


def _insert_feature(layer: ogr.Layer, geom_wkt: str, attrs: dict) -> None:
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(ogr.CreateGeometryFromWkt(geom_wkt))
    for key, val in attrs.items():
        feat.SetField(str(key), val)
    if layer.CreateFeature(feat) != 0:
        raise RuntimeError(f"failed creating feature in layer '{layer.GetName()}'")


def create_case_gpkg(out_gpkg: Path) -> dict:
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if out_gpkg.exists():
        out_gpkg.unlink()

    drv = ogr.GetDriverByName("GPKG")
    if drv is None:
        raise RuntimeError("GDAL GPKG driver unavailable")
    ds = drv.CreateDataSource(str(out_gpkg))
    if ds is None:
        raise RuntimeError(f"failed to create geopackage: {out_gpkg}")

    # Geometry/control constants (feet)
    x0 = 0.0
    x1 = 200.0
    y0 = 0.0
    y1 = 50.0

    sample_x = 100.0

    bottom_width_ft = 10.0
    side_slope_h_to_v = 3.0
    bankfull_depth_ft = 6.0
    target_depth_ft = 4.0
    slope = 0.005
    n_mann = 0.02

    q_in_cfs = manning_trapezoid_q_cfs(
        depth_ft=target_depth_ft,
        bottom_width_ft=bottom_width_ft,
        side_slope_h_to_v=side_slope_h_to_v,
        slope=slope,
        manning_n=n_mann,
    )

    # Core topology layers used by GUI combos.
    topo_nodes = _create_layer(
        ds,
        "swe2d_topo_nodes",
        ogr.wkbPoint,
        [("node_id", ogr.OFTInteger, 0, 0)],
    )
    topo_arcs = _create_layer(
        ds,
        "swe2d_topo_arcs",
        ogr.wkbLineString,
        [
            ("arc_id", ogr.OFTInteger, 0, 0),
            ("node0", ogr.OFTInteger, 0, 0),
            ("node1", ogr.OFTInteger, 0, 0),
            ("region_id", ogr.OFTInteger, 0, 0),
            ("arc_role", ogr.OFTString, 24, 0),
            ("use_global_arc_ctrl", ogr.OFTInteger, 0, 0),
            ("arc_mode_override", ogr.OFTString, 24, 0),
            ("arc_soft_size_override", ogr.OFTReal, 0, 6),
            ("arc_soft_dist_override", ogr.OFTReal, 0, 6),
        ],
    )
    regions = _create_layer(
        ds,
        "swe2d_topo_regions",
        ogr.wkbPolygon,
        [
            ("region_id", ogr.OFTInteger, 0, 0),
            ("target_size", ogr.OFTReal, 0, 6),
            ("cell_type", ogr.OFTString, 32, 0),
            ("channel_generator_type", ogr.OFTString, 32, 0),
            ("edge_len_1", ogr.OFTReal, 0, 6),
            ("edge_len_2", ogr.OFTReal, 0, 6),
            ("edge_len_3", ogr.OFTReal, 0, 6),
            ("edge_len_4", ogr.OFTReal, 0, 6),
        ],
    )
    _create_layer(
        ds,
        "swe2d_topo_constraints",
        ogr.wkbPolygon,
        [
            ("constraint_id", ogr.OFTInteger, 0, 0),
            ("target_size", ogr.OFTReal, 0, 6),
            ("cell_type", ogr.OFTString, 32, 0),
            ("edge_len_1", ogr.OFTReal, 0, 6),
            ("edge_len_2", ogr.OFTReal, 0, 6),
            ("edge_len_3", ogr.OFTReal, 0, 6),
            ("edge_len_4", ogr.OFTReal, 0, 6),
        ],
    )
    _create_layer(
        ds,
        "swe2d_topo_quad_edges",
        ogr.wkbLineString,
        [
            ("region_id", ogr.OFTInteger, 0, 0),
            ("edge_id", ogr.OFTInteger, 0, 0),
            ("target_size", ogr.OFTReal, 0, 6),
            ("n_layers", ogr.OFTInteger, 0, 0),
            ("first_height", ogr.OFTReal, 0, 6),
            ("growth_rate", ogr.OFTReal, 0, 6),
        ],
    )

    manning = _create_layer(
        ds,
        "swe2d_manning_zones",
        ogr.wkbPolygon,
        [("zone_id", ogr.OFTInteger, 0, 0), ("n_mann", ogr.OFTReal, 0, 6), ("priority", ogr.OFTInteger, 0, 0)],
    )
    bc_lines = _create_layer(
        ds,
        "swe2d_bc_lines",
        ogr.wkbLineString,
        [
            ("bc_type", ogr.OFTInteger, 0, 0),
            ("bc_value", ogr.OFTReal, 0, 8),
            ("priority", ogr.OFTInteger, 0, 0),
            ("hydrograph", ogr.OFTString, 1024, 0),
            ("hydrograph_id", ogr.OFTString, 64, 0),
            ("hydrograph_layer", ogr.OFTString, 128, 0),
        ],
    )
    sample_lines = _create_layer(
        ds,
        "swe2d_sample_lines",
        ogr.wkbLineString,
        [("line_id", ogr.OFTInteger, 0, 0), ("name", ogr.OFTString, 128, 0), ("enabled", ogr.OFTInteger, 0, 0), ("priority", ogr.OFTInteger, 0, 0)],
    )

    # Region/domain polygon.
    region_ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    # Populate topology nodes/arcs for the rectangular boundary.
    node_xy = [
        (1, x0, y0),
        (2, x1, y0),
        (3, x1, y1),
        (4, x0, y1),
    ]
    for nid, x, y in node_xy:
        _insert_feature(
            topo_nodes,
            f"POINT ({x:.8f} {y:.8f})",
            {"node_id": int(nid)},
        )

    arc_rows = [
        (1, 1, 2, [(x0, y0), (x1, y0)]),
        (2, 2, 3, [(x1, y0), (x1, y1)]),
        (3, 3, 4, [(x1, y1), (x0, y1)]),
        (4, 4, 1, [(x0, y1), (x0, y0)]),
    ]
    for arc_id, n0, n1, pts in arc_rows:
        _insert_feature(
            topo_arcs,
            _linestring_wkt(pts),
            {
                "arc_id": int(arc_id),
                "node0": int(n0),
                "node1": int(n1),
                "region_id": 1,
                "arc_role": "boundary",
                "use_global_arc_ctrl": 1,
                "arc_mode_override": "",
                "arc_soft_size_override": 0.0,
                "arc_soft_dist_override": 0.0,
            },
        )

    _insert_feature(
        regions,
        _polygon_wkt(region_ring),
        {
            "region_id": 1,
            "target_size": 5.0,
            "cell_type": "cartesian",
            "channel_generator_type": "",
            "edge_len_1": 5.0,
            "edge_len_2": 5.0,
            "edge_len_3": 5.0,
            "edge_len_4": 5.0,
        },
    )

    # Uniform Manning zone.
    _insert_feature(
        manning,
        _polygon_wkt(region_ring),
        {
            "zone_id": 1,
            "n_mann": n_mann,
            "priority": 0,
        },
    )

    # Upstream inflow Q (type=2), downstream normal-depth slope (type=7).
    _insert_feature(
        bc_lines,
        _linestring_wkt([(x0, y0), (x0, y1)]),
        {
            "bc_type": 2,
            "bc_value": float(q_in_cfs),
            "priority": 0,
            "hydrograph": "",
            "hydrograph_id": "",
            "hydrograph_layer": "",
        },
    )
    _insert_feature(
        bc_lines,
        _linestring_wkt([(x1, y0), (x1, y1)]),
        {
            "bc_type": 7,
            "bc_value": slope,
            "priority": 0,
            "hydrograph": "",
            "hydrograph_id": "",
            "hydrograph_layer": "",
        },
    )

    # Sample line 100 ft from upstream boundary.
    _insert_feature(
        sample_lines,
        _linestring_wkt([(sample_x, y0), (sample_x, y1)]),
        {
            "line_id": 1,
            "name": "Q_validation_x100ft",
            "enabled": 1,
            "priority": 0,
        },
    )

    ds = None

    summary = {
        "units": "US customary (ft, cfs)",
        "domain": {
            "length_ft": 200.0,
            "width_ft": 50.0,
            "target_mesh_size_ft": 5.0,
            "cell_type": "cartesian",
        },
        "channel_cross_section": {
            "bottom_width_ft": bottom_width_ft,
            "side_slope_h_to_v": side_slope_h_to_v,
            "bankfull_depth_ft": bankfull_depth_ft,
            "target_flow_depth_ft": target_depth_ft,
        },
        "hydraulics": {
            "slope": slope,
            "manning_n": n_mann,
            "computed_inflow_q_cfs": q_in_cfs,
            "upstream_bc": {"bc_type": 2, "description": "constant total inflow Q", "bc_value": q_in_cfs},
            "downstream_bc": {"bc_type": 7, "description": "normal-depth friction slope Sf", "bc_value": slope},
        },
        "sample_line": {
            "line_id": 1,
            "name": "Q_validation_x100ft",
            "x_ft": sample_x,
            "y0_ft": y0,
            "y1_ft": y1,
        },
        "notes": [
            "Use the generated bed grid CSV to assign terrain/node bed elevations with the requested trapezoid + longitudinal slope.",
            "This package preconfigures topology, Manning zone, BC lines, and sample line for flow-integration validation.",
        ],
    }
    return summary


def write_bed_grid_csv(out_csv: Path) -> None:
    """Write 5-ft node grid bed elevations for the requested trapezoidal channel."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    x0 = 0.0
    x1 = 200.0
    y0 = 0.0
    y1 = 50.0
    dx = 5.0
    dy = 5.0

    slope = 0.005
    center_y = 25.0
    bottom_half_width = 5.0  # 10-ft bottom width
    side_slope_h_to_v = 3.0
    bank_depth = 6.0

    def bed_z_ft(x: float, y: float) -> float:
        # Longitudinal bed: 0.5% descending from upstream (x=0) to downstream (x=200).
        z_long = slope * (x1 - float(x))
        # Trapezoid cross section with 10-ft bottom and 3H:1V sides, capped at 6-ft banks.
        yy = abs(float(y) - center_y)
        if yy <= bottom_half_width:
            z_cross = 0.0
        else:
            z_cross = min(bank_depth, (yy - bottom_half_width) / side_slope_h_to_v)
        return z_long + z_cross

    xs = [x0 + i * dx for i in range(int(round((x1 - x0) / dx)) + 1)]
    ys = [y0 + j * dy for j in range(int(round((y1 - y0) / dy)) + 1)]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_ft", "y_ft", "bed_z_ft"])
        for y in ys:
            for x in xs:
                w.writerow([f"{x:.3f}", f"{y:.3f}", f"{bed_z_ft(x, y):.6f}"])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_gpkg = repo_root / "example_project" / "trapezoid_sampleline_validation.gpkg"
    out_json = repo_root / "example_project" / "trapezoid_sampleline_validation.json"
    out_bed_csv = repo_root / "example_project" / "trapezoid_sampleline_bed_grid_5ft.csv"

    summary = create_case_gpkg(out_gpkg)
    write_bed_grid_csv(out_bed_csv)
    summary["bed_grid_csv"] = str(out_bed_csv)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    q_cfs = float(summary["hydraulics"]["computed_inflow_q_cfs"])
    print(f"Wrote {out_gpkg}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_bed_csv}")
    print(f"Computed Manning inflow Q for 4 ft depth: {q_cfs:.6f} cfs")


if __name__ == "__main__":
    main()
