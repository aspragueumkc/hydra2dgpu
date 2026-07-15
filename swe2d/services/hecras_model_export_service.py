"""
HEC-RAS model export service.

Converts a HYDRA2D model (mesh + drainage + structures + BCs) into a
complete HEC-RAS 7.0 project that can be run under Wine.

Output files written:

  <project_name>.prj   — project file (registered plans/geometries/flows)
  <project_name>.g##   — geometry text file (2D flow area, BC lines, connections)
  <project_name>.g##.hdf — geometry HDF5 (mesh, pipe conduits/nodes, structures)
  <project_name>.u##   — unsteady flow file (BC hydrographs)
  <project_name>.p##   — plan file (solver settings, simulation intervals)

Pure Python + numpy + h5py — no Qt/PyQt5 imports.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "HecrasModelWriter",
    "write_hecras_project",
]


# ---------------------------------------------------------------------------
# Output data containers — these bridge HYDRA2D types → HEC-RAS fields
# ---------------------------------------------------------------------------


@dataclass
class Hecras2DFlowArea:
    """Parameters for one HEC-RAS 2D flow area."""
    name: str = "Perimeter 1"
    manning_n: float = 0.03
    cell_volume_tolerance: float = 0.01
    cell_min_area_fraction: float = 0.01
    face_profile_tolerance: float = 0.01
    face_area_tolerance: float = 0.01
    face_conveyance_ratio: float = 0.02
    laminar_depth: float = 0.01
    min_face_length_ratio: float = 0.05
    spacing_dx: float = 10.0
    spacing_dy: float = 10.0
    point_gen_params: str = ",,10,10"


@dataclass
class HecrasCulvertGroup:
    """One culvert group within a SA/2D connection."""
    shape: int = 2              # 1=Circular, 2=Box, 3=Pipe Arch
    shape_name: str = "Box"
    rise: float = 8.0
    span: float = 8.0
    length: float = 80.0
    manning_n: float = 0.02
    entrance_loss: float = 0.4
    exit_loss: float = 0.8
    chart: int = 8
    scale: int = 3
    us_invert: float = 877.0
    ds_invert: float = 876.0
    flap_gate: int = 0
    group_name: str = "Group #1"
    use_momentum: int = 1
    barrels: int = 1
    culvert_code: int = -1  # -1 means unset


@dataclass
class HecrasSA2DConnection:
    """SA/2D connection with optional culvert groups."""
    name: str = "SA2D Conn 1"
    upstream_area: str = "Perimeter 1"
    downstream_area: str = "Perimeter 1"
    centerline: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    weir_width: float = 60.0
    weir_coef: float = 2.6
    overflow_method_2d: bool = True
    routing_type: int = 1
    simple_spill_pos: float = 0.05
    simple_spill_neg: float = 0.05
    weir_submergence: float = 0.98
    weir_min_elevation: float = np.nan
    weir_station_elevation: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2))
    )
    culvert_groups: List[HecrasCulvertGroup] = field(default_factory=list)


@dataclass
class HecrasBCLine:
    """Boundary condition line on a 2D flow area perimeter."""
    name: str
    storage_area: str
    coordinates: np.ndarray  # (n, 2) float64


@dataclass
class HecrasPipeConduit:
    """One pipe conduit in the geometry HDF5."""
    name: str
    system_name: str = "Base"
    us_node: str = ""
    ds_node: str = ""
    modeling_approach: str = "Hydraulic"
    length: float = 0.0
    max_cell_length: float = 10.0
    shape: str = "Circular"
    rise: float = 2.5
    span: float = 3.0
    manning_n: float = 0.012
    us_offset: float = 0.0
    ds_offset: float = 0.0
    us_elevation: float = 0.0
    ds_elevation: float = 0.0
    slope: float = 0.0
    entrance_loss_k: float = 0.2
    exit_loss_k: float = 0.4
    us_backflow_k: float = 0.4
    ds_backflow_k: float = 0.2
    ds_gate_type: str = "None"
    major_group: str = ""
    minor_group: str = ""
    # geometry
    polyline: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))


@dataclass
class HecrasPipeNode:
    """One pipe node (manhole/junction/outfall) in the geometry HDF5."""
    name: str
    system_name: str = "Base"
    node_type: str = "Junction"
    node_status: str = ""
    us_connections: int = 0
    ds_connections: int = 0
    invert_elevation: float = 0.0
    base_area: float = 25.0
    terrain_elevation: float = 0.0
    terrain_elevation_override: float = np.nan
    depth: float = 5.0
    top_inlet_type: str = ""
    top_inlet_elevation: float = np.nan
    side_inlet_type: str = ""
    side_inlet_elevation: float = np.nan
    total_connections: int = 1
    # geometry
    x: float = 0.0
    y: float = 0.0
    # optional coupling to 2D cell
    surface_cell_id: Optional[int] = None
    surface_layer: str = "2D Flow Area"


@dataclass
class HecrasTopInlet:
    """Top inlet (grate/manhole) template."""
    name: str
    weir_length: float = 6.0
    weir_coef: float = 3.0
    orifice_area: float = 3.6
    orifice_coef: float = 0.67
    surcharge_only: bool = False


# ---------------------------------------------------------------------------
# Geometry text writer — .g## file
# ---------------------------------------------------------------------------

_COORD_FMT = "{:>16.8f}{:>16.8f}"
_SENTINEL = " 1.79769313486232E+308 , 1.79769313486232E+308 "


def _format_coord_pair(x: float, y: float) -> str:
    return "{:>16.8f}{:>16.8f}".format(x, y)


def _write_coord_block(coords: np.ndarray, lines_per_row: int = 4) -> List[str]:
    """Write coordinate pairs, N per line, fixed-width format."""
    pairs = [_format_coord_pair(x, y) for x, y in coords]
    out = []
    for i in range(0, len(pairs), lines_per_row):
        out.append("".join(pairs[i:i + lines_per_row]))
    return out


def _build_geom_header(title: str, viewing_rect: Tuple[float, float, float, float]) -> List[str]:
    xmin, xmax, ymax, ymin = viewing_rect
    return [
        f"Geom Title={title}",
        "Program Version=7.00",
        f"Viewing Rectangle= {xmin} , {xmax} , {ymax} , {ymin} ",
        "",
    ]


def _build_2d_flow_area_block(
    area: Hecras2DFlowArea,
    perimeter: np.ndarray,
    cell_centers: np.ndarray,
) -> List[str]:
    """Build the Storage Area block for one 2D flow area."""
    lines: List[str] = []
    lines.append(f"Storage Area={area.name:<16s},,")
    lines.append(f"Storage Area Surface Line= {len(perimeter)} ")
    lines.extend(_write_coord_block(perimeter, lines_per_row=2))
    lines.append(f"Storage Area Type= 1 ")
    lines.append(f"Storage Area Area=")
    lines.append(f"Storage Area Min Elev=")
    lines.append(f"Storage Area Is2D=-1")
    lines.append(f"Storage Area Point Generation Data={area.point_gen_params}")
    # Cell center seeds — HEC-RAS uses these to generate the Voronoi mesh.
    lines.append(f"Storage Area 2D Points= {len(cell_centers)} ")
    lines.extend(_write_coord_block(cell_centers, lines_per_row=4))
    now = datetime.now(timezone.utc).strftime("%d%b%Y %H:%M:%S").upper()
    lines.append(f"Storage Area 2D PointsPerimeterTime={now}")
    lines.append(f"Storage Area Mannings={area.manning_n}")
    lines.append(f"2D Cell Volume Filter Tolerance={area.cell_volume_tolerance}")
    lines.append(f"2D Cell Minimum Area Fraction={area.cell_min_area_fraction}")
    lines.append(f"2D Face Profile Filter Tolerance={area.face_profile_tolerance}")
    lines.append(f"2D Face Area Elevation Profile Filter Tolerance={area.face_area_tolerance}")
    lines.append(f"2D Face Area Elevation Conveyance Ratio={area.face_conveyance_ratio}")
    lines.append(f"2D Face Min Length Ratio={area.min_face_length_ratio}")
    lines.append(f"2D Face Area Laminar Depth={area.laminar_depth}")
    lines.append(f"2D Multiple Face Mann n=0")
    lines.append(f"2D Composite LC=0")
    return lines


def _build_sa2d_connection_block(conn: HecrasSA2DConnection) -> List[str]:
    """Build a SA/2D Connection block with optional culvert groups."""
    lines: List[str] = []
    now = datetime.now(timezone.utc).strftime("%b/%d/%Y %H:%M:%S")
    cx, cy = float(np.mean(conn.centerline[:, 0])), float(np.mean(conn.centerline[:, 1]))
    lines.append(f"Connection={conn.name:<16s},{cx},{cy}")
    lines.append("Connection Desc=")
    lines.append(f"Connection Line= {len(conn.centerline)} ")
    lines.extend(_write_coord_block(conn.centerline, lines_per_row=2))
    lines.append("Connection Centerline Profile=0")
    lines.append(f"Connection Last Edited Time={now}")
    lines.append("Conn Near Repeats=0")
    lines.append("Conn Protection Radius=-1")
    lines.append(f"Connection Up SA={conn.upstream_area:<16s}")
    lines.append(f"Connection Dn SA={conn.downstream_area:<16s}")
    lines.append(f"Conn Routing Type= {conn.routing_type} ")
    lines.append("Conn Use RC Family=False")
    lines.append(f"Conn OverFlow Method 2D={conn.overflow_method_2d}")
    lines.append(f"Conn Weir WD={conn.weir_width}")
    lines.append(f"Conn Weir Coef={conn.weir_coef}")
    lines.append("Conn Weir WSCriteria= 0 ")
    lines.append("Conn Weir Is Ogee= 0 ")
    lines.append(f"Conn Simple Spill Pos Coef={conn.simple_spill_pos}")
    lines.append(f"Conn Simple Spill Neg Coef={conn.simple_spill_neg}")

    if len(conn.weir_station_elevation) > 0:
        lines.append(f"Conn Weir SE= {len(conn.weir_station_elevation)} ")
        # station-elevation pairs, 4 per line
        se_pairs = []
        for st, el in conn.weir_station_elevation:
            se_pairs.append(f"{st:>7.1f}{el:>10.2f}")
        for i in range(0, len(se_pairs), 4):
            lines.append("".join(se_pairs[i:i + 4]))

    for cg in conn.culvert_groups:
        chart = cg.chart if cg.chart >= 0 else 8
        scale = cg.scale if cg.scale >= 0 else 3
        culv_code = cg.culvert_code if cg.culvert_code >= 0 else -1
        lines.append(
            f"Connection Culv={cg.shape},{cg.span},{cg.rise},{cg.length},"
            f"{cg.manning_n},{cg.entrance_loss},{cg.exit_loss},"
            f"{chart},{scale},{cg.us_invert},{cg.ds_invert},"
            f" {cg.flap_gate} ,{cg.group_name:<20s}, 0 ,,{culv_code}"
        )
        lines.append("                ")
        # Barrel
        barrel_coords = np.array([
            [conn.centerline[0, 0], conn.centerline[0, 1]],
            [conn.centerline[-1, 0], conn.centerline[-1, 1]],
        ])
        lines.append(f"Conn Culvert Barrel=1,Barrel #1,{len(barrel_coords)}")
        lines.append(f"{_format_coord_pair(barrel_coords[0, 0], barrel_coords[0, 1])}"
                     f"{_format_coord_pair(barrel_coords[1, 0], barrel_coords[1, 1])}")
        lines.append(f"Conn Culv Bottom n={cg.manning_n}")
        lines.append("")

    lines.append("Conn Outlet Rating Curve= 0 ,False,,")
    lines.append("Conn BR: Bridge=0,0,0,0, 0 ,0.3,0.5")
    lines.append("Conn BR: Pressure-Weir=,,,,")
    lines.append("Conn BR: Deck Dist Width WeirC Skew NumUp NumDn MinLoCord MaxHiCord "
                 "MaxSubmerge Is_Ogee")
    lines.append(f",{conn.weir_width},{conn.weir_coef},0, 0, 0, , , 0.98, 0, 0,0,,")
    lines.append("Conn BR: BR SE=1,0")
    lines.append("Conn BR: BR Bank Stations=1,,")
    lines.append("Conn BR: BR Mann=1,0")
    lines.append("Conn BR: BR SE=2,0")
    lines.append("Conn BR: BR Bank Stations=2,,")
    lines.append("Conn BR: BR Mann=2,0")
    lines.append("Conn BR: BR Coef=-1 , 0 , 0 ,,,0.8,-1,,0,,0")
    lines.append("Conn BR: BR Skew=0")
    lines.append("Conn BR: XS SE=1,0")
    lines.append("Conn BR: XS Bank Stations=1,,")
    lines.append("Conn BR: XS Mann=1,0")
    lines.append("Conn BR: XS SE=2,0")
    lines.append("Conn BR: XS Bank Stations=2,,")
    lines.append("Conn BR: XS Mann=2,0")
    return lines


def _build_bc_line_block(bc: HecrasBCLine) -> List[str]:
    """Build a BC Line block."""
    lines: List[str] = []
    name_padded = bc.name.ljust(40)
    area_padded = bc.storage_area.ljust(16)
    coords = np.asarray(bc.coordinates, dtype=np.float64)
    x_start, y_start = coords[0]
    x_end, y_end = coords[-1]
    x_mid = float(np.mean(coords[:, 0]))
    y_mid = float(np.mean(coords[:, 1]))
    lines.append(f"BC Line Name={name_padded}")
    lines.append(f"BC Line Storage Area={area_padded}")
    lines.append(f"BC Line Start Position= {x_start} , {y_start} ")
    lines.append(f"BC Line Middle Position= {x_mid} , {y_mid} ")
    lines.append(f"BC Line End Position= {x_end} , {y_end} ")
    lines.append(f"BC Line Arc= {len(coords)} ")
    lines.extend(_write_coord_block(coords, lines_per_row=2))
    lines.append(f"BC Line Text Position={_SENTINEL}")
    return lines


def write_geom_text(
    path: Path,
    title: str,
    viewing_rect: Tuple[float, float, float, float],
    flow_area: Hecras2DFlowArea,
    perimeter: np.ndarray,
    cell_centers: np.ndarray,
    bc_lines: List[HecrasBCLine],
    connections: List[HecrasSA2DConnection],
) -> None:
    """Write a complete .g## geometry text file.

    Parameters
    ----------
    path : Path
        Output .g## file path (e.g. ``project.g01``).
    title : str
        Geometry title.
    viewing_rect : (xmin, xmax, ymax, ymin)
        Viewing rectangle for RAS Mapper.
    flow_area : Hecras2DFlowArea
        2D flow area parameters.
    perimeter : (n, 2) ndarray
        Perimeter polygon coordinates (closed ring).
    cell_centers : (n, 2) ndarray
        Cell center point coordinates.
    bc_lines : list of HecrasBCLine
        Boundary condition line definitions.
    connections : list of HecrasSA2DConnection
        SA/2D connection definitions.
    """
    lines: List[str] = []
    lines.extend(_build_geom_header(title, viewing_rect))
    lines.extend(_build_2d_flow_area_block(flow_area, perimeter, cell_centers))
    lines.append("")
    for conn in connections:
        lines.extend(_build_sa2d_connection_block(conn))
        lines.append("")
    for bc in bc_lines:
        lines.extend(_build_bc_line_block(bc))
        lines.append("")
    lines.append("LCMann Time=Dec/30/1899 00:00:00")
    lines.append("LCMann Region Time=Dec/30/1899 00:00:00")
    lines.append("LCMann Table=0")
    lines.append("Chan Stop Cuts=-1 ")
    lines.append("")
    lines.append("")
    lines.append("Use User Specified Reach Order=0")
    lines.append("GIS Ratio Cuts To Invert=-1")
    lines.append("GIS Limit At Bridges=0")
    lines.append("Composite Channel Slope=5")
    lines.append("")

    path.write_text("\r\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Geometry HDF5 writer — .g##.hdf file (full compiled geometry)
# ---------------------------------------------------------------------------


def _require_h5py():
    try:
        import h5py as _h5py_local  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "h5py is not installed.  Run: pip install h5py"
        ) from exc
    return __import__("h5py")


def _dt_attr() -> np.dtype:
    return np.dtype([
        ("Name", "S16"),
        ("Locked", "u1"),
        ("Mann", "<f4"),
        ("Multiple Face Mann n", "u1"),
        ("Composite LC", "u1"),
        ("Cell Vol Tol", "<f4"),
        ("Cell Min Area Fraction", "<f4"),
        ("Face Profile Tol", "<f4"),
        ("Face Area Tol", "<f4"),
        ("Face Conv Ratio", "<f4"),
        ("Laminar Depth", "<f4"),
        ("Min Face Length Ratio", "<f4"),
        ("Spacing dx", "<f4"),
        ("Spacing dy", "<f4"),
        ("Shift dx", "<f4"),
        ("Shift dy", "<f4"),
        ("Cell Count", "<i4"),
    ])


def _dt_pipe_conduit() -> np.dtype:
    return np.dtype([
        ("Name", "S9"),
        ("System Name", "S4"),
        ("US Node", "S6"),
        ("DS Node", "S6"),
        ("Modeling Approach", "S9"),
        ("Conduit Length", "<f8"),
        ("Max Cell Length", "<f4"),
        ("Shape", "S8"),
        ("Rise", "<f4"),
        ("Span", "<f4"),
        ("Manning's n", "<f4"),
        ("US Offset", "<f4"),
        ("DS Offset", "<f4"),
        ("US Elevation", "<f4"),
        ("DS Elevation", "<f4"),
        ("Slope", "<f4"),
        ("US Entrance Loss Coefficient", "<f4"),
        ("DS Exit Loss Coefficient", "<f4"),
        ("US Backflow Loss Coefficient", "<f4"),
        ("DS Backflow Loss Coefficient", "<f4"),
        ("DS Gate Type", "S4"),
        ("Major Group", "S1"),
        ("Minor Group", "S1"),
    ])


def _dt_pipe_node() -> np.dtype:
    return np.dtype([
        ("Name", "S6"),
        ("System Name", "S4"),
        ("Node Type", "S15"),
        ("Node Status", "S28"),
        ("Condtui Connections (US:DS)", "S3"),
        ("Invert Elevation", "<f4"),
        ("Base Area", "<f4"),
        ("Terrain Elevation", "<f4"),
        ("Terrain Elevation Override", "<f4"),
        ("Depth", "<f4"),
        ("Top Inlet Type", "S34"),
        ("Top Inlet Elevation", "<f4"),
        ("Side Inlet Type", "S1"),
        ("Side Inlet Elevation", "<f4"),
        ("Total Connection Count", "<i4"),
    ])


def _dt_top_inlet() -> np.dtype:
    return np.dtype([
        ("Name", "S34"),
        ("Weir Length", "<f4"),
        ("Weir Coef", "<f4"),
        ("Orifice Area", "<f4"),
        ("Orifice Coef", "<f4"),
        ("Surcharge Only", "S5"),
    ])


def _dt_side_inlet() -> np.dtype:
    return np.dtype([
        ("Name", "S36"),
        ("Inlet Shape", "S3"),
        ("Rise", "<f4"),
        ("Span", "<f4"),
        ("Weir Coef", "<f4"),
        ("Orifice Coef", "<f4"),
        ("Surcharge Only", "S5"),
    ])


def _dt_bc_line() -> np.dtype:
    return np.dtype([
        ("Name", "S32"),
        ("SA-2D", "S16"),
        ("Type", "S8"),
    ])


def _mark_perimeter_vertices(
    perimeter: np.ndarray,
    facepoints: np.ndarray,
    tol: float = 1.0,
) -> np.ndarray:
    is_perim = np.zeros(len(facepoints), dtype=np.int32)
    for px, py in perimeter:
        dists = np.hypot(facepoints[:, 0] - px, facepoints[:, 1] - py)
        idx = int(np.argmin(dists))
        if dists[idx] < tol:
            is_perim[idx] = 1
    return is_perim


def _build_external_faces(
    faces_fp_idx: np.ndarray,
    is_perim: np.ndarray,
) -> np.ndarray:
    """Build External Faces — interior faces whose vertices touch the perimeter."""
    ext_list = []
    for fi in range(len(faces_fp_idx)):
        fp0, fp1 = faces_fp_idx[fi]
        if is_perim[fp0] or is_perim[fp1]:
            ext_list.append((0, fi, int(fp0), int(fp1), 0.0, 0.0))
    if not ext_list:
        return np.zeros((0,), dtype=[
            ("BC Line ID", "<i4"), ("Face Index", "<i4"),
            ("FP Start Index", "<i4"), ("FP End Index", "<i4"),
            ("Station Start", "<f4"), ("Station End", "<f4"),
        ])
    return np.array(ext_list, dtype=[
        ("BC Line ID", "<i4"), ("Face Index", "<i4"),
        ("FP Start Index", "<i4"), ("FP End Index", "<i4"),
        ("Station Start", "<f4"), ("Station End", "<f4"),
    ])


def write_geom_hdf5(
    path: Path,
    mesh_data: dict,
    flow_area: Hecras2DFlowArea,
    perimeter: np.ndarray,
    is_us_customary: bool = False,
    manning_n: float = 0.03,
    projection_wkt: str = 'LOCAL_CS["Unknown"]',
    terrain_path: Optional[str] = None,
    terrain_hdf_path: Optional[str] = None,
    terrain_layername: str = "Terrain",
    pipe_conduits: Optional[List[HecrasPipeConduit]] = None,
    pipe_nodes: Optional[List[HecrasPipeNode]] = None,
    top_inlets: Optional[List[HecrasTopInlet]] = None,
    bc_lines: Optional[List[HecrasBCLine]] = None,
) -> None:
    """Write a complete .g##.hdf geometry HDF5 file.

    Parameters
    ----------
    path : Path
        Output .hdf file path (e.g. ``project.g01.hdf``).
    mesh_data : dict
        Mesh geometry dict with keys ``node_x``, ``node_y``, ``node_z``,
        and either ``cell_face_offsets`` + ``cell_face_nodes`` or
        ``cell_nodes`` (triangular).
    flow_area : Hecras2DFlowArea
        2D flow area parameters.
    perimeter : (n, 2) ndarray
        Perimeter polygon (closed ring).
    is_us_customary : bool
        Unit system flag.
    manning_n : float
        Manning's n value.
    projection_wkt : str
        CRS WKT string.
    terrain_path : str or None
        Path to terrain raster (HEC-RAS native path convention).
    terrain_hdf_path : str or None
        Path to terrain HDF (for HEC-RAS terrain).
    terrain_layername : str
        Terrain layer name.
    pipe_conduits : list of HecrasPipeConduit or None
        Pipe conduits to write.
    pipe_nodes : list of HecrasPipeNode or None
        Pipe nodes to write.
    top_inlets : list of HecrasTopInlet or None
        Top inlet templates.
    bc_lines : list of HecrasBCLine or None
        Boundary condition lines (for External Faces).
    """
    h5py = _require_h5py()

    node_x = np.asarray(mesh_data["node_x"], dtype=np.float64)
    node_y = np.asarray(mesh_data["node_y"], dtype=np.float64)
    node_z = np.asarray(mesh_data.get("node_z", np.zeros_like(node_x)), dtype=np.float64)
    face_offsets = mesh_data.get("cell_face_offsets")
    face_nodes_arr = mesh_data.get("cell_face_nodes")
    cell_nodes_tri = mesh_data.get("cell_nodes")

    # Dense cell-vertex index array
    if face_offsets is not None and face_nodes_arr is not None:
        offsets = face_offsets.astype(np.int32)
        n_cells = len(offsets) - 1
        max_vp = max(offsets[i + 1] - offsets[i] for i in range(n_cells))
        fp_idx = np.full((n_cells, max_vp), -1, dtype=np.int32)
        cell_cx = np.empty(n_cells, dtype=np.float64)
        cell_cy = np.empty(n_cells, dtype=np.float64)
        cell_solver_z = np.empty(n_cells, dtype=np.float64)
        for i in range(n_cells):
            s, e = int(offsets[i]), int(offsets[i + 1])
            ring = face_nodes_arr[s:e].astype(np.int32)
            fp_idx[i, :e - s] = ring
            cell_cx[i] = float(np.mean(node_x[ring]))
            cell_cy[i] = float(np.mean(node_y[ring]))
            cell_solver_z[i] = float(np.mean(node_z[ring]))
    else:
        tri = cell_nodes_tri.reshape(-1, 3).astype(np.int32)
        n_cells = tri.shape[0]
        fp_idx = tri
        cell_cx = np.mean(node_x[tri], axis=1)
        cell_cy = np.mean(node_y[tri], axis=1)
        cell_solver_z = np.mean(node_z[tri], axis=1)

    # Face data (edges between cells) — vectorized
    facepoints = np.column_stack([node_x, node_y])
    n_facepoints = len(facepoints)

    # Extract all unique edges from fp_idx using the edge data from the mesh
    all_edge_pairs: List[np.ndarray] = []
    for ci in range(n_cells):
        row = fp_idx[ci]
        valid = row[row >= 0]
        nv = len(valid)
        if nv < 3:
            continue
        edges = np.column_stack([valid, np.roll(valid, -1)])
        ae = np.minimum(edges[:, 0], edges[:, 1])
        be = np.maximum(edges[:, 0], edges[:, 1])
        all_edge_pairs.append(np.column_stack([ae, be]))
    all_edges = np.vstack(all_edge_pairs)

    # Unique edges
    edge_dt = np.dtype("i4,i4")
    edge_struct = all_edges.view(edge_dt).ravel()
    _, uniq_idx = np.unique(edge_struct, return_index=True)
    all_faces = all_edges[np.sort(uniq_idx)].astype(np.int32)
    n_all_faces = len(all_faces)

    # Count cells per face
    fp0_all = all_faces[:, 0].astype(np.int64)
    fp1_all = all_faces[:, 1].astype(np.int64)
    edge_to_fi_all = {int(fp0_all[i]) << 32 | int(fp1_all[i]): i for i in range(n_all_faces)}

    face_cell_counts = np.zeros(n_all_faces, dtype=np.int32)
    all_faces_cell_idx = np.full((n_all_faces, 2), -1, dtype=np.int32)
    for ci in range(n_cells):
        row = fp_idx[ci]
        valid = row[row >= 0]
        nv = len(valid)
        if nv < 3:
            continue
        for vi in range(nv):
            a, b = int(valid[vi]), int(valid[(vi + 1) % nv])
            if a > b:
                a, b = b, a
            key = int(a) << 32 | int(b)
            fi = edge_to_fi_all.get(key)
            if fi is not None and face_cell_counts[fi] < 2:
                all_faces_cell_idx[fi, face_cell_counts[fi]] = ci
                face_cell_counts[fi] += 1

    # HEC-RAS requires ALL faces to have 2 valid cell indices.
    # Filter out boundary faces (only 1 cell) — they are tracked via External Faces.
    interior_mask = face_cell_counts >= 2
    faces_fp_idx_arr = all_faces[interior_mask]
    faces_cell_idx = all_faces_cell_idx[interior_mask]
    n_faces = len(faces_fp_idx_arr)

    # Face normals and lengths
    edge_to_fi = {int(faces_fp_idx_arr[i, 0]) << 32 | int(faces_fp_idx_arr[i, 1]): i
                  for i in range(n_faces)}
    fp_a = facepoints[faces_fp_idx_arr[:, 0]]
    fp_b = facepoints[faces_fp_idx_arr[:, 1]]
    dx = fp_b[:, 0] - fp_a[:, 0]
    dy = fp_b[:, 1] - fp_a[:, 1]
    face_lengths = np.hypot(dx, dy)
    face_nx = dy / np.maximum(face_lengths, 1e-12)
    face_ny = -dx / np.maximum(face_lengths, 1e-12)
    face_normal_length = np.column_stack([face_nx, face_ny, face_lengths])
    face_min_elev = np.minimum(node_z[faces_fp_idx_arr[:, 0]], node_z[faces_fp_idx_arr[:, 1]])

    # Perimeter vertices
    is_perim = _mark_perimeter_vertices(perimeter, facepoints)

    area_name = flow_area.name
    now = datetime.now(timezone.utc).strftime("%d%b%Y %H:%M:%S").upper()
    geom_time = now

    with h5py.File(path, "w") as f:
        # File-level attrs
        f.attrs["File Type"] = np.bytes_("HEC-RAS Geometry")
        f.attrs["File Version"] = np.bytes_("HEC-RAS 7.0 April 2026")
        f.attrs["Units System"] = np.bytes_(
            "US Customary" if is_us_customary else "SI"
        )
        f.attrs["Projection"] = np.bytes_(projection_wkt.encode("utf-8"))

        geo = f.require_group("Geometry")
        geo.attrs["Complete Geometry"] = np.bytes_("True")
        geo.attrs["Extents"] = np.array([
            float(np.min(node_x)), float(np.max(node_x)),
            float(np.min(node_y)), float(np.max(node_y)),
        ], dtype=np.float64)
        geo.attrs["Geometry Time"] = np.bytes_(geom_time)
        geo.attrs["SI Units"] = np.bytes_("False" if is_us_customary else "True")
        geo.attrs["Title"] = np.bytes_(flow_area.name.encode())
        geo.attrs["Version"] = np.bytes_("1.0.22 (07Apr2026)")
        if terrain_hdf_path:
            geo.attrs["Terrain File Date"] = np.bytes_(now)
            geo.attrs["Terrain Filename"] = np.bytes_(
                terrain_hdf_path.replace("/", "\\")
            )
            geo.attrs["Terrain Layername"] = np.bytes_(terrain_layername)

        # ---- 2D Flow Areas ----
        flow_areas_grp = f.require_group("Geometry/2D Flow Areas")

        # Attributes
        attr_row = np.array([
            (
                area_name.encode(),
                0,
                np.float32(manning_n),
                0, 0,
                np.float32(flow_area.cell_volume_tolerance),
                np.float32(flow_area.cell_min_area_fraction),
                np.float32(flow_area.face_profile_tolerance),
                np.float32(flow_area.face_area_tolerance),
                np.float32(flow_area.face_conveyance_ratio),
                np.float32(flow_area.laminar_depth),
                np.float32(flow_area.min_face_length_ratio),
                np.float32(flow_area.spacing_dx),
                np.float32(flow_area.spacing_dy),
                np.float32(np.nan),
                np.float32(np.nan),
                np.int32(n_cells),
            )
        ], dtype=_dt_attr())
        flow_areas_grp.create_dataset("Attributes", data=attr_row)

        # Polygon
        flow_areas_grp.create_dataset("Polygon Info",
            data=np.array([[0, len(perimeter), 0, 1]], dtype=np.int32))
        flow_areas_grp.create_dataset("Polygon Parts",
            data=np.array([[0, len(perimeter)]], dtype=np.int32))
        flow_areas_grp.create_dataset("Polygon Points",
            data=perimeter.astype(np.float64))

        # Cell Info/Points (cell centers)
        flow_areas_grp.create_dataset("Cell Info",
            data=np.array([[0, n_cells]], dtype=np.int32))
        flow_areas_grp.create_dataset("Cell Points",
            data=np.column_stack([cell_cx, cell_cy]).astype(np.float64))

        # Per-area group
        area_grp = flow_areas_grp.require_group(area_name)
        extents = np.array([
            float(np.min(node_x)), float(np.max(node_x)),
            float(np.min(node_y)), float(np.max(node_y)),
        ], dtype=np.float64)
        area_grp.attrs["Cell Average Size"] = np.float64(
            float(np.mean(face_lengths))
        )
        area_grp.attrs["Cell Maximum Index"] = np.int32(n_cells - 1)
        area_grp.attrs["Cell Maximum Size"] = np.float64(
            float(np.max(face_lengths))
        )
        area_grp.attrs["Cell Minimum Area Fraction"] = np.float32(
            flow_area.cell_min_area_fraction
        )
        area_grp.attrs["Cell Minimum Size"] = np.float64(
            float(np.min(face_lengths))
        )
        area_grp.attrs["Cell Volume Tolerance"] = np.float32(
            flow_area.cell_volume_tolerance
        )
        area_grp.attrs["Composite LC"] = np.uint8(0)
        area_grp.attrs["Data Date"] = np.bytes_(now)
        area_grp.attrs["Extents"] = extents
        area_grp.attrs["Face Area Conveyance Ratio"] = np.float32(
            flow_area.face_conveyance_ratio
        )
        area_grp.attrs["Face Area Elevation Tolerance"] = np.float32(
            flow_area.face_area_tolerance
        )
        area_grp.attrs["Face Profile Tolerance"] = np.float32(
            flow_area.face_profile_tolerance
        )
        area_grp.attrs["Laminar Depth"] = np.float32(flow_area.laminar_depth)
        area_grp.attrs["Manning's n"] = np.float32(manning_n)
        area_grp.attrs["Min Face Length Ratio"] = np.float32(
            flow_area.min_face_length_ratio
        )
        area_grp.attrs["Multiple Face Mann n"] = np.uint8(0)

        # FacePoints
        area_grp.create_dataset("FacePoints Coordinate", data=facepoints)
        area_grp.create_dataset("FacePoints Elevation",
            data=node_z.astype(np.float32))
        area_grp.create_dataset("FacePoints Is Perimeter", data=is_perim)

        # FacePoints Cell Info/Values
        fp_cell_info = np.zeros((n_facepoints, 2), dtype=np.int32)
        fp_cell_vals = np.full(n_facepoints, -1, dtype=np.int32)
        for fi in range(n_faces):
            for side in (0, 1):
                ci = faces_cell_idx[fi, side]
                if ci >= 0:
                    for fp in faces_fp_idx_arr[fi]:
                        if fp_cell_vals[fp] < 0:
                            fp_cell_vals[fp] = ci
        # Build cell info (how many facepoints per cell index)
        cell_fp_counts: Dict[int, int] = {}
        for fp_idx_val in fp_cell_vals:
            if fp_idx_val >= 0:
                cell_fp_counts[fp_idx_val] = cell_fp_counts.get(fp_idx_val, 0) + 1
        ci = 0
        for cid in sorted(cell_fp_counts.keys()):
            cnt = cell_fp_counts[cid]
            fp_cell_info[ci, 0] = cid
            fp_cell_info[ci, 1] = cnt
            ci += 1
        area_grp.create_dataset("FacePoints Cell Info",
            data=fp_cell_info[:ci])
        area_grp.create_dataset("FacePoints Cell Index Values",
            data=fp_cell_vals)

        # Faces
        area_grp.create_dataset("Faces Cell Indexes", data=faces_cell_idx)
        area_grp.create_dataset("Faces FacePoint Indexes",
            data=faces_fp_idx_arr)
        area_grp.create_dataset("Faces Minimum Elevation",
            data=face_min_elev.astype(np.float32))
        area_grp.create_dataset(
            "Faces NormalUnitVector and Length",
            data=face_normal_length.astype(np.float32),
        )

        # Faces Perimeter Info/Values
        # Identify perimeter faces (face has a perimeter vertex on both sides)
        perim_face_info = np.full((n_faces, 2), -1, dtype=np.int32)
        perim_face_vals: List[Tuple[int, int]] = []
        for fi in range(n_faces):
            fp0, fp1 = faces_fp_idx_arr[fi]
            if is_perim[fp0] or is_perim[fp1]:
                perim_face_info[fi] = [fi, len(perim_face_vals)]
                perim_face_vals.append((fi, fi))
        if perim_face_vals:
            area_grp.create_dataset("Faces Perimeter Info",
                data=perim_face_info)
            area_grp.create_dataset("Faces Perimeter Values",
                data=np.array(perim_face_vals, dtype=np.float64))

        # Perimeter
        area_grp.create_dataset("Perimeter",
            data=perimeter.astype(np.float64))

        # Cells
        area_grp.create_dataset("Cells Center Coordinate",
            data=np.column_stack([cell_cx, cell_cy]).astype(np.float64))
        area_grp.create_dataset("Cells Center Manning's n",
            data=np.full(n_cells, manning_n, dtype=np.float32))
        area_grp.create_dataset("Cells FacePoint Indexes", data=fp_idx)
        area_grp.create_dataset("Cells Minimum Elevation",
            data=cell_solver_z.astype(np.float32))
        # Surface area per cell (approximate from face data)
        cell_area = np.zeros(n_cells, dtype=np.float32)
        for fi in range(n_faces):
            for side in (0, 1):
                ci = faces_cell_idx[fi, side]
                if ci >= 0:
                    cell_area[ci] += face_lengths[fi] * 0.5
        area_grp.create_dataset("Cells Surface Area", data=cell_area)

        # ---- GeomPreprocess/Storage Areas ----
        geom_pre = f.require_group("Geometry/GeomPreprocess")
        sa_grp = geom_pre.require_group("Storage Areas")
        sa_grp.create_dataset("NSAC", data=np.array([0], dtype=np.int32))

        # AutoUpdateParameters
        auto = f.require_group("Geometry/AutoUpdateParameters")
        for key in [
            "SA Elev-Vol Info", "Structure River Stations", "XS Bank Stations",
            "XS Blocked Obstruct", "XS Elevations - Channel",
            "XS Elevations - Overbanks", "XS Ineffective Areas",
            "XS Mannings n Values", "XS Reach Lengths", "XS River Stations",
        ]:
            auto.attrs[key] = np.bytes_("False")

        # ---- Land Cover ----
        lc = f.require_group("Geometry/Land Cover (Manning's n)")
        lc.create_dataset("Calibration Table",
            data=np.array([(b"", np.float32(np.nan))],
                dtype=[("Land Cover Name", "S32"),
                       ("Base Manning's n Value", "<f4")]))

        # ---- Boundary Condition Lines ----
        if bc_lines:
            bc_grp = f.require_group("Geometry/Boundary Condition Lines")

            bc_attr_data = []
            bc_poly_points: List[np.ndarray] = []
            bc_poly_info: List[Tuple[int, int, int, int]] = []
            bc_poly_parts: List[Tuple[int, int]] = []

            point_offset = 0
            for bc in bc_lines:
                coords = np.asarray(bc.coordinates, dtype=np.float64)
                n_pts = len(coords)
                bc_attr_data.append((
                    bc.name.encode(),
                    bc.storage_area.encode(),
                    b"External",
                ))
                bc_poly_info.append((point_offset, n_pts, len(bc_poly_parts), 1))
                bc_poly_parts.append((0, n_pts))
                bc_poly_points.append(coords)
                point_offset += n_pts

            bc_grp.create_dataset("Attributes",
                data=np.array(bc_attr_data, dtype=_dt_bc_line()))

            if bc_poly_points:
                all_points = np.vstack(bc_poly_points)
                bc_grp.create_dataset("Polyline Info",
                    data=np.array(bc_poly_info, dtype=np.int32))
                bc_grp.create_dataset("Polyline Parts",
                    data=np.array(bc_poly_parts, dtype=np.int32))
                bc_grp.create_dataset("Polyline Points",
                    data=all_points)

            # External Faces — interior faces touching the perimeter
            ext_faces = _build_external_faces(faces_fp_idx_arr, is_perim)
            if len(ext_faces) > 0:
                bc_grp.create_dataset("External Faces", data=ext_faces)

        # ---- Structures ----
        struct_grp = f.require_group("Geometry/Structures")
        struct_grp.attrs["Bridge/Culvert Count"] = np.int32(0)
        struct_grp.attrs["Connection Count"] = np.int32(0)
        struct_grp.attrs["Has Bridge Opening (2D)"] = np.int32(0)
        struct_grp.attrs["Inline Structure Count"] = np.int32(0)
        struct_grp.attrs["Lateral Structure Count"] = np.int32(0)

        # ---- Pipe Conduits ----
        if pipe_conduits:
            pc_grp = f.require_group("Geometry/Pipe Conduits")
            pc_grp.attrs["Version"] = np.bytes_("1.0.1")

            n_pc = len(pipe_conduits)
            pc_attr = np.zeros(n_pc, dtype=_dt_pipe_conduit())
            pc_poly_info = np.zeros((n_pc, 4), dtype=np.int32)
            pc_poly_parts = np.zeros((n_pc, 2), dtype=np.int32)
            all_pc_points: List[np.ndarray] = []

            point_offset = 0
            for i, conduit in enumerate(pipe_conduits):
                us_ds = f"{conduit.us_node},{conduit.ds_node}"
                pc_attr[i] = (
                    conduit.name.encode(),
                    conduit.system_name.encode(),
                    conduit.us_node.encode(),
                    conduit.ds_node.encode(),
                    conduit.modeling_approach.encode(),
                    conduit.length,
                    conduit.max_cell_length,
                    conduit.shape.encode(),
                    conduit.rise,
                    conduit.span,
                    conduit.manning_n,
                    conduit.us_offset,
                    conduit.ds_offset,
                    conduit.us_elevation,
                    conduit.ds_elevation,
                    conduit.slope,
                    conduit.entrance_loss_k,
                    conduit.exit_loss_k,
                    conduit.us_backflow_k,
                    conduit.ds_backflow_k,
                    conduit.ds_gate_type.encode(),
                    conduit.major_group.encode(),
                    conduit.minor_group.encode(),
                )
                poly = np.asarray(conduit.polyline, dtype=np.float64)
                n_pts = poly.shape[0]
                pc_poly_info[i] = (point_offset, n_pts, i, 1)
                pc_poly_parts[i] = (0, n_pts)
                all_pc_points.append(poly)
                point_offset += n_pts

            pc_grp.create_dataset("Attributes", data=pc_attr)
            pc_grp.create_dataset("Polyline Info", data=pc_poly_info)
            pc_grp.create_dataset("Polyline Parts", data=pc_poly_parts)
            if all_pc_points:
                pc_grp.create_dataset("Polyline Points",
                    data=np.vstack(all_pc_points))

        # ---- Pipe Nodes ----
        if pipe_nodes:
            pn_grp = f.require_group("Geometry/Pipe Nodes")
            pn_grp.attrs["Version"] = np.bytes_("1.0.1")

            n_pn = len(pipe_nodes)
            pn_attr = np.zeros(n_pn, dtype=_dt_pipe_node())
            pn_points = np.zeros((n_pn, 2), dtype=np.float64)

            for i, node in enumerate(pipe_nodes):
                conn_str = f"{node.us_connections}:{node.ds_connections}"
                pn_attr[i] = (
                    node.name.encode(),
                    node.system_name.encode(),
                    node.node_type.encode()[:15],
                    node.node_status.encode()[:28],
                    conn_str.encode(),
                    node.invert_elevation,
                    node.base_area,
                    node.terrain_elevation,
                    node.terrain_elevation_override,
                    node.depth,
                    node.top_inlet_type.encode()[:34],
                    node.top_inlet_elevation,
                    node.side_inlet_type.encode()[:1],
                    node.side_inlet_elevation,
                    node.total_connections,
                )
                pn_points[i] = (node.x, node.y)

            pn_grp.create_dataset("Attributes", data=pn_attr)
            pn_grp.create_dataset("Points", data=pn_points)

            # Top Inlets
            if top_inlets:
                ti_grp = pn_grp.require_group("Top Inlets")
                n_ti = len(top_inlets)
                ti_attr = np.zeros(n_ti, dtype=_dt_top_inlet())
                for i, inlet in enumerate(top_inlets):
                    ti_attr[i] = (
                        inlet.name.encode()[:34],
                        inlet.weir_length,
                        inlet.weir_coef,
                        inlet.orifice_area,
                        inlet.orifice_coef,
                        b"True" if inlet.surcharge_only else b"False",
                    )
                ti_grp.create_dataset("Attributes", data=ti_attr)
                ti_grp.create_dataset("Points Info",
                    data=np.zeros((n_ti, 2), dtype=np.int32))
                ti_grp.create_dataset("Points Points",
                    data=np.zeros((0, 2), dtype=np.float64))

            # Side Inlets (empty stub)
            si_grp = pn_grp.require_group("Side Inlets")
            si_grp.create_dataset("Attributes",
                data=np.zeros((0,), dtype=_dt_side_inlet()))
            si_grp.create_dataset("Points Info",
                data=np.zeros((0, 2), dtype=np.int32))
            si_grp.create_dataset("Points Points",
                data=np.zeros((0, 0), dtype=np.float64))


# ---------------------------------------------------------------------------
# Unsteady flow text writer — .u## file
# ---------------------------------------------------------------------------


@dataclass
class HecrasBoundary:
    """One boundary condition entry in the unsteady flow file."""
    bc_line_name: str = ""
    area_2d: str = ""
    flow_type: str = "Flow Hydrograph"  # Flow Hydrograph, Stage Hydrograph, Normal Depth
    hydrograph_values: Optional[List[float]] = None
    hydrograph_times: Optional[List[float]] = None  # in same units as simulation
    friction_slope: float = 0.001
    interval: str = "1HOUR"


def write_unsteady_flow(
    path: Path,
    title: str,
    boundaries: List[HecrasBoundary],
    start_date: str = "01JAN2000",
    start_time: str = "0000",
    computation_interval: str = "1MIN",
) -> None:
    """Write a .u## unsteady flow file.

    Parameters
    ----------
    path : Path
        Output file path.
    title : str
        Flow title.
    boundaries : list of HecrasBoundary
        Boundary condition definitions.
    start_date : str
        Simulation start date (ddMONyyyy).
    start_time : str
        Simulation start time (HHMM).
    computation_interval : str
        Computation interval (e.g. '1MIN', '5SEC').
    """
    lines: List[str] = []
    lines.append(f"Flow Title={title}")
    lines.append("Program Version=7.00")
    lines.append("Use Restart= 0 ")

    for bc in boundaries:
        area_field = bc.area_2d.ljust(16) if bc.area_2d else " " * 16
        bc_name_field = bc.bc_line_name.ljust(32) if bc.bc_line_name else " " * 32

        # HEC-RAS 7.0 Boundary Location format: 10 comma-separated fields
        #  1=River Name(16) 2=Reach Name(16) 3=RS(8) 4=Node Name(8)
        #  5=Profile Name(16) 6=SA/2D Area(16) 7=River(16) 8=BC Line(32)
        #  9=Setting(32) 10=Additional(32)
        lines.append(
            f"Boundary Location={''.ljust(16)},"
            f"{''.ljust(16)},"
            f"{''.ljust(8)},"
            f"{''.ljust(8)},"
            f"{''.ljust(16)},"
            f"{area_field},"
            f"{''.ljust(16)},"
            f"{bc_name_field},"
            f"{''.ljust(32)},"
            f"{''.ljust(32)}"
        )

        if bc.flow_type == "Normal Depth":
            lines.append(f"Friction Slope={bc.friction_slope}")
        elif bc.flow_type in ("Flow Hydrograph", "Stage Hydrograph"):
            lines.append(f"Interval={bc.interval}")
            vals = bc.hydrograph_values or []
            n_vals = len(vals)
            lines.append(f"{bc.flow_type}= {n_vals} ")
            fmt_vals = [f"{v:>10.3f}" for v in vals]
            for i in range(0, len(fmt_vals), 10):
                lines.append("".join(fmt_vals[i:i + 10]))
            lines.append("Stage Hydrograph TW Check=0")
            lines.append(f"{bc.flow_type} Slope= {bc.friction_slope} ")
            lines.append("DSS Path=")
            lines.append("Use DSS=False")
            lines.append("Use Fixed Start Time=False")
            lines.append("Fixed Start Date/Time=,")
            lines.append("Is Critical Boundary=False")
            lines.append("Critical Boundary Flow=")

    lines.append("Met Point Raster Parameters=,,,,")
    lines.append("Precipitation Mode=Disable")
    lines.append("Wind Mode=No Wind Forces")
    lines.append("Air Density Mode=")
    lines.append("Wave Mode=No Wave Forcing")

    path.write_text("\r\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Plan text writer — .p## file
# ---------------------------------------------------------------------------


@dataclass
class HecrasPlanParams:
    """HEC-RAS plan (solver) parameters."""
    title: str = "HYDRA2D_to_HECRAS"
    short_identifier: str = "HYD2HEC"
    simulation_date_start: str = "01JAN2000"
    simulation_time_start: str = "0000"
    simulation_date_end: str = "01JAN2000"
    simulation_time_end: str = "0030"
    computation_interval: str = "2SEC"
    output_interval: str = "1MIN"
    mapping_interval: str = "1MIN"
    geom_number: str = "01"
    flow_number: str = "01"
    # 2D solver params
    theta: float = 1.0
    theta_warmup: float = 1.0
    z_tolerance: float = 0.01
    volume_tolerance: float = 0.01
    max_iterations: int = 20
    equation_set: int = 2  # 0=DWE, 1=SWE-ELM, 2=SWE Diffusion
    eddy_viscosity: float = 0.3
    turbulence: str = "None"
    solver_type: str = "PARDISO (Direct)"
    gravity: float = 32.17405  # ft/s² (US customary)
    ramp_up_fraction: float = 0.1
    # Courant
    use_courant: bool = True
    max_courant: float = 0.7
    min_courant: float = 0.3


def write_plan(
    path: Path,
    params: HecrasPlanParams,
    area_name: str = "Perimeter 1",
) -> None:
    """Write a .p## plan file.

    Parameters
    ----------
    path : Path
        Output file path.
    params : HecrasPlanParams
        Plan parameters.
    area_name : str
        2D flow area name for per-area solver settings.
    """
    lines: List[str] = []
    lines.append(f"Plan Title={params.title}")
    lines.append("Program Version=7.00")
    lines.append(f"Short Identifier={params.short_identifier:<60s}")
    lines.append(
        f"Simulation Date={params.simulation_date_start},{params.simulation_time_start},"
        f"{params.simulation_date_end},{params.simulation_time_end}"
    )
    lines.append(f"Geom File=g{params.geom_number}")
    lines.append(f"Flow File=u{params.flow_number}")
    lines.append("Subcritical Flow")
    lines.append("K Sum by GR= 0 ")
    lines.append("Std Step Tol= 0.01 ")
    lines.append("Critical Tol= 0.01 ")
    lines.append("Num of Std Step Trials= 20 ")
    lines.append("Max Error Tol= 0.3 ")
    lines.append("Flow Tol Ratio= 0.001 ")
    lines.append("Split Flow NTrial= 30 ")
    lines.append("Split Flow Tol= 0.02 ")
    lines.append("Split Flow Ratio= 0.02 ")
    lines.append("Log Output Level= 0 ")
    lines.append("Friction Slope Method= 1 ")
    lines.append("Unsteady Friction Slope Method= 2 ")
    lines.append("Unsteady Bridges Friction Slope Method= 1 ")
    lines.append("Parabolic Critical Depth")
    lines.append("Global Vel Dist= 0 , 0 , 0 ")
    lines.append("Global Log Level= 0 ")
    lines.append("CheckData=True")
    lines.append("Encroach Param=-1 ,0,0, 0 ")
    lines.append(f"Computation Interval={params.computation_interval}")
    lines.append(f"Output Interval={params.output_interval}")
    lines.append(f"Instantaneous Interval={params.output_interval}")
    lines.append(f"Mapping Interval={params.mapping_interval}")
    if params.use_courant:
        lines.append("Computation Time Step Use Courant=       -1")
        lines.append("Computation Time Step Use Time Series=    0")
        lines.append(f"Computation Time Step Max Courant={params.max_courant}")
        lines.append(f"Computation Time Step Min Courant={params.min_courant}")
        lines.append("Computation Time Step Count To Double=3")
        lines.append("Computation Time Step Max Doubling=2")
        lines.append("Computation Time Step Max Halving=3")
        lines.append("Computation Time Step Residence Courant=0")
    else:
        lines.append("Computation Time Step Use Courant=        0")
        lines.append("Computation Time Step Use Time Series=    0")
    lines.append("Run HTab=-1 ")
    lines.append("Run UNet=-1 ")
    lines.append("Run Sediment= 0 ")
    lines.append("Run PostProcess=-1 ")
    lines.append("Run WQNet= 0 ")
    lines.append("Run RASMapper=-1 ")
    lines.append(f"UNET Gravity={params.gravity}")
    lines.append("UNET 1D Methodology=Finite Difference")
    lines.append("UNET DSS MLevel= 4 ")
    lines.append("UNET Pardiso=0")
    lines.append("UNET DZMax Abort= 100 ")
    lines.append("UNET Use Existing IB Tables=-1 ")
    lines.append("UNET Froude Reduction=False")
    lines.append("UNET Froude Limit= 0.8 ")
    lines.append("UNET Froude Power= 4 ")
    lines.append("UNET D1 Cores= 0 ")
    lines.append("UNET WindReference=Eulerian")
    lines.append("UNET WindDragFormulation=Hsu (1988)")
    lines.append("UNET D2 Coriolis=0")
    lines.append("UNET D2 Cores= 0 ")
    lines.append(f"UNET D2 Theta= {params.theta} ")
    lines.append(f"UNET D2 Theta Warmup= {params.theta_warmup} ")
    lines.append(f"UNET D2 Z Tol= {params.z_tolerance} ")
    lines.append(f"UNET D2 Volume Tol= {params.volume_tolerance} ")
    lines.append(f"UNET D2 Max Iterations= {params.max_iterations} ")
    lines.append("UNET D2 Advanced Convergence=0")
    lines.append("UNET D2 WS Max Tol=0.15")
    lines.append("UNET D2 WS RMS Tol=0.002")
    lines.append("UNET D2 WS Stall Tol=1")
    lines.append(f"UNET D2 Equation= {params.equation_set} ")
    lines.append("UNET D2 TotalICTime=")
    lines.append(f"UNET D2 RampUpFraction={params.ramp_up_fraction}")
    lines.append("UNET D2 TimeSlices= 1 ")
    lines.append(f"UNET D2 Turbulence Formulation={params.turbulence}")
    lines.append(f"UNET D2 Eddy Viscosity={params.eddy_viscosity}")
    lines.append("UNET D2 Transverse Eddy Viscosity=0.1")
    lines.append("UNET D2 Smagorinsky Mixing=0.05")
    lines.append("UNET D2 BCVolumeCheck=0")
    lines.append("UNET D2 Latitude=")
    lines.append("UNET D2 Cores=0")
    lines.append(f"UNET D2 SolverType={params.solver_type}")
    lines.append("UNET D2 Minimum Iterations= 3 ")
    lines.append("UNET D2 Maximum Iterations= 30 ")
    lines.append("UNET D2 Restart Number= 10 ")
    lines.append("UNET D2 Relaxation Coeff=1.3")
    lines.append("UNET D2 SOR Precondition Iterations= 10 ")
    lines.append("UNET D2 ILUT Maximum Fill= 8 ")
    lines.append("UNET D2 ILUT Tolerance=1E-08")
    lines.append("UNET D2 Convergence Tolerance=1E-05")
    lines.append("Secondary Flow Approach=None")
    # Per-area 2D solver settings
    lines.append(f"UNET D2 Name={area_name}")
    lines.append(f"UNET D2 Theta= {params.theta} ")
    lines.append(f"UNET D2 Theta Warmup= {params.theta_warmup} ")
    lines.append(f"UNET D2 Z Tol= {params.z_tolerance} ")
    lines.append(f"UNET D2 Volume Tol= {params.volume_tolerance} ")
    lines.append(f"UNET D2 Max Iterations= {params.max_iterations} ")
    lines.append(f"UNET D2 Equation= {params.equation_set} ")
    lines.append(f"UNET D2 RampUpFraction={params.ramp_up_fraction}")
    lines.append(f"UNET D2 Turbulence Formulation={params.turbulence}")
    lines.append(f"UNET D2 Eddy Viscosity={params.eddy_viscosity}")
    lines.append("UNET D2 SolverType=PARDISO (Direct)")
    # PS params
    lines.append("PS Theta= 1 ")
    lines.append("PS WS Tol= 0.01 ")
    lines.append("PS Volume Tol= 0.01 ")
    lines.append("PS Max Iterations= 20 ")
    lines.append("PS Equation= 0 ")
    lines.append("PS Cores=0")
    lines.append("UNET D1D2 MaxIter= 0 ")
    lines.append("UNET D1D2 ZTol=0.01")
    lines.append("UNET D1D2 QTol=0.1")
    lines.append("UNET D1D2 MinQTol=1")
    lines.append("DSS File=dss")
    lines.append("Write HDF5 File= 1 ")
    lines.append("Write Binary Output= 1 ")

    path.write_text("\r\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Project file writer — .prj
# ---------------------------------------------------------------------------


def write_project_file(
    path: Path,
    title: str,
    geom_numbers: List[str],
    flow_numbers: List[str],
    plan_numbers: List[str],
    is_us_customary: bool = True,
    dss_file: Optional[str] = None,
) -> None:
    """Write a .prj HEC-RAS project file.

    Parameters
    ----------
    path : Path
        Output file path.
    title : str
        Project title.
    geom_numbers : list of str
        Geometry file numbers (e.g. ``["01"]``).
    flow_numbers : list of str
        Unsteady flow file numbers (e.g. ``["01"]``).
    plan_numbers : list of str
        Plan file numbers (e.g. ``["01"]``).
    is_us_customary : bool
        Unit system.
    dss_file : str or None
        DSS file path.
    """
    lines: List[str] = []
    lines.append(f"Proj Title={title}")
    lines.append(f"Current Plan=p{plan_numbers[-1]}")
    lines.append("Default Exp/Contr=0.3,0.1")
    lines.append("English Units" if is_us_customary else "SI Units")
    for g in geom_numbers:
        lines.append(f"Geom File=g{g}")
    for f in flow_numbers:
        lines.append(f"Unsteady File=u{f}")
    for p in plan_numbers:
        lines.append(f"Plan File=p{p}")
    lines.append("Y Axis Title=Elevation")
    lines.append("X Axis Title(PF)=Main Channel Distance")
    lines.append("X Axis Title(XS)=Station")
    lines.append("BEGIN DESCRIPTION:")
    lines.append("")
    lines.append("END DESCRIPTION:")
    lines.append("DSS Start Date=")
    lines.append("DSS Start Time=")
    lines.append("DSS End Date=")
    lines.append("DSS End Time=")
    if dss_file:
        lines.append(f"DSS File={dss_file}")
    lines.append("DSS Export Filename=")
    lines.append("DSS Export Rating Curves= 0 ")
    lines.append("DSS Export Rating Curve Sorted= 0 ")
    lines.append("DSS Export Volume Flow Curves= 0 ")
    lines.append("DXF Filename=")
    lines.append("DXF OffsetX= 0 ")
    lines.append("DXF OffsetY= 0 ")
    lines.append("DXF ScaleX= 1 ")
    lines.append("DXF ScaleY= 10 ")
    lines.append("GIS Export Profiles= 0 ")
    path.write_text("\r\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def write_hecras_project(
    output_dir: Path,
    project_name: str,
    mesh_data: dict,
    flow_area: Hecras2DFlowArea,
    perimeter: np.ndarray,
    cell_centers: np.ndarray,
    bc_lines: List[HecrasBCLine],
    unsteady_boundaries: List[HecrasBoundary],
    connections: Optional[List[HecrasSA2DConnection]] = None,
    pipe_conduits: Optional[List[HecrasPipeConduit]] = None,
    pipe_nodes: Optional[List[HecrasPipeNode]] = None,
    top_inlets: Optional[List[HecrasTopInlet]] = None,
    plan_params: Optional[HecrasPlanParams] = None,
    geom_number: str = "01",
    flow_number: str = "01",
    plan_number: str = "01",
    is_us_customary: bool = False,
    manning_n: float = 0.03,
    projection_wkt: str = 'LOCAL_CS["Unknown"]',
    terrain_path: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Path:
    """Write a complete HEC-RAS 7.0 project from HYDRA2D model data.

    Creates the following files:

        {project_name}.prj       — project file
        {project_name}.g{geom}   — geometry text
        {project_name}.g{geom}.hdf — geometry HDF5
        {project_name}.u{flow}   — unsteady flow
        {project_name}.p{plan}   — plan

    Parameters
    ----------
    output_dir : Path
        Directory to write output files.
    project_name : str
        Project name (filename stem).
    mesh_data : dict
        Mesh geometry dict (node_x, node_y, node_z, cell_face_offsets,
        cell_face_nodes or cell_nodes).
    flow_area : Hecras2DFlowArea
        2D flow area parameters.
    perimeter : (n, 2) ndarray
        Perimeter polygon (closed ring).
    cell_centers : (n, 2) ndarray
        Cell center coordinates.
    bc_lines : list of HecrasBCLine
        BC line geometry definitions.
    unsteady_boundaries : list of HecrasBoundary
        BC hydrograph/ND boundary definitions for .u## file.
    connections : list of HecrasSA2DConnection or None
        SA/2D connections with optional culvert groups.
    pipe_conduits : list of HecrasPipeConduit or None
        Pipe conduits for network.
    pipe_nodes : list of HecrasPipeNode or None
        Pipe nodes for network.
    top_inlets : list of HecrasTopInlet or None
        Top inlet templates.
    plan_params : HecrasPlanParams or None
        Plan/solver parameters (defaults used if None).
    geom_number : str
        Geometry file number (e.g. '01' → .g01).
    flow_number : str
        Unsteady flow number (e.g. '01' → .u01).
    plan_number : str
        Plan file number (e.g. '01' → .p01).
    is_us_customary : bool
        Unit system flag.
    manning_n : float
        Manning's n for 2D flow area.
    projection_wkt : str
        CRS WKT string.
    terrain_path : str or None
        Terrain raster path (for HEC-RAS geometry reference).
    log_fn : callable or None
        Optional logging function.

    Returns
    -------
    Path
        The project directory.
    """
    if log_fn is None:
        log_fn = print

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = plan_params or HecrasPlanParams()
    connections = connections or []
    pipe_conduits = pipe_conduits or []
    pipe_nodes = pipe_nodes or []

    # Compute viewing rectangle from perimeter
    xmin = float(np.min(perimeter[:, 0]))
    xmax = float(np.max(perimeter[:, 0]))
    ymin = float(np.min(perimeter[:, 1]))
    ymax = float(np.max(perimeter[:, 1]))

    # ---- .g## text ----
    geom_path = output_dir / f"{project_name}.g{geom_number}"
    log_fn(f"Writing geometry text: {geom_path}")
    write_geom_text(
        path=geom_path,
        title=project_name,
        viewing_rect=(xmin, xmax, ymax, ymin),
        flow_area=flow_area,
        perimeter=perimeter,
        cell_centers=cell_centers,
        bc_lines=bc_lines,
        connections=connections,
    )

    # ---- .g##.hdf ----
    geom_hdf_path = output_dir / f"{project_name}.g{geom_number}.hdf"
    log_fn(f"Writing geometry HDF5: {geom_hdf_path}")
    terrain_hdf_path = None
    if terrain_path:
        terrain_hdf_path = terrain_path.replace("/", "\\")
    write_geom_hdf5(
        path=geom_hdf_path,
        mesh_data=mesh_data,
        flow_area=flow_area,
        perimeter=perimeter,
        is_us_customary=is_us_customary,
        manning_n=manning_n,
        projection_wkt=projection_wkt,
        terrain_path=terrain_path,
        terrain_hdf_path=terrain_hdf_path,
        pipe_conduits=pipe_conduits or None,
        pipe_nodes=pipe_nodes or None,
        top_inlets=top_inlets or None,
        bc_lines=bc_lines,
    )

    # ---- .u## text ----
    unsteady_path = output_dir / f"{project_name}.u{flow_number}"
    log_fn(f"Writing unsteady flow: {unsteady_path}")
    write_unsteady_flow(
        path=unsteady_path,
        title=f"{project_name}_unsteady",
        boundaries=unsteady_boundaries,
        start_date=params.simulation_date_start,
        start_time=params.simulation_time_start,
        computation_interval=params.computation_interval,
    )

    # ---- .p## text ----
    plan_path = output_dir / f"{project_name}.p{plan_number}"
    log_fn(f"Writing plan: {plan_path}")
    write_plan(
        path=plan_path,
        params=params,
        area_name=flow_area.name,
    )

    # ---- .prj text ----
    prj_path = output_dir / f"{project_name}.prj"
    log_fn(f"Writing project file: {prj_path}")
    write_project_file(
        path=prj_path,
        title=project_name,
        geom_numbers=[geom_number],
        flow_numbers=[flow_number],
        plan_numbers=[plan_number],
        is_us_customary=is_us_customary,
    )

    log_fn(f"Project written to {output_dir}")
    return output_dir
