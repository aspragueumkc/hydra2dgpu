from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from swe2d.mesh.meshing import MeshResult


@dataclass(frozen=True)
class MfemPreset:
    name: str
    args: Tuple[str, ...]
    description: str


MFEM_PRESETS: Dict[str, MfemPreset] = {
    "shape_unit": MfemPreset(
        name="shape_unit",
        args=("-mid", "2", "-tid", "1", "-qo", "4"),
        description="Shape optimization toward unit ideal elements.",
    ),
    "shape_equal_size": MfemPreset(
        name="shape_equal_size",
        args=("-mid", "2", "-tid", "2", "-qo", "4"),
        description="Shape optimization toward equal-size ideal elements.",
    ),
    "shape_initial_size": MfemPreset(
        name="shape_initial_size",
        args=("-mid", "2", "-tid", "3", "-qo", "4"),
        description="Shape optimization around the initial size field.",
    ),
    "shape_size_given": MfemPreset(
        name="shape_size_given",
        args=("-mid", "2", "-tid", "5", "-qo", "4"),
        description="Mixed shape plus given-size target with normalization.",
    ),
    "balanced_shape_size": MfemPreset(
        name="balanced_shape_size",
        args=("-mid", "14", "-tid", "2", "-qo", "4"),
        description="Balanced transition-focused metric with equal-size target for smoother interfaces.",
    ),
    "shape_size_orientation": MfemPreset(
        name="shape_size_orientation",
        args=("-mid", "14", "-tid", "5", "-qo", "4"),
        description="Shape, size, and orientation optimization with fixed boundary.",
    ),
}


def available_mfem_presets() -> List[str]:
    return list(MFEM_PRESETS.keys())


def build_mfem_tmop_args(
    *,
    preset_name: str,
    max_iterations: int,
    quality_weight: float,
    boundary_fit_weight: float,
    interface_fit_weight: float,
    min_det_j: float,
    preserve_boundary: bool,
    lock_boundary_nodes: bool,
) -> List[str]:
    """Translate workbench MFEM controls to mesh-optimizer CLI arguments.

    The existing topology controls are interpreted as follows:
    - boundary-fit + lock/preserve boundary -> boundary motion policy.
    - quality weight -> solver aggressiveness and smoothing tolerance.
    - interface-fit weight -> transition strategy (metric combination / worst-case handling).
    - min det(J) -> untangling limiting constant and barrier mode.
    """

    args: List[str] = ["-ni", str(max(1, int(max_iterations)))]

    # Boundary conformance: default to fixed boundaries when requested by controls.
    if lock_boundary_nodes or preserve_boundary or boundary_fit_weight >= 0.5:
        args.append("-fix-bnd")
    else:
        args.append("-bnd")

    # Transition handling across interfaces: stronger interface weight pushes adapted-size combos.
    if interface_fit_weight >= 0.20:
        args.extend(["-cmb", "2", "-nor", "-wctype", "0"])
    elif interface_fit_weight >= 0.05:
        args.extend(["-cmb", "1", "-wctype", "0"])
    else:
        args.extend(["-cmb", "0", "-wctype", "0", "-no-nor"])

    # Smoothing/quality behavior tuning.
    if quality_weight >= 1.25:
        args.extend(["-rtol", "1e-12", "-art", "2", "-li", "180"])
    elif quality_weight <= 0.75:
        args.extend(["-rtol", "1e-9", "-art", "1", "-li", "80"])
    else:
        args.extend(["-rtol", "1e-10", "-art", "1", "-li", "120"])

    # Untangling floor and barrier selection.
    limit_const = max(float(min_det_j), 1.0e-12)
    args.extend(["-lc", f"{limit_const:.12g}"])
    # Shifted/pseudo barriers are supported only by specific metrics in MFEM.
    barrier_supported = preset_name in {"shape_size_orientation"}
    if limit_const > 1.0e-10 and barrier_supported:
        args.extend(["-btype", "1"])
    else:
        args.extend(["-btype", "0"])

    return args


def find_mesh_optimizer(repo_root: Optional[str] = None) -> Optional[str]:
    candidates: List[Path] = []
    if repo_root:
        root = Path(repo_root)
    else:
        root = Path(__file__).resolve().parents[2]
    candidates.append(root / "external" / "mfem" / "build" / "miniapps" / "meshing" / "mesh-optimizer")
    candidates.append(root / "external" / "mfem" / "build" / "miniapps" / "meshing" / "mesh-optimizer.exe")
    for cand in candidates:
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _mesh_faces(mesh: MeshResult) -> List[np.ndarray]:
    offs = np.asarray(mesh.cell_face_offsets, dtype=np.int32)
    nodes = np.asarray(mesh.cell_face_nodes, dtype=np.int32)
    out: List[np.ndarray] = []
    for i in range(max(0, offs.size - 1)):
        s = int(offs[i])
        e = int(offs[i + 1])
        out.append(nodes[s:e])
    return out


def _boundary_segments(mesh: MeshResult) -> List[Tuple[int, int]]:
    edge_count: Dict[Tuple[int, int], int] = {}
    oriented: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for face in _mesh_faces(mesh):
        n = int(face.size)
        if n < 2:
            continue
        for i in range(n):
            a = int(face[i])
            b = int(face[(i + 1) % n])
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1
            oriented.setdefault(key, (a, b))
    segments: List[Tuple[int, int]] = []
    for key, count in edge_count.items():
        if count == 1:
            segments.append(oriented[key])
    return segments


def write_mfem_mesh(mesh: MeshResult, path: str) -> None:
    faces = _mesh_faces(mesh)
    boundary = _boundary_segments(mesh)
    node_x = np.asarray(mesh.node_x, dtype=np.float64)
    node_y = np.asarray(mesh.node_y, dtype=np.float64)
    region_id = np.asarray(mesh.region_id, dtype=np.int32) if mesh.region_id is not None else np.full(len(faces), 1, dtype=np.int32)

    with open(path, "w", encoding="utf-8") as f:
        f.write("MFEM mesh v1.0\n\n")
        f.write("dimension\n2\n\n")
        f.write("elements\n")
        f.write(f"{len(faces)}\n")
        for i, face in enumerate(faces):
            attr = int(region_id[i]) if i < region_id.size else 1
            geom = 2 if int(face.size) == 3 else 3
            conn = " ".join(str(int(v)) for v in face.tolist())
            f.write(f"{attr} {geom} {conn}\n")
        f.write("\n")
        f.write("boundary\n")
        f.write(f"{len(boundary)}\n")
        for a, b in boundary:
            f.write(f"1 1 {int(a)} {int(b)}\n")
        f.write("\n")
        f.write("vertices\n")
        f.write(f"{node_x.size}\n\n")
        f.write("nodes\n")
        f.write("FiniteElementSpace\n")
        f.write("FiniteElementCollection: H1_2D_P1\n")
        f.write("VDim: 2\n")
        f.write("Ordering: 0\n\n")
        for x in node_x.tolist():
            f.write(f"{x:.17g}\n")
        for y in node_y.tolist():
            f.write(f"{y:.17g}\n")


def _find_section(lines: Sequence[str], name: str) -> int:
    target = name.strip().lower()
    for i, line in enumerate(lines):
        if line.strip().lower() == target:
            return i
    return -1


def read_mfem_mesh(path: str, seed_mesh: MeshResult) -> MeshResult:
    from swe2d.mesh.meshing import MeshResult

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    elem_idx = _find_section(lines, "elements")
    node_idx = _find_section(lines, "nodes")
    if elem_idx < 0 or node_idx < 0:
        raise RuntimeError(f"Failed to parse MFEM mesh file: {path}")

    n_elem = int(lines[elem_idx + 1].strip())
    face_offsets: List[int] = [0]
    face_nodes: List[int] = []
    region_id: List[int] = []
    cell_type: List[str] = []
    tri_plot: List[int] = []
    for i in range(n_elem):
        parts = lines[elem_idx + 2 + i].split()
        if len(parts) < 5:
            continue
        attr = int(parts[0])
        geom = int(parts[1])
        conn = [int(v) for v in parts[2:]]
        face_nodes.extend(conn)
        face_offsets.append(len(face_nodes))
        region_id.append(attr)
        if geom == 2:
            cell_type.append("triangular")
            tri_plot.extend(conn[:3])
        elif geom == 3:
            cell_type.append("quadrilateral")
            tri_plot.extend([conn[0], conn[1], conn[2], conn[0], conn[2], conn[3]])
        else:
            cell_type.append("triangular")

    vertex_count_idx = _find_section(lines, "vertices")
    if vertex_count_idx < 0:
        raise RuntimeError(f"Failed to parse vertex count from MFEM mesh file: {path}")
    n_vertices = int(lines[vertex_count_idx + 1].strip())

    # nodes section format for ordering 0 / VDim 2: x block then y block
    values: List[float] = []
    for line in lines[node_idx + 5:]:
        txt = line.strip()
        if not txt:
            continue
        try:
            values.append(float(txt))
        except ValueError:
            pass
    if len(values) < 2 * n_vertices:
        raise RuntimeError(f"Insufficient nodal coordinates in MFEM mesh file: {path}")
    node_x = np.asarray(values[:n_vertices], dtype=np.float64)
    node_y = np.asarray(values[n_vertices:2 * n_vertices], dtype=np.float64)

    target_size = np.asarray(seed_mesh.target_size, dtype=np.float64)
    if target_size.size != len(region_id):
        if target_size.size > 0:
            fill = float(np.mean(target_size))
        else:
            fill = 1.0
        target_size = np.full(len(region_id), fill, dtype=np.float64)

    return MeshResult(
        node_x=node_x,
        node_y=node_y,
        node_z=np.zeros_like(node_x),
        cell_nodes=np.asarray(tri_plot, dtype=np.int32),
        cell_face_offsets=np.asarray(face_offsets, dtype=np.int32),
        cell_face_nodes=np.asarray(face_nodes, dtype=np.int32),
        cell_type=np.asarray(cell_type, dtype=object),
        region_id=np.asarray(region_id, dtype=np.int32),
        target_size=target_size,
        quality_summary=seed_mesh.quality_summary,
    )


def optimize_with_mfem(
    mesh: MeshResult,
    preset_name: str,
    repo_root: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    max_iterations: int = 120,
    quality_weight: float = 1.0,
    boundary_fit_weight: float = 0.35,
    interface_fit_weight: float = 0.25,
    min_det_j: float = 1.0e-9,
    preserve_boundary: bool = True,
    lock_boundary_nodes: bool = True,
) -> MeshResult:
    preset = MFEM_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(f"Unknown MFEM preset: {preset_name!r}")

    exe = find_mesh_optimizer(repo_root=repo_root)
    if exe is None:
        raise RuntimeError("MFEM mesh-optimizer executable not found. Build external/mfem first.")

    with tempfile.TemporaryDirectory(prefix="mfem_opt_") as td:
        in_mesh = os.path.join(td, "input.mesh")
        write_mfem_mesh(mesh, in_mesh)
        tuned_args = build_mfem_tmop_args(
            preset_name=preset_name,
            max_iterations=max_iterations,
            quality_weight=quality_weight,
            boundary_fit_weight=boundary_fit_weight,
            interface_fit_weight=interface_fit_weight,
            min_det_j=min_det_j,
            preserve_boundary=preserve_boundary,
            lock_boundary_nodes=lock_boundary_nodes,
        )
        cmd = [exe, "-m", in_mesh, "-o", "1", "-no-vis", *preset.args, *tuned_args]
        if extra_args:
            cmd.extend([str(v) for v in extra_args])
        proc = subprocess.run(cmd, cwd=td, check=False, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "MFEM mesh-optimizer failed for preset "
                f"{preset_name}:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        out_mesh = os.path.join(td, "optimized.mesh")
        if not os.path.exists(out_mesh):
            raise RuntimeError(f"MFEM mesh-optimizer did not produce optimized.mesh for preset {preset_name}")
        return read_mfem_mesh(out_mesh, seed_mesh=mesh)
