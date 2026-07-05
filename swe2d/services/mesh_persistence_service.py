"""Mesh (de)serialization into the model GPKG.

Extracted from swe2d.workbench.studio_dialog so the dialog no longer performs
build/serialize/persist pipelines inline. Pure numpy + sqlite3 (no Qt).
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List

import numpy as np


def save_baked_mesh(mesh_data: Dict[str, np.ndarray], gpkg_path: str, mesh_name: str) -> int:
    """Serialize ``mesh_data`` via the hydra_swe2d C extension and persist it
    under ``mesh_name`` in the GPKG. Returns the number of cells in the baked
    BLOB so callers can log/sanity-check.
    """
    from hydra_swe2d import (
        swe2d_build_mesh, swe2d_build_mesh_poly,
        swe2d_serialize_mesh, swe2d_mesh_info,
    )
    from swe2d.services.gpkg_persistence_service import persist_baked_mesh

    nx = np.asarray(mesh_data["node_x"], dtype=np.float64)
    ny = np.asarray(mesh_data["node_y"], dtype=np.float64)
    nz = np.asarray(mesh_data["node_z"], dtype=np.float64)
    bc_n0 = np.asarray(mesh_data.get("bc_edge_node0", np.empty(0)), dtype=np.int32)
    bc_n1 = np.asarray(mesh_data.get("bc_edge_node1", np.empty(0)), dtype=np.int32)
    bc_tp = np.asarray(mesh_data.get("bc_edge_type", np.empty(0)), dtype=np.int32)
    bc_vl = np.asarray(mesh_data.get("bc_edge_val", np.empty(0)), dtype=np.float64)
    cfn = mesh_data.get("cell_face_nodes")
    if cfn is None:
        cfn = mesh_data.get("cell_nodes")
    cfo = mesh_data.get("cell_face_offsets")
    if cfn is not None and cfo is not None:
        pm = swe2d_build_mesh_poly(
            nx, ny, nz,
            np.asarray(cfo, dtype=np.int32),
            np.asarray(cfn, dtype=np.int32),
            bc_n0, bc_n1, bc_tp, bc_vl,
        )
    else:
        cn = np.asarray(mesh_data["cell_nodes"], dtype=np.int32)
        pm = swe2d_build_mesh(nx, ny, nz, cn, bc_n0, bc_n1, bc_tp, bc_vl)
    blob = swe2d_serialize_mesh(pm)
    info = swe2d_mesh_info(pm)
    persist_baked_mesh(
        gpkg_path, mesh_name, blob,
        info["n_nodes"], info["n_cells"], info["n_edges"],
    )
    return int(info["n_cells"])


def list_baked_mesh_names(gpkg_path: str) -> List[str]:
    """Return baked-mesh names in ``gpkg_path`` ordered newest-first."""
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT mesh_name FROM swe2d_baked_mesh ORDER BY created_utc DESC")
        return [str(r[0]) for r in cur.fetchall()]
    finally:
        conn.close()


def load_baked_mesh(gpkg_path: str, mesh_name: str) -> Dict[str, np.ndarray]:
    """Load a previously-saved mesh by name. Raises KeyError if not present.

    Mesh geometry is returned in solver (RCMK) order per the baked BLOB spec
    (§5.12). Results loaded from ``load_baked_snapshot`` are also RCMK, so no
    permutation is required when pairing mesh + results.
    """
    from hydra_swe2d import swe2d_deserialize_mesh
    from swe2d.services.gpkg_persistence_service import load_baked_mesh as _load_blob

    blob = _load_blob(gpkg_path, mesh_name)
    if blob is None:
        raise KeyError(mesh_name)
    pm = swe2d_deserialize_mesh(blob)
    mesh_data: Dict[str, np.ndarray] = {
        "mesh_name": str(mesh_name),
        "node_x": np.asarray(pm.node_x, dtype=np.float64),
        "node_y": np.asarray(pm.node_y, dtype=np.float64),
        "node_z": np.asarray(pm.node_z, dtype=np.float64),
    }
    if pm.cell_face_nodes is not None:
        mesh_data["cell_nodes"] = np.asarray(pm.cell_face_nodes, dtype=np.int32)
    else:
        mesh_data["cell_nodes"] = np.empty(0, dtype=np.int32)
    if pm.cell_face_offsets is not None:
        mesh_data["cell_face_offsets"] = np.asarray(pm.cell_face_offsets, dtype=np.int32)
    if pm.cell_face_nodes is not None:
        mesh_data["cell_face_nodes"] = np.asarray(pm.cell_face_nodes, dtype=np.int32)
    return mesh_data
