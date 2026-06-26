"""Mesh extraction service — converts QGIS vector layers to numpy mesh dicts.

Pure data extraction: reads node/cell features from QGIS vector layers
and returns a numpy-based mesh_data dictionary.  Zero Qt imports beyond
the QGIS core types that are inherent to layer-iteration.

This service was extracted from ``extracted/topology_and_io_methods.py``
(Phase B3 of the Extracted Migration Plan).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

__all__ = ["extract_mesh_from_layer_data"]


def _parse_face_node_ids(value: object, id_to_idx: Dict[int, int], log_fn) -> list:
    """Parse a face-node-id text field into a list of zero-based indices.

    Accepts semicolon- or comma-separated integer node ids, looks each
    up in *id_to_idx*.  Logs conversion errors via *log_fn* but never
    raises — a faulty id is simply skipped.
    """
    txt = str(value or "").strip()
    if not txt:
        return []
    out: List[int] = []
    for part in txt.replace(";", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            out.append(id_to_idx[int(p)])
        except Exception as e:
            log_fn(f"[ERROR] face node id lookup: {e}")
            continue
    return out


def extract_mesh_from_layer_data(
    nodes_layer,
    cells_layer,
    log_fn,
) -> dict:
    """Pure data extraction: convert QGIS vector layers to numpy mesh dict.

    Parameters
    ----------
    nodes_layer:
        QGIS vector layer (``QgsVectorLayer``) with node features.
        Expected fields: ``node_id`` (int, optional), ``bed_z`` (float).
    cells_layer:
        QGIS vector layer (``QgsVectorLayer``) with cell features.
        Supported field patterns: ``node_ids`` (string), ``n0``..``nN``
        (int), polygon geometry fallback.  Optional fields:
        ``cell_type``, ``region_id``, ``target_size``.
    log_fn:
        Callable for logging diagnostic/error messages.

    Returns
    -------
    dict
        A mesh_data dict with keys: ``node_x``, ``node_y``, ``node_z``,
        ``cell_nodes``, ``cell_face_offsets``, ``cell_face_nodes``,
        ``cell_type``, ``region_id``, ``target_size``, ``nx``, ``ny``,
        ``lx``, ``ly``.  Returns an empty dict when no valid data is found.
    """
    if nodes_layer is None or cells_layer is None:
        return {}

    nodes_by_id: Dict[int, Tuple[float, float, float]] = {}
    auto_id = 0
    for ft in nodes_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        pt = geom.asPoint()
        nid = ft["node_id"] if "node_id" in nodes_layer.fields().names() else None
        if nid is None:
            nid = auto_id
            auto_id += 1
        try:
            nid_i = int(nid)
        except Exception as e:
            log_fn(f"[ERROR] import node_id int conversion: {e}")
            continue
        z = 0.0
        if "bed_z" in nodes_layer.fields().names():
            try:
                z = float(ft["bed_z"])
            except Exception as e:
                log_fn(f"[ERROR] import bed_z float conversion: {e}")
                z = 0.0
        nodes_by_id[nid_i] = (float(pt.x()), float(pt.y()), z)

    if not nodes_by_id:
        return {}

    node_ids = sorted(nodes_by_id.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    node_x = np.array([nodes_by_id[nid][0] for nid in node_ids], dtype=np.float64)
    node_y = np.array([nodes_by_id[nid][1] for nid in node_ids], dtype=np.float64)
    node_z = np.array([nodes_by_id[nid][2] for nid in node_ids], dtype=np.float64)

    coord_to_idx = {
        (round(node_x[i], 9), round(node_y[i], 9)): i for i in range(node_x.shape[0])
    }

    face_list: List[List[int]] = []
    tri_list: List[int] = []
    cell_type_vals: List[str] = []
    region_vals: List[int] = []
    size_vals: List[float] = []

    cell_field_names = set(cells_layer.fields().names())
    for ft in cells_layer.getFeatures():
        ids: List[int] = []

        if "node_ids" in cell_field_names:
            try:
                ids = _parse_face_node_ids(ft["node_ids"], id_to_idx, log_fn)
            except Exception as e:
                log_fn(f"[ERROR] face node ids parse: {e}")
                ids = []

        if not ids:
            n_keys = sorted(k for k in cell_field_names if len(k) >= 2 and k[0] == "n" and k[1:].isdigit())
            raw_ids: List[int] = []
            for key in n_keys:
                try:
                    v = ft[key]
                    if v is None:
                        continue
                    raw_ids.append(int(v))
                except Exception as e:
                    log_fn(f"[ERROR] n_key {key} int conversion: {e}")
                    continue
            if len(raw_ids) >= 3:
                try:
                    ids = [id_to_idx[v] for v in raw_ids]
                except Exception as e:
                    log_fn(f"[ERROR] raw_ids to idx lookup: {e}")
                    ids = []

        if not ids:
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            poly = geom.asPolygon()
            if not poly or not poly[0]:
                continue
            ring = poly[0]
            verts: List[int] = []
            for p in ring[:-1]:
                key = (round(float(p.x()), 9), round(float(p.y()), 9))
                if key in coord_to_idx:
                    verts.append(coord_to_idx[key])
            uniq: List[int] = []
            for vid in verts:
                if vid not in uniq:
                    uniq.append(vid)
            ids = uniq

        if len(ids) >= 2 and ids[0] == ids[-1]:
            ids = ids[:-1]
        uniq_ids: List[int] = []
        for nid in ids:
            if nid not in uniq_ids:
                uniq_ids.append(int(nid))
        ids = uniq_ids
        if len(ids) < 3:
            continue

        face_list.append(ids)
        for k in range(1, len(ids) - 1):
            tri_list.extend([int(ids[0]), int(ids[k]), int(ids[k + 1])])

        ctype = ""
        if "cell_type" in cell_field_names:
            try:
                ctype = str(ft["cell_type"] or "").strip().lower()
            except Exception as e:
                log_fn(f"[ERROR] cell_type field read: {e}")
                ctype = ""
        if not ctype:
            ctype = "quadrilateral" if len(ids) == 4 else "triangular"
        cell_type_vals.append(ctype)

        reg_v = -1
        if "region_id" in cell_field_names:
            try:
                reg_v = int(ft["region_id"])
            except Exception as e:
                log_fn(f"[ERROR] region_id int conversion: {e}")
                reg_v = -1
        region_vals.append(reg_v)

        ts_v = 0.0
        if "target_size" in cell_field_names:
            try:
                ts_v = float(ft["target_size"])
            except Exception as e:
                log_fn(f"[ERROR] target_size float conversion: {e}")
                ts_v = 0.0
        size_vals.append(ts_v)

    if not face_list:
        return {}

    cell_nodes = np.array(tri_list, dtype=np.int32)
    if cell_nodes.size % 3 != 0:
        cell_nodes = cell_nodes[: (cell_nodes.size // 3) * 3]

    face_offsets = [0]
    face_nodes_flat: List[int] = []
    for ids in face_list:
        face_nodes_flat.extend(ids)
        face_offsets.append(face_offsets[-1] + len(ids))
    cell_face_nodes = np.asarray(face_nodes_flat, dtype=np.int32)
    cell_face_offsets = np.asarray(face_offsets, dtype=np.int32)

    if node_x.size >= 2:
        lx = float(np.max(node_x) - np.min(node_x))
        ly = float(np.max(node_y) - np.min(node_y))
    else:
        lx, ly = 1.0, 1.0

    return {
        "nx": np.array(max(2, int(round(np.sqrt(node_x.size))))),
        "ny": np.array(max(2, int(round(np.sqrt(node_x.size))))),
        "lx": np.array(max(lx, 1.0)),
        "ly": np.array(max(ly, 1.0)),
        "node_x": node_x,
        "node_y": node_y,
        "node_z": node_z,
        "cell_nodes": cell_nodes,
        "cell_face_offsets": cell_face_offsets,
        "cell_face_nodes": cell_face_nodes,
        "cell_type": np.asarray(cell_type_vals, dtype=object),
        "region_id": np.asarray(region_vals, dtype=np.int32),
        "target_size": np.asarray(size_vals, dtype=np.float64),
    }
