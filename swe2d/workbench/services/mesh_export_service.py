"""Mesh-export service — convert in-memory mesh arrays to QGIS vector layers.

Mesh-to-layer conversion and field mapping, built for the Studio architecture.
Zero Qt — no PyQt5 imports.
"""
from __future__ import annotations

import numpy as np


def _build_point_layer(crs_auth: str) -> "QgsVectorLayer":
    """build point layer."""
    from qgis.core import QgsVectorLayer
    return QgsVectorLayer(
        f"Point?crs={crs_auth}&field=node_id:integer&field=bed_z:double",
        "SWE2D_Mesh_Nodes",
        "memory",
    )


def _build_polygon_layer(crs_auth: str) -> "QgsVectorLayer":
    """build polygon layer."""
    from qgis.core import QgsVectorLayer
    return QgsVectorLayer(
        f"Polygon?crs={crs_auth}"
        "&field=cell_id:integer"
        "&field=n0:integer"
        "&field=n1:integer"
        "&field=n2:integer"
        "&field=n3:integer"
        "&field=node_ids:string(512)"
        "&field=cell_type:string(32)"
        "&field=region_id:integer"
        "&field=target_size:double",
        "SWE2D_Mesh_Cells",
        "memory",
    )


def build_nodes_vector_layer(
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
    crs_auth: str = "EPSG:4326",
) -> "QgsVectorLayer":
    """Build a QgsVectorLayer from mesh node coordinates."""
    from qgis.core import QgsFeature, QgsGeometry, QgsPointXY

    nodes_layer = _build_point_layer(crs_auth)
    feats = []
    for i in range(len(node_x)):
        f = QgsFeature(nodes_layer.fields())
        f.setAttribute("node_id", int(i))
        f.setAttribute("bed_z", float(node_z[i]))
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(
            float(node_x[i]), float(node_y[i]),
        )))
        feats.append(f)
    nodes_layer.dataProvider().addFeatures(feats)
    nodes_layer.updateExtents()
    return nodes_layer


def build_cells_polygon_layer(
    node_x: np.ndarray,
    node_y: np.ndarray,
    cell_nodes: np.ndarray,
    cell_face_offsets: np.ndarray | None = None,
    cell_face_nodes: np.ndarray | None = None,
    cell_type_meta: np.ndarray | None = None,
    region_meta: np.ndarray | None = None,
    size_meta: np.ndarray | None = None,
    crs_auth: str = "EPSG:4326",
) -> "QgsVectorLayer":
    """Build a polygon QgsVectorLayer from mesh cell connectivity."""
    from qgis.core import QgsFeature, QgsGeometry, QgsPointXY

    cells_layer = _build_polygon_layer(crs_auth)

    if cell_face_offsets is not None and cell_face_nodes is not None:
        offs = np.asarray(cell_face_offsets, dtype=np.int32).ravel()
        nodes = np.asarray(cell_face_nodes, dtype=np.int32).ravel()
        face_ids = [
            nodes[int(offs[i]):int(offs[i + 1])].tolist()
            for i in range(max(0, int(offs.size) - 1))
        ]
    else:
        face_ids = [tri.tolist() for tri in cell_nodes.reshape((-1, 3))]

    feats = []
    for cid, ids in enumerate(face_ids):
        ids_i = [int(v) for v in ids]
        if len(ids_i) < 3:
            continue
        poly = [QgsPointXY(float(node_x[n]), float(node_y[n])) for n in ids_i]
        poly.append(poly[0])
        f = QgsFeature(cells_layer.fields())
        f.setAttribute("cell_id", int(cid))
        f.setAttribute("n0", int(ids_i[0]) if len(ids_i) > 0 else None)
        f.setAttribute("n1", int(ids_i[1]) if len(ids_i) > 1 else None)
        f.setAttribute("n2", int(ids_i[2]) if len(ids_i) > 2 else None)
        f.setAttribute("n3", int(ids_i[3]) if len(ids_i) > 3 else None)
        f.setAttribute("node_ids", ",".join(str(n) for n in ids_i))
        if cell_type_meta is not None and cid < len(cell_type_meta):
            f.setAttribute("cell_type", str(cell_type_meta[cid]))
        else:
            f.setAttribute("cell_type", "quadrilateral" if len(ids_i) == 4 else "triangular")
        if region_meta is not None and cid < len(region_meta):
            f.setAttribute("region_id", int(region_meta[cid]))
        else:
            f.setAttribute("region_id", -1)
        if size_meta is not None and cid < len(size_meta):
            f.setAttribute("target_size", float(size_meta[cid]))
        else:
            f.setAttribute("target_size", 0.0)
        f.setGeometry(QgsGeometry.fromPolygonXY([poly]))
        feats.append(f)

    cells_layer.dataProvider().addFeatures(feats)
    cells_layer.updateExtents()
    return cells_layer
