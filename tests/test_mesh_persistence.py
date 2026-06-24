"""Test mesh save/load round-trip to GeoPackage."""
import os
import tempfile
import numpy as np
from swe2d.workbench.services.gpkg_persistence_service import (
    persist_mesh_to_geopackage,
    load_mesh_from_geopackage,
)


def test_mesh_round_trip_triangles():
    """Save a simple triangle mesh to GPKG and load it back."""
    mesh_data = {
        "node_x": np.array([0.0, 100.0, 50.0, 50.0], dtype=np.float64),
        "node_y": np.array([0.0, 0.0, 50.0, -50.0], dtype=np.float64),
        "node_z": np.array([10.0, 10.0, 8.0, 12.0], dtype=np.float64),
        "cell_nodes": np.array([0, 1, 2, 0, 3, 1], dtype=np.int32),
    }
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
        gpkg = f.name
    try:
        persist_mesh_to_geopackage(gpkg, "test_tri", mesh_data)
        loaded = load_mesh_from_geopackage(gpkg, "test_tri")
        assert loaded is not None, "load returned None"
        for key in ("node_x", "node_y", "node_z", "cell_nodes"):
            assert key in loaded, f"Missing key: {key}"
            np.testing.assert_array_almost_equal(loaded[key], mesh_data[key])
    finally:
        if os.path.exists(gpkg):
            os.unlink(gpkg)


def test_mesh_round_trip_with_bc():
    """Round trip with boundary condition arrays."""
    mesh_data = {
        "node_x": np.array([0.0, 100.0, 50.0, 50.0], dtype=np.float64),
        "node_y": np.array([0.0, 0.0, 50.0, -50.0], dtype=np.float64),
        "node_z": np.array([10.0, 10.0, 8.0, 12.0], dtype=np.float64),
        "cell_nodes": np.array([0, 1, 2, 0, 3, 1], dtype=np.int32),
        "bc_edge_node0": np.array([0, 0], dtype=np.int32),
        "bc_edge_node1": np.array([1, 3], dtype=np.int32),
        "bc_edge_type": np.array([1, 2], dtype=np.int32),
        "bc_edge_val": np.array([0.0, 5.0], dtype=np.float64),
    }
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
        gpkg = f.name
    try:
        persist_mesh_to_geopackage(gpkg, "test_bc", mesh_data)
        loaded = load_mesh_from_geopackage(gpkg, "test_bc")
        assert loaded is not None
        for key in ("bc_edge_node0", "bc_edge_node1", "bc_edge_type", "bc_edge_val"):
            assert key in loaded, f"Missing BC key: {key}"
            np.testing.assert_array_equal(loaded[key], mesh_data[key])
    finally:
        if os.path.exists(gpkg):
            os.unlink(gpkg)


def test_mesh_round_trip_nonexistent():
    """Loading a non-existent mesh returns None."""
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
        gpkg = f.name
    try:
        conn = __import__("sqlite3").connect(gpkg)
        conn.execute("""
            CREATE TABLE swe2d_mesh (
                mesh_name TEXT PRIMARY KEY,
                created_utc TEXT, nnodes INTEGER, ncells INTEGER,
                crs_wkt TEXT, hash TEXT,
                node_x BLOB, node_y BLOB, node_z BLOB,
                cell_nodes BLOB, face_offsets BLOB,
                bc_n0 BLOB, bc_n1 BLOB, bc_type BLOB, bc_val BLOB,
                description TEXT
            )
        """)
        conn.commit()
        conn.close()
        result = load_mesh_from_geopackage(gpkg, "no_such_mesh")
        assert result is None, "Expected None for missing mesh"
    finally:
        if os.path.exists(gpkg):
            os.unlink(gpkg)
