"""Tests for swe2d.services.mesh_persistence_service.

These methods used to live on studio_dialog.py. They were moved here so the
dialog no longer performs build/serialize/persist pipelines inline.
"""
import pathlib

import numpy as np
import pytest


@pytest.fixture
def tiny_mesh():
    return {
        "node_x": np.array([0.0, 1.0, 0.5, 1.5], dtype=np.float64),
        "node_y": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        "node_z": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "cell_nodes": np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
    }


def test_save_and_load_baked_mesh_roundtrip(tmp_path: pathlib.Path, tiny_mesh):
    from swe2d.services.mesh_persistence_service import (
        save_baked_mesh,
        load_baked_mesh,
    )

    gpkg = tmp_path / "mesh_roundtrip.gpkg"
    name = "tiny"
    save_baked_mesh(tiny_mesh, str(gpkg), name)

    loaded = load_baked_mesh(str(gpkg), name)
    np.testing.assert_array_equal(loaded["node_x"], tiny_mesh["node_x"])
    np.testing.assert_array_equal(loaded["node_y"], tiny_mesh["node_y"])
    loaded_cells = loaded["cell_nodes"].reshape(-1, 3)
    expected_cells = tiny_mesh["cell_nodes"]
    assert loaded_cells.shape == expected_cells.shape
    assert set(map(tuple, loaded_cells.tolist())) == set(map(tuple, expected_cells.tolist()))


def test_load_baked_mesh_unknown_name_raises(tmp_path: pathlib.Path, tiny_mesh):
    from swe2d.services.mesh_persistence_service import (
        save_baked_mesh,
        load_baked_mesh,
    )

    gpkg = tmp_path / "mesh_unknown.gpkg"
    save_baked_mesh(tiny_mesh, str(gpkg), "tiny")
    with pytest.raises(KeyError):
        load_baked_mesh(str(gpkg), "does_not_exist")
