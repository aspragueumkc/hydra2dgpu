"""Tests for boundary-edge classification and coupling validation."""
import numpy as np
import pytest


def test_classify_boundary_edges_basic():
    from swe2d.workbench.services.mesh_service import classify_boundary_edges
    node_x = np.array([0.0, 10.0, 10.0, 0.0], dtype=np.float64)
    node_y = np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64)
    bc_n0 = np.array([0, 1], dtype=np.int32)
    bc_n1 = np.array([1, 2], dtype=np.int32)
    edge_len, side_idx, side_names = classify_boundary_edges(
        node_x, node_y, bc_n0, bc_n1,
    )
    assert edge_len.shape == (2,)
    assert side_idx.shape == (2,)
    assert side_names == ["left", "right", "bottom", "top"]


def test_classify_boundary_edges_empty():
    from swe2d.workbench.services.mesh_service import classify_boundary_edges
    empty = np.array([], dtype=np.int32)
    edge_len, side_idx, _ = classify_boundary_edges(
        np.array([0.0]), np.array([0.0]), empty, empty,
    )
    assert edge_len.size == 0
    assert side_idx.size == 0


def test_validate_coupling_configs_no_configs():
    from swe2d.runtime.coupling import validate_coupling_configs
    report = validate_coupling_configs(
        pipe_cfg=None, struct_cfg=None, n_cells=10,
    )
    assert isinstance(report, list)
    assert any("not configured" in line for line in report)
