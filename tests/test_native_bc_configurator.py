"""Tests for swe2d.boundary_and_forcing.native_bc_forcing.BoundaryHydrographConfigurator.

Extracted from swe2d.runtime.native_bc_forcing. The configurator is pure logic;
it no longer touches the backend directly — it returns a payload that the
runtime applies.
"""
import numpy as np
import pytest


def test_configurator_classifies_edges_by_side():
    from swe2d.boundary_and_forcing.native_bc_forcing import (
        BoundaryHydrographConfigurator,
    )

    cfg = BoundaryHydrographConfigurator(
        edge_nodes=np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32),
        node_coords=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64
        ),
        edge_groups=np.array([0, 0, 1], dtype=np.int32),
    )
    payload = cfg.build_payload()
    assert set(payload["side_by_edge"].tolist()) == {"left", "right", "top"}


def test_configurator_converts_bc_codes():
    from swe2d.boundary_and_forcing.native_bc_forcing import (
        BoundaryHydrographConfigurator,
    )

    cfg = BoundaryHydrographConfigurator(
        edge_nodes=np.array([[0, 1]], dtype=np.int32),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        edge_groups=np.array([0], dtype=np.int32),
        bc_codes_input=np.array([102], dtype=np.int32),
    )
    payload = cfg.build_payload()
    assert payload["bc_codes_output"][0] == 2  # 102 -> 2 (Q→h)
