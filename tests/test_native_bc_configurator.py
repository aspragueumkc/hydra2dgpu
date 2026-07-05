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
        bc_n0=np.array([0, 1, 2], dtype=np.int32),
        bc_n1=np.array([1, 2, 3], dtype=np.int32),
        bc_tp=np.array([0, 0, 1], dtype=np.int32),
        node_x=np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
        node_y=np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        node_z=np.zeros(4, dtype=np.float64),
    )
    payload = cfg.build_payload()
    assert set(payload["side_by_edge"].tolist()) == {"bottom", "right", "top"}


def test_configurator_converts_bc_codes():
    from swe2d.boundary_and_forcing.native_bc_forcing import (
        BoundaryHydrographConfigurator,
    )

    cfg = BoundaryHydrographConfigurator(
        bc_n0=np.array([0], dtype=np.int32),
        bc_n1=np.array([1], dtype=np.int32),
        bc_tp=np.array([102], dtype=np.int32),
        node_x=np.array([0.0, 1.0], dtype=np.float64),
        node_y=np.array([0.0, 0.0], dtype=np.float64),
        node_z=np.zeros(2, dtype=np.float64),
        side_hydrographs={"bottom": (np.array([0.0]), np.array([1.0]))},
        inflow_q_bc_type=2,
        ts_flow_code=102,
        ts_stage_code=103,
    )
    payload = cfg.build_payload()
    assert payload["bc_type_native"][0] == 2  # 102 -> 2 (Q→h)


def test_configurator_returns_empty_payload_when_no_hydrographs():
    from swe2d.boundary_and_forcing.native_bc_forcing import (
        BoundaryHydrographConfigurator,
    )

    cfg = BoundaryHydrographConfigurator(
        bc_n0=np.array([0], dtype=np.int32),
        bc_n1=np.array([1], dtype=np.int32),
        bc_tp=np.array([102], dtype=np.int32),
        node_x=np.array([0.0, 1.0], dtype=np.float64),
        node_y=np.array([0.0, 0.0], dtype=np.float64),
        node_z=np.zeros(2, dtype=np.float64),
    )
    payload = cfg.build_payload()
    assert payload["native_bc_forcing"] is False
    assert payload["configured_edges"] == 0


def test_configurator_scales_flow_by_total_edge_length():
    from swe2d.boundary_and_forcing.native_bc_forcing import (
        BoundaryHydrographConfigurator,
    )

    # Two left-side edges, each length 1, sharing a total-Q hydrograph of 10.
    cfg = BoundaryHydrographConfigurator(
        bc_n0=np.array([0, 1], dtype=np.int32),
        bc_n1=np.array([1, 2], dtype=np.int32),
        bc_tp=np.array([102, 102], dtype=np.int32),
        node_x=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        node_y=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        node_z=np.zeros(3, dtype=np.float64),
        side_hydrographs={"left": (np.array([0.0]), np.array([10.0]))},
        inflow_q_bc_type=2,
        ts_flow_code=102,
    )
    payload = cfg.build_payload()
    assert payload["native_bc_forcing"] is True
    assert payload["configured_edges"] == 2
    # Total length is 2, so unit q at t=0 should be 10 / 2 = 5 per edge.
    np.testing.assert_allclose(payload["value_native"], [5.0, 5.0])


def test_configurator_builds_progressive_data_when_requested():
    from swe2d.boundary_and_forcing.native_bc_forcing import (
        BoundaryHydrographConfigurator,
    )

    # Two left-side edges at different elevations.
    cfg = BoundaryHydrographConfigurator(
        bc_n0=np.array([0, 1], dtype=np.int32),
        bc_n1=np.array([1, 2], dtype=np.int32),
        bc_tp=np.array([102, 102], dtype=np.int32),
        node_x=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        node_y=np.array([0.0, 1.0, 2.0], dtype=np.float64),
        node_z=np.array([0.0, 1.0, 0.5], dtype=np.float64),
        side_hydrographs={"left": (np.array([0.0]), np.array([10.0]))},
        progressive=True,
        inflow_q_bc_type=2,
        ts_flow_code=102,
    )
    payload = cfg.build_payload()
    assert payload["progressive_data"] is not None
    assert payload["progressive_data"]["n_groups"] == 1
    assert payload["progressive_data"]["n_edges_total"] == 2
