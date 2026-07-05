import numpy as np
from unittest.mock import patch


def test_distribute_total_flow_to_unit_q_logic_computes_invariants():
    from swe2d.workbench.services.runtime_source_application_service import (
        _distribute_total_flow_to_unit_q_logic,
    )

    edge_n0 = np.array([0], dtype=np.int32)
    edge_n1 = np.array([1], dtype=np.int32)
    node_x = np.array([0.0, 1.0])
    node_y = np.array([0.0, 0.0])
    node_z = np.array([0.0, 0.0])

    with patch(
        "swe2d.workbench.services.runtime_source_application_service.distribute_total_flow_to_unit_q"
    ) as mock_logic, patch(
        "swe2d.workbench.services.runtime_source_application_service._bc_side_classification"
    ) as mock_classify:
        expected_return = np.array([3.0])
        mock_logic.return_value = expected_return
        mock_classify.return_value = (
            np.array([3], dtype=np.int32),
            np.array([1.0]),
            np.array([0.0]),
        )

        result = _distribute_total_flow_to_unit_q_logic(
            edge_n0=edge_n0,
            edge_n1=edge_n1,
            bc_type_step=np.array([0], dtype=np.int32),
            bc_val_step=np.array([0.0]),
            bc_type_template=np.array([0], dtype=np.int32),
            side_hydrographs={},
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
            progressive=False,
        )

        mock_classify.assert_called_once_with(edge_n0, edge_n1, node_x, node_y, node_z)
        mock_logic.assert_called_once()
        kwargs = mock_logic.call_args.kwargs
        assert kwargs["progressive"] is False
        assert kwargs["ts_flow_code"] == 102
        assert kwargs["edge_groups"] is None
        assert np.array_equal(kwargs["_side_idx"], mock_classify.return_value[0])
        assert np.array_equal(kwargs["_edge_len"], mock_classify.return_value[1])
        assert np.array_equal(kwargs["_edge_z"], mock_classify.return_value[2])
        assert result is expected_return


def test_distribute_total_flow_to_unit_q_logic_uses_precomputed_invariants():
    from swe2d.workbench.services.runtime_source_application_service import (
        _distribute_total_flow_to_unit_q_logic,
    )

    edge_n0 = np.array([0], dtype=np.int32)
    edge_n1 = np.array([1], dtype=np.int32)
    node_x = np.array([0.0, 1.0])
    node_y = np.array([0.0, 0.0])
    node_z = np.array([0.0, 0.0])
    side_idx = np.array([3], dtype=np.int32)
    edge_len = np.array([1.0])
    edge_z = np.array([0.5])

    with patch(
        "swe2d.workbench.services.runtime_source_application_service.distribute_total_flow_to_unit_q"
    ) as mock_logic, patch(
        "swe2d.workbench.services.runtime_source_application_service._bc_side_classification"
    ) as mock_classify:
        expected_return = np.array([7.0])
        mock_logic.return_value = expected_return

        result = _distribute_total_flow_to_unit_q_logic(
            edge_n0=edge_n0,
            edge_n1=edge_n1,
            bc_type_step=np.array([0], dtype=np.int32),
            bc_val_step=np.array([0.0]),
            bc_type_template=np.array([0], dtype=np.int32),
            side_hydrographs={},
            node_x=node_x,
            node_y=node_y,
            node_z=node_z,
            progressive=False,
            _side_idx=side_idx,
            _edge_len=edge_len,
            _edge_z=edge_z,
        )

        mock_classify.assert_not_called()
        mock_logic.assert_called_once()
        kwargs = mock_logic.call_args.kwargs
        assert kwargs["progressive"] is False
        assert kwargs["ts_flow_code"] == 102
        assert kwargs["edge_groups"] is None
        assert kwargs["_side_idx"] is side_idx
        assert kwargs["_edge_len"] is edge_len
        assert kwargs["_edge_z"] is edge_z
        assert result is expected_return
