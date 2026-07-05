import numpy as np
from unittest.mock import MagicMock, patch


def test_distribute_total_flow_to_unit_q_logic_forwards_args():
    from swe2d.workbench.services.runtime_source_application_service import _distribute_total_flow_to_unit_q_logic
    with patch("swe2d.workbench.services.runtime_source_application_service.distribute_total_flow_to_unit_q") as mock_logic:
        _distribute_total_flow_to_unit_q_logic(
            edge_n0=np.array([0], dtype=np.int32),
            edge_n1=np.array([1], dtype=np.int32),
            bc_type_step=np.array([0], dtype=np.int32),
            bc_val_step=np.array([0.0]),
            bc_type_template=np.array([0], dtype=np.int32),
            side_hydrographs={},
            node_x=np.array([0.0, 1.0]),
            node_y=np.array([0.0, 0.0]),
            node_z=np.array([0.0, 0.0]),
            progressive=False,
        )
        mock_logic.assert_called_once()
