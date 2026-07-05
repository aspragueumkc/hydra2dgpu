import numpy as np


def test_sample_line_metrics_logic_empty_sample_map():
    from swe2d.workbench.services.runtime_source_application_service import _sample_line_metrics_logic
    ts, prof = _sample_line_metrics_logic(
        sample_map={},
        t_accum=0.0,
        h_s=np.array([1.0]),
        hu_s=np.array([0.0]),
        hv_s=np.array([0.0]),
        cell_solver_z=np.array([0.0]),
        gravity=9.81,
        h_min=1e-4,
        mesh_data={
            "node_x": np.array([0.0, 1.0, 0.0]),
            "node_y": np.array([0.0, 0.0, 1.0]),
            "cell_nodes": np.array([[0, 1, 2]], dtype=np.int32),
        },
    )
    assert ts == []
    assert prof == []
