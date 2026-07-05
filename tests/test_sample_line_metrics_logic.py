import numpy as np


def test_sample_line_metrics_logic_empty_sample_map():
    from swe2d.workbench.services.line_sampling_service import _sample_line_metrics_logic
    ts, prof = _sample_line_metrics_logic(
        sample_map=[],
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


def test_sample_line_metrics_logic_profiles():
    from swe2d.workbench.services.line_sampling_service import _sample_line_metrics_logic
    from swe2d.services.line_sampling_service import sample_line_metrics

    node_x = np.array([0.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0])
    cell_nodes = np.array([[0, 1, 2]], dtype=np.int32)
    h = np.array([2.0])
    hu = np.array([1.0])
    hv = np.array([0.5])
    bed = np.array([0.5])
    sample_map = [{
        "line_id": 1,
        "line_name": "test-line",
        "cell_idx": np.array([0], dtype=np.int32),
        "weights": np.array([1.0], dtype=np.float64),
        "normal_x": 0.0,
        "normal_y": 1.0,
        "station_m": np.array([0.0], dtype=np.float64),
        "profile_station_m": np.linspace(0.0, 1.0, 5, dtype=np.float64),
        "profile_cell_idx": np.full((5, 1), 0, dtype=np.int32),
        "profile_cell_w": np.ones((5, 1), dtype=np.float64),
    }]

    ts, prof = _sample_line_metrics_logic(
        sample_map=sample_map,
        t_accum=10.0,
        h_s=h,
        hu_s=hu,
        hv_s=hv,
        cell_solver_z=bed,
        gravity=9.81,
        h_min=1e-4,
        mesh_data={
            "node_x": node_x,
            "node_y": node_y,
            "cell_nodes": cell_nodes,
        },
    )

    assert len(ts) == 1
    assert ts[0]["line_id"] == 1
    assert ts[0]["t_s"] == 10.0
    assert len(prof) == 1
    profile = prof[0]
    assert profile["line_id"] == 1
    assert profile["station_m"].size == 5
    assert np.allclose(profile["depth_m"], 2.0)
    assert np.allclose(profile["bed_m"], 0.5)
    assert np.allclose(profile["wse_m"], 2.5)

    # Verify the underlying service was reached with the same sample_map.
    direct = sample_line_metrics(
        h=h, hu=hu, hv=hv, bed=bed,
        node_coords=np.column_stack([node_x, node_y]),
        cell_nodes=cell_nodes,
        line_xy=np.empty((0, 2), dtype=np.float64),
        h_min=1e-4,
        timestep_s=10.0,
        gravity=9.81,
        sample_map=sample_map[0],
    )
    assert np.allclose(profile["station_m"], direct["station_m"])
    assert np.allclose(profile["depth_m"], direct["depth_m"])
