"""Tests for line profile sampling.

The viewer's profile path depends on ``sample_line_metrics`` returning
per-station data when a pre-built ``sample_map`` is supplied.  Earlier code
aborted if ``line_xy`` was empty even when the sample map already contained
profile interpolation data.
"""

import numpy as np

from swe2d.workbench.services.mesh_service import sample_line_metrics


def test_sample_line_metrics_uses_prebuilt_profile_map_with_empty_line_xy():
    """A pre-built sample_map must be used even when line_xy is empty."""
    h = np.array([1.0], dtype=np.float64)
    hu = np.array([0.5], dtype=np.float64)
    hv = np.array([0.1], dtype=np.float64)
    bed = np.array([0.2], dtype=np.float64)
    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cell_nodes = np.array([[0, 1, 2]], dtype=np.int32)
    line_xy = np.empty((0, 2), dtype=np.float64)

    sample_map = {
        "cell_idx": np.array([0], dtype=np.int32),
        "weights": np.array([1.0], dtype=np.float64),
        "normal_x": 0.0,
        "normal_y": 1.0,
        "station_m": np.array([0.5], dtype=np.float64),
        "profile_station_m": np.array([0.25], dtype=np.float64),
        "profile_cell_idx": np.array([[0]], dtype=np.int32),
        "profile_cell_w": np.array([[1.0]], dtype=np.float64),
    }

    result = sample_line_metrics(
        h=h, hu=hu, hv=hv, bed=bed,
        node_coords=node_coords, cell_nodes=cell_nodes,
        line_xy=line_xy, h_min=1.0e-6, timestep_s=0.0,
        gravity=9.81, sample_map=sample_map,
    )

    assert result["station_m"].size == 1, "expected one profile station"
    np.testing.assert_allclose(result["station_m"], [0.25])
    np.testing.assert_allclose(result["depth_m"], [1.0])
    np.testing.assert_allclose(result["bed_m"], [0.2])
    np.testing.assert_allclose(result["velocity_ms"], [np.hypot(0.5, 0.1)])
    np.testing.assert_allclose(result["wet"], [1])


def test_sample_line_metrics_empty_line_xy_without_map_returns_empty():
    """Without a sample_map, empty line_xy must still return empty."""
    h = np.array([1.0], dtype=np.float64)
    hu = np.array([0.5], dtype=np.float64)
    hv = np.array([0.1], dtype=np.float64)
    bed = np.array([0.2], dtype=np.float64)
    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cell_nodes = np.array([[0, 1, 2]], dtype=np.int32)
    line_xy = np.empty((0, 2), dtype=np.float64)

    result = sample_line_metrics(
        h=h, hu=hu, hv=hv, bed=bed,
        node_coords=node_coords, cell_nodes=cell_nodes,
        line_xy=line_xy, h_min=1.0e-6, timestep_s=0.0,
        gravity=9.81, sample_map=None,
    )

    assert result["station_m"].size == 0
