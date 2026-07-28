import unittest
"""Round-trip tests for baked line time-series and profile persistence.

These tests verify that every variable written to the GeoPackage by the
runtime can be read back by the viewers. They intentionally exercise the
public persistence API without Qt.
"""

import numpy as np
import pytest

from swe2d.services.gpkg_persistence_service import (
    load_baked_line_profile,
    load_baked_line_timeseries,
    persist_baked_line_profile,
    persist_baked_line_ts,
)


@pytest.fixture
def run_id() -> str:
    return "run_roundtrip_001"


@pytest.fixture
def line_id() -> int:
    return 7


@pytest.fixture
def line_name() -> str:
    return "test_section"


def test_baked_line_timeseries_roundtrip(tmp_path, run_id, line_id, line_name):
    """All line time-series variables must round-trip through the GPKG."""
    gpkg_path = str(tmp_path / "line_results.gpkg")
    times = np.array([0.0, 1800.0, 3600.0], dtype=np.float64)
    depth = np.array([0.1, 0.25, 0.4], dtype=np.float64)
    velocity = np.array([0.5, 0.75, 1.0], dtype=np.float64)
    wse = np.array([10.1, 10.25, 10.4], dtype=np.float64)
    bed = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    flow = np.array([1.0, 2.5, 4.0], dtype=np.float64)
    wet_frac = np.array([1.0, 1.0, 0.5], dtype=np.float64)
    fr = np.array([0.05, 0.12, 0.20], dtype=np.float64)

    persist_baked_line_ts(
        gpkg_path, run_id, line_id, line_name, times,
        depth, velocity, wse, bed, flow, wet_frac, fr,
    )

    loaded = load_baked_line_timeseries(gpkg_path, run_id, line_id)

    assert set(loaded.keys()) == {
        "t_s", "depth", "velocity", "wse", "bed",
        "flow", "wet_frac", "fr",
    }
    np.testing.assert_array_equal(loaded["t_s"], times)
    np.testing.assert_array_equal(loaded["depth"], depth)
    np.testing.assert_array_equal(loaded["velocity"], velocity)
    np.testing.assert_array_equal(loaded["wse"], wse)
    np.testing.assert_array_equal(loaded["bed"], bed)
    np.testing.assert_array_equal(loaded["flow"], flow)
    np.testing.assert_array_equal(loaded["wet_frac"], wet_frac)
    np.testing.assert_array_equal(loaded["fr"], fr)


def test_baked_line_profile_roundtrip(tmp_path, run_id, line_id, line_name):
    """All line profile variables must round-trip through the GPKG.

    The profile viewer needs velocity_ms, flow_qn, fr, and wet in addition
    to the basic station/wse/bed/depth set.
    """
    gpkg_path = str(tmp_path / "line_results.gpkg")
    station = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
    times = np.array([0.0, 3600.0], dtype=np.float64)
    n_ts, n_sta = len(times), len(station)

    depth = np.tile([0.5, 1.0, 0.8, 0.2], (n_ts, 1)).astype(np.float64)
    velocity = np.tile([1.0, 2.0, 1.5, 0.5], (n_ts, 1)).astype(np.float64)
    wse = np.tile([10.5, 11.0, 10.8, 10.2], (n_ts, 1)).astype(np.float64)
    bed = np.tile([10.0, 10.0, 10.0, 10.0], (n_ts, 1)).astype(np.float64)
    flow_qn = np.tile([0.5, 2.0, 1.2, 0.1], (n_ts, 1)).astype(np.float64)
    fr = np.tile([0.45, 0.63, 0.53, 0.35], (n_ts, 1)).astype(np.float64)
    wet = np.tile([1, 1, 1, 0], (n_ts, 1)).astype(np.int32)

    persist_baked_line_profile(
        gpkg_path, run_id, line_id, line_name,
        station, times, depth, velocity, wse, bed, flow_qn, fr, wet,
    )

    loaded = load_baked_line_profile(gpkg_path, run_id, line_id, t_sec=0.0)

    expected_keys = {
        "station", "wse", "bed", "depth",
        "velocity", "flow_qn", "fr", "wet",
    }
    missing = expected_keys - set(loaded.keys())
    assert not missing, f"load_baked_line_profile is missing keys: {missing}"

    np.testing.assert_array_equal(loaded["station"], station)
    np.testing.assert_array_equal(loaded["depth"], depth[0])
    np.testing.assert_array_equal(loaded["velocity"], velocity[0])
    np.testing.assert_array_equal(loaded["wse"], wse[0])
    np.testing.assert_array_equal(loaded["bed"], bed[0])
    np.testing.assert_array_equal(loaded["flow_qn"], flow_qn[0])
    np.testing.assert_array_equal(loaded["fr"], fr[0])
    np.testing.assert_array_equal(loaded["wet"], wet[0])


def test_baked_line_profile_nearest_timestep_selection(tmp_path, run_id, line_id, line_name):
    """Loading a profile at an intermediate time must return the nearest snapshot."""
    gpkg_path = str(tmp_path / "line_results.gpkg")
    station = np.array([0.0, 10.0], dtype=np.float64)
    times = np.array([0.0, 3600.0], dtype=np.float64)
    depth = np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float64)
    velocity = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    wse = np.array([[10.1, 10.1], [11.0, 11.0]], dtype=np.float64)
    bed = np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float64)
    flow_qn = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    fr = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float64)
    wet = np.array([[1, 1], [1, 1]], dtype=np.int32)

    persist_baked_line_profile(
        gpkg_path, run_id, line_id, line_name,
        station, times, depth, velocity, wse, bed, flow_qn, fr, wet,
    )

    loaded = load_baked_line_profile(gpkg_path, run_id, line_id, t_sec=3500.0)
    np.testing.assert_array_equal(loaded["depth"], depth[1])
    np.testing.assert_array_equal(loaded["velocity"], velocity[1])

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
