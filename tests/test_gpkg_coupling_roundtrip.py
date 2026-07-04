"""Round-trip tests for baked structure/drainage coupling persistence.

The viewers read coupling records as rows with ``t_s`` and ``value`` keys.
These tests verify that loading persisted coupling data produces rows the
viewers can actually plot.
"""

import numpy as np
import pytest

from swe2d.results.data import SWE2DResultsData
from swe2d.results.run_service import RunRecord
from swe2d.services.gpkg_persistence_service import persist_baked_coupling


@pytest.fixture
def run_id() -> str:
    return "run_coupling_001"


@pytest.fixture
def component() -> str:
    return "structure"


@pytest.fixture
def object_id() -> str:
    return "culvert_1"


@pytest.fixture
def metric() -> str:
    return "flow"


def test_load_coupling_records_expands_baked_timeseries(
    tmp_path, run_id, component, object_id, metric,
):
    """load_coupling_records must return rows with t_s/value for GPKG runs."""
    gpkg_path = str(tmp_path / "coupling_results.gpkg")
    times = np.array([0.0, 1800.0, 3600.0], dtype=np.float64)
    values = np.array([10.0, 25.0, 40.0], dtype=np.float64)

    persist_baked_coupling(
        gpkg_path, run_id, component, object_id,
        "Culvert 1", metric, times, values,
    )

    data = SWE2DResultsData()
    data._run_records = [
        RunRecord(
            run_id=run_id,
            gpkg_path=gpkg_path,
            color=(31, 119, 180),
            enabled=True,
            label="test run",
        )
    ]

    data.load_coupling_records(run_id)
    records = data.get_coupling_records()

    assert len(records) == 3, f"expected 3 rows, got {len(records)}"
    for r in records:
        assert "t_s" in r, "coupling record missing t_s"
        assert "value" in r, "coupling record missing value"
        assert r["component"] == component
        assert r["object_id"] == object_id
        assert r["metric"] == metric

    np.testing.assert_array_equal(
        np.array([r["t_s"] for r in records], dtype=np.float64), times
    )
    np.testing.assert_array_equal(
        np.array([r["value"] for r in records], dtype=np.float64), values
    )


def test_load_coupling_records_filters_by_run_id(
    tmp_path, run_id, component, object_id, metric,
):
    """Loading one run_id must not return records from another run_id."""
    gpkg_path = str(tmp_path / "coupling_results.gpkg")
    persist_baked_coupling(
        gpkg_path, run_id, component, object_id,
        "Culvert 1", metric,
        np.array([0.0, 3600.0], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
    )
    persist_baked_coupling(
        gpkg_path, "other_run", component, object_id,
        "Culvert 1", metric,
        np.array([0.0, 3600.0], dtype=np.float64),
        np.array([99.0, 99.0], dtype=np.float64),
    )

    data = SWE2DResultsData()
    data._run_records = [
        RunRecord(
            run_id=run_id,
            gpkg_path=gpkg_path,
            color=(31, 119, 180),
            enabled=True,
        )
    ]

    data.load_coupling_records(run_id)
    records = data.get_coupling_records()

    assert len(records) == 2
    assert all(r["run_id"] == run_id for r in records) if "run_id" in records[0] else True
    np.testing.assert_array_equal(
        np.array([r["value"] for r in records], dtype=np.float64),
        np.array([1.0, 2.0], dtype=np.float64),
    )
