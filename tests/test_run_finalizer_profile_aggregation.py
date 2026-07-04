"""Tests for run_finalizer profile aggregation.

The finalizer must persist line profiles even when the sample callback returns
long-format rows (one dict per station point) and the first station is at 0.0 m.
Earlier code treated a 0-d station array with value 0.0 as falsy and skipped the
profile silently.
"""

import os
import sys

import numpy as np
import pytest

# Ensure repo root is on path for direct test execution
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from swe2d.runtime.run_finalizer import SWE2DRunFinalizer


class _MockView:
    """Minimal view protocol implementation for finalizer tests."""

    def __init__(self, tmp_path, results_data=None):
        self._log: list = []
        self._gpkg_path = str(tmp_path / "line_results.gpkg")
        self._results_data = results_data

    def log_message(self, msg: str) -> None:
        self._log.append(msg)

    def get_line_results_storage_path(self) -> str:
        return self._gpkg_path

    def sync_overlay_data(self) -> None:
        pass

    def refresh_plot(self) -> None:
        pass

    def results_table_name(self) -> str:
        return "swe2d_baked_results"

    def length_unit_name(self) -> str:
        return "m"

    def length_scale_si_to_model(self) -> float:
        return 1.0

    def results_data(self):
        return self._results_data

    def update_overlay_time(self, t: float) -> None:
        pass

    def runtime_log_lines(self) -> list:
        return self._log

    def collect_run_log_metadata(self) -> dict:
        return {}

    def persist_run_log(self, *args, **kwargs) -> None:
        pass

    def is_cancel_requested(self) -> bool:
        return False


def _wide_format_callback(_sample_map, t_s, _h, _hu, _hv, _bed):
    """Return wide-format profile rows (one row per line/timestep with arrays)."""
    ts_rows = [
        {
            "t_s": float(t_s),
            "line_id": 1,
            "line_name": "section_1",
            "depth_m": 1.0,
            "velocity_ms": 0.5,
            "wse_m": 11.0,
            "bed_m": 10.0,
            "flow_cms": 2.0,
            "wet_frac": 1.0,
            "fr": 0.1,
        }
    ]
    prof_rows = [
        {
            "t_s": float(t_s),
            "line_id": 1,
            "line_name": "section_1",
            "station_m": np.array([0.0, 10.0, 20.0], dtype=np.float64),
            "depth_m": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            "velocity_ms": np.array([0.5, 0.5, 0.5], dtype=np.float64),
            "wse_m": np.array([11.0, 11.0, 11.0], dtype=np.float64),
            "bed_m": np.array([10.0, 10.0, 10.0], dtype=np.float64),
            "flow_qn": np.array([0.5, 0.5, 0.5], dtype=np.float64),
            "fr": np.array([0.1, 0.1, 0.1], dtype=np.float64),
            "wet": np.array([1, 1, 1], dtype=np.int32),
        }
    ]
    return ts_rows, prof_rows


def test_finalizer_persists_line_profile_with_station_starting_at_zero(tmp_path):
    """Wide-format profile rows with a station at 0.0 must create the profile table."""
    view = _MockView(tmp_path)
    finalizer = SWE2DRunFinalizer(view)

    n_cells = 4
    h = np.full(n_cells, 1.0, dtype=np.float64)
    hu = np.full(n_cells, 0.5, dtype=np.float64)
    hv = np.full(n_cells, 0.0, dtype=np.float64)
    area = np.full(n_cells, 1.0, dtype=np.float64)

    snapshot_timesteps = [
        (0.0, h.copy(), hu.copy(), hv.copy()),
        (3600.0, h.copy(), hu.copy(), hv.copy()),
    ]

    status = finalizer.finalize_and_persist(
        h=h,
        hu=hu,
        hv=hv,
        final_sim_time_s=3600.0,
        n_area=n_cells,
        area_model=area,
        storage_start_model=0.0,
        source_budget_model={"rain": 0.0, "cell": 0.0, "coupling": 0.0},
        source_step_rows_model=[],
        run_duration_s=3600.0,
        boundary_flux_budget_model={},
        boundary_flux_step_rows_model=[],
        run_id="run_profile_long_fmt",
        output_interval_s=3600.0,
        line_output_interval_s=3600.0,
        run_perf_start=0.0,
        run_wallclock_start="",
        run_log_start_idx=0,
        thiessen_forcing=None,
        rain_stats_acc={"samples": 0, "rain_mm": 0.0, "excess_mm": 0.0},
        save_line_results=True,
        save_mesh_results=False,
        save_coupling_results=False,
        save_run_log=False,
        sample_map=[{"line_id": 1, "line_name": "section_1"}],
        cell_solver_z=np.full(n_cells, 10.0, dtype=np.float64),
        sample_line_metrics_callback=_wide_format_callback,
        snapshot_timesteps=snapshot_timesteps,
    )

    assert status["ok"], f"finalizer failed: {status['errors']}"
    assert os.path.exists(view._gpkg_path)

    import sqlite3

    conn = sqlite3.connect(view._gpkg_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='swe2d_baked_line_profiles'"
        ).fetchone()
        assert row is not None, "swe2d_baked_line_profiles table was not created"
    finally:
        conn.close()
