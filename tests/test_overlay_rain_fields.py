#!/usr/bin/env python3
"""Tests for rain/Manning/CN overlay field persistence and live-copy logic.

Test 4 — _copy_overlay_cell_data_from_coupling with mocked forcing + results_data
Test 5 — persist_all_baked_results + GPKG roundtrip for overlay_field_items
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure repo root and build dir are on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Test 4 — _copy_overlay_cell_data_from_coupling
# ---------------------------------------------------------------------------

def test_overlay_cell_data_live_copy():
    """_copy_overlay_cell_data_from_coupling populates results_data overlay arrays.

    Sets up a mock coupling controller (cc) with a mock rain forcing that has
    known cumulative_rain_mm and cumulative_excess_mm on its cn_model, then
    verifies the helper copies them into the results_data object correctly.
    """
    from swe2d.results.data import SWE2DResultsData
    from swe2d.workbench.services.non_gui_runtime_service import (
        _copy_overlay_cell_data_from_coupling,
    )

    # ── Mock SCSCurveNumberLoss with known cumulative values ────────────
    mock_loss = MagicMock()
    mock_loss.cumulative_rain_mm = np.array([0.0, 5.0, 10.0, 2.5], dtype=np.float64)
    mock_loss.cumulative_excess_mm = np.array([0.0, 2.0, 4.5, 1.0], dtype=np.float64)

    # ── Mock rain forcing attached to cc ─────────────────────────────────
    mock_forcing = MagicMock()
    mock_forcing.cn_model = mock_loss
    mock_forcing._loss_calculator = None
    # step_net_rainfall_mps returns (rate_mps, stats_dict)
    mock_forcing.step_net_rainfall_mps.return_value = (
        np.array([1e-5, 1e-5, 1e-5, 1e-5], dtype=np.float64),
        {},
    )

    # ── Mock coupling controller (cc) ───────────────────────────────────
    mock_cc = MagicMock()
    mock_cc._rain_forcing = mock_forcing
    mock_cc.drainage = None

    # ── Mock dsoa with link_roughness_n ─────────────────────────────────
    mock_dsoa = MagicMock()
    mock_dsoa.link_roughness_n = np.array([0.013, 0.015], dtype=np.float64)
    mock_cc._dsoa = mock_dsoa

    # ── SWE2DResultsData with empty overlay arrays ───────────────────────
    results_data = SWE2DResultsData()
    assert results_data.overlay_cell_cumulative_rain.size == 0
    assert results_data.overlay_cell_cumulative_excess.size == 0
    assert results_data.overlay_cell_mannings_n.size == 0

    # ── Call the helper ─────────────────────────────────────────────────
    _copy_overlay_cell_data_from_coupling(mock_cc, results_data, t_s=0.0)

    # ── Verify cumulative rain/excess were copied ───────────────────────
    np.testing.assert_array_equal(
        results_data.overlay_cell_cumulative_rain,
        np.array([0.0, 5.0, 10.0, 2.5], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        results_data.overlay_cell_cumulative_excess,
        np.array([0.0, 2.0, 4.5, 1.0], dtype=np.float64),
    )

    # ── Verify Manning's n was copied (from dsoa.link_roughness_n) ───────
    np.testing.assert_array_equal(
        results_data.overlay_cell_mannings_n,
        np.array([0.013, 0.015], dtype=np.float64),
    )

    print("test_overlay_cell_data_live_copy — PASSED")


# ---------------------------------------------------------------------------
# Test 5 — persist_all_baked_results + GPKG roundtrip
# ---------------------------------------------------------------------------

def test_overlay_gpkg_roundtrip():
    """persist_all_baked_results writes overlay_field_items; direct sqlite3 readback
    returns the same values.

    Verifies the 'cumulative_rain' metric roundtrips correctly through the
    swe2d_baked_overlay_fields table.
    """
    from swe2d.services.gpkg_persistence_service import persist_all_baked_results

    # ── Setup: temporary GPKG + minimal mesh snapshot ─────────────────────
    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
        gpkg_path = f.name

    try:
        run_id = "test_overlay_roundtrip_run"
        mesh_name = "test_overlay_mesh"
        n_cells = 4
        n_ts = 3

        # Minimal mesh snapshot (one wet cell)
        times = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        h = np.ones((n_ts, n_cells), dtype=np.float64)
        hu = np.zeros((n_ts, n_cells), dtype=np.float64)
        hv = np.zeros((n_ts, n_cells), dtype=np.float64)
        snapshots = []
        for i in range(n_ts):
            snapshots.append((float(i), h[i], hu[i], hv[i]))

        # ── Overlay field: cumulative_rain flattened row-major ─────────────────
        cumulative_rain_values = np.array([0.0, 5.0, 10.0, 2.5], dtype=np.float64)  # 4 cells
        # n_timesteps × n_cells = 3 × 4 = 12 elements
        overlay_values = np.repeat(cumulative_rain_values[np.newaxis, :], n_ts, axis=0).ravel()
        overlay_items = [
            {
                "metric": "cumulative_rain",
                "times": times,
                "values": overlay_values,
            }
        ]

        # ── Persist ────────────────────────────────────────────────────
        persist_all_baked_results(
            gpkg_path=gpkg_path,
            run_id=run_id,
            mesh_name=mesh_name,
            snapshot_timesteps=snapshots,
            overlay_field_items=overlay_items,
            log_fn=None,
        )

        # ── Read back via raw sqlite3 ──────────────────────────────────
        conn = sqlite3.connect(gpkg_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT metric, n_timesteps, times_blob, values_blob "
            "FROM swe2d_baked_overlay_fields WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        conn.close()

        assert row is not None, "No overlay field row found in GPKG"
        metric, n_timesteps, times_blob, values_blob = row
        assert metric == "cumulative_rain", f"Unexpected metric: {metric}"
        assert n_timesteps == 3, f"Unexpected n_timesteps: {n_timesteps}"

        read_times = np.frombuffer(times_blob, dtype=np.float64)
        read_values = np.frombuffer(values_blob, dtype=np.float64)

        np.testing.assert_array_equal(read_times, times)
        np.testing.assert_array_equal(read_values, overlay_values)

        print("test_overlay_gpkg_roundtrip — PASSED")

    finally:
        if os.path.exists(gpkg_path):
            os.unlink(gpkg_path)


if __name__ == "__main__":
    test_overlay_cell_data_live_copy()
    test_overlay_gpkg_roundtrip()
