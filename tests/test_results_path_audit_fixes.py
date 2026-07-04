"""Tests for issues found in the results-path audit (first pass).

Each test targets one finding from docs/AGENT_SESSION_RECOVERY_LOG.md or the
inline audit.  Pure-Python where possible; no QGIS app required for most.
"""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestGpkgExplorerDialogImport(unittest.TestCase):
    """G1.1: gpkg_explorer_dialog.py had a SyntaxError on the dead
    `conservation` branch (_table_kind).  The module must import cleanly."""

    def test_module_imports_without_syntax_error(self):
        from swe2d.workbench.dialogs import gpkg_explorer_dialog
        self.assertTrue(hasattr(gpkg_explorer_dialog, "__file__"))

    def test_table_kind_classifies_known_tables(self):
        # Defer Qt import until test runs so the test file itself is importable
        # without a display.
        from swe2d.workbench.dialogs.gpkg_explorer_dialog import (
            SWE2DModelGeoPackageExplorerDialog as Dialog,
        )
        # _table_kind is a pure string classifier — call it on an *instance*
        # is overkill; it doesn't touch Qt at classification time, but it is
        # a method so we exercise it via an unbound call on a fake self.
        self.assertEqual(Dialog._table_kind(None, "swe2d_baked_results"), "mesh_results")
        self.assertEqual(Dialog._table_kind(None, "swe2d_baked_mesh"), "mesh_results")
        self.assertEqual(Dialog._table_kind(None, "swe2d_baked_line_ts"), "line_results")
        self.assertEqual(Dialog._table_kind(None, "swe2d_baked_line_profiles"), "line_results")
        self.assertEqual(Dialog._table_kind(None, "swe2d_baked_coupling"), "coupling_results")
        self.assertEqual(Dialog._table_kind(None, "swe2d_run_logs"), "run_log")
        # Unknown tables fall through to the default branch — not "conservation".
        self.assertEqual(Dialog._table_kind(None, "some_random_table"), "table")


class TestNullBlobGuardInLineTimeseries(unittest.TestCase):
    """G3.10: load_baked_line_timeseries crashed on NULL BLOBs.
    The fix must guard each np.frombuffer call."""

    def test_load_line_timeseries_with_null_blobs_returns_empty(self):
        from swe2d.services.gpkg_persistence_service import (
            persist_baked_line_ts,
            load_baked_line_timeseries,
            persist_baked_mesh,
            persist_baked_results,
        )
        import sqlite3

        tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        tmp.close()
        gpkg = tmp.name
        try:
            # Persist mesh + results so the schema is in place
            persist_baked_mesh(gpkg, "m", b"\x00" * 8, n_nodes=4, n_cells=4, n_edges=4)
            times = np.array([0.0], dtype=np.float64)
            h = np.zeros(4, dtype=np.float64)
            persist_baked_results(
                gpkg, "rid", "m",
                snapshot_timesteps=[(0.0, h, h, h)],
            )
            # Persist then delete a real line_ts row to create the table
            persist_baked_line_ts(
                gpkg, "rid", 99, "delete_me", times,
                depth_m=h, velocity_ms=h, wse_m=h, bed_m=h,
                flow_cms=h, wet_frac=h, fr=h,
            )
            import sqlite3 as _sql
            _c = _sql.connect(gpkg)
            _c.execute("DELETE FROM swe2d_baked_line_ts WHERE line_id=99")
            _c.commit()
            _c.close()
            # Manually insert a line_ts row with NULL blobs
            conn = sqlite3.connect(gpkg)
            conn.execute(
                "INSERT INTO swe2d_baked_line_ts "
                "(run_id, line_id, line_name, n_timesteps, times_blob, depth_blob, "
                "vel_blob, wse_blob, bed_blob, flow_blob, wet_frac_blob, fr_blob) "
                "VALUES (?,1,'L',1,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)",
                ("rid",),
            )
            conn.commit()
            conn.close()

            # Live source path with empty data — emulate a results-data stub
            class _Stub:
                _live_line_ts = {}
                _live_times = np.array([], dtype=np.float64)

            out = load_baked_line_timeseries(gpkg, "rid", 1)
            # Must not crash; the depth/time arrays must come back empty
            self.assertEqual(len(out.get("t_s", [])), 0)
        finally:
            os.remove(gpkg)


class TestMaxTrackingFallbackLogs(unittest.TestCase):
    """G3.9: compute_max_tracking silently fell back to snapshot max when the
    max_h_blob column was NULL.  The fallback must now emit a logger.warning so
    the underestimation is visible."""

    def test_max_tracking_null_warns_and_falls_back(self):
        from swe2d.services.gpkg_persistence_service import (
            persist_baked_results,
            compute_max_tracking,
        )
        import logging

        tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        tmp.close()
        gpkg = tmp.name
        try:
            h0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            h1 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
            persist_baked_results(
                gpkg, "rid", "m",
                snapshot_timesteps=[(0.0, h0, h0, h0), (1.0, h1, h1, h1)],
            )
            with self.assertLogs(
                "swe2d.services.gpkg_persistence_service", level="WARNING"
            ) as cm:
                result = compute_max_tracking(gpkg, "rid")
            max_h = result["max_h"]
            # Snapshot-max fallback returns the per-cell max across the snapshot axis
            self.assertEqual(max_h.shape, (4,))
            self.assertTrue(any("max_h_blob" in m or "fall" in m.lower() for m in cm.output))
        finally:
            os.remove(gpkg)


class TestOrphanCleanupOnDeleteRun(unittest.TestCase):
    """G3.8: delete_run left orphan rows in swe2d_baked_* tables.
    After the fix all baked tables must be cleared for the deleted run_id."""

    def test_delete_run_removes_baked_rows(self):
        from swe2d.services.gpkg_persistence_service import (
            persist_baked_results,
            persist_baked_line_ts,
            persist_baked_line_profile,
            persist_baked_coupling,
        )
        from swe2d.workbench.services.gpkg_operations_service import delete_run
        import sqlite3

        tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        tmp.close()
        gpkg = tmp.name
        try:
            times = np.array([0.0, 1.0], dtype=np.float64)
            h = np.zeros(4, dtype=np.float64)
            persist_baked_results(
                gpkg, "rid", "m",
                snapshot_timesteps=[(0.0, h, h, h), (1.0, h, h, h)],
            )
            persist_baked_line_ts(
                gpkg, "rid", 1, "L", times,
                depth_m=h, velocity_ms=h, wse_m=h, bed_m=h,
                flow_cms=h, wet_frac=h, fr=h,
            )
            prof = np.zeros((2, 2), dtype=np.float64)
            wet = np.zeros((2, 2), dtype=np.int32)
            persist_baked_line_profile(
                gpkg, "rid", 1, "L",
                station_m=np.array([0.0, 1.0]),
                times=times,
                depth_m=prof, velocity_ms=prof, wse_m=prof, bed_m=prof,
                flow_qn=prof, fr=prof, wet=wet,
            )
            persist_baked_coupling(
                gpkg, "rid",
                component="structure", object_id="sid",
                object_name="S", metric="flow",
                times=times, values=h,
            )

            delete_run(gpkg, "rid")

            conn = sqlite3.connect(gpkg)
            for tbl in (
                "swe2d_baked_results",
                "swe2d_baked_line_ts",
                "swe2d_baked_line_profiles",
                "swe2d_baked_coupling",
            ):
                cur = conn.execute(f"SELECT COUNT(*) FROM {tbl} WHERE run_id=?", ("rid",))
                self.assertEqual(cur.fetchone()[0], 0, f"orphan rows left in {tbl}")
            conn.close()
        finally:
            os.remove(gpkg)


class TestLiveLineProfilePopulation(unittest.TestCase):
    """G2.5/G2.6: live line TS/profile population was broken end-to-end.
    The reporter never sampled line metrics during runs, the live fallback
    in load_baked_line_profile assumed ragged 1D lists instead of 2D arrays,
    and preallocate_line_profile_nstations (referenced in docstrings) didn't
    exist.  After the fix the data layer must expose a
    populate_live_line_metrics() entry point and load_baked_line_profile must
    read from the structured 2D storage."""

    def test_populate_live_line_metrics_then_load_profile(self):
        from swe2d.results.data import SWE2DResultsData

        data = SWE2DResultsData()
        # Two snapshots at t=0 and t=1
        h0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        h1 = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        zeros = np.zeros_like(h0)
        data.set_live_snapshot_timesteps([
            (0.0, h0, zeros, zeros),
            (1.0, h1, zeros, zeros),
        ])
        # Sample map: one line with two stations at cells 0 and 2
        sample_map = [{
            "line_id": 7,
            "line_name": "L7",
            "cell_idx": np.array([0, 2], dtype=np.int32),
            "station_m": np.array([0.0, 10.0], dtype=np.float64),
        }]

        def fake_callback(sm, t, h, hu, hv, cell_bed):
            ts_rows, prof_rows = [], []
            for m in sm:
                lid = int(m["line_id"])
                idx = np.asarray(m["cell_idx"], dtype=np.int32)
                hh = h[idx]
                zb = cell_bed[idx] if cell_bed is not None else np.zeros_like(hh)
                ts_rows.append({
                    "line_id": lid, "line_name": m["line_name"],
                    "depth_m": float(np.mean(hh)),
                    "velocity_ms": 0.0, "wse_m": float(np.mean(hh + zb)),
                    "bed_m": float(np.mean(zb)), "flow_cms": 0.0,
                    "wet_frac": 1.0, "fr": 0.0,
                })
                prof_rows.append({
                    "line_id": lid, "line_name": m["line_name"],
                    "station_m": np.asarray(m["station_m"], dtype=np.float64),
                    "depth_m": hh.astype(np.float64),
                    "velocity_ms": np.zeros_like(hh, dtype=np.float64),
                    "wse_m": (hh + zb).astype(np.float64),
                    "bed_m": zb.astype(np.float64),
                    "flow_qn": np.zeros_like(hh, dtype=np.float64),
                    "wet": np.ones_like(hh, dtype=np.int32),
                    "fr": np.zeros_like(hh, dtype=np.float64),
                })
            return ts_rows, prof_rows

        # Pre-fix this returned without populating _live_line_profile.
        data.populate_live_line_metrics(
            sample_map=sample_map,
            sample_callback=fake_callback,
            cell_solver_z=np.array([0.0, 0.0, 0.0]),
        )

        # The structured storage must now contain line 7
        self.assertIn(7, data._live_line_profile)
        # Profile arrays must be 2D (n_snaps × n_stations)
        depth = data._live_line_profile[7]["depth_m"]
        self.assertEqual(depth.shape, (2, 2))
        # Load at t=1 should return the second-snapshot values
        data._live_run_id = "rid"
        from swe2d.services.gpkg_persistence_service import load_baked_line_profile
        out = load_baked_line_profile(data, "rid", 7, 1.0)
        self.assertIn("station_m", out)
        np.testing.assert_allclose(out["depth_m"], [1.5, 3.5])
        # And at t=0 the first-snapshot values
        out0 = load_baked_line_profile(data, "rid", 7, 0.0)
        np.testing.assert_allclose(out0["depth_m"], [1.0, 3.0])


class TestResultsDataLineTsLivePath(unittest.TestCase):
    """Companion to the above — load_baked_line_timeseries live path must
    consume the same structured storage."""

    def test_load_line_timeseries_uses_populated_arrays(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.services.gpkg_persistence_service import load_baked_line_timeseries

        data = SWE2DResultsData()
        h0 = np.array([1.0, 2.0], dtype=np.float64)
        h1 = np.array([2.0, 3.0], dtype=np.float64)
        zeros = np.zeros_like(h0)
        data.set_live_snapshot_timesteps([
            (0.0, h0, zeros, zeros),
            (10.0, h1, zeros, zeros),
        ])
        sample_map = [{
            "line_id": 1, "line_name": "L1",
            "cell_idx": np.array([0, 1], dtype=np.int32),
            "station_m": np.array([0.0, 5.0], dtype=np.float64),
        }]

        def fake_cb(sm, t, h, hu, hv, cell_bed):
            ts_rows, prof_rows = [], []
            for m in sm:
                idx = np.asarray(m["cell_idx"], dtype=np.int32)
                hh = h[idx]
                ts_rows.append({
                    "line_id": int(m["line_id"]), "line_name": m["line_name"],
                    "depth_m": float(np.mean(hh)), "velocity_ms": 0.0,
                    "wse_m": float(np.mean(hh)), "bed_m": 0.0,
                    "flow_cms": 0.0, "wet_frac": 1.0, "fr": 0.0,
                })
                prof_rows.append({
                    "line_id": int(m["line_id"]), "line_name": m["line_name"],
                    "station_m": np.asarray(m["station_m"], dtype=np.float64),
                    "depth_m": hh.astype(np.float64),
                    "velocity_ms": np.zeros_like(hh, dtype=np.float64),
                    "wse_m": hh.astype(np.float64),
                    "bed_m": np.zeros_like(hh, dtype=np.float64),
                    "flow_qn": np.zeros_like(hh, dtype=np.float64),
                    "wet": np.ones_like(hh, dtype=np.int32),
                    "fr": np.zeros_like(hh, dtype=np.float64),
                })
            return ts_rows, prof_rows

        data.populate_live_line_metrics(
            sample_map=sample_map, sample_callback=fake_cb,
            cell_solver_z=np.array([0.0, 0.0]),
        )

        data._live_run_id = "rid"
        ts = load_baked_line_timeseries(data, "rid", 1)
        # t_s must reflect the snapshot times (0 and 10), not be empty
        np.testing.assert_allclose(np.sort(ts["t_s"]), [0.0, 10.0])
        # Depth averages: snapshot 0 mean=1.5, snapshot 1 mean=2.5
        np.testing.assert_allclose(np.sort(ts["depth_m"]), [1.5, 2.5])


class TestStructureFlowsLivePath(unittest.TestCase):
    """G2.4: load_structure_flows_at_time only read from GPKG.  The fix adds
    a live-data path so structure annotations appear during live runs."""

    def test_live_data_source_returns_structure_rows(self):
        from swe2d.results.queries import load_structure_flows_at_time

        class _Live:
            """Stand-in for SWE2DResultsData with live coupling data."""
            def get_structure_flows_at_time(self, run_id, t_sec):
                if run_id == "rid":
                    return [
                        {"object_id": "S1", "object_name": "Weir 1",
                         "value": 12.34, "t_s": float(t_sec)},
                    ]
                return []

        rows = load_structure_flows_at_time(_Live(), "rid", 100.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["object_id"], "S1")
        self.assertAlmostEqual(rows[0]["value"], 12.34)


class TestPersistenceErrorsPropagate(unittest.TestCase):
    """G3.7: SWE2DRunFinalizer swallowed persistence errors as log warnings.
    The finalizer must report failures back to its caller."""

    def test_finalize_and_persist_reports_persistence_failures(self):
        from swe2d.runtime.run_finalizer import SWE2DRunFinalizer
        import numpy as np

        class _BadView:
            _results_data = None
            _messages = []

            def log_message(self, msg):
                self._messages.append(str(msg))

            def get_line_results_storage_path(self):
                # Force a persistence failure in the mesh-results path
                return "/nonexistent/path.gpkg"

            def sync_overlay_data(self):
                pass

            def refresh_plot(self):
                pass

            def results_table_name(self):
                return "swe2d_baked_results"

            def length_unit_name(self):
                return "m"

            def length_scale_si_to_model(self):
                return 1.0

            def results_data(self):
                return self._results_data

            def update_overlay_time(self, t):
                pass

            def runtime_log_lines(self):
                return []

            def collect_run_log_metadata(self):
                return {}

            def persist_run_log(self, *a, **k):
                pass

            def is_cancel_requested(self):
                return False

        h = np.zeros(4, dtype=np.float64)
        fin = SWE2DRunFinalizer(_BadView())
        result = fin.finalize_and_persist(
            h=h, hu=h, hv=h,
            final_sim_time_s=0.0,
            n_area=4,
            area_model=np.ones(4, dtype=np.float64),
            storage_start_model=0.0,
            source_budget_model={"rain": 0.0, "cell": 0.0, "coupling": 0.0},
            source_step_rows_model=[],
            run_duration_s=1.0,
            boundary_flux_budget_model={},
            boundary_flux_step_rows_model=[],
            run_id="rid",
            output_interval_s=1.0,
            line_output_interval_s=1.0,
            run_perf_start=time.perf_counter(),
            run_wallclock_start="now",
            run_log_start_idx=0,
            thiessen_forcing=None,
            rain_stats_acc={"samples": 0, "rain_mm": 0.0, "excess_mm": 0.0},
            save_mesh_results=True,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("ok", result)
        self.assertFalse(result["ok"])
        self.assertTrue(len(result.get("errors", [])) > 0)


if __name__ == "__main__":
    unittest.main()
