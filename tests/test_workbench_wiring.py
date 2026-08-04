#!/usr/bin/env python3
"""Behavioral wiring tests for the workbench run-seam startup controllers.

Covers (plan ``docs/plans/2026-08-02-gui-test-coverage.md`` Task E.1,
spec ``docs/specs/2026-08-02-gui-test-coverage-design.md`` §3-§4, pattern P2):

- ``swe2d/workbench/controllers/run_component_wiring_controller.py``
  (``wire_startup_run_components``)
- ``swe2d/workbench/controllers/startup_bootstrap_controller.py``
  (``bootstrap_startup_run_components``)
- ``swe2d/workbench/controllers/finalization_adapter.py``
  (``FinalizationAdapter``)

All tests drive a real ``SWE2DWorkbenchStudioDialog`` against the real
headless QGIS harness.  No MagicMock Qgs* substitutes; plain-data dialog
attributes (``_cancel_requested``, ``_length_unit_name``) are set directly
per the harness contract.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest

import numpy as np

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    make_temp_results_gpkg,
    requires_qgis,
    stub_iface,
)

from swe2d.runtime import (
    SWE2DBackendInitializer,
    SWE2DRunController,
    SWE2DRunFinalizer,
    SWE2DRunLifecycle,
    SWE2DRunOrchestrator,
    SWE2DRunRequest,
)
from swe2d.workbench.controllers.finalization_adapter import FinalizationAdapter
from swe2d.workbench.controllers.run_component_wiring_controller import (
    wire_startup_run_components,
)
from swe2d.workbench.controllers.startup_bootstrap_controller import (
    bootstrap_startup_run_components,
)


def _make_dialog():
    """Construct a real Studio dialog (runs the full bootstrap path)."""
    from qgis.PyQt import QtWidgets
    from swe2d.workbench.studio_dialog import SWE2DWorkbenchStudioDialog

    iface = stub_iface()
    # Dock widgets need a real QWidget parent.
    iface.mainWindow.return_value = QtWidgets.QMainWindow()
    return SWE2DWorkbenchStudioDialog(iface=iface)


class _RecordingBackend:
    """Plain test double for the native backend (destroy() call tracking)."""

    def __init__(self) -> None:
        self.destroy_calls = 0

    def destroy(self) -> None:
        self.destroy_calls += 1


class _FailingBackend:
    """Plain test double whose destroy() raises (lifecycle error path)."""

    def destroy(self) -> None:
        raise RuntimeError("boom")


@requires_qgis
class TestWireStartupRunComponents(unittest.TestCase):
    """wire_startup_run_components: seam components built and live-wired."""

    @classmethod
    def setUpClass(cls):
        cls._app = ensure_qgis_app()

    def setUp(self):
        self.dlg = _make_dialog()

    def tearDown(self):
        self.dlg.close()
        delete_widgets_now(self.dlg)

    # ── construction ────────────────────────────────────────────────────

    def test_components_constructed_with_documented_types(self):
        """The dialog bootstrap wires all five run-seam components."""
        dlg = self.dlg
        self.assertIsInstance(dlg._run_orchestrator, SWE2DRunOrchestrator)
        self.assertIsInstance(dlg._run_controller, SWE2DRunController)
        self.assertIsInstance(dlg._backend_initializer, SWE2DBackendInitializer)
        self.assertIsInstance(dlg._run_finalizer, SWE2DRunFinalizer)
        self.assertIsInstance(dlg._run_lifecycle, SWE2DRunLifecycle)

    def test_documented_connections_are_bound_to_the_dialog(self):
        """Every callback the wiring function documents points at the dialog."""
        dlg = self.dlg
        # Orchestrator: execute + log callbacks.
        self.assertIs(dlg._run_orchestrator._execute_callback.__self__, dlg)
        self.assertIs(dlg._run_orchestrator._log_callback.__self__, dlg)
        # Run controller: preflight callbacks.
        self.assertIs(dlg._run_controller._ensure_mesh_callback.__self__, dlg)
        self.assertIs(dlg._run_controller._has_mesh_callback.__self__, dlg)
        self.assertIs(dlg._run_controller._backend_ready_callback.__self__, dlg)
        self.assertIs(
            dlg._run_controller._backend_unavailable_callback.__self__, dlg
        )
        self.assertIs(dlg._run_controller._log_callback.__self__, dlg)
        # Backend initializer: BC-processing callbacks.
        self.assertIs(
            dlg._backend_initializer._apply_timeseries_bc_values.__self__, dlg
        )
        self.assertIs(
            dlg._backend_initializer._distribute_total_flow_to_unit_q.__self__,
            dlg,
        )
        # Finalizer holds a FinalizationAdapter bridging to the dialog.
        self.assertIsInstance(dlg._run_finalizer._view, FinalizationAdapter)
        self.assertIs(dlg._run_finalizer._view._dialog, dlg)
        # Lifecycle holds the dialog as its view-protocol handle.
        self.assertIs(dlg._run_lifecycle._ui, dlg)

    # ── orchestrator behavior through the wired seam ────────────────────

    def test_orchestrator_run_dispatches_through_wired_callback(self):
        """run(request) reaches dialog._dispatch_run_request behaviorally."""
        dlg = self.dlg
        request = SWE2DRunRequest.from_ui_values(
            run_duration_text="1:00",
            output_interval_text="0:05",
            adaptive_dt_enabled=False,
            requested_dt=0.5,
        )
        n0 = len(dlg._runtime_log_lines)
        accepted = dlg._run_orchestrator.run(request)
        self.assertTrue(accepted)
        # _dispatch_run_request stores the request before delegating.
        self.assertIs(dlg._last_run_request, request)
        # The dispatch chain reached RunController.on_run, which aborts
        # loudly without a mesh.
        self.assertIn(
            "Run aborted: mesh not available after preflight.",
            dlg._runtime_log_lines[n0:],
        )

    def test_orchestrator_reentrant_run_logs_via_wired_log_callback(self):
        """A concurrent run request is rejected through dialog._log."""
        dlg = self.dlg
        request = SWE2DRunRequest.from_ui_values("0:10", "0:01", False, 0.1)
        dlg._run_orchestrator._run_active = True
        try:
            n0 = len(dlg._runtime_log_lines)
            accepted = dlg._run_orchestrator.run(request)
        finally:
            dlg._run_orchestrator._run_active = False
        self.assertFalse(accepted)
        self.assertIn(
            "Run request ignored: another run is already active.",
            dlg._runtime_log_lines[n0:],
        )

    # ── run controller preflight through the wired seam ─────────────────

    def test_run_controller_preflight_aborts_loudly_without_mesh(self):
        """run_preflight() uses the wired ensure/has-mesh + log callbacks."""
        dlg = self.dlg
        n0 = len(dlg._runtime_log_lines)
        ok = dlg._run_controller.run_preflight()
        self.assertFalse(ok)
        new_lines = dlg._runtime_log_lines[n0:]
        self.assertIn(
            "Run aborted: no mesh loaded. Import mesh from map layers first.",
            new_lines,
        )
        self.assertIn("Run preflight aborted: mesh is not available.", new_lines)

    # ── lifecycle cleanup against real widgets ──────────────────────────

    def test_lifecycle_finalize_cleanup_restores_real_button_states(self):
        """finalize_cleanup destroys the backend and re-enables Run."""
        dlg = self.dlg
        dlg.set_run_button_enabled(False)
        dlg.set_cancel_button_enabled(True)
        self.assertFalse(dlg._run_dock.run_btn.isEnabled())
        self.assertTrue(dlg._run_dock.cancel_btn.isEnabled())

        backend = _RecordingBackend()
        dlg._run_lifecycle.finalize_cleanup(backend)

        self.assertEqual(backend.destroy_calls, 1)
        self.assertTrue(dlg._run_dock.run_btn.isEnabled())
        self.assertFalse(dlg._run_dock.cancel_btn.isEnabled())

    def test_lifecycle_finalize_cleanup_tolerates_backend_destroy_failure(self):
        """A failing backend.destroy() is logged; button cleanup still runs."""
        dlg = self.dlg
        dlg.set_run_button_enabled(False)
        n0 = len(dlg._runtime_log_lines)
        dlg._run_lifecycle.finalize_cleanup(_FailingBackend())
        self.assertTrue(dlg._run_dock.run_btn.isEnabled())
        self.assertFalse(dlg._run_dock.cancel_btn.isEnabled())
        self.assertIn(
            "[BACKEND] Backend destroy() failed: boom",
            dlg._runtime_log_lines[n0:],
        )

    # ── real button click through the wired run dock ────────────────────

    def test_run_button_click_invokes_wired_controller_once(self):
        """Clicking the real Run button fires the wired on_run exactly once."""
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtTest import QSignalSpy, QTest

        dlg = self.dlg
        spy = QSignalSpy(dlg._run_dock.run_btn.clicked)
        n0 = len(dlg._runtime_log_lines)
        QTest.mouseClick(dlg._run_dock.run_btn, Qt.LeftButton)
        self.assertEqual(len(spy), 1)
        new_lines = dlg._runtime_log_lines[n0:]
        self.assertEqual(
            new_lines.count("Run aborted: mesh not available after preflight."),
            1,
        )


@requires_qgis
class TestBootstrapStartupRunComponents(unittest.TestCase):
    """bootstrap_startup_run_components: namespace handoff + idempotency."""

    @classmethod
    def setUpClass(cls):
        cls._app = ensure_qgis_app()

    def setUp(self):
        self.dlg = _make_dialog()

    def tearDown(self):
        self.dlg.close()
        delete_widgets_now(self.dlg)

    def _call_bootstrap(self, wire_fn):
        bootstrap_startup_run_components(
            self.dlg,
            wire_fn,
            run_orchestrator=SWE2DRunOrchestrator,
            run_request=SWE2DRunRequest,
            run_controller=SWE2DRunController,
            backend_initializer=SWE2DBackendInitializer,
            run_finalizer=SWE2DRunFinalizer,
            run_lifecycle=SWE2DRunLifecycle,
        )

    def test_bootstrap_passes_full_namespace_to_wire_fn(self):
        """wire_fn receives the dialog plus all six documented seam classes."""
        calls = []

        def recording_wire_fn(dialog, ns):
            calls.append((dialog, ns))

        orchestrator_before = self.dlg._run_orchestrator
        self._call_bootstrap(recording_wire_fn)

        self.assertEqual(len(calls), 1)
        dialog, ns = calls[0]
        self.assertIs(dialog, self.dlg)
        self.assertEqual(
            set(ns.keys()),
            {
                "SWE2DRunOrchestrator",
                "SWE2DRunRequest",
                "SWE2DRunController",
                "SWE2DBackendInitializer",
                "SWE2DRunFinalizer",
                "SWE2DRunLifecycle",
            },
        )
        self.assertIs(ns["SWE2DRunOrchestrator"], SWE2DRunOrchestrator)
        self.assertIs(ns["SWE2DRunRequest"], SWE2DRunRequest)
        self.assertIs(ns["SWE2DRunController"], SWE2DRunController)
        self.assertIs(ns["SWE2DBackendInitializer"], SWE2DBackendInitializer)
        self.assertIs(ns["SWE2DRunFinalizer"], SWE2DRunFinalizer)
        self.assertIs(ns["SWE2DRunLifecycle"], SWE2DRunLifecycle)
        # bootstrap itself only delegates — it must not rebuild components
        # when wire_fn doesn't.
        self.assertIs(self.dlg._run_orchestrator, orchestrator_before)

    def test_second_bootstrap_replaces_components_without_duplicate_wiring(self):
        """A second bootstrap is a safe re-wire: new component instances, no
        doubled signal delivery, still fully functional."""
        dlg = self.dlg
        first = (
            dlg._run_orchestrator,
            dlg._run_controller,
            dlg._backend_initializer,
            dlg._run_finalizer,
            dlg._run_lifecycle,
        )
        n0 = len(dlg._runtime_log_lines)
        self._call_bootstrap(wire_startup_run_components)

        second = (
            dlg._run_orchestrator,
            dlg._run_controller,
            dlg._backend_initializer,
            dlg._run_finalizer,
            dlg._run_lifecycle,
        )
        for old, new in zip(first, second):
            self.assertIsNot(old, new)
        # Bootstrap itself must not log or mutate dialog runtime state.
        self.assertEqual(len(dlg._runtime_log_lines), n0)

        # The re-wired components are functional.
        n1 = len(dlg._runtime_log_lines)
        self.assertFalse(dlg._run_controller.run_preflight())
        self.assertIn(
            "Run preflight aborted: mesh is not available.",
            dlg._runtime_log_lines[n1:],
        )

        # No duplicate Qt connections accumulated: one click → one emission
        # → exactly one abort line.
        from qgis.PyQt.QtCore import Qt
        from qgis.PyQt.QtTest import QSignalSpy, QTest

        spy = QSignalSpy(dlg._run_dock.run_btn.clicked)
        n2 = len(dlg._runtime_log_lines)
        QTest.mouseClick(dlg._run_dock.run_btn, Qt.LeftButton)
        self.assertEqual(len(spy), 1)
        self.assertEqual(
            dlg._runtime_log_lines[n2:].count(
                "Run aborted: mesh not available after preflight."
            ),
            1,
        )


@requires_qgis
class TestFinalizationAdapter(unittest.TestCase):
    """FinalizationAdapter: every protocol method delegates to the dialog."""

    @classmethod
    def setUpClass(cls):
        cls._app = ensure_qgis_app()

    def setUp(self):
        self.dlg = _make_dialog()
        # Use the adapter the production wiring installed on the finalizer.
        self.adapter = self.dlg._run_finalizer._view
        self.assertIsInstance(self.adapter, FinalizationAdapter)

    def tearDown(self):
        self.dlg.close()
        delete_widgets_now(self.dlg)

    def test_log_message_reaches_runtime_log_and_log_view(self):
        self.adapter.log_message("wiring probe message")
        self.assertEqual(self.dlg._runtime_log_lines[-1], "wiring probe message")
        self.assertIn("wiring probe message", self.dlg.log_view.toPlainText())

    def test_runtime_log_lines_returns_live_dialog_list(self):
        lines = self.adapter.runtime_log_lines()
        self.assertIs(lines, self.dlg._runtime_log_lines)

    def test_results_data_returns_dialog_results_data(self):
        rd = self.adapter.results_data()
        self.assertIsNotNone(rd)
        self.assertIs(rd, self.dlg._results_data)

    def test_length_unit_name_default_and_override(self):
        self.assertEqual(self.adapter.length_unit_name(), "m")
        self.dlg._length_unit_name = "ft"
        try:
            self.assertEqual(self.adapter.length_unit_name(), "ft")
        finally:
            self.dlg._length_unit_name = "m"

    def test_length_scale_matches_unit_service(self):
        from swe2d.workbench.services import unit_conversion_service

        scale = self.adapter.length_scale_si_to_model()
        self.assertIsInstance(scale, float)
        self.assertGreater(scale, 0.0)
        self.assertEqual(scale, unit_conversion_service.length_scale_si_to_model())

    def test_is_cancel_requested_reflects_dialog_flag(self):
        self.assertFalse(self.adapter.is_cancel_requested())
        self.dlg._cancel_requested = True
        try:
            self.assertTrue(self.adapter.is_cancel_requested())
        finally:
            self.dlg._cancel_requested = False

    def test_results_table_name_applies_widget_prefix(self):
        edit = self.dlg._model_tab_view.results_table_name_edit
        # No prefix configured → base passes through (with default).
        edit.setText("")
        self.assertEqual(self.adapter.results_table_name("base"), "base")
        self.assertEqual(
            self.adapter.results_table_name(""), "swe2d_baked_results"
        )
        # Prefix from the real widget is applied, not doubled.
        edit.setText("mypfx")
        try:
            self.assertEqual(
                self.adapter.results_table_name("base"), "mypfx_base"
            )
            self.assertEqual(
                self.adapter.results_table_name("mypfx_base"), "mypfx_base"
            )
        finally:
            edit.setText("")

    def test_get_line_results_storage_path_uses_widget_override(self):
        tmpdir = tempfile.mkdtemp(prefix="hydra_wiring_")
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        target = os.path.join(tmpdir, "out.gpkg")
        self.dlg._model_tab_view.results_gpkg_path_edit.setText(target)
        n0 = len(self.dlg._runtime_log_lines)
        path = self.adapter.get_line_results_storage_path()
        self.assertEqual(path, os.path.abspath(target))
        self.assertTrue(
            any(
                "[ResultsPath] using override" in line
                for line in self.dlg._runtime_log_lines[n0:]
            )
        )

    def test_overlay_and_plot_delegation_no_data_is_safe(self):
        """sync_overlay_data / update_overlay_time / refresh_plot delegate
        without raising when no mesh/results are loaded."""
        self.adapter.sync_overlay_data()
        self.adapter.update_overlay_time(1.0)
        n0 = len(self.dlg._runtime_log_lines)
        self.adapter.refresh_plot()
        self.assertTrue(
            any(
                "[PlotRefresh]" in line
                for line in self.dlg._runtime_log_lines[n0:]
            )
        )

    def test_collect_run_log_metadata_uses_fallback_branch(self):
        """The dialog has no collect_run_log_metadata; the adapter's
        fallback collects real widget state via collect_run_widget_params."""
        self.assertFalse(hasattr(self.dlg, "collect_run_log_metadata"))
        metadata = self.adapter.collect_run_log_metadata()
        self.assertIsInstance(metadata, dict)
        self.assertIn("workbench_widget_state", metadata)
        self.assertIsInstance(metadata["workbench_widget_state"], dict)

    def test_persist_run_log_roundtrip_through_production_reader(self):
        """persist_run_log writes a run log readable by the production loader."""
        with make_temp_results_gpkg() as gpkg_path:
            n0 = len(self.dlg._runtime_log_lines)
            self.adapter.persist_run_log(
                gpkg_path,
                "wiring_run_log",
                "2026-08-02 10:00:00",
                "2026-08-02 10:01:00",
                60.0,
                "alpha\nbeta",
                metadata={"solver": "wiring-test"},
            )
            # Loud: the dialog wrapper must not have swallowed a failure.
            self.assertFalse(
                any(
                    "Run log persistence skipped" in line
                    for line in self.dlg._runtime_log_lines[n0:]
                )
            )
            from swe2d.results.run_log_storage import (
                load_run_logs_from_geopackage,
            )

            logs = load_run_logs_from_geopackage(gpkg_path=gpkg_path)
            matches = [l for l in logs if l["run_id"] == "wiring_run_log"]
            self.assertEqual(len(matches), 1)
            self.assertIn("alpha", matches[0]["log_text"])
            self.assertIn("beta", matches[0]["log_text"])


@requires_qgis
class TestFinalizationFlow(unittest.TestCase):
    """End-to-end: SWE2DRunFinalizer driven through the wired adapter."""

    @classmethod
    def setUpClass(cls):
        cls._app = ensure_qgis_app()

    def setUp(self):
        self.dlg = _make_dialog()
        self.tmpdir = tempfile.mkdtemp(prefix="hydra_wiring_flow_")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.gpkg_path = os.path.join(self.tmpdir, "results.gpkg")
        self.dlg._model_tab_view.results_gpkg_path_edit.setText(self.gpkg_path)
        self.finalizer = self.dlg._run_finalizer

    def tearDown(self):
        self.dlg.close()
        delete_widgets_now(self.dlg)

    def _finalize(self, *, run_id, snapshot_timesteps=None):
        h = np.array([1.0, 0.5, 0.0, 2.0], dtype=np.float64)
        hu = np.zeros(4, dtype=np.float64)
        hv = np.full(4, 0.1, dtype=np.float64)
        return self.finalizer.finalize_and_persist(
            h=h,
            hu=hu,
            hv=hv,
            final_sim_time_s=10.0,
            n_area=4,
            area_model=np.ones(4, dtype=np.float64),
            storage_start_model=0.0,
            source_budget_model={"rain": 0.25, "cell": 0.0, "coupling": 0.0},
            source_step_rows_model=[],
            run_duration_s=10.0,
            boundary_flux_budget_model={"inflow": 0.1},
            boundary_flux_step_rows_model=[],
            run_id=run_id,
            output_interval_s=5.0,
            run_perf_start=time.perf_counter(),
            run_wallclock_start="2026-08-02 10:00:00",
            run_log_start_idx=0,
            thiessen_forcing=None,
            rain_stats_acc={"samples": 0, "rain_mm": 0.0, "excess_mm": 0.0},
            save_run_log=True,
            mesh_name="wiring_mesh",
            snapshot_timesteps=snapshot_timesteps,
        )

    @staticmethod
    def _assert_order(messages, *substrings):
        """Assert each substring appears, in the given relative order."""
        last = -1
        for sub in substrings:
            hits = [i for i, m in enumerate(messages) if sub in m]
            assert hits, f"missing finalization log line containing {sub!r}"
            idx = hits[0]
            assert idx > last, f"{sub!r} out of order in {messages!r}"
            last = idx

    def test_finalize_and_persist_documented_teardown_order(self):
        """Full teardown: mass balance → persist → overlay → run log → summary."""
        h = np.array([1.0, 0.5, 0.0, 2.0], dtype=np.float64)
        snapshots = [(10.0, h, np.zeros(4), np.full(4, 0.1))]
        status = self._finalize(run_id="wiring_e2e", snapshot_timesteps=snapshots)
        messages = self.finalizer.drain_log_messages()

        self.assertTrue(status["ok"], f"errors: {status['errors']}")
        self.assertEqual(status["errors"], [])
        self.assertEqual(status["warnings"], [])
        self._assert_order(
            messages,
            "Mass balance (explicit sources/storage)",
            "Mass balance (SI reference)",
            "Boundary flux volume by group",
            "all baked results saved",
            "overlay sync + update",
            "Run wallclock end",
            "Run wallclock duration",
            "run log saved",
            "Run complete.",
            "Depth range:",
        )
        # Drain consumes the buffer.
        self.assertEqual(self.finalizer.drain_log_messages(), [])
        # refresh_plot ran as the final UI teardown step.
        self.assertTrue(
            any("[PlotRefresh]" in line for line in self.dlg._runtime_log_lines)
        )
        # Run log persisted and readable via the production reader.
        from swe2d.results.run_log_storage import load_run_logs_from_geopackage

        logs = load_run_logs_from_geopackage(gpkg_path=self.gpkg_path)
        self.assertIn("wiring_e2e", [l["run_id"] for l in logs])

    def test_finalize_without_snapshots_stores_terminal_fallback(self):
        """No interval snapshots → fallback terminal snapshot is stored live."""
        status = self._finalize(run_id="wiring_e2e_fallback")
        messages = self.finalizer.drain_log_messages()

        self.assertTrue(status["ok"], f"errors: {status['errors']}")
        self.assertTrue(
            any("Snapshot capture fallback" in m for m in messages)
        )
        live = list(self.dlg._results_data.get_live_snapshot_timesteps())
        self.assertEqual(len(live), 1)
        self.assertAlmostEqual(float(live[0][0]), 10.0)
        # KNOWN PRODUCTION FINDING (reported, not fixed here): with a live
        # snapshot but no mesh, sync_overlay_data hits the "live" overlay
        # path and mesh_runtime_logic.mesh_cell_centroids raises a raw
        # TypeError ('NoneType' object is not subscriptable) before the
        # controller's loud guard can fire; the finalizer records it as a
        # non-fatal warning. Assert the current documented behavior.
        self.assertTrue(
            any("Overlay sync failed" in w for w in status["warnings"]),
            f"expected overlay-sync warning, got {status['warnings']}",
        )

    def test_finalize_reports_cancelled_run(self):
        """is_cancel_requested() flips the terminal summary line."""
        self.dlg._cancel_requested = True
        try:
            h = np.ones(2, dtype=np.float64)
            snapshots = [(10.0, h, np.zeros(2), np.zeros(2))]
            status = self._finalize(
                run_id="wiring_e2e_cancel", snapshot_timesteps=snapshots
            )
        finally:
            self.dlg._cancel_requested = False
        messages = self.finalizer.drain_log_messages()
        self.assertTrue(status["ok"], f"errors: {status['errors']}")
        self.assertTrue(any("Run canceled by user." in m for m in messages))
        self.assertFalse(any(m == "Run complete." for m in messages))


if __name__ == "__main__":
    unittest.main()
