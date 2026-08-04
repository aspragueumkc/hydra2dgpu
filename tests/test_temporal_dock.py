"""Behavioral tests for swe2d/workbench/views/temporal_dock.py (TemporalDockWidget).

Pattern P2 (dialog workflow) per docs/specs/2026-08-02-gui-test-coverage-design.md §4.

Real contract (production wiring in
``swe2d/workbench/views/studio_results_panel.py``): the dock is a pure view.
It is bound to a real ``SWE2DResultsData`` via ``set_data()``; navigation
buttons and the slider call through to the data object, and the lazily
created ``ResultsAnimationController`` (``data.anim``) emits
``current_timestep_changed(float, int)`` / ``play_state_changed(bool)``,
which the dock's public slots ``on_timestep_changed`` /
``on_play_state_changed`` consume.  These tests reproduce that exact wiring
with real objects — no mocks.
"""

import unittest

import numpy as np

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    requires_qgis,
)

_SNAP_TIMES_S = (0.0, 600.0, 1200.0, 1800.0)
_N_CELLS = 4


def _make_snapshots():
    """Deterministic (t_s, h, hu, hv) snapshot list for SWE2DResultsData."""
    snaps = []
    for k, t in enumerate(_SNAP_TIMES_S):
        h = np.full(_N_CELLS, 1.0 + 0.1 * k, dtype=np.float64)
        hu = np.full(_N_CELLS, 0.01 * k, dtype=np.float64)
        hv = np.full(_N_CELLS, -0.005 * k, dtype=np.float64)
        snaps.append((t, h, hu, hv))
    return snaps


@requires_qgis
class TestTemporalDockWidget(unittest.TestCase):
    """Full public surface of TemporalDockWidget against a real SWE2DResultsData."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()
        from qgis.PyQt.QtTest import QSignalSpy  # noqa: F401  (import check)

    def setUp(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.workbench.views.temporal_dock import TemporalDockWidget

        self.widget = TemporalDockWidget()
        self.widget.resize(640, 40)

        self.data = SWE2DResultsData()
        # Force lazy anim creation BEFORE loading timesteps so the controller
        # receives them (mirrors production: the panel accesses data.anim to
        # connect signals before snapshots arrive).
        anim = self.data.anim
        # Exact production wiring (studio_results_panel.py): anim signals feed
        # the dock's public slots.
        anim.current_timestep_changed.connect(self.widget.on_timestep_changed)
        anim.play_state_changed.connect(self.widget.on_play_state_changed)
        self.data.set_live_snapshot_timesteps(_make_snapshots())
        self.widget.set_data(self.data)

    def tearDown(self):
        self.data.pause()
        anim = self.data._anim if self.data._anim is not None else None
        delete_widgets_now(self.widget, anim)

    # ------------------------------------------------------------------
    # Construction / no-data contract
    # ------------------------------------------------------------------

    def test_initial_state_without_data(self):
        """Freshly constructed dock: empty slider range, zero label, 1.0x speed."""
        from swe2d.workbench.views.temporal_dock import TemporalDockWidget

        w = TemporalDockWidget()
        try:
            self.assertEqual(w._time_slider.minimum(), 0)
            self.assertEqual(w._time_slider.maximum(), 0)
            self.assertEqual(w._time_slider.value(), 0)
            self.assertEqual(w._time_lbl.text(), "T = 0.000 hr")
            self.assertEqual(w._speed_combo.currentIndex(), 2)
            self.assertEqual(w._speed_combo.currentData(), 1.0)

            # set_data(None) and all handlers are loud no-ops with no data.
            w.set_data(None)
            self.assertEqual(w._time_slider.maximum(), 0)
            w._step_back_btn.click()
            w._step_fwd_btn.click()
            w._play_btn.click()
            w._time_slider.setValue(0)
            w._speed_combo.setCurrentIndex(0)
            self.assertIsNone(w._data)

            # on_timestep_changed guards on _data is None: label must not change.
            w.on_timestep_changed(3600.0, 1)
            self.assertEqual(w._time_lbl.text(), "T = 0.000 hr")
            # Production contract: on_play_state_changed has NO no-data guard —
            # it always repaints the button (the slot is only connected once
            # data exists, so this path is unreachable in production wiring).
            w.on_play_state_changed(True)
            self.assertEqual(w._play_btn.text(), "⏸")
        finally:
            delete_widgets_now(w)

    # ------------------------------------------------------------------
    # set_data — time range reflected in internal state
    # ------------------------------------------------------------------

    def test_set_data_sets_slider_range_from_frame_count(self):
        """Slider range becomes 0 .. frame_count-1 after set_data."""
        self.assertEqual(self.data.frame_count, len(_SNAP_TIMES_S))
        self.assertEqual(self.widget._time_slider.minimum(), 0)
        self.assertEqual(
            self.widget._time_slider.maximum(), len(_SNAP_TIMES_S) - 1
        )

    # ------------------------------------------------------------------
    # Navigation — step buttons emit correct timestamps (QSignalSpy)
    # ------------------------------------------------------------------

    def test_step_forward_emits_timestep_and_syncs_view(self):
        """Step-forward button → data.step_forward → (t_s, idx) emission."""
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.data.anim.current_timestep_changed)
        self.widget._step_fwd_btn.click()

        self.assertEqual(len(spy), 1)
        t_s, idx = spy[0]
        self.assertAlmostEqual(float(t_s), 600.0)
        self.assertEqual(int(idx), 1)

        # View state reflects the new frame via on_timestep_changed.
        self.assertEqual(self.widget._time_slider.value(), 1)
        self.assertEqual(self.widget._time_lbl.text(), "T = 0.167 hr")
        # Data-layer state reflects it too (production read path).
        self.assertEqual(self.data.current_frame_idx, 1)
        self.assertAlmostEqual(self.data.current_time_sec, 600.0)

    def test_step_backward_moves_back_one_frame(self):
        from qgis.PyQt.QtTest import QSignalSpy

        self.data.set_index(2)
        spy = QSignalSpy(self.data.anim.current_timestep_changed)
        self.widget._step_back_btn.click()

        self.assertEqual(len(spy), 1)
        t_s, idx = spy[0]
        self.assertAlmostEqual(float(t_s), 600.0)
        self.assertEqual(int(idx), 1)
        self.assertEqual(self.widget._time_slider.value(), 1)

    # ------------------------------------------------------------------
    # Boundary behavior — clamp at start, wrap past end (real contract)
    # ------------------------------------------------------------------

    def test_step_backward_at_first_frame_clamps(self):
        """At frame 0, step-back clamps to 0 — no wrap, no new emission."""
        from qgis.PyQt.QtTest import QSignalSpy

        self.assertEqual(self.data.current_frame_idx, 0)
        spy = QSignalSpy(self.data.anim.current_timestep_changed)
        self.widget._step_back_btn.click()

        self.assertEqual(len(spy), 0)
        self.assertEqual(self.data.current_frame_idx, 0)
        self.assertEqual(self.widget._time_slider.value(), 0)
        self.assertEqual(self.widget._time_lbl.text(), "T = 0.000 hr")

    def test_step_forward_past_last_frame_wraps_to_zero(self):
        """Step-forward at the last frame wraps to frame 0 (production contract)."""
        from qgis.PyQt.QtTest import QSignalSpy

        self.data.set_index(len(_SNAP_TIMES_S) - 1)
        spy = QSignalSpy(self.data.anim.current_timestep_changed)
        self.widget._step_fwd_btn.click()

        self.assertEqual(len(spy), 1)
        t_s, idx = spy[0]
        self.assertAlmostEqual(float(t_s), 0.0)
        self.assertEqual(int(idx), 0)
        self.assertEqual(self.widget._time_slider.value(), 0)

    # ------------------------------------------------------------------
    # Paired controls — slider <-> time label stay in sync both directions
    # ------------------------------------------------------------------

    def test_slider_seek_drives_data_and_label(self):
        """Dragging the slider seeks the animation and updates the time label."""
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.data.anim.current_timestep_changed)
        self.widget._time_slider.setValue(2)

        self.assertEqual(len(spy), 1)
        t_s, idx = spy[0]
        self.assertAlmostEqual(float(t_s), 1200.0)
        self.assertEqual(int(idx), 2)
        self.assertEqual(self.widget._time_lbl.text(), "T = 0.333 hr")
        self.assertEqual(self.data.current_frame_idx, 2)
        self.assertAlmostEqual(self.data.current_time_sec, 1200.0)

    def test_on_timestep_changed_syncs_slider_without_reemit(self):
        """External timestep signal moves the slider with signals blocked —
        no feedback loop back into data.set_index."""
        from qgis.PyQt.QtTest import QSignalSpy

        self.data.set_index(1)
        spy = QSignalSpy(self.data.anim.current_timestep_changed)
        self.widget.on_timestep_changed(1800.0, 3)

        self.assertEqual(self.widget._time_slider.value(), 3)
        self.assertEqual(self.widget._time_lbl.text(), "T = 0.500 hr")
        # Slider signals were blocked: nothing re-entered the data layer.
        self.assertEqual(len(spy), 0)
        self.assertEqual(self.data.current_frame_idx, 1)

    # ------------------------------------------------------------------
    # Play / pause — signal emission and button state
    # ------------------------------------------------------------------

    def test_play_pause_toggles_state_and_button(self):
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.data.anim.play_state_changed)

        self.widget._play_btn.click()
        self.assertEqual(len(spy), 1)
        self.assertEqual(bool(spy[0][0]), True)
        self.assertTrue(self.data.is_playing)
        self.assertTrue(self.widget._play_btn.isChecked())
        self.assertEqual(self.widget._play_btn.text(), "⏸")

        self.widget._play_btn.click()
        self.assertEqual(len(spy), 2)
        self.assertEqual(bool(spy[1][0]), False)
        self.assertFalse(self.data.is_playing)
        self.assertFalse(self.widget._play_btn.isChecked())
        self.assertEqual(self.widget._play_btn.text(), "▶")

    def test_on_play_state_changed_updates_button_without_reemit(self):
        """External play-state signal updates the button with signals blocked."""
        from qgis.PyQt.QtTest import QSignalSpy

        # No data-layer round trip: clicking the button is what triggers
        # play/pause; the slot itself must only repaint.
        spy = QSignalSpy(self.data.anim.play_state_changed)
        self.widget.on_play_state_changed(True)
        self.assertTrue(self.widget._play_btn.isChecked())
        self.assertEqual(self.widget._play_btn.text(), "⏸")
        self.assertEqual(len(spy), 0)
        self.assertFalse(self.data.is_playing)

        self.widget.on_play_state_changed(False)
        self.assertFalse(self.widget._play_btn.isChecked())
        self.assertEqual(self.widget._play_btn.text(), "▶")
        self.assertEqual(len(spy), 0)

    # ------------------------------------------------------------------
    # Speed combo — drives data.set_frame_rate(4.0 * speed)
    # ------------------------------------------------------------------

    def test_speed_combo_updates_frame_rate(self):
        """Speed selection multiplies the 4 fps base rate."""
        self.widget._speed_combo.setCurrentIndex(3)  # 2.0x
        # No public fps getter exists; _anim_fps is the authoritative field
        # that set_frame_rate writes and the controller reads.
        self.assertAlmostEqual(self.data._anim_fps, 8.0)
        self.assertAlmostEqual(self.data.anim._fps, 8.0)

        self.widget._speed_combo.setCurrentIndex(0)  # 0.25x
        self.assertAlmostEqual(self.data._anim_fps, 1.0)
        self.assertAlmostEqual(self.data.anim._fps, 1.0)

        self.widget._speed_combo.setCurrentIndex(5)  # 8.0x
        self.assertAlmostEqual(self.data._anim_fps, 32.0)
        self.assertAlmostEqual(self.data.anim._fps, 32.0)

    # ------------------------------------------------------------------
    # Render smoke — the dock actually paints content offscreen
    # ------------------------------------------------------------------

    def test_grab_non_empty(self):
        self.assertTrue(
            grab_non_empty(self.widget),
            "temporal dock grabbed to an empty image",
        )


if __name__ == "__main__":
    unittest.main()
