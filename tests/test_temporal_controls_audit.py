"""Tests for temporal-control bugs found in the second-pass audit.

These focus on SWE2DResultsData.current_time_sec staying in sync with the
animation index so plot widgets render the correct timestep.
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestCurrentTimeSecSync(unittest.TestCase):
    """Bug 1/2: current_time_sec must be current when plot widgets refresh."""

    def _make_data(self, times):
        from swe2d.results.data import SWE2DResultsData
        data = SWE2DResultsData()
        data._all_timesteps = np.asarray(times, dtype=np.float64)
        data._anim.set_timesteps(data._all_timesteps)
        return data

    def test_set_index_updates_time_before_signal_emission(self):
        data = self._make_data([0.0, 10.0, 20.0])
        observed = []

        def slot(t, idx):
            observed.append((float(data.current_time_sec), int(idx)))

        data._anim.current_timestep_changed.connect(slot)
        try:
            data.set_index(1)
        finally:
            data._anim.current_timestep_changed.disconnect(slot)

        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], 10.0)
        self.assertEqual(observed[0][1], 1)
        self.assertEqual(data.current_time_sec, 10.0)

    def test_step_forward_updates_current_time(self):
        data = self._make_data([0.0, 10.0, 20.0])
        data.set_index(0)
        self.assertEqual(data.current_time_sec, 0.0)
        data.step_forward()
        self.assertEqual(data.current_time_sec, 10.0)
        self.assertEqual(data._anim_frame_idx, 1)

    def test_step_backward_updates_current_time(self):
        data = self._make_data([0.0, 10.0, 20.0])
        data.set_index(2)
        self.assertEqual(data.current_time_sec, 20.0)
        data.step_backward()
        self.assertEqual(data.current_time_sec, 10.0)
        self.assertEqual(data._anim_frame_idx, 1)

    def test_playback_advances_current_time(self):
        data = self._make_data([0.0, 10.0, 20.0])
        data.set_index(0)
        # Simulate a timer tick: the animation controller steps forward and
        # emits current_timestep_changed.  Our data layer must reflect the
        # new index/time when the signal is emitted.
        observed = []

        def slot(t, idx):
            observed.append((float(data.current_time_sec), int(idx)))

        data._anim.current_timestep_changed.connect(slot)
        try:
            data._anim._on_tick()
        finally:
            data._anim.current_timestep_changed.disconnect(slot)

        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], 10.0)
        self.assertEqual(observed[0][1], 1)
        self.assertEqual(data.current_time_sec, 10.0)


class TestRebuildTimestepUnionSync(unittest.TestCase):
    """Bug 3: _rebuild_timestep_union must set current_time_sec before
    emitting current_timestep_changed."""

    def test_rebuild_sets_time_before_signal(self):
        from swe2d.results.data import SWE2DResultsData
        from swe2d.results.run_service import RunRecord
        from swe2d.services.gpkg_persistence_service import persist_baked_results
        import tempfile, os, sqlite3

        tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        tmp.close()
        gpkg = tmp.name
        try:
            h0 = np.zeros(1, dtype=np.float64)
            persist_baked_results(
                gpkg, "r", "m",
                snapshot_timesteps=[(5.0, h0, h0, h0), (15.0, h0, h0, h0)],
            )
            data = SWE2DResultsData()
            rec = RunRecord(run_id="r", gpkg_path=gpkg, color=(31, 119, 180))
            rec.enabled = True
            data._run_records.append(rec)
            data._selected_run_keys.add(rec.key)

            observed = []

            def slot(t, idx):
                observed.append((float(data.current_time_sec), float(t), int(idx)))

            data._anim.current_timestep_changed.connect(slot)
            try:
                data._rebuild_timestep_union()
            finally:
                data._anim.current_timestep_changed.disconnect(slot)

            self.assertEqual(data.current_time_sec, 5.0)
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0][0], 5.0)
            self.assertEqual(observed[0][1], 5.0)
        finally:
            os.remove(gpkg)


if __name__ == "__main__":
    unittest.main()
