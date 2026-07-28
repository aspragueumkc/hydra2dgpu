"""tests/test_network_profile_dialog.py"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

import numpy as np
from qgis.PyQt.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from swe2d.workbench.dialogs.network_profile_dialog import NetworkProfileDialog


def _make_gpkg(path: str):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE swe2d_drainage_nodes (
            node_id TEXT, invert_elev REAL DEFAULT 0.0, rim_elev REAL DEFAULT 5.0, max_depth REAL DEFAULT 1.0)
    """)
    conn.execute("""
        CREATE TABLE swe2d_drainage_links (
            link_id TEXT, from_node TEXT, to_node TEXT,
            length REAL DEFAULT 100.0,
            inlet_invert_elev REAL DEFAULT 0.0, outlet_invert_elev REAL DEFAULT 0.0,
            link_shape TEXT DEFAULT 'circular',
            diameter REAL DEFAULT 2.0,
            rise REAL DEFAULT 0.0)
    """)
    conn.execute("INSERT INTO swe2d_drainage_nodes VALUES ('N1', 0.0, 5.0, 1.0)")
    conn.execute("INSERT INTO swe2d_drainage_nodes VALUES ('N2', 0.0, 5.0, 1.0)")
    conn.execute("INSERT INTO swe2d_drainage_links VALUES ('L1', 'N1', 'N2', 100.0, 0.0, 0.0, 'circular', 2.0, 0.0)")
    conn.execute("""
        CREATE TABLE swe2d_baked_pipe_cell_ts (
            run_id TEXT, link_id TEXT, cell_sub_idx INTEGER, metric TEXT,
            n_timesteps INTEGER, times_blob BLOB, values_blob BLOB,
            cell_invert REAL DEFAULT 0.0, cell_width REAL DEFAULT 2.0,
            cell_height REAL DEFAULT 2.0, cell_shape_type INTEGER DEFAULT 0,
            PRIMARY KEY (run_id, link_id, cell_sub_idx, metric))
    """)
    conn.execute("""
        CREATE TABLE swe2d_run_logs (run_id TEXT, created_utc TEXT)
    """)
    conn.execute("INSERT INTO swe2d_run_logs VALUES ('run_001', '2024-01-01')")
    times = np.linspace(0, 60, 3, dtype=np.float64)
    for sub in range(5):
        depth = np.full(3, 1.0, dtype=np.float64)
        conn.execute(
            "INSERT INTO swe2d_baked_pipe_cell_ts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run_001", "L1", sub, "depth", 3, times.tobytes(), depth.tobytes(),
             0.0, 2.0, 2.0, 0),
        )
    conn.commit()
    conn.close()


class TestNetworkProfileDialog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _make_gpkg(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_dialog_instantiates_with_gpkg(self):
        dlg = NetworkProfileDialog(self.path, run_id="run_001")
        self.assertEqual(dlg._run_id, "run_001")
        self.assertEqual(dlg._n_timesteps, 3)

    def test_dialog_renders_with_chain(self):
        from swe2d.workbench.services.profile_pipeline_service import ChainSpec
        dlg = NetworkProfileDialog(self.path, run_id="run_001")
        dlg._chain_widget.set_chain(ChainSpec(link_specs=[("L1", False)]))
        # After set_chain triggers chain_changed -> _render
        self.assertIsNotNone(dlg._profile)
        self.assertEqual(dlg._profile.station_m.shape, (5,))

    def test_dialog_time_slider_recomputes(self):
        from swe2d.workbench.services.profile_pipeline_service import ChainSpec
        dlg = NetworkProfileDialog(self.path, run_id="run_001")
        dlg._chain_widget.set_chain(ChainSpec(link_specs=[("L1", False)]))
        dlg._timestep_slider.setValue(2)
        dlg._on_slider_change(2)
        self.assertEqual(dlg._timestep_index, 2)
