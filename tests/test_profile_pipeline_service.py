"""tests/test_profile_pipeline_service.py"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

import numpy as np

from swe2d.workbench.services.drainage_graph_service import (
    DrainageGraph,
    load_drainage_graph,
)
from swe2d.workbench.services.profile_pipeline_service import (
    ChainSpec,
    ProfileArrays,
    assemble_chain_profile,
    load_pipe_cell_geometry,
    load_pipe_cell_records,
    profile_at_variable,
)


def _make_baked_gpkg(path: str):
    """Create a GPKG with drainage topology + pipe_cell_ts for a small chain."""
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
    for nid in ("N1", "N2", "N3"):
        conn.execute(
            "INSERT INTO swe2d_drainage_nodes VALUES (?, 0.0, 5.0, 1.0)", (nid,)
        )
    conn.execute("INSERT INTO swe2d_drainage_links VALUES ('L1', 'N1', 'N2', 100.0, 0.0, 0.0, 'circular', 2.0, 0.0)")
    conn.execute("INSERT INTO swe2d_drainage_links VALUES ('L2', 'N2', 'N3', 100.0, 0.0, 0.0, 'circular', 2.0, 0.0)")
    conn.execute("""
        CREATE TABLE swe2d_baked_pipe_cell_ts (
            run_id TEXT, link_id TEXT, cell_sub_idx INTEGER, metric TEXT,
            n_timesteps INTEGER,
            times_blob BLOB, values_blob BLOB,
            cell_invert REAL DEFAULT 0.0,
            cell_width REAL DEFAULT 2.0,
            cell_height REAL DEFAULT 2.0,
            cell_shape_type INTEGER DEFAULT 0,
            PRIMARY KEY (run_id, link_id, cell_sub_idx, metric))
    """)
    # Insert 5 sub-cells for L1 (depth metric, plus invert+geometry)
    times = np.linspace(0, 60, 6, dtype=np.float64)
    # L1: 5 sub-cells, each with its own real invert elevation (5 m at
    # upstream, dropping to ~3 m at downstream) and circular shape.
    for sub in range(5):
        depth = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=np.float64)
        velocity = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
        cell_invert = 5.0 - 0.5 * sub
        conn.execute(
            "INSERT INTO swe2d_baked_pipe_cell_ts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run_001", "L1", sub, "depth", 6, times.tobytes(), depth.tobytes(),
             cell_invert, 2.0, 2.0, 0),
        )
        conn.execute(
            "INSERT INTO swe2d_baked_pipe_cell_ts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run_001", "L1", sub, "velocity", 6, times.tobytes(), velocity.tobytes(),
             cell_invert, 2.0, 2.0, 0),
        )
    # L2: 3 sub-cells, rectangular (shape_type=1), invert 3→2 m, height 1.5 m.
    for sub in range(3):
        conn.execute(
            "INSERT INTO swe2d_baked_pipe_cell_ts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run_001", "L2", sub, "depth", 6, times.tobytes(),
             np.full(6, 1.0 + 0.1 * sub, dtype=np.float64).tobytes(),
             3.0 - 0.5 * sub, 2.0, 1.5, 1),
        )
    conn.commit()
    conn.close()


class TestChainSpec(unittest.TestCase):
    def test_cumulative_links(self):
        chain = ChainSpec(link_specs=[("L1", False), ("L2", True)])
        self.assertEqual(chain.cumulative_links(), ["L1", "L2"])
        self.assertFalse(chain.is_empty())
        self.assertFalse(ChainSpec(link_specs=[]).is_empty() == False)  # sanity


class TestLoadPipeCellRecords(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _make_baked_gpkg(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_load_returns_correct_keys(self):
        records = load_pipe_cell_records(self.path, "run_001", ["L1"])
        self.assertIn(("L1", 0, "depth"), records)
        self.assertIn(("L1", 4, "velocity"), records)
        self.assertEqual(records[("L1", 0, "depth")].shape, (6,))

    def test_load_missing_gpkg_returns_empty(self):
        self.assertEqual(load_pipe_cell_records("/nonexistent.gpkg", "x", ["L1"]), {})


class TestLoadPipeCellGeometry(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _make_baked_gpkg(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_returns_per_cell_geometry(self):
        geom = load_pipe_cell_geometry(self.path, "run_001", ["L1", "L2"])
        self.assertIn(("L1", 0), geom)
        # First L1 cell: invert=5.0, width=2.0, height=2.0, shape=0 (circular)
        self.assertAlmostEqual(geom[("L1", 0)].invert, 5.0)
        self.assertAlmostEqual(geom[("L1", 0)].width, 2.0)
        self.assertEqual(geom[("L1", 0)].shape_type, 0)
        # L2 is rectangular (shape_type=1) with height=1.5
        self.assertEqual(geom[("L2", 0)].shape_type, 1)
        self.assertAlmostEqual(geom[("L2", 0)].height, 1.5)

    def test_one_geometry_row_per_subcell(self):
        """Geometry should be deduped by (link_id, sub_idx) — even though
        the table stores the same geometry on every metric row."""
        geom = load_pipe_cell_geometry(self.path, "run_001", ["L1"])
        l1_subs = [k for k in geom if k[0] == "L1"]
        self.assertEqual(len(l1_subs), 5)

    def test_missing_gpkg_returns_empty(self):
        self.assertEqual(load_pipe_cell_geometry("/nonexistent.gpkg", "x", ["L1"]), {})

    def test_empty_link_ids_returns_empty(self):
        self.assertEqual(load_pipe_cell_geometry(self.path, "run_001", []), {})

    def test_legacy_7_column_table_uses_defaults(self):
        """If geometry columns are missing on a legacy table, the loader
        must NOT raise — it should return CellGeometry with default values
        so the pipeline can fall back to link-level invert interpolation."""
        tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        tmp.close()
        conn = sqlite3.connect(tmp.name)
        conn.execute("""
            CREATE TABLE swe2d_baked_pipe_cell_ts (
                run_id TEXT, link_id TEXT, cell_sub_idx INTEGER, metric TEXT,
                n_timesteps INTEGER, times_blob BLOB, values_blob BLOB,
                PRIMARY KEY (run_id, link_id, cell_sub_idx, metric))
        """)
        times = np.linspace(0, 1, 2, dtype=np.float64)
        conn.execute(
            "INSERT INTO swe2d_baked_pipe_cell_ts VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("r", "L1", 0, "depth", 2, times.tobytes(), times.tobytes()),
        )
        conn.commit()
        conn.close()
        try:
            geom = load_pipe_cell_geometry(tmp.name, "r", ["L1"])
            self.assertIn(("L1", 0), geom)
            # Defaults: invert=0.0, width=1.0, height=0.0, shape=0
            self.assertAlmostEqual(geom[("L1", 0)].invert, 0.0)
            self.assertAlmostEqual(geom[("L1", 0)].width, 1.0)
        finally:
            os.unlink(tmp.name)


class TestAssembleChainProfile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _make_baked_gpkg(self.path)
        self.graph = load_drainage_graph(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_assemble_single_link_forward(self):
        chain = ChainSpec(link_specs=[("L1", False)])
        p = assemble_chain_profile(self.path, "run_001", chain, self.graph, timestep_index=0)
        # L1 has 5 sub-cells
        self.assertEqual(p.station_m.shape, (5,))
        self.assertEqual(p.crown_m.shape, (5,))
        # Cell inverts in the fixture are 5.0, 4.5, 4.0, 3.5, 3.0 m.
        np.testing.assert_allclose(p.invert_m, [5.0, 4.5, 4.0, 3.5, 3.0])
        # Crown = invert + cell_width (circular) = invert + 2.0
        np.testing.assert_allclose(p.crown_m, p.invert_m + 2.0)
        # HGL = invert + depth (per cell)
        # depth at t=0 is 1.0 for every sub-cell
        np.testing.assert_allclose(p.hgl_m, p.invert_m + 1.0)

    def test_assemble_single_link_reverse(self):
        chain = ChainSpec(link_specs=[("L1", True)])
        p_fwd = assemble_chain_profile(self.path, "run_001",
            ChainSpec(link_specs=[("L1", False)]), self.graph, 0)
        p_rev = assemble_chain_profile(self.path, "run_001",
            ChainSpec(link_specs=[("L1", True)]), self.graph, 0)
        # Reversed should have depth AND invert AND hgl all reversed
        np.testing.assert_allclose(p_rev.depth_m, p_fwd.depth_m[::-1])
        np.testing.assert_allclose(p_rev.invert_m, p_fwd.invert_m[::-1])
        np.testing.assert_allclose(p_rev.hgl_m, p_fwd.hgl_m[::-1])

    def test_assemble_uses_real_cell_invert_not_zero(self):
        """Regression test: the pipeline must read cell_invert from the GPKG,
        not hard-code 0.0 — the plotted HGL/crown must be shifted to the
        cell invert, otherwise the visualization lies about elevations.
        """
        chain = ChainSpec(link_specs=[("L1", False)])
        p = assemble_chain_profile(self.path, "run_001", chain, self.graph, timestep_index=0)
        # The first cell in L1 has cell_invert = 5.0 m.  HGL = 5.0 + 1.0 = 6.0 m.
        # If cell_invert were 0.0, HGL would equal 1.0 m.
        self.assertGreater(float(p.hgl_m[0]), 5.0)
        np.testing.assert_allclose(p.hgl_m[0], 6.0)
        # Crown elevation of first cell = invert + width = 5.0 + 2.0 = 7.0
        np.testing.assert_allclose(p.crown_m[0], 7.0)

    def test_assemble_rectangular_link_uses_height(self):
        """Rectangular cells (shape_type=1) use cell_height, not cell_width,
        for the crown offset."""
        chain = ChainSpec(link_specs=[("L2", False)])
        p = assemble_chain_profile(self.path, "run_001", chain, self.graph, timestep_index=0)
        # L2 fixture: shape_type=1, cell_height=1.5, cell_width=2.0
        # Crown offset = height = 1.5 (rectangular), not width = 2.0
        self.assertTrue(np.all(p.crown_offset_m == 1.5))
        self.assertTrue(np.allclose(p.crown_m, p.invert_m + 1.5))
        # Style flag should be 'rectangular'
        self.assertEqual(p.crown_style, "rectangular")

    def test_mixed_crown_style(self):
        """A chain with one circular + one rectangular link reports 'mixed'."""
        chain = ChainSpec(link_specs=[("L1", False), ("L2", False)])
        p = assemble_chain_profile(self.path, "run_001", chain, self.graph, timestep_index=0)
        self.assertEqual(p.crown_style, "mixed")

    def test_assemble_two_link_chain(self):
        chain = ChainSpec(link_specs=[("L1", False), ("L2", False)])
        p = assemble_chain_profile(self.path, "run_001", chain, self.graph, 0)
        # 5 + 3 = 8 cells
        self.assertEqual(p.station_m.shape, (8,))
        # 3 node endpoints (N1, N2, N3)
        self.assertEqual(len(p.node_stations), 3)
        self.assertEqual(p.node_ids, ["N1", "N2", "N3"])

    def test_crown_circular(self):
        chain = ChainSpec(link_specs=[("L1", False)])
        p = assemble_chain_profile(self.path, "run_001", chain, self.graph, 0)
        self.assertEqual(p.crown_style, "circular")
        # Circular: crown = invert + cell_width (= 2.0 for every cell in the fixture)
        # Note: crown is NO LONGER flat 2.0 because invert is now read from the
        # geometry columns (5.0, 4.5, ...). The crown follows the invert.
        np.testing.assert_allclose(p.crown_m, p.invert_m + 2.0)
        np.testing.assert_allclose(p.crown_offset_m, 2.0)

    def test_ground_interpolation_at_nodes(self):
        chain = ChainSpec(link_specs=[("L1", False)])
        p = assemble_chain_profile(self.path, "run_001", chain, self.graph, 0)
        # Two endpoints (N1 at station 0, N2 at station 100)
        # Ground between two rim_elev=5.0 nodes should be 5.0 everywhere
        self.assertTrue(np.allclose(p.ground_m, 5.0))

    def test_clamps_timestep_index_out_of_range(self):
        chain = ChainSpec(link_specs=[("L1", False)])
        # Should not raise; should clamp to last index
        p = assemble_chain_profile(self.path, "run_001", chain, self.graph, timestep_index=999)
        self.assertEqual(p.depth_m.shape, (5,))


class TestProfileAtVariable(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _make_baked_gpkg(self.path)
        self.graph = load_drainage_graph(self.path)
        self.chain = ChainSpec(link_specs=[("L1", False)])
        self.profile = assemble_chain_profile(
            self.path, "run_001", self.chain, self.graph, 0
        )

    def tearDown(self):
        os.unlink(self.path)

    def test_returns_depth(self):
        vals, stations = profile_at_variable(self.profile, "depth")
        self.assertEqual(vals.shape, (5,))
        np.testing.assert_array_equal(vals, self.profile.depth_m)

    def test_returns_velocity(self):
        vals, _ = profile_at_variable(self.profile, "velocity")
        np.testing.assert_array_equal(vals, self.profile.velocity_ms)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            profile_at_variable(self.profile, "nonsense")
