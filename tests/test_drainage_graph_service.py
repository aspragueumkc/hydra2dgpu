"""tests/test_drainage_graph_service.py"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from swe2d.workbench.services.drainage_graph_service import (
    DrainageGraph,
    find_chain,
    link_orientation,
    load_drainage_graph,
)


def _make_gpkg(path: str, links: list[tuple[str, str, str]], nodes: list[tuple[str, float, float]] | None = None):
    """links: [(link_id, from_node, to_node)]; nodes: [(node_id, invert, rim)]"""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE swe2d_drainage_links (
            link_id TEXT, from_node TEXT, to_node TEXT, length REAL DEFAULT 100.0,
            inlet_invert_elev REAL DEFAULT 0.0, outlet_invert_elev REAL DEFAULT 0.0)
    """)
    conn.execute("""
        CREATE TABLE swe2d_drainage_nodes (
            node_id TEXT, invert_elev REAL DEFAULT 0.0, rim_elev REAL DEFAULT 1.0, max_depth REAL DEFAULT 1.0)
    """)
    for lid, fn, tn in links:
        conn.execute("INSERT INTO swe2d_drainage_links VALUES (?, ?, ?, 100.0, 0.0, 0.0)", (lid, fn, tn))
    for nid, inv, rim in (nodes or []):
        conn.execute("INSERT INTO swe2d_drainage_nodes VALUES (?, ?, ?, 1.0)", (nid, inv, rim))
    conn.commit()
    conn.close()


class TestLoadDrainageGraph(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()

    def tearDown(self):
        os.unlink(self.path)

    def test_load_empty_graph(self):
        g = load_drainage_graph(self.path)
        self.assertEqual(g.link_ids, [])
        self.assertEqual(g.node_ids, [])

    def test_load_single_link(self):
        _make_gpkg(self.path, [("L1", "A", "B")], [("A", 0.0, 1.0), ("B", 0.0, 1.0)])
        g = load_drainage_graph(self.path)
        self.assertEqual(g.link_ids, ["L1"])
        self.assertIn("A", g.node_ids)
        self.assertEqual(g.from_node["L1"], "A")
        self.assertEqual(g.to_node["L1"], "B")

    def test_load_branching_network(self):
        # N1 -> N2 via L1; N1 -> N3 via L2; N2 -> N3 via L3
        _make_gpkg(self.path, [
            ("L1", "N1", "N2"),
            ("L2", "N1", "N3"),
            ("L3", "N2", "N3"),
        ], [(n, 0.0, 1.0) for n in ("N1", "N2", "N3")])
        g = load_drainage_graph(self.path)
        self.assertEqual(len(g.link_ids), 3)
        self.assertEqual(sorted(g.node_ids), ["N1", "N2", "N3"])
        self.assertEqual(sorted(g.outgoing["N1"]), ["L1", "L2"])
        self.assertEqual(sorted(g.outgoing["N2"]), ["L3"])
        self.assertEqual(sorted(g.incoming["N3"]), ["L2", "L3"])

    def test_load_missing_gpkg_returns_empty(self):
        g = load_drainage_graph("/nonexistent/path.gpkg")
        self.assertEqual(g.link_ids, [])


class TestFindChain(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()

    def tearDown(self):
        os.unlink(self.path)

    def test_two_link_path(self):
        _make_gpkg(self.path, [("L1", "A", "B"), ("L2", "B", "C")])
        g = load_drainage_graph(self.path)
        self.assertEqual(find_chain(g, "A", "C"), ["L1", "L2"])

    def test_branching_chooses_shortest(self):
        # N1 -> N2 -> N4 via L1, L3 (2 hops)
        # N1 -> N3 -> N4 via L2, L4 (2 hops; both equally short, BFS picks alpha-first)
        _make_gpkg(self.path, [
            ("L1", "N1", "N2"), ("L3", "N2", "N4"),
            ("L2", "N1", "N3"), ("L4", "N3", "N4"),
        ])
        g = load_drainage_graph(self.path)
        result = find_chain(g, "N1", "N4")
        self.assertEqual(set(result), {"L1", "L3"} or {"L2", "L4"})
        # Length must be 2
        self.assertEqual(len(result), 2)

    def test_no_path_returns_empty(self):
        _make_gpkg(self.path, [("L1", "A", "B")])
        g = load_drainage_graph(self.path)
        self.assertEqual(find_chain(g, "A", "Z"), [])

    def test_same_start_end_returns_empty(self):
        _make_gpkg(self.path, [("L1", "A", "B")])
        g = load_drainage_graph(self.path)
        self.assertEqual(find_chain(g, "A", "A"), [])

    def test_one_node_isolated(self):
        _make_gpkg(self.path, [("L1", "A", "B")], [("Z", 0, 1)])
        g = load_drainage_graph(self.path)
        self.assertEqual(find_chain(g, "A", "Z"), [])


class TestLinkOrientation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()

    def tearDown(self):
        os.unlink(self.path)

    def test_link_orientation_forward(self):
        _make_gpkg(self.path, [("L1", "A", "B")])
        g = load_drainage_graph(self.path)
        self.assertTrue(link_orientation(g, "L1", "A"))

    def test_link_orientation_reverse(self):
        _make_gpkg(self.path, [("L1", "A", "B")])
        g = load_drainage_graph(self.path)
        self.assertFalse(link_orientation(g, "L1", "B"))

    def test_link_orientation_unknown_upstream(self):
        _make_gpkg(self.path, [("L1", "A", "B")])
        g = load_drainage_graph(self.path)
        # Expected upstream neither endpoint -> default to forward
        self.assertTrue(link_orientation(g, "L1", "Z"))
