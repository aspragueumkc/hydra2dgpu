"""tests/test_profile_persistence_service.py"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from swe2d.workbench.services.profile_persistence_service import (
    PERSISTED_TABLE,
    delete_profile,
    list_profiles,
    load_profile,
    save_profile,
)
from swe2d.workbench.services.profile_pipeline_service import ChainSpec


def _empty_gpkg(path: str):
    conn = sqlite3.connect(path)
    conn.close()


class TestSaveProfile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()
        _empty_gpkg(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_save_profile_creates_table(self):
        chain = ChainSpec(link_specs=[("L1", False), ("L2", True)])
        pid = save_profile(self.path, "test_profile", chain, run_id="run_001")
        self.assertGreater(pid, 0)
        conn = sqlite3.connect(self.path)
        try:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (PERSISTED_TABLE,),
            ).fetchone()
            self.assertIsNotNone(row)
        finally:
            conn.close()

    def test_save_profile_replaces_existing(self):
        chain1 = ChainSpec(link_specs=[("L1", False)])
        chain2 = ChainSpec(link_specs=[("L2", False)])
        pid1 = save_profile(self.path, "test", chain1)
        pid2 = save_profile(self.path, "test", chain2)
        self.assertEqual(pid1, pid2)
        rows = list_profiles(self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["link_ids"], "L2,F")

    def test_save_profile_round_trip(self):
        chain = ChainSpec(link_specs=[("L1", False), ("L3", True), ("L5", False)])
        save_profile(self.path, "rt", chain)
        rows = list_profiles(self.path)
        loaded = load_profile(self.path, rows[0]["profile_id"])
        self.assertEqual(loaded.link_specs, chain.link_specs)


class TestListProfiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()

    def tearDown(self):
        os.unlink(self.path)

    def test_list_returns_in_insertion_order(self):
        save_profile(self.path, "a", ChainSpec(link_specs=[("L1", False)]))
        save_profile(self.path, "b", ChainSpec(link_specs=[("L2", False)]))
        save_profile(self.path, "c", ChainSpec(link_specs=[("L3", False)]))
        names = [p["profile_name"] for p in list_profiles(self.path)]
        self.assertEqual(names, ["a", "b", "c"])

    def test_list_empty_gpkg(self):
        self.assertEqual(list_profiles(self.path), [])

    def test_list_missing_gpkg(self):
        self.assertEqual(list_profiles("/nonexistent/path.gpkg"), [])


class TestDeleteProfile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False)
        self.path = self._tmp.name
        self._tmp.close()

    def tearDown(self):
        os.unlink(self.path)

    def test_delete_removes_row(self):
        save_profile(self.path, "x", ChainSpec(link_specs=[("L1", False)]))
        pid = list_profiles(self.path)[0]["profile_id"]
        delete_profile(self.path, pid)
        self.assertEqual(list_profiles(self.path), [])

    def test_delete_unknown_id_no_op(self):
        # No exception
        delete_profile(self.path, 9999)
        self.assertEqual(list_profiles(self.path), [])
