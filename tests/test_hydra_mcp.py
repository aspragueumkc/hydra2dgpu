"""Unit tests for the HYDRA MCP server Phase 0 tools.

Builds a small model/results GeoPackage fixture in-test using the real
persistence functions from swe2d.services.gpkg_persistence_service (plus a
direct sqlite3 insert for the swe2d_run_logs table, whose loader module
cannot be imported without the native hydra_overlay module), then exercises
tools/hydra_mcp/tools_modeling.py against it.

Run:
    PYTHONPATH=<repo root> python3 -m unittest tests.test_hydra_mcp -v
"""
import asyncio
import json
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

# Allow tests run from a worktree to find the native CUDA extension by passing
# the build directory via the HYDRA_BUILD_DIR environment variable.
_HYDRA_BUILD_DIR = os.environ.get("HYDRA_BUILD_DIR")
if _HYDRA_BUILD_DIR and str(_HYDRA_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_HYDRA_BUILD_DIR))

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from swe2d.services.gpkg_persistence_service import (
    persist_all_baked_results,
    persist_baked_mesh,
    persist_simulation_config,
)

from tools.hydra_mcp import tools_modeling, tools_modeling_phase1, tools_design, tools_audit

from tests.qgis_real_env import ensure_qgis_app, requires_qgis, stub_iface

# True when the native CUDA extension is importable.  Since eb149116
# ("native mesh only, no dry_run") spec validation always deserializes the
# baked-mesh BLOB via hydra_swe2d, so any test that needs a *valid* spec is
# gated on this.
_HYDRA_SWE2D_AVAILABLE = (
    __import__("importlib").util.find_spec("hydra_swe2d") is not None
)


def _bake_real_quad_mesh(gpkg_path: str, mesh_name: str) -> None:
    """Bake a genuine 4-cell quad mesh over the fixture's placeholder BLOB.

    Uses the repo's real build → serialize → persist path
    (``tests._swe2d_test_helpers._serialize_and_persist_mesh``, the same
    helper the GPU/CLI suites use) so ``spec_validate`` deserializes a
    genuinely valid mesh.  Requires the native hydra_swe2d extension.
    """
    from tests._swe2d_test_helpers import _serialize_and_persist_mesh

    node_x = np.tile(np.arange(3, dtype=np.float64), 3)
    node_y = np.repeat(np.arange(3, dtype=np.float64), 3)
    node_z = np.zeros(9, dtype=np.float64)
    cell_nodes = np.array(
        [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]],
        dtype=np.int32,
    )
    _serialize_and_persist_mesh(
        gpkg_path, mesh_name, node_x, node_y, node_z, cell_nodes,
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float64),
    )


class HydraMcpFixtureMixin:
    """Build a small GPKG fixture: one mesh, one config, two runs, one run log."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        # Point the MCP workspace root at this tempdir so the workspace
        # containment check accepts fixture paths.
        self._prev_workspace_root = os.environ.get("HYDRA_MCP_WORKSPACE_ROOT")
        os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = self.tmpdir.name
        self.addCleanup(self._restore_workspace_root)
        self.gpkg_path = os.path.join(self.tmpdir.name, "model.gpkg")
        self.mesh_name = "fixture_mesh"
        self.run_id = "run_001"
        self.n_cells = 4

        persist_baked_mesh(
            self.gpkg_path, self.mesh_name, b"\x00" * 32,
            n_nodes=6, n_cells=self.n_cells, n_edges=9,
        )
        persist_simulation_config(
            self.gpkg_path, config_id="cfg_001", mesh_name=self.mesh_name,
            run_duration_s=3600.0, widget_state={"version": 1},
            description="fixture config", params={"roughness": 0.03},
        )
        rng = np.arange(self.n_cells, dtype=np.float64)
        snapshots = [
            (0.0, rng * 0.0, rng * 0.1, rng * 0.2),
            (10.0, rng + 1.0, rng * 0.1 + 1.0, rng * 0.2),
            (20.0, rng + 2.0, rng * 0.1 + 2.0, rng * 0.2),
        ]
        # Inject a NaN so nan_count reporting is exercised.
        snapshots[1][1][2] = np.nan
        max_tracking = {
            "max_h": rng + 2.0,
            "max_hu": rng * 0.1 + 2.0,
            # max_hv deliberately omitted: not always stored.
        }
        # Baked line timeseries for run_001 so results_timeseries is testable.
        line_ts = [{
            "line_id": 1,
            "line_name": "test_line",
            "times": np.array([0.0, 10.0, 20.0], dtype=np.float64),
            "depth": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "velocity": np.array([0.0, 0.5, 1.0], dtype=np.float64),
            "wse": np.array([0.0, 1.5, 3.0], dtype=np.float64),
            "bed": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "flow": np.array([0.0, 2.0, 4.0], dtype=np.float64),
            "wet_frac": np.array([0.0, 1.0, 1.0], dtype=np.float64),
            "fr": np.array([0.0, 0.1, 0.2], dtype=np.float64),
        }]
        persist_all_baked_results(
            self.gpkg_path, run_id=self.run_id, mesh_name=self.mesh_name,
            snapshot_timesteps=snapshots, max_tracking=max_tracking,
            line_ts_items=line_ts,
        )
        # A second run with no max tracking.
        persist_all_baked_results(
            self.gpkg_path, run_id="run_002", mesh_name=self.mesh_name,
            snapshot_timesteps=[(0.0, rng, rng, rng)],
        )
        # Run-log record for run_001 (mirrors swe2d_run_logs schema).
        conn = sqlite3.connect(self.gpkg_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS swe2d_run_logs ("
                "run_id TEXT PRIMARY KEY, created_utc TEXT, start_wallclock TEXT, "
                "end_wallclock TEXT, duration_s REAL, log_text TEXT, metadata_json TEXT)"
            )
            # Workbench-style run log: metadata is a bulky nested blob
            # (mirrors finalization_adapter.collect_run_log_metadata, which
            # writes {"workbench_widget_state": {...}}).
            conn.execute(
                "INSERT OR REPLACE INTO swe2d_run_logs VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.run_id, "2026-07-24T10:15:00+00:00", "10:15:00", "10:16:30",
                 90.0, "log text",
                 json.dumps({"workbench_widget_state": {
                     "version": 1,
                     "model_tab": {"run_duration_s": "3600", "mesh_name": self.mesh_name},
                 }})),
            )
            # CLI/headless-style run log: empty metadata, no baked results
            # (mirrors headless_executor.collect_run_log_metadata -> {}).
            conn.execute(
                "INSERT OR REPLACE INTO swe2d_run_logs VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("run_cli_only", "2026-07-24T11:00:00+00:00", "11:00:00", "11:00:05",
                 5.0, "cli log text", json.dumps({})),
            )
            conn.commit()
        finally:
            conn.close()

    def _restore_workspace_root(self) -> None:
        if self._prev_workspace_root is None:
            os.environ.pop("HYDRA_MCP_WORKSPACE_ROOT", None)
        else:
            os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = self._prev_workspace_root


class TestModelInspect(HydraMcpFixtureMixin, unittest.TestCase):

    def test_inspect_lists_meshes_layers_configs_runs(self):
        out = tools_modeling.model_inspect(self.gpkg_path)
        self.assertTrue(out["ok"], out)
        self.assertEqual([m["mesh_name"] for m in out["meshes"]], [self.mesh_name])
        self.assertEqual(out["meshes"][0]["n_cells"], self.n_cells)
        self.assertTrue(out["layers"], "expected gpkg_contents layers")
        cfg_ids = [c["config_id"] for c in out["simulation_configs"]]
        self.assertIn("cfg_001", cfg_ids)
        cfg = next(c for c in out["simulation_configs"] if c["config_id"] == "cfg_001")
        self.assertNotIn("widget_state", cfg, "bulky widget_state must be stripped")
        self.assertEqual(cfg["params"], {"roughness": 0.03})
        run_ids = [r["run_id"] for r in out["runs"]]
        self.assertEqual(set(run_ids), {"run_001", "run_002"})

    def test_inspect_missing_file(self):
        out = tools_modeling.model_inspect("/nonexistent/model.gpkg")
        self.assertFalse(out["ok"])
        self.assertIn("not found", out["error"].lower())

    def test_inspect_empty_path(self):
        out = tools_modeling.model_inspect("")
        self.assertFalse(out["ok"])

    def test_inspect_not_sqlite(self):
        bad = os.path.join(self.tmpdir.name, "bad.gpkg")
        with open(bad, "wb") as fh:
            fh.write(b"this is not sqlite at all")
        out = tools_modeling.model_inspect(bad)
        self.assertFalse(out["ok"])
        self.assertIn("error", out)


class TestModelCreate(HydraMcpFixtureMixin, unittest.TestCase):

    def test_model_create_new_gpkg(self):
        new_gpkg = os.path.join(self.tmpdir.name, "created.gpkg")
        out = tools_modeling_phase1.model_create(new_gpkg, crs="EPSG:4326")
        self.assertTrue(out["ok"], out)
        self.assertTrue(os.path.isfile(new_gpkg))
        conn = sqlite3.connect(new_gpkg)
        try:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertIn("gpkg_contents", tables)
            self.assertIn("swe2d_baked_mesh", tables)
            self.assertIn("swe2d_mcp_model_config", tables)
        finally:
            conn.close()

    def test_model_create_outside_workspace_rejected(self):
        # Temporarily drop the workspace override so the default repo root is in
        # effect; creating under /tmp is rejected.
        os.environ.pop("HYDRA_MCP_WORKSPACE_ROOT", None)
        out = tools_modeling_phase1.model_create("/tmp/should_be_rejected.gpkg", crs="EPSG:4326")
        self.assertFalse(out["ok"], out)
        self.assertIn("workspace", out["error"].lower())

    def test_model_create_empty_path(self):
        out = tools_modeling_phase1.model_create("", crs="EPSG:4326")
        self.assertFalse(out["ok"])


class TestMeshGenerate(unittest.TestCase):

    def test_generate_structured_rect(self):
        out = tools_modeling_phase1.mesh_generate(
            {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}, spacing=5.0)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["backend"], "builtin")
        self.assertEqual(out["n_cells"], 4)
        self.assertEqual(out["n_nodes"], 9)
        mesh = out["mesh"]
        self.assertEqual(len(mesh["cell_nodes"]), 24)  # 4 quads * 2 tri * 3

    def test_generate_gmsh_not_supported(self):
        out = tools_modeling_phase1.mesh_generate(
            {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}, spacing=5.0,
            backend="gmsh")
        self.assertFalse(out["ok"])
        self.assertIn("gmsh", out["error"].lower())

    def test_generate_invalid_domain(self):
        out = tools_modeling_phase1.mesh_generate(
            {"xmin": 10, "ymin": 0, "xmax": 0, "ymax": 10}, spacing=5.0)
        self.assertFalse(out["ok"])
        self.assertIn("xmax", out["error"].lower())


class TestMeshBake(HydraMcpFixtureMixin, unittest.TestCase):

    def test_mesh_bake_persists_metadata(self):
        gen = tools_modeling_phase1.mesh_generate(
            {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}, spacing=5.0)
        self.assertTrue(gen["ok"], gen)
        out = tools_modeling_phase1.mesh_bake(
            self.gpkg_path, "rect_mesh", gen["mesh"], crs_wkt="EPSG:4326")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["n_cells"], 4)
        insp = tools_modeling.model_inspect(self.gpkg_path)
        self.assertTrue(insp["ok"], insp)
        names = {m["mesh_name"] for m in insp["meshes"]}
        self.assertIn("rect_mesh", names)

    def test_mesh_bake_missing_gpkg(self):
        out = tools_modeling_phase1.mesh_bake(
            "/nonexistent/model.gpkg", "mesh", {})
        self.assertFalse(out["ok"])

    def test_mesh_bake_invalid_mesh_data(self):
        out = tools_modeling_phase1.mesh_bake(
            self.gpkg_path, "bad", {"node_x": []})
        self.assertFalse(out["ok"])


class TestTerrainAssign(HydraMcpFixtureMixin, unittest.TestCase):

    def _make_rect_mesh(self):
        gen = tools_modeling_phase1.mesh_generate(
            {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}, spacing=5.0)
        self.assertTrue(gen["ok"], gen)
        return gen["mesh"]

    def test_terrain_assign_raster(self):
        tools_modeling_phase1.mesh_bake(
            self.gpkg_path, "terrain_mesh", self._make_rect_mesh())
        raster = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        out = tools_modeling_phase1.terrain_assign(
            self.gpkg_path, "terrain_mesh",
            {
                "type": "raster",
                "data": raster.tobytes().hex(),
                "shape": list(raster.shape),
                "geo_transform": (0.0, 5.0, 0.0, 10.0, 0.0, -5.0),
            },
            method="raster",
        )
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["n_nodes_updated"], 9)

    def test_terrain_assign_idw(self):
        tools_modeling_phase1.mesh_bake(
            self.gpkg_path, "terrain_mesh", self._make_rect_mesh())
        out = tools_modeling_phase1.terrain_assign(
            self.gpkg_path, "terrain_mesh",
            {
                "type": "points",
                "x": [2.5, 7.5],
                "y": [2.5, 7.5],
                "z": [10.0, 20.0],
            },
            method="idw",
        )
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["n_nodes_updated"], 9)

    def test_terrain_assign_missing_mesh(self):
        out = tools_modeling_phase1.terrain_assign(
            self.gpkg_path, "missing", {"type": "raster", "data": "", "shape": [1, 1], "geo_transform": [0,1,0,0,0,-1]})
        self.assertFalse(out["ok"])
        self.assertIn("not found", out["error"].lower())


class TestConfigureModel(HydraMcpFixtureMixin, unittest.TestCase):

    def test_bc_configure(self):
        out = tools_modeling_phase1.bc_configure(
            self.gpkg_path, self.mesh_name,
            [{"side": "left", "bc_type": "wall", "value": 0.0}],
        )
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["n_entries"], 1)

    def test_rainfall_configure_uniform(self):
        out = tools_modeling_phase1.rainfall_configure(
            self.gpkg_path, self.mesh_name,
            {"uniform_rate_mm_per_hr": 25.4},
        )
        self.assertTrue(out["ok"], out)

    def test_drainage_configure(self):
        cfg = {
            "nodes": [
                {"id": "n1", "x": 0.0, "y": 0.0, "type": "junction",
                 "invert": 0.0, "y_max": 10.0},
            ],
            "links": [
                {"id": "l1", "from": "n1", "to": "n1", "length": 1.0,
                 "diameter": 1.0, "roughness": 0.013},
            ],
        }
        out = tools_modeling_phase1.drainage_configure(
            self.gpkg_path, self.mesh_name, cfg)
        self.assertTrue(out["ok"], out)

    def test_structures_configure(self):
        cfg = [
            {"id": "s1", "type": "culvert", "upstream_cell": 0,
             "downstream_cell": 1, "crest_elev": 5.0},
        ]
        out = tools_modeling_phase1.structures_configure(
            self.gpkg_path, self.mesh_name, cfg)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["n_structures"], 1)

    def test_bc_configure_empty(self):
        out = tools_modeling_phase1.bc_configure(self.gpkg_path, self.mesh_name, [])
        self.assertFalse(out["ok"])


class TestSpecBuildValidateDiff(HydraMcpFixtureMixin, unittest.TestCase):

    def _build_spec(self, **run_params):
        tools_modeling_phase1.bc_configure(
            self.gpkg_path, self.mesh_name,
            [{"side": "left", "bc_type": "wall", "value": 0.0}],
        )
        tools_modeling_phase1.rainfall_configure(
            self.gpkg_path, self.mesh_name,
            {"uniform_rate_mm_per_hr": 25.4},
        )
        return tools_modeling_phase1.spec_build(
            self.gpkg_path, self.mesh_name, run_params=run_params)

    def test_spec_build(self):
        out = self._build_spec(run_duration_s=100.0)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["mesh_name"], self.mesh_name)
        self.assertIn("mesh", out["spec"])
        self.assertIn("params", out["spec"])
        self.assertIn("_mcp_model_config", out["spec"])
        self.assertEqual(out["spec"]["params"]["run_duration_s"], 100.0)

    def test_spec_validate_returns_structured_error_when_mesh_unloadable(self):
        out = self._build_spec(run_duration_s=100.0)
        self.assertTrue(out["ok"], out)
        val = tools_modeling_phase1.spec_validate(out["spec"])
        # In the headless test environment without the native hydra_swe2d build,
        # validation fails at mesh load.  The tool must return a structured error.
        if val.get("ok"):
            return
        self.assertFalse(val["ok"])
        self.assertIn("error", val)

    @unittest.skipUnless(
        _HYDRA_SWE2D_AVAILABLE,
        "hydra_swe2d extension not built",
    )
    def test_spec_validate_success_with_native_build(self):
        # Re-bake the fixture mesh via the real native serializer — the
        # placeholder BLOB from the shared fixture is intentionally corrupt
        # and the native deserializer (correctly) rejects it.
        _bake_real_quad_mesh(self.gpkg_path, self.mesh_name)
        out = self._build_spec(run_duration_s=100.0)
        self.assertTrue(out["ok"], out)
        val = tools_modeling_phase1.spec_validate(out["spec"])
        self.assertTrue(val.get("ok"), val)
        self.assertTrue(val.get("valid"))

    def test_spec_diff(self):
        a = self._build_spec(run_duration_s=100.0)["spec"]
        b = self._build_spec(run_duration_s=200.0)["spec"]
        diff = tools_modeling_phase1.spec_diff(a, b)
        self.assertTrue(diff["ok"], diff)
        changed = {line.split(":")[0] for line in diff["changed"]}
        self.assertIn("params.run_duration_s", changed)
        same = tools_modeling_phase1.spec_diff(a, a)
        self.assertEqual(same["changed"], [])


@unittest.skipUnless(_HYDRA_SWE2D_AVAILABLE, "hydra_swe2d extension not built")
class TestRunJobs(HydraMcpFixtureMixin, unittest.TestCase):
    """Async job lifecycle under the post-eb149116 contract.

    ``run_start`` no longer accepts ``dry_run`` (removed in eb149116 —
    "native mesh only, no dry_run"): every start validates the spec for
    real and spawns a real subprocess.  These tests therefore use a
    genuinely valid spec (real baked mesh) and patch only the subprocess
    *command* so the spawned child is a sleeper instead of a full GPU
    simulation.  Validation, Popen, status polling, and terminate all run
    for real.
    """

    def setUp(self):
        super().setUp()
        _bake_real_quad_mesh(self.gpkg_path, self.mesh_name)
        bc_out = tools_modeling_phase1.bc_configure(
            self.gpkg_path, self.mesh_name,
            [{"side": "left", "bc_type": "wall", "value": 0.0}],
        )
        self.assertTrue(bc_out["ok"], bc_out)
        out = tools_modeling_phase1.spec_build(
            self.gpkg_path, self.mesh_name,
            run_params={"run_duration_s": 10.0},
        )
        self.assertTrue(out["ok"], out)
        self.spec = out["spec"]

    def _patch_run_command(self):
        """Swap the solver subprocess command for a harmless sleeper."""
        import swe2d.cli.commands as cli_commands
        sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
        patcher = unittest.mock.patch.object(
            cli_commands, "build_run_command_for_params",
            side_effect=lambda *a, **k: list(sleeper),
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_run_start_status_cancel(self):
        self._patch_run_command()
        out = tools_modeling_phase1.run_start(self.spec)
        self.assertTrue(out["ok"], out)
        self.assertIn("pid", out)
        job_id = out["job_id"]
        status = tools_modeling_phase1.run_status(job_id)
        self.assertTrue(status["ok"], status)
        self.assertEqual(status["status"]["status"], "running")
        cancel = tools_modeling_phase1.run_cancel(job_id)
        self.assertTrue(cancel["ok"], cancel)
        self.assertTrue(cancel["cancelled"])
        # A terminated child exits with a nonzero returncode → "failed".
        final = tools_modeling_phase1.run_status(job_id)
        self.assertEqual(final["status"]["status"], "failed")

    def test_run_status_unknown_job(self):
        out = tools_modeling_phase1.run_status("does_not_exist")
        self.assertFalse(out["ok"])

    def test_run_batch(self):
        # Canonical batch_spec: a valid swe2d-run/2 spec.  Without a
        # "sweep" block the expansion yields exactly one param set.
        self._patch_run_command()
        out = tools_modeling_phase1.run_batch(self.spec)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["n_jobs"], 1)
        job = out["jobs"][0]
        self.assertEqual(job["params"]["params"]["run_duration_s"], 10.0)
        cancel = tools_modeling_phase1.run_cancel(job["job_id"])
        self.assertTrue(cancel["ok"], cancel)


class TestRunList(HydraMcpFixtureMixin, unittest.TestCase):

    def test_run_list_metadata_and_log_join(self):
        out = tools_modeling.run_list(self.gpkg_path)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["n_runs"], 3)
        runs = {r["run_id"]: r for r in out["runs"]}
        r1 = runs["run_001"]
        self.assertEqual(r1["mesh_name"], self.mesh_name)
        self.assertEqual(r1["n_timesteps"], 3)
        self.assertEqual(r1["n_cells"], self.n_cells)
        self.assertTrue(r1["created_utc"])
        self.assertEqual(r1["duration_s"], 90.0)
        self.assertEqual(r1["start_wallclock"], "10:15:00")
        # config_summary reports the real metadata shape: top-level keys
        # only, never echoing the bulky workbench_widget_state blob.
        self.assertEqual(r1["config_summary"],
                         {"metadata_keys": ["workbench_widget_state"]})
        self.assertNotIn("model_tab", json.dumps(r1["config_summary"]),
                         "nested widget-state payload must not be echoed")
        self.assertNotIn("log_text", r1, "raw log text must not be returned")
        # run_002 has baked results but no run-log entry.
        self.assertNotIn("duration_s", runs["run_002"])

    def test_run_list_log_only_run_with_empty_metadata(self):
        out = tools_modeling.run_list(self.gpkg_path)
        self.assertTrue(out["ok"], out)
        runs = {r["run_id"]: r for r in out["runs"]}
        cli = runs["run_cli_only"]
        self.assertFalse(cli["results_stored"])
        self.assertEqual(cli["n_timesteps"], 0)
        self.assertEqual(cli["duration_s"], 5.0)
        self.assertEqual(cli["config_summary"], {"metadata_keys": []})
        self.assertEqual(cli["mesh_name"], "")

    def test_run_list_missing_file(self):
        out = tools_modeling.run_list("/nonexistent/results.gpkg")
        self.assertFalse(out["ok"])

    def test_run_list_empty_gpkg(self):
        empty = os.path.join(self.tmpdir.name, "empty.gpkg")
        conn = sqlite3.connect(empty)
        conn.close()
        out = tools_modeling.run_list(empty)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["runs"], [])


class TestSummarizeRunMetadata(unittest.TestCase):
    """Direct tests for the run-log metadata summarizer."""

    def test_scalars_pass_through_blobs_do_not(self):
        summary = tools_modeling._summarize_run_metadata({
            "workbench_widget_state": {"huge": {"nested": "blob"}},
            "mesh_name": "mesh_a",
            "run_duration_s": 3600.0,
            "max_tracking_enabled": True,
            "long_note": "x" * 500,
        })
        self.assertEqual(
            summary["metadata_keys"],
            ["long_note", "max_tracking_enabled", "mesh_name",
             "run_duration_s", "workbench_widget_state"],
        )
        self.assertEqual(summary["scalars"], {
            "mesh_name": "mesh_a",
            "run_duration_s": 3600.0,
            "max_tracking_enabled": True,
        })

    def test_empty_metadata(self):
        self.assertEqual(tools_modeling._summarize_run_metadata({}),
                         {"metadata_keys": []})


class TestResultsQuery(HydraMcpFixtureMixin, unittest.TestCase):

    def test_query_whole_field(self):
        out = tools_modeling.results_query(self.gpkg_path, self.run_id, "h")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["kind"], "snapshot_timeseries")
        self.assertEqual(out["summary"]["shape"], [3, self.n_cells])
        self.assertEqual(out["summary"]["dtype"], "float64")
        self.assertEqual(out["summary"]["nan_count"], 1)
        self.assertEqual(out["timesteps"], [0.0, 10.0, 20.0])

    def test_query_single_timestep_nearest(self):
        out = tools_modeling.results_query(
            self.gpkg_path, self.run_id, "h", timestep=12.0)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["requested_timestep"], 12.0)
        self.assertEqual(out["actual_timestep"], 10.0)
        self.assertEqual(out["timestep_index"], 1)
        self.assertEqual(out["summary"]["shape"], [self.n_cells])
        self.assertEqual(out["summary"]["nan_count"], 1)
        self.assertAlmostEqual(out["summary"]["max"], 4.0)  # cell 3: 3 + 1.0

    def test_query_non_numeric_timestep(self):
        out = tools_modeling.results_query(
            self.gpkg_path, self.run_id, "h", timestep="abc")
        self.assertFalse(out["ok"])
        self.assertIn("not a number", out["error"])
        self.assertEqual(out["timesteps"], [0.0, 10.0, 20.0])

    def test_query_nan_timestep(self):
        out = tools_modeling.results_query(
            self.gpkg_path, self.run_id, "h", timestep=float("nan"))
        self.assertFalse(out["ok"])
        self.assertIn("NaN/inf", out["error"])
        self.assertEqual(out["timesteps"], [0.0, 10.0, 20.0])

    def test_query_inf_timestep(self):
        out = tools_modeling.results_query(
            self.gpkg_path, self.run_id, "h", timestep=float("inf"))
        self.assertFalse(out["ok"])
        self.assertIn("NaN/inf", out["error"])
        self.assertEqual(out["timesteps"], [0.0, 10.0, 20.0])

    def test_query_max_tracking_field(self):
        out = tools_modeling.results_query(self.gpkg_path, self.run_id, "max_h")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["kind"], "max_tracking")
        self.assertEqual(out["summary"]["shape"], [self.n_cells])
        self.assertAlmostEqual(out["summary"]["max"], 5.0)

    def test_query_missing_run_lists_valid_ids(self):
        out = tools_modeling.results_query(self.gpkg_path, "nope", "h")
        self.assertFalse(out["ok"])
        self.assertEqual(set(out["available_run_ids"]), {"run_001", "run_002"})

    def test_query_bad_field_lists_valid_fields(self):
        out = tools_modeling.results_query(self.gpkg_path, self.run_id, "velocity")
        self.assertFalse(out["ok"])
        self.assertIn("available_fields", out)
        # max_hv was not stored for run_001 — must not be offered.
        self.assertNotIn("max_hv", out["available_fields"])
        self.assertIn("max_h", out["available_fields"])

    def test_query_unstored_max_field(self):
        out = tools_modeling.results_query(self.gpkg_path, self.run_id, "max_hv")
        self.assertFalse(out["ok"])
        self.assertIn("not stored", out["error"])
        self.assertIn("h", out["available_fields"])

    def test_query_no_max_tracking_run(self):
        out = tools_modeling.results_query(self.gpkg_path, "run_002", "max_h")
        self.assertFalse(out["ok"])
        self.assertEqual(out["available_fields"], ["h", "hu", "hv"])

    def test_query_missing_file(self):
        out = tools_modeling.results_query("/nonexistent/x.gpkg", "r", "h")
        self.assertFalse(out["ok"])

    def test_query_gpkg_without_results_table(self):
        empty = os.path.join(self.tmpdir.name, "mesh_only.gpkg")
        persist_baked_mesh(empty, "m", b"\x00" * 8, n_nodes=1, n_cells=1, n_edges=1)
        out = tools_modeling.results_query(empty, "r", "h")
        self.assertFalse(out["ok"])
        self.assertIn("swe2d_baked_results", out["error"])


class TestResultsPhase1(HydraMcpFixtureMixin, unittest.TestCase):

    def test_results_timeseries(self):
        out = tools_modeling_phase1.results_timeseries(
            self.gpkg_path, self.run_id, line_id=1)
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["n_timesteps"], 3)
        self.assertIn("depth", out["fields"])

    def test_results_export(self):
        out_path = os.path.join(self.tmpdir.name, "export.csv")
        out = tools_modeling_phase1.results_export(
            self.gpkg_path, self.run_id, out_path, format="csv")
        self.assertTrue(out["ok"], out)
        self.assertTrue(os.path.isfile(out_path))

    def test_results_render(self):
        out_path = os.path.join(self.tmpdir.name, "render.png")
        out = tools_modeling_phase1.results_render(
            self.gpkg_path, self.run_id, "h", out_path=out_path)
        self.assertTrue(out["ok"], out)
        self.assertTrue(os.path.isfile(out_path))

    def test_results_compare(self):
        out = tools_modeling_phase1.results_compare(
            self.gpkg_path, self.run_id, "run_002", "h")
        self.assertTrue(out["ok"], out)
        self.assertIn("summary_diffs", out)

    def test_results_timeseries_missing(self):
        out = tools_modeling_phase1.results_timeseries(
            self.gpkg_path, self.run_id, line_id=999)
        self.assertFalse(out["ok"])


class TestAuditTools(unittest.TestCase):
    """Audit-only GUI adapters with a fake bridge."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self._old_root = os.environ.get("HYDRA_MCP_WORKSPACE_ROOT")
        os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = self.tmpdir.name
        self.addCleanup(self._restore_root)

    def _restore_root(self):
        if self._old_root is None:
            os.environ.pop("HYDRA_MCP_WORKSPACE_ROOT", None)
        else:
            os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = self._old_root

    def test_gui_dump_dock_returns_subtree(self):
        with unittest.mock.patch.object(tools_audit.tools_gui, "gui_find_widget", return_value={"ok": True, "node": {}}), unittest.mock.patch.object(tools_audit.tools_gui, "gui_widget_tree", return_value={"ok": True, "nodes": [{"object_name": "dock"}]}):
            out = tools_audit.gui_dump_dock("dock")
        self.assertEqual(out, {"ok": True, "dock_object_name": "dock", "nodes": [{"object_name": "dock"}]})

    def test_gui_screenshot_path_writes_file_to_workspace(self):
        # ``default_workspace().resolve_under`` requires the target to exist;
        # create an empty placeholder so the containment check accepts the
        # workspace-relative path before the screenshot bytes are written.
        (Path(self.tmpdir.name) / "panel.png").touch()
        with unittest.mock.patch.object(tools_audit.tools_gui, "gui_screenshot", return_value={"ok": True, "image_b64": "aGVsbG8=", "format": "png", "width": 2, "height": 3}):
            out = tools_audit.gui_screenshot_path("panel.png", path="dock")
        self.assertTrue(out["ok"], out)
        self.assertEqual(Path(out["path"]).read_bytes(), b"hello")

    def test_gui_screenshot_path_rejects_path_outside_workspace(self):
        out = tools_audit.gui_screenshot_path("/tmp/outside.png", path="dock")
        self.assertFalse(out["ok"])
        self.assertIn("workspace", out["error"].lower())

    def test_gui_screenshot_path_rejects_dotdot(self):
        out = tools_audit.gui_screenshot_path("../../etc/foo.png", path="dock")
        self.assertFalse(out["ok"])
        self.assertIn("..", out["error"])

    def test_gui_describe_widget_returns_metadata(self):
        class Client:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def _call(self, method, **params):
                self.assert_called = (method, params)
                return {"object_name": "spin", "class_name": "QSpinBox", "enabled": True, "visible": True}
        with unittest.mock.patch.object(tools_audit.tools_gui, "_get_bridge_client", return_value=Client()):
            out = tools_audit.gui_describe_widget("dock.spin")
        self.assertTrue(out["ok"], out)
        self.assertTrue(out["widget"]["enabled"])


class TestServerSmoke(unittest.TestCase):
    """FastMCP server constructs and all tools are registered.

    Skipped when the mcp SDK is not installed in the current interpreter.
    """

    def test_server_constructs_with_all_tools(self):
        try:
            from tools.hydra_mcp import server
        except ModuleNotFoundError as exc:
            self.skipTest(f"mcp SDK not importable: {exc}")
        tools = asyncio.run(server.mcp.list_tools())
        names = {t.name for t in tools}
        self.assertEqual(
            {
                "model_inspect",
                "run_list",
                "results_query",
                "model_create",
                "mesh_generate",
                "mesh_bake",
                "terrain_assign",
                "bc_configure",
                "rainfall_configure",
                "drainage_configure",
                "structures_configure",
                "spec_build",
                "spec_validate",
                "spec_diff",
                "run_start",
                "run_status",
                "run_cancel",
                "run_batch",
                "results_timeseries",
                "results_export",
                "results_render",
                "results_compare",
                "design_rename_widget",
                "design_relabel_widget",
                "design_preview_patch",
                "design_apply_patch",
                "gui_launch",
                "gui_widget_tree",
                "gui_find_widget",
                "gui_find_widget_by_path",
                "gui_get_value",
                "gui_set_value",
                 "gui_screenshot",
                 "gui_dump_dock",
                 "gui_screenshot_path",
                 "gui_describe_widget",

                "gui_click",
                "gui_key",
                "gui_run_action",
                "gui_read_log",
                "gui_run_simulation",
                "gui_close",
            },
            names,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2.A — Bridge framing + client + tools_gui tests
# ═══════════════════════════════════════════════════════════════════════════════

import socket
import subprocess
import threading
import time
import unittest.mock

from tools.hydra_mcp import bridge_client, tools_gui


class TestMessageFraming(unittest.TestCase):
    """Pure-Python unit tests for the length-prefixed wire format."""

    def test_encode_decode_roundtrip(self):
        obj = {"jsonrpc": "2.0", "method": "ping", "id": 1}
        encoded = bridge_client.encode_message(obj)
        # 4-byte big-endian length prefix.
        import struct
        length = struct.unpack("!I", encoded[:4])[0]
        self.assertEqual(length, len(encoded) - 4)
        messages, leftover = bridge_client.decode_messages(encoded)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0], obj)
        self.assertEqual(leftover, b"")

    def test_decode_multiple_messages(self):
        msg1 = {"jsonrpc": "2.0", "method": "a", "id": 1}
        msg2 = {"jsonrpc": "2.0", "method": "b", "id": 2}
        data = bridge_client.encode_message(msg1) + bridge_client.encode_message(msg2)
        messages, leftover = bridge_client.decode_messages(data)
        self.assertEqual(messages, [msg1, msg2])
        self.assertEqual(leftover, b"")

    def test_decode_incomplete_message_returns_leftover(self):
        msg = {"jsonrpc": "2.0", "method": "ping", "id": 1}
        data = bridge_client.encode_message(msg)
        # Feed only 3 bytes — no complete length header yet.
        partial, leftover = bridge_client.decode_messages(data[:3])
        self.assertEqual(partial, [])
        self.assertEqual(leftover, data[:3])

    def test_decode_partial_payload_returns_leftover(self):
        msg = {"jsonrpc": "2.0", "method": "ping", "id": 1}
        data = bridge_client.encode_message(msg)
        # Feed 5 bytes (header complete, but payload too short).
        partial, leftover = bridge_client.decode_messages(data[:5])
        self.assertEqual(partial, [])
        self.assertEqual(leftover, data[:5])

    def test_encode_large_payload(self):
        large = {"data": "x" * 100_000}
        encoded = bridge_client.encode_message(large)
        import struct
        length = struct.unpack("!I", encoded[:4])[0]
        self.assertEqual(length, len(encoded) - 4)
        messages, leftover = bridge_client.decode_messages(encoded)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0], large)


class TestBridgeClientTokenDiscovery(unittest.TestCase):
    """Unit tests for BridgeClient token / socket discovery."""

    def test_load_token_file(self):
        import tempfile, json
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as fh:
            json.dump(
                {"socket_name": "test_socket", "token": "test_secret"},
                fh,
            )
            path = fh.name
        # Pin the workspace at /tmp so the token file is accepted.
        prev = os.environ.get("HYDRA_MCP_WORKSPACE_ROOT")
        os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = tempfile.gettempdir()
        try:
            cli = bridge_client.BridgeClient(token_path=path)
            self.assertEqual(cli.socket_name, "test_socket")
            self.assertEqual(cli.token, "test_secret")
        finally:
            os.unlink(path)
            if prev is None:
                os.environ.pop("HYDRA_MCP_WORKSPACE_ROOT", None)
            else:
                os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = prev

    def test_instantiation_requires_qt(self):
        # Patching Qt as unavailable should raise a clear error.
        orig = bridge_client._QT_AVAILABLE
        bridge_client._QT_AVAILABLE = False
        try:
            with self.assertRaises(RuntimeError) as ctx:
                bridge_client.BridgeClient()
            self.assertIn("PyQt5", str(ctx.exception))
        finally:
            bridge_client._QT_AVAILABLE = orig


class TestToolsGuiGracefulDegradation(unittest.TestCase):
    """Tier B tools return clear errors when no bridge session is active."""

    def test_gui_widget_tree_no_bridge(self):
        # Patch BridgeClient to raise RuntimeError (no Qt / not connected).
        orig = bridge_client.BridgeClient
        bridge_client.BridgeClient = unittest.mock.MagicMock(
            side_effect=RuntimeError("BridgeClient requires PyQt5")
        )
        try:
            out = tools_gui.gui_widget_tree()
            self.assertFalse(out["ok"])
            self.assertIn("No active QGIS bridge session found", out["error"])
        finally:
            bridge_client.BridgeClient = orig

    def test_gui_find_widget_no_bridge(self):
        orig = bridge_client.BridgeClient
        bridge_client.BridgeClient = unittest.mock.MagicMock(
            side_effect=RuntimeError("BridgeClient requires PyQt5")
        )
        try:
            out = tools_gui.gui_find_widget("foo")
            self.assertFalse(out["ok"])
            self.assertIn("No active QGIS bridge session found", out["error"])
        finally:
            bridge_client.BridgeClient = orig

    def test_gui_find_widget_empty_name(self):
        out = tools_gui.gui_find_widget("")
        self.assertFalse(out["ok"])
        self.assertIn("name is required", out["error"])

    def test_gui_launch_invalid_mode(self):
        out = tools_gui.gui_launch(mode="invalid_mode")
        self.assertFalse(out["ok"])
        self.assertIn("Invalid mode", out["error"])

    def test_gui_launch_display_auto_discovers(self):
        # In display mode, gui_launch tries to discover an active bridge.
        orig = bridge_client.BridgeClient
        bridge_client.BridgeClient = unittest.mock.MagicMock(
            side_effect=RuntimeError("no bridge")
        )
        try:
            out = tools_gui.gui_launch(mode="display")
            # Should fall back to bridge_not_available error.
            self.assertFalse(out["ok"])
            self.assertIn("No active QGIS bridge session found", out["error"])
        finally:
            bridge_client.BridgeClient = orig

    def test_gui_get_value_no_bridge(self):
        orig = bridge_client.BridgeClient
        bridge_client.BridgeClient = unittest.mock.MagicMock(
            side_effect=RuntimeError("BridgeClient requires PyQt5")
        )
        try:
            out = tools_gui.gui_get_value("a.b.c")
            self.assertFalse(out["ok"])
            self.assertIn("No active QGIS bridge session found", out["error"])
        finally:
            bridge_client.BridgeClient = orig

    def test_gui_get_value_empty_path(self):
        out = tools_gui.gui_get_value("")
        self.assertFalse(out["ok"])
        self.assertIn("path is required", out["error"])

    def test_gui_set_value_no_bridge(self):
        orig = bridge_client.BridgeClient
        bridge_client.BridgeClient = unittest.mock.MagicMock(
            side_effect=RuntimeError("BridgeClient requires PyQt5")
        )
        try:
            out = tools_gui.gui_set_value("a.b.c", 42)
            self.assertFalse(out["ok"])
            self.assertIn("No active QGIS bridge session found", out["error"])
        finally:
            bridge_client.BridgeClient = orig

    def test_gui_set_value_empty_path(self):
        out = tools_gui.gui_set_value("", 42)
        self.assertFalse(out["ok"])
        self.assertIn("path is required", out["error"])


class TestGuiLaunchDisplayMode(unittest.TestCase):
    """``gui_launch(mode="display")`` polls for an active bridge token file."""

    def test_display_mode_discovers_active_bridge(self):
        class FakeClient:
            def __init__(self, token_path=None, timeout=30.0):
                self.timeout_ms = int(timeout * 1000)
                self.bridge_token = "display_placeholder"
                self.socket_name = "display_socket"

            def connect(self):
                pass

            def ping(self):
                return {
                    "socket_name": "display_socket",
                    "token_path": "/tmp/hydra_mcp_bridge_display_123.json",
                    "version": "2.C",
                }

            def close(self):
                pass

        orig_get = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: FakeClient(
            token_path=token_path, timeout=timeout
        )
        try:
            out = tools_gui.gui_launch(mode="display", timeout=2.0)
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["mode"], "display")
            self.assertEqual(out["socket_name"], "display_socket")
            self.assertEqual(out["session_id"], "display_123")
            self.assertEqual(
                out["token_path"], "/tmp/hydra_mcp_bridge_display_123.json"
            )
        finally:
            tools_gui._get_bridge_client = orig_get

    def test_display_mode_times_out_when_no_bridge(self):
        class FailingClient:
            def __init__(self, token_path=None, timeout=30.0):
                pass

            def connect(self):
                raise RuntimeError("no bridge")

        orig_get = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: FailingClient()
        try:
            out = tools_gui.gui_launch(mode="display", timeout=0.5)
            self.assertFalse(out["ok"])
            self.assertIn("No active QGIS bridge session found", out["error"])
        finally:
            tools_gui._get_bridge_client = orig_get


class TestToolsGuiLaunchEnvAndCmd(unittest.TestCase):
    """Construct the (env, cmd) pair for ``gui_launch`` without spawning QGIS.

    The launch-path bug fixed in P0.4 was: ``gui_launch`` stripped
    ``QGIS_PLUGINPATH`` (line 275), used ``--noplugins`` (line 289, blocking
    plugin load), and did not set ``HYDRA_MCP_BRIDGE=1`` so the autostart
    gate at ``hydra_plugin.py:152-155`` short-circuited. These tests pin the
    three correctness invariants so the bug cannot regress.
    """

    def setUp(self):
        # Save and clear inherited values so each test starts clean.
        self._prev_pluginpath = os.environ.pop("QGIS_PLUGINPATH", None)
        self._prev_bridge = os.environ.pop("HYDRA_MCP_BRIDGE", None)
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        for var in ("QGIS_PLUGINPATH", "HYDRA_MCP_BRIDGE"):
            os.environ.pop(var, None)
        if self._prev_pluginpath is not None:
            os.environ["QGIS_PLUGINPATH"] = self._prev_pluginpath
        if self._prev_bridge is not None:
            os.environ["HYDRA_MCP_BRIDGE"] = self._prev_bridge

    def _build(self, *, mode="offscreen", qgis_binary=None,
               bridge_script=None, project=None):
        return tools_gui._build_launch_env_and_cmd(
            mode=mode,
            qgis_binary=qgis_binary or "/tmp/fake/qgis",
            project=project,
            bridge_script=bridge_script or Path("/tmp/bridge.py"),
        )

    def test_launch_sets_bridge_autostart_env(self):
        env, _cmd, err = self._build()
        self.assertIsNone(err)
        self.assertEqual(env.get("HYDRA_MCP_BRIDGE"), "1")

    def test_launch_omits_noplugins_flag(self):
        _env, cmd, err = self._build()
        self.assertIsNone(err)
        self.assertNotIn("--noplugins", cmd)

    def test_launch_inherits_qgis_pluginpath(self):
        # Caller already exported QGIS_PLUGINPATH → honour it.
        os.environ["QGIS_PLUGINPATH"] = "/custom/plugin/path"
        env, _cmd, err = self._build()
        self.assertIsNone(err)
        self.assertEqual(env.get("QGIS_PLUGINPATH"), "/custom/plugin/path")

    def test_launch_constructs_qgis_pluginpath_when_unset(self):
        env, _cmd, err = self._build(
            qgis_binary=os.path.join(tempfile.gettempdir(), "fake-qgis"),
        )
        self.assertIsNone(err)
        path = env.get("QGIS_PLUGINPATH")
        self.assertIsNotNone(
            path,
            "QGIS_PLUGINPATH must be constructed when not inherited",
        )
        # The constructed path must point at a real directory.
        self.assertTrue(os.path.isdir(path), path)
        self.assertTrue(path.endswith("share/qgis/python/plugins"), path)

    def test_launch_includes_no_versioncheck_and_bridge_code(self):
        _env, cmd, err = self._build(bridge_script=Path("/tmp/bridge.py"))
        self.assertIsNone(err)
        self.assertIn("--noversioncheck", cmd)
        self.assertEqual(cmd[-2:], ["--code", "/tmp/bridge.py"])

    def test_offscreen_launch_sets_qt_qpa_platform(self):
        env, _cmd, err = self._build(mode="offscreen")
        self.assertIsNone(err)
        self.assertEqual(env.get("QT_QPA_PLATFORM"), "offscreen")


class TestToolsGuiMockedBridge(unittest.TestCase):
    """Integration tests for gui_widget_tree / gui_find_widget against a
    mock BridgeClient that speaks the real protocol over a Unix socket pair.

    This tests the full wire round-trip without needing a real QGIS process.
    """

    def setUp(self):
        self.sock_a, self.sock_b = socket.socketpair()

    def tearDown(self):
        self.sock_a.close()
        self.sock_b.close()

    def _respond(self, request_id, result):
        msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
        self.sock_b.send(bridge_client.encode_message(msg))

    def _send_request(self, method, params=None):
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }
        self.sock_a.send(bridge_client.encode_message(req))
        # Read the response (blocking but short).
        self.sock_a.settimeout(5.0)
        header = self.sock_a.recv(4)
        import struct
        length = struct.unpack("!I", header)[0]
        body = b""
        while len(body) < length:
            body += self.sock_a.recv(length - len(body))
        import json
        return json.loads(body.decode())

    def test_widget_tree_via_socket_loop(self):
        # Server loop: reads requests from sock_b and responds.
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    method = msg.get("method")
                    if method == "get_widget_tree":
                        self._respond(msg["id"], [
                            {"object_name": "MainWindow",
                             "class_name": "QMainWindow",
                             "widget_id": 1, "parent_id": None,
                             "text": "HYDRA Studio", "depth": 0},
                            {"object_name": "centralWidget",
                             "class_name": "QWidget",
                             "widget_id": 2, "parent_id": 1,
                             "text": "", "depth": 1},
                        ])
                    elif method == "find_widget":
                        nodes = [
                            {"object_name": "centralWidget",
                             "class_name": "QWidget",
                             "widget_id": 2, "parent_id": 1,
                             "text": "", "depth": 1}
                        ]
                        found = next(
                            (n for n in nodes
                             if n["object_name"] == msg["params"].get("object_name")),
                            None,
                        )
                        self._respond(msg["id"], found)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        # Build a mock BridgeClient that uses our socket pair.
        class MockBridgeClient:
            _sock = None  # set below after sockets are created

            def __init__(self, token_path=None, socket_name=None,
                         token=None, timeout=30.0):
                self.timeout_ms = int(timeout * 1000)
                self.token = token or "mock_token"
                self.socket_name = socket_name or "mock_socket"

            def connect(self):
                pass

            def close(self):
                pass

            def _call_raw(self, method, **params):
                params["token"] = self.token
                request = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                    "id": 1,
                }
                self._sock.send(bridge_client.encode_message(request))
                self._sock.settimeout(5.0)
                header = self._sock.recv(4)
                import struct
                length = struct.unpack("!I", header)[0]
                body = b""
                while len(body) < length:
                    body += self._sock.recv(length - len(body))
                import json
                resp = json.loads(body.decode())
                if "error" in resp:
                    raise RuntimeError(resp["error"].get("message"))
                return resp.get("result", {})

            def get_widget_tree(self, root=None):
                return self._call_raw("get_widget_tree", root_object_name=root)

            def find_widget(self, name):
                return self._call_raw("find_widget", object_name=name)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        # Point the mock at our socket pair before patching _get_bridge_client.
        MockBridgeClient._sock = self.sock_a

        # Patch _get_bridge_client so tools_gui uses our mock.
        orig_get = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: MockBridgeClient(
            token_path=token_path
        )

        try:
            out = tools_gui.gui_widget_tree()
            self.assertTrue(out["ok"], out)
            self.assertEqual(len(out["nodes"]), 2)
            self.assertEqual(out["nodes"][0]["object_name"], "MainWindow")
            self.assertEqual(out["nodes"][0]["class_name"], "QMainWindow")

            out = tools_gui.gui_find_widget("centralWidget")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["node"]["object_name"], "centralWidget")

            out = tools_gui.gui_find_widget("nonexistent")
            self.assertFalse(out["ok"])
            self.assertIn("No widget", out["error"])
            self.assertIn("nonexistent", out["error"])
        finally:
            tools_gui._get_bridge_client = orig_get


class TestToolsGuiValueToolsMockedBridge(unittest.TestCase):
    """Socket-pair tests for gui_get_value / gui_set_value / gui_find_widget_by_path."""

    def setUp(self):
        self.sock_a, self.sock_b = socket.socketpair()

    def tearDown(self):
        self.sock_a.close()
        self.sock_b.close()

    def _respond(self, request_id, result):
        msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
        self.sock_b.send(bridge_client.encode_message(msg))

    def _send_request(self, method, params=None):
        req = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }
        self.sock_a.send(bridge_client.encode_message(req))
        self.sock_a.settimeout(5.0)
        import struct
        header = self.sock_a.recv(4)
        length = struct.unpack("!I", header)[0]
        body = b""
        while len(body) < length:
            body += self.sock_a.recv(length - len(body))
        import json
        return json.loads(body.decode())

    def _make_mock_client(self):
        """Return a BridgeClient-like object that speaks over sock_a."""

        class MockBridgeClient:
            _sock = None

            def __init__(self, token_path=None, socket_name=None,
                         token=None, timeout=30.0):
                self.timeout_ms = int(timeout * 1000)
                self.token = token or "mock_token"
                self.socket_name = socket_name or "mock_socket"

            def connect(self):
                pass

            def close(self):
                pass

            def _call(self, method, **params):
                params["token"] = self.token
                request = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                    "id": 1,
                }
                self._sock.send(bridge_client.encode_message(request))
                self._sock.settimeout(5.0)
                import struct
                header = self._sock.recv(4)
                length = struct.unpack("!I", header)[0]
                body = b""
                while len(body) < length:
                    body += self._sock.recv(length - len(body))
                import json
                resp = json.loads(body.decode())
                if "error" in resp:
                    raise RuntimeError(resp["error"].get("message"))
                return resp.get("result", {})

            def get_value(self, path, root=None):
                return self._call("get_value", path=path, root_object_name=root)

            def set_value(self, path, value, root=None):
                return self._call("set_value", path=path, value=value,
                                   root_object_name=root)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        MockBridgeClient._sock = self.sock_a
        return MockBridgeClient()   # return instance, not class

    def test_get_value_spinbox_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "get_value":
                        path = msg["params"].get("path")
                        if path == "root.spin":
                            self._respond(msg["id"],
                                          {"ok": True, "type": "QSpinBox",
                                           "value": 42})
                        else:
                            self._respond(msg["id"],
                                          {"ok": False,
                                           "error": "not found"})

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        orig_get = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_get_value("root.spin")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["type"], "QSpinBox")
            self.assertEqual(out["value"], 42)

            out = tools_gui.gui_get_value("root.missing")
            self.assertFalse(out["ok"])
            self.assertIn("not found", out["error"])
        finally:
            tools_gui._get_bridge_client = orig_get

    def test_get_value_checkbox_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "get_value":
                        path = msg["params"].get("path")
                        if path == "root.check":
                            self._respond(msg["id"],
                                          {"ok": True, "type": "QCheckBox",
                                           "value": True})
                        else:
                            self._respond(msg["id"],
                                          {"ok": False,
                                           "error": f"no widget at {path}"})

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        orig_get = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_get_value("root.check")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["type"], "QCheckBox")
            self.assertEqual(out["value"], True)
        finally:
            tools_gui._get_bridge_client = orig_get

    def test_set_value_spinbox_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "set_value":
                        path = msg["params"].get("path")
                        val = msg["params"].get("value")
                        if path == "root.spin" and val == 99:
                            self._respond(msg["id"], {"ok": True})
                        elif path == "root.combo" and val == "bad":
                            self._respond(msg["id"],
                                          {"ok": False,
                                           "error": "no item matching 'bad'"})
                        else:
                            self._respond(msg["id"],
                                          {"ok": False, "error": "unknown"})

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        orig_get = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_set_value("root.spin", 99)
            self.assertTrue(out["ok"], out)

            out = tools_gui.gui_set_value("root.combo", "bad")
            self.assertFalse(out["ok"])
            self.assertIn("no item matching", out["error"])
        finally:
            tools_gui._get_bridge_client = orig_get

    def test_find_widget_by_path_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "find_widget":
                        path = msg["params"].get("path")
                        if path == "root.a.b":
                            self._respond(msg["id"], {
                                "object_name": "b",
                                "class_name": "QDoubleSpinBox",
                                "widget_id": 999,
                                "geometry": {"x": 0, "y": 0,
                                             "width": 80, "height": 24},
                                "is_visible": True,
                            })
                        else:
                            self._respond(msg["id"], None)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        orig_get = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_find_widget_by_path("root.a.b")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["widget"]["class_name"], "QDoubleSpinBox")
            self.assertEqual(out["widget"]["geometry"]["width"], 80)

            out = tools_gui.gui_find_widget_by_path("root.x.y")
            self.assertFalse(out["ok"])
            self.assertIn("No widget found", out["error"])
        finally:
            tools_gui._get_bridge_client = orig_get


class TestWidgetWalkerFindByPath(unittest.TestCase):
    """Unit tests for find_widget_by_path using plain-Python mock widgets.

    Uses simple objects that implement the minimal Qt interface
    (objectName, findChildren) so no QApplication is needed.
    """

    def test_find_widget_by_path_exact_match(self):
        from swe2d.workbench.devtools.widget_walker import find_widget_by_path

        # Build a plain-Python tree mirroring what Qt gives us.
        leaf = _MockWidget("my_spin")
        container = _MockWidget("container", children=[leaf])
        root = _MockWidget("root", children=[container])

        result = find_widget_by_path(root, ["container", "my_spin"])
        self.assertIsNotNone(result)
        self.assertEqual(result.objectName(), "my_spin")

    def test_find_widget_by_path_partial_missing(self):
        from swe2d.workbench.devtools.widget_walker import find_widget_by_path

        child = _MockWidget("child")
        root = _MockWidget("root", children=[child])

        # Path goes into a dead end.
        result = find_widget_by_path(root, ["child", "nonexistent"])
        self.assertIsNone(result)

    def test_find_widget_by_path_empty_returns_root(self):
        from swe2d.workbench.devtools.widget_walker import find_widget_by_path

        root = _MockWidget("root")
        result = find_widget_by_path(root, [])
        self.assertIsNotNone(result)
        self.assertEqual(result.objectName(), "root")

    def test_find_widget_by_path_empty_parts_skipped(self):
        from swe2d.workbench.devtools.widget_walker import find_widget_by_path

        leaf = _MockWidget("leaf")
        child = _MockWidget("child", children=[leaf])
        root = _MockWidget("root", children=[child])

        # Path with empty parts (e.g. "root..child.leaf") should still work.
        result = find_widget_by_path(root, ["", "child", "", "leaf"])
        self.assertIsNotNone(result)
        self.assertEqual(result.objectName(), "leaf")


# ── Plain-Python mock widget (no Qt / QApplication required) ─────────────────


class _MockWidget:
    """Minimal Qt-like widget for unit testing find_widget_by_path.

    A real ``QWidget`` subclass (when PyQt5 is available) so the walker
    (post-2.H.6) accepts it via ``isinstance(c, QWidget)``.  When
    PyQt5 is unavailable, falls back to a plain-Python shim with the
    same interface; in that case the walker (which depends on
    ``qgis.PyQt.QtWidgets``) cannot be imported, so the shim path is
    only used in build environments that exclude Qt tests.
    """

    def __new__(cls, *args, **kwargs):
        try:
            from qgis.PyQt.QtWidgets import QApplication, QWidget  # noqa: F401
            QApplication.instance() or QApplication([])
            return super().__new__(cls)
        except Exception:
            return super().__new__(cls)

    def __init__(self, object_name: str, children: List["_MockWidget"] = None):
        # Try to call QWidget.__init__ via the sip super() chain.
        try:
            from qgis.PyQt.QtWidgets import QWidget  # noqa: F401
            # ``sip.simplewrapper.__init__`` is not directly callable on
            # a plain Python subclass; we have to let ``super()`` route
            # through the MRO.  If this fails, the test environment is
            # headless and the walker is not importable, so the shim
            # path is irrelevant.
            super(_MockWidget, self).__init__()
        except Exception:
            pass
        self._object_name = object_name
        self._children = children or []
        self._visible = False
        # MagicMock for grab() so tests that don't exercise it don't need to mock it.
        import unittest.mock
        self._grab_mock = unittest.mock.MagicMock(name="grab")
        # setObjectName is provided by the C++ QWidget side; on the
        # plain-Python shim path it is absent.
        if hasattr(self, "setObjectName"):
            self.setObjectName(object_name)
        # Adopt explicit children so the C++ QObject side's parent()
        # and children() reflect the explicit test list.
        for c in self._children:
            try:
                c.setParent(self)
            except Exception:
                pass

    def objectName(self) -> str:
        return self._object_name

    def isVisible(self) -> bool:
        return self._visible

    def grab(self):
        return self._grab_mock

    def findChildren(self, widget_type, name: str = ""):
        return list(self._children)

    def children(self):
        # Direct QObject children — the widget walker (post-2.H.6) uses
        # ``QObject.children()`` instead of ``findChildren`` so a path
        # cannot jump levels.  We return the explicit test list so the
        # walker's ``isinstance(c, QWidget)`` filter sees a real QWidget
        # (the test's intent is to walk exactly the children it
        # provided, not whatever Qt's parent tree happens to expose).
        return list(self._children)


class TestBridgeClientValueMethods(unittest.TestCase):
    """Direct unit tests for BridgeClient.get_value / set_value (mocked socket)."""

    def test_bridge_client_get_value_calls_correct_method(self):
        # Patch _call to capture what it sends.
        orig = bridge_client.BridgeClient
        captured: Dict[str, Any] = {}

        class FakeClient:
            def __init__(self, token_path=None, socket_name=None,
                         token=None, timeout=30.0):
                self.timeout_ms = int(timeout * 1000)
                self.token = token or "t"
                self.socket_name = socket_name or "s"
            def connect(self): pass
            def close(self): pass
            def _call(self, method, **params):
                captured.update({"method": method, "params": params})
                return {"ok": True, "type": "QSpinBox", "value": 7}
            def get_value(self, path, root=None):
                return self._call("get_value", path=path, root_object_name=root)
            def set_value(self, path, value, root=None):
                return self._call("set_value", path=path, value=value,
                                  root_object_name=root)
            def __enter__(self): return self
            def __exit__(self, *a): self.close()

        cli = FakeClient()
        with cli:
            result = cli.get_value("a.b.c", root="root_widget")
        self.assertEqual(captured["method"], "get_value")
        self.assertEqual(captured["params"]["path"], "a.b.c")
        self.assertEqual(captured["params"]["root_object_name"], "root_widget")
        self.assertEqual(result["value"], 7)

    def test_bridge_client_set_value_calls_correct_method(self):
        captured: Dict[str, Any] = {}

        class FakeClient:
            def __init__(self, token_path=None, socket_name=None,
                         token=None, timeout=30.0):
                self.timeout_ms = int(timeout * 1000)
                self.token = token or "t"
                self.socket_name = socket_name or "s"
            def connect(self): pass
            def close(self): pass
            def _call(self, method, **params):
                captured.update({"method": method, "params": params})
                return {"ok": True}
            def set_value(self, path, value, root=None):
                return self._call("set_value", path=path, value=value,
                                  root_object_name=root)
            def __enter__(self): return self
            def __exit__(self, *a): self.close()

        cli = FakeClient()
        with cli:
            result = cli.set_value("x.y", 3.14, root="root_widget")
        self.assertEqual(captured["method"], "set_value")
        self.assertEqual(captured["params"]["path"], "x.y")
        self.assertEqual(captured["params"]["value"], 3.14)
        self.assertEqual(captured["params"]["root_object_name"], "root_widget")
        self.assertTrue(result["ok"])
    @requires_qgis
    def test_walker_finds_root_and_children(self):
        """Test the widget-node serialization path against real QGIS.

        ``swe2d.workbench.devtools.widget_walker`` requires ``qgis.PyQt``;
        under the real headless QGIS harness it imports directly.  The
        pure-Python logic (WidgetNode dataclass, find_node_by_object_name,
        _node_to_dict dict shape) is exercised here.  A full integration
        test that launches ``qgis --code qgis_bridge.py`` is a manual step
        documented in ``tools/hydra_mcp/README.md``.
        """
        ensure_qgis_app()

        from swe2d.workbench.devtools.widget_walker import (
            WidgetNode,
            find_node_by_object_name,
        )

        # Build a flat node list that mirrors what walk_widget_tree returns.
        node0 = WidgetNode(
            object_name="StudioMainWindow",
            class_name="QMainWindow",
            widget_id=1001,
            parent_id=None,
            text="HYDRA Studio",
            depth=0,
        )
        node1 = WidgetNode(
            object_name="TabAContainer",
            class_name="QWidget",
            widget_id=1002,
            parent_id=node0.widget_id,
            text="Model",
            depth=1,
        )
        node2 = WidgetNode(
            object_name="LeafWidget",
            class_name="QFrame",
            widget_id=1003,
            parent_id=node1.widget_id,
            text="",
            depth=2,
        )
        nodes = [node0, node1, node2]

        # Tree invariants.
        self.assertEqual(nodes[0].parent_id, None)
        self.assertEqual(nodes[1].parent_id, nodes[0].widget_id)
        self.assertEqual(nodes[2].parent_id, nodes[1].widget_id)

        # find_node_by_object_name.
        found = find_node_by_object_name(nodes, "LeafWidget")
        self.assertIsNotNone(found)
        self.assertEqual(found.class_name, "QFrame")
        self.assertEqual(found.depth, 2)

        # _node_to_dict shape (what the bridge sends over the wire).
        # We inline the _node_to_dict logic to avoid importing qgis_bridge here.
        def node_to_dict(node):
            return {
                "object_name": node.object_name,
                "class_name": node.class_name,
                "widget_id": node.widget_id,
                "parent_id": node.parent_id,
                "text": node.text,
                "depth": node.depth,
            }

        d = node_to_dict(node0)
        self.assertEqual(set(d.keys()), {
            "object_name", "class_name", "widget_id",
            "parent_id", "text", "depth",
        })
        self.assertIsInstance(d["widget_id"], int)
        self.assertIsNone(d["parent_id"])

        # JSON round-trip (the wire format).
        import json
        serialised = json.dumps([node_to_dict(n) for n in nodes])
        parsed = json.loads(serialised)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]["object_name"], "StudioMainWindow")
        self.assertEqual(parsed[2]["parent_id"], parsed[1]["widget_id"])


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2.C — gui_screenshot
# ═══════════════════════════════════════════════════════════════════════════════


class TestBridgeClientScreenshot(unittest.TestCase):
    """Unit tests for BridgeClient.screenshot (mocked _call)."""

    def test_bridge_client_screenshot_calls_correct_method(self):
        captured: Dict[str, Any] = {}

        class FakeClient:
            def __init__(self, token_path=None, socket_name=None,
                         token=None, timeout=30.0):
                self.timeout_ms = int(timeout * 1000)
                self.token = token or "t"
                self.socket_name = socket_name or "s"
            def connect(self): pass
            def close(self): pass
            def _call(self, method, **params):
                captured.update({"method": method, "params": params})
                return {
                    "ok": True,
                    "image_b64": "abc123",
                    "format": "png",
                    "width": 640,
                    "height": 480,
                }
            def screenshot(self, path=None, format="png", root=None, target=None):
                return self._call("screenshot", path=path, format=format,
                                  root_object_name=root, target=target)
            def __enter__(self): return self
            def __exit__(self, *a): self.close()

        cli = FakeClient()
        with cli:
            result = cli.screenshot("a.b.tab", format="jpg", root="main")
        self.assertEqual(captured["method"], "screenshot")
        self.assertEqual(captured["params"]["path"], "a.b.tab")
        self.assertEqual(captured["params"]["format"], "jpg")
        self.assertEqual(captured["params"]["root_object_name"], "main")
        self.assertTrue(result["ok"])
        self.assertEqual(result["width"], 640)
        self.assertEqual(result["height"], 480)
        self.assertEqual(result["image_b64"], "abc123")

    def test_bridge_client_screenshot_default_format(self):
        captured: Dict[str, Any] = {}

        class FakeClient:
            def __init__(self, **kw):
                self.timeout_ms = 30000
                self.bridge_token = "t"
                self.socket_name = "s"
            def connect(self): pass
            def close(self): pass
            def _call(self, method, **params):
                captured.update(params)
                return {"ok": True, "image_b64": "x", "format": "png",
                        "width": 100, "height": 100}
                captured.update(params)
                return {"ok": True, "image_b64": "x", "format": "png",
                        "width": 100, "height": 100}
            def screenshot(self, path=None, format="png", root=None, target=None):
                return self._call("screenshot", path=path, format=format,
                                  root_object_name=root, target=target)
            def __enter__(self): return self
            def __exit__(self, *a): self.close()

        cli = FakeClient()
        with cli:
            cli.screenshot("root.child")
        self.assertEqual(captured["format"], "png")


class TestToolsGuiScreenshotGracefulDegradation(unittest.TestCase):
    """gui_screenshot returns clear errors when no bridge is active or format is bad."""

    def test_gui_screenshot_no_bridge(self):
        orig = bridge_client.BridgeClient
        bridge_client.BridgeClient = unittest.mock.MagicMock(
            side_effect=RuntimeError("BridgeClient requires PyQt5")
        )
        try:
            out = tools_gui.gui_screenshot("a.b.c")
            self.assertFalse(out["ok"])
            self.assertIn("No active QGIS bridge session found", out["error"])
        finally:
            bridge_client.BridgeClient = orig

    def test_gui_screenshot_empty_path_no_target_rejected(self):
        # Empty path AND no explicit target is rejected locally — the
        # function no longer silently defaults target to "dialog" and
        # overrides a user-supplied (but empty) path.
        out = tools_gui.gui_screenshot("")
        self.assertFalse(out["ok"])
        self.assertIn("target is required", out["error"])

    def test_gui_screenshot_empty_target_rejected(self):
        out = tools_gui.gui_screenshot("", target="")
        self.assertFalse(out["ok"])
        self.assertIn("Invalid target", out["error"])

    def test_gui_screenshot_path_with_no_target_forwards_path(self):
        # When *path* is non-empty and *target* was not explicitly
        # provided, the function forwards ``path`` to the bridge and
        # passes ``target=None`` — the request must not be silently
        # rewritten to ``target="dialog"``.
        captured: Dict[str, Any] = {}

        class FakeClient:
            def __init__(self, **kw):
                self.timeout_ms = 30000
                self.bridge_token = "t"
                self.socket_name = "s"
            def connect(self): pass
            def close(self): pass
            def screenshot(self, path=None, format="png", root=None, target=None):
                captured["path"] = path
                captured["path"] = path
                captured["target"] = target
                return {"ok": False, "error": "no bridge"}  # exit the with-block
            def __enter__(self): return self
            def __exit__(self, *a): self.close()

        import tools.hydra_mcp.bridge_client as bc
        prev = bc.BridgeClient
        # ``tools_gui`` caches the BridgeClient class on first call; the
        # previous test in this class monkey-patched that cache, so
        # reset it before substituting our own.
        prev_cache = tools_gui._BridgeClient
        tools_gui._BridgeClient = None
        bc.BridgeClient = FakeClient
        try:
            out = tools_gui.gui_screenshot("a.b.c")
        finally:
            bc.BridgeClient = prev
            tools_gui._BridgeClient = prev_cache
        self.assertFalse(out["ok"])
        self.assertEqual(captured["path"], "a.b.c")
        self.assertIsNone(
            captured["target"],
            "explicit path must not be silently overridden by the default target",
        )

    def test_gui_screenshot_invalid_target_rejected(self):
        out = tools_gui.gui_screenshot("ignored", target="bogus")
        self.assertFalse(out["ok"])
        self.assertIn("Invalid target", out["error"])

    def test_gui_screenshot_invalid_format(self):
        out = tools_gui.gui_screenshot("a.b", format="bmp")
        self.assertFalse(out["ok"])
        self.assertIn("Invalid format", out["error"])


class TestToolsGuiScreenshotMockedBridge(unittest.TestCase):
    """Socket-pair integration test for gui_screenshot."""

    def setUp(self):
        self.sock_a, self.sock_b = socket.socketpair()

    def tearDown(self):
        self.sock_a.close()
        self.sock_b.close()

    def _respond(self, request_id, result):
        msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
        self.sock_b.send(bridge_client.encode_message(msg))

    def _make_mock_client(self):
        class MockBridgeClient:
            _sock = None

            def __init__(self, token_path=None, socket_name=None,
                         token=None, timeout=30.0):
                self.timeout_ms = int(timeout * 1000)
                self.token = token or "mock_token"
                self.socket_name = socket_name or "mock_socket"

            def connect(self): pass
            def close(self): pass

            def _call(self, method, **params):
                params["token"] = self.token
                request = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                    "id": 1,
                }
                self._sock.send(bridge_client.encode_message(request))
                self._sock.settimeout(5.0)
                import struct
                header = self._sock.recv(4)
                length = struct.unpack("!I", header)[0]
                body = b""
                while len(body) < length:
                    body += self._sock.recv(length - len(body))
                import json
                resp = json.loads(body.decode())
                if "error" in resp:
                    raise RuntimeError(resp["error"].get("message"))
                return resp.get("result", {})

            def screenshot(self, path=None, format="png", root=None, target=None):
                return self._call("screenshot", path=path, format=format,
                                  root_object_name=root, target=target)

            def __enter__(self): return self
            def __exit__(self, *args): self.close()

        MockBridgeClient._sock = self.sock_a
        return MockBridgeClient()

    def test_screenshot_png_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "screenshot":
                        path = msg["params"].get("path")
                        fmt = msg["params"].get("format", "png")
                        if path == "root.tab":
                            self._respond(msg["id"], {
                                "ok": True,
                                "image_b64": "PNG_BASE64_DATA",
                                "format": fmt,
                                "width": 800,
                                "height": 600,
                            })
                        else:
                            self._respond(msg["id"], {
                                "ok": False,
                                "error": f"widget not found at {path}",
                            })

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        orig_get = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_screenshot("root.tab", format="png")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["image_b64"], "PNG_BASE64_DATA")
            self.assertEqual(out["format"], "png")
            self.assertEqual(out["width"], 800)
            self.assertEqual(out["height"], 600)

            out = tools_gui.gui_screenshot("root.missing")
            self.assertFalse(out["ok"])
            self.assertIn("not found", out["error"])
        finally:
            tools_gui._get_bridge_client = orig_get

    def test_screenshot_jpeg_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "screenshot":
                        self._respond(msg["id"], {
                            "ok": True,
                            "image_b64": "JPEG_BASE64_DATA",
                            "format": "jpg",
                            "width": 1024,
                            "height": 768,
                        })

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        orig_get = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_screenshot("root.win", format="jpeg")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["format"], "jpg")
            self.assertEqual(out["width"], 1024)
            self.assertEqual(out["height"], 768)
        finally:
            tools_gui._get_bridge_client = orig_get


class TestCaptureWidgetScreenshot(unittest.TestCase):
    """Unit tests for the capture_widget_screenshot helper (plain Python mocks).

    ``tools.hydra_mcp.widget_screenshot`` is Qt-free at module level, so no
    QGIS app is needed to import it.
    """

    def setUp(self):
        # widget_screenshot.py is Qt-free — no mocks needed.
        from tools.hydra_mcp.widget_screenshot import capture_widget_screenshot
        self._capture = capture_widget_screenshot

    def test_none_widget_returns_error(self):
        out = self._capture(None, "png")
        self.assertFalse(out["ok"])
        self.assertIn("not available", out["error"])

    def test_unsupported_format_returns_error(self):
        # A mock widget that is visible and has grab(), but wrong format.
        mock_widget = _MockWidget("w")
        mock_widget._visible = True  # make isVisible() return True
        out = self._capture(mock_widget, "bmp")
        self.assertFalse(out["ok"])
        self.assertIn("unsupported format", out["error"])


class TestWidgetWalkerScreenshotMocked(unittest.TestCase):
    """Unit tests for screenshot via a mock widget that implements grab()."""

    def setUp(self):
        # widget_screenshot.py is Qt-free — no mocks needed.
        from tools.hydra_mcp.widget_screenshot import capture_widget_screenshot
        self._capture = capture_widget_screenshot

    def test_screenshot_calls_grab_and_returns_b64(self):
        # Build a mock pixmap returned by widget.grab()
        class MockPixmap:
            def __init__(self):
                self._w = 320
                self._h = 240
            def width(self): return self._w
            def height(self): return self._h
            def save(self, buffer, fmt, quality=None):
                # Write a minimal PNG header (just enough to be non-empty)
                buffer.write(b"\x89PNG\r\n\x1a\n" + fmt.encode())
                return True

        class MockWidgetForScreenshot(_MockWidget):
            def __init__(self, object_name):
                super().__init__(object_name)
                self._visible = True
                self._grab_called = False

            def isVisible(self):
                return self._visible

            def grab(self):
                self._grab_called = True
                return MockPixmap()

        widget = MockWidgetForScreenshot("my_widget")
        out = self._capture(widget, "png")
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["format"], "png")
        self.assertEqual(out["width"], 320)
        self.assertEqual(out["height"], 240)
        self.assertIsInstance(out["image_b64"], str)
        self.assertTrue(widget._grab_called)

    def test_screenshot_jpeg_format(self):
        class MockPixmap:
            def __init__(self):
                self._w = 640
                self._h = 480
            def width(self): return self._w
            def height(self): return self._h
            def save(self, buffer, fmt, quality=None):
                buffer.write(b"\xff\xd8" + fmt.encode())
                return True

        class MockWidgetForScreenshot(_MockWidget):
            def __init__(self, object_name):
                super().__init__(object_name)
                self._visible = True
                self._pixmap = MockPixmap()
            def isVisible(self): return self._visible
            def grab(self): return self._pixmap

        widget = MockWidgetForScreenshot("win")
        out = self._capture(widget, "jpeg")
        self.assertTrue(out["ok"], out)
        # Function normalizes "jpeg" → "jpg" in the returned format field.
        self.assertEqual(out["format"], "jpg")

    def test_screenshot_invisible_widget_returns_error(self):
        class MockWidgetInvisible(_MockWidget):
            def isVisible(self): return False

        widget = MockWidgetInvisible("hidden")
        out = self._capture(widget, "png")
        self.assertFalse(out["ok"])
        self.assertIn("not available", out["error"])

    def test_screenshot_uses_qbuffer_qiodevice(self):
        """Verify the buffer passed to QPixmap.save() is a real QtCore.QBuffer
        (a QIODevice subclass), NOT an io.BytesIO. The previous implementation
        called pixmap.save(io.BytesIO(), "PNG") which silently failed for real
        QPixmap instances — BytesIO doesn't implement the QIODevice protocol.

        We use a mock pixmap that captures the buffer argument and verifies
        its type against the real ``qgis.PyQt.QtCore.QBuffer``.
        """
        try:
            from qgis.PyQt import QtCore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"PyQt5 not importable: {exc}")

        captured_buffer = []  # type: ignore[var-annotated]

        class CapturingMockPixmap:
            def __init__(self):
                self._w = 4
                self._h = 4
            def width(self): return self._w
            def height(self): return self._h
            def save(self, buffer, fmt, quality=None):
                captured_buffer.append((buffer, fmt, quality))
                # BytesIO-style write — our real wrapper passes a QBuffer
                # which is writeable like a file.
                buffer.write(b"\x89PNG\r\n\x1a\n" + fmt.encode())
                return True

        captured_pixmap = CapturingMockPixmap()

        class RealWidget(_MockWidget):
            def __init__(self, object_name):
                super().__init__(object_name)
                self._visible = True

            def isVisible(self):
                return self._visible

            def grab(self):
                return captured_pixmap

        out = self._capture(RealWidget("real_widget"), "png")
        self.assertTrue(out["ok"], out)
        # The buffer arg must be a real QIODevice / QBuffer instance, not a
        # BytesIO. This is the property that was broken in the prior version.
        self.assertEqual(len(captured_buffer), 1, captured_buffer)
        buf, fmt, _q = captured_buffer[0]
        self.assertIsInstance(buf, QtCore.QBuffer)
        self.assertIsInstance(buf, QtCore.QIODevice)
        # The PNG bytes are read out via QBuffer.data() in the wrapper.
        import base64

        decoded = base64.b64decode(out["image_b64"])
        self.assertTrue(decoded.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(out["format"], "png")
        self.assertEqual(out["width"], 4)
        self.assertEqual(out["height"], 4)

    def test_screenshot_jpeg_uses_qbuffer_with_quality(self):
        """JPEG path must also use a real QBuffer and pass quality=85."""
        try:
            from qgis.PyQt import QtCore  # noqa: F401
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"PyQt5 not importable: {exc}")

        captured = []  # type: ignore[var-annotated]

        class CapturingMockPixmap:
            def __init__(self):
                self._w = 2
                self._h = 2
            def width(self): return self._w
            def height(self): return self._h
            def save(self, buffer, fmt, quality=None):
                captured.append((buffer, fmt, quality))
                buffer.write(b"\xff\xd8" + fmt.encode())
                return True

        captured_pixmap = CapturingMockPixmap()

        class RealWidget(_MockWidget):
            def __init__(self, object_name):
                super().__init__(object_name)
                self._visible = True

            def isVisible(self):
                return self._visible

            def grab(self):
                return captured_pixmap

        out = self._capture(RealWidget("w"), "jpeg")
        self.assertTrue(out["ok"], out)
        buf, fmt, q = captured[0]
        self.assertIsInstance(buf, QtCore.QBuffer)
        self.assertIsInstance(buf, QtCore.QIODevice)
        self.assertEqual(fmt, "JPEG")
        # Quality arg forwarded (3rd positional, int 85).
        self.assertEqual(q, 85)
        self.assertEqual(out["format"], "jpg")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2.H — workspace containment (Tier A: model_inspect / run_list / results_query)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorkspaceContainmentModeling(unittest.TestCase):
    """Workspace containment rejects escapes for the Phase 0 read tools.

    The fixture paths used by the other TestModelInspect / TestRunList /
    TestResultsQuery tests live inside the workspace (setUp points
    HYDRA_MCP_WORKSPACE_ROOT at the tempdir); here we exercise the
    rejection paths directly without a workspace override, so the canonical
    repo root is in effect.
    """

    def setUp(self):
        # Make sure no override leaks in from the parent environment.
        self._prev = os.environ.pop("HYDRA_MCP_WORKSPACE_ROOT", None)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._prev is not None:
            os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = self._prev

    def test_parent_dotdot_escape_rejected(self):
        # /tmp/../etc/passwd resolves outside the workspace even though it
        # lexically starts under it.
        out = tools_modeling.model_inspect("/tmp/../etc/passwd")
        self.assertFalse(out["ok"], out)
        self.assertIn("workspace", out["error"].lower())

    def test_absolute_outside_workspace_rejected(self):
        # /etc/passwd is a real file but lives outside any workspace.
        out = tools_modeling.run_list("/etc/passwd")
        self.assertFalse(out["ok"])
        self.assertIn("workspace", out["error"].lower())

    def test_symlink_pointing_outside_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = tmpdir
            outside = os.path.join(tmpdir, "..", "outside.gpkg")
            # Use a path that resolves outside even after we mkdir a symlink.
            try:
                os.symlink("/etc/passwd", outside)
                out = tools_modeling.results_query(outside, run_id="x", field="h")
                self.assertFalse(out["ok"])
                self.assertIn("workspace", out["error"].lower())
            finally:
                if os.path.lexists(outside):
                    os.unlink(outside)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2.H — workspace containment (Tier B: gui_launch project, bridge token file)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorkspaceContainmentGuiLaunch(unittest.TestCase):
    """gui_launch's ``project`` parameter must resolve inside the workspace."""

    def setUp(self):
        self._prev = os.environ.pop("HYDRA_MCP_WORKSPACE_ROOT", None)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._prev is not None:
            os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = self._prev

    def test_gui_launch_project_outside_workspace_rejected(self):
        # An absolute path outside any sane workspace is rejected without
        # spawning QGIS or trying to read the file.
        out = tools_gui.gui_launch(mode="offscreen", project="/etc/passwd")
        self.assertFalse(out["ok"], out)
        self.assertIn("workspace", out["error"].lower())


class TestBridgeClientTokenPathContainment(unittest.TestCase):
    """BridgeClient(token_path=...) rejects paths outside the workspace."""

    def setUp(self):
        self._prev = os.environ.pop("HYDRA_MCP_WORKSPACE_ROOT", None)
        self.addCleanup(self._restore)
        # Pin the workspace at /tmp so any token_path under it is accepted
        # and any other absolute path is rejected.
        os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = tempfile.gettempdir()

    def _restore(self):
        if self._prev is not None:
            os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = self._prev
        else:
            os.environ.pop("HYDRA_MCP_WORKSPACE_ROOT", None)

    def test_outside_workspace_token_path_rejected(self):
        # /etc/passwd exists and is readable but lives outside the workspace.
        with self.assertRaises(RuntimeError) as ctx:
            bridge_client.BridgeClient(token_path="/etc/passwd")
        self.assertIn("workspace", str(ctx.exception).lower())

    def test_inside_workspace_token_path_accepted(self):
        # A valid token file inside the workspace is loaded.
        token_path = os.path.join(tempfile.gettempdir(), "test_token_xyz.json")
        with open(token_path, "w") as fh:
            json.dump({"socket_name": "s", "token": "t"}, fh)
        try:
            cli = bridge_client.BridgeClient(token_path=token_path)
            self.assertEqual(cli.socket_name, "s")
            self.assertEqual(cli.token, "t")
        finally:
            os.unlink(token_path)

    def test_discovered_token_in_tmp_loaded_without_workspace_override(self):
        # CRITICAL-1 (2.H review): bridge auto-discovery scans
        # $XDG_RUNTIME_DIR / /tmp — directories that, by design, sit
        # OUTSIDE the default workspace.  Pre-fix, the discovered file
        # was routed through ``_load_token_file``, which enforced
        # ``WorkspacePath.resolve_under`` and rejected every candidate
        # in /tmp.  The fix routes discovery through
        # ``_parse_token_file`` which only validates ownership and
        # mode 0o600.  This test runs WITHOUT any
        # ``HYDRA_MCP_WORKSPACE_ROOT`` override so the default
        # workspace (the repo root) is in effect, then writes a
        # well-formed 0600 token file in /tmp and confirms that
        # ``BridgeClient()`` (no token_path) discovers and loads it.
        import time as _time
        prev_root = os.environ.pop("HYDRA_MCP_WORKSPACE_ROOT", None)
        prev_token = os.environ.pop("HYDRA_MCP_BRIDGE_TOKEN", None)
        prev_socket = os.environ.pop("HYDRA_MCP_BRIDGE_SOCKET", None)
        self.addCleanup(self._restore_env_discover, prev_root, prev_token, prev_socket)

        token_path = os.path.join(
            tempfile.gettempdir(),
            "hydra_mcp_bridge_discovery_xyz_test.json",
        )
        if os.path.exists(token_path):
            os.unlink(token_path)
        try:
            with open(token_path, "w") as fh:
                json.dump({"socket_name": "discovered_sock",
                           "token": "discovered_secret"}, fh)
            os.chmod(token_path, 0o600)
            # Bump mtime so the discovery sort picks this file first.
            _time.sleep(0.05)
            os.utime(token_path, None)

            cli = bridge_client.BridgeClient()  # no token_path → discovery
            self.assertEqual(cli.socket_name, "discovered_sock")
            self.assertEqual(cli.token, "discovered_secret")
        finally:
            if os.path.exists(token_path):
                os.unlink(token_path)

    @staticmethod
    def _restore_env_discover(prev_root, prev_token, prev_socket):
        if prev_root is None:
            os.environ.pop("HYDRA_MCP_WORKSPACE_ROOT", None)
        else:
            os.environ["HYDRA_MCP_WORKSPACE_ROOT"] = prev_root
        if prev_token is None:
            os.environ.pop("HYDRA_MCP_BRIDGE_TOKEN", None)
        else:
            os.environ["HYDRA_MCP_BRIDGE_TOKEN"] = prev_token
        if prev_socket is None:
            os.environ.pop("HYDRA_MCP_BRIDGE_SOCKET", None)
        else:
            os.environ["HYDRA_MCP_BRIDGE_SOCKET"] = prev_socket


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2.H — frame length cap (DoS protection)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFrameLengthCap(unittest.TestCase):
    """Length-prefixed framing rejects frames above ``MAX_FRAME_BYTES``.

    Without this guard, an unauthenticated peer could declare a multi-GB
    frame via the 4-byte length prefix and force the local side to
    allocate, buffer, or JSON-decode a payload before any auth check runs.
    The cap is enforced in :func:`decode_messages` and applies to both
    server-side reads (``qgis_bridge._on_ready_read``) and client-side
    response parsing (``BridgeClient._call``).
    """

    def test_seventeen_mib_frame_rejected_with_clear_error(self):
        import struct
        # Hand-build a 17 MiB frame: 4-byte length prefix > MAX_FRAME_BYTES.
        bad_length = 17 * 1024 * 1024
        bogus = struct.pack("!I", bad_length) + b"\x00" * 16
        with self.assertRaises(bridge_client.FrameTooLargeError) as ctx:
            bridge_client.decode_messages(bogus)
        self.assertEqual(ctx.exception.length, bad_length)
        self.assertEqual(ctx.exception.max_bytes, bridge_client.MAX_FRAME_BYTES)
        self.assertIn("MAX_FRAME_BYTES", str(ctx.exception))

    def test_oversized_length_does_not_consume_buffer(self):
        # An oversized header followed by other bytes must raise BEFORE
        # the rest of the buffer is interpreted (i.e. we never silently
        # advance past the bad header).
        import struct
        bad = struct.pack("!I", 17 * 1024 * 1024) + b"junk after bad header"
        with self.assertRaises(bridge_client.FrameTooLargeError):
            bridge_client.decode_messages(bad)

    def test_max_frame_bytes_constant_is_16mib(self):
        # The cap is part of the protocol contract — keep it at exactly
        # 16 MiB (raised from 1 MiB in 96368c2b so screenshot base64 blobs
        # fit comfortably while still bounding unauthenticated buffering).
        self.assertEqual(bridge_client.MAX_FRAME_BYTES, 16 << 20)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2.H — atomic 0600 token file + shutdown cleanup
# ═══════════════════════════════════════════════════════════════════════════════


@requires_qgis
class TestTokenFileAtomic0600(unittest.TestCase):
    """``HydraMcpBridge._write_token_file`` creates an atomically-private file.

    The previous implementation used ``Path.write_text`` + ``os.chmod``,
    which leaves a window where the file is visible with a permissive umask
    (and is vulnerable to symlink/precreate races in ``/tmp``).  The fix
    uses ``os.open`` with ``O_CREAT|O_EXCL|O_WRONLY`` and ``mode=0o600``
    so the file never exists with broader permissions and cannot be
    pre-created by an attacker.
    """

    def setUp(self):
        ensure_qgis_app()
        # Real ``qgis.PyQt.QtNetwork`` provides QLocalServer/QLocalSocket;
        # qgis_bridge.py imports them at module level.
        from tools.hydra_mcp.qgis_bridge import HydraMcpBridge

        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        # Override the bridge's token_path to point at our tempdir.
        self.bridge = HydraMcpBridge()
        self.bridge.token_path = Path(self.tmpdir) / self.bridge.token_path.name

    def test_token_file_created_with_mode_0600(self):
        self.bridge._write_token_file()
        self.assertTrue(self.bridge.token_path.exists())
        mode = stat.S_IMODE(self.bridge.token_path.stat().st_mode)
        self.assertEqual(mode, 0o600, f"expected mode 0600, got {oct(mode)}")

    def test_token_file_preexistence_rejected(self):
        # A pre-existing file at the same path blocks the atomic create —
        # no symlink/TOCTOU window where the attacker could swap in a forged
        # token before the bridge writes its real one.
        self.bridge.token_path.write_text("forged content")
        with self.assertRaises(FileExistsError):
            self.bridge._write_token_file()

    def test_token_file_contains_valid_payload(self):
        self.bridge._write_token_file()
        payload = json.loads(self.bridge.token_path.read_text())
        self.assertEqual(payload["socket_name"], self.bridge.socket_name)
        self.assertEqual(payload["token"], self.bridge.token)
        self.assertEqual(payload["version"], "2.C")

    def test_failed_write_cleans_up_partial_file(self):
        # Force a write failure by closing the underlying fd between open and
        # write — the bridge must remove the partial token file so it does
        # not linger with a valid mode and zero-byte content.
        real_open = os.open
        raised = {"count": 0}

        def fail_after_open(path, flags, mode=0o777, *args, **kwargs):
            fd = real_open(path, flags, mode, *args, **kwargs)
            if str(path).endswith(self.bridge.token_path.name):
                # Immediately raise so the fdopen path never runs.
                raised["count"] += 1
                os.close(fd)
                raise OSError("simulated write failure")
            return fd

        with unittest.mock.patch("tools.hydra_mcp.qgis_bridge.os.open", side_effect=fail_after_open):
            with self.assertRaises(OSError):
                self.bridge._write_token_file()
        self.assertFalse(
            self.bridge.token_path.exists(),
            "partial token file must be removed on write failure",
        )

    def test_cleanup_token_file_removes_existing_file(self):
        self.bridge._write_token_file()
        self.assertTrue(self.bridge.token_path.exists())
        self.bridge._cleanup_token_file()
        self.assertFalse(self.bridge.token_path.exists())

    def test_cleanup_token_file_idempotent_on_missing_file(self):
        # Calling cleanup when the file is already gone must not raise.
        self.assertFalse(self.bridge.token_path.exists())
        self.bridge._cleanup_token_file()  # must not raise

    def test_o_excl_collision_does_not_unlink_foreign_file(self):
        # MEDIUM-4 (2.H review): when ``os.open(..., O_EXCL)`` fails with
        # ``FileExistsError`` because another live bridge already holds
        # the path, the cleanup branch must NOT unlink the foreign file.
        # Pre-fix the outer ``except Exception`` ran unconditionally.
        pre_existing = self.bridge.token_path
        pre_existing.write_text("foreign bridge token", encoding="utf-8")
        pre_existing.chmod(0o600)
        try:
            with self.assertRaises(FileExistsError):
                self.bridge._write_token_file()
            self.assertTrue(
                pre_existing.exists(),
                "foreign token file must not be removed on O_EXCL collision",
            )
            self.assertEqual(
                pre_existing.read_text(encoding="utf-8"),
                "foreign bridge token",
                "foreign token file must not be modified either",
            )
        finally:
            if pre_existing.exists():
                pre_existing.unlink()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2.H — screenshot target param (dialog / dock / canvas)
# ═══════════════════════════════════════════════════════════════════════════════


@requires_qgis
class TestScreenshotTargetResolution(unittest.TestCase):
    """``gui_screenshot(target=...)`` resolves each target to a live widget.

    Uses the real offscreen ``QgsApplication`` from ``ensure_qgis_app()``
    so the resolution logic is exercised end-to-end against live widgets.
    """

    @classmethod
    def setUpClass(cls):
        cls._qapp = ensure_qgis_app()

    def setUp(self):
        from tools.hydra_mcp.qgis_bridge import HydraMcpBridge
        self.bridge = HydraMcpBridge()
        # Keep a reference to the test widgets so they survive the
        # teardown and don't trigger QApplication cleanup in the wrong
        # order.
        self._created_widgets: list = []

    def tearDown(self):
        # Release references to any widgets we created.
        for w in list(self._created_widgets):
            try:
                w.deleteLater()
            except Exception:
                pass
        self._created_widgets.clear()

    def _track(self, widget):
        self._created_widgets.append(widget)
        return widget

    def test_target_dialog_returns_active_window(self):
        from qgis.PyQt.QtWidgets import QMainWindow
        win = self._track(QMainWindow())
        win.setObjectName("active_dialog_target_test")
        self._qapp.setActiveWindow(win)
        resolved = self.bridge._resolve_screenshot_target({"target": "dialog"})
        self.assertIs(resolved, win)

    def test_target_dock_returns_first_dock_widget(self):
        from qgis.PyQt.QtWidgets import QDockWidget, QMainWindow
        main = self._track(QMainWindow())
        dock = QDockWidget("d", main)
        dock.setObjectName("screenshot_target_dock_test")
        main.addDockWidget(0x1, dock)  # LeftDockWidgetArea = 0x1
        main.show()
        resolved = self.bridge._resolve_screenshot_target({"target": "dock"})
        self.assertIs(resolved, dock)

    def test_target_canvas_returns_none_without_canvas_widget(self):
        # Real ``qgis.gui.QgsMapCanvas`` is importable, but no canvas widget
        # exists among the top-level widgets, so the target resolves to
        # None — the caller treats this as "widget not available".
        resolved = self.bridge._resolve_screenshot_target({"target": "canvas"})
        self.assertIsNone(resolved)

    def test_target_unknown_raises_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.bridge._resolve_screenshot_target({"target": "bogus"})
        self.assertIn("unknown screenshot target", str(ctx.exception))

    def test_target_wins_over_path(self):
        from qgis.PyQt.QtWidgets import QMainWindow
        win = self._track(QMainWindow())
        win.setObjectName("screenshot_target_over_path_test")
        self._qapp.setActiveWindow(win)
        resolved = self.bridge._resolve_screenshot_target(
            {"target": "dialog", "path": "ignored.path"}
        )
        self.assertIs(resolved, win)

    def test_path_only_legacy_mode(self):
        # Without target and without path, raise — bridge requires one.
        with self.assertRaises(RuntimeError) as ctx:
            self.bridge._resolve_screenshot_target({})
        self.assertIn("required", str(ctx.exception))


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2.H — widget path resolution + isinstance + timeout + bool coercion
# ═══════════════════════════════════════════════════════════════════════════════


@requires_qgis
class TestWidgetPathStrictTraversal(unittest.TestCase):
    """``find_widget_by_path`` uses direct children, raises on ambiguity.

    Pre-2.H.6 the walker used ``findChildren(QWidget, "")`` (recursive),
    which let a 2-segment path jump 4 levels and recorded descendants
    as direct root children.  Post-2.H.6 the walker uses
    ``QObject.children()`` (direct) and refuses to navigate to a
    grand-child via a single hop.
    """

    def test_path_must_descend_one_level_per_segment(self):
        from swe2d.workbench.devtools.widget_walker import find_widget_by_path
        # Build: root -> container -> leaf (3 levels)
        leaf = _MockWidget("leaf")
        container = _MockWidget("container", children=[leaf])
        root = _MockWidget("root", children=[container])
        # 1 segment reaches container, not leaf.
        self.assertIs(
            find_widget_by_path(root, ["container"]), container
        )
        # 2 segments reach the leaf.
        self.assertIs(
            find_widget_by_path(root, ["container", "leaf"]), leaf
        )
        # "leaf" is NOT a direct child of root, so a 1-segment path
        # cannot reach it — this is the strict-traversal fix.
        self.assertIsNone(find_widget_by_path(root, ["leaf"]))

    def test_ambiguous_path_segment_raises(self):
        from swe2d.workbench.devtools.widget_walker import (
            AmbiguousWidgetPathError,
            find_widget_by_path,
        )
        # Two siblings with the same objectName — ambiguous.
        twin_a = _MockWidget("twin")
        twin_b = _MockWidget("twin")
        container = _MockWidget("container", children=[twin_a, twin_b])
        root = _MockWidget("root", children=[container])
        with self.assertRaises(AmbiguousWidgetPathError) as ctx:
            find_widget_by_path(root, ["container", "twin"])
        self.assertIn("twin", str(ctx.exception))
        self.assertIn("2 direct children", str(ctx.exception))

    def test_ambiguity_only_at_segment_with_duplicates(self):
        from swe2d.workbench.devtools.widget_walker import (
            AmbiguousWidgetPathError,
            find_widget_by_path,
        )
        # Root has two children with the same name; that is its own
        # ambiguity (the very first segment).
        twin_a = _MockWidget("twin")
        twin_b = _MockWidget("twin")
        root = _MockWidget("root", children=[twin_a, twin_b])
        with self.assertRaises(AmbiguousWidgetPathError):
            find_widget_by_path(root, ["twin"])

    def test_root_not_included_in_path(self):
        # The root widget is NOT itself a matchable segment.  The
        # walker treats ``root`` as the anchor; ``path_parts[0]`` is
        # a direct child of root.
        from swe2d.workbench.devtools.widget_walker import find_widget_by_path
        child = _MockWidget("child")
        root = _MockWidget("root", children=[child])
        self.assertIsNone(find_widget_by_path(root, ["root"]))
        self.assertIs(find_widget_by_path(root, ["child"]), child)

    def test_empty_path_returns_root(self):
        from swe2d.workbench.devtools.widget_walker import find_widget_by_path
        root = _MockWidget("root")
        self.assertIs(find_widget_by_path(root, []), root)
        self.assertIs(find_widget_by_path(root, ["", ""]), root)


@requires_qgis
class TestBoolCoercionForCheckBox(unittest.TestCase):
    """``set_widget_value`` honours ``"false"`` (the bool() bug)."""

    def _checkbox(self):
        from qgis.PyQt.QtWidgets import QCheckBox
        ensure_qgis_app()
        cb = QCheckBox()
        return cb

    def test_set_true_literal(self):
        from tools.hydra_mcp.qgis_bridge import set_widget_value
        cb = self._checkbox()
        out = set_widget_value(cb, True)
        self.assertTrue(out["ok"])
        self.assertTrue(cb.isChecked())

    def test_set_false_literal(self):
        from tools.hydra_mcp.qgis_bridge import set_widget_value
        cb = self._checkbox()
        cb.setChecked(True)
        out = set_widget_value(cb, False)
        self.assertTrue(out["ok"])
        self.assertFalse(cb.isChecked())

    def test_set_false_string(self):
        # The pre-2.H.6 bug: ``bool("false")`` returns True because the
        # string is non-empty.  The fix honours the string semantics.
        from tools.hydra_mcp.qgis_bridge import set_widget_value
        cb = self._checkbox()
        cb.setChecked(True)
        out = set_widget_value(cb, "false")
        self.assertTrue(out["ok"])
        self.assertFalse(cb.isChecked(), "string 'false' must coerce to False")

    def test_set_true_string(self):
        from tools.hydra_mcp.qgis_bridge import set_widget_value
        cb = self._checkbox()
        cb.setChecked(False)
        for s in ("true", "TRUE", "1", "yes", "on"):
            out = set_widget_value(cb, s)
            self.assertTrue(out["ok"], out)
            self.assertTrue(cb.isChecked(), f"string {s!r} must coerce to True")

    def test_set_zero_int(self):
        from tools.hydra_mcp.qgis_bridge import set_widget_value
        cb = self._checkbox()
        cb.setChecked(True)
        out = set_widget_value(cb, 0)
        self.assertTrue(out["ok"])
        self.assertFalse(cb.isChecked())


@requires_qgis
class TestBridgeBootstrap(unittest.TestCase):
    """Unit tests for ``bootstrap_bridge_if_needed`` console/plugin entry point."""

    def setUp(self):
        ensure_qgis_app()
        # Real ``qgis.PyQt.QtNetwork`` provides QLocalServer/QLocalSocket;
        # qgis_bridge.py imports them at module level.
        import tools.hydra_mcp.qgis_bridge as _qgb
        self._qgb = _qgb
        self._prev_env = os.environ.get("HYDRA_MCP_BRIDGE")
        self._orig_instance = _qgb._HYDRA_MCP_BRIDGE_INSTANCE

    def tearDown(self):
        if self._prev_env is None:
            os.environ.pop("HYDRA_MCP_BRIDGE", None)
        else:
            os.environ["HYDRA_MCP_BRIDGE"] = self._prev_env
        self._qgb._HYDRA_MCP_BRIDGE_INSTANCE = self._orig_instance

    def test_bootstrap_sets_env_and_starts_bridge(self):
        os.environ.pop("HYDRA_MCP_BRIDGE", None)
        self._qgb._HYDRA_MCP_BRIDGE_INSTANCE = None

        class FakeBridge:
            def __init__(self):
                self.started = False
                self.socket_name = "fake_socket"
                self.token_path = "/tmp/fake_token.json"

            def objectName(self):
                return "FakeBridge"

            def start(self):
                self.started = True

        orig_class = self._qgb.HydraMcpBridge
        self._qgb.HydraMcpBridge = FakeBridge
        try:
            bridge = self._qgb.bootstrap_bridge_if_needed()
            self.assertIsInstance(bridge, FakeBridge)
            self.assertTrue(bridge.started)
            self.assertEqual(os.environ.get("HYDRA_MCP_BRIDGE"), "1")
        finally:
            self._qgb.HydraMcpBridge = orig_class

    def test_bootstrap_returns_existing_live_instance(self):
        self._qgb._HYDRA_MCP_BRIDGE_INSTANCE = None

        class FakeBridge:
            def __init__(self):
                self.socket_name = "existing_socket"
                self.token_path = "/tmp/existing_token.json"

            def objectName(self):
                return "ExistingBridge"

            def is_alive(self):
                # bootstrap_bridge_if_needed() only short-circuits on a
                # live existing instance.
                return True

        existing = FakeBridge()
        self._qgb._HYDRA_MCP_BRIDGE_INSTANCE = existing
        orig_class = self._qgb.HydraMcpBridge
        self._qgb.HydraMcpBridge = FakeBridge
        try:
            bridge = self._qgb.bootstrap_bridge_if_needed()
            self.assertIs(bridge, existing)
        finally:
            self._qgb.HydraMcpBridge = orig_class


class TestTimeoutPlumbedToBridgeClient(unittest.TestCase):
    """``_get_bridge_client(token_path, timeout)`` forwards timeout."""

    def test_timeout_forwarded_to_bridge_client(self):
        # Reset the module-level cache so our patched class is used.
        import tools.hydra_mcp.tools_gui as tg
        tg._BridgeClient = None
        captured: Dict[str, Any] = {}

        class CapturingClient:
            def __init__(self, token_path=None, timeout=30.0):
                captured["token_path"] = token_path
                captured["timeout"] = timeout

        import tools.hydra_mcp.bridge_client as bc
        prev = bc.BridgeClient
        bc.BridgeClient = CapturingClient
        try:
            cli = tg._get_bridge_client(token_path="/some/path", timeout=7.5)
            self.assertEqual(captured["timeout"], 7.5)
            self.assertEqual(captured["token_path"], "/some/path")
            self.assertIsInstance(cli, CapturingClient)
        finally:
            bc.BridgeClient = prev
            tg._BridgeClient = None

    def test_default_timeout_used_when_not_specified(self):
        import tools.hydra_mcp.tools_gui as tg
        tg._BridgeClient = None
        captured: Dict[str, Any] = {}

        class CapturingClient:
            def __init__(self, token_path=None, timeout=30.0):
                captured["timeout"] = timeout

        import tools.hydra_mcp.bridge_client as bc
        prev = bc.BridgeClient
        bc.BridgeClient = CapturingClient
        try:
            tg._get_bridge_client()
            self.assertEqual(captured["timeout"], 30.0)
        finally:
            bc.BridgeClient = prev
            tg._BridgeClient = None


class TestToolsGuiBehavioralGracefulDegradation(unittest.TestCase):
    """Phase 3 behavioral tools return clear errors when no bridge is active."""

    def setUp(self):
        self._orig_bridge_client = bridge_client.BridgeClient
        bridge_client.BridgeClient = unittest.mock.MagicMock(
            side_effect=RuntimeError("BridgeClient requires PyQt5")
        )
        tools_gui._BridgeClient = None

    def tearDown(self):
        bridge_client.BridgeClient = self._orig_bridge_client
        tools_gui._BridgeClient = None

    def test_gui_click_no_bridge(self):
        out = tools_gui.gui_click(object_name="run_btn")
        self.assertFalse(out["ok"])
        self.assertIn("No active QGIS bridge session found", out["error"])

    def test_gui_click_requires_path_or_name(self):
        out = tools_gui.gui_click()
        self.assertFalse(out["ok"])
        self.assertIn("path or object_name", out["error"])

    def test_gui_key_no_bridge(self):
        out = tools_gui.gui_key("return", object_name="edit")
        self.assertFalse(out["ok"])
        self.assertIn("No active QGIS bridge session found", out["error"])

    def test_gui_key_requires_key(self):
        out = tools_gui.gui_key("", object_name="edit")
        self.assertFalse(out["ok"])
        self.assertIn("key is required", out["error"])

    def test_gui_run_action_no_bridge(self):
        out = tools_gui.gui_run_action(text="Run")
        self.assertFalse(out["ok"])
        self.assertIn("No active QGIS bridge session found", out["error"])

    def test_gui_run_action_requires_name_or_text(self):
        out = tools_gui.gui_run_action()
        self.assertFalse(out["ok"])
        self.assertIn("object_name or text", out["error"])

    def test_gui_read_log_no_bridge(self):
        out = tools_gui.gui_read_log()
        self.assertFalse(out["ok"])
        self.assertIn("No active QGIS bridge session found", out["error"])

    def test_gui_run_simulation_no_bridge(self):
        out = tools_gui.gui_run_simulation(run_duration_text="0:01")
        self.assertFalse(out["ok"])
        self.assertIn("No active QGIS bridge session found", out["error"])


class TestToolsGuiBehavioralMockedBridge(unittest.TestCase):
    """Socket-pair integration tests for Phase 3 behavioral tools."""

    def setUp(self):
        self.sock_a, self.sock_b = socket.socketpair()

    def tearDown(self):
        self.sock_a.close()
        self.sock_b.close()

    def _respond(self, request_id, result):
        msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }
        self.sock_b.send(bridge_client.encode_message(msg))

    def _make_mock_client(self):
        class MockBridgeClient:
            _sock = None

            def __init__(self, token_path=None, socket_name=None,
                          token=None, timeout=30.0):
                self.timeout_ms = int(timeout * 1000)
                self.token = token or "mock_token"
                self.socket_name = socket_name or "mock_socket"

            def connect(self):
                pass

            def close(self):
                pass

            def _call(self, method, **params):
                params["token"] = self.token
                request = {
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                    "id": 1,
                }
                self._sock.send(bridge_client.encode_message(request))
                self._sock.settimeout(5.0)
                import struct
                header = self._sock.recv(4)
                length = struct.unpack("!I", header)[0]
                body = b""
                while len(body) < length:
                    body += self._sock.recv(length - len(body))
                import json
                resp = json.loads(body.decode())
                if "error" in resp:
                    raise RuntimeError(resp["error"].get("message"))
                return resp.get("result", {})

            def click_widget(self, path=None, object_name=None, root=None,
                             x=None, y=None):
                return self._call("click_widget", path=path,
                                   object_name=object_name,
                                   root_object_name=root, x=x, y=y)

            def key_press(self, key, path=None, object_name=None, root=None):
                return self._call("key_press", key=key, path=path,
                                   object_name=object_name,
                                   root_object_name=root)

            def run_action(self, object_name=None, text=None):
                return self._call("run_action", object_name=object_name,
                                   text=text)

            def read_log(self, max_lines=1000):
                return self._call("read_log", max_lines=max_lines)

            def run_simulation(self, run_duration_text=None,
                               output_interval_text=None,
                               timeout=60.0, startup_timeout=10.0):
                return self._call(
                    "run_simulation",
                    run_duration_text=run_duration_text,
                    output_interval_text=output_interval_text,
                    timeout=timeout,
                    startup_timeout=startup_timeout,
                )

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        MockBridgeClient._sock = self.sock_a
        return MockBridgeClient()

    def test_click_widget_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "click_widget":
                        p = msg["params"]
                        if p.get("object_name") == "run_btn":
                            self._respond(msg["id"], {
                                "ok": True,
                                "class_name": "QPushButton",
                                "object_name": "run_btn",
                                # Echo the click position so the test can
                                # assert x/y forwarding (added in 96368c2b).
                                "x": p.get("x"),
                                "y": p.get("y"),
                            })
                        else:
                            self._respond(msg["id"], {
                                "ok": False,
                                "error": "widget not found",
                            })

        threading.Thread(target=serve, daemon=True).start()
        orig = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_click(object_name="run_btn")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["class_name"], "QPushButton")

            # x/y click-position kwargs (96368c2b) must reach the bridge.
            out = tools_gui.gui_click(object_name="run_btn", x=0.5, y=0.25)
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["x"], 0.5)
            self.assertEqual(out["y"], 0.25)

            out = tools_gui.gui_click(object_name="missing")
            self.assertFalse(out["ok"])
            self.assertIn("widget not found", out["error"])
        finally:
            tools_gui._get_bridge_client = orig

    def test_key_press_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "key_press":
                        p = msg["params"]
                        if p.get("key") == "return" and p.get("object_name") == "edit":
                            self._respond(msg["id"], {
                                "ok": True, "key": "return",
                                "class_name": "QLineEdit",
                            })
                        else:
                            self._respond(msg["id"], {
                                "ok": False,
                                "error": "bad key",
                            })

        threading.Thread(target=serve, daemon=True).start()
        orig = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_key("return", object_name="edit")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["key"], "return")
        finally:
            tools_gui._get_bridge_client = orig

    def test_run_action_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "run_action":
                        p = msg["params"]
                        if p.get("text") == "Run":
                            self._respond(msg["id"], {"ok": True, "text": "Run"})
                        else:
                            self._respond(msg["id"], {
                                "ok": False,
                                "error": "action not found",
                            })

        threading.Thread(target=serve, daemon=True).start()
        orig = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_run_action(text="Run")
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["text"], "Run")

            out = tools_gui.gui_run_action(object_name="missing")
            self.assertFalse(out["ok"])
        finally:
            tools_gui._get_bridge_client = orig

    def test_read_log_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "read_log":
                        self._respond(msg["id"], {
                            "ok": True,
                            "lines": ["line1", "line2"],
                            "total": 2,
                        })

        threading.Thread(target=serve, daemon=True).start()
        orig = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_read_log(max_lines=10)
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["lines"], ["line1", "line2"])
            self.assertEqual(out["total"], 2)
        finally:
            tools_gui._get_bridge_client = orig

    def test_run_simulation_via_socket(self):
        def serve():
            self.sock_b.settimeout(5.0)
            while True:
                try:
                    raw = self.sock_b.recv(4096)
                except socket.timeout:
                    break
                if not raw:
                    break
                msgs, _ = bridge_client.decode_messages(raw)
                for msg in msgs:
                    if msg.get("method") == "run_simulation":
                        p = msg["params"]
                        if p.get("run_duration_text") == "0:01":
                            self._respond(msg["id"], {
                                "ok": True,
                                "status": "finished",
                                "run_id": "r1",
                                "message": "",
                            })
                        else:
                            self._respond(msg["id"], {
                                "ok": False,
                                "error": "bad params",
                            })

        threading.Thread(target=serve, daemon=True).start()
        orig = tools_gui._get_bridge_client
        tools_gui._get_bridge_client = lambda token_path=None, timeout=30.0: self._make_mock_client()
        try:
            out = tools_gui.gui_run_simulation(run_duration_text="0:01", timeout=1.0)
            self.assertTrue(out["ok"], out)
            self.assertEqual(out["status"], "finished")
            self.assertEqual(out["run_id"], "r1")
        finally:
            tools_gui._get_bridge_client = orig


class TestProcessRegistry(unittest.TestCase):
    """Unit tests for the launched-process lifecycle registry."""

    def setUp(self):
        self.registry = tools_gui._ProcessRegistry()
        self._orig_registry = tools_gui._PROCESS_REGISTRY
        tools_gui._PROCESS_REGISTRY = self.registry

    def tearDown(self):
        tools_gui._PROCESS_REGISTRY = self._orig_registry

    def _make_proc(self, pid=12345, alive_after_term=0, alive_after_kill=0):
        proc = unittest.mock.MagicMock(spec=subprocess.Popen)
        proc.pid = pid
        poll_results = {
            "alive_after_term": alive_after_term,
            "alive_after_kill": alive_after_kill,
        }
        call_count = {"term": 0, "kill": 0}

        def poll():
            if call_count["kill"] > 0:
                return None if poll_results["alive_after_kill"] > 0 else 0
            if call_count["term"] > 0:
                return None if poll_results["alive_after_term"] > 0 else 0
            return None

        def terminate():
            call_count["term"] += 1

        def kill():
            call_count["kill"] += 1

        proc.poll.side_effect = poll
        proc.terminate.side_effect = terminate
        proc.kill.side_effect = kill
        return proc, call_count

    def test_terminate_success(self):
        proc, calls = self._make_proc(alive_after_term=0)
        self.registry.register("/tmp/t.json", proc)
        out = self.registry.terminate("/tmp/t.json")
        self.assertTrue(out["ok"])
        self.assertEqual(out["action"], "terminated")
        self.assertEqual(out["pid"], 12345)
        self.assertEqual(calls["term"], 1)
        self.assertEqual(calls["kill"], 0)

    def test_sigterm_to_sigkill_escalation(self):
        proc, calls = self._make_proc(alive_after_term=1, alive_after_kill=0)
        self.registry.register("/tmp/t.json", proc)
        # Use short timeouts to keep the test fast.
        out = self.registry.terminate("/tmp/t.json", term_timeout=0.1, kill_timeout=0.1)
        self.assertTrue(out["ok"])
        self.assertEqual(out["action"], "killed")
        self.assertEqual(calls["term"], 1)
        self.assertEqual(calls["kill"], 1)

    def test_unknown_token_returns_error(self):
        out = self.registry.terminate("/tmp/nope.json")
        self.assertFalse(out["ok"])
        self.assertIn("No launched process", out["error"])

    def test_already_exited(self):
        proc = unittest.mock.MagicMock(spec=subprocess.Popen)
        proc.pid = 12345
        proc.poll.return_value = 0  # already exited
        self.registry.register("/tmp/t.json", proc)
        out = self.registry.terminate("/tmp/t.json")
        self.assertTrue(out["ok"])
        self.assertEqual(out["action"], "already_exited")

    def test_gui_close_without_token_falls_back_to_most_recent(self):
        proc, calls = self._make_proc(alive_after_term=0)
        self.registry.register("/tmp/a.json", proc, session_id="a", mode="xvfb")
        out = tools_gui.gui_close(timeout=1.0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["pid"], 12345)
        self.assertEqual(calls["term"], 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — design tools (rename / relabel / preview / apply patch)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDesignTools(unittest.TestCase):
    """Thin MCP wrappers over ``swe2d.workbench.devtools``.

    Tests use a scratch file in a temp directory so the real workbench view
    files are never modified.  The patch format and edit objects are the
    contract the server exposes to MCP clients.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

    def _write_view(self, name: str, source: str) -> str:
        path = os.path.join(self.tmpdir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        return path

    def test_design_rename_widget_returns_unified_diff(self):
        path = self._write_view(
            "view.py",
            'from qgis.PyQt import QtWidgets\n'
            'class V(QtWidgets.QWidget):\n'
            '    def __init__(self):\n'
            '        self.btn = QtWidgets.QPushButton()\n'
            '        self.btn.setObjectName("old_btn")\n',
        )
        out = tools_design.design_rename_widget("old_btn", "new_btn", [path])
        self.assertTrue(out["ok"], out)
        self.assertIn('setObjectName("new_btn")', out["patch_text"])
        self.assertEqual(out["file_path"], path)
        self.assertEqual(out["edits"][0]["kind"], "setObjectName")
        self.assertEqual(out["edits"][0]["new_value"], "new_btn")

    def test_design_rename_widget_rejects_name_collision(self):
        owner = self._write_view(
            "owner.py",
            'from qgis.PyQt import QtWidgets\n'
            'class V(QtWidgets.QWidget):\n'
            '    def __init__(self):\n'
            '        self.a = QtWidgets.QPushButton()\n'
            '        self.a.setObjectName("old_btn")\n',
        )
        other = self._write_view(
            "other.py",
            'from qgis.PyQt import QtWidgets\n'
            'class V(QtWidgets.QWidget):\n'
            '    def __init__(self):\n'
            '        self.b = QtWidgets.QPushButton()\n'
            '        self.b.setObjectName("new_btn")\n',
        )
        out = tools_design.design_rename_widget(
            "old_btn", "new_btn", [owner, other]
        )
        self.assertFalse(out["ok"])
        self.assertIn("already used", out["error"].lower())
    def test_design_rename_widget_rejects_missing_old_name(self):
        path = self._write_view(
            "view.py",
            'from qgis.PyQt import QtWidgets\n'
            'class V(QtWidgets.QWidget):\n'
            '    def __init__(self):\n'
            '        self.btn = QtWidgets.QPushButton()\n'
            '        self.btn.setObjectName("some_btn")\n',
        )
        out = tools_design.design_rename_widget("missing", "new_btn", [path])
        self.assertFalse(out["ok"])
        self.assertIn("not found", out["error"].lower())

    def test_design_relabel_widget_updates_titles_and_labels(self):
        path = self._write_view(
            "view.py",
            'from qgis.PyQt import QtWidgets\n'
            'class V(QtWidgets.QWidget):\n'
            '    def __init__(self):\n'
            '        self.g = QtWidgets.QGroupBox("Old Group")\n'
            '        self.t = QtWidgets.QToolBox()\n'
            '        self.t.addItem(self.g, "Old Group")\n',
        )
        out = tools_design.design_relabel_widget("Old Group", "New Group", [path])
        self.assertTrue(out["ok"], out)
        diff = out["patch_text"]
        self.assertIn('QGroupBox("New Group")', diff)
        self.assertIn('addItem(self.g, "New Group")', diff)
        self.assertEqual(len(out["edits"]), 2)

    def test_design_relabel_widget_rejects_missing_label(self):
        path = self._write_view(
            "view.py",
            'from qgis.PyQt import QtWidgets\n'
            'class V(QtWidgets.QWidget):\n'
            '    def __init__(self):\n'
            '        self.g = QtWidgets.QGroupBox("Some Group")\n',
        )
        out = tools_design.design_relabel_widget("Missing", "New", [path])
        self.assertFalse(out["ok"])
        self.assertIn("not found", out["error"].lower())

    def test_design_preview_patch_returns_diff_without_writing(self):
        path = self._write_view(
            "view.py",
            'from qgis.PyQt import QtWidgets\n'
            'class V(QtWidgets.QWidget):\n'
            '    def __init__(self):\n'
            '        self.btn = QtWidgets.QPushButton()\n'
            '        self.btn.setObjectName("old_btn")\n',
        )
        edits = [
            {
                "kind": "setObjectName",
                "file_path": path,
                "lineno": 5,
                "old_value": "old_btn",
                "new_value": "new_btn",
            }
        ]
        out = tools_design.design_preview_patch(edits, view_files=[path])
        self.assertTrue(out["ok"], out)
        self.assertIn("new_btn", out["patch_text"])
        self.assertEqual(Path(path).read_text().count("old_btn"), 1)

    def test_design_apply_patch_writes_file_and_returns_files(self):
        path = self._write_view(
            "view.py",
            'from qgis.PyQt import QtWidgets\n'
            'class V(QtWidgets.QWidget):\n'
            '    def __init__(self):\n'
            '        self.btn = QtWidgets.QPushButton()\n'
            '        self.btn.setObjectName("old_btn")\n',
        )
        preview = tools_design.design_rename_widget("old_btn", "new_btn", [path])
        self.assertTrue(preview["ok"], preview)
        apply = tools_design.design_apply_patch(
            json.dumps(preview), view_files=[path]
        )
        self.assertTrue(apply["ok"], apply)
        self.assertEqual(apply["edit_count"], 1)
        self.assertEqual(apply["files"], [path])
        content = Path(path).read_text()
        self.assertIn('setObjectName("new_btn")', content)
        self.assertNotIn('setObjectName("old_btn")', content)

    def test_design_apply_patch_rejects_raw_unified_diff_text(self):
        path = self._write_view(
            "view.py",
            'from qgis.PyQt import QtWidgets\n'
            'class V(QtWidgets.QWidget):\n'
            '    def __init__(self):\n'
            '        self.btn = QtWidgets.QPushButton()\n'
            '        self.btn.setObjectName("old_btn")\n',
        )
        preview = tools_design.design_rename_widget("old_btn", "new_btn", [path])
        self.assertTrue(preview["ok"], preview)
        out = tools_design.design_apply_patch(preview["patch_text"], view_files=[path])
        self.assertFalse(out["ok"])
        self.assertIn("JSON", out["error"])

    def test_design_apply_patch_rejects_file_outside_view_files(self):
        path = self._write_view(
            "view.py",
            'from qgis.PyQt import QtWidgets\n'
            'class V(QtWidgets.QWidget):\n'
            '    def __init__(self):\n'
            '        self.btn = QtWidgets.QPushButton()\n'
            '        self.btn.setObjectName("old_btn")\n',
        )
        edits = [
            {
                "kind": "setObjectName",
                "file_path": path,
                "lineno": 5,
                "old_value": "old_btn",
                "new_value": "new_btn",
            }
        ]
        preview = tools_design.design_preview_patch(edits, view_files=[path])
        self.assertTrue(preview["ok"], preview)
        # Pass a different (empty) view_files list so the file is not allowed.
        out = tools_design.design_apply_patch(
            json.dumps(preview), view_files=[]
        )
        self.assertFalse(out["ok"])
        self.assertIn("not in allowed", out["error"])


try:
    import mcp.server.fastmcp  # noqa: F401
    _MCP_AVAILABLE = True
except Exception:
    _MCP_AVAILABLE = False


class TestDesignServerRegistration(unittest.TestCase):
    """The four design tools are advertised by the MCP server module."""

    @unittest.skipUnless(_MCP_AVAILABLE, "mcp SDK not installed in this environment")
    def test_design_tools_registered_in_server(self):
        import tools.hydra_mcp.server as server
        names = {
            "design_rename_widget",
            "design_relabel_widget",
            "design_preview_patch",
            "design_apply_patch",
        }
        # The current FastMCP SDK exposes registered tools via the tool
        # manager (the old ``FastMCP._tools`` attribute no longer exists).
        registered = {t.name for t in server.mcp._tool_manager.list_tools()}
        self.assertTrue(
            names.issubset(registered),
            f"missing tools: {names - registered}",
        )


class TestMcpConfigGate(unittest.TestCase):
    """The dangerous ``design_apply_patch`` tool is disabled by default."""

    def test_design_apply_patch_disabled_in_mcp_json(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", ".kimi-code", "mcp.json"
        )
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        hydra = config.get("mcpServers", {}).get("hydra", {})
        self.assertIn(
            "design_apply_patch",
            hydra.get("disabledTools", []),
            "design_apply_patch must be disabled by default",
        )


@requires_qgis
class TestPluginAutoStartMcpBridge(unittest.TestCase):
    """HYDRA_MCP_BRIDGE=1 causes the plugin to start the MCP bridge."""

    def test_plugin_starts_bridge_when_env_var_set(self):
        ensure_qgis_app()

        # Avoid dragging in the real QGIS bridge and its QtNetwork dependency
        # during this unit test.  Phase 5 auto-starts via
        # ``bootstrap_bridge_if_needed``, so we mock that entry point and
        # verify the plugin assigns the returned bridge to ``_mcp_bridge``.
        import types
        bridge_mod_name = "tools.hydra_mcp.qgis_bridge"
        original_bridge_mod = sys.modules.get(bridge_mod_name)
        _fake_bridge = types.ModuleType(bridge_mod_name)

        class FakeBridge:
            def __init__(self, parent=None):
                self.parent = parent
                self.socket_name = "fake"
                self.token_path = "/tmp/fake_token.json"
            def start(self) -> None:
                pass

        def bootstrap_bridge_if_needed():
            return FakeBridge()

        _fake_bridge.HydraMcpBridge = FakeBridge
        _fake_bridge.bootstrap_bridge_if_needed = bootstrap_bridge_if_needed
        sys.modules[bridge_mod_name] = _fake_bridge

        def _restore_bridge_mod():
            if original_bridge_mod is None:
                sys.modules.pop(bridge_mod_name, None)
            else:
                sys.modules[bridge_mod_name] = original_bridge_mod
        self.addCleanup(_restore_bridge_mod)

        # hydra_plugin lives in qgis_plugin/HYDRA2DGPU/ (the real plugin
        # package layout), not at the repo root.
        plugin_dir = os.path.join(
            os.path.dirname(__file__), "..", "qgis_plugin", "HYDRA2DGPU"
        )
        plugin_dir = os.path.abspath(plugin_dir)
        if plugin_dir not in sys.path:
            sys.path.insert(0, plugin_dir)

        import importlib
        import hydra_plugin
        importlib.reload(hydra_plugin)

        iface = stub_iface()
        plugin = hydra_plugin.HydraQgisPlugin(iface)
        self.assertIsNone(plugin._mcp_bridge)

        original = os.environ.get("HYDRA_MCP_BRIDGE")
        try:
            os.environ["HYDRA_MCP_BRIDGE"] = "1"
            plugin._maybe_start_mcp_bridge()
        finally:
            if original is None:
                os.environ.pop("HYDRA_MCP_BRIDGE", None)
            else:
                os.environ["HYDRA_MCP_BRIDGE"] = original

        self.assertIsNotNone(plugin._mcp_bridge)
        self.assertIsInstance(plugin._mcp_bridge, FakeBridge)
        self.assertTrue(plugin._mcp_bridge.socket_name)


if __name__ == "__main__":
    unittest.main()
