"""Tests for swe2d.cli.commands — single source of truth for batch subprocess commands.

Both the CLI ``batch_runner`` and the workbench ``BatchWorker`` must produce
the same subprocess argv for the same spec — these tests guard that
contract.  See ``docs/CLI_FIRST_REFACTOR_PLAN.md`` Phase 3.4 for the
architectural rule.
"""
import json
import os
import sys
import tempfile
import unittest


class TestBuildRunCommand(unittest.TestCase):
    """Test the explicit-arg builder (used by the workbench BatchWorker)."""

    def test_build_run_command_minimal(self):
        """Minimal call: only spec_path, no status file."""
        from swe2d.cli.commands import build_run_command

        cmd = build_run_command("/tmp/spec.json")
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1:4], ["-m", "swe2d.cli", "replay"])
        self.assertIn("--replay-file", cmd)
        self.assertEqual(cmd[cmd.index("--replay-file") + 1], "/tmp/spec.json")
        # No status-file-path when not requested.
        self.assertNotIn("--status-file-path", cmd)

    def test_build_run_command_with_status_file(self):
        """``status_file_path`` is appended after the spec."""
        from swe2d.cli.commands import build_run_command

        cmd = build_run_command(
            "/tmp/spec.json",
            status_file_path="/tmp/status.json",
            status_interval_s=2.5,
        )
        self.assertIn("--status-file-path", cmd)
        idx = cmd.index("--status-file-path")
        self.assertEqual(cmd[idx + 1], "/tmp/status.json")
        self.assertIn("--status-interval", cmd)
        self.assertEqual(cmd[cmd.index("--status-interval") + 1], "2.5")


class TestBuildRunCommandForParams(unittest.TestCase):
    """Test the dict-arg builder (used by the CLI batch_runner)."""

    def test_replay_spec_writes_temp_file(self):
        """``swe2d-replay/1`` payload → writes temp file + replay command."""
        from swe2d.cli.commands import build_run_command_for_params, cleanup_temp_specs

        spec = {
            "schema_version": "swe2d-replay/1",
            "run_id": "test1",
            "mesh": {"mesh_name": "m", "gpkg_path": "/tmp/m.gpkg"},
            "params": {"n_mann": 0.04},
        }
        try:
            cmd = build_run_command_for_params(spec)
            self.assertIn("replay", cmd)
            self.assertIn("--replay-file", cmd)
            # The temp file exists and contains the spec
            spec_path = cmd[cmd.index("--replay-file") + 1]
            self.assertTrue(os.path.exists(spec_path))
            with open(spec_path) as f:
                data = json.load(f)
            self.assertEqual(data["run_id"], "test1")
        finally:
            cleanup_temp_specs()

    def test_legacy_spec_uses_run_command(self):
        """Non-replay spec → ``run`` command with mesh + JSON args."""
        from swe2d.cli.commands import build_run_command_for_params

        legacy = {
            "mesh_path": "/tmp/m.gpkg",
            "params": {"n_mann": 0.04},
        }
        cmd = build_run_command_for_params(legacy, results_gpkg="/tmp/r.gpkg")
        self.assertIn("run", cmd)
        # Mesh path comes immediately after "run"
        run_idx = cmd.index("run")
        self.assertEqual(cmd[run_idx + 1], "/tmp/m.gpkg")
        # JSON string is the next positional
        self.assertIn("n_mann", cmd[run_idx + 2])
        # --results is appended
        self.assertIn("--results", cmd)
        self.assertEqual(cmd[cmd.index("--results") + 1], "/tmp/r.gpkg")

    def test_shared_builder_used_by_both_paths(self):
        """Both batch_runner and batch_worker call the same builder.

        We invoke the builder directly and check the resulting argv
        shape: ``[python, -m, swe2d.cli, replay, --replay-file, ...,
        --status-file-path, ...]``.
        """
        from swe2d.cli.commands import build_run_command_for_params, cleanup_temp_specs

        spec = {
            "schema_version": "swe2d-replay/1",
            "run_id": "shared1",
            "mesh": {"mesh_name": "m", "gpkg_path": "/tmp/m.gpkg"},
        }
        try:
            cmd = build_run_command_for_params(
                spec,
                status_file_path="/tmp/status.json",
                status_interval_s=1.0,
            )
            self.assertEqual(cmd[0], sys.executable)
            self.assertEqual(cmd[1:4], ["-m", "swe2d.cli", "replay"])
            # Both flags honored.
            self.assertIn("--replay-file", cmd)
            self.assertIn("--status-file-path", cmd)
            self.assertIn("--status-interval", cmd)
        finally:
            cleanup_temp_specs()


class TestIsReplaySpec(unittest.TestCase):
    """``is_replay_spec`` distinguishes replay payloads from legacy forms."""

    def test_replay_payload(self):
        from swe2d.cli.commands import is_replay_spec

        self.assertTrue(is_replay_spec({"schema_version": "swe2d-replay/1"}))

    def test_legacy_payload(self):
        from swe2d.cli.commands import is_replay_spec

        self.assertFalse(is_replay_spec({"mesh_path": "/tmp/m.gpkg"}))

    def test_empty_dict(self):
        from swe2d.cli.commands import is_replay_spec

        self.assertFalse(is_replay_spec({}))


class TestBatchRunnerUsesSharedBuilder(unittest.TestCase):
    """The CLI batch_runner._run_one must delegate to ``commands.build_run_command_for_params``.

    We assert this by inspecting the source of ``batch_runner._run_one`` —
    the canonical source of truth for the subprocess argv is in
    ``commands.py``, not duplicated in ``batch_runner.py``.
    """

    def test_run_one_calls_shared_builder(self):
        import inspect
        from swe2d.cli import batch_runner

        # ``_run_one`` is a closure inside ``run_batch`` — inspect the
        # outer function instead.
        src = inspect.getsource(batch_runner.run_batch)
        self.assertIn("build_run_command_for_params", src)
        # No more inline list literals forming the cmd list.
        self.assertNotIn("sys.executable, \"-m\", \"swe2d.cli\"", src)
        # The BatchOrchestrator class is gone (Phase 3.4).
        self.assertFalse(hasattr(batch_runner, "BatchOrchestrator"))


class TestBatchWorkerUsesSharedBuilder(unittest.TestCase):
    """The workbench BatchWorker._build_command must delegate to the same builder.

    This test loads BatchWorker only if PyQt5 is available.  Without
    PyQt5 the test skips — the CLI part of the contract is exercised
    by ``TestBatchRunnerUsesSharedBuilder``.
    """

    def test_build_command_calls_shared_builder(self):
        try:
            from swe2d.workbench.workers.batch_worker import BatchWorker
        except Exception as exc:
            self.skipTest(f"PyQt5 not available: {exc}")
            return
        import inspect
        src = inspect.getsource(BatchWorker._build_command)
        self.assertIn("build_run_command_for_params", src)
        # The docstring's "poll its status file" guarantee: the new
        # _build_command must pass --status-file-path.
        self.assertIn("status_file_path", src)


if __name__ == "__main__":
    unittest.main()
