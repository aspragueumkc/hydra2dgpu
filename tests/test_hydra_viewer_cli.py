"""Tests for tools/hydra_viewer_cli.py — headless GPU renderer CLI.

Strategy: mock the GPKG loaders (heavy infrastructure) but exercise the real
GPU render path (Phase 2.1 binding) + real Pillow PNG writer. This gives
end-to-end coverage of the CLI orchestration without the GPKG bake setup.

Auto-skipped without GPU via @pytest.mark.{solver,gpu}.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUILD_DIR = os.environ.get("HYDRA_BUILD_DIR") or os.path.join(_REPO_ROOT, "build")
for _p in (_REPO_ROOT, _BUILD_DIR):
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)

from tools import hydra_viewer_cli as cli  # noqa: E402
from tests._swe2d_test_helpers import _make_rect_mesh  # noqa: E402


def _binding_present():
    try:
        import hydra_swe2d as m
        return hasattr(m, "swe2d_gpu_render_field_to_rgba")
    except ImportError:
        return False


def _gpu_available():
    try:
        import hydra_swe2d as m
        return m.swe2d_gpu_available()
    except Exception:
        return False


@pytest.mark.solver
@pytest.mark.gpu
@unittest.skipUnless(_binding_present(), "swe2d_gpu_render_field_to_rgba not in binding")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
class TestHydraViewerCLI(unittest.TestCase):
    """End-to-end CLI tests with GPKG loaders mocked."""

    NX = 10
    NY = 4
    LX = 100.0
    LY = 40.0

    def _synthetic_mesh_and_field(self):
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(
            self.NX, self.NY, self.LX, self.LY,
        )
        info_n_cells = 2 * self.NX * self.NY  # 2 triangles per quad
        cell_x = np.zeros(info_n_cells, dtype=np.float64)
        cell_y = np.zeros(info_n_cells, dtype=np.float64)
        for ci in range(info_n_cells):
            n0 = cell_nodes[3*ci]
            n1 = cell_nodes[3*ci + 1]
            n2 = cell_nodes[3*ci + 2]
            cell_x[ci] = (node_x[n0] + node_x[n1] + node_x[n2]) / 3.0
            cell_y[ci] = (node_y[n0] + node_y[n1] + node_y[n2]) / 3.0
        # Field: ramp from 0 to 2 along x.
        field = 2.0 * (cell_x / self.LX)
        return {
            "node_x": node_x.astype(np.float64),
            "node_y": node_y.astype(np.float64),
            "node_z": node_z.astype(np.float64),
            "cell_nodes": cell_nodes.astype(np.int32),
            "cell_x": cell_x,
            "cell_y": cell_y,
        }, field

    def _make_args(self, **kwargs):
        """Build a minimal argparse.Namespace for cmd_single."""
        defaults = {
            "mode": "single",
            "gpkg": "/tmp/synthetic.gpkg",
            "run_id": "synthetic",
            "field": "depth",
            "timestep": None,
            "width": 320,
            "height": 180,
            "cmap": "turbo",
            "vmin": None,
            "vmax": None,
            "output": "/tmp/synthetic_out.png",
            "output_dir": None,
            "solver": None,
        }
        defaults.update(kwargs)
        import argparse
        return argparse.Namespace(**defaults)

    def test_build_colormap_lut_returns_256_rgba(self):
        lut = cli.build_colormap_lut("turbo")
        self.assertEqual(lut.shape, (256, 4))
        self.assertEqual(lut.dtype, np.uint8)
        # Alpha must be 255 for every entry.
        self.assertTrue((lut[:, 3] == 255).all())
        # At least one non-zero color channel per entry.
        self.assertTrue((lut[:, :3].sum(axis=1) > 0).all())

    def test_build_colormap_lut_unknown_raises(self):
        with self.assertRaises(ValueError):
            cli.build_colormap_lut("not-a-cmap")

    def test_save_png_writes_valid_file(self):
        rgba = np.zeros((40, 60, 4), dtype=np.uint8)
        rgba[..., 0] = 200  # red
        rgba[..., 3] = 255  # opaque
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "sub", "frame.png")
            cli.save_png(rgba, out)
            self.assertTrue(os.path.isfile(out))
            self.assertGreater(os.path.getsize(out), 0)
            # Validate PNG header.
            with open(out, "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")

    def test_cmd_single_writes_png(self):
        """End-to-end: mock GPKG loaders, real GPU render, real PNG write."""
        mesh, field = self._synthetic_mesh_and_field()

        with tempfile.TemporaryDirectory() as d:
            out_png = os.path.join(d, "frame.png")
            args = self._make_args(output=out_png)

            with patch.object(cli, "load_mesh_from_gpkg", return_value=mesh), \
                 patch.object(cli, "load_field_from_gpkg", return_value=field):
                rc = cli.cmd_single(args)
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.isfile(out_png))
            self.assertGreater(os.path.getsize(out_png), 0)

    def test_cmd_single_propagates_vmin_vmax(self):
        """Explicit vmin/vmax must be passed through (no override)."""
        mesh, field = self._synthetic_mesh_and_field()
        import hydra_swe2d as m

        with tempfile.TemporaryDirectory() as d:
            out_png = os.path.join(d, "frame.png")
            args = self._make_args(output=out_png, vmin=0.25, vmax=1.75)

            captured = {}

            real_render = cli.render_field_rgba

            def spy_render(solver, field_key, vmin, vmax, *args_, **kw):
                captured["vmin"] = vmin
                captured["vmax"] = vmax
                return real_render(solver, field_key, vmin, vmax, *args_, **kw)

            with patch.object(cli, "load_mesh_from_gpkg", return_value=mesh), \
                 patch.object(cli, "load_field_from_gpkg", return_value=field), \
                 patch.object(cli, "render_field_rgba", spy_render):
                cli.cmd_single(args)
            self.assertEqual(captured["vmin"], 0.25)
            self.assertEqual(captured["vmax"], 1.75)

    def test_main_single_returns_2_on_missing_required_args(self):
        """single mode without --gpkg/--run-id/--output → exit code 2."""
        rc = cli.main(["--mode", "single"])
        self.assertEqual(rc, 2)

    def test_argparse_rejects_unknown_mode(self):
        """argparse itself rejects unknown --mode values."""
        with self.assertRaises(SystemExit):
            cli.main(["--mode", "bogus"])

    def test_cmd_multi_writes_one_png_per_timestep(self):
        """multi mode: render every timestep → N PNGs in output dir."""
        mesh, _ = self._synthetic_mesh_and_field()
        # 5 synthetic timesteps, with the field varying per timestep.
        timesteps = [0.0, 0.5, 1.0, 1.5, 2.0]

        def fake_load_field(gpkg, run_id, field, ts):
            # Vary the field linearly with time so each frame is distinct.
            n = mesh["cell_x"].size
            return np.linspace(0.0, 2.0, n) * (1.0 + ts)

        with tempfile.TemporaryDirectory() as d:
            args = self._make_args(mode="multi", output_dir=d)
            with patch.object(cli, "load_mesh_from_gpkg", return_value=mesh), \
                 patch.object(cli, "load_field_from_gpkg", side_effect=fake_load_field), \
                 patch.object(cli, "load_timesteps_from_gpkg", return_value=timesteps):
                rc = cli.cmd_multi(args)
            self.assertEqual(rc, 0)
            frames = sorted(p for p in os.listdir(d) if p.startswith("frame_"))
            self.assertEqual(len(frames), 5)
            for fname in frames:
                self.assertGreater(os.path.getsize(os.path.join(d, fname)), 0)

    def test_main_multi_returns_2_on_missing_required_args(self):
        rc = cli.main(["--mode", "multi"])
        self.assertEqual(rc, 2)

    def test_cmd_live_writes_one_png_per_snapshot_then_stops(self):
        """live mode with --max-frames exits cleanly after N snapshots.

        Patches the snapshot readback binding + the render call so the
        test runs without a real GPU/solver.  cmd_live no longer wraps
        the snapshot readback in a LiveSnapshotReader class — it calls
        hydra_swe2d.swe2d_gpu_snapshot_count / swe2d_gpu_read_snapshots
        directly.
        """
        mesh, _ = self._synthetic_mesh_and_field()

        n_cells = mesh["cell_x"].size
        # Snapshots as (t_s, h) pairs.
        snapshots = [
            (0.0, np.full(n_cells, 0.5, dtype=np.float64)),
            (1.0, np.full(n_cells, 1.0, dtype=np.float64)),
            (2.0, np.full(n_cells, 1.5, dtype=np.float64)),
            (3.0, np.full(n_cells, 2.0, dtype=np.float64)),
        ]
        idx = {"n": 0}

        def fake_snapshot_count(_solver):
            # 0 until we've exhausted our fake list.
            return 1 if idx["n"] < len(snapshots) else 0

        def fake_read_snapshots(_solver):
            i = idx["n"]
            if i >= len(snapshots):
                return {}
            t_s, h = snapshots[i]
            idx["n"] += 1
            return {
                "t_s": np.array([t_s], dtype=np.float64),
                "h":   np.atleast_2d(h),
                "hu":  np.zeros((1, n_cells), dtype=np.float64),
                "hv":  np.zeros((1, n_cells), dtype=np.float64),
            }

        # Mock the GPU render call — the fake solver isn't a real PySolver
        # so the binding would reject it.  Live mode's contract is just
        # "render the latest snapshot and save PNG"; the render itself is
        # already covered by Phase 2.1 + 2.2 tests.
        def fake_render(solver, field_key, vmin, vmax, width, height,
                        cell_x, cell_y, lut):
            return np.zeros((height, width, 4), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as d:
            args = self._make_args(
                mode="live", output_dir=d, max_frames=3,
                gpkg="/tmp/synthetic.gpkg", run_id="synthetic", solver="fake",
                width=160, height=90,
            )
            # cmd_live does `import hydra_swe2d as _h` inside the function —
            # we patch both the module-level access and the function body
            # access by creating a fake module attribute.
            import hydra_swe2d as _h_real
            _h_real.swe2d_gpu_snapshot_count = fake_snapshot_count
            _h_real.swe2d_gpu_read_snapshots = fake_read_snapshots
            try:
                with patch.object(cli, "load_mesh_from_gpkg", return_value=mesh), \
                     patch.object(cli, "render_field_rgba", side_effect=fake_render):
                    rc = cli.cmd_live(args)
            finally:
                # Restore the original bindings (don't pollute module state
                # for other tests).
                import hydra_swe2d as _h_real2
                if hasattr(_h_real2, "swe2d_gpu_snapshot_count"):
                    del _h_real2.swe2d_gpu_snapshot_count
                if hasattr(_h_real2, "swe2d_gpu_read_snapshots"):
                    del _h_real2.swe2d_gpu_read_snapshots
            self.assertEqual(rc, 0)
            frames = sorted(p for p in os.listdir(d) if p.startswith("live_"))
            self.assertEqual(len(frames), 3)
            for fname in frames:
                self.assertGreater(os.path.getsize(os.path.join(d, fname)), 0)

    def test_main_live_returns_2_on_missing_required_args(self):
        rc = cli.main(["--mode", "live"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()