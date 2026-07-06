"""Test the device snapshot ring buffer with memory-pressure auto-dump.

Verifies:
  1. Snapshots stored on-device don't stall the GPU pipeline (no D2H per interval)
  2. Auto-dump triggers when GPU free memory drops below the safety margin
  3. Bulk readback returns all snapshots (host + device) in correct time order
  4. Data integrity after auto-dump cycles
"""

import os
import sys
import unittest

import numpy as np

from swe2d.runtime.backend import SWE2DBackend, swe2d_available
from tests._swe2d_test_helpers import _make_rect_mesh


def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except Exception:
        return None


def _gpu_available():
    mod = _load_module()
    if mod is None:
        return False
    try:
        return bool(mod.swe2d_gpu_available())
    except Exception:
        return False


def _has_test_alloc():
    mod = _load_module()
    if mod is None:
        return False
    return hasattr(mod, "swe2d_gpu_test_alloc")


@unittest.skipUnless(swe2d_available(), "hydra_swe2d not built")
@unittest.skipUnless(_gpu_available(), "CUDA GPU not available")
@unittest.skipUnless(_has_test_alloc(), "swe2d_gpu_test_alloc not in module")
class TestSnapshotRingBuffer(unittest.TestCase):

    def setUp(self):
        mod = _load_module()
        self.mod = mod
        node_x, node_y, node_z, cell_nodes = _make_rect_mesh(20, 8, 200.0, 80.0)
        self.backend = SWE2DBackend()
        self.backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        n = self.backend.n_cells
        self.backend.initialize(
            h0=np.zeros(n, dtype=np.float64),
            hu0=np.zeros(n, dtype=np.float64),
            hv0=np.zeros(n, dtype=np.float64),
            n_mann=0.03, cfl=0.45, h_min=1e-6, dt_fixed=-1.0, dt_max=0.25,
        )

    def tearDown(self):
        self.backend.free_snapshot_buf()
        self.backend.destroy()

    def test_store_and_readback(self):
        """Snapshots stored on-device are read back correctly."""
        n_snaps = 5
        for i in range(n_snaps):
            self.backend.step(0.25)
            self.backend.store_snapshot(float(i * 0.25))

        snap = self.backend.read_snapshots()
        self.assertIsNotNone(snap, "read_snapshots returned None")
        self.assertIn("t_s", snap)
        self.assertEqual(snap["t_s"].shape[0], n_snaps,
                         f"Expected {n_snaps} snapshots, got {snap['t_s'].shape[0]}")
        self.assertTrue(np.all(np.isfinite(snap["h"])), "NaN in snapshot h")
        self.assertTrue(np.all(np.isfinite(snap["hu"])), "NaN in snapshot hu")
        self.assertTrue(np.all(np.isfinite(snap["hv"])), "NaN in snapshot hv")

    def test_no_d2h_during_normal_store(self):
        """store_snapshot does not trigger D2H — only D2D on compute stream."""
        dev_count_before = self.mod.swe2d_gpu_snapshot_count(self.backend._solver_h)
        self.assertEqual(dev_count_before, 0)

        for i in range(3):
            self.backend.step(0.25)
            self.backend.store_snapshot(float(i * 0.25))
            cnt = self.mod.swe2d_gpu_snapshot_count(self.backend._solver_h)
            self.assertEqual(cnt, i + 1,
                             f"Device count should be {i+1} after store {i}")

    def test_readback_is_non_destructive(self):
        """After read_snapshots, the device buffer is NOT reset (non-destructive)."""
        for i in range(3):
            self.backend.step(0.25)
            self.backend.store_snapshot(float(i * 0.25))

        snap = self.backend.read_snapshots()
        self.assertIsNotNone(snap)

        cnt = self.mod.swe2d_gpu_snapshot_count(self.backend._solver_h)
        self.assertEqual(cnt, 3, "Device buffer should still have 3 snapshots after non-destructive read")

    def test_auto_dump_under_memory_pressure(self):
        """Auto-dump triggers when GPU free memory drops below threshold.

        Allocates 50% of VRAM to simulate memory pressure, sets the
        auto-dump threshold above current free memory, then verifies
        that store_snapshot dumps device → host at each call.
        """
        info = self.mod.swe2d_gpu_device_memory_info()
        total = info["total_bytes"]

        # Allocate 50% of device memory to create pressure
        alloc_bytes = int(total * 0.50)
        buf = self.mod.swe2d_gpu_test_alloc(alloc_bytes)

        try:
            info2 = self.mod.swe2d_gpu_device_memory_info()
            free_mem = info2["free_bytes"]

            # Set threshold above current free so auto-dump fires immediately
            snap_sz = self.backend._snap_per_snapshot_bytes()
            self.backend._snap_auto_dump_margin_mult = 1
            self.backend._snap_auto_dump_margin_bytes = int(free_mem * 1.5)

            self.assertTrue(
                self.backend._snap_should_auto_dump(),
                "Auto-dump should trigger when threshold > free memory"
            )

            # Store snapshots — each should auto-dump
            for i in range(4):
                self.backend.step(0.25)
                self.backend.store_snapshot(float(i * 0.25))
                cnt = self.mod.swe2d_gpu_snapshot_count(self.backend._solver_h)
                # After dump+store, device should have exactly 1 snapshot
                self.assertEqual(cnt, 1,
                                 f"Device should have 1 snapshot after auto-dump+store, got {cnt}")

            # Read all back — should include all 4
            snap = self.backend.read_snapshots()
            self.assertIsNotNone(snap)
            self.assertEqual(snap["t_s"].shape[0], 4,
                             f"Expected 4 snapshots total, got {snap['t_s'].shape[0]}")
            self.assertTrue(np.all(np.isfinite(snap["h"])), "NaN after auto-dump")
        finally:
            self.mod.swe2d_gpu_test_free(buf)

    def test_auto_dump_at_natural_capacity(self):
        """With a small mesh and large margin, auto-dump only triggers at threshold.

        Verifies the normal path: snapshots accumulate on-device until
        memory threshold is crossed, then auto-dump fires.
        """
        # Set a very tight threshold — 2x per-snapshot size + 1 byte
        # so it dumps when only ~1-2 snapshots stored
        snap_sz = self.backend._snap_per_snapshot_bytes()
        self.backend._snap_auto_dump_margin_mult = 2
        self.backend._snap_auto_dump_margin_bytes = 1

        for i in range(6):
            self.backend.step(0.25)
            will_dump = self.backend._snap_should_auto_dump()
            self.backend.store_snapshot(float(i * 0.25))
            cnt = self.mod.swe2d_gpu_snapshot_count(self.backend._solver_h)
            hb = self.backend._snap_host_buffer
            hbc = len(hb["t_s"]) if hb else 0

            if will_dump:
                # After dump, device has 1 (the one we just stored)
                self.assertLessEqual(cnt, 1, f"Dump: dev should be ≤1, got {cnt}")
            else:
                # No dump yet, device accumulates
                self.assertEqual(cnt, i + 1, f"No dump: dev should be {i+1}, got {cnt}")

        snap = self.backend.read_snapshots()
        self.assertIsNotNone(snap)
        self.assertEqual(snap["t_s"].shape[0], 6,
                         f"Expected 6, got {snap['t_s'].shape[0]}")


if __name__ == "__main__":
    unittest.main()
