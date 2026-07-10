import unittest
import os
import sys

import numpy as np

import swe2d.runtime.backend as backend_mod
from swe2d.extensions.extension_models import SolverModelOptions


class _FakeModuleBase:
    def swe2d_gpu_available(self):
        return True

    def swe2d_build_mesh(self, *args, **kwargs):
        return {"mesh": True}

    def swe2d_mesh_info(self, _mesh_h):
        return {"n_cells": 1}

    def swe2d_boundary_edges(self, _mesh_h):
        empty_i = np.empty(0, dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_i, empty_i, empty_i, empty_f

    def swe2d_get_cell_perm(self, _mesh_h):
        return np.array([0], dtype=np.int32)

    def swe2d_destroy(self, _solver_h):
        return None


class _FakeModuleNew(_FakeModuleBase):
    def __init__(self):
        self.create_kwargs = None
        self.last_run_diag_batch_size = None

    def swe2d_create_solver(self, *args, **kwargs):
        self.create_kwargs = dict(kwargs)
        return {"solver": True}

    def swe2d_run_to_time(self, _solver_h, t_end, dt_request, diag_batch_size):
        self.last_run_diag_batch_size = int(diag_batch_size)
        return {
            "diags": [],
            "steps_completed": 0,
            "cancelled": False,
            "final_time": float(t_end),
        }


class _FakeModuleOld(_FakeModuleBase):
    def __init__(self):
        self.create_kwargs = None

    def swe2d_create_solver(self, *args, **kwargs):
        # Emulate pybind11 signature mismatch from an older extension build.
        if kwargs:
            raise TypeError("incompatible function arguments")
        self.create_kwargs = {}
        return {"solver": True}


class TestSWE2DBackendTinyModeConfig(unittest.TestCase):
    def setUp(self):
        self._old_mod = backend_mod._swe2d_mod
        self._old_err = backend_mod._swe2d_load_error

    def tearDown(self):
        backend_mod._swe2d_mod = self._old_mod
        backend_mod._swe2d_load_error = self._old_err

    def _build_minimal_mesh(self, backend):
        node_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.array([0, 1, 2], dtype=np.int32)
        backend.build_mesh(node_x, node_y, node_z, cell_nodes)

    def test_initialize_passes_tiny_mode_kwargs_to_new_extension(self):
        fake = _FakeModuleNew()
        backend_mod._swe2d_mod = fake
        backend_mod._swe2d_load_error = None

        b = backend_mod.SWE2DBackend()
        self._build_minimal_mesh(b)
        b.initialize(np.array([0.1], dtype=np.float64))

        self.assertIsNotNone(fake.create_kwargs)
        self.assertEqual(fake.create_kwargs["gpu_diag_sync_interval_steps"], 50)
        self.assertEqual(fake.create_kwargs["tiny_mode"], 1)
        self.assertEqual(fake.create_kwargs["tiny_cell_threshold"], 8000)
        self.assertEqual(fake.create_kwargs["tiny_edge_threshold"], 24000)
        self.assertEqual(fake.create_kwargs["tiny_wet_cell_threshold"], 2000)
        self.assertNotIn("tiny_persistent_chunk_substeps", fake.create_kwargs)
        self.assertNotIn("tiny_active_compaction_stride_steps", fake.create_kwargs)
        self.assertNotIn("tiny_enable_active_compaction", fake.create_kwargs)

    def test_initialize_compat_filters_new_kwargs_for_old_extension(self):
        fake = _FakeModuleOld()
        backend_mod._swe2d_mod = fake
        backend_mod._swe2d_load_error = None

        b = backend_mod.SWE2DBackend()
        self._build_minimal_mesh(b)
        b.initialize(np.array([0.1], dtype=np.float64))

        self.assertEqual(fake.create_kwargs, {})

    def test_run_uses_zero_batching_for_all_tiny_modes(self):
        for mode in (0, 1, 2, 3):
            fake = _FakeModuleNew()
            backend_mod._swe2d_mod = fake
            backend_mod._swe2d_load_error = None

            b = backend_mod.SWE2DBackend()
            self._build_minimal_mesh(b)
            b.initialize(
                np.array([0.1], dtype=np.float64),
                tiny_mode=mode,
            )
            b.run(t_end=1.0, dt_request=0.1)

            self.assertEqual(fake.last_run_diag_batch_size, 0, f"mode={mode}")


if __name__ == "__main__":
    unittest.main()
