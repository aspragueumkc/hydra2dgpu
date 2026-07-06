import time
from unittest.mock import patch

import numpy as np
from qgis.PyQt.QtWidgets import QApplication


def _make_mock_backend_class(ctx, cell_perm=None):
    """Return a mock backend class that captures the supplied RunContext."""

    class MockBackend:
        _cell_perm = cell_perm
        _mesh_h = None
        instances = []

        def __init__(self):
            self.destroyed = False
            MockBackend.instances.append(self)

        def build_mesh(self, *args, **kwargs):
            pass

        def initialize(self, *args, **kwargs):
            pass

        def supports_dynamic_boundary_update(self):
            return True

        def boundary_edge_cells(self):
            return None

        def step(self, dt_request):
            return {
                "dt": float(ctx.dt_request),
                "max_courant": 0.1,
                "wet_cells": 1,
                "gpu_active": False,
            }

        def store_snapshot(self, t_s):
            pass

        def read_snapshots(self):
            return {}

        def get_state(self):
            return ctx.h0.copy(), ctx.hu0.copy(), ctx.hv0.copy()

        def get_max_tracking(self):
            return None

        def gpu_active(self):
            return False

        def destroy(self):
            self.destroyed = True

    return MockBackend


def _make_context(**overrides):
    defaults = dict(
        run_id="r1",
        run_wallclock_start="now",
        run_log_start_idx=0,
        run_duration_s=0.05,
        output_interval_s=0.1,
        dt_cfg=0.05,
        dt_request=0.05,
        node_x=np.array([0.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        node_z=np.array([0.0, 0.0, 0.0]),
        cell_nodes=np.array([[0, 1, 2]], dtype=np.int32),
        h0=np.array([1.0]),
        hu0=np.array([0.0]),
        hv0=np.array([0.0]),
        cell_areas=np.array([0.5]),
        mesh_cell_areas=lambda: np.array([0.5]),
    )
    defaults.update(overrides)
    from swe2d.workbench.workers.run_context import RunContext
    return RunContext(**defaults)


def test_simulation_worker_emits_progress_and_finishes():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.workers.simulation_worker import SimulationWorker

    ctx = _make_context()
    MockBackend = _make_mock_backend_class(ctx)
    MockBackend.instances.clear()

    with patch("swe2d.workbench.workers.simulation_worker.SWE2DBackend", new=MockBackend):
        worker = SimulationWorker(ctx)
        progress = []
        worker.progress_percent.connect(progress.append)
        worker.compute_finished.connect(lambda r: progress.append("done"))
        worker.start()
        deadline = time.perf_counter() + 5.0
        while worker.isRunning() and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    assert "done" in progress
    assert 100 in progress


def test_simulation_worker_calls_backend_destroy():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.workers.simulation_worker import SimulationWorker

    ctx = _make_context()
    MockBackend = _make_mock_backend_class(ctx)
    MockBackend.instances.clear()

    with patch("swe2d.workbench.workers.simulation_worker.SWE2DBackend", new=MockBackend):
        worker = SimulationWorker(ctx)
        finished = []
        worker.compute_finished.connect(finished.append)
        worker.start()
        deadline = time.perf_counter() + 5.0
        while worker.isRunning() and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    assert len(finished) == 1
    assert len(MockBackend.instances) == 1
    assert MockBackend.instances[0].destroyed is True


def test_simulation_worker_requests_mesh_permutation_from_main_thread():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.workers.simulation_worker import SimulationWorker

    ctx = _make_context()
    MockBackend = _make_mock_backend_class(ctx, cell_perm=np.array([0], dtype=np.int32))
    MockBackend.instances.clear()

    received = {}

    def _on_mesh_permutation_ready(cell_perm, result_holder):
        received["cell_perm"] = np.asarray(cell_perm, dtype=np.int32).copy()
        result_holder.sample_map = [{"line_id": "test"}]
        result_holder.cell_solver_z = np.array([0.0])
        result_holder.event.set()

    with patch("swe2d.workbench.workers.simulation_worker.SWE2DBackend", new=MockBackend):
        worker = SimulationWorker(ctx)
        worker.mesh_permutation_ready.connect(_on_mesh_permutation_ready)
        finished = []
        worker.compute_finished.connect(finished.append)
        worker.start()
        deadline = time.perf_counter() + 5.0
        while worker.isRunning() and time.perf_counter() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    assert len(finished) == 1
    assert "cell_perm" in received
    assert received["cell_perm"].size == 1
