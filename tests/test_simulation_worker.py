"""Qt signal/lifecycle tests for the thin simulation worker wrapper.

The solver execution body now lives in ``swe2d.core.executor.execute_run``; these
tests mock that function to verify that ``SimulationWorker`` wires the
``Sink`` callbacks to the correct Qt signals.
"""

import threading
import time
from unittest.mock import patch

import numpy as np
from qgis.PyQt.QtWidgets import QApplication


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
        h_min=1e-6,
    )
    defaults.update(overrides)
    from swe2d.core.run_context import RunContext

    return RunContext(**defaults)


def _fake_result(ctx):
    from swe2d.core.executor import ComputeResult

    return ComputeResult(
        ok=True,
        h=ctx.h0,
        hu=ctx.hu0,
        hv=ctx.hv0,
        final_sim_time_s=ctx.run_duration_s,
        n_area=1,
        area_model=np.array([1.0]),
        storage_start_model=0.0,
        source_budget_model={"rain": 0.0, "cell": 0.0, "coupling": 0.0},
        source_step_rows_model=[],
        run_duration_s=ctx.run_duration_s,
        boundary_flux_budget_model={},
        boundary_flux_step_rows_model=[],
        run_id=ctx.run_id,
        mesh_name=ctx.mesh_name or "mesh",
        output_interval_s=ctx.output_interval_s,
        run_perf_start=0.0,
        run_wallclock_start=ctx.run_wallclock_start,
        run_log_start_idx=ctx.run_log_start_idx,
        thiessen_forcing=None,
        rain_stats_acc={"rain_mm": 0.0, "excess_mm": 0.0, "samples": 0},
        max_tracking=None,
        snapshot_timesteps=[(ctx.run_duration_s, ctx.h0, ctx.hu0, ctx.hv0)],
        coupling_snapshots={},
        save_line_results=False,
        save_coupling_results=False,
        save_mesh_results=True,
        save_run_log=True,
        save_max_only=False,
        h_min=ctx.h_min,
        pipe_cell_items=None,
        precomputed_line_results=None,
        cancelled=False,
    )


def _make_fake_execute_run(ctx, *, request_perm=False):
    """Return a stub ``execute_run`` that exercises the sink protocol."""

    def _execute_run(context, sink):
        sink.log("fake start")
        sink.progress(0.0, {})
        sink.progress(50.0, {})
        sink.progress(100.0, {})
        if request_perm:
            perm = np.array([0], dtype=np.int32)
            result = type(
                "PermutationResult",
                (),
                {
                    "sample_map": None,
                    "cell_solver_z": None,
                    "event": threading.Event(),
                    "error": None,
                },
            )()
            sink.permutation(perm, result)
        return _fake_result(context)

    return _execute_run


def test_simulation_worker_emits_progress_and_finishes():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.workers.simulation_worker import SimulationWorker

    ctx = _make_context()
    fake = _make_fake_execute_run(ctx)

    with patch("swe2d.workbench.workers.simulation_worker.execute_run", new=fake):
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


def test_simulation_worker_calls_execute_run():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.workers.simulation_worker import SimulationWorker

    ctx = _make_context()
    calls = []

    def _fake(context, sink):
        calls.append((context, sink))
        return _fake_result(context)

    with patch("swe2d.workbench.workers.simulation_worker.execute_run", new=_fake):
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
    assert len(calls) == 1
    assert calls[0][0] is ctx


def test_simulation_worker_requests_mesh_permutation_from_main_thread():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.workers.simulation_worker import SimulationWorker

    ctx = _make_context()
    fake = _make_fake_execute_run(ctx, request_perm=True)

    received = {}

    def _on_mesh_permutation_ready(cell_perm, result_holder):
        received["cell_perm"] = np.asarray(cell_perm, dtype=np.int32).copy()
        result_holder.sample_map = [{"line_id": "test"}]
        result_holder.cell_solver_z = np.array([0.0])
        result_holder.event.set()

    with patch("swe2d.workbench.workers.simulation_worker.execute_run", new=fake):
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

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
