import tempfile
import time
import numpy as np
from qgis.PyQt.QtWidgets import QApplication

def test_persistence_worker_finishes_without_error():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.workers.persistence_worker import PersistenceWorker
    from swe2d.workbench.workers.simulation_worker import ComputeResult

    with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as f:
        path = f.name

    result = ComputeResult(
        ok=True,
        h=np.array([1.0]),
        hu=np.array([0.0]),
        hv=np.array([0.0]),
        final_sim_time_s=1.0,
        n_area=1,
        area_model=np.array([1.0]),
        storage_start_model=0.0,
        source_budget_model={"rain": 0.0, "cell": 0.0, "coupling": 0.0},
        source_step_rows_model=[],
        run_duration_s=1.0,
        boundary_flux_budget_model={},
        boundary_flux_step_rows_model=[],
        run_id="r1",
        mesh_name="mesh",
        output_interval_s=1.0,
        run_perf_start=0.0,
        run_wallclock_start="now",
        run_log_start_idx=0,
        thiessen_forcing=None,
        rain_stats_acc={"rain_mm": 0.0, "excess_mm": 0.0, "samples": 0},
        max_tracking=None,
        snapshot_timesteps=[(1.0, np.array([1.0]), np.array([0.0]), np.array([0.0]))],
        coupling_snapshots={},
        precomputed_line_results=None,
    )

    view = type("View", (), {
        "log_message": lambda self, msg: None,
        "get_line_results_storage_path": lambda self: path,
        "sync_overlay_data": lambda self: None,
        "refresh_plot": lambda self: None,
        "results_table_name": lambda self, base: base,
        "length_unit_name": lambda self: "m",
        "length_scale_si_to_model": lambda self: 1.0,
        "results_data": lambda self: None,
        "update_overlay_time": lambda self, t: None,
        "runtime_log_lines": lambda self: [],
        "collect_run_log_metadata": lambda self: {},
        "persist_run_log": lambda *a, **k: None,
        "is_cancel_requested": lambda self: False,
    })()

    worker = PersistenceWorker(view, result)
    finished = []
    worker.persist_finished.connect(lambda s: finished.append(s))
    worker.start()
    deadline = time.perf_counter() + 5.0
    while worker.isRunning() and time.perf_counter() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert len(finished) == 1
    assert finished[0]["ok"] is True
