import threading
import time
import numpy as np
from unittest.mock import MagicMock
from qgis.PyQt.QtWidgets import QApplication

def test_simulation_worker_emits_progress_and_finishes():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.workers.simulation_worker import SimulationWorker
    from swe2d.workbench.workers.run_context import RunContext

    ctx = RunContext(
        run_id="r1",
        run_wallclock_start="now",
        run_log_start_idx=0,
        run_duration_s=0.05,
        output_interval_s=0.1,
        line_output_interval_s=0.1,
        node_x=np.array([0.0, 1.0, 0.0]),
        node_y=np.array([0.0, 0.0, 1.0]),
        node_z=np.array([0.0, 0.0, 0.0]),
        cell_nodes=np.array([[0, 1, 2]], dtype=np.int32),
        h0=np.array([1.0]),
        hu0=np.array([0.0]),
        hv0=np.array([0.0]),
        cell_areas=np.array([0.5]),
    )

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
