import time
from unittest.mock import MagicMock, patch
from qgis.PyQt.QtWidgets import QApplication

def test_run_controller_starts_simulation_worker():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.controllers.run_controller import RunController
    ctrl = RunController(view=MagicMock())
    with patch("swe2d.workbench.controllers.run_controller.SimulationWorker") as MockW, \
         patch.object(ctrl, "_build_run_context", return_value=MagicMock()):
        instance = MagicMock()
        MockW.return_value = instance
        ctrl.on_run()
        instance.start.assert_called_once()


def test_run_controller_builds_context_from_view():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.controllers.run_controller import RunController
    ctrl = RunController(view=MagicMock())
    with patch.object(ctrl, "_build_run_context", return_value=MagicMock()) as mock_build, \
         patch("swe2d.workbench.controllers.run_controller.SimulationWorker"):
        ctrl.on_run()
        mock_build.assert_called_once()


def test_run_controller_skips_when_worker_running():
    app = QApplication.instance() or QApplication([])
    from swe2d.workbench.controllers.run_controller import RunController
    mock_view = MagicMock()
    ctrl = RunController(view=mock_view)
    fake_worker = MagicMock()
    fake_worker.isRunning.return_value = True
    ctrl._simulation_worker = fake_worker
    ctrl.on_run()
    mock_view._log.assert_called_once()
    assert "already active" in mock_view._log.call_args[0][0]
