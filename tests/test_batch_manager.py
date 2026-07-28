"""Tests for BatchManager service."""
import threading
from unittest.mock import MagicMock, patch

from swe2d.workbench.services.batch_manager import BatchManager, SimState, BatchConfig


def test_batch_manager_initial_state():
    bm = BatchManager()
    assert not bm.is_running()
    assert bm.get_status() == {}


def test_start_batch_sets_running():
    bm = BatchManager()
    config = BatchConfig(max_workers=1, results_dir="/tmp/test", mesh_path="/tmp/mesh.gpkg")
    with patch("swe2d.workbench.workers.batch_worker.BatchWorker") as MockWorker:
        mock_worker = MagicMock()
        MockWorker.return_value = mock_worker
        bm.start_batch([{"id": "sim_001", "params": {}}], config)
        assert bm.is_running()
        mock_worker.start.assert_called_once()


def test_cancel_batch_sets_event():
    bm = BatchManager()
    config = BatchConfig(max_workers=1, results_dir="/tmp/test", mesh_path="/tmp/mesh.gpkg")
    with patch("swe2d.workbench.workers.batch_worker.BatchWorker") as MockWorker:
        mock_worker = MagicMock()
        MockWorker.return_value = mock_worker
        bm.start_batch([{"id": "sim_001", "params": {}}], config)
        bm.cancel_batch()
        assert bm._cancel_event.is_set()


def test_get_status_returns_sim_states():
    bm = BatchManager()
    bm._sim_states["sim_001"] = SimState(
        sim_id="sim_001", status="completed", progress=100.0,
        status_text="Done", results_path="/tmp/out.gpkg",
        error=None, log_file="/tmp/sim_001.log",
    )
    status = bm.get_status()
    assert "sim_001" in status
    assert status["sim_001"].status == "completed"

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
