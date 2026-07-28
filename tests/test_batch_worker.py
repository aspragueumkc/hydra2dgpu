"""Tests for BatchWorker thread."""
import os
import tempfile
import threading
from unittest.mock import patch, MagicMock

from swe2d.workbench.workers.batch_worker import BatchWorker
from swe2d.workbench.services.batch_manager import BatchConfig


def test_batch_worker_sequential_mode():
    """Test that BatchWorker runs sims sequentially when max_workers=1."""
    config = BatchConfig(max_workers=1, results_dir=tempfile.mkdtemp(), mesh_path="/tmp/mesh.gpkg")
    cancel_event = threading.Event()
    params = [{"id": "sim_001", "params": {"n_mann": 0.03}}]

    worker = BatchWorker(params_list=params, config=config, cancel_event=cancel_event)

    with patch.object(worker, "_launch_sim") as mock_launch:
        mock_launch.return_value = None
        worker._run_sequential()
        assert mock_launch.call_count == 1
        mock_launch.assert_called_with(params[0])


def test_batch_worker_cancel_stops_loop():
    """Test that cancelling stops the sequential loop."""
    config = BatchConfig(max_workers=1, results_dir=tempfile.mkdtemp(), mesh_path="/tmp/mesh.gpkg")
    cancel_event = threading.Event()
    params = [
        {"id": "sim_001", "params": {}},
        {"id": "sim_002", "params": {}},
        {"id": "sim_003", "params": {}},
    ]

    worker = BatchWorker(params_list=params, config=config, cancel_event=cancel_event)

    call_count = 0

    def fake_launch(p):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            cancel_event.set()

    with patch.object(worker, "_launch_sim", side_effect=fake_launch):
        worker._run_sequential()
        assert call_count == 2  # sim_001 ran, sim_002 set cancel, sim_003 skipped


def test_batch_worker_creates_log_dir():
    """Test that _launch_sim creates the batch_runs log directory."""
    config = BatchConfig(max_workers=1, results_dir=tempfile.mkdtemp(), mesh_path="/tmp/mesh.gpkg")
    cancel_event = threading.Event()
    worker = BatchWorker(
        params_list=[{"id": "sim_001", "params": {}}],
        config=config, cancel_event=cancel_event,
    )

    log_dir = os.path.join(config.results_dir, "batch_runs")
    assert not os.path.exists(log_dir)

    with patch("subprocess.Popen") as MockPopen:
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_proc.returncode = 0
        mock_proc.poll.return_value = 0
        MockPopen.return_value = mock_proc

        worker._launch_sim({"id": "sim_001", "params": {}})
        assert os.path.isdir(log_dir)


def test_batch_worker_failed_exit_emits_failed():
    """Test that a non-zero exit code emits sim_failed."""
    config = BatchConfig(max_workers=1, results_dir=tempfile.mkdtemp(), mesh_path="/tmp/mesh.gpkg")
    cancel_event = threading.Event()
    worker = BatchWorker(
        params_list=[{"id": "sim_001", "params": {}}],
        config=config, cancel_event=cancel_event,
    )
    failed_sids = []
    worker._sim_failed.connect(lambda sid, msg: failed_sids.append(sid))

    with patch("subprocess.Popen") as MockPopen:
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])
        mock_proc.returncode = 1
        mock_proc.poll.return_value = 1
        MockPopen.return_value = mock_proc

        worker._launch_sim({"id": "sim_001", "params": {}})
        assert "sim_001" in failed_sids

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
