"""BatchWorker: QThread that orchestrates batch simulation subprocesses.

Runs on a background thread. Launches subprocesses, polls their status,
writes per-sim log files, and emits progress signals via BatchManager.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from qgis.PyQt.QtCore import QThread, pyqtSignal

from swe2d.cli.commands import (
    build_run_command_for_params,
    cleanup_temp_specs,
)

logger = logging.getLogger(__name__)


class BatchWorker(QThread):
    """Background worker that orchestrates batch simulation subprocesses.

    Emits signals that BatchManager forwards to the UI layer.
    """

    _sim_started = pyqtSignal(str)                      # sim_id
    _sim_progress = pyqtSignal(str, float, str)       # sim_id, percent, status_text
    _sim_completed = pyqtSignal(str, str)               # sim_id, results_path
    _sim_failed = pyqtSignal(str, str)                  # sim_id, error_message
    _log_line = pyqtSignal(str, str)                    # sim_id, line
    _batch_done = pyqtSignal(bool, int, int)            # cancelled, completed, failed

    def __init__(
        self,
        params_list: List[Dict[str, Any]],
        config: Any,  # BatchConfig
        cancel_event: threading.Event,
        parent: Optional[QThread] = None,
    ):
        super().__init__(parent)
        self._params_list = params_list
        self._config = config
        self._cancel_event = cancel_event
        self._completed = 0
        self._failed = 0
        self._count_lock = threading.Lock()

    def run(self):
        """Main orchestration loop. Runs on background thread."""
        try:
            os.makedirs(os.path.join(self._config.results_dir, "batch_runs"), exist_ok=True)

            if self._config.max_workers > 1:
                self._run_parallel()
            else:
                self._run_sequential()
        except Exception as exc:
            logger.exception("BatchWorker failed")
        finally:
            with self._count_lock:
                completed = self._completed
                failed = self._failed
            self._batch_done.emit(
                self._cancel_event.is_set(),
                completed,
                failed,
            )

    def request_cancel(self):
        """Signal the worker to stop launching new sims."""
        self._cancel_event.set()

    def _run_sequential(self):
        """Launch each sim as subprocess.Popen, poll status JSON, emit progress."""
        for params in self._params_list:
            if self._cancel_event.is_set():
                break
            self._launch_sim(params)

    def _run_parallel(self):
        """Use ThreadPoolExecutor with MPS for concurrent GPU execution.

        Each sim runs in its own subprocess (swe2d.cli). The thread pool
        manages the launch/polling threads in parallel; MPS schedules the
        CUDA subprocesses on the GPU.
        """
        from swe2d.cli.batch_runner import _ensure_mps, _stop_mps_if_we_started

        mps_started = _ensure_mps()
        try:
            with ThreadPoolExecutor(max_workers=self._config.max_workers) as pool:
                futures = {}
                for params in self._params_list:
                    if self._cancel_event.is_set():
                        break
                    future = pool.submit(self._launch_sim, params)
                    futures[future] = params.get("id", "unknown")

                for future in as_completed(futures):
                    sim_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        with self._count_lock:
                            self._failed += 1
                        self._sim_failed.emit(sim_id, str(exc))
        finally:
            _stop_mps_if_we_started(mps_started)

    def _build_command(self, params: Dict[str, Any]) -> List[str]:
        """Build the CLI command for a single simulation.

        Phase 3.4: delegate to ``swe2d.cli.commands.build_run_command_for_params``
        so the GUI batch and the CLI batch use the same argv construction.
        Passes ``--status-file-path`` so the docstring's "poll its status
        file" guarantee is honored (the BatchManager polls the JSON file
        written by the subprocess).
        """
        # Write the per-sim status file inside the log dir; the dialog
        # can read it to display per-cell progress.
        log_dir = os.path.join(self._config.results_dir, "batch_runs")
        os.makedirs(log_dir, exist_ok=True)
        status_file_path = os.path.join(log_dir, f"{params.get('id', 'unknown')}.status.json")
        return build_run_command_for_params(
            params,
            results_gpkg=os.path.join(
                self._config.results_dir, "batch_results.gpkg"
            ),
            status_file_path=status_file_path,
            status_interval_s=5.0,
        )

    def _cleanup_replay_files(self):
        """Remove temporary replay files created for this batch.

        Phase 3.4: ``build_run_command_for_params`` tracks temp specs at
        module scope; we forward to the shared cleanup.  Per-sim status
        JSON files are kept (the BatchManager reads them).
        """
        cleanup_temp_specs()

    def _launch_sim(self, params: Dict[str, Any]) -> None:
        """Launch a single sim subprocess, poll its status file, write log."""
        sim_id = str(params.get("id", "unknown"))
        self._sim_started.emit(sim_id)

        log_dir = os.path.join(self._config.results_dir, "batch_runs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{sim_id}.log")

        cmd = self._build_command(params)
        if not cmd:
            self._sim_failed.emit(sim_id, "Failed to build command")
            self._failed += 1
            return

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self._sim_failed.emit(sim_id, f"Failed to launch: {exc}")
            self._failed += 1
            return

        # Stream stdout to log file and emit log lines
        try:
            with open(log_path, "w") as log_file:
                for line in proc.stdout:
                    log_file.write(line)
                    log_file.flush()
                    self._log_line.emit(sim_id, line.rstrip("\n"))

                proc.wait(timeout=self._config.timeout_per_sim)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            self._sim_failed.emit(sim_id, "Timed out")
            self._failed += 1
            return
        except Exception as exc:
            proc.kill()
            proc.wait()
            self._sim_failed.emit(sim_id, f"Error: {exc}")
            self._failed += 1
            return

        # Determine results path
        results_path = os.path.join(
            self._config.results_dir, "batch_results.gpkg"
        )

        if proc.returncode == 0:
            with self._count_lock:
                self._completed += 1
            self._sim_completed.emit(sim_id, results_path)
        else:
            with self._count_lock:
                self._failed += 1
            self._sim_failed.emit(sim_id, f"Exit code {proc.returncode}")
