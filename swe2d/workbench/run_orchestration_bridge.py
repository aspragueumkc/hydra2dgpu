"""Run orchestration delegation bridge for swe2d_workbench_qt.py."""

from typing import Dict, Tuple
import time


def prepare_run_orchestration(
    dialog,
    cancel_flag_holder: Dict[str, bool],
) -> Tuple[bool, str]:
    """
    Prepare run orchestration state and perform preflight checks.
    
    Returns:
        (success: bool, error_msg: str)
    """
    if dialog._mesh_data is None:
        return False, "Run aborted: mesh not available after preflight."
    
    if dialog.SWE2DBackend is None:
        return False, "Run aborted: native backend not available after preflight."
    
    required_components = [
        ("_run_data_builder", "run data builder"),
        ("_run_options_builder", "run options builder"),
        ("_backend_initializer", "backend initializer"),
        ("_run_finalizer", "run finalizer"),
        ("_run_lifecycle", "run lifecycle"),
    ]
    
    if not dialog._require_run_components(required_components, context_label="Run"):
        return False, "Run aborted: required components missing."
    
    # Initialize run state
    cancel_flag_holder["cancelled"] = False
    dialog.run_btn.setEnabled(False)
    dialog.cancel_btn.setEnabled(True)
    dialog.progress_bar.setValue(0)
    
    return True, ""


def initialize_run_timing_and_logging(
    dialog,
) -> Tuple[float, int]:
    """
    Initialize run timing and logging state.
    
    Returns:
        (perf_counter_start: float, log_line_start_idx: int)
    """
    perf_start = time.perf_counter()
    log_start_idx = len(dialog._runtime_log_lines)
    return perf_start, log_start_idx


def finalize_run_ui_state(dialog, success: bool) -> None:
    """
    Restore UI state after run completion or failure.
    """
    dialog.run_btn.setEnabled(True)
    dialog.cancel_btn.setEnabled(False)
    if success:
        dialog.progress_bar.setValue(100)
