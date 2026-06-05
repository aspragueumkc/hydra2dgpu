#!/usr/bin/env python3
"""Constructor startup state initialization for SWE2D workbench dialog."""

from __future__ import annotations

import os
from typing import Any, Callable, Tuple

from swe2d import units as _u


def initialize_workbench_startup_state(
    dialog: Any,
    *,
    qtcore_module: Any,
    concurrent_futures_module: Any,
    try_import_matplotlib_qt: Callable[[], Tuple[Any, Any, Any]],
) -> None:
    """Populate constructor-owned startup state before UI build/wiring."""
    dialog._backend = None
    dialog._cancel_requested = False
    dialog._mesh_data = None
    dialog._result_data = None
    dialog._snapshot_timesteps = []
    dialog._snapshot_mesh_fingerprint = ""
    dialog._line_snapshot_rows = []
    dialog._line_snapshot_profile_rows = []
    dialog._coupling_snapshot_rows = []
    dialog._three_d_patch_snapshots = []
    dialog._line_results_latest_run_id = ""
    dialog._line_results_latest_db_path = ""
    dialog._coupling_results_latest_run_id = ""
    dialog._coupling_results_latest_db_path = ""
    dialog._run_log_latest_run_id = ""
    dialog._run_log_latest_db_path = ""
    dialog._swe3d_geom_gate_last_config = {}
    dialog._swe3d_geom_gate_last_metrics = {}
    dialog._swe3d_geom_gate_last_violations = []
    dialog._results_mesh_layer_id = ""
    dialog._results_mesh_source_path = ""
    dialog._results_mesh_snapshot_count = -1
    dialog._results_mesh_mode_enabled = True
    dialog._runtime_log_lines = []
    dialog._model_gpkg_path = ""
    dialog._mesh_nodes_layer_id = None
    dialog._mesh_cells_layer_id = None
    dialog._unit_system = "SI"
    dialog._length_unit_name = "m"
    dialog._gravity = _u.gravity()
    dialog._topology_mesh_future = None
    dialog._topology_mesh_backend = None
    dialog._topology_mesh_default_cell_type = None
    dialog._topology_mesh_run_mode = "full"
    dialog._topology_mesh_auto_fallback_used = False
    dialog._topology_mesh_conceptual = None
    dialog._topology_mesh_options = {}
    dialog._topology_mesh_thread_pool = concurrent_futures_module.ThreadPoolExecutor(max_workers=1)
    dialog._topology_mesh_process_pool = None
    dialog._topology_mesh_timer = qtcore_module.QTimer(dialog)
    dialog._topology_mesh_timer.setInterval(120)
    dialog._topology_mesh_timer.timeout.connect(dialog._poll_topology_mesh_future)
    dialog._topology_mesh_started_at = None
    dialog._topology_mesh_poll_count = 0
    dialog._topology_mesh_active_timeout_sec = 0.0
    dialog._topology_mesh_checkpoint_path = ""
    dialog._topology_mesh_progress_path = ""
    dialog._topology_mesh_progress_last_seq = -1
    dialog._topology_mesh_progress_last_sig = ""
    dialog._topology_mesh_progress = None
    dialog._project_layer_state_blocked = False
    dialog._initial_layer_restore_pending = True
    dialog._experimental_3d_bc_widget_attrs = []
    dialog._experimental_3d_bc_signal_specs = []
    try:
        timeout_sec = float(os.environ.get("BACKWATER_TOPOLOGY_MESH_TIMEOUT_SEC", "3000"))
    except Exception:
        timeout_sec = 3000.0
    dialog._topology_mesh_timeout_sec = max(30.0, timeout_sec)
    dialog._topology_mesh_active_timeout_sec = dialog._topology_mesh_timeout_sec

    dialog._runtime_log_detached_dialogs = []
    dialog._mesh_view_detached_dialogs = []
    dialog._runtime_log_detached_dialog = None
    dialog._mesh_view_detached_dialog = None
    dialog._detached_panel_dialogs = []

    figure_canvas, figure, mtri = try_import_matplotlib_qt()
    dialog._FigureCanvas = figure_canvas
    dialog._Figure = figure
    dialog._mtri = mtri
    dialog._have_mpl = figure_canvas is not None and figure is not None and mtri is not None
    dialog._view_adapter = None
    dialog._run_orchestrator = None
    dialog._run_controller = None
    dialog._run_data_builder = None
    dialog._run_options_builder = None
    dialog._backend_initializer = None
    dialog._run_finalizer = None
    dialog._run_lifecycle = None
    dialog._last_run_request = None
    dialog._startup_run_component_errors = []
