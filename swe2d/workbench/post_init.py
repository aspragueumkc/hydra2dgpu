#!/usr/bin/env python3
"""Post-bootstrap constructor setup for SWE2D workbench dialog."""

from __future__ import annotations

from typing import Any

import numpy as np


def run_workbench_post_bootstrap_setup(
    dialog: Any,
    *,
    swe2d_gpu_available_fn,
    gmsh_available_fn,
) -> None:
    """Run constructor setup that must happen after startup seam bootstrap."""
    dialog._connect_project_workbench_state_signals()
    dialog._connect_project_save_state_signals()
    dialog._initial_layer_restore_pending = False
    # Workbench state restoration remains in showEvent() for reliable timing.
    dialog._update_unit_system_from_crs()
    dialog._log(
        f"GPU backend: {'available' if swe2d_gpu_available_fn() else 'unavailable'}"
    )
    dialog._log(
        "Meshing: Gmsh "
        f"{'available' if gmsh_available_fn() else 'NOT INSTALLED — use Structured backend or: pip install gmsh'}"
    )

    # Sprint 0: dockable results panel (created lazily on first show).
    dialog._results_panel = None
    dialog._results_data = None
    dialog._high_perf_canvas_overlay_item = None
    dialog._high_perf_canvas_overlay_enabled = False
    dialog._high_perf_overlay_cell_x = np.empty(0, dtype=np.float64)
    dialog._high_perf_overlay_cell_y = np.empty(0, dtype=np.float64)
    dialog._high_perf_overlay_cell_bed = np.empty(0, dtype=np.float64)
    dialog._high_perf_overlay_node_x = np.empty(0, dtype=np.float64)
    dialog._high_perf_overlay_node_y = np.empty(0, dtype=np.float64)
    dialog._high_perf_overlay_cell_nodes = np.empty(0, dtype=np.int32)
    dialog._velocity_vectors_layer_id = None
    dialog._velocity_overlay_sources = []
    dialog._velocity_overlay_layer_ids = {}
    dialog._velocity_overlay_feature_ids = {}
    dialog._velocity_overlay_source_mode_logged = {}
    dialog._velocity_cell_xy_cache = {}
    dialog._velocity_base_len_cache = {}
    dialog._streamline_overlay_layer_ids = {}
    dialog._velocity_overlay_manual_gpkg_path = ""
    dialog._velocity_overlay_manual_run_id = ""
    dialog._velocity_overlay_manual_layer_name = ""
    dialog._velocity_overlay_manual_table_name = ""
    dialog._velocity_overlay_refresh_token = 0
    dialog._velocity_overlay_frame_counter = 0
    dialog._velocity_overlay_perf_log_every = 30
    dialog._streamline_overlay_frame_counter = 0
    dialog._streamline_overlay_perf_log_every = 30
    dialog._three_d_patch_surface_layer_id = None
    dialog._three_d_patch_last_spec = None
