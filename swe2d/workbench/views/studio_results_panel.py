"""Results panel handlers for the Studio dialog.

Extracted from SWE2DWorkbenchStudioDialog. Each function takes the dialog
as the first parameter — this module is stateless.
"""
import logging
import os

from qgis.PyQt import QtCore, QtWidgets

logger = logging.getLogger(__name__)


def _safe_log(dialog, message: str) -> None:
    """Log a message to the dialog's _log if available, else to the module logger.

    Never raises — used inside error-handling paths where the dialog itself
    may be torn down.
    """
    log_fn = getattr(dialog, "_log", None)
    if log_fn is not None:
        try:
            log_fn(message)
            return
        except Exception:
            logger.warning("_safe_log: dialog._log raised", exc_info=True)
    logger.warning(message)


def on_run_selection_changed(dialog) -> None:
    """Update run count label and refresh overlay when run selection changes."""
    dialog.results_toolbox.update_run_count()
    dialog._overlay_controller.refresh_high_perf_canvas_overlay(None)
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is not None:
        viewer.refresh()


def on_results_refresh(dialog) -> None:
    """Re-scan GPKG for runs and rebuild the run list."""
    data = dialog.results_toolbox.get_results_data()
    if data is None:
        show_results_panel(dialog)
        data = dialog.results_toolbox.get_results_data()
    if data is not None:
        dialog.results_toolbox.refresh_run_list()
        data._rebuild_timestep_union()
        data.load_coupling_for_first_enabled_run()
    # Sync temporal dock slider range with updated timesteps
    temporal = getattr(dialog, "_temporal_dock", None)
    if temporal is not None:
        temporal.set_data(data)
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is not None:
        # Notify all viewer widgets of updated data
        for w in viewer.plot_widgets.values():
            if hasattr(w, "set_data"):
                w.set_data(result_data=data)
        viewer.refresh()


def on_results_add(dialog) -> None:
    """Open file dialog to add results GeoPackages and select runs to load."""
    from swe2d.results.run_service import collect_runs_from_gpkg
    from swe2d.workbench.dialogs.run_selection_dialog import RunSelectionDialog

    paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
        dialog, "Add GeoPackage Results", "",
        "GeoPackage (*.gpkg);;All Files (*)",
    )
    if not paths:
        return
    data = dialog.results_toolbox.get_results_data()
    if data is None:
        show_results_panel(dialog)
        data = dialog.results_toolbox.get_results_data()
    if data is None:
        dialog._log("[ERROR] _on_results_add: no SWE2DResultsData")
        return

    all_candidates = []
    for gpkg in paths:
        gpkg = str(gpkg or "").strip()
        if not gpkg:
            continue
        if not os.path.exists(gpkg):
            dialog._log(f"[ERROR] _on_results_add: file not found: {gpkg}")
            continue
        try:
            candidates = collect_runs_from_gpkg(gpkg)
        except Exception as exc:
            dialog._log(f"[ERROR] _on_results_add: collect_runs_from_gpkg failed for {gpkg}: {exc}")
            continue
        if not candidates:
            dialog._log(f"[WARNING] _on_results_add: no SWE2D runs found in {gpkg}")
            continue
        all_candidates.extend(candidates)
        data.add_manual_gpkg(gpkg)

    if not all_candidates:
        dialog._log("[WARNING] _on_results_add: no runs found in selected files")
        return

    dlg = RunSelectionDialog(all_candidates, parent=dialog)
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return
    selected = dlg.selected_keys()
    if not selected:
        return

    # Replace selected_run_keys per GPKG: remove old keys for GPKGs the user
    # is explicitly adding in THIS session.  Runs from GPKGs the user is NOT
    # re-adding are cleared so old selections don't bleed across sessions.
    gpkg_paths_in_this_batch = {r.gpkg_path for r in all_candidates}
    old_keys = set(data._selected_run_keys)
    for k in old_keys:
        data._selected_run_keys.discard(k)
    data._selected_run_keys.update(selected)

    # Clear _run_records so the viewer shows ONLY the runs from this action.
    # The auto-loaded non-batch runs (from dialog startup) are discarded.
    data._run_records = []

    # Only scan the GPKGs the user explicitly added in this action —
    # do NOT scan previously-added manual GPKGs.  Each "Add Results"
    # action is self-contained.
    data.discover_runs(scan_paths=list(gpkg_paths_in_this_batch))
    data._rebuild_timestep_union()
    dialog.results_toolbox.refresh_run_list()
    temporal = getattr(dialog, "_temporal_dock", None)
    if temporal is not None:
        temporal.set_data(data)
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is not None:
        for w in viewer.plot_widgets.values():
            if hasattr(w, "set_data"):
                w.set_data(result_data=data)
        viewer.refresh()


def on_results_remove(dialog) -> None:
    """Remove selected runs from the data layer."""
    selected = dialog.results_toolbox.get_run_list_widget().selectedItems()
    if not selected:
        return
    keys = {str(it.data(QtCore.Qt.ItemDataRole.UserRole) or "") for it in selected}
    keys.discard("")
    if not keys:
        return
    data = dialog.results_toolbox.get_results_data()
    if data is not None:
        data.remove_runs(keys)
    dialog.results_toolbox.refresh_run_list()
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is not None:
        viewer.refresh()


def on_results_show_all(dialog) -> None:
    """Enable visibility for all runs."""
    data = dialog.results_toolbox.get_results_data()
    if data is not None:
        data.set_all_runs_visible()
    dialog.results_toolbox.refresh_run_list()
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is not None:
        viewer.refresh()


def on_results_hide_all(dialog) -> None:
    """Disable visibility for all runs."""
    data = dialog.results_toolbox.get_results_data()
    if data is not None:
        data.set_all_runs_hidden()
    dialog.results_toolbox.refresh_run_list()
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is not None:
        viewer.refresh()


def on_results_line_selected(dialog, line_id: int) -> None:
    """Set the active line ID and refresh the viewer."""
    data = dialog.results_toolbox.get_results_data()
    if data is not None and line_id >= 0:
        data.set_line_id(line_id)
    dialog._studio_viewer.refresh()


def on_results_ts_var_changed(dialog, var_key: str) -> None:
    """Set the time-series variable key and refresh the viewer."""
    data = dialog.results_toolbox.get_results_data()
    if data is not None and var_key:
        data.ts_var_key = var_key
    # Sync the pyqtgraph time-series widget's combo so it matches the
    # toolbox selection (otherwise its refresh reads the stale combo value).
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is not None:
        ts_widget = viewer.plot_widgets.get("Time Series")
        if ts_widget is not None and hasattr(ts_widget, "selected_metric"):
            ts_widget.selected_metric = var_key
    dialog._studio_viewer.refresh()


def on_results_prof_var_changed(dialog, var_key: str) -> None:
    """Set the profile variable key and refresh the viewer."""
    data = dialog.results_toolbox.get_results_data()
    if data is not None and var_key:
        data.prof_var_key = var_key
    # Sync the pyqtgraph profile widget's combo so it matches the
    # toolbox selection (otherwise its refresh reads the stale combo value).
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is not None:
        prof_widget = viewer.plot_widgets.get("Profile")
        if prof_widget is not None and hasattr(prof_widget, "selected_metric"):
            prof_widget.selected_metric = var_key
    dialog._studio_viewer.refresh()





def on_results_panel_timestep_changed(dialog, t_s: float, frame_idx: int = 0) -> None:
    """Handle animation timestep change — sync temporal dock, overlay, and viewer."""
    temporal = getattr(dialog, "_temporal_dock", None)
    if temporal is not None:
        temporal.on_timestep_changed(t_s, frame_idx)
        # Sync slider range — _rebuild_timestep_union may have changed frame_count.
        # Block signals while changing the range to avoid a spurious valueChanged
        # emission when the new range is smaller than the current slider value.
        data = dialog.results_toolbox.get_results_data()
        if data is not None and hasattr(data, "frame_count"):
            slider = temporal._time_slider
            slider.blockSignals(True)
            try:
                slider.setRange(0, max(0, data.frame_count - 1))
            finally:
                slider.blockSignals(False)
    if bool(getattr(dialog, "_high_perf_canvas_overlay_enabled", False)):
        # Baked path: load_mesh_snapshot_for_overlay handles both live and GPKG transparently
        last_ts = getattr(dialog, "_overlay_last_loaded_t_s", None)
        if last_ts is None or abs(float(last_ts) - float(t_s)) > 1.0e-3:
            dialog._overlay_controller.load_mesh_snapshot_for_overlay(t_s)
        dialog._update_high_perf_overlay_time(float(t_s))
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is not None:
        viewer.refresh()


def show_results_panel(dialog):
    """Show the results panel, wiring data layer and animation signals."""
    data = getattr(dialog, "_results_data", None)
    if data is None:
        dialog._overlay_controller.maybe_create_results_data()
        data = getattr(dialog, "_results_data", None)
    if data is None:
        QtWidgets.QMessageBox.warning(
            dialog, "Results Panel",
            "Could not create results data layer.\n"
            "Check the plugin log for '[Results]' details."
        )
        return
    temporal = getattr(dialog, "_temporal_dock", None)
    # Wire signals per-animation object, not once per dialog.  If
    # dialog._results_data is recreated (new SWE2DResultsData), the new
    # animation controller must be connected or the slider/play buttons stop
    # updating the overlay and plots.
    anim = getattr(data, "_anim", None)
    wired_anim = getattr(dialog, "_results_wired_anim", None)
    if anim is not None and wired_anim is not anim:
        from swe2d.workbench.signal_helpers import safe_disconnect
        safe_disconnect(anim.current_timestep_changed, dialog._on_results_panel_timestep_changed)
        anim.current_timestep_changed.connect(dialog._on_results_panel_timestep_changed)
        if temporal is not None:
            safe_disconnect(anim.play_state_changed, temporal.on_play_state_changed)
            anim.play_state_changed.connect(temporal.on_play_state_changed)
        dialog._results_wired_anim = anim
    toolbox = dialog.results_toolbox
    if toolbox is not None:
        toolbox.set_data(data)
        if hasattr(toolbox, "set_overlay_refresh_callback"):
            toolbox.set_overlay_refresh_callback(
                lambda: dialog._overlay_controller.refresh_high_perf_canvas_overlay(None)
            )
    if temporal is not None:
        temporal.set_data(data)
    try:
        dialog._refresh_plot()
    except Exception as exc:
        logger.warning("show_results_panel: _refresh_plot failed", exc_info=True)
        _safe_log(dialog, f"[ERROR] show_results_panel: {exc}")


def on_coupling_metric_changed(dialog, metric: str) -> None:
    """Handle coupling metric combo change — update the viewer plot."""
    if not metric:
        return
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is None:
        return
    widget = viewer.current_widget
    if widget is None:
        return
    data = dialog.results_toolbox.get_results_data()
    if data is not None:
        for rec in getattr(data, "_run_records", []):
            if rec.enabled and hasattr(rec, 'run_id'):
                data.load_coupling_records(rec.run_id)
                break
    widget._populate_metric_combo()
    widget.selected_metric = metric
    viewer.refresh()


def on_coupling_element_changed(dialog, element_id: str) -> None:
    """Handle coupling element combo change — update the viewer plot."""
    if not element_id:
        return
    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is None:
        return
    widget = viewer.current_widget
    if widget is None:
        return
    widget.selected_element_id = element_id
    viewer.refresh()


def auto_load_results_panel(dialog, gpkg_path: str = "", snapshot_run_id: str = ""):
    """Auto-load results panel with the most recent run from a GeoPackage.

    Only adds the most recent (or named) run — never scans in all runs.
    Does NOT switch data_source — caller is responsible for that.
    """
    if not gpkg_path:
        from swe2d.services.gpkg_persistence_service import current_line_results_storage_path
        gpkg_path = current_line_results_storage_path(dialog)
    if not gpkg_path:
        dialog._log("[Auto-Load] No GeoPackage path to load results into panel.")
        return
    data = getattr(dialog, "_results_data", None)
    if data is None:
        show_results_panel(dialog)
        data = getattr(dialog, "_results_data", None)
    if data is None:
        dialog._log("[Auto-Load] Results data layer could not be created.")
        return

    # Find the single target run (most recent, or the named snapshot)
    from swe2d.results.run_service import collect_runs_from_gpkg
    candidates = collect_runs_from_gpkg(gpkg_path)
    if not candidates:
        dialog._log("[Auto-Load] No SWE2D runs found in GPKG.")
        return

    target_key = None
    if snapshot_run_id:
        for c in candidates:
            if c.run_id == snapshot_run_id:
                target_key = c.key
                break
    if target_key is None:
        target_key = candidates[0].key  # most recent

    data._selected_run_keys.add(target_key)

    try:
        data.discover_runs(
            scan_paths=[gpkg_path] if gpkg_path else None,
        )
    except Exception as e:
        dialog._log(f"[Auto-Load] discover_runs failed: {e}")
        return

    records = data.get_run_records()
    if not records:
        dialog._log("[Auto-Load] No run records found after discover_runs.")
        return

    if snapshot_run_id:
        for rec in records:
            rec.enabled = (rec.run_id == snapshot_run_id)
    else:
        data.keep_only_most_recent_run()

    data._rebuild_timestep_union()
    dialog.results_toolbox.refresh_run_list()

    temporal = getattr(dialog, "_temporal_dock", None)
    if temporal is not None:
        temporal.set_data(data)

    viewer = getattr(dialog, "_studio_viewer", None)
    if viewer is not None:
        for w in viewer.plot_widgets.values():
            if hasattr(w, "set_data"):
                w.set_data(result_data=data)

    try:
        if bool(getattr(dialog, "_high_perf_canvas_overlay_enabled", False)):
            t_s = float(data.current_time_sec)
            dialog._update_high_perf_overlay_time(t_s)
    except Exception as e:
        dialog._log(f"[Auto-Load] overlay refresh failed: {e}")
