"""Results panel handlers for the Studio dialog.

Extracted from SWE2DWorkbenchStudioDialog. Each function takes the dialog
as the first parameter — this module is stateless.
"""
import os

from qgis.PyQt import QtCore, QtWidgets


def _reload_coupling_combos(dialog) -> None:
    """Re-populate toolbox coupling combos after discover_runs loads _coupling_records."""
    toolbox = getattr(dialog, "_results_toolbox", None)
    if toolbox is None or not hasattr(toolbox, "populate_coupling_combos"):
        return
    data = getattr(toolbox, "_data", None)
    cart_records = data.get_coupling_records()
    if cart_records:
        toolbox.populate_coupling_combos(cart_records)


def on_run_selection_changed(dialog) -> None:
    """Update run count label and refresh overlay when run selection changes."""
    dialog._results_toolbox._update_run_count()
    dialog._overlay_controller.refresh_high_perf_canvas_overlay(None)


def on_results_refresh(dialog) -> None:
    """Re-scan GPKG for runs and rebuild the run list and coupling combos."""
    data = getattr(dialog._results_toolbox, "_data", None)
    if data is None:
        show_results_panel(dialog)
        data = getattr(dialog._results_toolbox, "_data", None)
    if data is not None:
        dialog._results_toolbox._rebuild_run_list()
        data._rebuild_timestep_union()
        data.load_coupling_for_first_enabled_run()
        _reload_coupling_combos(dialog)
        line_id = dialog._results_toolbox.line_combo.currentData()
        if isinstance(line_id, (list, tuple)):
            line_id = int(line_id[0])
        if line_id is not None and int(line_id) >= 0:
            on_results_line_selected(dialog, int(line_id))


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
    data = dialog._results_toolbox._data
    if data is None:
        show_results_panel(dialog)
        data = dialog._results_toolbox._data
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

    # Replace selected_run_keys per GPKG: remove old keys for the same GPKGs,
    # then add the new user selection. This ensures the dialog is the sole
    # controller of which runs appear — old runs from the same GPKG that the
    # user un-checked are removed, and runs from other GPKGs are preserved.
    gpkg_paths_in_this_batch = {r.gpkg_path for r in all_candidates}
    old_keys = set(data._selected_run_keys)
    for k in old_keys:
        if any(k.startswith(f"{p}::") for p in gpkg_paths_in_this_batch):
            data._selected_run_keys.discard(k)
    data._selected_run_keys.update(selected)

    data.discover_runs()
    data._rebuild_timestep_union()
    _reload_coupling_combos(dialog)
    dialog._results_toolbox._rebuild_run_list()
    line_id = dialog._results_toolbox.line_combo.currentData()
    if isinstance(line_id, (list, tuple)):
        line_id = int(line_id[0])
    if line_id is not None and int(line_id) >= 0:
        on_results_line_selected(dialog, int(line_id))


def on_results_remove(dialog) -> None:
    """Remove selected runs from the data layer."""
    selected = dialog._results_toolbox.run_list.selectedItems()
    if not selected:
        return
    keys = {str(it.data(QtCore.Qt.ItemDataRole.UserRole) or "") for it in selected}
    keys.discard("")
    if not keys:
        return
    data = dialog._results_toolbox._data
    if data is not None:
        data.remove_runs(keys)
    dialog._results_toolbox._rebuild_run_list()


def on_results_show_all(dialog) -> None:
    """Enable visibility for all runs."""
    data = dialog._results_toolbox._data
    if data is not None:
        data.set_all_runs_visible()
    dialog._results_toolbox._rebuild_run_list()


def on_results_hide_all(dialog) -> None:
    """Disable visibility for all runs."""
    data = dialog._results_toolbox._data
    if data is not None:
        data.set_all_runs_hidden()
    dialog._results_toolbox._rebuild_run_list()


def on_results_line_selected(dialog, line_id: int) -> None:
    """Set the active line ID and refresh the viewer."""
    data = dialog._results_toolbox._data
    if data is not None and line_id >= 0:
        data.set_line_id(line_id)
    dialog._studio_viewer.refresh()


def on_results_ts_var_changed(dialog, var_key: str) -> None:
    """Set the time-series variable key and refresh the viewer."""
    data = dialog._results_toolbox._data
    if data is not None and var_key:
        data.ts_var_key = var_key
    dialog._studio_viewer.refresh()


def on_results_prof_var_changed(dialog, var_key: str) -> None:
    """Set the profile variable key and refresh the viewer."""
    data = dialog._results_toolbox._data
    if data is not None and var_key:
        data.prof_var_key = var_key
    dialog._studio_viewer.refresh()


def on_results_profile_options_changed(dialog) -> None:
    """Sync profile options from toolbox to data layer and refresh the viewer."""
    data = dialog._results_toolbox._data
    tb = dialog._results_toolbox
    if data is not None:
        data.prof_fill_key = str(tb.prof_fill_combo.currentData() or "none")
        data.prof_cmap = str(tb.prof_cmap_combo.currentData() or "viridis")
        data.prof_show_structures = bool(tb.show_structures_chk.isChecked())
    dialog._studio_viewer.refresh()


def on_results_panel_timestep_changed(dialog, t_s: float, frame_idx: int = 0) -> None:
    """Handle animation timestep change — sync temporal dock, overlay, and viewer."""
    temporal = getattr(dialog, "_temporal_dock", None)
    if temporal is not None:
        temporal.on_timestep_changed(t_s, frame_idx)
        # Sync slider range — _rebuild_timestep_union may have changed frame_count
        data = getattr(dialog._results_toolbox, "_data", None)
        if data is not None and hasattr(data, "frame_count"):
            temporal._time_slider.setRange(0, max(0, data.frame_count - 1))
    if bool(getattr(dialog, "_high_perf_canvas_overlay_enabled", False)):
        _snapshots = getattr(getattr(dialog, "_results_data", None), "get_live_snapshot_timesteps", lambda: [])()
        n_ts = len(_snapshots)
        data_source = getattr(getattr(dialog, "_results_data", None), "data_source", "none")
        if n_ts > 0 and data_source != "gpkg":
            dialog._update_high_perf_overlay_time(float(t_s))
        else:
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
    from swe2d.workbench.services.gpkg_persistence_service import current_line_results_storage_path
    gpkg = current_line_results_storage_path(dialog)
    if gpkg and gpkg != data.gpkg_path:
        data.set_gpkg_path(gpkg)
    if not getattr(dialog, "_results_anim_wired", False):
        dialog._results_anim_wired = True
        anim = getattr(data, "_anim", None)
        from swe2d.workbench.signal_helpers import safe_disconnect
        if anim is not None:
            safe_disconnect(anim.current_timestep_changed, dialog._on_results_panel_timestep_changed)
            anim.current_timestep_changed.connect(dialog._on_results_panel_timestep_changed)
            safe_disconnect(anim.play_state_changed, dialog._on_results_play_state_changed)
            anim.play_state_changed.connect(dialog._on_results_play_state_changed)
    toolbox = getattr(dialog, "_results_toolbox", None)
    if toolbox is not None:
        toolbox.set_data(data)
        line_id = toolbox.line_combo.currentData()
        if isinstance(line_id, (list, tuple)):
            line_id = int(line_id[0])
        if line_id is not None and int(line_id) >= 0:
            on_results_line_selected(dialog, int(line_id))
    temporal = getattr(dialog, "_temporal_dock", None)
    if temporal is not None:
        temporal.set_data(data)
    try:
        dialog._refresh_plot()
    except Exception:
        pass


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
    data = getattr(dialog._results_toolbox, "_data", None)
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
    """Auto-load results panel with the most recent run from a GeoPackage."""
    if not gpkg_path:
        from swe2d.workbench.services.gpkg_persistence_service import current_line_results_storage_path
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
    if gpkg_path != data.gpkg_path:
        data.set_gpkg_path(gpkg_path)
    # Auto-select the most recent run from the GPKG when nothing selected yet
    if not data._selected_run_keys:
        from swe2d.results.run_service import collect_runs_from_gpkg
        candidates = collect_runs_from_gpkg(gpkg_path)
        if candidates:
            data._selected_run_keys.add(candidates[0].key)
    try:
        data.discover_runs()
    except Exception as e:
        dialog._log(f"[Auto-Load] discover_runs failed: {e}")
        return
    records = data.get_run_records()
    if not records:
        dialog._log("[Auto-Load] No run records found in GPKG — hiding panel.")
        return
    _reload_coupling_combos(dialog)
    if snapshot_run_id:
        # Prefer the snapshot run — enable only it, disable others
        found = False
        for rec in records:
            rec.enabled = (rec.run_id == snapshot_run_id)
            if rec.run_id == snapshot_run_id:
                found = True
        if not found:
            dialog._log(f"[Auto-Load] Snapshot run {snapshot_run_id} not found — enabling first run")
            records[0].enabled = True
        data._rebuild_timestep_union()
        dialog._results_toolbox._rebuild_run_list()
    else:
        data.keep_only_most_recent_run()
        dialog._results_toolbox._rebuild_run_list()
    try:
        if bool(getattr(dialog, "_high_perf_canvas_overlay_enabled", False)):
            t_s = float(data.current_time_sec)
            dialog._update_high_perf_overlay_time(t_s)
    except Exception as e:
        dialog._log(f"[Auto-Load] overlay refresh failed: {e}")
