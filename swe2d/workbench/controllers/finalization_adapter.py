class FinalizationAdapter:
    def __init__(self, dialog):
        self._dialog = dialog

    def log_message(self, msg: str) -> None:
        self._dialog._log(msg)

    def get_line_results_storage_path(self) -> str:
        return self._dialog._current_line_results_storage_path()

    def sync_overlay_data(self) -> None:
        self._dialog._sync_high_perf_overlay_data()

    def refresh_plot(self) -> None:
        self._dialog._refresh_plot()

    def results_table_name(self) -> str:
        return self._dialog._results_table_name()

    def results_data(self):
        return getattr(self._dialog, "_results_data", None)

    def length_unit_name(self) -> str:
        return getattr(self._dialog, "_length_unit_name", "m")

    def length_scale_si_to_model(self) -> float:
        return self._dialog._length_scale_si_to_model()

    def update_overlay_time(self, t: float) -> None:
        self._dialog._update_high_perf_overlay_time(t)

    def runtime_log_lines(self):
        return self._dialog._runtime_log_lines

    def collect_run_log_metadata(self):
        return self._dialog._collect_run_log_metadata()

    def persist_run_log(self, gpkg_path, run_id, run_wallclock_start,
                        run_wallclock_end, run_duration_wallclock_s,
                        run_log_text, *, metadata):
        self._dialog._persist_run_log_to_geopackage(
            gpkg_path, run_id, run_wallclock_start, run_wallclock_end,
            run_duration_wallclock_s, run_log_text, metadata=metadata,
        )

    def is_cancel_requested(self) -> bool:
        return bool(getattr(self._dialog, "_cancel_requested", False))

    def line_snapshot_rows(self):
        rd = getattr(self._dialog, "_results_data", None)
        if rd is not None:
            return rd.get_live_line_snapshot_rows()
        return getattr(self._dialog, "_line_snapshot_rows", [])

    def line_snapshot_profile_rows(self):
        rd = getattr(self._dialog, "_results_data", None)
        if rd is not None:
            return rd.get_live_line_profile_rows()
        return getattr(self._dialog, "_line_snapshot_profile_rows", [])

    def coupling_snapshot_rows(self):
        rd = getattr(self._dialog, "_results_data", None)
        if rd is not None:
            return rd.get_live_coupling_snapshot_rows()
        return getattr(self._dialog, "_coupling_snapshot_rows", [])

    def persist_coupling_results(self, gpkg_path, run_id, rows, *, interval_s):
        self._dialog._persist_coupling_results_to_geopackage(
            gpkg_path, run_id, rows, interval_s=interval_s,
        )

    def build_mesh_snapshot_rows(self):
        return self._dialog._build_mesh_snapshot_rows()

    def selected_mesh_results_table_name(self):
        return self._dialog._selected_mesh_results_table_name()

    def persist_mesh_results(self, gpkg_path, run_id, rows, *, interval_s, table_name):
        self._dialog._persist_mesh_results_to_geopackage(
            gpkg_path, run_id, rows, interval_s=interval_s, table_name=table_name,
        )

    def persist_conservation_forensics(
        self, gpkg_path, run_id, storage_rows, boundary_rows,
        conservation_summary, *, source_step_rows,
    ) -> None:
        self._dialog._persist_conservation_forensics_to_geopackage(
            gpkg_path, run_id, storage_rows, boundary_rows,
            conservation_summary, source_step_rows=list(source_step_rows),
        )
