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

    def results_table_name(self, base_name: str) -> str:
        return self._dialog._results_table_name(base_name)

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
