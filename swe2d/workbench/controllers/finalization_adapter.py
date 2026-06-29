"""Finalization adapter: bridges the RunFinalizationView protocol to the Studio dialog."""


class FinalizationAdapter:
    """View-protocol adapter bridging run finalization logic to the Studio dialog.

    Implements the ``RunFinalizationView`` protocol by delegating to private
    methods on the dialog, keeping the finalization seam free of Qt imports.
    """

    def __init__(self, dialog):
        self._dialog = dialog

    def log_message(self, msg: str) -> None:
        """Append a message to the runtime log panel."""
        self._dialog._log(msg)

    def get_line_results_storage_path(self) -> str:
        """Return the GeoPackage path for line-results persistence."""
        return self._dialog._current_line_results_storage_path()

    def sync_overlay_data(self) -> None:
        """Refresh the high-performance canvas overlay with current results."""
        self._dialog._sync_high_perf_overlay_data()

    def refresh_plot(self) -> None:
        """Redraw the results time-series plot."""
        self._dialog._refresh_plot()

    def results_table_name(self, base_name: str) -> str:
        """Return the prefixed results table name for the given base name."""
        return self._dialog._results_table_name(base_name)

    def results_data(self):
        """Return the current ``SWE2DResultsData`` instance (or None)."""
        return getattr(self._dialog, "_results_data", None)

    def length_unit_name(self) -> str:
        """Return the CRS length unit label ('m' or 'ft')."""
        return getattr(self._dialog, "_length_unit_name", "m")

    def length_scale_si_to_model(self) -> float:
        """Return the SI-to-model-unit length conversion factor."""
        return self._dialog._length_scale_si_to_model()

    def update_overlay_time(self, t: float) -> None:
        """Advance the overlay animation to simulation time *t*."""
        self._dialog._update_high_perf_overlay_time(t)

    def runtime_log_lines(self):
        """Return the list of runtime log lines accumulated during the run."""
        return self._dialog._runtime_log_lines

    def collect_run_log_metadata(self):
        """Collect run-log metadata (solver options, CRS, etc.) as a dict."""
        return self._dialog._collect_run_log_metadata()

    def persist_run_log(self, gpkg_path, run_id, run_wallclock_start,
                        run_wallclock_end, run_duration_wallclock_s,
                        run_log_text, *, metadata):
        """Write the run log to the results GeoPackage."""
        self._dialog._persist_run_log_to_geopackage(
            gpkg_path, run_id, run_wallclock_start, run_wallclock_end,
            run_duration_wallclock_s, run_log_text, metadata=metadata,
        )

    def is_cancel_requested(self) -> bool:
        """Return True if the user pressed Cancel during the run."""
        return bool(getattr(self._dialog, "_cancel_requested", False))
