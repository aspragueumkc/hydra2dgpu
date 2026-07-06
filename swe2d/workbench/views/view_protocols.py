"""Typed view protocols for MVP architecture.

Each domain controller / consumer receives a typed View protocol — not
the full dialog. This prevents reaching through to Qt widgets directly
and makes consumers testable with simple mocks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

from qgis.PyQt import QtWidgets


class ModelTabViewProtocol(Protocol):
    """Typed access to Model tab widgets (controllers never touch widgets directly)."""

    def get_h_min(self) -> float:
        """Minimum water depth threshold."""

    def get_rain_update_interval_s(self) -> int:
        """Rain rate update interval in seconds."""

    def get_run_time_hours(self) -> str:
        """Run duration text (decimal hours or HH:MM)."""

    def get_run_time_hours_parsed(self) -> float:
        """Run duration parsed as decimal hours."""

    def get_n_mann(self) -> float:
        """Manning n roughness coefficient."""

    def get_ia_ratio(self) -> float:
        """SCS initial abstraction ratio."""

    def get_infiltration_method(self) -> str:
        """Selected infiltration method key."""

    def get_rain_boundary_buffer_rings(self) -> int:
        """Boundary buffer ring count for rain."""

    def get_cn_default(self) -> float:
        """Default SCS curve number."""

    def get_drainage_solver_mode(self) -> int:
        """Drainage equation set integer key."""

    def get_drainage_gpu_method(self) -> str:
        """Drainage GPU method key."""

    def get_drainage_coupling_substeps(self) -> int:
        """Number of drainage substeps per SWE2D step."""

    def get_drainage_max_coupling_substeps(self) -> int:
        """Max adaptive substeps for drainage."""

    def get_drainage_head_deadband(self) -> float:
        """Head deadband below which no drainage flow."""

    def get_drainage_dynamic_relaxation(self) -> float:
        """Relaxation factor for drainage coupling."""

    def get_drainage_adaptive_depth_fraction(self) -> float:
        """Fraction of cell water depth drainable per step."""

    def get_drainage_adaptive_wave_courant(self) -> float:
        """Courant target for adaptive drainage."""

    def get_drainage_implicit_iters(self) -> int:
        """Implicit solver iterations for GPU drainage."""

    def get_drainage_implicit_relax(self) -> float:
        """Relaxation factor for implicit drainage on GPU."""

    def collect_params(self) -> Dict[str, Any]:
        """Return all model parameter values as a flat dict."""

    def is_inflow_progressive(self) -> bool:
        """Inflow progressive activation checkbox."""

    def is_uniform_inflow(self) -> bool:
        """Uniform inflow velocity checkbox is checked."""

    def get_inflow_progressive_chk(self) -> Optional[QtWidgets.QCheckBox]:
        """Inflow progressive checkbox widget."""

    def get_default_bc_type(self) -> int:
        """Default boundary condition type code."""

    def is_save_mesh(self) -> bool:
        """Save mesh results checkbox is checked."""

    def is_save_line(self) -> bool:
        """Save line results checkbox is checked."""

    def is_save_coupling(self) -> bool:
        """Save coupling results checkbox is checked."""

    def is_save_max_only(self) -> bool:
        """Save max-only results checkbox is checked."""

    def is_save_log(self) -> bool:
        """Save run log checkbox is checked."""

    def get_storage_checkboxes(self) -> Dict[str, QtWidgets.QCheckBox]:
        """Return storage checkboxes by key."""

    def collect_storage_params(self) -> Dict[str, Any]:
        """Return storage-checkbox parameter values as a flat dict."""


class ResultsToolboxProtocol(Protocol):
    """Typed access to results toolbox public interface (Display-only).

    Storage checkboxes were moved to the Model tab of the Model Setup panel.
    See ModelTabViewProtocol for the storage accessors.
    """

    def refresh_run_list(self) -> None:
        """Rebuild the run list from data layer."""

    def get_results_data(self):
        """Return the bound SWE2DResultsData (or None)."""

    def get_run_list_widget(self) -> QtWidgets.QListWidget:
        """Return the run list QListWidget."""


class MapTabViewProtocol(Protocol):
    """Typed access to Map tab widgets."""

    def set_layer_status_text(self, text: str) -> None:
        """Update the status label for the active layer."""


class RunDockProtocol(Protocol):
    """Typed access to Run dock widgets (execution surface only).

    Output-config widgets (output_interval, results_table_name,
    results_gpkg_path, results_gpkg_browse, preview overrides/coupling,
    load/save config) live on the Simulation tab's Output page — see
    ModelTabViewProtocol. Read them from the dialog or from the model tab
    view directly.
    """

    def set_run_button_enabled(self, enabled: bool) -> None:
        """Enable/disable the Run button."""

    def set_cancel_button_enabled(self, enabled: bool) -> None:
        """Enable/disable the Cancel button."""

    def set_progress_bar_value(self, value: int) -> None:
        """Set the progress bar value."""

    def get_run_btn(self) -> Optional[QtWidgets.QPushButton]:
        """Run button widget (for signal wiring)."""

    def get_cancel_btn(self) -> Optional[QtWidgets.QPushButton]:
        """Cancel button widget."""

    def get_progress_bar(self) -> Optional[QtWidgets.QProgressBar]:
        """Progress bar widget."""
