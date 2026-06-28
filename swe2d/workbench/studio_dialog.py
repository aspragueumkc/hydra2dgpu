"""Standalone Studio dialog for SWE2D workbench.

Extracted from swe2d_workbench_qt.py to break the SWE2DWorkbenchDialog inheritance.
"""

from __future__ import annotations

import copy
import csv
import datetime
import json
import logging
import math
import os
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from qgis.PyQt import QtCore, QtGui, QtWidgets

from swe2d.workbench.views.studio_component_view import StudioComponent
# ponytail: tab views imported lazily via studio_tab_builder.py
from swe2d.workbench.views.studio_viewer import SWE2DStudioViewer
from swe2d.workbench.views.results_controls import ResultsToolbox
from swe2d.workbench.views.temporal_dock import TemporalDockWidget

from swe2d.workbench.controllers.run_controller import RunController as WorkbenchController
from swe2d.workbench.workbench_dialog_builder import WorkbenchDialogBuilder
from swe2d.workbench.services import unit_conversion_service as _unit_svc
from swe2d.workbench.services.text_parser_service import (
    parse_hydrograph_text as _parse_hydrograph_text_logic,
    parse_time_hours,
)
from swe2d.services import mesh_computation_service as _mesh_svc
from swe2d.workbench.services import widget_persistence_service as _wp_svc

from swe2d.workbench.bridges.project_settings_bridge import (
    WORKBENCH_STATE_KEY,
    load_project_json,
    write_project_json,
)
from swe2d.workbench.signal_helpers import connect_lambda, safe_disconnect, safe_teardown
from swe2d.mesh.gmsh_backend import _gmsh_available
from swe2d.workbench.views.studio_host_methods import (
    launch_swe2d_workbench_studio,
    _normalize_workbench_host_mode,
    _resolve_workbench_iface,
    _remove_workbench_dock_instance,
    _remove_workbench_studio_dock,
    _studio_host_main_window,
    _clear_studio_host_controls,
    _install_studio_host_controls,
)

logger_wb = logging.getLogger(__name__)

try:
    from qgis.core import (
        QgsFeature,
        QgsField,
        QgsGeometry,
        QgsMeshLayer,
        QgsProject,
        QgsPointXY,
        QgsRasterLayer,
        QgsSettings,
        QgsUnitTypes,
        QgsVectorFileWriter,
        QgsVectorLayer,
        QgsWkbTypes,
    )
    from qgis.PyQt.QtCore import QVariant
    _HAVE_QGIS_CORE = True
except Exception:
    QgsEditorWidgetSetup = QgsFieldConstraints = QgsSettings = None
    QgsFeature = QgsField = QgsGeometry = QgsPointXY = QgsProject = None
    QgsMeshLayer = None
    QgsRasterLayer = QgsVectorLayer = QgsWkbTypes = None
    QgsUnitTypes = QgsVectorFileWriter = None
    QVariant = None
    _HAVE_QGIS_CORE = False
    logger_wb.warning(
        "qgis.core import failed — running outside QGIS or QGIS not fully initialized",
        exc_info=True,
    )

try:
    from swe2d.runtime.backend import SWE2DBackend, swe2d_gpu_available
except ImportError:
    SWE2DBackend = None
    swe2d_gpu_available = lambda: False

try:
    from swe2d.runtime.backend import (
        TemporalScheme,
        SpatialDiscretization,
        SolverModelOptions,
    )
except ImportError:
    TemporalScheme = SpatialDiscretization = SolverModelOptions = None

def _try_import_matplotlib_qt():
    """Try importing matplotlib Qt backend with fallback."""
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        return FigureCanvas, Figure
    except ImportError:
        logger_wb.warning("Exception in primary path — attempting fallback", exc_info=True)
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            return FigureCanvas, Figure
        except ImportError:
            logger_wb.warning("Graceful degradation — Exception returned fallback value", exc_info=True)
            return None, None


_BC_OPTIONS = [
    ("Wall (zero normal flux)", 1),
    ("Inflow Q (total discharge)", 2),
    ("Stage (prescribed WSE)", 3),
    ("Normal Depth (prescribed depth)", 6),
    ("Normal Depth (friction slope Sf)", 7),
    ("Timeseries Flow Q", 102),
    ("Timeseries Stage", 103),
    ("Open (zero-gradient)", 4),
    ("Reflecting", 5),
]

_BC_TS_FLOW = 102
_BC_TS_STAGE = 103
_BC_INFLOW_Q = 2


class _RuntimeLogHandler(logging.Handler):
    def __init__(self, log_fn):
        super().__init__(level=logging.WARNING)
        self._log_fn = log_fn

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the runtime log callback."""
        msg = self.format(record)
        try:
            self._log_fn(msg)
        except Exception as _e:

            try:

                self._log(f"[ERROR] Exception in studio_dialog.py: {_e}")

            except Exception:

                pass


_SWE2D_WORKBENCH_STUDIO_DOCK = None
_SWE2D_STUDIO_HOST_TOOLBAR = None
_SWE2D_STUDIO_HOST_MENU = None
_SWE2D_WORKBENCH_STUDIO_WINDOWS: List[QtWidgets.QDialog] = []
# Tracks the currently active Studio dialog for duplicate launch prevention
_studio_active_dialog: Optional["SWE2DWorkbenchStudioDialog"] = None

# ── Controls tab constants (mirrored from swe2d/results/panel.py) ──────

_CTRL_TS_VARIABLES = [
    ("Flow", "flow_cms"),
    ("Depth", "depth_m"),
    ("Velocity", "velocity_ms"),
    ("Water Surface", "wse_m"),
    ("Bed Elevation", "bed_m"),
]

_CTRL_PROF_VARIABLES = [
    ("WSE + Bed", "wse_bed"),
    ("EGL", "egl_m"),
    ("Depth", "depth_m"),
    ("Velocity", "velocity_ms"),
    ("Froude", "fr"),
    ("Normal Flow", "flow_qn"),
]

_CTRL_PROFILE_FILL_OPTIONS = [
    ("None", "none"),
    ("Depth", "depth_m"),
    ("Velocity", "velocity_ms"),
    ("Froude", "fr"),
    ("Normal Flow", "flow_qn"),
]

_CTRL_PROFILE_CMAP_OPTIONS = [
    ("Viridis", "viridis"),
    ("Plasma", "plasma"),
    ("Turbo", "turbo"),
    ("Inferno", "inferno"),
    ("Magma", "magma"),
    ("Cividis", "cividis"),
]

_CTRL_TIME_UNIT = "hr"


class SWE2DWorkbenchStudioDialog(QtWidgets.QDialog):
    """Dock-inspired workspace layout with persistent side inspector.

    Standalone dialog (no inheritance from SWE2DWorkbenchDialog).
    Shared business logic is composed from module-level functions.
    """

    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.iface = iface
        WorkbenchDialogBuilder(self).configure()

    # ---- Inlined base methods from SWE2DWorkbenchDialog ---------------

    _UP_RAINBOW = [
        "#FF0000", "#FF7F00", "#FFFF00", "#00FF00",
        "#0000FF", "#4B0082", "#9400D3",
    ]

    def _log(self, msg: str):
        """Append a message to the runtime log view."""
        import logging
        msg_txt = str(msg)
        self._runtime_log_lines.append(msg_txt)
        lv = getattr(self, "log_view", None)
        if lv is None:
            logging.getLogger(__name__).info(msg_txt)
            return
        try:
            if msg_txt.startswith("[ERROR]"):
                try:
                    lv.appendHtml(
                        f'<span style="color:red;font-weight:bold;">{msg_txt}</span>')
                except Exception:
                    lv.appendPlainText(msg_txt)
            elif msg_txt.startswith("[WARNING]"):
                try:
                    lv.appendHtml(
                        f'<span style="color:#FF8C00;font-weight:normal;">{msg_txt}</span>')
                except Exception:
                    lv.appendPlainText(msg_txt)
            elif "They go UP" in msg_txt:
                spans = []
                ci = 0
                for ch in msg_txt:
                    if ch == " ":
                        spans.append(ch)
                    else:
                        color = self._UP_RAINBOW[ci % len(self._UP_RAINBOW)]
                        spans.append(
                            f'<span style="color:{color};font-weight:bold;">{ch}</span>')
                        ci += 1
                lv.appendHtml("".join(spans))
            else:
                lv.appendPlainText(msg_txt)
        except Exception:
            # Absolute fallback — never recurse back into _log
            logging.getLogger(__name__).warning("_log fallback: %s", msg_txt)
        for dlg in list(self._state.runtime_log_detached_dialogs):
            try:
                if dlg is not None:
                    dlg.append_text(msg_txt)
            except Exception as _e:

                try:

                    self._log(f"[ERROR] Exception in studio_dialog.py: {_e}")

                except Exception:

                    pass
        now = time.perf_counter()
        last = float(getattr(self, "_last_log_process_events_wall", 0.0) or 0.0)
        if (now - last) >= 0.10:
            self._last_log_process_events_wall = now
            QtWidgets.QApplication.processEvents()

    def _wire_runtime_log_handler(self) -> None:
        """Wire runtime log handler to SWE2D loggers."""
        _handler = _RuntimeLogHandler(self._log)
        _handler.setFormatter(logging.Formatter("%(levelname)s|%(name)s: %(message)s"))
        for logger_name in [
            "swe2d.mesh.meshing",
            "swe2d.boundary_and_forcing.boundary_qgis_adapter",
            "swe2d.boundary_and_forcing.spatial_forcing_qgis_adapter",
        ]:
            logging.getLogger(logger_name).addHandler(_handler)

    def _try_import_matplotlib_qt(self):
        """Try importing matplotlib Qt backend with fallback."""
        return _try_import_matplotlib_qt()

    def _iter_project_layers(self):
        """Iterate all map layers in the QGIS project."""
        try:
            from qgis.core import QgsProject
            return list(QgsProject.instance().mapLayers().values())
        except Exception as e:
            self._log(f"[ERROR] iter project layers failed: {e}")
            return []

    def _combo_layer(self, combo, expected_kind: str):
        """Get the selected layer from a combo box by expected type."""
        from qgis.core import QgsVectorLayer, QgsRasterLayer
        idx = combo.currentIndex()
        if idx < 0:
            return None
        lid = combo.itemData(idx)
        if not lid:
            return None
        for lyr in self._iter_project_layers():
            try:
                if lyr.id() != lid:
                    continue
                if expected_kind == "vector" and isinstance(lyr, QgsVectorLayer):
                    return lyr
                if expected_kind == "raster" and isinstance(lyr, QgsRasterLayer):
                    return lyr
            except Exception as e:
                self._log(f"[ERROR] combo layer failed: {e}")
                continue
        return None

    def _open_detached_runtime_log(self):
        """Open a detached runtime log dialog."""
        from swe2d.workbench.dialogs.detached_log_dialog import SWE2DDetachedRuntimeLogDialog
        text = "\n".join(self._runtime_log_lines)
        dlg = SWE2DDetachedRuntimeLogDialog(initial_text=text, parent=self)
        self._state.runtime_log_detached_dialogs.append(dlg)
        def _cleanup(_result=None, dialog=dlg):
            """Remove the detached dialog from the tracked list on close."""
            try:
                if dialog in self._state.runtime_log_detached_dialogs:
                    self._state.runtime_log_detached_dialogs.remove(dialog)
            except Exception as e:
                self._log(f"[ERROR] cleanup failed: {e}")
        dlg.finished.connect(_cleanup)
        dlg.show()

    def _resolve_qgis_iface(self):
        """Resolve the QGIS interface instance via parent chain or qgis.utils."""
        if getattr(self, "iface", None) is not None:
            return self.iface
        parent = self.parent()
        if parent is not None:
            if hasattr(parent, "_get_qgis_iface") and callable(getattr(parent, "_get_qgis_iface")):
                try:
                    iface_obj = parent._get_qgis_iface()
                    if iface_obj is not None:
                        self.iface = iface_obj
                        return iface_obj
                except Exception as e:
                    self._log(f"[ERROR] resolve qgis iface failed: {e}")
                    pass
            if hasattr(parent, "iface"):
                try:
                    iface_obj = getattr(parent, "iface")
                    if iface_obj is not None:
                        self.iface = iface_obj
                        return iface_obj
                except Exception as e:
                    self._log(f"[ERROR] resolve qgis iface failed: {e}")
                    pass
        try:
            import qgis.utils as _qutils
            iface_obj = getattr(_qutils, "iface", None)
            if iface_obj is not None:
                self.iface = iface_obj
                return iface_obj
        except Exception as e:
            self._log(f"[ERROR] resolve qgis iface failed: {e}")
            pass
        return None

    def _resolve_map_canvas(self):
        """Resolve the map canvas from the QGIS interface."""
        iface_obj = self._resolve_qgis_iface()
        if iface_obj is None or not hasattr(iface_obj, "mapCanvas"):
            return None
        try:
            return iface_obj.mapCanvas()
        except Exception as e:
            self._log(f"[ERROR] resolve map canvas failed: {e}")
            return None

    def _get_qgis_main_window(self):
        """Get the QGIS main window instance."""
        try:
            iface = self._resolve_qgis_iface()
            if iface is not None and hasattr(iface, "mainWindow"):
                mw = iface.mainWindow()
                if mw is not None:
                    return mw
        except Exception:
            self._log("[WARNING] Unexpected Exception silently caught \u2014 review this handler")
        return None

    def _build_map_tab(self):
        """Build the map tab via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import build_map_tab as _fn
        return _fn(self)

    def _build_topology_tab(self):
        """Build the topology tab via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import build_topology_tab as _fn
        return _fn(self)

    def _build_model_tab(self):
        """Build the model tab via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import build_model_tab as _fn
        return _fn(self)

    def _compose_left_pane(self, left_host: QtWidgets.QWidget) -> QtWidgets.QWidget:
        """Compose the left pane via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import compose_left_pane as _fn
        return _fn(self, left_host)

    def _build_map_tab_page(self):
        """Build the map tab page via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import build_map_tab_page as _fn
        return _fn(self)

    def _wire_map_tab_data_signals(self) -> None:
        """Wire map tab data signals via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import wire_map_tab_data_signals as _fn
        _fn(self)

    def _wire_map_tab_action_signals(self) -> None:
        """Wire map tab action signals via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import wire_map_tab_action_signals as _fn
        _fn(self)

    def _wire_map_tab_tools_signals(self) -> None:
        """Wire map tab tools signals via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import wire_map_tab_tools_signals as _fn
        _fn(self)

    def _build_topology_tab_page(self) -> QtWidgets.QWidget:
        """Build the topology tab page via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import build_topology_tab_page as _fn
        return _fn(self)

    def _wire_topology_tab_static_signals(self) -> None:
        """Wire topology tab static signals via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import wire_topology_tab_static_signals as _fn
        _fn(self)

    def _write_memory_layer_to_gpkg(self, lyr, gpkg_path: str, layer_name: str, *, create_file: bool = False) -> None:
        from swe2d.services.lumped_hydrology_service import write_memory_layer_to_gpkg as _fn
        _fn(lyr, gpkg_path, layer_name, create_file=create_file)

    def _open_line_results_viewer(self) -> None:
        self._log("_open_line_results_viewer: not yet wired to QGIS GeoPackage loading")

    def _build_topology_meshing_options(self) -> dict:
        v = self._topology_tab_view
        opts = {}

        def _r(name, default=None):
            w = getattr(v, name, None) or v._topo_widgets.get(name)
            if w is None:
                return default
            m = getattr(w, "currentData", None)
            if m is not None:
                d = m()
                return default if d is None else d
            m = getattr(w, "isChecked", None)
            if m is not None:
                return bool(m())
            m = getattr(w, "value", None)
            if m is not None:
                return m()
            m = getattr(w, "text", None)
            if m is not None:
                return str(m()).strip() or default
            return default

        opts["gmsh_tri_algorithm"] = _r("topo_gmsh_tri_algo_combo", 6)
        opts["gmsh_quad_algorithm"] = _r("topo_gmsh_quad_algo_combo", 6)
        opts["gmsh_recombination_algorithm"] = _r("topo_gmsh_recombine_algo_combo", 1)
        opts["gmsh_global_recombine"] = _r("topo_gmsh_global_recombine_chk", False)
        opts["gmsh_quad_full_region_flow_align"] = _r("topo_gmsh_quad_full_region_flow_align_chk", True)
        opts["gmsh_smoothing"] = _r("topo_gmsh_smoothing_spin", 0)
        opts["gmsh_optimize_iters"] = _r("topo_gmsh_optimize_iters_spin", 0)
        opts["gmsh_verbosity"] = _r("topo_gmsh_verbosity_spin", 2)
        opts["gmsh_optimize_netgen"] = _r("topo_gmsh_optimize_netgen_chk", False)
        opts["gmsh_quality_enable"] = _r("topo_gmsh_quality_enable_chk", False)
        opts["gmsh_quality_strict"] = _r("topo_quality_strict_chk", False)
        opts["gmsh_min_angle_deg"] = _r("topo_quality_min_angle_spin", 5.0)
        opts["gmsh_max_aspect_ratio"] = _r("topo_quality_max_aspect_spin", 20.0)
        opts["gmsh_max_non_orth_deg"] = _r("topo_quality_max_non_orth_spin", 82.0)
        opts["gmsh_min_area_rel_bbox"] = _r("topo_quality_min_area_edit", "1e-14")
        opts["gmsh_quality_max_iterations"] = _r("topo_gmsh_quality_max_iters_spin", 2)
        opts["gmsh_quality_time_limit_s"] = _r("topo_gmsh_quality_time_limit_spin", 55.0)
        opts["gmsh_quality_size_scales"] = _r("topo_quality_size_scales_edit", "1.0,0.9,0.8,0.7")
        opts["gmsh_quality_smooth_increments"] = _r("topo_quality_smooth_increments_edit", "0,2,4,6")
        opts["gmsh_quality_recombine_topology_passes"] = _r("topo_gmsh_quality_recombine_topology_passes_edit", "5,12,20")
        opts["gmsh_quality_recombine_minimum_quality"] = _r("topo_gmsh_quality_recombine_min_quality_edit", "0.01,0.03,0.06")
        opts["gmsh_quality_random_factors"] = _r("topo_gmsh_quality_random_factors_edit", "1e-9,1e-7,1e-6")
        opts["gmsh_quality_optimize_methods"] = _r("topo_gmsh_quality_optimize_methods_edit", "Laplace2D,Relocate2D")
        opts["gmsh_algorithm_switch_on_failure"] = _r("topo_gmsh_algo_switch_on_failure_chk", False)
        opts["gmsh_quality_recombine_node_repositioning"] = _r("topo_gmsh_recombine_node_repositioning_chk", True)
        opts["gmsh_mesh_size_min"] = _r("topo_gmsh_mesh_size_min_spin", 0.0)
        opts["gmsh_tolerance_edge_length"] = _r("topo_gmsh_tolerance_edge_length_spin", 0.0)
        opts["gmsh_mesh_size_from_points"] = _r("topo_gmsh_mesh_size_from_points_chk", True)
        opts["gmsh_arc_mode"] = _r("topo_gmsh_arc_mode_combo", "hard_embed")
        opts["gmsh_arc_soft_size_factor"] = _r("topo_gmsh_arc_soft_size_factor_spin", 0.5)
        opts["gmsh_arc_soft_dist_factor"] = _r("topo_gmsh_arc_soft_dist_factor_spin", 2.0)
        opts["gmsh_interface_transition_enable"] = _r("topo_gmsh_interface_transition_enable_chk", True)
        opts["gmsh_interface_transition_dist_factor"] = _r("topo_gmsh_interface_transition_dist_factor_spin", 2.5)
        opts["gmsh_interface_transition_min_ratio"] = _r("topo_gmsh_interface_transition_min_ratio_spin", 1.25)
        opts["gmsh_interface_conformance"] = _r("topo_gmsh_interface_conformance_chk", False)
        opts["gmsh_transverse_interface_centroid_merge"] = _r("topo_gmsh_transverse_interface_centroid_merge_chk", False)
        opts["gmsh_interface_snap_tol"] = _r("topo_gmsh_interface_snap_tol_spin", 1.0)
        opts["gmsh_interface_reject_near_unshared"] = _r("topo_gmsh_interface_reject_near_unshared_chk", True)
        opts["gmsh_interface_reject_tol"] = _r("topo_gmsh_interface_reject_tol_spin", 1e-3)
        opts["gmsh_num_threads"] = _r("topo_gmsh_num_threads_spin", 1)
        opts["gmsh_max_num_threads_2d"] = _r("topo_gmsh_max_num_threads_2d_spin", 0)
        opts["gmsh_transfinite_shared_interface_harmonize"] = _r(
            "topo_gmsh_transfinite_shared_interface_harmonize_chk", False
        )
        opts["gmsh_transfinite_opposite_subset_start"] = _r(
            "topo_gmsh_transfinite_opposite_subset_start_spin", 0.30
        )
        opts["gmsh_transfinite_opposite_subset_end"] = _r(
            "topo_gmsh_transfinite_opposite_subset_end_spin", 0.70
        )
        opts["gmsh_transfinite_opposite_subset_density_scale"] = _r(
            "topo_gmsh_transfinite_opposite_subset_density_scale_spin", 0.50
        )
        opts["gmsh_transfinite_interface_debug"] = _r(
            "topo_gmsh_transfinite_interface_debug_chk", False
        )
        opts["gmsh_transfinite_subset_containment_enable"] = _r(
            "topo_gmsh_transfinite_subset_containment_enable_chk", True
        )
        opts["gmsh_transfinite_subset_containment_high_overlap"] = _r(
            "topo_gmsh_transfinite_subset_containment_high_overlap_spin", 0.95
        )
        opts["gmsh_transfinite_subset_containment_min_overlap"] = _r(
            "topo_gmsh_transfinite_subset_containment_min_overlap_spin", 0.02
        )
        opts["gmsh_transfinite_subset_containment_max_length_ratio"] = _r(
            "topo_gmsh_transfinite_subset_containment_max_length_ratio_spin", 0.35
        )
        return opts

    def _infer_workspace_root_for_meshing(self) -> str:
        """ponytail: returns empty, add workspace resolution when needed."""
        return ""

    def _build_model_tab_page(self):
        """Build the model tab page via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import build_model_tab_page as _fn
        return _fn(self)

    def _wire_run_tab_signals(self) -> None:
        """Wire run tab signals via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import wire_run_tab_signals as _fn
        _fn(self)

    # ── Migrated method — real impl from extracted/topology_and_io_methods ──

    def _update_topology_control_summary(self) -> None:
        """Update topology control summary label from current topology state.

        Delegates to controller method (inlined from extracted module).
        """
        self._topology_tab_view.update_control_summary()

    def _refresh_topology_status(self) -> None:
        """Refresh topology status display (delegates to controller)."""
        self._topology_controller._refresh_topology_status()

    def _populate_gmsh_quality_controls(self) -> None:
        """Populate gmsh quality controls (delegates to controller)."""
        self._topology_controller._populate_gmsh_quality_controls()

    # ── TopologyMeshView protocol methods ─────────────────────────────

    @property
    def topo_status_lbl(self):
        """Delegate to topology tab view's status label."""
        return getattr(self._topology_tab_view, "topo_status_lbl", None)

    def update_topo_status(self, text: str) -> None:
        """Set topology status label text (TopologyMeshView protocol)."""
        self._topology_tab_view.update_topo_status(text)

    def update_topo_controls_summary(self, text: str) -> None:
        """Set topology controls summary text (TopologyMeshView protocol)."""
        self._topology_tab_view.update_topo_controls_summary(text)

    def get_topo_widget_value(self, attr: str):
        """Read a topology widget value by attribute name (TopologyMeshView protocol)."""
        return self._topology_tab_view.get_topo_widget_value(attr)

    def set_topo_widget_visible(self, attr: str, visible: bool) -> None:
        """Set topology widget visibility (TopologyMeshView protocol)."""
        self._topology_tab_view.set_topo_widget_visible(attr, visible)

    def get_topo_combo_data(self, attr: str):
        """Return currentData() of a topology combo (TopologyMeshView protocol)."""
        return self._topology_tab_view.get_topo_combo_data(attr)

    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed seconds for display (TopologyMeshView protocol)."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds - minutes * 60
        return f"{minutes}m {secs:.0f}s"

    # ── Stub methods for Phase A migration (wire handler compatibility) ──────
    # These are stopgap stubs. Real implementations will replace them in
    # later phases (B-D) of the extracted/ migration.

    def _create_lumped_hydrology_geopackage(self) -> None:
        """Create a lumped hydrology GeoPackage."""
        self._mesh_controller.create_lumped_hydrology_geopackage()

    def _export_mesh_to_layers(self) -> None:
        """Export mesh to QGIS vector layers."""
        self._mesh_controller.export_mesh_to_layers()

    def _export_mesh_to_ugrid(self) -> None:
        """Export mesh to UGRID NetCDF."""
        self._mesh_controller.export_mesh_to_ugrid()

    def _save_mesh_to_gpkg(self) -> None:
        """Save current mesh to a GeoPackage as a baked BLOB."""
        from qgis.PyQt import QtWidgets
        mesh_data = getattr(self, "_mesh_data", None)
        if mesh_data is None or mesh_data.get("node_x") is None:
            QtWidgets.QMessageBox.warning(self, "Save Mesh", "No mesh loaded.")
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select GeoPackage to save mesh into",
            getattr(self, "_model_gpkg_path", ""),
            "GeoPackage (*.gpkg);;All Files (*)",
        )
        if not path:
            return

        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Mesh", "Mesh entry name:",
            text="",
        )
        if not ok or not name.strip():
            return
        name = str(name).strip()

        try:
            from swe2d.services.gpkg_persistence_service import persist_baked_mesh
            from hydra_swe2d import (
                swe2d_build_mesh_poly, swe2d_serialize_mesh, swe2d_mesh_info,
            )
            import numpy as np
            nx = np.asarray(mesh_data["node_x"], dtype=np.float64)
            ny = np.asarray(mesh_data["node_y"], dtype=np.float64)
            nz = np.asarray(mesh_data["node_z"], dtype=np.float64)
            bc_n0 = np.asarray(mesh_data.get("bc_edge_node0", np.empty(0)), dtype=np.int32)
            bc_n1 = np.asarray(mesh_data.get("bc_edge_node1", np.empty(0)), dtype=np.int32)
            bc_tp = np.asarray(mesh_data.get("bc_edge_type", np.empty(0)), dtype=np.int32)
            bc_vl = np.asarray(mesh_data.get("bc_edge_val", np.empty(0)), dtype=np.float64)
            cfn = mesh_data.get("cell_face_nodes") or mesh_data.get("cell_nodes")
            cfo = mesh_data.get("cell_face_offsets")
            if cfn is not None and cfo is not None:
                pm = swe2d_build_mesh_poly(nx, ny, nz, np.asarray(cfo, dtype=np.int32),
                    np.asarray(cfn, dtype=np.int32), bc_n0, bc_n1, bc_tp, bc_vl)
            else:
                cn = np.asarray(mesh_data["cell_nodes"], dtype=np.int32)
                pm = swe2d_build_mesh(nx, ny, nz, cn, bc_n0, bc_n1, bc_tp, bc_vl)
            blob = swe2d_serialize_mesh(pm)
            info = swe2d_mesh_info(pm)
            persist_baked_mesh(path, name, blob, info["n_nodes"], info["n_cells"], info["n_edges"])
            QtWidgets.QMessageBox.information(
                self, "Save Mesh",
                f"Mesh '{name}' saved to {os.path.basename(path)}.",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Mesh Error", str(exc))

    def _load_mesh_from_gpkg(self) -> None:
        """Open file dialog to select GPKG, then load a baked mesh from it."""
        from qgis.PyQt import QtWidgets
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select GeoPackage with Mesh", "",
            "GeoPackage (*.gpkg);;All Files (*)",
        )
        if not path or not os.path.isfile(path):
            return
        import sqlite3
        mesh_names = []
        try:
            conn = sqlite3.connect(path)
            try:
                cur = conn.cursor()
                cur.execute("SELECT mesh_name FROM swe2d_baked_mesh ORDER BY created_utc DESC")
                mesh_names = [str(r[0]) for r in cur.fetchall()]
            finally:
                conn.close()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Load Mesh Error",
                f"Could not read {os.path.basename(path)}: {exc}",
            )
            return
        if not mesh_names:
            QtWidgets.QMessageBox.warning(
                self, "Load Mesh",
                f"No baked meshes found in {os.path.basename(path)}.",
            )
        from qgis.PyQt.QtWidgets import QInputDialog
        name, ok = QInputDialog.getItem(
            self, "Select Mesh", "Mesh:", mesh_names, 0, False,
        )
        if not ok or not name:
            return
        try:
            from hydra_swe2d import swe2d_deserialize_mesh
            from swe2d.services.gpkg_persistence_service import load_baked_mesh
            blob = load_baked_mesh(path, name)
            if blob is None:
                QtWidgets.QMessageBox.warning(self, "Load Mesh", f"Mesh '{name}' not found.")
                return
            pm = swe2d_deserialize_mesh(blob)
            mesh_data = {
                "node_x": np.asarray(pm.node_x, dtype=np.float64),
                "node_y": np.asarray(pm.node_y, dtype=np.float64),
                "node_z": np.asarray(pm.node_z, dtype=np.float64),
                "cell_nodes": np.asarray(pm.cell_face_nodes, dtype=np.int32) if pm.cell_face_nodes is not None else np.empty(0, dtype=np.int32),
            }
            cfo = pm.cell_face_offsets
            if cfo is not None:
                mesh_data["cell_face_offsets"] = np.asarray(cfo, dtype=np.int32)
            cfn = pm.cell_face_nodes
            if cfn is not None:
                mesh_data["cell_face_nodes"] = np.asarray(cfn, dtype=np.int32)
            if mesh_data is None or mesh_data.get("node_x") is None:
                QtWidgets.QMessageBox.warning(self, "Load Mesh", f"Mesh '{name}' not found.")
                return
            self._mesh_data = mesh_data
            self._log(f"Mesh '{name}' loaded from {os.path.basename(path)} ({mesh_data['node_x'].size} nodes)")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load Mesh Error", str(exc))

    def _assign_node_z_from_terrain(self) -> None:
        """Assign node z-values from terrain raster."""
        self._mesh_controller.assign_node_z_from_terrain()

    def _export_results_to_ugrid(self) -> None:
        """Export simulation results to UGRID NetCDF format."""
        self._mesh_controller.export_results_to_ugrid()

    def _pull_node_z_from_layer(self) -> None:
        """Pull node z-values from a vector layer."""
        self._mesh_controller.pull_node_z_from_layer()

    def _open_model_gpkg_explorer(self) -> None:
        """Open model GeoPackage explorer dialog."""
        self._topology_controller.open_model_gpkg_explorer()

    def _open_run_log_viewer(self) -> None:
        """Open the run log viewer dialog."""
        self._mesh_controller.open_run_log_viewer()

    def _open_topology_region_table(self) -> None:
        """Open topology region attribute table dialog."""
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.dialogs.topo_attr_table_dialog import TopologyAttributeTableDialog
        from swe2d.workbench.services.constants_service import CELL_TYPE_OPTIONS as _ctypes

        lyr = self._combo_layer(self._topology_tab_view.topo_regions_combo, "vector")
        if lyr is None:
            QtWidgets.QMessageBox.information(
                self, "Topology Editor", "Select a topology regions layer first.",
            )
            return

        dlg = TopologyAttributeTableDialog(
            lyr,
            "Topology Region Controls",
            [
                ("region_id", "Region ID", "int"),
                ("target_size", "Target Size", "float"),
                ("cell_type", "Cell Type", "enum", _ctypes),
                ("edge_len_1", "Edge Len 1", "float"),
                ("edge_len_2", "Edge Len 2", "float"),
                ("edge_len_3", "Edge Len 3", "float"),
                ("edge_len_4", "Edge Len 4", "float"),
            ],
            sort_fields=["region_id"],
            note=(
                "Use one polygon per block. For structured/cartesian blocks, "
                "edge_len_1..4 define per-edge target spacing."
            ),
            parent=self,
        )
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._topology_tab_view.update_control_summary()

    def _open_topology_quad_edge_table(self) -> None:
        """Open topology quad-edge attribute table dialog."""
        from qgis.PyQt import QtWidgets
        from swe2d.workbench.dialogs.topo_attr_table_dialog import TopologyAttributeTableDialog

        lyr = self._combo_layer(self._topology_tab_view.topo_quad_edges_combo, "vector")
        if lyr is None:
            QtWidgets.QMessageBox.information(
                self, "Topology Editor",
                "Select a quad-edge / transition-layer layer first.",
            )
            return

        dlg = TopologyAttributeTableDialog(
            lyr,
            "Topology Transition Layers",
            [
                ("region_id", "Region ID", "int"),
                ("edge_id", "Edge ID", "int"),
                ("target_size", "Target Size", "float"),
                ("n_layers", "N Layers", "int"),
                ("first_height", "First Height", "float"),
                ("growth_rate", "Growth Rate", "float"),
            ],
            sort_fields=["region_id", "edge_id"],
            note=(
                "Define one line per region edge for a complete four-edge "
                "structured block."
            ),
            parent=self,
        )
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._topology_tab_view.update_control_summary()

    def _create_topology_template_layers(self) -> None:
        """Create topology template layers in the QGIS project."""
        self._topology_controller.create_topology_template_layers()

    def _generate_mesh_from_topology_layers(self) -> None:
        """Generate mesh from topology layer data."""
        self._topology_controller.generate_mesh_from_topology_layers()

    def _on_terminate_topology_mesh(self) -> None:
        """Terminate running topology-to-mesh generation."""
        self._topology_controller.on_terminate_topology_mesh()

    def _start_topology_mesh_async(self, conceptual=None, backend=None,
                                    default_cell_type=None, mesh_options=None,
                                    run_mode="full") -> None:
        """Start async topology-to-mesh generation."""
        self._topology_controller.start_topology_mesh_async(
            conceptual=conceptual, backend_name=backend or "gmsh",
            default_cell_type=default_cell_type or "Triangle",
            mesh_options=mesh_options, run_mode=run_mode,
        )

    def _set_topology_mesh_busy(self, busy: bool, status_msg: str = "") -> None:
        """Toggle UI busy state during topology meshing."""
        from qgis.PyQt import QtCore, QtWidgets

        topo = self._topology_tab_view
        try:
            topo.topo_generate_btn.setEnabled(not busy)
        except Exception as _e:

            try:

                self._log(f"[ERROR] Exception in studio_dialog.py: {_e}")

            except Exception:

                pass
        try:
            topo.topo_terminate_btn.setEnabled(bool(busy))
        except Exception as _e:

            try:

                self._log(f"[ERROR] Exception in studio_dialog.py: {_e}")

            except Exception:

                pass
        if status_msg:
            topo.topo_status_lbl.setText(status_msg)
        if busy:
            topo.progress_bar.setRange(0, 0)
            QtWidgets.QApplication.setOverrideCursor(
                QtCore.Qt.CursorShape.WaitCursor,
            )
        else:
            topo.progress_bar.setRange(0, 100)
            topo.progress_bar.setValue(0)
            try:
                QtWidgets.QApplication.restoreOverrideCursor()
            except Exception as _e:

                try:

                    self._log(f"[ERROR] Exception in studio_dialog.py: {_e}")

                except Exception:

                    pass

    def _load_2d_model_geopackage(self, path_override=None) -> None:
        """Load a 2D model GeoPackage into the workbench."""
        self._mesh_controller.load_2d_model_geopackage(path_override=path_override)

    def _iter_all_persistable_widgets(self):
        """Yield (attr_name, widget) pairs across all tab views and toolbox."""
        return _wp_svc.iter_all_persistable_widgets(
            dialog=self,
            tab_views=[getattr(self, a, None) for a in
                       ("_model_tab_view", "_map_tab_view",
                        "_topology_tab_view", "_mesh_tab_view",
                        "_boundary_tab_view", "_results_toolbox")],
            persistable_types=(
                QtWidgets.QSpinBox,
                QtWidgets.QDoubleSpinBox,
                QtWidgets.QComboBox,
                QtWidgets.QCheckBox,
                QtWidgets.QLineEdit,
            ),
        )

    def _is_project_workbench_state_persist_blocked(self) -> bool:
        """Check if persistence should be suppressed (during restore)."""
        return _wp_svc.is_project_workbench_state_persist_blocked(self._state)

    def _connect_project_workbench_state_signals(self) -> None:
        """Wire every persistable widget's value-changed signal to auto-save."""
        safe_disconnect = globals().get("safe_disconnect")
        if safe_disconnect is None:
            from swe2d.workbench.signal_helpers import safe_disconnect

        for attr_name, widget in self._iter_all_persistable_widgets():
            signal_attr = None
            if isinstance(widget, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                signal_attr = "valueChanged"
            elif isinstance(widget, QtWidgets.QComboBox):
                signal_attr = "currentIndexChanged"
            elif isinstance(widget, QtWidgets.QCheckBox):
                signal_attr = "toggled"
            elif isinstance(widget, QtWidgets.QLineEdit):
                signal_attr = "editingFinished"
            if signal_attr is None:
                continue
            signal = getattr(widget, signal_attr, None)
            if signal is None:
                continue
            safe_disconnect(signal, self._persist_project_workbench_state)
            try:
                signal.connect(self._persist_project_workbench_state)
            except Exception as _e:

                try:

                    self._log(f"[ERROR] Exception in studio_dialog.py: {_e}")

                except Exception:

                    pass

    def _connect_project_save_state_signals(self) -> None:
        """Connect project save/close events to persist workbench state."""
        if not _HAVE_QGIS_CORE or QgsProject is None:
            return
        try:
            safe_disconnect = globals().get("safe_disconnect")
            if safe_disconnect is None:
                from swe2d.workbench.signal_helpers import safe_disconnect
            safe_disconnect(QgsProject.instance().writeProject,
                            self._persist_project_workbench_state)
            QgsProject.instance().writeProject.connect(
                self._persist_project_workbench_state
            )
        except Exception:
            self._log("[ERROR] Could not connect project save signals")

    def _selected_results_table_prefix(self) -> str:
        """Build the sanitized prefix for results table names from UI."""
        raw = ""
        tbl_edit = getattr(self._model_tab_view, "results_table_name_edit", None)
        if tbl_edit is not None:
            raw = str(tbl_edit.text() or "").strip()
        if not raw:
            return ""
        cleaned_chars = []
        for ch in raw:
            cleaned_chars.append(ch if ch.isalnum() or ch == "_" else "_")
        cleaned = "".join(cleaned_chars).strip("_")
        if not cleaned:
            return ""
        if not (cleaned[0].isalpha() or cleaned[0] == "_"):
            cleaned = f"p_{cleaned}"
        return cleaned

    def _results_table_name(self, base_name: str) -> str:
        """Build a qualified results table name with optional prefix."""
        base = str(base_name or "").strip() or "swe2d_mesh_results"
        prefix = str(self._selected_results_table_prefix() or "").strip("_")
        if not prefix:
            return base
        if base.startswith(prefix + "_"):
            return base
        return f"{prefix}_{base}"

    def _load_run_logs_from_geopackage(self, gpkg_path: str) -> list:
        """Load run log entries from a GeoPackage."""
        from swe2d.results.run_log_storage import load_run_logs_from_geopackage as _loader
        return _loader(
            gpkg_path=gpkg_path,
            table_prefix=self._selected_results_table_prefix(),
        )

    def _apply_run_log_metadata_to_ui(self, metadata: dict) -> int:
        """Restore UI widget state from run log metadata."""
        if not isinstance(metadata, dict):
            return 0
        state_payload = metadata.get("workbench_widget_state")
        if not isinstance(state_payload, dict):
            return 0
        widgets_data = state_payload.get("widgets")
        if not isinstance(widgets_data, dict):
            return 0
        from swe2d.workbench.bridges.project_settings_bridge import restore_workbench_widget_state
        restored = restore_workbench_widget_state(
            ui=self,
            widgets_data=widgets_data,
            qtwidgets_module=QtWidgets,
            log_callback=self._log,
        )
        self._log(f"Applied run metadata settings: restored_widgets={int(restored)}")
        return int(restored)

    def _wrap_left_tab_page(self, widget):
        """Wrap a widget as a left tab page via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import wrap_left_tab_page as _fn
        return _fn(self, widget)

    def _expand_toolbox_pages(self, toolbox):
        """Expand toolbox pages via studio_tab_builder."""
        from swe2d.workbench.views.studio_tab_builder import expand_toolbox_pages as _fn
        _fn(self, toolbox)

    def _show_tab_detach_menu(self, tab_widget, pos):
        """Show the tab detach context menu at a given position."""
        bar = tab_widget.tabBar()
        idx = bar.tabAt(pos)
        if idx < 0:
            return
        title = str(tab_widget.tabText(idx) or "Tab")
        menu = QtWidgets.QMenu(self)
        detach_action = menu.addAction(f"Detach '{title}'")
        chosen = menu.exec(bar.mapToGlobal(pos))
        if chosen == detach_action:
            self._detach_tab(tab_widget, idx)

    def _show_panel_detach_menu(self, panel_kind: str, global_pos: QtCore.QPoint) -> None:
        """Show the panel detach context menu."""
        menu = QtWidgets.QMenu(self)
        if str(panel_kind) == "log":
            action = menu.addAction("Detach Runtime Log")
            chosen = menu.exec(global_pos)
            if chosen == action:
                self._open_detached_runtime_log()
            return

    def _detach_tab(self, tab_widget, index):
        """Detach a tab page into a standalone dialog."""
        if tab_widget is None or index < 0:
            return
        page = tab_widget.widget(index)
        if page is None:
            return
        try:
            page.setParent(None)
        except Exception as e:
            self._log(f"[ERROR] detach tab failed: {e}")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(str(tab_widget.tabText(index) or ""))
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page)
        tab_widget.removeTab(index)
        self._detached_panel_dialogs.append(dlg)
        dlg.finished.connect(lambda: self._detached_panel_dialogs.remove(dlg) if dlg in self._detached_panel_dialogs else None)
        dlg.resize(600, 500)
        dlg.show()

    def _poll_topology_mesh_future(self):
        """Poll the topology mesh generation future and update state."""
        state = self._topology_controller.poll_topology_mesh_future(self._topology_mesh_state)
        if state is not None:
            self._topology_mesh_state = state
        return state

    def _ensure_high_perf_canvas_overlay_item(self):
        """Ensure the high-performance canvas overlay item exists."""
        return self._overlay_controller.ensure_high_perf_canvas_overlay_item()

    def _sync_high_perf_overlay_data(self):
        """Sync high-performance overlay data."""
        self._overlay_controller.sync_high_perf_overlay_data()

    def show_warning_message(self, title: str, message: str) -> None:
        """Show a warning message box with the given title and message."""
        from qgis.PyQt import QtWidgets
        QtWidgets.QMessageBox.warning(self, str(title), str(message))

    def show_get_save_file(self, title: str, start_dir: str, filter_str: str) -> str:
        """Show a save-file dialog and return the chosen path."""
        from qgis.PyQt import QtWidgets
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, str(title), str(start_dir), str(filter_str))
        return str(path or "")

    def show_get_double(self, title: str, label: str, value: float, min_v: float, max_v: float) -> Tuple[float, bool]:
        """Show a double-input dialog and return the value and OK flag."""
        from qgis.PyQt import QtWidgets
        val, ok = QtWidgets.QInputDialog.getDouble(self, str(title), str(label), float(value), float(min_v), float(max_v), 4)
        return float(val), bool(ok)

    def refresh_map_canvas(self) -> None:
        """Refresh the QGIS map canvas."""
        iface = self._resolve_qgis_iface()
        if iface is not None and hasattr(iface, "mapCanvas"):
            try:
                iface.mapCanvas().refresh()
            except Exception as _e:

                try:

                    self._log(f"[ERROR] Exception in studio_dialog.py: {_e}")

                except Exception:

                    pass

    def _update_high_perf_overlay_time(self, t_s: float):
        """Update the high-performance overlay time."""
        self._overlay_controller.update_high_perf_overlay_time(t_s)

    def _resolve_overlay_time(self, t_s):
        """Resolve overlay time via controller."""
        return self._overlay_controller.resolve_overlay_time(t_s)

    def _apply_overlay_frame(self, frame):
        """Apply an overlay frame via controller."""
        self._overlay_controller.apply_overlay_frame(frame)

    def _refresh_high_perf_canvas_overlay(self, t_s):
        """Refresh the high-performance canvas overlay at time t_s."""
        self._overlay_controller.refresh_high_perf_canvas_overlay(t_s)

    def _reset_runtime_snapshot_overlay_cache(self, reason=""):
        """Reset the runtime snapshot overlay cache."""
        self._overlay_controller.reset_runtime_snapshot_overlay_cache(reason)

    # ------------------------------------------------------------------
    # Results toolbox handlers
    # ------------------------------------------------------------------

    def _on_run_selection_changed(self):
        """Handle run selection changed event."""
        from swe2d.workbench.views.studio_results_panel import on_run_selection_changed as _fn
        _fn(self)

    def _on_results_refresh(self):
        """Handle results refresh request."""
        from swe2d.workbench.views.studio_results_panel import on_results_refresh as _fn
        _fn(self)

    def _on_results_add(self):
        """Handle results add request."""
        from swe2d.workbench.views.studio_results_panel import on_results_add as _fn
        _fn(self)

    def _on_results_remove(self):
        """Handle results remove request."""
        from swe2d.workbench.views.studio_results_panel import on_results_remove as _fn
        _fn(self)

    def _on_results_show_all(self):
        """Handle show-all-runs request."""
        from swe2d.workbench.views.studio_results_panel import on_results_show_all as _fn
        _fn(self)

    def _on_results_hide_all(self):
        """Handle hide-all-runs request."""
        from swe2d.workbench.views.studio_results_panel import on_results_hide_all as _fn
        _fn(self)

    def _on_results_line_selected(self, line_id: int):
        """Handle line selection in results panel."""
        from swe2d.workbench.views.studio_results_panel import on_results_line_selected as _fn
        _fn(self, line_id)

    def _on_results_ts_var_changed(self, var_key: str):
        """Handle time-series variable change."""
        from swe2d.workbench.views.studio_results_panel import on_results_ts_var_changed as _fn
        _fn(self, var_key)

    def _on_results_prof_var_changed(self, var_key: str):
        """Handle profile variable change."""
        from swe2d.workbench.views.studio_results_panel import on_results_prof_var_changed as _fn
        _fn(self, var_key)



    def _on_coupling_metric_changed(self, metric: str) -> None:
        """Handle coupling metric change in results panel."""
        from swe2d.workbench.views.studio_results_panel import on_coupling_metric_changed as _fn
        _fn(self, metric)

    def _on_coupling_element_changed(self, element_id: str) -> None:
        """Handle coupling element selection change."""
        from swe2d.workbench.views.studio_results_panel import on_coupling_element_changed as _fn
        _fn(self, element_id)

    def _current_line_results_storage_path(self) -> str:
        """Determine the current GeoPackage path for line result storage."""
        mtv = getattr(self, "_model_tab_view", None)
        if mtv is not None:
            path_edit = getattr(mtv, "results_gpkg_path_edit", None)
            if path_edit is not None:
                override_raw = str(path_edit.text() or "").strip()
                if override_raw:
                    override = os.path.abspath(os.path.expanduser(override_raw))
                    parent_dir = os.path.dirname(override) or "."
                    if os.path.isdir(parent_dir):
                        self._log(
                            f"[ResultsPath] using override: {override} "
                            f"(from results_gpkg_path_edit on Model→Run/Output)"
                        )
                        return override
                    else:
                        self._log(
                            f"[ResultsPath] override parent dir missing: "
                            f"parent_dir={parent_dir!r}, override={override!r}"
                        )
                else:
                    self._log("[ResultsPath] results_gpkg_path_edit is empty")
            else:
                self._log("[ResultsPath] results_gpkg_path_edit not found on _model_tab_view")
        else:
            self._log("[ResultsPath] _model_tab_view not found")
        if self._model_gpkg_path and os.path.exists(self._model_gpkg_path):
            self._log(
                f"[ResultsPath] falling back to _model_gpkg_path: "
                f"{self._model_gpkg_path}"
            )
            return self._model_gpkg_path
        if hasattr(self, "_map_tab_view") and hasattr(self._map_tab_view, "sample_lines_layer_combo"):
            lyr = self._combo_layer(self._map_tab_view.sample_lines_layer_combo, "vector")
            if lyr is not None:
                try:
                    src = str(lyr.dataProvider().dataSourceUri())
                    gpkg = src.split("|", 1)[0]
                    if gpkg.lower().endswith(".gpkg") and os.path.exists(gpkg):
                        return gpkg
                except Exception as e:
                    self._log(f"[ERROR] current line results storage path failed: {e}")
                    pass
        import tempfile
        return os.path.join(tempfile.gettempdir(), "swe2d_line_results.gpkg")

    def _persist_snapshot_to_gpkg(self, gpkg_path: str, run_id: str, accumulate: bool = False) -> None:
        """[DEPRECATED] Snapshot persistence now deferred to run_finalizer.py."""
        logger_wb.warning("_persist_snapshot_to_gpkg called but snapshots are in-memory only until finalization")

    def _persist_run_log_to_geopackage(
        self, gpkg_path: str, run_id: str,
        start_wallclock: str, end_wallclock: str,
        duration_s: float, log_text: str,
        metadata: dict = None,
    ) -> None:
        """Persist the accumulated runtime log to the GPKG."""
        if not gpkg_path or not run_id:
            return
        try:
            from swe2d.results.run_log_storage import persist_run_log_to_geopackage as _persist
            _persist(
                gpkg_path=gpkg_path, run_id=run_id,
                start_wallclock=start_wallclock, end_wallclock=end_wallclock,
                duration_s=duration_s, log_text=log_text,
                metadata=metadata or {},
            )
        except Exception as e:
            self._log(f"[WARNING] Run log persistence skipped: {e}")

    def _show_results_panel(self):
        """Show the results panel via studio_results_panel."""
        from swe2d.workbench.views.studio_results_panel import show_results_panel as _fn
        _fn(self)

    def _auto_load_results_panel(self, gpkg_path: str = "", snapshot_run_id: str = ""):
        """Auto-load the results panel from a GeoPackage."""
        from swe2d.workbench.views.studio_results_panel import auto_load_results_panel as _fn
        _fn(self, gpkg_path, snapshot_run_id)

    def _write_ugrid_nc(self, path, timesteps=None):
        """Export simulation results to UGRID NetCDF format."""
        from swe2d.services.ugrid_export_service import write_ugrid_nc
        return write_ugrid_nc(self, path, timesteps)

    def _on_results_panel_timestep_changed(self, t_s: float, frame_idx: int = 0):
        """Handle results panel timestep change event."""
        from swe2d.workbench.views.studio_results_panel import on_results_panel_timestep_changed as _fn
        _fn(self, t_s, frame_idx)

    def _persist_project_workbench_state(self, *_args):
        """Persist workbench widget state to project."""
        _wp_svc.persist_project_workbench_state(
            have_qgis_core=_HAVE_QGIS_CORE,
            qgs_project_cls=QgsProject,
            workbench_state_key=WORKBENCH_STATE_KEY,
            state_obj=self._state,
            iter_widgets_fn=self._iter_all_persistable_widgets,
            write_project_json_fn=write_project_json,
            log_fn=self._log,
        )

    def _restore_project_workbench_state(self, *_args):
        """Restore workbench widget state from project."""
        _wp_svc.restore_project_workbench_state(
            have_qgis_core=_HAVE_QGIS_CORE,
            qgs_project_cls=QgsProject,
            workbench_state_key=WORKBENCH_STATE_KEY,
            state_obj=self._state,
            iter_widgets_fn=self._iter_all_persistable_widgets,
            load_project_json_fn=load_project_json,
            log_fn=self._log,
        )

    def _refresh_plot(self):
        """Refresh the plot viewer with current mesh and results data."""
        viewer = getattr(self, "_studio_viewer", None)
        if viewer is None:
            self._log("[PlotRefresh] _studio_viewer not found")
            return
        mesh = getattr(self, "_mesh_data", None)
        results = getattr(self, "_results_data", None)
        n_runs = len(results.get_run_records()) if results is not None and hasattr(results, "get_run_records") else -1
        n_enabled = len(results.get_enabled_run_records()) if results is not None and hasattr(results, "get_enabled_run_records") else -1
        self._log(
            f"[PlotRefresh] mesh={'loaded' if mesh else 'none'}, "
            f"results={'present' if results else 'none'}, "
            f"runs={n_runs}, enabled={n_enabled}, "
            f"gpkg={getattr(results, 'gpkg_path', '?')}"
        )
        viewer.set_mesh_data(mesh)
        viewer.set_result_data(results)
        try:
            h_min = float(self._model_tab_view.h_min_spin.value())
        except Exception:
            h_min = 1.0e-6
        viewer.set_h_min(h_min)
        viewer.refresh()

    def _studio_project_scope_key(self) -> str:
        """Generate a unique scope key for the current QGIS project."""
        project_key = "default"
        if _HAVE_QGIS_CORE and QgsProject is not None:
            try:
                proj = QgsProject.instance()
                file_name = str(proj.fileName() or "").strip()
                if file_name:
                    project_key = file_name
                else:
                    project_key = str(proj.homePath() or "").strip() or project_key
            except Exception as e:
                self._log(f"[ERROR] studio project scope key failed: {e}")
                pass
        safe = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in project_key)
        if not safe:
            safe = "default"
        return safe

    def _studio_select_tab(self, name: str) -> None:
        """Select a left tab by name."""
        if not hasattr(self, "_left_tabs") or self._left_tabs is None:
            return
        target = str(name or "").strip().lower()
        for idx in range(self._left_tabs.count()):
            if str(self._left_tabs.tabText(idx) or "").strip().lower() == target:
                self._left_tabs.setCurrentIndex(idx)
                return

    def _studio_set_feature_enabled(self, feature: str, enabled: bool) -> None:
        """Set a Studio feature flag and re-apply visibility filters.

        Valid feature keys are defined in self._state.studio_feature_flags.
        After updating the flag, calls _studio_apply_feature_filters() to
        immediately show/hide matching widgets and tabs.

        To add a new feature:
          1. Add the key to self._state.studio_feature_flags in the dialog __init__
          2. Add keyword entries in _studio_feature_keywords() below
          3. Add menu + toolbar toggles in _install_studio_host_controls()
        See docs/STUDIO_UI_ARCHITECTURE.md section C.
        """
        key = str(feature or "").strip().lower()
        if key not in self._state.studio_feature_flags:
            return
        self._state.studio_feature_flags[key] = bool(enabled)
        self._studio_apply_feature_filters()

    def _studio_feature_keywords(self) -> Dict[str, Tuple[str, ...]]:
        """Return keyword mappings from feature flags to widget text patterns."""
        return {
            "rainfall": ("rain", "gauge", "hyet", "storm", "runoff", "precip"),
            "drainage_structures": (
                "drain", "node", "link", "inlet", "outfall", "pipe", "network",
                "structure", "culvert", "weir", "orifice", "gate", "spillway",
                "coupling",
            ),
        }

    def _studio_widget_text_blob(self, widget: QtWidgets.QWidget) -> str:
        """Build a lowercase text blob from a widget for feature matching."""
        parts = [str(widget.objectName() or "")]
        try:
            if hasattr(widget, "text") and callable(widget.text):
                parts.append(str(widget.text() or ""))
        except Exception as e:
            self._log(f"[ERROR] studio widget text blob failed: {e}")
            pass
        try:
            if hasattr(widget, "title") and callable(widget.title):
                parts.append(str(widget.title() or ""))
        except Exception as e:
            self._log(f"[ERROR] studio widget text blob failed: {e}")
            pass
        try:
            parts.append(str(widget.toolTip() or ""))
        except Exception as e:
            self._log(f"[ERROR] studio widget text blob failed: {e}")
            pass
        return " ".join(parts).lower()

    def _studio_apply_feature_filters(self) -> None:
        """Apply feature visibility filters to all widgets and tabs."""
        if not hasattr(self, "_left_tabs") or self._left_tabs is None:
            return
        keywords = self._studio_feature_keywords()
        for widget in self._left_tabs.findChildren(QtWidgets.QWidget):
            if widget is self._left_tabs:
                continue
            blob = self._studio_widget_text_blob(widget)
            matched = []
            for feature, words in keywords.items():
                if any(word in blob for word in words):
                    matched.append(feature)
            if not matched:
                continue
            visible = all(self._state.studio_feature_flags.get(feature, True) for feature in matched)
            try:
                widget.setVisible(visible)
            except Exception as e:
                self._log(f"[ERROR] studio apply feature filters failed: {e}")
                pass
        # Sync tab page visibility: hide/show tabs whose page or content matches
        # a feature flag, so the tab bar entry disappears when the feature is off.
        tabs = self._left_tabs
        for i in range(tabs.count()):
            page = tabs.widget(i)
            if page is None:
                continue
            blob = self._studio_widget_text_blob(page)
            matched = []
            for feature, words in keywords.items():
                if any(word in blob for word in words):
                    matched.append(feature)
            if not matched:
                continue
            visible = all(self._state.studio_feature_flags.get(feature, True) for feature in matched)
            try:
                tabs.setTabVisible(i, visible)
            except Exception as e:
                self._log(f"[ERROR] studio apply feature filters failed: {e}")
                pass

    # ── Component Registration API ──────────────────────────────────────



    def _studio_update_status(self) -> None:
        """Update the status bar with project and host-mode info."""
        if self._state.studio_status_label is None:
            return
        project_name = "(no project)"
        project_home = ""
        if _HAVE_QGIS_CORE and QgsProject is not None:
            try:
                proj = QgsProject.instance()
                project_name = str(proj.baseName() or "").strip() or "(unnamed project)"
                project_home = str(proj.homePath() or "").strip()
            except Exception as e:
                self._log(f"[ERROR] studio update status failed: {e}")
                pass
        mode_txt = str(getattr(self, "_swe2d_workbench_host_mode", "window") or "window")
        detail = f"Project: {project_name}"
        if project_home:
            detail += f" | Home: {project_home}"
        detail += f" | Host mode: {mode_txt}"
        self._state.studio_status_label.setText(detail)

    # ── View methods (MVP View layer — read widget values, return data) ──

    def _update_unit_system_from_crs(self) -> None:
        """Update unit-system labels from CRS-derived map units."""
        result = _unit_svc.update_unit_system_from_crs(
            have_qgis_core=_HAVE_QGIS_CORE,
            project=QgsProject.instance() if _HAVE_QGIS_CORE and QgsProject is not None else None,
            log_fn=self._log,
        )
        self._unit_system = result["sys_name"]
        self._length_unit_name = result["unit_name"]
        self._gravity = result["gravity"]
        self._k_mann = result["k_mann"]
        self._log(
            f"Detected unit system: {result['sys_name']} "
            f"(CRS: {result.get('crs_desc', 'unknown')}, "
            f"length unit: {result['unit_name']}, "
            f"gravity={result['gravity']:.3f})"
        )
        mtv = getattr(self, "_model_tab_view", None)
        if mtv is not None and hasattr(mtv, "unit_system_lbl"):
            mtv.unit_system_lbl.setText(
                f"Unit system: {result['sys_name']} "
                f"(CRS length unit: {result['unit_name']}, "
                f"gravity={result['gravity']:.3f})"
            )

    def _length_scale_si_to_model(self) -> float:
        """Return the SI-to-model length scale factor."""
        return _unit_svc.length_scale_si_to_model()

    def _rain_mm_to_model_depth(self) -> float:
        """Return the rain mm-to-model depth conversion factor."""
        return _unit_svc.rain_mm_to_model_depth()

    def _rain_rate_si_to_model(self, rain_rate_mps):
        """Convert an SI rain rate to model units."""
        return _unit_svc.rain_rate_si_to_model(rain_rate_mps)

    def _flow_si_to_model(self, flow_cms):
        """Convert an SI flow rate to model units."""
        return _unit_svc.flow_si_to_model(flow_cms)

    def _parse_time_hours(self, token: str) -> float:
        """Parse a time string into hours using hydrograph logic."""
        from swe2d.boundary_and_forcing.hydrograph_logic import parse_time_hours as _logic
        return _logic(token)

    def _parse_run_duration_seconds(self) -> float:
        """Parse run duration from UI and return seconds."""
        hrs = self._parse_time_hours(self._model_tab_view.run_time_edit.text())
        if hrs <= 0.0:
            raise ValueError("run duration must be > 0")
        return 3600.0 * hrs

    def _collect_boundary_arrays(self):
        """Collect boundary condition arrays from the map view."""
        from swe2d.boundary_and_forcing.boundary_runtime_logic import collect_boundary_arrays as _logic
        default_bc_type = 0
        default_bc_combo = getattr(self._map_tab_view, "default_bc_type_combo", None)
        if default_bc_combo is not None:
            default_bc_type = int(default_bc_combo.currentData())
        return _logic(
            mesh_data=self._mesh_data,
            mesh_boundary_edges_fn=self._mesh_boundary_edges,
            default_bc_type=default_bc_type,
            apply_bc_layer_overrides_fn=self._apply_bc_layer_overrides,
            log_fn=self._log,
        )

    def _collect_bc_layer_hydrographs(self, edge_n0: np.ndarray, edge_n1: np.ndarray) -> Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
        """Collect boundary condition hydrographs from BC layer."""
        from swe2d.boundary_and_forcing.boundary_qgis_adapter import collect_bc_layer_hydrographs_qgis as _logic
        from swe2d.workbench.services.constants_service import BC_TS_FLOW as _BC_TS_FLOW, BC_TS_STAGE as _BC_TS_STAGE
        from qgis.core import QgsVectorLayer, QgsGeometry, QgsPointXY
        return _logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            bc_lines_layer_combo=getattr(self._map_tab_view, "bc_lines_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            iter_project_layers_fn=self._iter_project_layers,
            hydrograph_from_layer_fn=self._hydrograph_from_layer,
            parse_hydrograph_text_fn=self._parse_hydrograph_text,
            edge_n0=edge_n0,
            edge_n1=edge_n1,
            ts_flow_code=_BC_TS_FLOW,
            ts_stage_code=_BC_TS_STAGE,
            qgs_vector_layer_cls=QgsVectorLayer,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
            log_fn=self._log,
        )

    def _collect_bc_layer_edge_groups(self, edge_n0: np.ndarray, edge_n1: np.ndarray) -> Dict[int, str]:
        """Collect boundary edge group labels from BC layer."""
        from swe2d.boundary_and_forcing.boundary_qgis_adapter import collect_bc_layer_edge_groups_qgis as _logic
        from qgis.core import QgsGeometry, QgsPointXY
        return _logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            bc_lines_layer_combo=getattr(self._map_tab_view, "bc_lines_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            edge_n0=edge_n0,
            edge_n1=edge_n1,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
        )

    def _initial_state(self, bc_n0=None, bc_n1=None, bc_tp=None):
        """Build the initial state arrays for mesh cells."""
        from swe2d.mesh.mesh_runtime_logic import initial_state as _logic
        mtv = self._model_tab_view
        return _logic(
            mesh_data=self._mesh_data,
            mode=str(mtv.initial_condition_combo.currentData()),
            initial_depth=float(mtv.initial_depth_spin.value()),
            initial_wse=float(mtv.initial_wse_spin.value()),
            h_min=float(mtv.h_min_spin.value()),
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            log_fn=self._log,
        )

    def _build_spatial_manning_array(self) -> Optional[np.ndarray]:
        """Build a spatial Manning's n array from a raster layer."""
        from swe2d.boundary_and_forcing.spatial_forcing_qgis_adapter import build_spatial_manning_array_qgis as _logic
        return _logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            manning_layer_combo=getattr(self._map_tab_view, "manning_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            mesh_cell_centroids_fn=self._mesh_cell_centroids,
            default_n=float(self._model_tab_view.n_mann_spin.value()),
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
            log_fn=self._log,
        )

    def _preview_spatial_manning(self):
        """Preview the spatial Manning's n array statistics."""
        mann_arr = self._build_spatial_manning_array()
        name = ""
        layer = getattr(self._map_tab_view, "manning_layer_combo", None)
        if layer is not None:
            name = str(layer.currentText() or "")
        if mann_arr is not None and mann_arr.size > 0:
            return mann_arr, int(np.sum(mann_arr > 0)), int(mann_arr.size), name
        return None, 0, 0, name

    def _bc_code_label(self, code: int) -> str:
        """Return the human-readable label for a boundary condition code."""
        from swe2d.workbench.services.constants_service import BC_VALUE_MAP as _BC_VALUE_MAP
        for label, val in _BC_VALUE_MAP.items():
            if int(val) == code:
                return label
        return f"BC(code={code})"

    def _build_side_hydrographs(self):
        """Build side hydrographs dict (stub, returns empty)."""
        return {}

    def _build_internal_flow_forcing(self) -> Optional[Dict[str, object]]:
        """Build internal flow forcing configuration from QGIS layers."""
        from swe2d.boundary_and_forcing.internal_flow_qgis_adapter import build_internal_flow_forcing_qgis as _logic
        return _logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            internal_flow_layer_combo=getattr(self._model_tab_view, "internal_flow_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            requested_field_name=str(getattr(self._model_tab_view, "internal_flow_field_edit", None).text() or "q_cms") if hasattr(self._model_tab_view, "internal_flow_field_edit") and self._model_tab_view.internal_flow_field_edit is not None else "q_cms",
            iter_project_layers_fn=self._iter_project_layers,
            mesh_cell_centroids_fn=self._mesh_cell_centroids,
            parse_hydrograph_text_fn=self._parse_hydrograph_text,
            hydrograph_from_layer_fn=self._hydrograph_from_layer,
            qgs_vector_layer_cls=QgsVectorLayer,
            qgs_wkb_types=QgsWkbTypes,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
            log_fn=self._log,
        )

    def _internal_flow_source_cms_at_time(self, forcing: Optional[Dict[str, object]], t_sec: float) -> Optional[np.ndarray]:
        """Compute internal flow source rate at a given time."""
        from swe2d.boundary_and_forcing.runtime_source_logic import internal_flow_source_cms_at_time as _logic
        return _logic(forcing, t_sec, self._interp_hydrograph)

    def _build_thiessen_rain_cn_forcing(self):
        """Build Thiessen polygon rain/CN forcing from gauge layers."""
        from swe2d.boundary_and_forcing.spatial_forcing_qgis_adapter import build_thiessen_rain_cn_forcing_qgis as _logic
        from swe2d.boundary_and_forcing.rainfall_hydrology import (
            Gauge as _Gauge,
            ThiessenRainCNForcing as _ThiessenRainCNForcing,
            build_hyetograph as _build_hyetograph,
            assign_cells_to_nearest_gauge as _assign_cells_to_nearest_gauge,
            inspect_hyetograph_rows as _inspect_hyetograph_rows,
        )
        return _logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            thiessen_rain_cn_forcing_cls=_ThiessenRainCNForcing,
            gauge_cls=_Gauge,
            build_hyetograph_fn=_build_hyetograph,
            assign_cells_to_nearest_gauge_fn=_assign_cells_to_nearest_gauge,
            inspect_hyetograph_rows_fn=_inspect_hyetograph_rows,
            use_spatial_rain_cn=bool(self._model_tab_view.use_spatial_rain_cn_chk.isChecked()),
            rain_gage_layer_combo=getattr(self._map_tab_view, "rain_gage_layer_combo", None),
            hyetograph_layer_combo=getattr(self._map_tab_view, "hyetograph_layer_combo", None),
            storm_area_layer_combo=getattr(self._model_tab_view, "storm_area_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            mesh_cell_centroids_fn=self._mesh_cell_centroids,
            boundary_buffer_cells_fn=self._boundary_buffer_cells,
            build_spatial_cn_array_fn=self._build_spatial_cn_array,
            ia_ratio=float(self._model_tab_view.ia_ratio_spin.value()),
            infiltration_method=str(self._model_tab_view.infiltration_method_combo.currentData() or "scs_cn"),
            rain_boundary_buffer_rings=int(self._model_tab_view.rain_boundary_buffer_rings_spin.value()),
            qgs_wkb_types=QgsWkbTypes,
            qgs_geometry_cls=QgsGeometry,
            qgs_pointxy_cls=QgsPointXY,
            log_fn=self._log,
        )

    def _build_pipe_network_config(self):
        """Build pipe network configuration from drain layers."""
        from swe2d.workbench.services.pipe_network_config_service import (
            build_pipe_network_config_from_widgets,
        )
        from swe2d.extensions.extension_models import PipeNetworkConfig
        map_view = self._map_tab_view
        if self._mesh_data is None:
            self._log("[Drainage] _build_pipe_network_config: mesh_data is None")
            return None
        if not _HAVE_QGIS_CORE:
            self._log("[Drainage] _build_pipe_network_config: QGIS core not available")
            return None
        if PipeNetworkConfig is None:
            self._log("[Drainage] _build_pipe_network_config: PipeNetworkConfig import failed")
            return None
        if map_view is None:
            self._log("[Drainage] _build_pipe_network_config: map_view is None")
            return None
        if not hasattr(map_view, "drain_nodes_layer_combo"):
            self._log("[Drainage] _build_pipe_network_config: drain_nodes_layer_combo missing on map_view")
            return None
        node_layer = self._combo_layer(map_view.drain_nodes_layer_combo, "vector")
        link_layer = self._combo_layer(map_view.drain_links_layer_combo, "vector") if hasattr(map_view, "drain_links_layer_combo") else None
        inlet_layer = self._combo_layer(map_view.drain_inlets_layer_combo, "vector") if hasattr(map_view, "drain_inlets_layer_combo") else None
        node_inlet_layer = self._combo_layer(map_view.drain_node_inlets_layer_combo, "vector") if hasattr(map_view, "drain_node_inlets_layer_combo") else None
        if node_layer is None:
            self._log("[Drainage] _build_pipe_network_config: no drain_nodes layer selected")
            return None
        if link_layer is None:
            self._log("[Drainage] _build_pipe_network_config: no drain_links layer selected")
            return None
        cell_min_bed = self._mesh_cell_min_bed()
        from swe2d.mesh.mesh_runtime_logic import nearest_cell_index, mesh_cell_centroids
        cell_cx, cell_cy = mesh_cell_centroids(self._mesh_data)
        def _nearest_cell(x, y):
            """Find the nearest mesh cell index to a given coordinate."""
            return nearest_cell_index(x, y, cell_cx, cell_cy)
        solver_mode_combo = self._model_tab_view.drainage_solver_mode_combo
        return build_pipe_network_config_from_widgets(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            pipe_network_config_cls=PipeNetworkConfig,
            node_layer=node_layer,
            link_layer=link_layer,
            inlet_layer=inlet_layer,
            node_inlet_layer=node_inlet_layer,
            cell_min_bed=cell_min_bed,
            nearest_cell_fn=_nearest_cell,
            gravity=self._gravity,
            solver_mode_name=solver_mode_combo.currentText(),
            solver_mode=solver_mode_combo.currentData(),
            coupling_substeps=int(self._model_tab_view.drainage_coupling_substeps_spin.value()),
            max_coupling_substeps=int(self._model_tab_view.drainage_max_coupling_substeps_spin.value()),
            gpu_method=str(self._model_tab_view.drainage_gpu_method_combo.currentData()),
            head_deadband=float(self._model_tab_view.drainage_head_deadband_spin.value()),
            dynamic_relaxation=float(self._model_tab_view.drainage_dynamic_relaxation_spin.value()),
            adaptive_depth_fraction=float(self._model_tab_view.drainage_adaptive_depth_fraction_spin.value()),
            adaptive_wave_courant=float(self._model_tab_view.drainage_adaptive_wave_courant_spin.value()),
            implicit_iters=int(self._model_tab_view.drainage_implicit_iters_spin.value()),
            implicit_relax=float(self._model_tab_view.drainage_implicit_relax_spin.value()),
            log_fn=self._log,
        )
        self._log(f"[Drainage] _build_pipe_network_config: built config successfully (solver={solver_mode_combo.currentText()})")

    def _build_hydraulic_structure_config(self):
        """Build hydraulic structure configuration from structure layer."""
        from swe2d.extensions.extension_models import (
            HydraulicStructureConfig, StructureType, HydraulicStructure)
        if self._mesh_data is None:
            self._log("[Structures] mesh_data is None")
            return None
        if not _HAVE_QGIS_CORE:
            self._log("[Structures] QGIS core not available")
            return None
        if HydraulicStructureConfig is None or StructureType is None:
            self._log("[Structures] Structure model imports failed")
            return None
        structures_layer_combo = getattr(self._map_tab_view, "structures_layer_combo", None)
        if structures_layer_combo is None:
            self._log("[Structures] structures_layer_combo not found on map_view")
            return None
        layer = self._combo_layer(structures_layer_combo, "vector")
        if layer is None:
            self._log("[Structures] no structures layer selected in combo")
            return None
        from swe2d.workbench.services.structure_config_service import (
            build_hydraulic_structure_config_from_layer as _logic,
        )
        cfg = _logic(
            mesh_data=self._mesh_data,
            have_qgis_core=_HAVE_QGIS_CORE,
            hydraulic_structure_config_cls=HydraulicStructureConfig,
            structure_type_cls=StructureType,
            hydraulic_structure_cls=HydraulicStructure,
            structures_layer=layer,
        )
        self._log(f"[Structures] built config: {type(cfg).__name__ if cfg is not None else 'None'}")
        return cfg

    def _apply_timeseries_bc_values(self, edge_n0, edge_n1, bc_type, bc_val, side_hydrographs, t_sec, edge_hydrographs=None):
        """Apply time-series boundary condition values at time t_sec."""
        from swe2d.boundary_and_forcing.bc_logic import apply_timeseries_bc_values as _logic
        from swe2d.boundary_and_forcing.boundary_runtime_logic import classify_boundary_edges as _bc_side_classification
        bc_cache = self.__dict__.setdefault("_bc_geom_cache", {})
        if "_side_idx" not in bc_cache:
            side_idx = _bc_side_classification(
                edge_n0, edge_n1, self._mesh_data["node_x"], self._mesh_data["node_y"],
            )
            bc_cache["_side_idx"] = side_idx
        return _logic(
            edge_n0=edge_n0, edge_n1=edge_n1, bc_type=bc_type, bc_val=bc_val,
            side_hydrographs=side_hydrographs,
            node_x=self._mesh_data["node_x"], node_y=self._mesh_data["node_y"],
            t_sec=t_sec,
            ts_flow_code=1, ts_stage_code=2,
            edge_hydrographs=edge_hydrographs,
            _side_idx=bc_cache["_side_idx"],
        )

    def _distribute_total_flow_to_unit_q(
        self,
        edge_n0: np.ndarray,
        edge_n1: np.ndarray,
        bc_type_step: np.ndarray,
        bc_val_step: np.ndarray,
        bc_type_template: np.ndarray,
        side_hydrographs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        edge_hydrographs: Optional[Dict[int, Tuple[int, Tuple[np.ndarray, np.ndarray]]]] = None,
        edge_groups: Optional[Dict[int, str]] = None,
    ) -> np.ndarray:
        """Distribute total flow BC values to unit discharge per edge."""
        from swe2d.boundary_and_forcing.bc_logic import distribute_total_flow_to_unit_q as _logic
        from swe2d.boundary_and_forcing.bc_logic import _bc_side_classification
        progressive = bool(getattr(self._map_tab_view, "inflow_progressive_chk", None) is not None
                          and self._map_tab_view.inflow_progressive_chk.isChecked())
        if edge_groups is None:
            eg = getattr(self, "_cached_edge_groups", None)
            if eg is not None:
                edge_groups = eg
            else:
                try:
                    edge_groups = self._collect_bc_layer_edge_groups(edge_n0, edge_n1)
                    self._cached_edge_groups = edge_groups
                except Exception as e:
                    self._log(f"[ERROR] distribute total flow to unit q failed: {e}")
        bc_cache = self.__dict__.setdefault("_bc_geom_cache", {})
        if "_side_idx" not in bc_cache or "_edge_len" not in bc_cache or "_edge_z" not in bc_cache:
            side_idx, edge_len, edge_z, *_ = _bc_side_classification(
                edge_n0, edge_n1,
                self._mesh_data["node_x"], self._mesh_data["node_y"],
                self._mesh_data.get("node_z"),
            )
            bc_cache["_side_idx"] = side_idx
            bc_cache["_edge_len"] = edge_len
            bc_cache["_edge_z"] = edge_z
        return _logic(
            edge_n0=edge_n0, edge_n1=edge_n1,
            bc_type_step=bc_type_step, bc_val_step=bc_val_step,
            bc_type_template=bc_type_template,
            side_hydrographs=side_hydrographs,
            node_x=self._mesh_data["node_x"],
            node_y=self._mesh_data["node_y"],
            node_z=self._mesh_data["node_z"],
            progressive=progressive,
            ts_flow_code=102,
            edge_hydrographs=edge_hydrographs,
            edge_groups=edge_groups,
        )

    def _apply_external_sources(self, backend, dt_step, rain_rate_model, cell_source_model=None, coupled_source_rate=None):
        """Apply external source terms (rain, cell sources) to the backend."""
        from swe2d.boundary_and_forcing.runtime_source_logic import apply_external_sources as _logic
        mtab = self._model_tab_view
        _cell_areas = None
        if cell_source_model is not None:
            _cell_areas = backend.cell_areas()
        _logic(
            backend=backend,
            dt_step=dt_step,
            rain_rate_model=rain_rate_model,
            cell_source_model=cell_source_model,
            coupled_source_rate=coupled_source_rate,
            mesh_cell_areas=_cell_areas,
            max_source_rate=float(mtab.max_source_rate_spin.value()),
            h_min=float(mtab.h_min_spin.value()),
            max_rel_depth_increase=float(mtab.max_rel_depth_increase_spin.value()),
            max_source_depth_step=float(mtab.max_source_depth_step_spin.value()),
            shallow_damping_depth=float(mtab.shallow_damping_depth_spin.value()),
            momentum_cap_min_speed=float(mtab.momentum_cap_min_speed_spin.value()),
            momentum_cap_celerity_mult=float(mtab.momentum_cap_celerity_mult_spin.value()),
        )

    def _sample_line_metrics(self, sample_map, t_accum, h_s, hu_s, hv_s, cell_solver_z):
        """Sample line metrics (time-series and profile) from solver state."""
        if not sample_map:
            return [], []
        from swe2d.workbench.services.mesh_service import sample_line_metrics as _svc
        from swe2d.workbench.services.mesh_service import sample_line_aggregate_ts_row as _agg_svc
        g = float(self._gravity)
        h_min = float(self._model_tab_view.h_min_spin.value())
        h = np.asarray(h_s, dtype=np.float64)
        hu = np.asarray(hu_s, dtype=np.float64)
        hv = np.asarray(hv_s, dtype=np.float64)
        cell_bed = np.asarray(cell_solver_z, dtype=np.float64)
        if self._mesh_data is not None:
            _nx = np.asarray(self._mesh_data.get("node_x", np.empty(0)), dtype=np.float64).ravel()
            _ny = np.asarray(self._mesh_data.get("node_y", np.empty(0)), dtype=np.float64).ravel()
            _nc = np.asarray(self._mesh_data.get("cell_nodes", np.empty((0, 3), dtype=np.int32)))
            node_coords = np.column_stack([_nx, _ny]) if _nx.size > 0 and _ny.size > 0 else np.empty((0, 2), dtype=np.float64)
            cell_nodes = _nc.reshape((-1, 3)).astype(np.int32) if _nc.size > 0 else np.empty((0, 3), dtype=np.int32)
        else:
            node_coords = np.empty((0, 2), dtype=np.float64)
            cell_nodes = np.empty((0, 3), dtype=np.int32)
        out_ts: List[Dict[str, object]] = []
        out_prof: List[Dict[str, object]] = []
        for sm in sample_map:
            agg = _agg_svc(sm, h, hu, hv, cell_bed, h_min, g, t_accum)
            if agg is None:
                continue
            out_ts.append({
                "t_s": agg["t_s"], "line_id": agg["line_id"],
                "line_name": agg["line_name"],
                "depth_m": agg["depth_m"], "velocity_ms": agg["velocity_ms"],
                "wse_m": agg["wse_m"], "bed_m": agg["bed_m"],
                "flow_cms": agg["flow_cms"],
                "flow_cell_cms": agg["flow_cell_cms"],
                "flow_fv_cms": agg["flow_fv_cms"],
                "wet_frac": agg["wet_frac"], "fr": agg["fr"],
            })
            result = _svc(
                h=h, hu=hu, hv=hv, bed=cell_bed,
                node_coords=node_coords, cell_nodes=cell_nodes,
                line_xy=np.empty((0, 2), dtype=np.float64),
                h_min=h_min, timestep_s=float(t_accum), gravity=g,
                sample_map=sm,
            )
            station_arr = np.asarray(result.get("station_m", np.empty(0)), dtype=np.float64)
            if station_arr.size > 0:
                depth_arr = np.asarray(result.get("depth_m", np.empty(0)), dtype=np.float64)
                vel_arr = np.asarray(result.get("velocity_ms", np.empty(0)), dtype=np.float64)
                wse_arr = np.asarray(result.get("wse_m", np.empty(0)), dtype=np.float64)
                bed_arr = np.asarray(result.get("bed_m", np.empty(0)), dtype=np.float64)
                qn_arr = np.asarray(result.get("flow_qn", np.empty(0)), dtype=np.float64)
                wet_arr = np.asarray(result.get("wet", np.empty(0)), dtype=np.int32)
                fr_arr_p = np.asarray(result.get("froude", np.empty(0)), dtype=np.float64)
                n_p = min(station_arr.size, depth_arr.size, vel_arr.size, wse_arr.size, bed_arr.size, qn_arr.size, wet_arr.size, fr_arr_p.size)
                for j in range(n_p):
                    if not np.isfinite(station_arr[j]):
                        continue
                    out_prof.append({
                        "t_s": float(t_accum), "line_id": int(sm.get("line_id", -1)),
                        "line_name": str(sm.get("line_name", "") or ""),
                        "station_m": float(station_arr[j]),
                        "depth_m": float(depth_arr[j]) if np.isfinite(depth_arr[j]) else float("nan"),
                        "velocity_ms": float(vel_arr[j]) if np.isfinite(vel_arr[j]) else float("nan"),
                        "wse_m": float(wse_arr[j]) if np.isfinite(wse_arr[j]) else float("nan"),
                        "bed_m": float(bed_arr[j]) if np.isfinite(bed_arr[j]) else float("nan"),
                        "flow_qn": float(qn_arr[j]) if np.isfinite(qn_arr[j]) else float("nan"),
                        "wet": int(wet_arr[j]),
                        "fr": float(fr_arr_p[j]) if np.isfinite(fr_arr_p[j]) else float("nan"),
                    })
            else:
                idx = agg["_idx"]
                hh = agg["_hh"]; zb = agg["_zb"]; vel = agg["_vel"]
                qn = agg["_qn"]; wet = agg["_wet"]; fr_arr = agg["_fr_arr"]
                sta = np.asarray(sm.get("station_m", np.arange(idx.size, dtype=np.float64)), dtype=np.float64)
                if sta.size != idx.size:
                    sta = np.linspace(0.0, float(idx.size - 1), idx.size, dtype=np.float64)
                for j in range(idx.size):
                    out_prof.append({
                        "t_s": float(t_accum), "line_id": int(sm.get("line_id", -1)),
                        "line_name": str(sm.get("line_name", "") or ""),
                        "station_m": float(sta[j]), "depth_m": float(hh[j]),
                        "velocity_ms": float(vel[j]), "wse_m": float(hh[j] + zb[j]),
                        "bed_m": float(zb[j]), "flow_qn": float(qn[j]),
                        "wet": int(bool(wet[j])), "fr": float(fr_arr[j]),
                    })
        return out_ts, out_prof

    def _execute_run_request(self, request):
        """Execute a run request via the controller."""
        self._last_run_request = request
        self._controller.on_run(request)

    def _preflight_validate_mesh(self) -> dict:
        """Validate mesh and backend before a run (delegates to controller)."""
        return self._controller._preflight_validate_mesh()

    def _collect_bc_for_edges(self, edge_n0, edge_n1) -> dict:
        """Collect boundary conditions for edges (delegates to controller)."""
        return self._controller._collect_bc_for_edges(edge_n0, edge_n1)

    def _prepare_run_inputs(self) -> dict:
        """Prepare all run inputs (delegates to controller)."""
        return self._controller._prepare_run_inputs()

    def _collect_simulation_settings(self) -> dict:
        """Collect simulation settings from UI (delegates to controller)."""
        return self._controller._collect_simulation_settings()

    def _ensure_mesh_for_run_preflight(self):
        """Log an error if no mesh is loaded (preflight check)."""
        if self._mesh_data is None:
            self._log("Run aborted: no mesh loaded. Import mesh from map layers first.")
            return

    def _on_generate_mesh(self):
        """Handle generate-mesh request (stub, logs unavailable message)."""
        self._log("Structured mesh generation not available — import mesh from map layers instead.")

    def _has_mesh_for_run_preflight(self) -> bool:
        """Check whether mesh data is loaded."""
        return self._mesh_data is not None

    def _backend_ready_for_run_preflight(self) -> bool:
        """Check whether the GPU backend is available."""
        from swe2d.runtime.backend import swe2d_gpu_available
        return bool(swe2d_gpu_available())

    def _show_backend_unavailable_for_run_preflight(self, message: str):
        """Show a critical error dialog for backend unavailability."""
        QtWidgets.QMessageBox.critical(self, "2D SWE", str(message))

    # ── View helper methods (read widget values, delegate to services) ──

    def _mesh_boundary_edges(self):
        """Get boundary edge indices for the current mesh."""
        return _mesh_svc.mesh_boundary_edges(self._mesh_data)


    def _apply_bc_layer_overrides(self, edge_n0, edge_n1, bc_type, bc_val):
        """Apply boundary condition overrides from BC vector layer."""
        from swe2d.boundary_and_forcing.boundary_qgis_adapter import apply_bc_layer_overrides_qgis as _logic
        return _logic(
            mesh_data=self._mesh_data, have_qgis_core=_HAVE_QGIS_CORE,
            bc_lines_layer_combo=getattr(self._map_tab_view, "bc_lines_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            edge_n0=edge_n0, edge_n1=edge_n1, bc_type=bc_type, bc_val=bc_val,
            qgs_geometry_cls=QgsGeometry, qgs_pointxy_cls=QgsPointXY, log_fn=self._log,
        )

    def _mesh_cell_centroids(self):
        """Get cell centroid coordinates for the current mesh."""
        return _mesh_svc.mesh_cell_centroids(self._mesh_data)

    def _mesh_cell_areas(self):
        """Get cell areas for the current mesh."""
        return _mesh_svc.mesh_cell_areas(self._mesh_data)

    def _mesh_cell_min_bed(self):
        """Get minimum bed elevation across mesh cells."""
        return _mesh_svc.mesh_cell_min_bed(self._mesh_data)

    def _mesh_cell_solver_bed(self):
        """Get solver bed elevation values for mesh cells."""
        return _mesh_svc.mesh_cell_solver_bed(self._mesh_data)

    def get_n_mann_value(self) -> float:
        """Get the Manning's n roughness coefficient from the UI."""
        mtv = getattr(self, "_model_tab_view", None)
        if mtv is not None:
            spin = getattr(mtv, "n_mann_spin", None)
            if spin is not None:
                return float(spin.value())
        return 0.03

    def _mesh_cell_polygons(self) -> List:
        """Build QgsGeometry polygons for every mesh cell (needed by line sampling)."""
        if self._mesh_data is None:
            return []
        node_x = np.asarray(self._mesh_data["node_x"], dtype=np.float64)
        node_y = np.asarray(self._mesh_data["node_y"], dtype=np.float64)
        out: List = []
        if "cell_face_offsets" in self._mesh_data and "cell_face_nodes" in self._mesh_data:
            offs = np.asarray(self._mesh_data["cell_face_offsets"], dtype=np.int32)
            faces = np.asarray(self._mesh_data["cell_face_nodes"], dtype=np.int32)
            for i in range(offs.size - 1):
                s = int(offs[i])
                e = int(offs[i + 1])
                ids = faces[s:e]
                if ids.size < 3:
                    out.append(QgsGeometry())
                    continue
                ring = [QgsPointXY(float(node_x[n]), float(node_y[n])) for n in ids]
                ring.append(ring[0])
                out.append(QgsGeometry.fromPolygonXY([ring]))
            return out
        tris = np.asarray(self._mesh_data["cell_nodes"], dtype=np.int32).reshape((-1, 3))
        for tri in tris:
            ring = [
                QgsPointXY(float(node_x[int(tri[0])]), float(node_y[int(tri[0])])),
                QgsPointXY(float(node_x[int(tri[1])]), float(node_y[int(tri[1])])),
                QgsPointXY(float(node_x[int(tri[2])]), float(node_y[int(tri[2])])),
            ]
            ring.append(ring[0])
            out.append(QgsGeometry.fromPolygonXY([ring]))
        return out

    def _build_line_sampling_map(self) -> List[Dict[str, object]]:
        """Build line sampling map from the sample lines layer (delegates to service)."""
        from swe2d.workbench.services.line_sampling_service import build_line_sampling_map
        sample_lines_combo = getattr(self._map_tab_view, "sample_lines_layer_combo", None)
        line_layer = self._combo_layer(sample_lines_combo, "vector") if sample_lines_combo is not None else None
        if line_layer is None:
            self._log("[LineSampling] No sample lines layer selected — line results will be empty.")
        else:
            n_features = line_layer.featureCount() if hasattr(line_layer, "featureCount") else "?"
            self._log(f"[LineSampling] Using sample lines layer: {line_layer.name()} ({n_features} features)")
        smap = build_line_sampling_map(
            mesh_data=self._mesh_data,
            line_layer=line_layer,
            log_fn=self._log,
            mesh_cell_polygons_fn=self._mesh_cell_polygons,
            mesh_cell_centroids_fn=self._mesh_cell_centroids,
            mesh_cell_areas_fn=self._mesh_cell_areas,
        )
        self._log(f"[LineSampling] Built sampling map with {len(smap)} line(s)")
        return smap

    def _interp_hydrograph(self, hg, t_sec):
        """Interpolate a hydrograph at a given time."""
        from swe2d.boundary_and_forcing.hydrograph_logic import interp_hydrograph as _logic
        return _logic(hg, t_sec)

    def _parse_hydrograph_text(self, text):
        """Parse hydrograph text into time-value pairs."""
        from swe2d.boundary_and_forcing.hydrograph_logic import parse_hydrograph_text as _logic
        return _logic(text, parse_time_hours_fn=self._parse_time_hours)

    def _hydrograph_from_layer(self, layer, hydrograph_id="", bc_type=None):
        """Extract a hydrograph from a QGIS vector layer."""
        from swe2d.boundary_and_forcing.hydrograph_logic import hydrograph_from_layer as _logic
        return _logic(layer, hydrograph_id=hydrograph_id, bc_type=bc_type,
                      parse_time_hours_fn=self._parse_time_hours,
                      vector_layer_type=QgsVectorLayer)

    def _detect_map_unit(self):
        """Detect the map CRS length unit."""
        return _unit_svc.detect_map_unit(
            have_qgis_core=_HAVE_QGIS_CORE,
            project=QgsProject.instance() if _HAVE_QGIS_CORE and QgsProject is not None else None,
            log_fn=self._log,
        )

    def _build_spatial_cn_array(self):
        """Build a spatial curve number array from a CN layer."""
        from swe2d.boundary_and_forcing.spatial_forcing_qgis_adapter import build_spatial_cn_array_qgis as _logic
        return _logic(
            mesh_data=self._mesh_data, have_qgis_core=_HAVE_QGIS_CORE,
            cn_layer_combo=getattr(self._map_tab_view, "cn_layer_combo", None),
            combo_layer_fn=self._combo_layer,
            mesh_cell_centroids_fn=self._mesh_cell_centroids,
            default_cn=float(self._model_tab_view.cn_default_spin.value()),
            qgs_geometry_cls=QgsGeometry, qgs_pointxy_cls=QgsPointXY, log_fn=self._log,
        )

    def _boundary_buffer_cells(self, n_rings):
        """Get indices of cells within n rings of the mesh boundary."""
        return _mesh_svc.boundary_buffer_cells(self._mesh_data, n_rings)

    def _is_us_customary_units(self) -> bool:
        """Check whether the current unit system is US customary."""
        return _unit_svc.is_us_customary_units(
            getattr(self, "_length_unit_name", "m")
        )

    # ── View Protocol Methods (called by Service Layer, never widgets directly) ──

    def _open_batch_simulation_dialog(self) -> None:
        """Open the batch simulation dialog for parameter sweeps."""
        from swe2d.workbench.dialogs.batch_simulation_dialog import BatchSimulationDialog

        base_params = {
            "mesh": "",
            "params": {
                "rain_rate_mmhr": 0.0,
                "n_mann": 0.035,
                "duration_s": 3600.0,
            },
        }

        # Auto-populate mesh GPKG path from the current model if available
        gpkg = getattr(self, "_model_gpkg_path", "")
        if not gpkg or not os.path.isfile(gpkg):
            # Fall back to results GPKG path on the Model → Run/Output page
            mtv = getattr(self, "_model_tab_view", None)
            if mtv:
                pe = getattr(mtv, "results_gpkg_path_edit", None)
                if pe:
                    gpkg = str(pe.text() or "").strip()

        dlg = BatchSimulationDialog(
            parent=self,
            base_params=base_params,
            mesh_gpkg=gpkg,
        )
        dlg.exec()

    def set_run_button_enabled(self, enabled: bool) -> None:
        """Enable or disable the Run button."""
        if hasattr(self, "_model_tab_view") and hasattr(self._model_tab_view, "run_btn"):
            self._model_tab_view.run_btn.setEnabled(enabled)

    def set_cancel_button_enabled(self, enabled: bool) -> None:
        """Enable or disable the Cancel button."""
        if hasattr(self, "_model_tab_view") and hasattr(self._model_tab_view, "cancel_btn"):
            self._model_tab_view.cancel_btn.setEnabled(enabled)

    def set_run_progress(self, value: int) -> None:
        """Set the run progress bar value."""
        if hasattr(self, "_model_tab_view") and hasattr(self._model_tab_view, "progress_bar"):
            self._model_tab_view.progress_bar.setValue(value)

    def get_uniform_inflow_velocity(self) -> bool:
        """Return whether uniform inflow velocity is enabled (MapView protocol)."""
        mtv = getattr(self, "_map_tab_view", None)
        if mtv is None:
            return False
        chk = getattr(mtv, "uniform_inflow_velocity_chk", None)
        return bool(chk.isChecked()) if chk is not None else False

    def populate_layer_combo(self, combo_attr: str, layers: List,
                              layer_type_hint: str = "") -> None:
        """Populate a layer combo box with project layers."""
        combo = self._resolve_combo_attr(combo_attr)
        if combo is None:
            return
        from qgis.core import QgsRasterLayer
        current_id = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(none)", None)
        for layer in layers:
            if layer_type_hint == "raster" and not isinstance(layer, QgsRasterLayer):
                continue
            combo.addItem(str(layer.name()), layer.id())
        if current_id is not None:
            idx = combo.findData(current_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def get_combo_current_text(self, combo_attr: str) -> str:
        """Get the current text of a combo box by attribute name."""
        combo = self._resolve_combo_attr(combo_attr)
        return str(combo.currentText()) if combo is not None else ""

    def select_layer_in_combo(self, combo_attr: str, layer_id: str) -> None:
        """Select a layer in a combo box by layer ID."""
        combo = self._resolve_combo_attr(combo_attr)
        if combo is not None:
            idx = combo.findData(layer_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def get_topo_combo(self, attr: str) -> Any:
        """Get a topology tab combo widget by attribute name."""
        tv = getattr(self, "_topology_tab_view", None)
        if tv is None:
            return None
        return getattr(tv, attr, None)

    def _resolve_combo_attr(self, attr: str) -> Any:
        """Resolve a combo widget by attribute name across all tab views."""
        for tab_attr in ("_map_tab_view", "_model_tab_view", "_topology_tab_view"):
            tab = getattr(self, tab_attr, None)
            if tab is not None:
                w = getattr(tab, attr, None)
                if w is not None:
                    return w
        return None

    def set_layer_status_text(self, text: str) -> None:
        """Set the layer status label text."""
        if hasattr(self, "_map_tab_view"):
            try:
                self._map_tab_view.layer_status_lbl.setText(text)
            except RuntimeError:
                pass

    def set_results_gpkg_path(self, path: str) -> None:
        """Set the results GeoPackage path in the UI."""
        mtv = getattr(self, "_model_tab_view", None)
        edit = getattr(mtv, "results_gpkg_path_edit", None) if mtv is not None else None
        if edit is not None:
            try:
                edit.setText(str(path))
            except RuntimeError:
                pass

    def get_combo_selected_layer(self, combo_attr: str, kind: str = "vector") -> Any:
        """Get the currently selected layer from a combo box."""
        combo = self._resolve_combo_attr(combo_attr)
        if combo is None:
            return None
        return self._combo_layer(combo, kind)

    def get_combo_widget(self, combo_attr: str) -> Any:
        """Get a combo widget by attribute name."""
        return self._resolve_combo_attr(combo_attr)

    def sync_overlay_widget_states(self) -> None:
        """Sync overlay widget enable states based on current selections."""
        tb = getattr(self, "_results_toolbox", None)
        if tb is None:
            return
        field_key = str(tb.field_combo.currentData() or "depth")
        lock_canvas = bool(tb.lock_canvas_chk.isChecked())
        tb.res_combo.setEnabled(not lock_canvas)
        tb.wse_render_combo.setEnabled(field_key == "wse")
        arrows_on = bool(tb.arrows_chk.isChecked())
        for attr in ("arrow_density_spin", "arrow_length_spin",
                     "arrow_head_length_spin", "arrow_head_width_spin"):
            w = getattr(tb, attr, None)
            if w is not None:
                w.setEnabled(arrows_on)
        stream_on = bool(tb.streamlines_chk.isChecked())
        for attr in ("streamline_backend_combo", "streamline_seed_spin", "streamline_steps_spin"):
            w = getattr(tb, attr, None)
            if w is not None:
                w.setEnabled(stream_on)

    def get_overlay_export_field(self) -> str:
        """Get the current overlay export field name."""
        tb = self._results_toolbox
        combo = getattr(tb, "field_combo", None)
        if combo is None:
            return "depth"
        return str(combo.currentData() or "depth")

    def get_overlay_export_cmap(self) -> str:
        """Get the current overlay export colormap."""
        tb = self._results_toolbox
        combo = getattr(tb, "cmap_combo", None)
        if combo is None:
            return "turbo"
        return str(combo.currentData() or "turbo")

    def get_overlay_export_wse_render_mode(self) -> str:
        """Get the current WSE render mode for overlay export."""
        tb = self._results_toolbox
        combo = getattr(tb, "wse_render_combo", None)
        if combo is None:
            return "cell"
        return str(combo.currentData() or "cell")

    def get_overlay_auto_contrast(self) -> bool:
        """Get auto-contrast setting for overlay."""
        tb = self._results_toolbox
        return bool(tb.auto_contrast_chk.isChecked())

    def collect_run_widget_params(self) -> dict:
        """Collect all run parameter values from UI widgets."""
        mtab = self._model_tab_view
        rtb = self._results_toolbox
        # Commit any in-progress editor text before reading values.
        for w in (mtab.gpu_diag_sync_interval_spin,):
            try:
                w.interpretText()
            except Exception as _e:

                try:

                    self._log(f"[ERROR] Exception in studio_dialog.py: {_e}")

                except Exception:

                    pass
        return {
            "gravity": float(self._gravity),
            "k_mann": float(self._k_mann),
            "n_mann_spin": float(mtab.n_mann_spin.value()),
            "cfl_spin": float(mtab.cfl_spin.value()),
            "h_min_spin": float(mtab.h_min_spin.value()),
            "dt_spin": float(mtab.dt_spin.value()),
            "initial_dt_spin": float(mtab.initial_dt_spin.value()),
            "adaptive_cfl_dt_chk": bool(mtab.adaptive_cfl_dt_chk.isChecked()),
            "reconstruction_combo": int(mtab.reconstruction_combo.currentData()),
            "reconstruction_combo_text": str(mtab.reconstruction_combo.currentText()).strip(),
            "temporal_order_combo": int(mtab.temporal_order_combo.currentData()),
            "temporal_order_combo_text": str(mtab.temporal_order_combo.currentText()).strip(),
            "cfl_lambda_cap_spin": float(mtab.cfl_lambda_cap_spin.value()),
            "gpu_diag_sync_interval_spin": int(mtab.gpu_diag_sync_interval_spin.value()),
            "max_rel_depth_increase_spin": float(mtab.max_rel_depth_increase_spin.value()),
            "max_source_depth_step_spin": float(mtab.max_source_depth_step_spin.value()),
            "max_source_rate_spin": float(mtab.max_source_rate_spin.value()),
            "extreme_rain_mode_chk": bool(mtab.extreme_rain_mode_chk.isChecked()),
            "source_cfl_beta_spin": float(mtab.source_cfl_beta_spin.value()),
            "source_max_substeps_spin": int(mtab.source_max_substeps_spin.value()),
            "source_true_subcycling_chk": bool(mtab.source_true_subcycling_chk.isChecked()),
            "source_imex_split_chk": bool(mtab.source_imex_split_chk.isChecked()),
            "shallow_damping_depth_spin": float(mtab.shallow_damping_depth_spin.value()),
            "depth_cap_spin": float(mtab.depth_cap_spin.value()),
            "momentum_cap_min_speed_spin": float(mtab.momentum_cap_min_speed_spin.value()),
            "momentum_cap_celerity_mult_spin": float(mtab.momentum_cap_celerity_mult_spin.value()),
            "max_inv_area_spin": float(mtab.max_inv_area_spin.value()),
            "rain_rate_spin": float(mtab.rain_rate_spin.value()),
            "output_interval_edit": str(mtab.output_interval_edit.text()),
            "line_output_interval_edit": str(mtab.line_output_interval_edit.text()),
            "tiny_mode_combo": int(mtab.tiny_mode_combo.currentData()),
            "tiny_wet_cell_threshold_spin": int(mtab.tiny_wet_cell_threshold_spin.value()),
            "source_stage_coupled_imex_rk2_chk": bool(mtab.source_stage_coupled_imex_rk2_chk.isChecked()),
            "inflow_progressive_chk": bool(self._map_tab_view.inflow_progressive_chk.isChecked()),
            "use_redistribution_chk": bool(mtab.use_redistribution_chk.isChecked()),
            "gpu_diag_sync_interval_raw": int(mtab.gpu_diag_sync_interval_spin.value()),
            "extended_outputs_chk": bool(rtb.extended_outputs_chk.isChecked()),
            "swe2d_perf_mode_chk": bool(mtab.swe2d_perf_mode_chk.isChecked()),
            "culvert_face_flux_chk": bool(mtab.culvert_face_flux_chk.isChecked()),
            "enable_cuda_graphs_chk": bool(mtab.enable_cuda_graphs_chk.isChecked()),
            "save_mesh_results_to_gpkg_chk": bool(rtb.save_mesh_chk.isChecked()) and not bool(rtb.save_max_only_chk.isChecked()),
            "save_line_results_to_gpkg_chk": bool(rtb.save_line_chk.isChecked()) and not bool(rtb.save_max_only_chk.isChecked()),
            "save_coupling_results_to_gpkg_chk": bool(rtb.save_coupling_chk.isChecked()) and not bool(rtb.save_max_only_chk.isChecked()),
            "save_max_only_chk": bool(rtb.save_max_only_chk.isChecked()),
            "save_run_log_to_gpkg_chk": bool(rtb.save_log_chk.isChecked()),
            "degen_mode": int(mtab.degen_mode_combo.currentData()),
            "front_flux_damping_spin": float(mtab.front_flux_damping_spin.value()),
            "active_set_hysteresis_chk": bool(mtab.active_set_hysteresis_chk.isChecked()),
            "drainage_gpu_method": str(mtab.drainage_gpu_method_combo.currentData()),
            "culvert_solver_mode": int(mtab.culvert_solver_mode_combo.currentData()),
            "bridge_coupling_mode": str(mtab.bridge_stacked_coupling_mode_combo.currentData()),
        }

    def _log_exception(self, context: str, exc: Exception) -> None:
        """Log an exception with traceback to the runtime log."""
        self._log(f"{context}: {exc}")
        tb_txt = traceback.format_exc()
        if tb_txt:
            self._log("--- traceback begin ---")
            for ln in tb_txt.rstrip().splitlines():
                self._log(ln)
            self._log("--- traceback end ---")
