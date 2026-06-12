from __future__ import annotations

from swe2d import units as _u

# Extracted methods depend on symbols defined in swe2d_workbench_qt.
from swe2d_workbench_qt import *  # type: ignore F401,F403
from swe2d_workbench_qt import (
    _BC_VALUE_MAP,
    _CELL_TYPE_OPTIONS,
    _DRAIN_LINK_SHAPE_VALUE_MAP,
    _DRAIN_LINK_TYPE_VALUE_MAP,
    _DRAIN_NODE_TYPE_VALUE_MAP,
    _HAVE_H5PY,
    _HAVE_NETCDF4,
    _HAVE_QGIS_CORE,
    _HYETOGRAPH_UNITS_VALUE_MAP,
    _HYETOGRAPH_VALUE_TYPE_MAP,
    _NETCDF4_IMPORT_ERROR,
    _RAIN_GAGE_UNITS_VALUE_MAP,
    _STRUCTURE_TYPE_VALUE_MAP,
    _ensure_netcdf4_available,
    _gmsh_available,
    _h5py,
    _netCDF4,
)

def _bind_topology_tab_dynamic_controls(self, topology_tab_page: QtWidgets.QWidget, topo_layout: QtWidgets.QGridLayout) -> None:
    def _ensure(widget: QtWidgets.QWidget, row: int, col: int, row_span: int = 1, col_span: int = 1) -> None:
        idx = topo_layout.indexOf(widget)
        if idx >= 0:
            try:
                cur_row, cur_col, cur_row_span, cur_col_span = topo_layout.getItemPosition(idx)
                if (
                    int(cur_row) == int(row)
                    and int(cur_col) == int(col)
                    and int(cur_row_span) == int(row_span)
                    and int(cur_col_span) == int(col_span)
                ):
                    return
            except Exception as e:
                self._log(f"[ERROR] layout position check: {e}")
            topo_layout.removeWidget(widget)
        topo_layout.addWidget(widget, row, col, row_span, col_span)

    def _find_child_robust(widget_type: type, name: str):
        """Work around PyQt5 findChild intermittently returning None for widgets
        that DO exist in the tree (found by findChildren).  Always use
        findChildren first, then return the first match or None."""
        children = topology_tab_page.findChildren(widget_type, name)
        return children[0] if children else None

    def _find_or_create_combo(name: str, row: int) -> QtWidgets.QComboBox:
        w = _find_child_robust(QtWidgets.QComboBox, name)
        if w is None:
            w = QtWidgets.QComboBox()
            w.setObjectName(name)
        _ensure(w, row, 1)
        return w

    def _set_combo_items(
        combo: QtWidgets.QComboBox,
        items: List[Tuple[str, object]],
        default_data: Optional[object] = None,
    ) -> None:
        prev_data = combo.currentData()
        prev_text = combo.currentText()
        combo.blockSignals(True)
        try:
            combo.clear()
            for label, data in items:
                combo.addItem(label, data)
            idx = -1
            if prev_data is not None:
                idx = combo.findData(prev_data)
            if idx < 0 and default_data is not None:
                idx = combo.findData(default_data)
            if idx < 0 and prev_text:
                idx = combo.findText(prev_text)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)

    def _find_or_create_double_spin(name: str) -> QtWidgets.QDoubleSpinBox:
        w = _find_child_robust(QtWidgets.QDoubleSpinBox, name)
        if w is None:
            w = QtWidgets.QDoubleSpinBox()
            w.setObjectName(name)
        return w

    def _find_or_create_spin(name: str) -> QtWidgets.QSpinBox:
        w = _find_child_robust(QtWidgets.QSpinBox, name)
        if w is None:
            w = QtWidgets.QSpinBox()
            w.setObjectName(name)
        return w

    def _find_or_create_line_edit(name: str, text: str) -> QtWidgets.QLineEdit:
        w = _find_child_robust(QtWidgets.QLineEdit, name)
        if w is None:
            w = QtWidgets.QLineEdit(text)
            w.setObjectName(name)
        if not str(w.text() or "").strip():
            w.setText(text)
        return w

    def _find_or_create_check(name: str, text: str) -> QtWidgets.QCheckBox:
        w = _find_child_robust(QtWidgets.QCheckBox, name)
        if w is None:
            w = QtWidgets.QCheckBox(text)
            w.setObjectName(name)
        if not str(w.text() or "").strip():
            w.setText(text)
        return w

    def _find_or_create_form_container(
        name: str,
        row: int,
        col: int = 1,
        row_span: int = 1,
        col_span: int = 1,
    ) -> QtWidgets.QFormLayout:
        container = _find_child_robust(QtWidgets.QWidget, name)
        if container is None:
            container = QtWidgets.QWidget()
            container.setObjectName(name)
        _ensure(container, row, col, row_span, col_span)
        layout = container.layout()
        if not isinstance(layout, QtWidgets.QFormLayout):
            layout = QtWidgets.QFormLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        return layout

    def _ensure_form_row(form: QtWidgets.QFormLayout, widget: QtWidgets.QWidget, label: Optional[str] = None) -> None:
        try:
            row, _ = form.getWidgetPosition(widget)
            if row >= 0:
                return
        except Exception as e:
            self._log(f"[ERROR] form row position check: {e}")
        if label is None:
            form.addRow(widget)
        else:
            form.addRow(label, widget)

    def _reconnect(signal: object, callback: Callable[[], None]) -> None:
        safe_disconnect(signal, callback)
        signal.connect(callback)

    self.topo_nodes_combo = _find_or_create_combo("topo_nodes_combo", 0)
    self.topo_arcs_combo = _find_or_create_combo("topo_arcs_combo", 1)
    self.topo_regions_combo = _find_or_create_combo("topo_regions_combo", 2)
    self.topo_constraints_combo = _find_or_create_combo("topo_constraints_combo", 3)
    self.topo_quad_edges_combo = _find_or_create_combo("topo_quad_edges_combo", 4)
    self.topo_backend_combo = _find_or_create_combo("topo_backend_combo", 5)
    self.topo_default_size_spin = _find_or_create_double_spin("topo_default_size_spin")
    _ensure(self.topo_default_size_spin, 6, 1)
    self.topo_default_cell_type_combo = _find_or_create_combo("topo_default_cell_type_combo", 7)

    _set_combo_items(self.topo_constraints_combo, [("(none)", None)], default_data=None)
    _set_combo_items(self.topo_quad_edges_combo, [("(none)", None)], default_data=None)

    _gmsh_label = "Gmsh (recommended)" if _gmsh_available() else "Gmsh (install: pip install gmsh)"
    _set_combo_items(
        self.topo_backend_combo,
        [
            (_gmsh_label, "gmsh"),
            ("Structured (built-in fallback)", "structured"),
        ],
        default_data="gmsh",
    )
    _set_combo_items(
        self.topo_default_cell_type_combo,
        [
            ("triangular", "triangular"),
            ("quadrilateral", "quadrilateral"),
            ("cartesian", "cartesian"),
            ("empty", "empty"),
        ],
        default_data="triangular",
    )

    self.topo_default_size_spin.setRange(0.01, 1.0e6)
    self.topo_default_size_spin.setDecimals(3)
    self.topo_default_size_spin.setValue(20.0)

    gmsh_form = _find_or_create_form_container("topo_gmsh_controls_widget", 8, col=1)
    quality_form = _find_or_create_form_container("topo_quality_controls_widget", 9, col=1)
    self.topo_gmsh_controls_widget = _find_child_robust(QtWidgets.QWidget, "topo_gmsh_controls_widget")
    self.topo_quality_controls_widget = _find_child_robust(QtWidgets.QWidget, "topo_quality_controls_widget")
    self.topo_backend_controls_lbl = _find_child_robust(QtWidgets.QLabel, "topo_gmsh_controls_lbl")
    if self.topo_backend_controls_lbl is None:
        self.topo_backend_controls_lbl = QtWidgets.QLabel("Backend advanced controls:")
        self.topo_backend_controls_lbl.setObjectName("topo_gmsh_controls_lbl")
    _ensure(self.topo_backend_controls_lbl, 8, 0)

    self.topo_quality_controls_lbl = _find_child_robust(QtWidgets.QLabel, "topo_quality_controls_lbl")
    if self.topo_quality_controls_lbl is None:
        self.topo_quality_controls_lbl = QtWidgets.QLabel("Quality controls (Gmsh):")
        self.topo_quality_controls_lbl.setObjectName("topo_quality_controls_lbl")
    _ensure(self.topo_quality_controls_lbl, 9, 0)

    # ── Quality controls: all widgets now live in the .ui form layouts ──
    # No more programmatic group box creation – widgets stay where Qt Designer put them.

    self.topo_gmsh_tri_algo_combo = _find_child_robust(QtWidgets.QComboBox, "topo_gmsh_tri_algo_combo")
    if self.topo_gmsh_tri_algo_combo is None:
        self.topo_gmsh_tri_algo_combo = QtWidgets.QComboBox()
        self.topo_gmsh_tri_algo_combo.setObjectName("topo_gmsh_tri_algo_combo")
        gmsh_form.addRow("Triangle algorithm:", self.topo_gmsh_tri_algo_combo)
    _set_combo_items(
        self.topo_gmsh_tri_algo_combo,
        [
            ("Frontal-Delaunay (quality)", 6),
            ("Delaunay (faster)", 5),
        ],
        default_data=6,
    )

    self.topo_gmsh_quad_algo_combo = _find_child_robust(QtWidgets.QComboBox, "topo_gmsh_quad_algo_combo")
    if self.topo_gmsh_quad_algo_combo is None:
        self.topo_gmsh_quad_algo_combo = QtWidgets.QComboBox()
        self.topo_gmsh_quad_algo_combo.setObjectName("topo_gmsh_quad_algo_combo")
        gmsh_form.addRow("Quadrilateral algorithm:", self.topo_gmsh_quad_algo_combo)
    _set_combo_items(
        self.topo_gmsh_quad_algo_combo,
        [
            ("Frontal + Blossom recombine", 6),
            ("Delaunay + Blossom recombine", 5),
            ("Packing of Parallelograms", 9),
        ],
        default_data=6,
    )

    self.topo_gmsh_recombine_algo_combo = _find_child_robust(QtWidgets.QComboBox, "topo_gmsh_recombine_algo_combo")
    if self.topo_gmsh_recombine_algo_combo is None:
        self.topo_gmsh_recombine_algo_combo = QtWidgets.QComboBox()
        self.topo_gmsh_recombine_algo_combo.setObjectName("topo_gmsh_recombine_algo_combo")
        gmsh_form.addRow("Recombine algorithm:", self.topo_gmsh_recombine_algo_combo)
    _set_combo_items(
        self.topo_gmsh_recombine_algo_combo,
        [
            ("Simple", 0),
            ("Blossom", 1),
            ("Simple full-quad", 2),
        ],
        default_data=1,
    )

    self.topo_gmsh_global_recombine_chk = _find_or_create_check(
        "topo_gmsh_global_recombine_chk",
        "Apply global recombine pass after mesh generation",
    )
    self.topo_gmsh_global_recombine_chk.setChecked(False)
    self.topo_gmsh_global_recombine_chk.setToolTip(
        "If enabled, runs gmsh.model.mesh.recombine() globally after mesh generation. "
        "Default is off to avoid recombining non-quad-targeted regions."
    )
    if self.topo_gmsh_global_recombine_chk.parent() is None:
        gmsh_form.addRow(self.topo_gmsh_global_recombine_chk)

    self.topo_gmsh_quad_full_region_flow_align_chk = _find_or_create_check(
        "topo_gmsh_quad_full_region_flow_align_chk",
        "Gmsh full-region flow-aligned quads",
    )
    self.topo_gmsh_quad_full_region_flow_align_chk.setChecked(True)
    self.topo_gmsh_quad_full_region_flow_align_chk.setToolTip(
        "For quadrilateral/channel-generator regions with complete quad-edge controls, "
        "apply TransfiniteCurve + TransfiniteSurface + Recombine so edge-aligned spacing "
        "propagates across the full region."
    )
    if self.topo_gmsh_quad_full_region_flow_align_chk.parent() is None:
        gmsh_form.addRow(self.topo_gmsh_quad_full_region_flow_align_chk)

    self.topo_gmsh_smoothing_spin = _find_or_create_spin("topo_gmsh_smoothing_spin")
    self.topo_gmsh_smoothing_spin.setRange(0, 100)
    self.topo_gmsh_smoothing_spin.setValue(0)
    if self.topo_gmsh_smoothing_spin.parent() is None:
        gmsh_form.addRow("Smoothing passes:", self.topo_gmsh_smoothing_spin)

    self.topo_gmsh_optimize_iters_spin = _find_or_create_spin("topo_gmsh_optimize_iters_spin")
    self.topo_gmsh_optimize_iters_spin.setRange(0, 100)
    self.topo_gmsh_optimize_iters_spin.setValue(0)
    if self.topo_gmsh_optimize_iters_spin.parent() is None:
        gmsh_form.addRow("Optimize iterations:", self.topo_gmsh_optimize_iters_spin)

    self.topo_gmsh_verbosity_spin = _find_or_create_spin("topo_gmsh_verbosity_spin")
    self.topo_gmsh_verbosity_spin.setRange(0, 10)
    self.topo_gmsh_verbosity_spin.setValue(2)
    if self.topo_gmsh_verbosity_spin.parent() is None:
        gmsh_form.addRow("Verbosity:", self.topo_gmsh_verbosity_spin)

    self.topo_gmsh_num_threads_spin = _find_or_create_spin("topo_gmsh_num_threads_spin")
    self.topo_gmsh_num_threads_spin.setRange(0, 256)
    self.topo_gmsh_num_threads_spin.setValue(1)
    self.topo_gmsh_num_threads_spin.setToolTip(
        "General.NumThreads. Set 0 to use Gmsh default/auto behavior."
    )
    if self.topo_gmsh_num_threads_spin.parent() is None:
        gmsh_form.addRow("Num threads:", self.topo_gmsh_num_threads_spin)

    self.topo_gmsh_max_num_threads_2d_spin = _find_or_create_spin("topo_gmsh_max_num_threads_2d_spin")
    self.topo_gmsh_max_num_threads_2d_spin.setRange(0, 256)
    self.topo_gmsh_max_num_threads_2d_spin.setValue(0)
    self.topo_gmsh_max_num_threads_2d_spin.setToolTip(
        "Mesh.MaxNumThreads2D cap. Set 0 to keep Gmsh default/auto behavior."
    )
    if self.topo_gmsh_max_num_threads_2d_spin.parent() is None:
        gmsh_form.addRow("Max 2D threads:", self.topo_gmsh_max_num_threads_2d_spin)

    self.topo_gmsh_optimize_netgen_chk = _find_or_create_check("topo_gmsh_optimize_netgen_chk", "Enable Netgen optimize")
    if self.topo_gmsh_optimize_netgen_chk.parent() is None:
        gmsh_form.addRow(self.topo_gmsh_optimize_netgen_chk)

    self.topo_gmsh_arc_mode_combo = _find_child_robust(QtWidgets.QComboBox, "topo_gmsh_arc_mode_combo")
    if self.topo_gmsh_arc_mode_combo is None:
        self.topo_gmsh_arc_mode_combo = QtWidgets.QComboBox()
        self.topo_gmsh_arc_mode_combo.setObjectName("topo_gmsh_arc_mode_combo")
    _set_combo_items(
        self.topo_gmsh_arc_mode_combo,
        [
            ("Hard embed arcs (strict)", "hard_embed"),
            ("Soft arc size hint (non-strict)", "soft_size_hint"),
            ("Disable arc influence", "disabled"),
        ],
        default_data="hard_embed",
    )
    self.topo_gmsh_arc_soft_size_factor_spin = _find_or_create_double_spin("topo_gmsh_arc_soft_size_factor_spin")
    self.topo_gmsh_arc_soft_size_factor_spin.setRange(0.05, 1.0)
    self.topo_gmsh_arc_soft_size_factor_spin.setDecimals(3)
    self.topo_gmsh_arc_soft_size_factor_spin.setSingleStep(0.05)
    self.topo_gmsh_arc_soft_size_factor_spin.setValue(0.5)
    self.topo_gmsh_arc_soft_size_factor_spin.setToolTip(
        "Soft arc mode target-size factor near arcs. Lower values force finer cells along arc corridors."
    )
    self.topo_gmsh_arc_soft_dist_factor_spin = _find_or_create_double_spin("topo_gmsh_arc_soft_dist_factor_spin")
    self.topo_gmsh_arc_soft_dist_factor_spin.setRange(0.1, 10.0)
    self.topo_gmsh_arc_soft_dist_factor_spin.setDecimals(3)
    self.topo_gmsh_arc_soft_dist_factor_spin.setSingleStep(0.1)
    self.topo_gmsh_arc_soft_dist_factor_spin.setValue(2.0)
    self.topo_gmsh_arc_soft_dist_factor_spin.setToolTip(
        "Soft arc mode influence distance factor. Higher values widen arc-driven refinement corridors."
    )
    self.topo_gmsh_interface_transition_enable_chk = _find_or_create_check(
        "topo_gmsh_interface_transition_enable_chk",
        "Enable interface transition grading",
    )
    self.topo_gmsh_interface_transition_enable_chk.setChecked(True)
    self.topo_gmsh_interface_transition_enable_chk.setToolTip(
        "Apply Distance/Threshold grading near shared interfaces on non-transfinite regions only."
    )
    self.topo_gmsh_interface_transition_dist_factor_spin = _find_or_create_double_spin(
        "topo_gmsh_interface_transition_dist_factor_spin"
    )
    self.topo_gmsh_interface_transition_dist_factor_spin.setRange(0.25, 20.0)
    self.topo_gmsh_interface_transition_dist_factor_spin.setDecimals(3)
    self.topo_gmsh_interface_transition_dist_factor_spin.setSingleStep(0.25)
    self.topo_gmsh_interface_transition_dist_factor_spin.setValue(2.5)
    self.topo_gmsh_interface_transition_dist_factor_spin.setToolTip(
        "Distance multiplier for interface grading influence width. Higher values widen the transition band."
    )
    self.topo_gmsh_interface_transition_min_ratio_spin = _find_or_create_double_spin(
        "topo_gmsh_interface_transition_min_ratio_spin"
    )
    self.topo_gmsh_interface_transition_min_ratio_spin.setRange(1.0, 10.0)
    self.topo_gmsh_interface_transition_min_ratio_spin.setDecimals(3)
    self.topo_gmsh_interface_transition_min_ratio_spin.setSingleStep(0.05)
    self.topo_gmsh_interface_transition_min_ratio_spin.setValue(1.25)
    self.topo_gmsh_interface_transition_min_ratio_spin.setToolTip(
        "Only apply interface grading when adjacent region target sizes differ by at least this ratio."
    )
    self.topo_gmsh_interface_conformance_chk = _find_or_create_check(
        "topo_gmsh_interface_conformance_chk",
        "Enable transverse interface conformance post-process",
    )
    self.topo_gmsh_interface_conformance_chk.setChecked(False)
    self.topo_gmsh_interface_conformance_chk.setToolTip(
        "Snap and weld mixed-interface nodes after Gmsh extraction to enforce shared boundary topology."
    )
    self.topo_gmsh_transverse_interface_centroid_merge_chk = _find_or_create_check(
        "topo_gmsh_transverse_interface_centroid_merge_chk",
        "Use centroid merge for matched transverse interface nodes",
    )
    self.topo_gmsh_transverse_interface_centroid_merge_chk.setChecked(False)
    self.topo_gmsh_transverse_interface_centroid_merge_chk.setToolTip(
        "Move matched interface-node groups to their centroid before welding instead of one-sided snapping."
    )
    self.topo_gmsh_interface_snap_tol_spin = _find_or_create_double_spin("topo_gmsh_interface_snap_tol_spin")
    self.topo_gmsh_interface_snap_tol_spin.setRange(1.0e-6, 1.0e5)
    self.topo_gmsh_interface_snap_tol_spin.setDecimals(6)
    self.topo_gmsh_interface_snap_tol_spin.setValue(1.0)
    self.topo_gmsh_interface_snap_tol_spin.setToolTip(
        "Distance tolerance used by transverse interface conformance snapping."
    )
    self.topo_gmsh_interface_reject_near_unshared_chk = _find_or_create_check(
        "topo_gmsh_interface_reject_near_unshared_chk",
        "Reject mixed interfaces with near-coincident unshared nodes",
    )
    self.topo_gmsh_interface_reject_near_unshared_chk.setChecked(True)
    self.topo_gmsh_interface_reject_near_unshared_chk.setToolTip(
        "Fail meshing when a transfinite/tri interface shows hanging-node style near-miss pairs."
    )
    self.topo_gmsh_interface_reject_tol_spin = _find_or_create_double_spin("topo_gmsh_interface_reject_tol_spin")
    self.topo_gmsh_interface_reject_tol_spin.setRange(1.0e-6, 1.0e3)
    self.topo_gmsh_interface_reject_tol_spin.setDecimals(6)
    self.topo_gmsh_interface_reject_tol_spin.setValue(1.0e-3)
    self.topo_gmsh_interface_reject_tol_spin.setToolTip(
        "Tolerance for detecting near-coincident unshared interface nodes (hanging-node signature)."
    )
    self.topo_gmsh_mesh_size_min_spin = _find_or_create_double_spin("topo_gmsh_mesh_size_min_spin")
    self.topo_gmsh_mesh_size_min_spin.setRange(0.0, 1.0e6)
    self.topo_gmsh_mesh_size_min_spin.setDecimals(6)
    self.topo_gmsh_mesh_size_min_spin.setValue(0.0)
    if self.topo_gmsh_mesh_size_min_spin.parent() is None:
        gmsh_form.addRow("Global min cell size:", self.topo_gmsh_mesh_size_min_spin)

    self.topo_gmsh_tolerance_edge_length_spin = _find_or_create_double_spin("topo_gmsh_tolerance_edge_length_spin")
    self.topo_gmsh_tolerance_edge_length_spin.setRange(0.0, 1.0e6)
    self.topo_gmsh_tolerance_edge_length_spin.setDecimals(6)
    self.topo_gmsh_tolerance_edge_length_spin.setValue(0.0)
    if self.topo_gmsh_tolerance_edge_length_spin.parent() is None:
        gmsh_form.addRow("Ignore edges shorter than:", self.topo_gmsh_tolerance_edge_length_spin)

    self.topo_gmsh_mesh_size_from_points_chk = _find_or_create_check(
        "topo_gmsh_mesh_size_from_points_chk", "Use region target_size for mesh sizing"
    )
    self.topo_gmsh_mesh_size_from_points_chk.setChecked(True)
    if self.topo_gmsh_mesh_size_from_points_chk.parent() is None:
        gmsh_form.addRow(self.topo_gmsh_mesh_size_from_points_chk)

    self.topo_gmsh_quality_enable_chk = _find_or_create_check(
        "topo_gmsh_quality_enable_chk", "Enable Gmsh iterative quality loop"
    )
    self.topo_gmsh_quality_enable_chk.setChecked(False)

    self.topo_gmsh_quality_max_iters_spin = _find_or_create_spin("topo_gmsh_quality_max_iters_spin")
    self.topo_gmsh_quality_max_iters_spin.setRange(1, 50)
    self.topo_gmsh_quality_max_iters_spin.setValue(2)

    self.topo_gmsh_quality_time_limit_spin = _find_or_create_double_spin("topo_gmsh_quality_time_limit_spin")
    self.topo_gmsh_quality_time_limit_spin.setRange(1.0, 3600.0)
    self.topo_gmsh_quality_time_limit_spin.setDecimals(1)
    self.topo_gmsh_quality_time_limit_spin.setValue(55.0)

    self.topo_quality_min_angle_spin = _find_or_create_double_spin("topo_quality_min_angle_spin")
    self.topo_quality_min_angle_spin.setRange(0.0, 89.0)
    self.topo_quality_min_angle_spin.setDecimals(1)
    self.topo_quality_min_angle_spin.setValue(5.0)

    self.topo_quality_max_aspect_spin = _find_or_create_double_spin("topo_quality_max_aspect_spin")
    self.topo_quality_max_aspect_spin.setRange(1.0, 1.0e4)
    self.topo_quality_max_aspect_spin.setDecimals(2)
    self.topo_quality_max_aspect_spin.setValue(20.0)

    self.topo_quality_max_non_orth_spin = _find_or_create_double_spin("topo_quality_max_non_orth_spin")
    self.topo_quality_max_non_orth_spin.setRange(1.0, 89.9)
    self.topo_quality_max_non_orth_spin.setDecimals(1)
    self.topo_quality_max_non_orth_spin.setValue(82.0)

    self.topo_quality_min_area_edit = _find_or_create_line_edit("topo_quality_min_area_edit", "1e-14")

    self.topo_quality_size_scales_edit = _find_or_create_line_edit("topo_quality_size_scales_edit", "1.0,0.9,0.8,0.7")
    self.topo_quality_size_scales_edit.setToolTip(
        "Comma-separated per-attempt size multipliers. Example: 1.0,0.9,0.8"
    )

    self.topo_quality_smooth_increments_edit = _find_or_create_line_edit("topo_quality_smooth_increments_edit", "0,2,4,6")
    self.topo_quality_smooth_increments_edit.setToolTip(
        "Comma-separated extra smoothing passes added per attempt. Example: 0,2,4,6"
    )

    self.topo_gmsh_quality_recombine_topology_passes_edit = _find_or_create_line_edit(
        "topo_gmsh_quality_recombine_topology_passes_edit", "5,12,20"
    )
    self.topo_gmsh_quality_recombine_topology_passes_edit.setToolTip(
        "Comma-separated topological optimization passes for quad recombination per attempt. "
        "Higher values can improve quad layout but cost more runtime."
    )

    self.topo_gmsh_quality_recombine_min_quality_edit = _find_or_create_line_edit(
        "topo_gmsh_quality_recombine_min_quality_edit", "0.01,0.03,0.06"
    )
    self.topo_gmsh_quality_recombine_min_quality_edit.setToolTip(
        "Comma-separated minimum acceptable recombined quad quality per attempt. "
        "Typical range: 0.0 to 0.2."
    )

    self.topo_gmsh_quality_random_factors_edit = _find_or_create_line_edit(
        "topo_gmsh_quality_random_factors_edit", "1e-9,1e-7,1e-6"
    )
    self.topo_gmsh_quality_random_factors_edit.setToolTip(
        "Comma-separated Mesh.RandomFactor values per attempt. "
        "Use small positive values to perturb deterministic local minima."
    )

    self.topo_gmsh_quality_optimize_methods_edit = _find_or_create_line_edit(
        "topo_gmsh_quality_optimize_methods_edit", "Laplace2D,Relocate2D"
    )
    self.topo_gmsh_quality_optimize_methods_edit.setToolTip(
        "Comma-separated gmsh.model.mesh.optimize methods applied each attempt. "
        "Example: Laplace2D,Relocate2D"
    )

    self.topo_gmsh_algo_switch_on_failure_chk = _find_or_create_check(
        "topo_gmsh_algo_switch_on_failure_chk", "Gmsh algorithm switch on failure"
    )
    self.topo_gmsh_algo_switch_on_failure_chk.setChecked(False)
    self.topo_gmsh_algo_switch_on_failure_chk.setToolTip(
        "Enable Mesh.AlgorithmSwitchOnFailure. Gmsh may switch 2D algorithms (e.g. to MeshAdapt) on failure."
    )

    self.topo_gmsh_recombine_node_repositioning_chk = _find_or_create_check(
        "topo_gmsh_recombine_node_repositioning_chk", "Allow recombine node repositioning"
    )
    self.topo_gmsh_recombine_node_repositioning_chk.setChecked(True)
    self.topo_gmsh_recombine_node_repositioning_chk.setToolTip(
        "Enable node repositioning during quad recombination (Mesh.RecombineNodeRepositioning)."
    )

    self.topo_quality_strict_chk = _find_or_create_check("topo_quality_strict_chk", "Strict quality acceptance")

    _reconnect(self.topo_backend_combo.currentIndexChanged, self._update_topology_control_summary)
    _reconnect(self.topo_regions_combo.currentIndexChanged, self._update_topology_control_summary)
    _reconnect(self.topo_constraints_combo.currentIndexChanged, self._update_topology_control_summary)
    _reconnect(self.topo_quad_edges_combo.currentIndexChanged, self._update_topology_control_summary)
    _reconnect(self.topo_quality_min_angle_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_quality_max_aspect_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_quality_max_non_orth_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_quality_min_area_edit.textChanged, self._update_topology_control_summary)
    _reconnect(self.topo_quality_strict_chk.toggled, self._update_topology_control_summary)
    _reconnect(self.topo_quality_size_scales_edit.textChanged, self._update_topology_control_summary)
    _reconnect(self.topo_quality_smooth_increments_edit.textChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_quad_full_region_flow_align_chk.toggled, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_quality_recombine_topology_passes_edit.textChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_quality_recombine_min_quality_edit.textChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_quality_random_factors_edit.textChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_quality_optimize_methods_edit.textChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_algo_switch_on_failure_chk.toggled, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_recombine_node_repositioning_chk.toggled, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_global_recombine_chk.toggled, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_arc_mode_combo.currentIndexChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_arc_soft_size_factor_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_arc_soft_dist_factor_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_interface_transition_enable_chk.toggled, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_interface_transition_dist_factor_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_interface_transition_min_ratio_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_mesh_size_min_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_tolerance_edge_length_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_mesh_size_from_points_chk.toggled, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_num_threads_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_max_num_threads_2d_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_quality_enable_chk.toggled, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_quality_max_iters_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_quality_time_limit_spin.valueChanged, self._update_topology_control_summary)

    self._update_topology_control_summary()





def _build_pipe_network_config(self):
    if (
        self._mesh_data is None
        or not _HAVE_QGIS_CORE
        or PipeNetworkConfig is None
        or not hasattr(self, "drain_nodes_layer_combo")
    ):
        return None
    node_layer = self._combo_layer(self.drain_nodes_layer_combo, "vector")
    link_layer = self._combo_layer(self.drain_links_layer_combo, "vector") if hasattr(self, "drain_links_layer_combo") else None
    inlet_layer = self._combo_layer(self.drain_inlets_layer_combo, "vector") if hasattr(self, "drain_inlets_layer_combo") else None
    node_inlet_layer = self._combo_layer(self.drain_node_inlets_layer_combo, "vector") if hasattr(self, "drain_node_inlets_layer_combo") else None
    if node_layer is None or link_layer is None:
        return None

    node_fields = set(node_layer.fields().names())
    nodes: List[DrainageNode] = []
    node_by_id: Dict[str, DrainageNode] = {}
    node_cell_by_id: Dict[str, int] = {}
    node_zero_storage_by_id: Dict[str, bool] = {}
    cell_min_bed = self._mesh_cell_min_bed()

    def _opt_float(value, fallback=None):
        if value in (None, ""):
            return fallback
        try:
            return float(value)
        except Exception as e:
            self._log(f"[ERROR] opt_float conversion: {e}")
            return fallback

    def _opt_bool(value, fallback=False):
        if value in (None, ""):
            return fallback
        if value is True:
            return True
        if value is False:
            return False
        sval = str(value).strip().lower()
        if sval in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if sval in {"0", "false", "f", "no", "n", "off"}:
            return False
        try:
            return float(value) != 0.0
        except Exception as e:
            self._log(f"[ERROR] opt_bool float conversion: {e}")
            return fallback

    for ft in node_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            pt = geom.asPoint()
        except Exception as e:
            self._log(f"[ERROR] node feature geometry asPoint: {e}")
            continue
        node_id = str(ft["node_id"] if "node_id" in node_fields else ft.id()).strip()
        if not node_id:
            continue
        x = float(pt.x())
        y = float(pt.y())
        invert = _opt_float(ft["invert_elev"] if "invert_elev" in node_fields else None, 0.0)
        node_type = str(ft["node_type"] if "node_type" in node_fields else "junction").strip().lower() or "junction"
        ci = self._nearest_cell_index_for_xy(x, y)
        bed_here = float(cell_min_bed[ci]) if ci >= 0 and ci < int(cell_min_bed.size) else invert
        rim = _opt_float(ft["rim_elev"] if "rim_elev" in node_fields else None, None)
        if rim is None:
            rim = max(invert, bed_here)
        max_depth = _opt_float(ft["max_depth"] if "max_depth" in node_fields else None, None)
        if max_depth is None:
            if node_type == "outfall":
                max_depth = 10.0
            else:
                max_depth = max(0.1, float(rim) - float(invert))
        crest = _opt_float(ft["crest_elev"] if "crest_elev" in node_fields else None, None)
        if crest is None:
            crest = float(invert if node_type == "outfall" else rim)

        node = DrainageNode(
            node_id=node_id,
            x=x,
            y=y,
            invert_elev=float(invert),
            max_depth=float(max_depth),
            crest_elev=float(crest),
            rim_elev=float(rim),
            node_type=node_type,
            metadata={
                "surface_area": float(ft["surface_area"] if "surface_area" in node_fields and ft["surface_area"] not in (None, "") else 50.0),
                "outfall_area_m2": float(ft["outfall_area"] if "outfall_area" in node_fields and ft["outfall_area"] not in (None, "") else 0.0),
            },
        )
        nodes.append(node)
        node_by_id[node_id] = node
        node_cell_by_id[node_id] = int(ci)
        node_zero_storage_by_id[node_id] = _opt_bool(ft["zero_storage"] if "zero_storage" in node_fields else None, False)
    if not nodes:
        return None

    link_fields = set(link_layer.fields().names())
    links: List[DrainageLink] = []
    links_missing_capacity: List[str] = []

    def _ellipse_perimeter(a: float, b: float) -> float:
        # Ramanujan approximation; stable and cheap for geometry-derived hydraulics.
        if a <= 0.0 or b <= 0.0:
            return 0.0
        return math.pi * (3.0 * (a + b) - math.sqrt(max(0.0, (3.0 * a + b) * (a + 3.0 * b))))

    for ft in link_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        link_id = str(ft["link_id"] if "link_id" in link_fields else ft.id()).strip()
        from_node = str(ft["from_node"] if "from_node" in link_fields else "").strip()
        to_node = str(ft["to_node"] if "to_node" in link_fields else "").strip()
        if not link_id or not from_node or not to_node:
            continue

        link_shape = str(ft["link_shape"] if "link_shape" in link_fields else "").strip().lower()
        if link_shape in ("", "none", "null"):
            link_shape = "circular"

        diameter_val = None
        for nm in ("diameter", "diameter_m", "equiv_diameter", "equiv_diameter_m"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    d_try = float(ft[nm])
                    if d_try > 0.0:
                        diameter_val = d_try
                        break
                except Exception as e:
                    self._log(f"[ERROR] link diameter float conversion: {e}")

        area_val = None
        for nm in ("area_m2", "area", "cross_area"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    a_try = float(ft[nm])
                    if a_try > 0.0:
                        area_val = a_try
                        break
                except Exception as e:
                    self._log(f"[ERROR] link area float conversion: {e}")

        span_val = None
        for nm in ("span", "span_m", "width", "width_m"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    s_try = float(ft[nm])
                    if s_try > 0.0:
                        span_val = s_try
                        break
                except Exception as e:
                    self._log(f"[ERROR] link span float conversion: {e}")

        rise_val = None
        for nm in ("rise", "rise_m", "height", "height_m"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    r_try = float(ft[nm])
                    if r_try > 0.0:
                        rise_val = r_try
                        break
                except Exception as e:
                    self._log(f"[ERROR] link rise float conversion: {e}")

        equiv_d_val = None
        for nm in ("equiv_diameter_m", "equiv_diameter"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    eq_try = float(ft[nm])
                    if eq_try > 0.0:
                        equiv_d_val = eq_try
                        break
                except Exception as e:
                    self._log(f"[ERROR] link equiv_diameter float conversion: {e}")

        if (area_val is None or area_val <= 0.0):
            if link_shape == "circular" and diameter_val is not None and diameter_val > 0.0:
                area_val = 0.25 * math.pi * float(diameter_val) * float(diameter_val)
            elif link_shape in ("box", "rectangular", "rect") and span_val is not None and rise_val is not None:
                area_val = float(span_val) * float(rise_val)
            elif link_shape == "pipe_arch" and span_val is not None and rise_val is not None:
                area_val = 0.25 * math.pi * float(span_val) * float(rise_val)

        if (equiv_d_val is None or equiv_d_val <= 0.0):
            if diameter_val is not None and diameter_val > 0.0:
                equiv_d_val = float(diameter_val)
            elif area_val is not None and area_val > 0.0:
                if link_shape in ("box", "rectangular", "rect") and span_val is not None and rise_val is not None:
                    perim = 2.0 * (float(span_val) + float(rise_val))
                    if perim > 0.0:
                        equiv_d_val = 4.0 * float(area_val) / perim
                elif link_shape == "pipe_arch" and span_val is not None and rise_val is not None:
                    perim = _ellipse_perimeter(0.5 * float(span_val), 0.5 * float(rise_val))
                    if perim > 0.0:
                        equiv_d_val = 4.0 * float(area_val) / perim
                if equiv_d_val is None or equiv_d_val <= 0.0:
                    equiv_d_val = math.sqrt(4.0 * float(area_val) / math.pi)

        if (diameter_val is None or diameter_val <= 0.0) and equiv_d_val is not None and equiv_d_val > 0.0:
            diameter_val = float(equiv_d_val)

        if (diameter_val is None or diameter_val <= 0.0) and (area_val is None or area_val <= 0.0) and (equiv_d_val is None or equiv_d_val <= 0.0):
            links_missing_capacity.append(link_id)

        link_type_raw = str(ft["link_type"] if "link_type" in link_fields else "conduit").strip() or "conduit"
        is_culvert = (link_type_raw.lower() == "culvert")

        # Parse culvert-specific fields when link_type is "culvert"
        culvert_shape_val = None
        if is_culvert:
            raw_shape = str(ft.get("culvert_shape", "") or "").strip().lower()
            if raw_shape in ("", "none", "null"):
                raw_shape = "circular"
            culvert_shape_val = raw_shape

        culvert_code_val = 1
        if is_culvert and "culvert_code" in link_fields:
            try:
                culvert_code_val = int(round(float(ft["culvert_code"])))
            except Exception as e:
                self._log(f"[ERROR] culvert_code conversion: {e}")

        culvert_rise_val = None
        if is_culvert and "culvert_rise" in link_fields:
            try:
                rv = float(ft["culvert_rise"])
                if rv > 0.0:
                    culvert_rise_val = rv
            except Exception as e:
                self._log(f"[ERROR] culvert_rise float conversion: {e}")

        culvert_span_val = None
        if is_culvert and "culvert_span" in link_fields:
            try:
                sv = float(ft["culvert_span"])
                if sv > 0.0:
                    culvert_span_val = sv
            except Exception as e:
                self._log(f"[ERROR] culvert_span float conversion: {e}")

        inlet_invert_val = None
        if is_culvert and "inlet_invert_elev" in link_fields:
            try:
                iiv = float(ft["inlet_invert_elev"])
                inlet_invert_val = iiv
            except Exception as e:
                self._log(f"[ERROR] inlet_invert_elev float conversion: {e}")

        outlet_invert_val = None
        if is_culvert and "outlet_invert_elev" in link_fields:
            try:
                oiv = float(ft["outlet_invert_elev"])
                outlet_invert_val = oiv
            except Exception as e:
                self._log(f"[ERROR] outlet_invert_elev float conversion: {e}")

        entrance_loss_val = 0.5
        if is_culvert:
            for cand in ("entrance_loss_k", "inlet_loss_k", "entry_loss_k"):
                if cand in link_fields and ft[cand] not in (None, ""):
                    try:
                        entrance_loss_val = float(ft[cand])
                        break
                    except Exception as e:
                        self._log(f"[ERROR] entrance_loss_k float conversion: {e}")

        exit_loss_val = 1.0
        if is_culvert:
            for cand in ("exit_loss_k", "outlet_loss_k"):
                if cand in link_fields and ft[cand] not in (None, ""):
                    try:
                        exit_loss_val = float(ft[cand])
                        break
                    except Exception as e:
                        self._log(f"[ERROR] exit_loss_k float conversion: {e}")

        barrel_count_val = 1
        if is_culvert and "culvert_barrels" in link_fields:
            try:
                bc = int(round(float(ft["culvert_barrels"])))
                if bc >= 1:
                    barrel_count_val = bc
            except Exception as e:
                self._log(f"[ERROR] culvert_barrels int conversion: {e}")

        links.append(
            DrainageLink(
                link_id=link_id,
                from_node_id=from_node,
                to_node_id=to_node,
                link_type=link_type_raw,
                length=float(ft["length"]) if "length" in link_fields and ft["length"] not in (None, "") else float(geom.length()),
                roughness_n=float(ft["roughness_n"] if "roughness_n" in link_fields and ft["roughness_n"] not in (None, "") else 0.013),
                diameter=diameter_val,
                max_flow=float(ft["max_flow"]) if "max_flow" in link_fields and ft["max_flow"] not in (None, "") else None,
                culvert_shape=culvert_shape_val,
                culvert_code=culvert_code_val,
                culvert_rise=culvert_rise_val,
                culvert_span=culvert_span_val,
                inlet_invert_elev=inlet_invert_val,
                outlet_invert_elev=outlet_invert_val,
                entrance_loss_k=entrance_loss_val,
                exit_loss_k=exit_loss_val,
                barrel_count=barrel_count_val,
                cd=float(ft["cd"] if "cd" in link_fields and ft["cd"] not in (None, "") else 0.75),
                metadata={
                    "area_m2": float(area_val) if area_val is not None else 0.0,
                    "equiv_diameter_m": float(equiv_d_val) if equiv_d_val is not None else 0.0,
                    "cd": float(ft["cd"] if "cd" in link_fields and ft["cd"] not in (None, "") else 0.75),
                    "entry_loss_k": float(ft["entry_loss_k"] if "entry_loss_k" in link_fields and ft["entry_loss_k"] not in (None, "") else 0.5),
                    "exit_loss_k": float(ft["exit_loss_k"] if "exit_loss_k" in link_fields and ft["exit_loss_k"] not in (None, "") else 1.0),
                    "pipe_end_inlet_loss_k": float(
                        ft["pipe_end_inlet_loss_k"]
                        if "pipe_end_inlet_loss_k" in link_fields and ft["pipe_end_inlet_loss_k"] not in (None, "")
                        else (
                            ft["inlet_loss_k"]
                            if "inlet_loss_k" in link_fields and ft["inlet_loss_k"] not in (None, "")
                            else 0.5
                        )
                    ),
                    "pipe_end_outlet_loss_k": float(
                        ft["pipe_end_outlet_loss_k"]
                        if "pipe_end_outlet_loss_k" in link_fields and ft["pipe_end_outlet_loss_k"] not in (None, "")
                        else (
                            ft["outlet_loss_k"]
                            if "outlet_loss_k" in link_fields and ft["outlet_loss_k"] not in (None, "")
                            else 1.0
                        )
                    ),
                    "link_shape": link_shape,
                    "span_m": float(span_val) if span_val is not None else 0.0,
                    "rise_m": float(rise_val) if rise_val is not None else 0.0,
                    # Culvert fields also mirrored in metadata for back-compat
                    "culvert_shape": culvert_shape_val or "circular",
                    "culvert_code": float(culvert_code_val),
                    "culvert_rise": float(culvert_rise_val) if culvert_rise_val is not None else 0.0,
                    "culvert_span": float(culvert_span_val) if culvert_span_val is not None else 0.0,
                    "inlet_invert_elev": float(inlet_invert_val) if inlet_invert_val is not None else 0.0,
                    "outlet_invert_elev": float(outlet_invert_val) if outlet_invert_val is not None else 0.0,
                    "entrance_loss_k": entrance_loss_val,
                    "exit_loss_k": exit_loss_val,
                    "culvert_barrels": float(barrel_count_val),
                },
            )
        )
    if not links:
        return None

    if links_missing_capacity:
        preview = ", ".join(links_missing_capacity[:8])
        suffix = "" if len(links_missing_capacity) <= 8 else f", ... (+{len(links_missing_capacity) - 8} more)"
        self._log(
            "Drainage warning: link(s) missing hydraulic geometry (diameter/area/equiv_diameter/shape dimensions); "
            "link flow will stay zero for these IDs: "
            f"{preview}{suffix}"
        )

    inlets: List[InletExchange] = []
    inlet_types: List[InletType] = []
    node_inlets: List[NodeInletAssignment] = []
    inlet_types_by_id: Dict[str, InletType] = {}

    # New schema: tabular inlet-type catalog + node assignment table.
    if inlet_layer is not None:
        inlet_fields = set(inlet_layer.fields().names())
        has_new_inlet_schema = "inlet_type_id" in inlet_fields
        if has_new_inlet_schema:
            for ft in inlet_layer.getFeatures():
                inlet_type_id = str(ft["inlet_type_id"] if "inlet_type_id" in inlet_fields else "").strip()
                if not inlet_type_id:
                    continue
                inlet_type = InletType(
                    inlet_type_id=inlet_type_id,
                    name=str(ft["name"] if "name" in inlet_fields and ft["name"] not in (None, "") else inlet_type_id),
                    length=float(ft["weir_length"] if "weir_length" in inlet_fields and ft["weir_length"] not in (None, "") else 1.0),
                    area=float(ft["orifice_area"] if "orifice_area" in inlet_fields and ft["orifice_area"] not in (None, "") else 0.0),
                    coeff_weir=float(ft["coeff_weir"] if "coeff_weir" in inlet_fields and ft["coeff_weir"] not in (None, "") else 1.70),
                    coeff_orifice=float(ft["coeff_orifice"] if "coeff_orifice" in inlet_fields and ft["coeff_orifice"] not in (None, "") else 0.62),
                    max_capture=float(ft["max_capture"]) if "max_capture" in inlet_fields and ft["max_capture"] not in (None, "") else None,
                )
                inlet_types.append(inlet_type)
                inlet_types_by_id[inlet_type_id] = inlet_type

            if node_inlet_layer is not None:
                assign_fields = set(node_inlet_layer.fields().names())
                for ft in node_inlet_layer.getFeatures():
                    node_id = str(ft["node_id"] if "node_id" in assign_fields else "").strip()
                    inlet_type_id = str(ft["inlet_type_id"] if "inlet_type_id" in assign_fields else "").strip()
                    if not node_id or not inlet_type_id:
                        continue
                    node_inlets.append(
                        NodeInletAssignment(
                            node_id=node_id,
                            inlet_type_id=inlet_type_id,
                            multiplier=float(ft["inlet_count"] if "inlet_count" in assign_fields and ft["inlet_count"] not in (None, "") else 1.0),
                            crest_offset=float(ft["crest_offset"] if "crest_offset" in assign_fields and ft["crest_offset"] not in (None, "") else 0.0),
                        )
                    )

            for a in node_inlets:
                if a.node_id not in node_by_id:
                    continue
                it = inlet_types_by_id.get(a.inlet_type_id)
                if it is None:
                    continue
                node = node_by_id[a.node_id]
                crest = float((node.crest_elev if node.crest_elev is not None else node.invert_elev) + a.crest_offset)
                inlets.append(
                    InletExchange(
                        inlet_id=f"{a.node_id}:{a.inlet_type_id}",
                        cell_id=int(node_cell_by_id.get(a.node_id, self._nearest_cell_index_for_xy(node.x, node.y))),
                        node_id=a.node_id,
                        crest_elev=crest,
                        length=max(0.0, float(it.length)) * max(0.0, float(a.multiplier)),
                        area=max(0.0, float(it.area)) * max(0.0, float(a.multiplier)),
                        coeff_weir=max(0.0, float(it.coeff_weir)),
                        coeff_orifice=max(0.0, float(it.coeff_orifice)),
                        max_capture=it.max_capture,
                    )
                )
        else:
            # Legacy schema: spatial inlets with node and geometry.
            for ft in inlet_layer.getFeatures():
                geom = ft.geometry()
                if geom is None or geom.isEmpty():
                    continue
                try:
                    pt = geom.asPoint()
                except Exception as e:
                    self._log(f"[ERROR] inlet geometry asPoint: {e}")
                    try:
                        c = geom.centroid()
                        pt = c.asPoint() if c is not None and not c.isEmpty() else None
                    except Exception as e2:
                        self._log(f"[ERROR] inlet centroid asPoint: {e2}")
                        pt = None
                if pt is None:
                    continue
                node_id = str(ft["node_id"] if "node_id" in inlet_fields else "").strip()
                if not node_id:
                    continue
                inlets.append(
                    InletExchange(
                        inlet_id=str(ft["inlet_id"] if "inlet_id" in inlet_fields else ft.id()).strip(),
                        cell_id=self._nearest_cell_index_for_xy(float(pt.x()), float(pt.y())),
                        node_id=node_id,
                        crest_elev=float(ft["crest_elev"] if "crest_elev" in inlet_fields and ft["crest_elev"] not in (None, "") else 0.0),
                        length=float(ft["width"] if "width" in inlet_fields and ft["width"] not in (None, "") else 1.0),
                        area=float(ft["area"] if "area" in inlet_fields and ft["area"] not in (None, "") else 0.0),
                        coeff_weir=float(ft["coeff_weir"] if "coeff_weir" in inlet_fields and ft["coeff_weir"] not in (None, "") else 1.70),
                        coeff_orifice=float(ft["coefficient"] if "coefficient" in inlet_fields and ft["coefficient"] not in (None, "") else 0.62),
                        max_capture=float(ft["max_capture"]) if "max_capture" in inlet_fields and ft["max_capture"] not in (None, "") else None,
                    )
                )

    # Build outfall exchange objects for outfall-type nodes located within the mesh.
    # Prefer explicit outfall area on node features; fall back to connected-link
    # hydraulic capacity when area is not explicitly provided.
    outfalls: List[OutfallExchange] = []
    if OutfallExchange is not None:
        _node_connected_area: dict = {}
        _node_connected_diameter: dict = {}
        for lnk in links:
            area_lnk = float(lnk.metadata.get("area_m2", 0.0) or 0.0)
            d_lnk = float(lnk.diameter or 0.0)
            if d_lnk <= 0.0:
                d_lnk = float(lnk.metadata.get("equiv_diameter_m", 0.0) or 0.0)
            if d_lnk <= 0.0:
                if area_lnk > 0.0:
                    d_lnk = math.sqrt(4.0 * area_lnk / math.pi)
            if area_lnk <= 0.0 and d_lnk > 0.0:
                area_lnk = 0.25 * math.pi * d_lnk * d_lnk
            for nid in (lnk.from_node_id, lnk.to_node_id):
                cur_a = float(_node_connected_area.get(nid, 0.0))
                if area_lnk > cur_a:
                    _node_connected_area[nid] = area_lnk
                cur = float(_node_connected_diameter.get(nid, 0.0))
                if d_lnk > cur:
                    _node_connected_diameter[nid] = d_lnk

        outfalls_missing_capacity: List[str] = []
        for node in nodes:
            if str(node.node_type).strip().lower() != "outfall":
                continue
            cell_id = self._nearest_cell_index_for_xy(float(node.x), float(node.y))
            area_outfall = max(0.0, float(node.metadata.get("outfall_area_m2", 0.0) or 0.0))
            if area_outfall <= 0.0:
                area_outfall = max(0.0, float(_node_connected_area.get(node.node_id, 0.0) or 0.0))
            diameter = float(_node_connected_diameter.get(node.node_id, 0.0) or 0.0)
            if diameter <= 0.0 and area_outfall > 0.0:
                diameter = math.sqrt(4.0 * area_outfall / math.pi)
            if area_outfall <= 0.0 and diameter <= 0.0:
                outfalls_missing_capacity.append(str(node.node_id))
            outfalls.append(
                OutfallExchange(
                    outfall_id=node.node_id,
                    cell_id=cell_id,
                    node_id=node.node_id,
                    invert_elev=float(node.invert_elev),
                    area_m2=area_outfall,
                    diameter=diameter,
                    coefficient=0.82,
                    max_flow=None,
                    zero_storage=bool(node_zero_storage_by_id.get(node.node_id, False)),
                )
            )
        if outfalls_missing_capacity:
            preview = ", ".join(outfalls_missing_capacity[:8])
            suffix = "" if len(outfalls_missing_capacity) <= 8 else f", ... (+{len(outfalls_missing_capacity) - 8} more)"
            self._log(
                "Drainage warning: outfall node(s) missing outfall_area and connected link capacity; "
                f"outfall exchange will stay zero for IDs: {preview}{suffix}"
            )

    pipe_ends: List[PipeEndExchange] = []
    if PipeEndExchange is not None:
        pipe_end_link_types = {
            "pipe_end", "pipe-end", "daylighted_pipe", "daylighted", "daylight_pipe",
            "culvert",  # culvert links terminating at pipe_end nodes
        }
        pipe_end_nodes = {
            str(n.node_id) for n in nodes
            if str(n.node_type).strip().lower() == "pipe_end"
        }
        assigned_pipe_end_nodes: set = set()
        for lnk in links:
            ltype = str(lnk.link_type or "").strip().lower()
            if (
                ltype not in pipe_end_link_types
                and str(lnk.from_node_id) not in pipe_end_nodes
                and str(lnk.to_node_id) not in pipe_end_nodes
            ):
                continue

            for nid in (str(lnk.from_node_id), str(lnk.to_node_id)):
                if nid in assigned_pipe_end_nodes:
                    continue
                node = node_by_id.get(nid)
                if node is None:
                    continue
                if str(node.node_type).strip().lower() != "pipe_end":
                    continue

                cell_id = int(node_cell_by_id.get(nid, self._nearest_cell_index_for_xy(float(node.x), float(node.y))))
                diameter = float(lnk.diameter or lnk.metadata.get("equiv_diameter_m", 0.0) or 0.0)
                area_pipe = max(0.0, float(lnk.metadata.get("area_m2", 0.0) or 0.0))
                if area_pipe <= 0.0 and diameter > 0.0:
                    area_pipe = 0.25 * math.pi * diameter * diameter

                pipe_ends.append(
                    PipeEndExchange(
                        pipe_end_id=f"pipe_end:{nid}",
                        cell_id=cell_id,
                        node_id=nid,
                        invert_elev=float(node.invert_elev),
                        diameter=diameter,
                        area_m2=area_pipe,
                        coefficient=float(lnk.metadata.get("cd", 0.82) or 0.82),
                        max_flow=lnk.max_flow,
                        inlet_loss_k=float(lnk.metadata.get("pipe_end_inlet_loss_k", lnk.metadata.get("entry_loss_k", 0.5)) or 0.5),
                        outlet_loss_k=float(lnk.metadata.get("pipe_end_outlet_loss_k", lnk.metadata.get("exit_loss_k", 1.0)) or 1.0),
                    )
                )
                assigned_pipe_end_nodes.add(nid)

    gravity = float(getattr(self, "_gravity", _u.gravity()))
    solver_mode = int(self.drainage_solver_mode_combo.currentData() if hasattr(self, "drainage_solver_mode_combo") else 0)
    solver_mode_name = str(self.drainage_solver_mode_combo.currentText() if hasattr(self, "drainage_solver_mode_combo") else "EGL")
    self._log(
        f"Drainage coupling configured: nodes={len(nodes)}, links={len(links)}, "
        f"inlets={len(inlets)}, inlet_types={len(inlet_types)}, node_inlets={len(node_inlets)}, "
        f"outfalls={len(outfalls)}, pipe_ends={len(pipe_ends)}, gravity={gravity:.3f}, mode={solver_mode_name}, "
        f"substeps={int(self.drainage_coupling_substeps_spin.value()) if hasattr(self, 'drainage_coupling_substeps_spin') else 1}, "
        f"max_substeps={int(self.drainage_max_coupling_substeps_spin.value()) if hasattr(self, 'drainage_max_coupling_substeps_spin') else 64}, "
        f"gpu_method={str(self.drainage_gpu_method_combo.currentData()) if hasattr(self, 'drainage_gpu_method_combo') else 'step'}, "
        f"deadband={float(self.drainage_head_deadband_spin.value()) if hasattr(self, 'drainage_head_deadband_spin') else 1.0e-3:.4g}, "
        f"relax={float(self.drainage_dynamic_relaxation_spin.value()) if hasattr(self, 'drainage_dynamic_relaxation_spin') else 1.0:.3f}"
    )
    return PipeNetworkConfig(
        enabled=True,
        nodes=nodes,
        links=links,
        inlet_types=inlet_types,
        node_inlets=node_inlets,
        inlets=inlets,
        outfalls=outfalls,
        pipe_ends=pipe_ends,
        gravity=gravity,
        solver_mode=solver_mode,
        coupling_substeps=int(self.drainage_coupling_substeps_spin.value()) if hasattr(self, "drainage_coupling_substeps_spin") else 1,
        max_coupling_substeps=int(self.drainage_max_coupling_substeps_spin.value()) if hasattr(self, "drainage_max_coupling_substeps_spin") else 64,
        head_deadband_m=float(self.drainage_head_deadband_spin.value()) if hasattr(self, "drainage_head_deadband_spin") else 1.0e-3,
        dynamic_flow_relaxation=float(self.drainage_dynamic_relaxation_spin.value()) if hasattr(self, "drainage_dynamic_relaxation_spin") else 1.0,
        adaptive_depth_fraction=float(self.drainage_adaptive_depth_fraction_spin.value()) if hasattr(self, "drainage_adaptive_depth_fraction_spin") else 0.2,
        adaptive_wave_courant=float(self.drainage_adaptive_wave_courant_spin.value()) if hasattr(self, "drainage_adaptive_wave_courant_spin") else 0.5,
        implicit_coupling_iterations=int(self.drainage_implicit_iters_spin.value()) if hasattr(self, "drainage_implicit_iters_spin") else 2,
        implicit_coupling_relaxation=float(self.drainage_implicit_relax_spin.value()) if hasattr(self, "drainage_implicit_relax_spin") else 0.5,
    )





def _write_ugrid_nc(self, path: str, timesteps=None):
    """Write a UGRID 1.0 NetCDF4 file readable by QGIS MDAL.

    The file follows the CF-1.8 + UGRID 1.0 conventions.  QGIS MDAL's
    UGRID driver natively pairs (velocity_u, velocity_v) into an arrow
    vector dataset without requiring any naming hacks.

    Parameters
    ----------
    path : str
        Output .nc file path.
    timesteps : list of (time_seconds, h, hu, hv) or None
        When supplied, result variables are written; otherwise topology only.
    """
    if not _ensure_netcdf4_available():
        detail = ""
        if _NETCDF4_IMPORT_ERROR is not None:
            detail = f" Import error: {_NETCDF4_IMPORT_ERROR}"
        raise RuntimeError(
            "netCDF4 is unavailable (missing or binary-incompatible in current QGIS Python)."
            " Install a compatible netCDF4 build for this QGIS environment." + detail
        )
    if self._mesh_data is None:
        raise RuntimeError("No mesh data available")

    node_x = self._mesh_data["node_x"]
    node_y = self._mesh_data["node_y"]
    node_z = self._mesh_data.get("node_z", np.zeros_like(node_x))

    # Build face→node connectivity (zero-based, row per face, -1 padded)
    face_offsets = self._mesh_data.get("cell_face_offsets")
    face_nodes_arr = self._mesh_data.get("cell_face_nodes")
    cell_nodes_tri = self._mesh_data.get("cell_nodes")

    if face_offsets is not None and face_nodes_arr is not None:
        offsets = face_offsets.astype(np.int32)
        n_cells = int(offsets.size - 1)
        max_vp = int(max(offsets[i + 1] - offsets[i] for i in range(n_cells)))
        face_node = np.full((n_cells, max_vp), -1, dtype=np.int32)
        cell_cx = np.empty(n_cells, dtype=np.float64)
        cell_cy = np.empty(n_cells, dtype=np.float64)
        cell_solver_z = np.empty(n_cells, dtype=np.float64)
        for i in range(n_cells):
            s, e = int(offsets[i]), int(offsets[i + 1])
            ring = face_nodes_arr[s:e].astype(np.int32)
            face_node[i, : e - s] = ring
            cell_cx[i] = float(np.mean(node_x[ring]))
            cell_cy[i] = float(np.mean(node_y[ring]))
            cell_solver_z[i] = float(np.mean(node_z[ring]))
    else:
        tri = cell_nodes_tri.reshape(-1, 3).astype(np.int32)
        n_cells = tri.shape[0]
        max_vp = 3
        face_node = tri
        cell_cx = np.mean(node_x[tri], axis=1)
        cell_cy = np.mean(node_y[tri], axis=1)
        cell_solver_z = np.mean(node_z[tri], axis=1)

    n_nodes = int(node_x.size)

    # CRS info
    epsg_code = None
    crs_wkt = 'LOCAL_CS["Unknown"]'
    if _HAVE_QGIS_CORE:
        try:
            project_crs = QgsProject.instance().crs()
            if project_crs is not None and project_crs.isValid():
                crs_wkt = project_crs.toWkt()
                epsg_code = project_crs.postgisSrid() or None
        except Exception as e:
            self._log(f"[ERROR] CRS query for NetCDF export: {e}")

    include_extra = bool(getattr(self, "extended_outputs_chk", None) is None or self.extended_outputs_chk.isChecked())

    with _netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        # Global attributes (CF + UGRID)
        ds.Conventions = "CF-1.8 UGRID-1.0"
        ds.title = "SWE2D HYDRA model results"
        ds.institution = "qgis-hydra-plugin"
        ds.history = "Created by swe2d_workbench_qt"
        ds.featureType = "mesh2D"
        len_unit = self._length_unit_name if self._length_unit_name else "m"
        vel_unit = f"{len_unit} s-1"
        mom_unit = f"{len_unit}2 s-1"
        manning_unit = "s ft-1/3" if self._is_us_customary_units() else "s m-1/3"

        # Dimensions
        ds.createDimension("node", n_nodes)
        ds.createDimension("face", n_cells)
        ds.createDimension("max_face_nodes", max_vp)
        if timesteps:
            ds.createDimension("time", len(timesteps))

        # ---- Mesh topology container variable ----
        mesh = ds.createVariable("mesh2d", "i4")
        mesh.cf_role = "mesh_topology"
        mesh.topology_dimension = 2
        mesh.node_coordinates = "node_x node_y"
        mesh.face_node_connectivity = "face_node"
        mesh.face_coordinates = "face_x face_y"

        # Node coordinates
        nx_var = ds.createVariable("node_x", "f8", ("node",))
        nx_var.standard_name = "projection_x_coordinate"
        nx_var.units = len_unit
        nx_var.mesh = "mesh2d"
        nx_var.location = "node"
        nx_var.grid_mapping = "crs"
        nx_var[:] = node_x.astype(np.float64)

        ny_var = ds.createVariable("node_y", "f8", ("node",))
        ny_var.standard_name = "projection_y_coordinate"
        ny_var.units = len_unit
        ny_var.mesh = "mesh2d"
        ny_var.location = "node"
        ny_var.grid_mapping = "crs"
        ny_var[:] = node_y.astype(np.float64)

        nz_var = ds.createVariable("node_z", "f8", ("node",))
        nz_var.standard_name = "altitude"
        nz_var.long_name = "bed elevation at node"
        nz_var.units = len_unit
        nz_var.mesh = "mesh2d"
        nz_var.location = "node"
        nz_var.grid_mapping = "crs"
        nz_var[:] = node_z.astype(np.float64)

        # Face centroid coordinates
        fx_var = ds.createVariable("face_x", "f8", ("face",))
        fx_var.standard_name = "projection_x_coordinate"
        fx_var.units = len_unit
        fx_var.mesh = "mesh2d"
        fx_var.location = "face"
        fx_var.grid_mapping = "crs"
        fx_var[:] = cell_cx.astype(np.float64)

        fy_var = ds.createVariable("face_y", "f8", ("face",))
        fy_var.standard_name = "projection_y_coordinate"
        fy_var.units = len_unit
        fy_var.mesh = "mesh2d"
        fy_var.location = "face"
        fy_var.grid_mapping = "crs"
        fy_var[:] = cell_cy.astype(np.float64)

        # Face bed elevation consistent with solver cell_zb.
        fz_var = ds.createVariable("face_z", "f8", ("face",))
        fz_var.long_name = "face bed elevation (mean vertex bed, solver-consistent)"
        fz_var.units = len_unit
        fz_var.mesh = "mesh2d"
        fz_var.location = "face"
        fz_var.grid_mapping = "crs"
        fz_var[:] = cell_solver_z.astype(np.float64)

        # Face→node connectivity (1-indexed as UGRID standard; -1 = fill)
        fn_var = ds.createVariable(
            "face_node", "i4", ("face", "max_face_nodes"),
            fill_value=-1,
        )
        fn_var.cf_role = "face_node_connectivity"
        fn_var.long_name = "face to node connectivity"
        fn_var.start_index = 0  # zero-based
        fn_var[:] = face_node

        # CRS variable
        crs_var = ds.createVariable("crs", "i4")
        crs_var.grid_mapping_name = "unknown"
        crs_var.crs_wkt = crs_wkt
        if epsg_code:
            crs_var.epsg_code = f"EPSG:{epsg_code}"

        # ---- Time-dependent results ----
        if timesteps:
            times_s = np.array([t for t, *_ in timesteps], dtype=np.float64)

            t_var = ds.createVariable("time", "f8", ("time",))
            t_var.standard_name = "time"
            t_var.long_name = "simulation time"
            t_var.units = "seconds since 2000-01-01 00:00:00"
            t_var.calendar = "proleptic_gregorian"
            t_var[:] = times_s

            depth_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
            wse_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
            vel_u_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
            vel_v_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
            vel_mag_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
            if include_extra:
                mom_u_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                mom_v_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                qmag_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                wet_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                froude_arr = np.zeros((len(timesteps), n_cells), dtype=np.float32)
                h_min = float(self.h_min_spin.value())
                g = float(self._gravity)

            for ti, (_, h, hu, hv) in enumerate(timesteps):
                h_f = np.asarray(h, dtype=np.float64)[:n_cells]
                hu_f = np.asarray(hu, dtype=np.float64)[:n_cells]
                hv_f = np.asarray(hv, dtype=np.float64)[:n_cells]
                wet = (h_f > h_min)
                hmag = np.maximum(h_f, 1e-12)
                u = np.where(wet, hu_f / hmag, 0.0)
                v = np.where(wet, hv_f / hmag, 0.0)
                depth_arr[ti] = h_f.astype(np.float32)
                wse_arr[ti] = (h_f + cell_solver_z[:n_cells]).astype(np.float32)
                vel_u_arr[ti] = u.astype(np.float32)
                vel_v_arr[ti] = v.astype(np.float32)
                vel_mag_arr[ti] = np.sqrt(u ** 2 + v ** 2).astype(np.float32)
                if include_extra:
                    mom_u_arr[ti] = hu_f.astype(np.float32)
                    mom_v_arr[ti] = hv_f.astype(np.float32)
                    qmag_arr[ti] = np.sqrt(hu_f ** 2 + hv_f ** 2).astype(np.float32)
                    wet_arr[ti] = wet.astype(np.float32)
                    froude_arr[ti] = np.where(wet, np.sqrt(u ** 2 + v ** 2) / np.sqrt(np.maximum(g * h_f, 1.0e-12)), 0.0).astype(np.float32)

            d_var = ds.createVariable(
                "water_depth", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
            )
            d_var.standard_name = "water_depth"
            d_var.long_name = "water depth"
            d_var.units = len_unit
            d_var.mesh = "mesh2d"
            d_var.location = "face"
            d_var.coordinates = "face_x face_y"
            d_var.grid_mapping = "crs"
            d_var[:] = depth_arr

            w_var = ds.createVariable(
                "water_surface_elevation", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
            )
            w_var.standard_name = "water_surface_elevation"
            w_var.long_name = "water surface elevation"
            w_var.units = len_unit
            w_var.mesh = "mesh2d"
            w_var.location = "face"
            w_var.coordinates = "face_x face_y"
            w_var.grid_mapping = "crs"
            w_var[:] = wse_arr

            # MDAL's UGRID driver infers vectors from component wording in
            # long_name, not just standard_name, on many QGIS builds.
            u_var = ds.createVariable(
                "velocity_u", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
            )
            u_var.standard_name = "eastward_water_velocity"
            u_var.long_name = "eastward component of velocity"
            u_var.units = vel_unit
            u_var.mesh = "mesh2d"
            u_var.location = "face"
            u_var.coordinates = "face_x face_y"
            u_var.grid_mapping = "crs"
            u_var[:] = vel_u_arr

            v_var = ds.createVariable(
                "velocity_v", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
            )
            v_var.standard_name = "northward_water_velocity"
            v_var.long_name = "northward component of velocity"
            v_var.units = vel_unit
            v_var.mesh = "mesh2d"
            v_var.location = "face"
            v_var.coordinates = "face_x face_y"
            v_var.grid_mapping = "crs"
            v_var[:] = vel_v_arr

            vm_var = ds.createVariable(
                "velocity_magnitude", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
            )
            vm_var.long_name = "velocity magnitude"
            vm_var.units = vel_unit
            vm_var.mesh = "mesh2d"
            vm_var.location = "face"
            vm_var.coordinates = "face_x face_y"
            vm_var.grid_mapping = "crs"
            vm_var[:] = vel_mag_arr

            if include_extra:
                mu_var = ds.createVariable(
                    "momentum_x", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                mu_var.long_name = "x momentum per unit width"
                mu_var.units = mom_unit
                mu_var.mesh = "mesh2d"
                mu_var.location = "face"
                mu_var.coordinates = "face_x face_y"
                mu_var.grid_mapping = "crs"
                mu_var[:] = mom_u_arr

                mv_var = ds.createVariable(
                    "momentum_y", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                mv_var.long_name = "y momentum per unit width"
                mv_var.units = mom_unit
                mv_var.mesh = "mesh2d"
                mv_var.location = "face"
                mv_var.coordinates = "face_x face_y"
                mv_var.grid_mapping = "crs"
                mv_var[:] = mom_v_arr

                qmag_var = ds.createVariable(
                    "unit_discharge_magnitude", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                qmag_var.long_name = "unit discharge magnitude"
                qmag_var.units = mom_unit
                qmag_var.mesh = "mesh2d"
                qmag_var.location = "face"
                qmag_var.coordinates = "face_x face_y"
                qmag_var.grid_mapping = "crs"
                qmag_var[:] = qmag_arr

                wet_var = ds.createVariable(
                    "wet_mask", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                wet_var.long_name = "wet mask"
                wet_var.units = "1"
                wet_var.mesh = "mesh2d"
                wet_var.location = "face"
                wet_var.coordinates = "face_x face_y"
                wet_var.grid_mapping = "crs"
                wet_var[:] = wet_arr

                fr_var = ds.createVariable(
                    "froude_number", "f4", ("time", "face"), fill_value=np.float32(-9999.0)
                )
                fr_var.long_name = "Froude number"
                fr_var.units = "1"
                fr_var.mesh = "mesh2d"
                fr_var.location = "face"
                fr_var.coordinates = "face_x face_y"
                fr_var.grid_mapping = "crs"
                fr_var[:] = froude_arr

        if include_extra:
            if self._result_data is not None and "n_mann_cell" in self._result_data:
                n_face = np.asarray(self._result_data["n_mann_cell"], dtype=np.float64)[:n_cells]
            else:
                n_face = np.full(n_cells, float(self.n_mann_spin.value()), dtype=np.float64)
            n_var = ds.createVariable("manning_n_face", "f4", ("face",), fill_value=np.float32(-9999.0))
            n_var.long_name = "Manning roughness at face"
            n_var.units = manning_unit
            n_var.mesh = "mesh2d"
            n_var.location = "face"
            n_var.coordinates = "face_x face_y"
            n_var.grid_mapping = "crs"
            n_var[:] = n_face.astype(np.float32)





def _write_hecras_hdf5(self, path: str, timesteps=None):
    """Write a HEC-RAS 2D compatible HDF5 file readable by QGIS MDAL.

    Parameters
    ----------
    path : str
        Output .h5 file path.
    timesteps : list of (time_seconds, h, hu, hv) or None
        When supplied, results datasets are written; otherwise geometry only.
    """
    if not _HAVE_H5PY:
        raise RuntimeError("h5py is not installed.  Run: pip install h5py")
    if self._mesh_data is None:
        raise RuntimeError("No mesh data available")

    node_x = self._mesh_data["node_x"]
    node_y = self._mesh_data["node_y"]
    node_z = self._mesh_data.get("node_z", np.zeros_like(node_x))

    # Build dense cell-vertex index array (HEC-RAS FacePoint Indexes,
    # -1 padded to maximum ring length).
    face_offsets = self._mesh_data.get("cell_face_offsets")
    face_nodes_arr = self._mesh_data.get("cell_face_nodes")
    cell_nodes_tri = self._mesh_data.get("cell_nodes")

    if face_offsets is not None and face_nodes_arr is not None:
        offsets = face_offsets.astype(np.int32)
        n_cells = int(offsets.size - 1)
        max_vp = int(max(offsets[i + 1] - offsets[i] for i in range(n_cells)))
        fp_idx = np.full((n_cells, max_vp), -1, dtype=np.int32)
        cell_cx = np.empty(n_cells, dtype=np.float64)
        cell_cy = np.empty(n_cells, dtype=np.float64)
        cell_solver_z = np.empty(n_cells, dtype=np.float64)
        for i in range(n_cells):
            s, e = int(offsets[i]), int(offsets[i + 1])
            ring = face_nodes_arr[s:e].astype(np.int32)
            fp_idx[i, : e - s] = ring
            cell_cx[i] = float(np.mean(node_x[ring]))
            cell_cy[i] = float(np.mean(node_y[ring]))
            cell_solver_z[i] = float(np.mean(node_z[ring]))
    else:
        tri = cell_nodes_tri.reshape(-1, 3).astype(np.int32)
        n_cells = tri.shape[0]
        fp_idx = tri
        cell_cx = np.mean(node_x[tri], axis=1)
        cell_cy = np.mean(node_y[tri], axis=1)
        cell_solver_z = np.mean(node_z[tri], axis=1)

    area_name = "Perimeter 1"

    include_extra = bool(getattr(self, "extended_outputs_chk", None) is None or self.extended_outputs_chk.isChecked())

    with _h5py.File(path, "w") as f:
        f.attrs["File Type"] = np.bytes_(b"HEC-RAS Results")
        f.attrs["File Version"] = np.bytes_(b"HEC-RAS 7.0 April 2026")
        f.attrs["Units System"] = np.bytes_(
            b"US Customary" if self._is_us_customary_units() else b"SI"
        )
        projection_wkt = 'LOCAL_CS["Unknown"]'
        if _HAVE_QGIS_CORE:
            try:
                project_crs = QgsProject.instance().crs()
                if project_crs is not None and project_crs.isValid():
                    projection_wkt = project_crs.toWkt()
            except Exception as e:
                self._log(f"[ERROR] CRS query for HDF5 export: {e}")
        f.attrs["Projection"] = np.bytes_(projection_wkt.encode("utf-8"))

        # ---- Geometry ----
        geo = f.require_group("Geometry")
        geo.attrs["Complete Geometry"] = np.bytes_(b"True")
        geo.attrs["SI Units"] = np.bytes_(b"False" if self._is_us_customary_units() else b"True")
        geo.attrs["Title"] = np.bytes_(b"Generated Geometry")
        geo.attrs["Version"] = np.bytes_(b"1.0")
        flow_areas_grp = geo.require_group("2D Flow Areas")

        # MDAL's HEC-RAS driver discovers 2D flow areas from
        # Geometry/2D Flow Areas/Attributes and expects the HEC-RAS 5.0.5+
        # field names, not the ad hoc top-level dataset used initially.
        attrs_dt = np.dtype(
            [
                ("Name", "S16"),
                ("Locked", np.uint8),
                ("Mann", np.float32),
                ("Multiple Face Mann n", np.uint8),
                ("Composite LC", np.uint8),
                ("Cell Vol Tol", np.float32),
                ("Cell Min Area Fraction", np.float32),
                ("Face Profile Tol", np.float32),
                ("Face Area Tol", np.float32),
                ("Face Conv Ratio", np.float32),
                ("Laminar Depth", np.float32),
                ("Min Face Length Ratio", np.float32),
                ("Spacing dx", np.float32),
                ("Spacing dy", np.float32),
                ("Shift dx", np.float32),
                ("Shift dy", np.float32),
                ("Cell Count", np.int32),
            ]
        )
        flow_areas_grp.create_dataset(
            "Attributes",
            data=np.array(
                [
                    (
                        area_name.encode(),
                        0,
                        np.float32(0.03),
                        0,
                        0,
                        np.float32(0.01),
                        np.float32(0.01),
                        np.float32(0.01),
                        np.float32(0.01),
                        np.float32(0.02),
                        np.float32(0.2),
                        np.float32(0.05),
                        np.float32(1.0),
                        np.float32(1.0),
                        np.float32(np.nan),
                        np.float32(np.nan),
                        n_cells,
                    )
                ],
                dtype=attrs_dt,
            ),
        )

        area_grp = flow_areas_grp.require_group(area_name)

        # Vertices ("FacePoints" in HEC-RAS 2D parlance)
        area_grp.create_dataset(
            "FacePoints Coordinate",
            data=np.column_stack([node_x, node_y]).astype(np.float64),
        )
        # Cell centroids
        area_grp.create_dataset(
            "Cells Center Coordinate",
            data=np.column_stack([cell_cx, cell_cy]).astype(np.float64),
        )
        # Solver-consistent bed elevation per cell.
        area_grp.create_dataset(
            "Cells Minimum Elevation",
            data=cell_solver_z.astype(np.float32),
        )
        if include_extra:
            if self._result_data is not None and "n_mann_cell" in self._result_data:
                n_face = np.asarray(self._result_data["n_mann_cell"], dtype=np.float64)[:n_cells]
            else:
                n_face = np.full(n_cells, float(self.n_mann_spin.value()), dtype=np.float64)
            area_grp.create_dataset("Cells Manning n", data=n_face.astype(np.float32))
        # Connectivity: nCells × maxVerts, -1 padded
        area_grp.create_dataset("Cells FacePoint Indexes", data=fp_idx)

        # ---- Results ----
        if timesteps:
            n_t = len(timesteps)
            times_hr = np.array([t / 3600.0 for t, *_ in timesteps], dtype=np.float32)

            ts_base = (
                "Results/Unsteady/Output/Output Blocks/"
                "Base Output/Unsteady Time Series"
            )
            ds_time = f.create_dataset(f"{ts_base}/Time", data=times_hr)
            ds_time.attrs["Number of actual Time Steps"] = np.array([n_t], dtype=np.int32)
            ds_time.attrs["Time"] = np.bytes_(b"Hours")

            # String time stamps (ddMONyyyy HH:MM:SS) — used by some MDAL versions
            stamps = []
            for t_s, *_ in timesteps:
                total_min = int(t_s / 60)
                hh, mm = divmod(total_min, 60)
                stamps.append(f"01JAN2000 {hh:02d}:{mm:02d}:00".encode())
            f.create_dataset(
                f"{ts_base}/Time Date Stamp",
                data=np.array(stamps, dtype="S26"),
            )

            depth_arr = np.zeros((n_t, n_cells), dtype=np.float32)
            wse_arr = np.zeros((n_t, n_cells), dtype=np.float32)
            vel_arr = np.zeros((n_t, n_cells), dtype=np.float32)
            vel_u_arr = np.zeros((n_t, n_cells), dtype=np.float32)
            vel_v_arr = np.zeros((n_t, n_cells), dtype=np.float32)
            if include_extra:
                mom_u_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                mom_v_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                qmag_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                wet_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                froude_arr = np.zeros((n_t, n_cells), dtype=np.float32)
                h_min = float(self.h_min_spin.value())
                g = float(self._gravity)

            for ti, (_, h, hu, hv) in enumerate(timesteps):
                h_f = np.asarray(h, dtype=np.float64)[:n_cells]
                hu_f = np.asarray(hu, dtype=np.float64)[:n_cells]
                hv_f = np.asarray(hv, dtype=np.float64)[:n_cells]
                wet = (h_f > h_min)
                hmag = np.maximum(h_f, 1e-12)
                u = np.where(wet, hu_f / hmag, 0.0)
                v = np.where(wet, hv_f / hmag, 0.0)
                depth_arr[ti] = h_f.astype(np.float32)
                wse_arr[ti] = (h_f + cell_solver_z[:n_cells]).astype(np.float32)
                vel_arr[ti] = np.sqrt(u ** 2 + v ** 2).astype(np.float32)
                vel_u_arr[ti] = u.astype(np.float32)
                vel_v_arr[ti] = v.astype(np.float32)
                if include_extra:
                    mom_u_arr[ti] = hu_f.astype(np.float32)
                    mom_v_arr[ti] = hv_f.astype(np.float32)
                    qmag_arr[ti] = np.sqrt(hu_f ** 2 + hv_f ** 2).astype(np.float32)
                    wet_arr[ti] = wet.astype(np.float32)
                    froude_arr[ti] = np.where(wet, np.sqrt(u ** 2 + v ** 2) / np.sqrt(np.maximum(g * h_f, 1.0e-12)), 0.0).astype(np.float32)

            ar = f.require_group(f"{ts_base}/2D Flow Areas/{area_name}")
            ar.create_dataset("Depth", data=depth_arr)
            ar.create_dataset("Water Surface", data=wse_arr)
            ar.create_dataset("Cell Velocity - Magnitude", data=vel_arr)
            ar.create_dataset("Cell Velocity - X", data=vel_u_arr)
            ar.create_dataset("Cell Velocity - Y", data=vel_v_arr)
            # Alias names improve vector pairing across MDAL/QGIS versions.
            ar.create_dataset("Cell Velocity X", data=vel_u_arr)
            ar.create_dataset("Cell Velocity Y", data=vel_v_arr)
            ar.create_dataset("Velocity X", data=vel_u_arr)
            ar.create_dataset("Velocity Y", data=vel_v_arr)
            if include_extra:
                ar.create_dataset("Cell Momentum - X", data=mom_u_arr)
                ar.create_dataset("Cell Momentum - Y", data=mom_v_arr)
                ar.create_dataset("Unit Discharge - Magnitude", data=qmag_arr)
                ar.create_dataset("Wet Mask", data=wet_arr)
                ar.create_dataset("Cell Froude Number", data=froude_arr)

            # MDAL's HEC-RAS reader expects Summary Output to exist when a
            # Results tree is present, even if most summary datasets are not.
            f.require_group(
                "Results/Unsteady/Output/Output Blocks/"
                f"Base Output/Summary Output/2D Flow Areas/{area_name}"
            )





def _mesh_cell_centers_for_gpkg(
    self,
    gpkg_path: str,
    run_id: str = "",
    table_name: str = "swe2d_mesh_results",
) -> Tuple[Dict[int, Tuple[float, float]], float]:
    gpkg_path = str(gpkg_path or "").strip()
    run_id = str(run_id or "").strip()
    table_name = str(table_name or "swe2d_mesh_results").strip() or "swe2d_mesh_results"
    cache_key = f"{gpkg_path}|{table_name}|{run_id}"
    if cache_key in self._velocity_cell_xy_cache:
        return (
            self._velocity_cell_xy_cache.get(cache_key, {}),
            float(self._velocity_base_len_cache.get(cache_key, 1.0)),
        )

    cell_xy: Dict[int, Tuple[float, float]] = {}
    base_len = 1.0
    mesh_layer_name = ""

    def _quote_ident(name: str) -> str:
        return '"' + str(name or "").replace('"', '""') + '"'

    expected_n_cells = 0
    candidate_layers: List[str] = [
        "swe2d_mesh_cells",
        "SWE2D_Mesh_Cells",
        "SWE2D_Mesh_Cells refined 2",
        "struct_SWE2D_Mesh_Cells",
        "smol_SWE2D_Mesh_Cells",
        "GMSH_SWE2D_Mesh_Cells",
    ]

    if gpkg_path and os.path.exists(gpkg_path):
        try:
            conn = sqlite3.connect(gpkg_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND lower(name) LIKE '%mesh_cells%'"
            )
            for (nm,) in cur.fetchall():
                nm = str(nm or "").strip()
                if nm and nm not in candidate_layers:
                    candidate_layers.append(nm)

            if run_id:
                cur.execute(
                    f"SELECT COUNT(DISTINCT cell_id) FROM {_quote_ident(table_name)} WHERE run_id = ?",
                    (run_id,),
                )
                row = cur.fetchone()
                expected_n_cells = int(row[0]) if row and row[0] is not None else 0

            best_layer = ""
            best_score = None
            for lname in candidate_layers:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {_quote_ident(lname)}")
                    row = cur.fetchone()
                    n_cells = int(row[0]) if row and row[0] is not None else 0
                except Exception as e:
                    self._log(f"[ERROR] SQL count for layer {lname}: {e}")
                    continue
                if n_cells <= 0:
                    continue
                if expected_n_cells > 0:
                    score = abs(n_cells - expected_n_cells)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_layer = lname
                        if score == 0:
                            break
                elif not best_layer:
                    best_layer = lname
            mesh_layer_name = best_layer
        except Exception as e:
            self._log(f"[ERROR] velocity overlay GPKG scan: {e}")
            mesh_layer_name = ""
        finally:
            try:
                conn.close()
            except Exception as e:
                self._log(f"[ERROR] velocity overlay GPKG close: {e}")

    if _HAVE_QGIS_CORE and QgsVectorLayer is not None and gpkg_path and os.path.exists(gpkg_path):
        for lname in ([mesh_layer_name] if mesh_layer_name else []) + ["swe2d_mesh_cells", "SWE2D_Mesh_Cells"]:
            try:
                lyr = QgsVectorLayer(f"{gpkg_path}|layername={lname}", lname, "ogr")
                if lyr is None or not lyr.isValid():
                    continue
                if lyr.fields().indexFromName("cell_id") < 0:
                    continue

                areas = []
                for ft in lyr.getFeatures():
                    try:
                        cid = int(ft["cell_id"])
                        geom = ft.geometry()
                        if geom is None or geom.isEmpty():
                            continue
                        cgeom = geom.centroid()
                        if cgeom is None or cgeom.isEmpty():
                            continue
                        pt = cgeom.asPoint()
                        cell_xy[cid] = (float(pt.x()), float(pt.y()))
                        try:
                            a = float(geom.area())
                            if a > 0.0:
                                areas.append(a)
                        except Exception as e:
                            self._log(f"[ERROR] cell area float conversion: {e}")
                    except Exception as e:
                        self._log(f"[ERROR] cell feature iteration: {e}")
                        continue

                if cell_xy:
                    if areas:
                        base_len = max(0.05, float(np.sqrt(max(float(np.nanmean(np.asarray(areas))), 1.0e-9))))
                    if expected_n_cells > 0 and abs(int(len(cell_xy)) - int(expected_n_cells)) > 0:
                        self._log(
                            "Velocity overlay warning: selected mesh layer does not exactly match run cell count "
                            f"(run_id={run_id}, table={table_name}, expected={expected_n_cells}, got={len(cell_xy)}, layer={lname})."
                        )
                    break
            except Exception as e:
                self._log(f"[ERROR] velocity overlay layer iteration: {e}")
                continue

    # Fallback for current active in-memory mesh if mesh layer was unavailable.
    if not cell_xy and self._mesh_data is not None:
        try:
            cx, cy = self._mesh_cell_centroids()
            n_cells = min(int(cx.size), int(cy.size))
            cell_xy = {i: (float(cx[i]), float(cy[i])) for i in range(n_cells)}
            area = np.asarray(self._mesh_cell_areas(), dtype=np.float64)
            base_len = max(0.05, float(np.sqrt(max(float(np.nanmean(area)), 1.0e-9))))
        except Exception as e:
            self._log(f"[ERROR] velocity overlay fallback centroids: {e}")
            cell_xy = {}
            base_len = 1.0

    # Handle common 1-based cell_id schemas by also exposing shifted keys.
    if cell_xy and 0 not in cell_xy and 1 in cell_xy:
        shifted = {}
        for cid, xy in cell_xy.items():
            if cid > 0:
                shifted[cid - 1] = xy
        cell_xy.update(shifted)

    self._velocity_cell_xy_cache[cache_key] = cell_xy
    self._velocity_base_len_cache[cache_key] = float(base_len)
    return cell_xy, float(base_len)





def _persist_line_results_to_geopackage(
    self,
    gpkg_path: str,
    run_id: str,
    rows: List[Dict[str, object]],
    mesh_interval_s: float,
    line_interval_s: float,
    profile_rows: Optional[List[Dict[str, object]]] = None,
) -> None:
    if not gpkg_path or not rows:
        return
    profile_rows = list(profile_rows or [])
    runs_table = "swe2d_line_results_runs"
    ts_table = "swe2d_line_results_ts"
    profile_table = "swe2d_line_results_profile"
    if hasattr(self, "_results_table_name"):
        try:
            runs_table = str(self._results_table_name(runs_table) or runs_table)
            ts_table = str(self._results_table_name(ts_table) or ts_table)
            profile_table = str(self._results_table_name(profile_table) or profile_table)
        except Exception as e:
            self._log(f"[ERROR] results table name resolution: {e}")
            runs_table = "swe2d_line_results_runs"
            ts_table = "swe2d_line_results_ts"
            profile_table = "swe2d_line_results_profile"

    def _q(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    q_runs = _q(runs_table)
    q_ts = _q(ts_table)
    q_profile = _q(profile_table)

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {q_runs} (
                run_id TEXT PRIMARY KEY,
                created_utc TEXT,
                mesh_interval_s REAL,
                line_interval_s REAL,
                row_count INTEGER
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {q_ts} (
                run_id TEXT,
                t_s REAL,
                line_id INTEGER,
                line_name TEXT,
                depth_m REAL,
                velocity_ms REAL,
                wse_m REAL,
                bed_m REAL,
                flow_cms REAL,
                wet_frac REAL,
                fr REAL,
                PRIMARY KEY (run_id, t_s, line_id)
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{ts_table}_run_line_t ON {q_ts}(run_id, line_id, t_s)"
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {q_profile} (
                run_id TEXT,
                t_s REAL,
                line_id INTEGER,
                line_name TEXT,
                station_m REAL,
                depth_m REAL,
                velocity_ms REAL,
                wse_m REAL,
                bed_m REAL,
                flow_qn REAL,
                wet INTEGER,
                fr REAL,
                PRIMARY KEY (run_id, t_s, line_id, station_m)
            )
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{profile_table}_run_line_t_s ON {q_profile}(run_id, line_id, t_s, station_m)"
        )
        cur.execute(f"DELETE FROM {q_ts} WHERE run_id = ?", (run_id,))
        cur.execute(f"DELETE FROM {q_profile} WHERE run_id = ?", (run_id,))
        cur.execute(
            f"""
            INSERT OR REPLACE INTO {q_runs}
            (run_id, created_utc, mesh_interval_s, line_interval_s, row_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
                float(mesh_interval_s),
                float(line_interval_s),
                int(len(rows)),
            ),
        )
        batch = [
            (
                str(run_id),
                float(r.get("t_s", 0.0)),
                int(r.get("line_id", -1)),
                str(r.get("line_name", "") or ""),
                float(r.get("depth_m", float("nan"))),
                float(r.get("velocity_ms", float("nan"))),
                float(r.get("wse_m", float("nan"))),
                float(r.get("bed_m", float("nan"))),
                float(r.get("flow_cms", float("nan"))),
                float(r.get("wet_frac", float("nan"))),
                float(r.get("fr", float("nan"))),
            )
            for r in rows
        ]
        cur.executemany(
            f"""
            INSERT OR REPLACE INTO {q_ts}
            (run_id, t_s, line_id, line_name, depth_m, velocity_ms, wse_m, bed_m, flow_cms, wet_frac, fr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        if profile_rows:
            prof_batch = [
                (
                    str(run_id),
                    float(r.get("t_s", 0.0)),
                    int(r.get("line_id", -1)),
                    str(r.get("line_name", "") or ""),
                    float(r.get("station_m", 0.0)),
                    float(r.get("depth_m", float("nan"))),
                    float(r.get("velocity_ms", float("nan"))),
                    float(r.get("wse_m", float("nan"))),
                    float(r.get("bed_m", float("nan"))),
                    float(r.get("flow_qn", float("nan"))),
                    int(r.get("wet", 0)),
                    float(r.get("fr", float("nan"))),
                )
                for r in profile_rows
            ]
            cur.executemany(
                f"""
                INSERT OR REPLACE INTO {q_profile}
                (run_id, t_s, line_id, line_name, station_m, depth_m, velocity_ms, wse_m, bed_m, flow_qn, wet, fr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prof_batch,
            )
        conn.commit()
        self._line_results_latest_run_id = str(run_id)
        self._line_results_latest_db_path = str(gpkg_path)
        self._log(
            f"Stored sample line results in GeoPackage: {gpkg_path} "
            f"(run_id={run_id}, ts_rows={len(rows)}, profile_rows={len(profile_rows)})"
        )
    finally:
        conn.close()





def _create_2d_model_geopackage(self):
    if not _HAVE_QGIS_CORE:
        self._log("QGIS layer API unavailable; cannot create model GeoPackage.")
        return

    out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
        self,
        "Create 2D Model GeoPackage",
        "swe2d_model.gpkg",
        "GeoPackage (*.gpkg)",
    )
    if not out_path:
        return
    if not out_path.lower().endswith(".gpkg"):
        out_path += ".gpkg"

    crs_auth = "EPSG:4326"
    try:
        crs = QgsProject.instance().crs()
        if crs is not None and crs.isValid():
            crs_auth = crs.authid() or crs_auth
    except Exception as e:
        self._log(f"[ERROR] CRS auth for GPKG export: {e}")

    nodes = QgsVectorLayer(f"Point?crs={crs_auth}&field=node_id:integer", "swe2d_topo_nodes", "memory")
    arcs = QgsVectorLayer(
        f"LineString?crs={crs_auth}&field=arc_id:integer&field=node0:integer&field=node1:integer"
        "&field=use_global_arc_ctrl:integer&field=arc_mode_override:string(24)"
        "&field=arc_soft_size_override:double&field=arc_soft_dist_override:double",
        "swe2d_topo_arcs",
        "memory",
    )
    regions = QgsVectorLayer(
        f"Polygon?crs={crs_auth}&field=region_id:integer&field=target_size:double&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double&field=edge_len_3:double&field=edge_len_4:double",
        "swe2d_topo_regions",
        "memory",
    )
    constraints = QgsVectorLayer(
        f"Polygon?crs={crs_auth}&field=constraint_id:integer&field=target_size:double&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double&field=edge_len_3:double&field=edge_len_4:double",
        "swe2d_topo_constraints",
        "memory",
    )
    quad_edges = QgsVectorLayer(
        f"LineString?crs={crs_auth}&field=region_id:integer&field=edge_id:integer&field=target_size:double&field=n_layers:integer&field=first_height:double&field=growth_rate:double",
        "swe2d_topo_quad_edges",
        "memory",
    )
    manning = QgsVectorLayer(
        f"Polygon?crs={crs_auth}&field=zone_id:integer&field=n_mann:double&field=priority:integer",
        "swe2d_manning_zones",
        "memory",
    )
    bc_lines = QgsVectorLayer(
        f"LineString?crs={crs_auth}&field=bc_type:integer&field=bc_value:double&field=priority:integer&field=hydrograph:string(1024)&field=hydrograph_id:string(64)&field=hydrograph_layer:string(128)",
        "swe2d_bc_lines",
        "memory",
    )
    sample_lines = QgsVectorLayer(
        f"LineString?crs={crs_auth}&field=line_id:integer&field=name:string(128)&field=enabled:integer&field=priority:integer",
        "swe2d_sample_lines",
        "memory",
    )
    rain_gages = QgsVectorLayer(
        f"Point?crs={crs_auth}&field=gage_id:string(64)&field=name:string(128)&field=hyetograph_id:string(64)&field=units:string(32)&field=priority:integer",
        "swe2d_rain_gages",
        "memory",
    )
    storm_areas = QgsVectorLayer(
        f"Polygon?crs={crs_auth}&field=storm_id:integer&field=name:string(128)&field=priority:integer",
        "swe2d_storm_areas",
        "memory",
    )
    cn_zones = QgsVectorLayer(
        f"Polygon?crs={crs_auth}&field=zone_id:integer&field=cn:double&field=priority:integer",
        "swe2d_cn_zones",
        "memory",
    )
    hyetographs = QgsVectorLayer(
        "None?field=hyetograph_id:string(64)&field=Time:string(32)&field=Value:double&field=value_type:string(24)&field=units:string(24)&field=description:string(256)",
        "swe2d_hyetographs",
        "memory",
    )
    hydro = QgsVectorLayer(
        "None?field=hydrograph_id:string(64)&field=bc_type:integer&field=Time:string(32)&field=Value:double&field=description:string(256)",
        "swe2d_hydrographs",
        "memory",
    )
    drainage_nodes = QgsVectorLayer(
        f"Point?crs={crs_auth}&field=node_id:string(64)&field=invert_elev:double&field=max_depth:double&field=rim_elev:double&field=crest_elev:double&field=node_type:string(32)&field=surface_area:double&field=outfall_area:double&field=zero_storage:integer",
        "swe2d_drainage_nodes",
        "memory",
    )
    drainage_links = QgsVectorLayer(
        f"LineString?crs={crs_auth}&field=link_id:string(64)&field=from_node:string(64)&field=to_node:string(64)&field=link_type:string(32)&field=link_shape:string(32)&field=length:double&field=roughness_n:double&field=diameter:double&field=span:double&field=rise:double&field=area_m2:double&field=equiv_diameter_m:double&field=max_flow:double&field=cd:double",
        "swe2d_drainage_links",
        "memory",
    )
    drainage_inlets = QgsVectorLayer(
        "None?field=inlet_type_id:string(64)&field=name:string(128)&field=weir_length:double&field=orifice_area:double&field=coeff_weir:double&field=coeff_orifice:double&field=max_capture:double&field=description:string(256)",
        "swe2d_drainage_inlets",
        "memory",
    )
    drainage_node_inlets = QgsVectorLayer(
        "None?field=node_id:string(64)&field=inlet_type_id:string(64)&field=inlet_count:double&field=crest_offset:double&field=description:string(256)",
        "swe2d_drainage_node_inlets",
        "memory",
    )
    structures = QgsVectorLayer(
        f"LineString?crs={crs_auth}&field=structure_id:string(64)&field=structure_type:integer&field=crest_elev:double&field=enabled:integer&field=width:double&field=height:double&field=diameter:double&field=culvert_shape:string(32)&field=culvert_code:integer&field=culvert_rise:double&field=culvert_span:double&field=culvert_area_m2:double&field=culvert_barrels:integer&field=culvert_slope:double&field=inlet_invert_elev:double&field=outlet_invert_elev:double&field=entrance_loss_k:double&field=exit_loss_k:double&field=embankment_enabled:integer&field=embankment_crest_elev:double&field=embankment_overflow_width:double&field=embankment_weir_coeff:double&field=length:double&field=roughness_n:double&field=coeff:double&field=cd:double&field=opening:double&field=q_pump:double&field=max_flow:double&field=inlet_loss_k:double&field=outlet_loss_k:double&field=stacked_enabled:integer&field=influence_width_m:double&field=upstream_buffer_m:double&field=downstream_buffer_m:double&field=deck_soffit_elev:double&field=deck_top_elev:double&field=model_top_elev:double&field=under_layers:integer&field=over_layers:integer&field=pier_count:integer&field=pier_width:double",
        "swe2d_structures",
        "memory",
    )

    model_layers = [
        nodes,
        arcs,
        regions,
        constraints,
        quad_edges,
        manning,
        bc_lines,
        sample_lines,
        rain_gages,
        storm_areas,
        cn_zones,
        hyetographs,
        hydro,
        drainage_nodes,
        drainage_links,
        drainage_inlets,
        drainage_node_inlets,
        structures,
    ]
    for lyr in model_layers:
        self._configure_swe2d_layer_editors(lyr)

    # Persist as a single GeoPackage file.
    for i, lyr in enumerate(model_layers):
        self._write_memory_layer_to_gpkg(lyr, out_path, lyr.name(), create_file=(i == 0))
    self._persist_model_layer_bindings(out_path)

    self._log(f"Created 2D model GeoPackage: {out_path}")
    self.layer_status_lbl.setText("2D model GeoPackage created.")
    self._load_2d_model_geopackage(path_override=out_path)





def _migrate_2d_model_geopackage(self):
    """Add missing layers and columns to an existing 2D model GeoPackage."""
    if not _HAVE_QGIS_CORE:
        self._log("QGIS layer API unavailable; cannot migrate GeoPackage.")
        return

    gpkg_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Select 2D Model GeoPackage to Update",
        "",
        "GeoPackage (*.gpkg)",
    )
    if not gpkg_path:
        return

    crs_auth = "EPSG:4326"
    try:
        crs = QgsProject.instance().crs()
        if crs is not None and crs.isValid():
            crs_auth = crs.authid() or crs_auth
    except Exception as e:
        self._log(f"[ERROR] CRS auth for GPKG migration: {e}")

    # Canonical schema: list of (layer_name, memory_uri) pairs.
    # Geometry-less tables use "None?" as the URI prefix.
    layer_specs = [
        ("swe2d_topo_nodes",
         f"Point?crs={crs_auth}&field=node_id:integer"),
        ("swe2d_topo_arcs",
         f"LineString?crs={crs_auth}&field=arc_id:integer&field=node0:integer&field=node1:integer"
         "&field=use_global_arc_ctrl:integer&field=arc_mode_override:string(24)"
         "&field=arc_soft_size_override:double&field=arc_soft_dist_override:double"),
        ("swe2d_topo_regions",
         f"Polygon?crs={crs_auth}&field=region_id:integer&field=target_size:double"
         "&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double"
         "&field=edge_len_3:double&field=edge_len_4:double"),
        ("swe2d_topo_constraints",
         f"Polygon?crs={crs_auth}&field=constraint_id:integer&field=target_size:double"
         "&field=cell_type:string(32)&field=edge_len_1:double&field=edge_len_2:double"
         "&field=edge_len_3:double&field=edge_len_4:double"),
        ("swe2d_topo_quad_edges",
         f"LineString?crs={crs_auth}&field=region_id:integer&field=edge_id:integer"
         "&field=target_size:double&field=n_layers:integer&field=first_height:double"
         "&field=growth_rate:double"),
        ("swe2d_manning_zones",
         f"Polygon?crs={crs_auth}&field=zone_id:integer&field=n_mann:double&field=priority:integer"),
        ("swe2d_bc_lines",
         f"LineString?crs={crs_auth}&field=bc_type:integer&field=bc_value:double"
         "&field=priority:integer&field=hydrograph:string(1024)"
         "&field=hydrograph_id:string(64)&field=hydrograph_layer:string(128)"),
        ("swe2d_sample_lines",
         f"LineString?crs={crs_auth}&field=line_id:integer&field=name:string(128)"
         "&field=enabled:integer&field=priority:integer"),
        ("swe2d_rain_gages",
         f"Point?crs={crs_auth}&field=gage_id:string(64)&field=name:string(128)"
         "&field=hyetograph_id:string(64)&field=units:string(32)&field=priority:integer"),
        ("swe2d_storm_areas",
         f"Polygon?crs={crs_auth}&field=storm_id:integer&field=name:string(128)&field=priority:integer"),
        ("swe2d_cn_zones",
         f"Polygon?crs={crs_auth}&field=zone_id:integer&field=cn:double&field=priority:integer"),
        ("swe2d_hyetographs",
         "None?field=hyetograph_id:string(64)&field=Time:string(32)&field=Value:double"
         "&field=value_type:string(24)&field=units:string(24)&field=description:string(256)"),
        ("swe2d_hydrographs",
         "None?field=hydrograph_id:string(64)&field=bc_type:integer&field=Time:string(32)"
         "&field=Value:double&field=description:string(256)"),
        ("swe2d_drainage_nodes",
         f"Point?crs={crs_auth}&field=node_id:string(64)&field=invert_elev:double"
         "&field=max_depth:double&field=rim_elev:double&field=crest_elev:double"
         "&field=node_type:string(32)&field=surface_area:double"
         "&field=outfall_area:double&field=zero_storage:integer"),
        ("swe2d_drainage_links",
         f"LineString?crs={crs_auth}&field=link_id:string(64)&field=from_node:string(64)"
         "&field=to_node:string(64)&field=link_type:string(32)&field=link_shape:string(32)"
         "&field=length:double&field=roughness_n:double&field=diameter:double"
         "&field=span:double&field=rise:double&field=area_m2:double"
         "&field=equiv_diameter_m:double&field=max_flow:double&field=cd:double"),
        ("swe2d_drainage_inlets",
         "None?field=inlet_type_id:string(64)&field=name:string(128)"
         "&field=weir_length:double&field=orifice_area:double"
         "&field=coeff_weir:double&field=coeff_orifice:double"
         "&field=max_capture:double&field=description:string(256)"),
        ("swe2d_drainage_node_inlets",
         "None?field=node_id:string(64)&field=inlet_type_id:string(64)"
         "&field=inlet_count:double&field=crest_offset:double&field=description:string(256)"),
        ("swe2d_structures",
         f"LineString?crs={crs_auth}&field=structure_id:string(64)"
         "&field=structure_type:integer&field=crest_elev:double&field=enabled:integer"
            "&field=width:double&field=height:double&field=diameter:double"
            "&field=culvert_shape:string(32)&field=culvert_code:integer"
            "&field=culvert_rise:double&field=culvert_span:double"
            "&field=culvert_area_m2:double&field=culvert_barrels:integer"
            "&field=culvert_slope:double&field=inlet_invert_elev:double"
            "&field=outlet_invert_elev:double&field=entrance_loss_k:double"
            "&field=exit_loss_k:double&field=embankment_enabled:integer"
            "&field=embankment_crest_elev:double&field=embankment_overflow_width:double"
            "&field=embankment_weir_coeff:double&field=length:double&field=roughness_n:double&field=coeff:double"
            "&field=cd:double&field=opening:double&field=q_pump:double&field=max_flow:double"
            "&field=inlet_loss_k:double&field=outlet_loss_k:double"
            "&field=stacked_enabled:integer&field=influence_width_m:double"
            "&field=upstream_buffer_m:double&field=downstream_buffer_m:double"
            "&field=deck_soffit_elev:double&field=deck_top_elev:double"
            "&field=model_top_elev:double&field=under_layers:integer"
            "&field=over_layers:integer&field=pier_count:integer&field=pier_width:double"),
    ]

    def _uri_fields(uri: str):
        """Parse field names and SQLite column types from a memory layer URI string."""
        fields = []
        for part in uri.split("&"):
            if not part.startswith("field="):
                continue
            spec = part[len("field="):]
            if ":" not in spec:
                continue
            fname, ftype_raw = spec.split(":", 1)
            ftype_lower = ftype_raw.lower()
            if ftype_lower.startswith("integer") or ftype_lower.startswith("int"):
                sql_type = "INTEGER"
            elif ftype_lower.startswith("double") or ftype_lower.startswith("real"):
                sql_type = "REAL"
            else:
                sql_type = "TEXT"
            fields.append((fname, sql_type))
        return fields

    layers_added = []
    columns_added = []

    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        for layer_name, uri in layer_specs:
            # Check if the table already exists.
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (layer_name,),
            )
            exists = cur.fetchone() is not None

            if not exists:
                # Write empty layer via QGIS driver (handles geometry and GPKG metadata).
                mem_lyr = QgsVectorLayer(uri, layer_name, "memory")
                if mem_lyr.isValid():
                    self._write_memory_layer_to_gpkg(
                        mem_lyr, gpkg_path, layer_name, create_file=False
                    )
                    layers_added.append(layer_name)
            else:
                # Check for missing columns.
                expected = _uri_fields(uri)
                cur.execute(f"PRAGMA table_info(\"{layer_name}\")")
                existing_cols = {row[1].lower() for row in cur.fetchall()}
                for fname, sql_type in expected:
                    if fname.lower() not in existing_cols:
                        try:
                            cur.execute(
                                f"ALTER TABLE \"{layer_name}\" ADD COLUMN \"{fname}\" {sql_type}"
                            )
                            columns_added.append(f"{layer_name}.{fname}")
                        except Exception as col_err:
                            self._log(
                                f"[Migrate] Could not add column {layer_name}.{fname}: {col_err}"
                            )
        conn.commit()
    finally:
        conn.close()

    summary_parts = []
    if layers_added:
        summary_parts.append(f"Added {len(layers_added)} layer(s): {', '.join(layers_added)}")
    if columns_added:
        summary_parts.append(f"Added {len(columns_added)} column(s): {', '.join(columns_added)}")
    if not summary_parts:
        summary_parts.append("GeoPackage schema is already up to date — no changes needed.")

    summary = "; ".join(summary_parts)
    self._log(f"[Migrate] {summary}")
    self.layer_status_lbl.setText(f"GeoPackage updated: {summary}")





def _update_topology_control_summary(self):
    if not hasattr(self, "topo_controls_summary_lbl"):
        return

    # Guard against stale widget references after mesh termination
    try:
        import sip
        def _alive(w):
            return w is not None and not sip.isdeleted(w)
    except ImportError:
        def _alive(w):
            return w is not None

    if not _alive(self.topo_controls_summary_lbl):
        return

    def _safe_checked(name: str, default: bool = False) -> bool:
        w = getattr(self, name, None)
        if not _alive(w):
            return bool(default)
        try:
            return bool(w.isChecked())
        except Exception as e:
            self._log(f"[ERROR] _safe_checked {name}: {e}")
            return bool(default)

    def _safe_spin_value(name: str, default: float = 0.0) -> float:
        w = getattr(self, name, None)
        if not _alive(w):
            return float(default)
        try:
            return float(w.value())
        except Exception as e:
            self._log(f"[ERROR] _safe_spin_value {name}: {e}")
            return float(default)

    def _safe_line_text(name: str, default: str = "") -> str:
        w = getattr(self, name, None)
        if not _alive(w):
            return str(default)
        try:
            return str(w.text()).strip()
        except Exception as e:
            self._log(f"[ERROR] _safe_line_text {name}: {e}")
            return str(default)

    def _safe_combo_data(name: str, default: object = None):
        w = getattr(self, name, None)
        if not _alive(w):
            return default
        try:
            data = w.currentData()
            return default if data is None else data
        except Exception as e:
            self._log(f"[ERROR] _safe_combo_data {name}: {e}")
            return default

    topo_backend_combo = getattr(self, "topo_backend_combo", None)
    topo_regions_combo = getattr(self, "topo_regions_combo", None)
    topo_constraints_combo = getattr(self, "topo_constraints_combo", None)
    topo_quad_edges_combo = getattr(self, "topo_quad_edges_combo", None)

    backend_name = str(_safe_combo_data("topo_backend_combo", "gmsh") or "gmsh")
    regions_layer = self._combo_layer(topo_regions_combo, "vector") if _alive(topo_regions_combo) else None
    constraints_layer = self._combo_layer(topo_constraints_combo, "vector") if _alive(topo_constraints_combo) else None
    quad_edges_layer = self._combo_layer(topo_quad_edges_combo, "vector") if _alive(topo_quad_edges_combo) else None

    gmsh_controls_widget = getattr(self, "topo_gmsh_controls_widget", None)
    quality_controls_widget = getattr(self, "topo_quality_controls_widget", None)
    backend_controls_lbl = getattr(self, "topo_backend_controls_lbl", None)
    quality_controls_lbl = getattr(self, "topo_quality_controls_lbl", None)
    if gmsh_controls_widget is None and hasattr(self, "findChild"):
        gmsh_controls_widget = self.findChild(QtWidgets.QWidget, "topo_gmsh_controls_widget")
    if quality_controls_widget is None and hasattr(self, "findChild"):
        quality_controls_widget = self.findChild(QtWidgets.QWidget, "topo_quality_controls_widget")
    if backend_controls_lbl is None and hasattr(self, "findChild"):
        backend_controls_lbl = self.findChild(QtWidgets.QLabel, "topo_gmsh_controls_lbl")
    if quality_controls_lbl is None and hasattr(self, "findChild"):
        quality_controls_lbl = self.findChild(QtWidgets.QLabel, "topo_quality_controls_lbl")

    is_gmsh = backend_name == "gmsh"
    if _alive(gmsh_controls_widget):
        gmsh_controls_widget.setVisible(is_gmsh)
    if _alive(backend_controls_lbl):
        backend_controls_lbl.setVisible(is_gmsh)
        if is_gmsh:
            backend_controls_lbl.setText("Gmsh advanced controls:")
        else:
            backend_controls_lbl.setText("Backend advanced controls:")
    if _alive(quality_controls_widget):
        quality_controls_widget.setVisible(is_gmsh)
    if _alive(quality_controls_lbl):
        quality_controls_lbl.setVisible(is_gmsh)

    gmsh_quality_only = [
        getattr(self, "topo_gmsh_quality_enable_chk", None),
        getattr(self, "topo_gmsh_quality_max_iters_spin", None),
        getattr(self, "topo_gmsh_quality_time_limit_spin", None),
        getattr(self, "topo_gmsh_quality_recombine_topology_passes_edit", None),
        getattr(self, "topo_gmsh_quality_recombine_min_quality_edit", None),
        getattr(self, "topo_gmsh_quality_random_factors_edit", None),
        getattr(self, "topo_gmsh_quality_optimize_methods_edit", None),
        getattr(self, "topo_gmsh_algo_switch_on_failure_chk", None),
        getattr(self, "topo_gmsh_recombine_node_repositioning_chk", None),
        getattr(self, "topo_gmsh_global_recombine_chk", None),
    ]
    generic_quality_widgets = [
        getattr(self, "topo_quality_min_angle_spin", None),
        getattr(self, "topo_quality_max_aspect_spin", None),
        getattr(self, "topo_quality_max_non_orth_spin", None),
        getattr(self, "topo_quality_min_area_edit", None),
        getattr(self, "topo_quality_size_scales_edit", None),
        getattr(self, "topo_quality_smooth_increments_edit", None),
        getattr(self, "topo_quality_strict_chk", None),
    ]
    for widget in gmsh_quality_only:
        if _alive(widget):
            widget.setVisible(is_gmsh)
    for widget in generic_quality_widgets:
        if _alive(widget):
            widget.setVisible(is_gmsh)

    if backend_name == "gmsh":
        backend_hint = (
            "Gmsh: use multiple region polygons for multiblock meshes. "
            "Set region cell_type to 'cartesian' or 'quadrilateral' and populate edge_len_1..4 "
            "for per-edge structured spacing. Opposite edges are matched automatically. "
            "Enable full-region flow-aligned quads to drive transfinite spacing from quad-edge controls. "
            "Region interior rings plus empty regions/constraints are meshed as cutout holes."
        )
    else:
        backend_hint = (
            "Structured fallback: honors per-region target_size and cell_type, "
            "supports cutout holes from region interior rings and empty zones, "
            "but does not apply quad-edge transition layers or exact transfinite edge counts."
        )

    quality_hint = (
        " Quality UI: min angle >= {min_angle:.1f} deg, max aspect <= {max_aspect:.2f}, "
        "max non-orth <= {max_non_orth:.1f} deg, min area/bbox >= {min_area}, strict={strict}; "
        "retry scales={size_scales}, smooth increments={smooth_increments}, "
        "recombine topology={recombine_topology}, recombine minQ={recombine_minq}, rand={random_factors}, "
        "optimize methods={opt_methods}, algo-switch={algo_switch}, node-reposition={node_reposition}, "
        "global-recombine={global_recombine}; "
        "gmsh-full-align={gmsh_full_align}; "
        "arc-mode={arc_mode}, soft-size={arc_soft_size:.3g}, soft-dist={arc_soft_dist:.3g}, "
        "iface-transition={iface_transition}, iface-dist={iface_dist:.3g}, iface-ratio>={iface_ratio:.3g}, "
        "threads={gmsh_threads}, max2d={gmsh_max2d_threads}, "
        "min-cell={mesh_size_min:.6g}, edge-tol={edge_tol:.6g}, "
        "point-refine={point_refine}; Gmsh loop={gmsh_loop}, attempts={attempts}, budget={budget:.1f}s; "
        ""
    ).format(
        min_angle=_safe_spin_value("topo_quality_min_angle_spin", 0.0),
        max_aspect=_safe_spin_value("topo_quality_max_aspect_spin", 0.0),
        max_non_orth=_safe_spin_value("topo_quality_max_non_orth_spin", 0.0),
        min_area=_safe_line_text("topo_quality_min_area_edit", "0"),
        strict="on" if _safe_checked("topo_quality_strict_chk", False) else "off",
        size_scales=_safe_line_text("topo_quality_size_scales_edit", "1.0"),
        smooth_increments=_safe_line_text("topo_quality_smooth_increments_edit", "0"),
        recombine_topology=_safe_line_text("topo_gmsh_quality_recombine_topology_passes_edit", "5"),
        recombine_minq=_safe_line_text("topo_gmsh_quality_recombine_min_quality_edit", "0.01"),
        random_factors=_safe_line_text("topo_gmsh_quality_random_factors_edit", "1e-9"),
        opt_methods=_safe_line_text("topo_gmsh_quality_optimize_methods_edit", "Laplace2D"),
        algo_switch="on" if _safe_checked("topo_gmsh_algo_switch_on_failure_chk", False) else "off",
        node_reposition="on" if _safe_checked("topo_gmsh_recombine_node_repositioning_chk", False) else "off",
        global_recombine="on" if _safe_checked("topo_gmsh_global_recombine_chk", False) else "off",
        gmsh_full_align="on" if _safe_checked("topo_gmsh_quad_full_region_flow_align_chk", False) else "off",
        arc_mode=str(_safe_combo_data("topo_gmsh_arc_mode_combo", "hard_embed") or "hard_embed"),
        arc_soft_size=_safe_spin_value("topo_gmsh_arc_soft_size_factor_spin", 0.5),
        arc_soft_dist=_safe_spin_value("topo_gmsh_arc_soft_dist_factor_spin", 2.0),
        iface_transition="on" if _safe_checked("topo_gmsh_interface_transition_enable_chk", False) else "off",
        iface_dist=_safe_spin_value("topo_gmsh_interface_transition_dist_factor_spin", 2.5),
        iface_ratio=_safe_spin_value("topo_gmsh_interface_transition_min_ratio_spin", 1.25),
        gmsh_threads=int(round(_safe_spin_value("topo_gmsh_num_threads_spin", 1.0))),
        gmsh_max2d_threads=int(round(_safe_spin_value("topo_gmsh_max_num_threads_2d_spin", 0.0))),
        mesh_size_min=_safe_spin_value("topo_gmsh_mesh_size_min_spin", 0.0),
        edge_tol=_safe_spin_value("topo_gmsh_tolerance_edge_length_spin", 0.0),
        point_refine="on" if _safe_checked("topo_gmsh_mesh_size_from_points_chk", False) else "off",
        gmsh_loop="on" if _safe_checked("topo_gmsh_quality_enable_chk", False) else "off",
        attempts=int(round(_safe_spin_value("topo_gmsh_quality_max_iters_spin", 0.0))),
        budget=_safe_spin_value("topo_gmsh_quality_time_limit_spin", 0.0),
    )

    details: List[str] = []
    if regions_layer is not None:
        try:
            region_fields = set(regions_layer.fields().names())
            region_count = 0
            cartesian_count = 0
            empty_count = 0
            size_values = set()
            missing_edge_lengths = 0
            for ft in regions_layer.getFeatures():
                region_count += 1
                ctype = str(ft["cell_type"]).strip().lower() if "cell_type" in region_fields and ft["cell_type"] not in (None, "") else ""
                if ctype in {"cartesian", "quadrilateral"}:
                    cartesian_count += 1
                    edge_fields = [f"edge_len_{i}" for i in range(1, 5)]
                    edge_ok = True
                    for name in edge_fields:
                        if name not in region_fields or ft[name] in (None, ""):
                            edge_ok = False
                            break
                        try:
                            if float(ft[name]) <= 0.0:
                                edge_ok = False
                                break
                        except Exception as e:
                            self._log(f"[ERROR] edge_len float conversion: {e}")
                            edge_ok = False
                            break
                    if not edge_ok:
                        missing_edge_lengths += 1
                if ctype == "empty":
                    empty_count += 1
                if "target_size" in region_fields and ft["target_size"] not in (None, ""):
                    try:
                        size_values.add(round(float(ft["target_size"]), 6))
                    except Exception as e:
                        self._log(f"[ERROR] target_size float conversion: {e}")
            details.append(f"regions={region_count}")
            if cartesian_count > 0:
                details.append(f"structured-block-regions={cartesian_count}")
            if empty_count > 0:
                details.append(f"empty-regions={empty_count}")
            if len(size_values) > 1:
                details.append(f"multi-block sizes={len(size_values)}")
            if missing_edge_lengths > 0:
                details.append(f"structured regions missing edge_len_1..4={missing_edge_lengths}")
        except Exception as e:
            self._log(f"[ERROR] regions layer summary: {e}")

    if constraints_layer is not None and _alive(topo_constraints_combo) and topo_constraints_combo.currentData() is not None:
        try:
            c_fields = set(constraints_layer.fields().names())
            constraint_count = 0
            empty_constraints = 0
            for ft in constraints_layer.getFeatures():
                constraint_count += 1
                ctype = str(ft["cell_type"]).strip().lower() if "cell_type" in c_fields and ft["cell_type"] not in (None, "") else ""
                if ctype == "empty":
                    empty_constraints += 1
            details.append(f"constraints={constraint_count}")
            if empty_constraints > 0:
                details.append(f"empty-constraints={empty_constraints}")
        except Exception as e:
            self._log(f"[ERROR] constraints layer summary: {e}")

    if quad_edges_layer is not None and _alive(topo_quad_edges_combo) and topo_quad_edges_combo.currentData() is not None:
        try:
            q_fields = set(quad_edges_layer.fields().names())
            edge_count = 0
            layered_edges = 0
            total_layers = 0
            for ft in quad_edges_layer.getFeatures():
                edge_count += 1
                if "n_layers" in q_fields and ft["n_layers"] not in (None, ""):
                    nl = max(0, int(ft["n_layers"]))
                    total_layers += nl
                    if nl > 0:
                        layered_edges += 1
            details.append(f"quad-edges={edge_count}")
            if layered_edges > 0:
                details.append(f"transition-layer-edges={layered_edges}")
                details.append(f"total-n_layers={total_layers}")
        except Exception as e:
            self._log(f"[ERROR] quad_edges layer summary: {e}")

    suffix = " | ".join(details)
    if suffix:
        self.topo_controls_summary_lbl.setText(f"{backend_hint}{quality_hint} Current layers: {suffix}.")
    else:
        self.topo_controls_summary_lbl.setText(f"{backend_hint}{quality_hint}")





def _configure_swe2d_layer_editors(self, layer):
    if layer is None or not isinstance(layer, QgsVectorLayer):
        return
    lname = str(layer.name()).lower()

    try:
        from qgis.core import QgsEditFormConfig
        cfg = layer.editFormConfig()
        if hasattr(QgsEditFormConfig, "DragAndDrop") and hasattr(cfg, "setLayout"):
            cfg.setLayout(QgsEditFormConfig.DragAndDrop)
            layer.setEditFormConfig(cfg)
    except Exception as e:
        self._log(f"[ERROR] set layer edit form config: {e}")

    def _set_alias(field_name: str, alias: str) -> None:
        try:
            idx = layer.fields().indexOf(field_name)
            if idx >= 0:
                layer.setFieldAlias(idx, alias)
        except Exception as e:
            self._log(f"[ERROR] set field alias {field_name}: {e}")

    is_region = "topo_regions" in lname or lname.endswith("swe2d_topo_regions")
    is_arc = "topo_arcs" in lname or lname.endswith("swe2d_topo_arcs")
    is_constraint = "topo_constraints" in lname or lname.endswith("swe2d_topo_constraints")
    is_quad_edges = "topo_quad_edges" in lname or lname.endswith("swe2d_topo_quad_edges")
    is_bc_lines = "bc_lines" in lname
    is_sample_lines = "sample_lines" in lname
    is_manning = "manning" in lname
    is_cn_zone = "cn_zones" in lname
    is_rain_gage = "rain_gages" in lname
    is_hyetograph = "hyetographs" in lname
    is_drain_nodes = "drainage_nodes" in lname
    is_drain_links = "drainage_links" in lname
    is_drain_inlets = "drainage_inlets" in lname
    is_drain_node_inlets = "drainage_node_inlets" in lname
    is_structures = ("structures" in lname) and ("hydrographs" not in lname)

    if is_region or is_constraint:
        self._set_value_map_editor(
            layer,
            "cell_type",
            {s.capitalize(): s for s in _CELL_TYPE_OPTIONS},
        )
        allowed = ", ".join(f"'{s}'" for s in _CELL_TYPE_OPTIONS)
        self._set_expression_constraint(layer, "cell_type", f"\"cell_type\" IN ({allowed})")
        self._set_expression_constraint(layer, "target_size", '"target_size" > 0')
        for nm in ("edge_len_1", "edge_len_2", "edge_len_3", "edge_len_4"):
            self._set_expression_constraint(layer, nm, f'"{nm}" IS NULL OR "{nm}" > 0')

    if is_arc:
        self._set_value_map_editor(
            layer,
            "arc_mode_override",
            {
                "Hard embed arcs": "hard_embed",
                "Soft arc size hint": "soft_size_hint",
                "Disable arc influence": "disabled",
            },
        )
        self._set_expression_constraint(layer, "use_global_arc_ctrl", '"use_global_arc_ctrl" IS NULL OR "use_global_arc_ctrl" IN (0,1)')
        self._set_expression_constraint(
            layer,
            "arc_mode_override",
            '"arc_mode_override" IS NULL OR "arc_mode_override" IN (\'hard_embed\',\'soft_size_hint\',\'disabled\')',
        )
        self._set_expression_constraint(layer, "arc_soft_size_override", '"arc_soft_size_override" IS NULL OR "arc_soft_size_override" > 0')
        self._set_expression_constraint(layer, "arc_soft_dist_override", '"arc_soft_dist_override" IS NULL OR "arc_soft_dist_override" > 0')

    if is_bc_lines:
        self._set_value_map_editor(layer, "bc_type", _BC_VALUE_MAP)
        self._set_expression_constraint(layer, "bc_type", '"bc_type" IN (1,2,3,4,5,6,7,102,103)')
        self._set_expression_constraint(layer, "priority", '"priority" >= 0')

    if is_quad_edges:
        self._set_expression_constraint(layer, "region_id", '"region_id" >= 0')
        self._set_expression_constraint(layer, "edge_id", '"edge_id" IN (1,2,3,4)')
        self._set_expression_constraint(layer, "target_size", '"target_size" IS NULL OR "target_size" > 0')
        self._set_expression_constraint(layer, "n_layers", '"n_layers" >= 0')
        self._set_expression_constraint(layer, "first_height", '"first_height" IS NULL OR "first_height" > 0')
        self._set_expression_constraint(layer, "growth_rate", '"growth_rate" IS NULL OR "growth_rate" > 0')

    if is_manning:
        self._set_expression_constraint(layer, "n_mann", '"n_mann" >= 0')
        self._set_expression_constraint(layer, "priority", '"priority" >= 0')

    if is_cn_zone:
        self._set_expression_constraint(layer, "cn", '"cn" >= 1 AND "cn" <= 100')
        self._set_expression_constraint(layer, "priority", '"priority" >= 0')

    if is_rain_gage:
        self._set_value_map_editor(layer, "units", _RAIN_GAGE_UNITS_VALUE_MAP)
        self._set_expression_constraint(layer, "gage_id", 'length(trim("gage_id")) > 0')
        self._set_expression_constraint(layer, "hyetograph_id", 'length(trim("hyetograph_id")) > 0')
        self._set_expression_constraint(layer, "units", '"units" IS NULL OR "units" IN (\'mm/hr\',\'in/hr\',\'mm\',\'in\')')

    if is_hyetograph:
        self._set_value_map_editor(layer, "value_type", _HYETOGRAPH_VALUE_TYPE_MAP)
        self._set_value_map_editor(layer, "units", _HYETOGRAPH_UNITS_VALUE_MAP)
        self._set_expression_constraint(layer, "hyetograph_id", 'length(trim("hyetograph_id")) > 0')
        self._set_expression_constraint(layer, "Time", 'length(trim("Time")) > 0')
        self._set_expression_constraint(layer, "Value", '"Value" >= 0')
        self._set_expression_constraint(layer, "value_type", '"value_type" IS NULL OR "value_type" IN (\'intensity\',\'incremental\',\'cumulative\')')
        self._set_expression_constraint(layer, "units", '"units" IS NULL OR "units" IN (\'mm/hr\',\'in/hr\',\'mm\',\'in\')')

    if is_sample_lines:
        self._set_expression_constraint(layer, "line_id", '"line_id" IS NULL OR "line_id" >= 0')
        self._set_expression_constraint(layer, "enabled", '"enabled" IS NULL OR "enabled" IN (0,1)')
        self._set_expression_constraint(layer, "priority", '"priority" IS NULL OR "priority" >= 0')

    if is_drain_nodes:
        node_field_names = set(layer.fields().names())
        self._set_value_map_editor(layer, "node_type", _DRAIN_NODE_TYPE_VALUE_MAP)
        self._set_expression_constraint(layer, "node_id", 'length(trim("node_id")) > 0')
        self._set_expression_constraint(layer, "node_type", '"node_type" IN (\'junction\',\'outfall\',\'storage\',\'inlet\',\'pipe_end\')')
        self._set_expression_constraint(layer, "max_depth", '"max_depth" IS NULL OR "max_depth" > 0')
        self._set_expression_constraint(layer, "rim_elev", '"rim_elev" IS NULL OR "rim_elev" >= "invert_elev"')
        self._set_expression_constraint(layer, "crest_elev", '"crest_elev" IS NULL OR "crest_elev" >= "invert_elev"')
        self._set_expression_constraint(layer, "surface_area", '"surface_area" IS NULL OR "surface_area" > 0')
        if "outfall_area" in node_field_names:
            self._set_expression_constraint(layer, "outfall_area", '"outfall_area" IS NULL OR "outfall_area" > 0')
        if "zero_storage" in node_field_names:
            self._set_expression_constraint(layer, "zero_storage", '"zero_storage" IS NULL OR "zero_storage" IN (0,1)')

    if is_drain_links:
        self._set_value_map_editor(layer, "link_type", _DRAIN_LINK_TYPE_VALUE_MAP)
        self._set_value_map_editor(layer, "link_shape", _DRAIN_LINK_SHAPE_VALUE_MAP)
        self._set_expression_constraint(layer, "link_id", 'length(trim("link_id")) > 0')
        self._set_expression_constraint(layer, "from_node", 'length(trim("from_node")) > 0')
        self._set_expression_constraint(layer, "to_node", 'length(trim("to_node")) > 0')
        self._set_expression_constraint(layer, "link_type", '"link_type" IN (\'conduit\',\'lateral_simple\',\'pump\',\'weir\',\'orifice\',\'culvert\')')
        self._set_expression_constraint(layer, "link_shape", '"link_shape" IS NULL OR "link_shape" IN (\'circular\',\'box\',\'pipe_arch\',\'custom\')')
        self._set_expression_constraint(layer, "length", '"length" IS NULL OR "length" > 0')
        self._set_expression_constraint(layer, "roughness_n", '"roughness_n" IS NULL OR "roughness_n" > 0')
        self._set_expression_constraint(layer, "diameter", '"diameter" IS NULL OR "diameter" > 0')
        self._set_expression_constraint(layer, "span", '"span" IS NULL OR "span" > 0')
        self._set_expression_constraint(layer, "rise", '"rise" IS NULL OR "rise" > 0')
        self._set_expression_constraint(layer, "area_m2", '"area_m2" IS NULL OR "area_m2" > 0')

    if is_drain_inlets:
        self._set_expression_constraint(layer, "inlet_type_id", 'length(trim("inlet_type_id")) > 0')
        self._set_expression_constraint(layer, "weir_length", '"weir_length" IS NULL OR "weir_length" > 0')
        self._set_expression_constraint(layer, "orifice_area", '"orifice_area" IS NULL OR "orifice_area" > 0')
        self._set_expression_constraint(layer, "coeff_weir", '"coeff_weir" IS NULL OR "coeff_weir" > 0')
        self._set_expression_constraint(layer, "coeff_orifice", '"coeff_orifice" IS NULL OR "coeff_orifice" > 0')
        self._set_expression_constraint(layer, "max_capture", '"max_capture" IS NULL OR "max_capture" > 0')

    if is_drain_node_inlets:
        self._set_expression_constraint(layer, "node_id", 'length(trim("node_id")) > 0')
        self._set_expression_constraint(layer, "inlet_type_id", 'length(trim("inlet_type_id")) > 0')
        self._set_expression_constraint(layer, "inlet_count", '"inlet_count" IS NULL OR "inlet_count" > 0')

    if is_structures:
        self._set_value_map_editor(layer, "structure_type", _STRUCTURE_TYPE_VALUE_MAP)
        self._set_value_map_editor(layer, "culvert_shape", {"Circular": "circular", "Box": "box", "Rectangular": "rectangular"})
        self._set_value_map_editor(layer, "embankment_enabled", {"No": 0, "Yes": 1})
        self._set_expression_constraint(layer, "structure_id", 'length(trim("structure_id")) > 0')
        self._set_expression_constraint(layer, "structure_type", '"structure_type" IN (1,2,3,4,5)')
        self._set_expression_constraint(layer, "enabled", '"enabled" IS NULL OR "enabled" IN (0,1)')
        self._set_expression_constraint(layer, "culvert_code", '"culvert_code" IS NULL OR "culvert_code" >= 1')
        self._set_expression_constraint(layer, "culvert_rise", '"culvert_rise" IS NULL OR "culvert_rise" > 0')
        self._set_expression_constraint(layer, "culvert_span", '"culvert_span" IS NULL OR "culvert_span" > 0')
        self._set_expression_constraint(layer, "culvert_area_m2", '"culvert_area_m2" IS NULL OR "culvert_area_m2" > 0')
        self._set_expression_constraint(layer, "culvert_barrels", '"culvert_barrels" IS NULL OR "culvert_barrels" >= 1')
        self._set_expression_constraint(layer, "length", '"length" IS NULL OR "length" > 0')
        self._set_expression_constraint(layer, "roughness_n", '"roughness_n" IS NULL OR "roughness_n" > 0')
        self._set_expression_constraint(layer, "entrance_loss_k", '"entrance_loss_k" IS NULL OR "entrance_loss_k" >= 0')
        self._set_expression_constraint(layer, "exit_loss_k", '"exit_loss_k" IS NULL OR "exit_loss_k" >= 0')
        self._set_expression_constraint(layer, "embankment_enabled", '"embankment_enabled" IS NULL OR "embankment_enabled" IN (0,1)')
        self._set_expression_constraint(layer, "embankment_overflow_width", '"embankment_overflow_width" IS NULL OR "embankment_overflow_width" >= 0')
        self._set_expression_constraint(layer, "embankment_weir_coeff", '"embankment_weir_coeff" IS NULL OR "embankment_weir_coeff" > 0')

        for field_name, alias in (
            ("culvert_shape", "Culvert Shape"),
            ("culvert_code", "FHWA Culvert Code"),
            ("culvert_rise", "Culvert Rise"),
            ("culvert_span", "Culvert Span"),
            ("culvert_area_m2", "Override Area"),
            ("culvert_barrels", "Barrel Count"),
            ("culvert_slope", "Culvert Slope"),
            ("inlet_invert_elev", "Inlet Invert Elev."),
            ("outlet_invert_elev", "Outlet Invert Elev."),
            ("entrance_loss_k", "Entrance Loss K"),
            ("exit_loss_k", "Exit Loss K"),
            ("embankment_enabled", "Enable Embankment Overflow"),
            ("embankment_crest_elev", "Embankment Crest Elev."),
            ("embankment_overflow_width", "Overflow Width"),
            ("embankment_weir_coeff", "Weir Coefficient"),
        ):
            _set_alias(field_name, alias)

        try:
            from qgis.core import QgsEditFormConfig
            cfg = layer.editFormConfig()
            form_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "forms", "swe2d_structures_culvert_form.ui")
            if os.path.exists(form_path) and hasattr(cfg, "setUiForm"):
                cfg.setUiForm(form_path)
                if hasattr(QgsEditFormConfig, "UiFileLayout") and hasattr(cfg, "setLayout"):
                    cfg.setLayout(QgsEditFormConfig.UiFileLayout)
                layer.setEditFormConfig(cfg)
        except Exception as e:
            self._log(f"[ERROR] culvert form config: {e}")





def _cleanup_topology_mesh_checkpoint(self) -> None:
    cp_path = str(getattr(self, "_topology_mesh_checkpoint_path", "") or "").strip()
    if cp_path:
        try:
            os.remove(cp_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            self._log(f"[ERROR] checkpoint cleanup remove: {e}")
    self._topology_mesh_checkpoint_path = ""

    progress_path = str(getattr(self, "_topology_mesh_progress_path", "") or "").strip()
    if progress_path:
        try:
            os.remove(progress_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            self._log(f"[ERROR] progress cleanup remove: {e}")
    self._topology_mesh_progress_path = ""
    self._topology_mesh_progress_last_seq = -1
    self._topology_mesh_progress_last_sig = ""
    self._topology_mesh_progress = None


def _recover_topology_mesh_checkpoint(self, backend_name: str, run_mode: str, elapsed: float) -> bool:
    if str(backend_name).strip().lower() != "gmsh":
        return False

    cp_path = str(getattr(self, "_topology_mesh_checkpoint_path", "") or "").strip()
    if not cp_path or not os.path.exists(cp_path):
        return False

    try:
        with np.load(cp_path, allow_pickle=False) as cp:
            node_x = np.asarray(cp["node_x"], dtype=np.float64)
            node_y = np.asarray(cp["node_y"], dtype=np.float64)
            node_z = np.asarray(cp["node_z"], dtype=np.float64)
            cell_nodes = np.asarray(cp["cell_nodes"], dtype=np.int32)
            cell_face_offsets = np.asarray(cp["cell_face_offsets"], dtype=np.int32)
            cell_face_nodes = np.asarray(cp["cell_face_nodes"], dtype=np.int32)
            cell_type = np.asarray(cp["cell_type"]).astype(object)
            region_id = np.asarray(cp["region_id"], dtype=np.int32)
            target_size = np.asarray(cp["target_size"], dtype=np.float64)
            quality_summary = None
            if "quality_summary_json" in cp.files:
                try:
                    import json as _json
                    raw = str(np.asarray(cp["quality_summary_json"]).item())
                    quality_summary = _json.loads(raw) if raw else None
                except Exception as e:
                    self._log(f"[ERROR] checkpoint quality_summary json parse: {e}")
                    quality_summary = None
    except Exception as exc:
        self._log(f"mesh> checkpoint-read-fail path={cp_path} error={exc}")
        return False

    n_nodes = int(node_x.size)
    n_faces = max(0, int(cell_face_offsets.size) - 1)
    if n_nodes <= 0 or n_faces <= 0:
        return False

    self._mesh_data = {
        "nx": np.array(max(2, int(round(np.sqrt(node_x.size))))),
        "ny": np.array(max(2, int(round(np.sqrt(node_x.size))))),
        "lx": np.array(max(float(np.max(node_x) - np.min(node_x)), 1.0)),
        "ly": np.array(max(float(np.max(node_y) - np.min(node_y)), 1.0)),
        "node_x": node_x,
        "node_y": node_y,
        "node_z": node_z,
        "cell_nodes": cell_nodes,
        "cell_face_offsets": cell_face_offsets,
        "cell_face_nodes": cell_face_nodes,
        "cell_type": cell_type,
        "region_id": region_id,
        "target_size": target_size,
    }
    if isinstance(quality_summary, dict):
        self._mesh_data["quality_summary"] = dict(quality_summary)
    if hasattr(self, "_reset_runtime_snapshot_overlay_cache"):
        self._reset_runtime_snapshot_overlay_cache("topology mesh checkpoint recovered")

    n_tris = int(cell_nodes.size // 3)
    self.mesh_info_lbl.setText(f"Topology mesh: nodes={node_x.size}, faces={n_faces}, plot_triangles={n_tris}")
    self.topo_status_lbl.setText(
        f"Recovered {n_faces} computational faces from latest Gmsh attempt after timeout "
        f"(elapsed={elapsed:.2f}s, backend='{backend_name}')."
    )
    self._log(
        "mesh> recovered-checkpoint "
        f"backend={backend_name} mode={run_mode} nodes={node_x.size} faces={n_faces} elapsed={elapsed:.2f}s"
    )
    if isinstance(quality_summary, dict):
        best_stats = quality_summary.get("best_stats", {})
        try:
            self._log(
                "mesh> gmsh-quality-summary "
                f"attempts={int(quality_summary.get('attempts', 0))} "
                f"strict={bool(quality_summary.get('strict_requested', False))} "
                f"passed={bool(quality_summary.get('had_passing_candidate', False))} "
                f"fail_cells(any/angle/aspect/area/non_orth)="
                f"{int(float(best_stats.get('failed_any_cells', 0.0)))}/"
                f"{int(float(best_stats.get('failed_min_angle_cells', 0.0)))}/"
                f"{int(float(best_stats.get('failed_max_aspect_cells', 0.0)))}/"
                f"{int(float(best_stats.get('failed_min_area_cells', 0.0)))}/"
                f"{int(float(best_stats.get('failed_max_non_orth_cells', 0.0)))}"
            )
        except Exception as e:
            self._log(f"[ERROR] checkpoint quality summary log: {e}")

    self._result_data = None
    self.view_mode_combo.setCurrentText("Mesh")
    self._refresh_plot()
    return True


def _poll_tqmesh_progress(self) -> None:
    backend_name = str(getattr(self, "_topology_mesh_backend", "") or "").strip().lower()
    if backend_name not in {"tqmesh", "gmsh"}:
        return

    progress_path = str(getattr(self, "_topology_mesh_progress_path", "") or "").strip()
    if not progress_path or not os.path.exists(progress_path):
        return

    try:
        import json as _json

        with open(progress_path, "r", encoding="utf-8") as fh:
            payload = _json.load(fh)
    except Exception as e:
        self._log(f"[ERROR] progress json read: {e}")
        return

    if not isinstance(payload, dict):
        return

    try:
        seq = int(payload.get("seq", -1))
    except Exception as e:
        self._log(f"[ERROR] progress seq parse: {e}")
        seq = -1

    last_seq = int(getattr(self, "_topology_mesh_progress_last_seq", -1))
    payload_sig = (
        f"{payload.get('stage', '')}|{payload.get('region_id', '')}|"
        f"{payload.get('attempt', '')}|{payload.get('detail', '')}|{seq}"
    )
    last_sig = str(getattr(self, "_topology_mesh_progress_last_sig", "") or "")
    if (seq >= 0 and seq == last_seq) or (payload_sig == last_sig):
        return
    self._topology_mesh_progress_last_seq = seq
    self._topology_mesh_progress_last_sig = payload_sig
    self._topology_mesh_progress = dict(payload)

    stage = str(payload.get("stage", "")).strip() or "update"
    detail = str(payload.get("detail", "")).strip()
    region_id = payload.get("region_id", None)
    attempt = payload.get("attempt", None)
    elapsed_s = payload.get("elapsed_s", None)

    parts = [f"stage={stage}"]
    if region_id is not None:
        parts.append(f"region={region_id}")
    if attempt is not None:
        parts.append(f"attempt={attempt}")
    if elapsed_s is not None:
        try:
            parts.append(f"elapsed={float(elapsed_s):.2f}s")
        except Exception as e:
            self._log(f"[ERROR] progress elapsed_s format: {e}")
    if detail:
        parts.append(f"detail={detail}")
    self._log(f"mesh> {backend_name}-progress " + " ".join(parts))


def _poll_topology_mesh_future(self):
    fut = self._topology_mesh_future
    if fut is None:
        self._topology_mesh_timer.stop()
        self._set_topology_mesh_busy(False)
        return

    elapsed = 0.0
    if self._topology_mesh_started_at is not None:
        elapsed = max(0.0, time.perf_counter() - self._topology_mesh_started_at)

    if elapsed > self._topology_mesh_active_timeout_sec and not fut.done():
        backend_name = self._topology_mesh_backend or "unknown"
        run_mode = self._topology_mesh_run_mode
        self._topology_mesh_timer.stop()
        self._topology_mesh_future = None
        self._topology_mesh_started_at = None
        self._topology_mesh_poll_count = 0

        # For process-executed backends (gmsh/tqmesh), terminate and recreate
        # the pool to ensure stuck native meshing work is not left running.
        if backend_name in {"gmsh", "tqmesh"} and self._topology_mesh_process_pool is not None:
            try:
                self._topology_mesh_process_pool.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                self._log(f"[ERROR] process pool shutdown: {e}")
            self._topology_mesh_process_pool = None

        recovered = _recover_topology_mesh_checkpoint(self, backend_name=backend_name, run_mode=run_mode, elapsed=elapsed)

        if not recovered:
            self.topo_status_lbl.setText(
                f"Topology meshing timed out after {self._topology_mesh_active_timeout_sec:.0f}s "
                f"(backend '{backend_name}')."
            )
        self._log(
            "mesh> timeout "
            f"backend={backend_name} mode={run_mode} elapsed={elapsed:.2f}s "
            f"limit={self._topology_mesh_active_timeout_sec:.0f}s"
        )

        if recovered:
            self._log(
                "mesh> timeout-recovery "
                f"backend={backend_name} mode={run_mode} action=loaded_latest_attempt"
            )

        self._set_topology_mesh_busy(False)
        _cleanup_topology_mesh_checkpoint(self)
        return

    if not fut.done():
        self._topology_mesh_poll_count += 1
        _poll_tqmesh_progress(self)
        # Emit lightweight runtime heartbeat at ~1 second cadence.
        if self._topology_mesh_poll_count % 8 == 0:
            backend_running = str(self._topology_mesh_backend or "unknown").strip().lower()
            spinner = "|/-\\"[(self._topology_mesh_poll_count // 8) % 4]
            if backend_running == "gmsh":
                try:
                    status_txt = str(self.topo_status_lbl.text() or "").strip()
                except Exception as e:
                    self._log(f"[ERROR] gmsh status text read: {e}")
                    status_txt = ""
                elapsed_s = 0.0
                if self._topology_mesh_started_at is not None:
                    elapsed_s = max(0.0, time.perf_counter() - self._topology_mesh_started_at)
                self._topology_mesh_progress = {
                    "backend": "gmsh",
                    "stage": "running",
                    "spinner": str(spinner),
                    "elapsed_s": float(elapsed_s),
                    "detail": str(status_txt),
                }
                parts = [
                    "stage=running",
                    f"status={spinner}",
                    f"elapsed={self._format_elapsed(self._topology_mesh_started_at)}",
                ]
                if status_txt:
                    parts.append(f"detail={status_txt}")
                self._log("mesh> gmsh-progress " + " ".join(parts))
            else:
                self._log(
                    "mesh> run "
                    f"status={spinner} backend={self._topology_mesh_backend or 'unknown'} "
                    f"elapsed={self._format_elapsed(self._topology_mesh_started_at)}"
                )
        return

    self._topology_mesh_timer.stop()
    backend_name = self._topology_mesh_backend or "unknown"
    default_cell_type = self._topology_mesh_default_cell_type or "triangular"
    run_mode = self._topology_mesh_run_mode
    elapsed_str = self._format_elapsed(self._topology_mesh_started_at)
    self._topology_mesh_future = None
    self._topology_mesh_started_at = None
    self._topology_mesh_poll_count = 0
    self._topology_mesh_progress = None
    fallback_restarted = False

    try:
        mesh = fut.result()
        n_nodes = int(np.asarray(mesh.node_x).size)
        n_faces = max(0, int(np.asarray(mesh.cell_face_offsets).size) - 1)
        if n_nodes <= 0 or n_faces <= 0:
            raise RuntimeError(
                f"Topology backend '{backend_name}' produced an empty mesh "
                f"(nodes={n_nodes}, faces={n_faces})."
            )
        self._mesh_data = {
            "nx": np.array(max(2, int(round(np.sqrt(mesh.node_x.size))))),
            "ny": np.array(max(2, int(round(np.sqrt(mesh.node_x.size))))),
            "lx": np.array(max(float(np.max(mesh.node_x) - np.min(mesh.node_x)), 1.0)),
            "ly": np.array(max(float(np.max(mesh.node_y) - np.min(mesh.node_y)), 1.0)),
            "node_x": mesh.node_x,
            "node_y": mesh.node_y,
            "node_z": mesh.node_z,
            "cell_nodes": mesh.cell_nodes,
            "cell_face_offsets": mesh.cell_face_offsets,
            "cell_face_nodes": mesh.cell_face_nodes,
            "cell_type": mesh.cell_type,
            "region_id": mesh.region_id,
            "target_size": mesh.target_size,
        }
        quality_summary = getattr(mesh, "quality_summary", None)
        if isinstance(quality_summary, dict):
            self._mesh_data["quality_summary"] = dict(quality_summary)
        if hasattr(self, "_reset_runtime_snapshot_overlay_cache"):
            self._reset_runtime_snapshot_overlay_cache("topology mesh regenerated")
        n_faces = int(mesh.cell_face_offsets.size - 1)
        n_tris = int(mesh.cell_nodes.size // 3)
        self.mesh_info_lbl.setText(f"Topology mesh: nodes={mesh.node_x.size}, faces={n_faces}, plot_triangles={n_tris}")
        if run_mode == "fallback-no-constraints":
            self.topo_status_lbl.setText(
                f"Generated {n_faces} computational faces using backend '{backend_name}' "
                "after automatic fallback with constraints disabled. "
                "Review/repair constraint polygons and regenerate when ready."
            )
        else:
            self.topo_status_lbl.setText(
                f"Generated {n_faces} computational faces using backend '{backend_name}'. "
                "Cell metadata (type/size/region) stored in mesh state."
            )
        self._log(
            "mesh> done "
            f"backend={backend_name} default_cell_type={default_cell_type} "
            f"mode={run_mode} "
            f"nodes={mesh.node_x.size} faces={n_faces} elapsed={elapsed_str}"
        )
        quality_summary = getattr(mesh, "quality_summary", None)
        if isinstance(quality_summary, dict):
            best_stats = quality_summary.get("best_stats", {})
            try:
                self._log(
                    "mesh> gmsh-quality-summary "
                    f"attempts={int(quality_summary.get('attempts', 0))} "
                    f"strict={bool(quality_summary.get('strict_requested', False))} "
                    f"passed={bool(quality_summary.get('had_passing_candidate', False))} "
                    f"fail_cells(any/angle/aspect/area/non_orth)="
                    f"{int(float(best_stats.get('failed_any_cells', 0.0)))}/"
                    f"{int(float(best_stats.get('failed_min_angle_cells', 0.0)))}/"
                    f"{int(float(best_stats.get('failed_max_aspect_cells', 0.0)))}/"
                    f"{int(float(best_stats.get('failed_min_area_cells', 0.0)))}/"
                    f"{int(float(best_stats.get('failed_max_non_orth_cells', 0.0)))}"
                )
            except Exception as e:
                self._log(f"[ERROR] quality summary log in poll: {e}")
        self._result_data = None
        self.view_mode_combo.setCurrentText("Mesh")
        self._refresh_plot()
    except NotImplementedError as exc:
        self.topo_status_lbl.setText(str(exc))
        self._log(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
    except RuntimeError as exc:
        err_txt = str(exc)
        err_l = err_txt.lower()
        empty_mesh_failure = ("empty mesh" in err_l) or ("non-empty mesh" in err_l)
        conceptual = self._topology_mesh_conceptual
        can_retry_without_constraints = (
            backend_name == "gmsh"
            and run_mode == "full"
            and not self._topology_mesh_auto_fallback_used
            and conceptual is not None
            and bool(getattr(conceptual, "constraints", []))
        )
        if empty_mesh_failure and can_retry_without_constraints:
            try:
                fallback_conceptual = _clone_conceptual_without_constraints(conceptual)
                self._topology_mesh_auto_fallback_used = True
                self._log(
                    "mesh> fallback "
                    f"backend={backend_name} action=retry_without_constraints "
                    f"reason=empty-mesh elapsed={elapsed_str}"
                )
                self._start_topology_mesh_async(
                    fallback_conceptual,
                    backend_name,
                    default_cell_type,
                    self._topology_mesh_options,
                    run_mode="fallback-no-constraints",
                )
                fallback_restarted = True
                return
            except Exception as fallback_exc:
                self._log(
                    "mesh> fallback-fail "
                    f"backend={backend_name} elapsed={elapsed_str} error={fallback_exc}"
                )
        self.topo_status_lbl.setText(err_txt)
        self._log(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
    except Exception as exc:
        self.topo_status_lbl.setText(f"Topology meshing failed: {exc}")
        self._log(f"mesh> fail backend={backend_name} mode={run_mode} elapsed={elapsed_str} error={exc}")
    finally:
        if not fallback_restarted:
            self._set_topology_mesh_busy(False)
        _cleanup_topology_mesh_checkpoint(self)





def _import_mesh_from_layers(self):
    if not _HAVE_QGIS_CORE:
        return
    nodes_layer = self._combo_layer(self.nodes_layer_combo, "vector")
    cells_layer = self._combo_layer(self.cells_layer_combo, "vector")
    if nodes_layer is None or cells_layer is None:
        self._log("Select both nodes and cells vector layers.")
        return

    nodes_by_id: Dict[int, Tuple[float, float, float]] = {}
    auto_id = 0
    for ft in nodes_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        pt = geom.asPoint()
        nid = ft["node_id"] if "node_id" in nodes_layer.fields().names() else None
        if nid is None:
            nid = auto_id
            auto_id += 1
        try:
            nid_i = int(nid)
        except Exception as e:
            self._log(f"[ERROR] import node_id int conversion: {e}")
            continue
        z = 0.0
        if "bed_z" in nodes_layer.fields().names():
            try:
                z = float(ft["bed_z"])
            except Exception as e:
                self._log(f"[ERROR] import bed_z float conversion: {e}")
                z = 0.0
        nodes_by_id[nid_i] = (float(pt.x()), float(pt.y()), z)

    if not nodes_by_id:
        self._log("No valid node features found in selected nodes layer.")
        return

    node_ids = sorted(nodes_by_id.keys())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    node_x = np.array([nodes_by_id[nid][0] for nid in node_ids], dtype=np.float64)
    node_y = np.array([nodes_by_id[nid][1] for nid in node_ids], dtype=np.float64)
    node_z = np.array([nodes_by_id[nid][2] for nid in node_ids], dtype=np.float64)

    coord_to_idx = {
        (round(node_x[i], 9), round(node_y[i], 9)): i for i in range(node_x.shape[0])
    }

    face_list: List[List[int]] = []
    tri_list: List[int] = []
    cell_type_vals: List[str] = []
    region_vals: List[int] = []
    size_vals: List[float] = []

    def _parse_face_node_ids(value: object) -> List[int]:
        txt = str(value or "").strip()
        if not txt:
            return []
        out: List[int] = []
        for part in txt.replace(";", ",").split(","):
            p = part.strip()
            if not p:
                continue
            try:
                out.append(id_to_idx[int(p)])
            except Exception as e:
                self._log(f"[ERROR] face node id lookup: {e}")
                continue
        return out

    cell_field_names = set(cells_layer.fields().names())
    for ft in cells_layer.getFeatures():
        ids: List[int] = []

        if "node_ids" in cell_field_names:
            try:
                ids = _parse_face_node_ids(ft["node_ids"])
            except Exception as e:
                self._log(f"[ERROR] face node ids parse: {e}")
                ids = []

        if not ids:
            n_keys = sorted(k for k in cell_field_names if len(k) >= 2 and k[0] == "n" and k[1:].isdigit())
            raw_ids: List[int] = []
            for key in n_keys:
                try:
                    v = ft[key]
                    if v is None:
                        continue
                    raw_ids.append(int(v))
                except Exception as e:
                    self._log(f"[ERROR] n_key {key} int conversion: {e}")
                    continue
            if len(raw_ids) >= 3:
                try:
                    ids = [id_to_idx[v] for v in raw_ids]
                except Exception as e:
                    self._log(f"[ERROR] raw_ids to idx lookup: {e}")
                    ids = []

        if not ids:
            geom = ft.geometry()
            if geom is None or geom.isEmpty():
                continue
            poly = geom.asPolygon()
            if not poly or not poly[0]:
                continue
            ring = poly[0]
            verts: List[int] = []
            for p in ring[:-1]:
                key = (round(float(p.x()), 9), round(float(p.y()), 9))
                if key in coord_to_idx:
                    verts.append(coord_to_idx[key])
            uniq: List[int] = []
            for vid in verts:
                if vid not in uniq:
                    uniq.append(vid)
            ids = uniq

        if len(ids) >= 2 and ids[0] == ids[-1]:
            ids = ids[:-1]
        uniq_ids: List[int] = []
        for nid in ids:
            if nid not in uniq_ids:
                uniq_ids.append(int(nid))
        ids = uniq_ids
        if len(ids) < 3:
            continue

        face_list.append(ids)
        for k in range(1, len(ids) - 1):
            tri_list.extend([int(ids[0]), int(ids[k]), int(ids[k + 1])])

        ctype = ""
        if "cell_type" in cell_field_names:
            try:
                ctype = str(ft["cell_type"] or "").strip().lower()
            except Exception as e:
                self._log(f"[ERROR] cell_type field read: {e}")
                ctype = ""
        if not ctype:
            ctype = "quadrilateral" if len(ids) == 4 else "triangular"
        cell_type_vals.append(ctype)

        reg_v = -1
        if "region_id" in cell_field_names:
            try:
                reg_v = int(ft["region_id"])
            except Exception as e:
                self._log(f"[ERROR] region_id int conversion: {e}")
                reg_v = -1
        region_vals.append(reg_v)

        ts_v = 0.0
        if "target_size" in cell_field_names:
            try:
                ts_v = float(ft["target_size"])
            except Exception as e:
                self._log(f"[ERROR] target_size float conversion: {e}")
                ts_v = 0.0
        size_vals.append(ts_v)

    if not face_list:
        self._log("No valid polygon cells found in selected cells layer.")
        return

    cell_nodes = np.array(tri_list, dtype=np.int32)
    if cell_nodes.size % 3 != 0:
        cell_nodes = cell_nodes[: (cell_nodes.size // 3) * 3]

    face_offsets = [0]
    face_nodes_flat: List[int] = []
    for ids in face_list:
        face_nodes_flat.extend(ids)
        face_offsets.append(face_offsets[-1] + len(ids))
    cell_face_nodes = np.asarray(face_nodes_flat, dtype=np.int32)
    cell_face_offsets = np.asarray(face_offsets, dtype=np.int32)

    if node_x.size >= 2:
        lx = float(np.max(node_x) - np.min(node_x))
        ly = float(np.max(node_y) - np.min(node_y))
    else:
        lx, ly = 1.0, 1.0

    self._mesh_data = {
        "nx": np.array(max(2, int(round(np.sqrt(node_x.size))))),
        "ny": np.array(max(2, int(round(np.sqrt(node_x.size))))),
        "lx": np.array(max(lx, 1.0)),
        "ly": np.array(max(ly, 1.0)),
        "node_x": node_x,
        "node_y": node_y,
        "node_z": node_z,
        "cell_nodes": cell_nodes,
        "cell_face_offsets": cell_face_offsets,
        "cell_face_nodes": cell_face_nodes,
        "cell_type": np.asarray(cell_type_vals, dtype=object),
        "region_id": np.asarray(region_vals, dtype=np.int32),
        "target_size": np.asarray(size_vals, dtype=np.float64),
    }
    if hasattr(self, "_reset_runtime_snapshot_overlay_cache"):
        self._reset_runtime_snapshot_overlay_cache("mesh imported from selected layers")
    n_faces = int(max(0, cell_face_offsets.size - 1))
    n_tris = int(cell_nodes.size // 3)
    self.mesh_info_lbl.setText(f"Loaded map mesh: nodes={node_x.size}, faces={n_faces}, triangles={n_tris}")
    self.layer_status_lbl.setText("Mesh loaded from selected map layers.")
    self._log(f"Imported mesh from map layers: nodes={node_x.size}, faces={n_faces}, triangles={n_tris}")
    self._result_data = None
    self.view_mode_combo.setCurrentText("Mesh")
    self._refresh_plot()





