from __future__ import annotations

# NOTE:
# These extracted methods intentionally depend on symbols defined in
# swe2d_workbench_qt_impl. They are imported lazily by delegate wrappers to
# avoid import-cycle issues during module initialization.
from swe2d_workbench_qt_impl import *  # type: ignore F401,F403

def _bind_model_tab_hydrology_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
    def _ensure_row(label: str, widget: QtWidgets.QWidget) -> None:
        if param_form.indexOf(widget) >= 0:
            return
        param_form.addRow(label, widget)

    def _find_or_create_double_spin(name: str, label: str) -> QtWidgets.QDoubleSpinBox:
        w = model_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
        if w is None:
            w = QtWidgets.QDoubleSpinBox()
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    def _find_or_create_spin(name: str, label: str) -> QtWidgets.QSpinBox:
        w = model_tab_page.findChild(QtWidgets.QSpinBox, name)
        if w is None:
            w = QtWidgets.QSpinBox()
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    def _find_or_create_check(name: str, label: str, text: str) -> QtWidgets.QCheckBox:
        w = model_tab_page.findChild(QtWidgets.QCheckBox, name)
        if w is None:
            w = QtWidgets.QCheckBox(text)
            w.setObjectName(name)
        if not str(w.text() or "").strip():
            w.setText(text)
        _ensure_row(label, w)
        return w

    def _find_or_create_combo(name: str, label: str) -> QtWidgets.QComboBox:
        w = model_tab_page.findChild(QtWidgets.QComboBox, name)
        if w is None:
            w = QtWidgets.QComboBox()
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    self.max_rel_depth_increase_spin = _find_or_create_double_spin(
        "max_rel_depth_increase_spin", "Max rel depth increase:"
    )
    self.max_rel_depth_increase_spin.setRange(0.0, 1000.0)
    self.max_rel_depth_increase_spin.setDecimals(3)
    self.max_rel_depth_increase_spin.setValue(2.0)
    self.max_rel_depth_increase_spin.setToolTip(
        "Per-step depth growth limiter on GPU update:\n"
        "h_new <= h_old + factor * max(h_old, h_min).\n"
        "Lower values are more robust near advancing wet/dry fronts."
    )

    self.max_source_depth_step_spin = _find_or_create_double_spin(
        "max_source_depth_step_spin", "Max source dh/step:"
    )
    self.max_source_depth_step_spin.setRange(0.0, 10.0)
    self.max_source_depth_step_spin.setDecimals(6)
    self.max_source_depth_step_spin.setValue(0.0)
    self.max_source_depth_step_spin.setToolTip(
        "Absolute cap on positive source-driven depth increase per step (model units).\n"
        "0 disables the cap. Useful for suppressing rain/CN impulse spikes."
    )

    self.max_source_rate_spin = _find_or_create_double_spin("max_source_rate_spin", "Max source rate:")
    self.max_source_rate_spin.setRange(0.0, 100.0)
    self.max_source_rate_spin.setDecimals(6)
    self.max_source_rate_spin.setValue(0.0)
    self.max_source_rate_spin.setToolTip(
        "Cap on positive net source rate (model units per second).\n"
        "0 disables the cap. Applies before per-step depth update."
    )

    self.extreme_rain_mode_chk = _find_or_create_check("extreme_rain_mode_chk", "Extreme rain mode:", "Enable")
    self.extreme_rain_mode_chk.setChecked(False)
    self.extreme_rain_mode_chk.setToolTip(
        "Adaptive source-CFL limiter for extreme rainfall/source events.\n"
        "When enabled, positive source terms are reduced using an equivalent\n"
        "substepping factor so dt*source remains bounded by beta*h_ref."
    )

    self.source_cfl_beta_spin = _find_or_create_double_spin("source_cfl_beta_spin", "Source CFL beta:")
    self.source_cfl_beta_spin.setRange(0.01, 2.0)
    self.source_cfl_beta_spin.setDecimals(3)
    self.source_cfl_beta_spin.setSingleStep(0.05)
    self.source_cfl_beta_spin.setValue(0.25)
    self.source_cfl_beta_spin.setToolTip(
        "Target source-CFL beta in dt*source <= beta*h_ref.\n"
        "Lower beta is more conservative."
    )

    self.source_max_substeps_spin = _find_or_create_spin("source_max_substeps_spin", "Source max substeps:")
    self.source_max_substeps_spin.setRange(1, 512)
    self.source_max_substeps_spin.setValue(16)
    self.source_max_substeps_spin.setToolTip(
        "Maximum equivalent source substeps used by adaptive source limiter."
    )

    self.source_true_subcycling_chk = _find_or_create_check(
        "source_true_subcycling_chk", "True source subcycling:", "Enable"
    )
    self.source_true_subcycling_chk.setChecked(False)
    self.source_true_subcycling_chk.setToolTip(
        "Apply true source subcycling (real sub-iterations over dt) instead of\n"
        "equivalent one-shot source scaling."
    )

    self.source_imex_split_chk = _find_or_create_check("source_imex_split_chk", "IMEX source split:", "Enable")
    self.source_imex_split_chk.setChecked(False)
    self.source_imex_split_chk.setToolTip(
        "IMEX-style split: apply flux update first, then source/friction subcycling.\n"
        "Most useful when true source subcycling is enabled."
    )

    self.source_stage_coupled_imex_rk2_chk = _find_or_create_check(
        "source_stage_coupled_imex_rk2_chk", "Stage-coupled IMEX-RK2 sources:", "Enable"
    )
    self.source_stage_coupled_imex_rk2_chk.setChecked(False)
    self.source_stage_coupled_imex_rk2_chk.setToolTip(
        "Stage-coupled IMEX-RK2 for external coupling sources (drainage/structures).\n"
        "Runs a predictor/corrector source update each step (GPU native injection path).\n"
        "Best for stiff coupling; costs extra compute per step."
    )

    self.shallow_damping_depth_spin = _find_or_create_double_spin(
        "shallow_damping_depth_spin", "Shallow damping depth:"
    )
    self.shallow_damping_depth_spin.setRange(1.0e-8, 10.0)
    self.shallow_damping_depth_spin.setDecimals(6)
    self.shallow_damping_depth_spin.setValue(1.0e-4)
    self.shallow_damping_depth_spin.setToolTip(
        "Depth threshold for smooth momentum damping in shallow cells."
    )

    self.shallow_front_recon_fallback_chk = _find_or_create_check(
        "shallow_front_recon_fallback_chk", "Shallow-front recon fallback:", "Enable"
    )
    self.shallow_front_recon_fallback_chk.setChecked(True)
    self.shallow_front_recon_fallback_chk.setToolTip(
        "If enabled, force first-order reconstruction on shallow wet/dry-front\n"
        "edge pairs to improve stability for higher-order schemes."
    )

    self.front_flux_damping_spin = _find_or_create_double_spin("front_flux_damping_spin", "Front flux damping:")
    self.front_flux_damping_spin.setRange(0.0, 1.0)
    self.front_flux_damping_spin.setDecimals(2)
    self.front_flux_damping_spin.setSingleStep(0.05)
    self.front_flux_damping_spin.setValue(0.5)
    self.front_flux_damping_spin.setToolTip(
        "Momentum-flux scale factor applied to edges on the wet/dry front.\n"
        "0.0 = fully damp momentum at the front (most stable, some diffusion).\n"
        "1.0 = no damping (default HLLC).\n"
        "0.5 is a good starting value for oscillating fronts."
    )

    self.active_set_hysteresis_chk = _find_or_create_check(
        "active_set_hysteresis_chk", "Active-set hysteresis:", "Enable"
    )
    self.active_set_hysteresis_chk.setChecked(True)
    self.active_set_hysteresis_chk.setToolTip(
        "Keep cells active for one extra step after they dry below h_min.\n"
        "Prevents rapid oscillatory wet/dry switching at the advancing front.\n"
        "Has negligible performance overhead."
    )

    self.depth_cap_spin = _find_or_create_double_spin("depth_cap_spin", "Depth cap:")
    self.depth_cap_spin.setRange(0.001, 1.0e7)
    self.depth_cap_spin.setDecimals(3)
    self.depth_cap_spin.setValue(1.0e6)
    self.depth_cap_spin.setToolTip("Absolute depth cap for robustness.")

    self.momentum_cap_min_speed_spin = _find_or_create_double_spin(
        "momentum_cap_min_speed_spin", "Momentum cap min speed:"
    )
    self.momentum_cap_min_speed_spin.setRange(0.1, 1.0e4)
    self.momentum_cap_min_speed_spin.setDecimals(3)
    self.momentum_cap_min_speed_spin.setValue(50.0)
    self.momentum_cap_min_speed_spin.setToolTip(
        "Minimum speed floor used by momentum clipping."
    )

    self.momentum_cap_celerity_mult_spin = _find_or_create_double_spin(
        "momentum_cap_celerity_mult_spin", "Momentum cap celerity mult:"
    )
    self.momentum_cap_celerity_mult_spin.setRange(0.1, 1000.0)
    self.momentum_cap_celerity_mult_spin.setDecimals(3)
    self.momentum_cap_celerity_mult_spin.setValue(20.0)
    self.momentum_cap_celerity_mult_spin.setToolTip(
        "Momentum clipping speed cap multiplier on sqrt(g*h)."
    )

    self.max_inv_area_spin = _find_or_create_double_spin("max_inv_area_spin", "Max inv area:")
    self.max_inv_area_spin.setRange(1.0, 1.0e12)
    self.max_inv_area_spin.setDecimals(1)
    self.max_inv_area_spin.setValue(1.0e6)
    self.max_inv_area_spin.setToolTip(
        "Cap on inverse cell area used in flux and update kernels."
    )

    self.cfl_lambda_cap_spin = _find_or_create_double_spin("cfl_lambda_cap_spin", "CFL lambda cap:")
    self.cfl_lambda_cap_spin.setRange(1.0, 1.0e12)
    self.cfl_lambda_cap_spin.setDecimals(1)
    self.cfl_lambda_cap_spin.setValue(1.0e6)
    self.cfl_lambda_cap_spin.setToolTip(
        "Cap on local CFL lambda used in dt reduction and diagnostics."
    )

    self.rain_rate_spin = _find_or_create_double_spin("rain_rate_spin", "Rain rate:")
    self.rain_rate_spin.setRange(0.0, 2000.0)
    self.rain_rate_spin.setDecimals(3)
    self.rain_rate_spin.setValue(0.0)
    self.rain_rate_spin.setSuffix(" mm/hr")

    self.cn_default_spin = _find_or_create_double_spin("cn_default_spin", "Default CN:")
    self.cn_default_spin.setRange(1.0, 100.0)
    self.cn_default_spin.setDecimals(1)
    self.cn_default_spin.setValue(75.0)

    self.ia_ratio_spin = _find_or_create_double_spin("ia_ratio_spin", "SCS Ia/S ratio:")
    self.ia_ratio_spin.setRange(0.0, 1.0)
    self.ia_ratio_spin.setDecimals(3)
    self.ia_ratio_spin.setSingleStep(0.01)
    self.ia_ratio_spin.setValue(0.2)
    self.ia_ratio_spin.setToolTip(
        "Initial abstraction ratio (Ia/S) for SCS Curve Number losses.\n"
        "Typical default is 0.20."
    )

    self.use_spatial_rain_cn_chk = _find_or_create_check(
        "use_spatial_rain_cn_chk",
        "Spatial rainfall:",
        "Use Thiessen gage rainfall when layers are available",
    )
    self.use_spatial_rain_cn_chk.setChecked(True)

    self.infiltration_method_combo = _find_or_create_combo("infiltration_method_combo", "Infiltration method:")
    prev_data = self.infiltration_method_combo.currentData()
    prev_text = self.infiltration_method_combo.currentText()
    self.infiltration_method_combo.blockSignals(True)
    try:
        self.infiltration_method_combo.clear()
        self.infiltration_method_combo.addItem("SCS Curve Number", "scs_cn")
        self.infiltration_method_combo.addItem("None (no infiltration)", "none")
        idx = self.infiltration_method_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.infiltration_method_combo.findText(prev_text)
        if idx < 0:
            idx = self.infiltration_method_combo.findData("scs_cn")
        if idx >= 0:
            self.infiltration_method_combo.setCurrentIndex(idx)
    finally:
        self.infiltration_method_combo.blockSignals(False)
    self.infiltration_method_combo.setToolTip(
        "Infiltration/loss method applied to rainfall before it enters the 2D surface as runoff.\n"
        "SCS Curve Number: NRCS CN abstraction (default).\n"
        "None: all rainfall becomes direct runoff - no abstraction."
    )

    self.storm_area_layer_combo = _find_or_create_combo("storm_area_layer_combo", "Storm area layer (optional):")
    prev_data = self.storm_area_layer_combo.currentData()
    prev_text = self.storm_area_layer_combo.currentText()
    self.storm_area_layer_combo.blockSignals(True)
    try:
        self.storm_area_layer_combo.clear()
        self.storm_area_layer_combo.addItem("(none)", None)
        idx = self.storm_area_layer_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.storm_area_layer_combo.findText(prev_text)
        if idx >= 0:
            self.storm_area_layer_combo.setCurrentIndex(idx)
    finally:
        self.storm_area_layer_combo.blockSignals(False)

    self.rain_boundary_buffer_rings_spin = _find_or_create_spin(
        "rain_boundary_buffer_rings_spin", "Rain boundary buffer rings:"
    )
    self.rain_boundary_buffer_rings_spin.setRange(0, 10)
    self.rain_boundary_buffer_rings_spin.setValue(1)
    self.rain_boundary_buffer_rings_spin.setToolTip(
        "Boundary rain buffer rings (Thiessen + CN forcing).\n"
        "0: no exclusion. 1: exclude boundary cells.\n"
        "N>1: also exclude N-1 inward neighbor rings."
    )



def _bind_model_tab_3d_patch_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
    patch_form = model_tab_page.findChild(QtWidgets.QFormLayout, "patch_3d_form") or param_form

    def _ensure_row(label: str, widget: QtWidgets.QWidget) -> None:
        if patch_form.indexOf(widget) >= 0:
            return
        patch_form.addRow(label, widget)

    def _ensure_widget_row(widget: QtWidgets.QWidget) -> None:
        if patch_form.indexOf(widget) >= 0:
            return
        patch_form.addRow(widget)

    def _find_or_create_check(name: str, label: str, text: str) -> QtWidgets.QCheckBox:
        w = model_tab_page.findChild(QtWidgets.QCheckBox, name)
        if w is None:
            w = QtWidgets.QCheckBox(text)
            w.setObjectName(name)
        if not str(w.text() or "").strip():
            w.setText(text)
        _ensure_row(label, w)
        return w

    def _find_or_create_combo(name: str, label: str) -> QtWidgets.QComboBox:
        w = model_tab_page.findChild(QtWidgets.QComboBox, name)
        if w is None:
            w = QtWidgets.QComboBox()
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    def _find_or_create_double_spin(name: str, label: str) -> QtWidgets.QDoubleSpinBox:
        w = model_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
        if w is None:
            w = QtWidgets.QDoubleSpinBox()
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    def _find_or_create_line_edit(name: str, label: str) -> QtWidgets.QLineEdit:
        w = model_tab_page.findChild(QtWidgets.QLineEdit, name)
        if w is None:
            w = QtWidgets.QLineEdit()
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
        w = model_tab_page.findChild(QtWidgets.QPushButton, name)
        if w is None:
            w = QtWidgets.QPushButton(text)
            w.setObjectName(name)
        if not str(w.text() or "").strip():
            w.setText(text)
        _ensure_widget_row(w)
        return w

    def _find_or_create_label(name: str, text: str) -> QtWidgets.QLabel:
        w = model_tab_page.findChild(QtWidgets.QLabel, name)
        if w is None:
            w = QtWidgets.QLabel(text)
            w.setObjectName(name)
        if not str(w.text() or "").strip():
            w.setText(text)
        _ensure_widget_row(w)
        return w

    self.experimental_3d_mode_chk = _find_or_create_check(
        "experimental_3d_mode_chk", "3D patch execution mode:", "Run 3D patch solver (GPU)"
    )
    self.experimental_3d_mode_chk.setChecked(False)
    self.experimental_3d_mode_chk.setToolTip(
        "Experimental 3D patch solver mode for validation/smoke testing.\n"
        "Enables SINGLE_PHASE_FREE_SURFACE_VOF and optional 2D-3D coupling."
    )
    self._experimental_3d_mode_supported = bool(
        SolverModelOptions is not None
        and SWE2DThreeDSolverModel is not None
        and SWE2DThreeDCouplingMode is not None
    )

    self.experimental_3d_coupling_mode_combo = _find_or_create_combo(
        "experimental_3d_coupling_mode_combo", "3D patch coupling mode:"
    )
    prev_data = self.experimental_3d_coupling_mode_combo.currentData()
    prev_text = self.experimental_3d_coupling_mode_combo.currentText()
    self.experimental_3d_coupling_mode_combo.blockSignals(True)
    try:
        self.experimental_3d_coupling_mode_combo.clear()
        if SWE2DThreeDCouplingMode is not None:
            self.experimental_3d_coupling_mode_combo.addItem(
                "Off (uncoupled)", int(SWE2DThreeDCouplingMode.OFF)
            )
            self.experimental_3d_coupling_mode_combo.addItem(
                "One-way (2D -> 3D)", int(SWE2DThreeDCouplingMode.ONE_WAY_2D_TO_3D)
            )
            self.experimental_3d_coupling_mode_combo.addItem(
                "Two-way (2D <-> 3D)", int(SWE2DThreeDCouplingMode.TWO_WAY_2D_3D)
            )
        else:
            self.experimental_3d_coupling_mode_combo.addItem("Off (uncoupled)", 0)
            self.experimental_3d_coupling_mode_combo.addItem("One-way (2D -> 3D)", 1)
            self.experimental_3d_coupling_mode_combo.addItem("Two-way (2D <-> 3D)", 2)
        idx = self.experimental_3d_coupling_mode_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.experimental_3d_coupling_mode_combo.findText(prev_text)
        if idx < 0:
            idx = 0
        if idx >= 0:
            self.experimental_3d_coupling_mode_combo.setCurrentIndex(idx)
    finally:
        self.experimental_3d_coupling_mode_combo.blockSignals(False)
    self.experimental_3d_coupling_mode_combo.setToolTip(
        "Select 2D-3D exchange mode for the 3D patch runtime.\n"
        "When coupling is ON, the GUI auto-builds and uploads a boundary-edge interface contract."
    )

    self.experimental_3d_patch_face_len_x_spin = _find_or_create_double_spin(
        "experimental_3d_patch_face_len_x_spin", "3D patch target face length x:"
    )
    self.experimental_3d_patch_face_len_x_spin.setRange(1.0e-4, 1.0e6)
    self.experimental_3d_patch_face_len_x_spin.setDecimals(6)
    self.experimental_3d_patch_face_len_x_spin.setSingleStep(0.5)
    self.experimental_3d_patch_face_len_x_spin.setValue(5.0)
    self.experimental_3d_patch_face_len_x_spin.setToolTip(
        "Target x-face length for 3D patch cells (model units).\n"
        "Runtime resolves nx = ceil((xmax-xmin)/target_len_x)."
    )

    self.experimental_3d_patch_face_len_y_spin = _find_or_create_double_spin(
        "experimental_3d_patch_face_len_y_spin", "3D patch target face length y:"
    )
    self.experimental_3d_patch_face_len_y_spin.setRange(1.0e-4, 1.0e6)
    self.experimental_3d_patch_face_len_y_spin.setDecimals(6)
    self.experimental_3d_patch_face_len_y_spin.setSingleStep(0.5)
    self.experimental_3d_patch_face_len_y_spin.setValue(5.0)
    self.experimental_3d_patch_face_len_y_spin.setToolTip(
        "Target y-face length for 3D patch cells (model units).\n"
        "Runtime resolves ny = ceil((ymax-ymin)/target_len_y)."
    )

    self.experimental_3d_patch_face_len_z_spin = _find_or_create_double_spin(
        "experimental_3d_patch_face_len_z_spin", "3D patch target face length z:"
    )
    self.experimental_3d_patch_face_len_z_spin.setRange(1.0e-4, 1.0e6)
    self.experimental_3d_patch_face_len_z_spin.setDecimals(6)
    self.experimental_3d_patch_face_len_z_spin.setSingleStep(0.25)
    self.experimental_3d_patch_face_len_z_spin.setValue(2.0)
    self.experimental_3d_patch_face_len_z_spin.setToolTip(
        "Target z-face length for 3D patch cells (model units).\n"
        "Runtime resolves nz = ceil((zmax-zmin)/target_len_z)."
    )

    self.experimental_3d_patch_xmin_edit = _find_or_create_line_edit(
        "experimental_3d_patch_xmin_edit", "3D patch x min:"
    )
    self.experimental_3d_patch_xmax_edit = _find_or_create_line_edit(
        "experimental_3d_patch_xmax_edit", "3D patch x max:"
    )
    self.experimental_3d_patch_ymin_edit = _find_or_create_line_edit(
        "experimental_3d_patch_ymin_edit", "3D patch y min:"
    )
    self.experimental_3d_patch_ymax_edit = _find_or_create_line_edit(
        "experimental_3d_patch_ymax_edit", "3D patch y max:"
    )
    self.experimental_3d_patch_zmin_edit = _find_or_create_line_edit(
        "experimental_3d_patch_zmin_edit", "3D patch z min:"
    )
    self.experimental_3d_patch_zmax_edit = _find_or_create_line_edit(
        "experimental_3d_patch_zmax_edit", "3D patch z max:"
    )
    for _w in (
        self.experimental_3d_patch_xmin_edit,
        self.experimental_3d_patch_xmax_edit,
        self.experimental_3d_patch_ymin_edit,
        self.experimental_3d_patch_ymax_edit,
        self.experimental_3d_patch_zmin_edit,
        self.experimental_3d_patch_zmax_edit,
    ):
        _w.setPlaceholderText("auto from mesh")
    self.experimental_3d_patch_zmin_edit.setPlaceholderText("auto from terrain")

    self.experimental_3d_patch_set_roi_btn = _find_or_create_button(
        "experimental_3d_patch_set_roi_btn", "Set ROI From Current Mesh"
    )
    self.experimental_3d_patch_set_roi_btn.setToolTip(
        "Populate x/y/z min-max fields from the current 2D mesh extents.\n"
        "Used only when Experimental 3D patch mode is enabled."
    )
    try:
        self.experimental_3d_patch_set_roi_btn.clicked.disconnect(self._set_3d_patch_roi_from_mesh)
    except Exception:
        pass
    self.experimental_3d_patch_set_roi_btn.clicked.connect(self._set_3d_patch_roi_from_mesh)

    self.experimental_3d_patch_hint_lbl = _find_or_create_label(
        "experimental_3d_patch_hint_lbl",
        "3D patch ROI/resolution override (experimental): resolution is driven by target face lengths; "
        "leave min/max empty to auto-use mesh extents; z-min is terrain-driven when a DEM is available.",
    )
    self.experimental_3d_patch_hint_lbl.setWordWrap(True)

    self._experimental_3d_bc_widget_attrs = []
    self._experimental_3d_bc_signal_specs = []
    self.experimental_3d_patch_bc_widget = model_tab_page.findChild(
        QtWidgets.QWidget, "experimental_3d_patch_bc_widget"
    )
    if self.experimental_3d_patch_bc_widget is None:
        self.experimental_3d_patch_bc_widget = QtWidgets.QWidget()
        self.experimental_3d_patch_bc_widget.setObjectName("experimental_3d_patch_bc_widget")
    _ensure_row("3D patch face BCs:", self.experimental_3d_patch_bc_widget)

    existing_layout = self.experimental_3d_patch_bc_widget.layout()
    if isinstance(existing_layout, QtWidgets.QGridLayout):
        while existing_layout.count():
            item = existing_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        bc_grid = existing_layout
    else:
        bc_grid = QtWidgets.QGridLayout(self.experimental_3d_patch_bc_widget)
    bc_grid.setContentsMargins(0, 0, 0, 0)
    bc_grid.setHorizontalSpacing(4)
    bc_grid.setVerticalSpacing(2)

    bc_headers = ["Face", "Mode", "Q", "U", "V", "W", "VOF", "P"]
    for col, label in enumerate(bc_headers):
        hdr = QtWidgets.QLabel(label)
        hdr.setStyleSheet("font-weight: 600;")
        bc_grid.addWidget(hdr, 0, col)

    for row, face in enumerate(_SWE3D_PATCH_FACES, start=1):
        face_key = str(face).lower()
        bc_grid.addWidget(QtWidgets.QLabel(face), row, 0)

        mode_combo = QtWidgets.QComboBox()
        for mode_label, mode_value in _SWE3D_BC_MODE_OPTIONS:
            mode_combo.addItem(str(mode_label), int(mode_value))
        mode_combo.setCurrentIndex(0)
        mode_combo.setToolTip(
            "Boundary mode for this 3D patch face "
            "(0=Wall, 1=Inflow(U/V/W), 2=Outflow(zero-gradient), 3=Free Surface, 4=Volumetric Inlet(Q))."
        )
        mode_attr = f"experimental_3d_bc_{face_key}_mode_combo"
        setattr(self, mode_attr, mode_combo)
        self._experimental_3d_bc_widget_attrs.append(mode_attr)
        self._experimental_3d_bc_signal_specs.append((mode_attr, "currentIndexChanged"))
        bc_grid.addWidget(mode_combo, row, 1)

        for col, field_name in enumerate(("q", "u", "v", "w", "vof", "p"), start=2):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(6)
            if field_name == "q":
                spin.setRange(-1.0e9, 1.0e9)
                spin.setSingleStep(1.0)
            elif field_name == "vof":
                spin.setRange(0.0, 1.0)
                spin.setSingleStep(0.05)
            elif field_name == "p":
                spin.setRange(-1.0e9, 1.0e9)
                spin.setSingleStep(1000.0)
            else:
                spin.setRange(-1.0e6, 1.0e6)
                spin.setSingleStep(0.1)
            spin.setValue(float(_SWE3D_BC_FIELD_DEFAULTS.get(field_name, 0.0)))
            spin.setMaximumWidth(100)
            if field_name == "q":
                spin.setToolTip(
                    f"Prescribed volumetric flow rate Q [m^3/s] for {face} when mode=Volumetric Inlet (Q)."
                )
            else:
                spin.setToolTip(
                    f"Prescribed {field_name.upper()} state for {face} when mode uses boundary state input."
                )
            field_attr = f"experimental_3d_bc_{face_key}_{field_name}_spin"
            setattr(self, field_attr, spin)
            self._experimental_3d_bc_widget_attrs.append(field_attr)
            self._experimental_3d_bc_signal_specs.append((field_attr, "valueChanged"))
            bc_grid.addWidget(spin, row, col)

    self.experimental_3d_patch_bc_hint_lbl = _find_or_create_label(
        "experimental_3d_patch_bc_hint_lbl",
        "3D face BCs map to BACKWATER_SWE3D_BC_<FACE>_<FIELD> env overrides; "
        "Outflow is zero-gradient, and Volumetric Inlet uses Q [m^3/s] for the face-normal inflow target.",
    )
    self.experimental_3d_patch_bc_hint_lbl.setWordWrap(True)

    if not self._experimental_3d_mode_supported:
        self.experimental_3d_mode_chk.setChecked(False)
        self.experimental_3d_mode_chk.setEnabled(False)
        self.experimental_3d_coupling_mode_combo.setEnabled(False)
        self.experimental_3d_mode_chk.setText("3D patch solver unavailable in this runtime")
        self.experimental_3d_mode_chk.setToolTip(
            "3D patch runtime enums (SolverModelOptions / SWE2DThreeD*) are unavailable.\n"
            "This session will run 2D only until the Python runtime imports swe2d_extensions fully."
        )



def _bind_model_tab_3d_subgrid_drainage_controls(
    self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout
) -> None:
    patch_form = model_tab_page.findChild(QtWidgets.QFormLayout, "patch_3d_form") or param_form

    def _ensure_row(label: str, widget: QtWidgets.QWidget, target_form: Optional[QtWidgets.QFormLayout] = None) -> None:
        form = target_form or patch_form
        if form.indexOf(widget) >= 0:
            return
        form.addRow(label, widget)

    def _ensure_widget_row(widget: QtWidgets.QWidget, target_form: Optional[QtWidgets.QFormLayout] = None) -> None:
        form = target_form or patch_form
        if form.indexOf(widget) >= 0:
            return
        form.addRow(widget)

    def _find_or_create_check(
        name: str,
        label: str,
        text: str,
        target_form: Optional[QtWidgets.QFormLayout] = None,
    ) -> QtWidgets.QCheckBox:
        w = model_tab_page.findChild(QtWidgets.QCheckBox, name)
        if w is None:
            w = QtWidgets.QCheckBox(text)
            w.setObjectName(name)
        if not str(w.text() or "").strip():
            w.setText(text)
        _ensure_row(label, w, target_form)
        return w

    def _find_or_create_combo(
        name: str,
        label: str,
        target_form: Optional[QtWidgets.QFormLayout] = None,
    ) -> QtWidgets.QComboBox:
        w = model_tab_page.findChild(QtWidgets.QComboBox, name)
        if w is None:
            w = QtWidgets.QComboBox()
            w.setObjectName(name)
        _ensure_row(label, w, target_form)
        return w

    def _find_or_create_line_edit(
        name: str,
        label: str,
        text: str = "",
        target_form: Optional[QtWidgets.QFormLayout] = None,
    ) -> QtWidgets.QLineEdit:
        w = model_tab_page.findChild(QtWidgets.QLineEdit, name)
        if w is None:
            w = QtWidgets.QLineEdit(text)
            w.setObjectName(name)
        _ensure_row(label, w, target_form)
        return w

    def _find_or_create_double_spin(
        name: str,
        label: str,
        target_form: Optional[QtWidgets.QFormLayout] = None,
    ) -> QtWidgets.QDoubleSpinBox:
        w = model_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
        if w is None:
            w = QtWidgets.QDoubleSpinBox()
            w.setObjectName(name)
        _ensure_row(label, w, target_form)
        return w

    def _find_or_create_spin(
        name: str,
        label: str,
        target_form: Optional[QtWidgets.QFormLayout] = None,
    ) -> QtWidgets.QSpinBox:
        w = model_tab_page.findChild(QtWidgets.QSpinBox, name)
        if w is None:
            w = QtWidgets.QSpinBox()
            w.setObjectName(name)
        _ensure_row(label, w, target_form)
        return w

    self.experimental_3d_obj_solids_chk = _find_or_create_check(
        "experimental_3d_obj_solids_chk", "3D sub-grid solids:", "Enable"
    )
    self.experimental_3d_obj_solids_chk.setChecked(True)
    self.experimental_3d_obj_solids_chk.setToolTip(
        "Upload static sub-grid geometry tensors (phi/ax/ay/az) before run start.\n"
        "Sources geometry from an OBJ instance point layer and optional terrain DEM solid fill."
    )

    self.experimental_3d_obj_method_combo = _find_or_create_combo(
        "experimental_3d_obj_method_combo", "3D sub-grid method:"
    )
    prev_data = self.experimental_3d_obj_method_combo.currentData()
    prev_text = self.experimental_3d_obj_method_combo.currentText()
    self.experimental_3d_obj_method_combo.blockSignals(True)
    try:
        self.experimental_3d_obj_method_combo.clear()
        self.experimental_3d_obj_method_combo.addItem("Fractional cut-cell (current)", "fractional_cutcell")
        self.experimental_3d_obj_method_combo.addItem("Porosity (Hirt-Nichols/FAVOR-like)", "favor1981_porosity")
        idx = self.experimental_3d_obj_method_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.experimental_3d_obj_method_combo.findText(prev_text)
        if idx < 0:
            idx = 0
        if idx >= 0:
            self.experimental_3d_obj_method_combo.setCurrentIndex(idx)
    finally:
        self.experimental_3d_obj_method_combo.blockSignals(False)
    self.experimental_3d_obj_method_combo.setToolTip(
        "Static-obstacle tensor reconstruction method.\n"
        "Fractional cut-cell: current phi + pair-min face openness.\n"
        "Porosity/FAVOR-like: direct directional face-open sampling."
    )

    self.experimental_3d_obj_layer_combo = _find_or_create_combo(
        "experimental_3d_obj_layer_combo", "3D OBJ instances layer:"
    )
    prev_data = self.experimental_3d_obj_layer_combo.currentData()
    prev_text = self.experimental_3d_obj_layer_combo.currentText()
    self.experimental_3d_obj_layer_combo.blockSignals(True)
    try:
        self.experimental_3d_obj_layer_combo.clear()
        self.experimental_3d_obj_layer_combo.addItem("(none)", None)
        idx = self.experimental_3d_obj_layer_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.experimental_3d_obj_layer_combo.findText(prev_text)
        if idx >= 0:
            self.experimental_3d_obj_layer_combo.setCurrentIndex(idx)
    finally:
        self.experimental_3d_obj_layer_combo.blockSignals(False)

    self.experimental_3d_obj_path_field_edit = _find_or_create_line_edit(
        "experimental_3d_obj_path_field_edit", "3D OBJ path field:", "model_path"
    )
    if not str(self.experimental_3d_obj_path_field_edit.text() or "").strip():
        self.experimental_3d_obj_path_field_edit.setText("model_path")
    self.experimental_3d_obj_path_field_edit.setPlaceholderText("attribute with OBJ file path")

    self.experimental_3d_obj_default_path_edit = _find_or_create_line_edit(
        "experimental_3d_obj_default_path_edit", "3D OBJ fallback path:"
    )
    self.experimental_3d_obj_default_path_edit.setPlaceholderText("fallback OBJ path (optional)")

    self.experimental_3d_obj_scale_field_edit = _find_or_create_line_edit(
        "experimental_3d_obj_scale_field_edit", "3D OBJ scale field:", "scale"
    )
    if not str(self.experimental_3d_obj_scale_field_edit.text() or "").strip():
        self.experimental_3d_obj_scale_field_edit.setText("scale")
    self.experimental_3d_obj_scale_field_edit.setPlaceholderText("optional scale field (1 or sx,sy,sz)")

    self.experimental_3d_obj_yaw_field_edit = _find_or_create_line_edit(
        "experimental_3d_obj_yaw_field_edit", "3D OBJ yaw field:", "yaw_deg"
    )
    if not str(self.experimental_3d_obj_yaw_field_edit.text() or "").strip():
        self.experimental_3d_obj_yaw_field_edit.setText("yaw_deg")
    self.experimental_3d_obj_yaw_field_edit.setPlaceholderText("optional yaw field (degrees)")

    self.experimental_3d_obj_z_offset_field_edit = _find_or_create_line_edit(
        "experimental_3d_obj_z_offset_field_edit", "3D OBJ z-offset field:", "z_offset"
    )
    if not str(self.experimental_3d_obj_z_offset_field_edit.text() or "").strip():
        self.experimental_3d_obj_z_offset_field_edit.setText("z_offset")
    self.experimental_3d_obj_z_offset_field_edit.setPlaceholderText("optional per-instance z offset")

    self.experimental_3d_obj_inside_points_layer_combo = _find_or_create_combo(
        "experimental_3d_obj_inside_points_layer_combo", "3D OBJ outside-point layer:"
    )
    prev_data = self.experimental_3d_obj_inside_points_layer_combo.currentData()
    prev_text = self.experimental_3d_obj_inside_points_layer_combo.currentText()
    self.experimental_3d_obj_inside_points_layer_combo.blockSignals(True)
    try:
        self.experimental_3d_obj_inside_points_layer_combo.clear()
        self.experimental_3d_obj_inside_points_layer_combo.addItem("(none)", None)
        idx = self.experimental_3d_obj_inside_points_layer_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.experimental_3d_obj_inside_points_layer_combo.findText(prev_text)
        if idx >= 0:
            self.experimental_3d_obj_inside_points_layer_combo.setCurrentIndex(idx)
    finally:
        self.experimental_3d_obj_inside_points_layer_combo.blockSignals(False)

    self.experimental_3d_obj_instance_id_field_edit = _find_or_create_line_edit(
        "experimental_3d_obj_instance_id_field_edit", "3D OBJ instance id field:", "instance_id"
    )
    if not str(self.experimental_3d_obj_instance_id_field_edit.text() or "").strip():
        self.experimental_3d_obj_instance_id_field_edit.setText("instance_id")
    self.experimental_3d_obj_instance_id_field_edit.setPlaceholderText("optional OBJ instance id field")

    self.experimental_3d_obj_inside_id_field_edit = _find_or_create_line_edit(
        "experimental_3d_obj_inside_id_field_edit", "3D OBJ outside-point id field:", "instance_id"
    )
    if not str(self.experimental_3d_obj_inside_id_field_edit.text() or "").strip():
        self.experimental_3d_obj_inside_id_field_edit.setText("instance_id")
    self.experimental_3d_obj_inside_id_field_edit.setPlaceholderText("optional outside-point id field")

    self.experimental_3d_obj_inside_z_field_edit = _find_or_create_line_edit(
        "experimental_3d_obj_inside_z_field_edit", "3D OBJ outside-point z field:", "z"
    )
    if not str(self.experimental_3d_obj_inside_z_field_edit.text() or "").strip():
        self.experimental_3d_obj_inside_z_field_edit.setText("z")
    self.experimental_3d_obj_inside_z_field_edit.setPlaceholderText("optional outside-point z field")

    self.experimental_3d_obj_use_terrain_chk = _find_or_create_check(
        "experimental_3d_obj_use_terrain_chk", "3D terrain solid:", "Use terrain layer as bed solid"
    )
    self.experimental_3d_obj_use_terrain_chk.setChecked(True)
    self.experimental_3d_obj_use_terrain_chk.setToolTip(
        "Treat cells below sampled terrain DEM elevation as solid (phi=0)."
    )

    self.experimental_3d_obj_ab_compare_chk = _find_or_create_check(
        "experimental_3d_obj_ab_compare_chk", "3D A/B compare:", "A/B compare methods (startup probe)"
    )
    self.experimental_3d_obj_ab_compare_chk.setChecked(False)
    self.experimental_3d_obj_ab_compare_chk.setToolTip(
        "Run a short pre-run probe on temporary backends to compare fractional cut-cell and FAVOR-like methods.\n"
        "Logs mass drift proxy, max Courant, p_max_abs, and u_rms deltas before the main run starts."
    )

    self.experimental_3d_obj_ab_probe_steps_spin = _find_or_create_spin(
        "experimental_3d_obj_ab_probe_steps_spin", "3D A/B probe steps:"
    )
    self.experimental_3d_obj_ab_probe_steps_spin.setRange(1, 64)
    self.experimental_3d_obj_ab_probe_steps_spin.setValue(8)
    self.experimental_3d_obj_ab_probe_steps_spin.setToolTip(
        "Number of adaptive 3D probe steps used for each obstacle method in A/B compare mode."
    )

    self.experimental_3d_obj_export_obj_chk = _find_or_create_check(
        "experimental_3d_obj_export_obj_chk", "3D export voxel shell OBJ:", "Export voxelized solid shell OBJ"
    )
    self.experimental_3d_obj_export_obj_chk.setChecked(False)
    self.experimental_3d_obj_export_obj_chk.setToolTip(
        "Write the reconstructed solid representation (from phi thresholding) as an OBJ mesh for inspection."
    )

    self.experimental_3d_obj_export_obj_path_edit = _find_or_create_line_edit(
        "experimental_3d_obj_export_obj_path_edit", "3D solid OBJ export path:"
    )
    self.experimental_3d_obj_export_obj_path_edit.setPlaceholderText("optional OBJ output path (auto if empty)")

    self.experimental_3d_geom_sanitize_chk = _find_or_create_check(
        "experimental_3d_geom_sanitize_chk",
        "3D sanitize tensors:",
        "Sanitize upload tensors (clamp/snap tiny phi/area)",
    )
    self.experimental_3d_geom_sanitize_chk.setChecked(True)
    self.experimental_3d_geom_sanitize_chk.setToolTip(
        "Preprocess uploaded phi/ax/ay/az tensors for numerical robustness.\n"
        "Clamps all tensors to [0,1], snaps tiny phi cells to solid, and snaps tiny face-open areas to zero."
    )

    self.experimental_3d_geom_phi_snap_spin = _find_or_create_double_spin(
        "experimental_3d_geom_phi_snap_spin", "3D sanitize phi snap min:"
    )
    self.experimental_3d_geom_phi_snap_spin.setRange(0.0, 1.0)
    self.experimental_3d_geom_phi_snap_spin.setDecimals(6)
    self.experimental_3d_geom_phi_snap_spin.setSingleStep(0.001)
    self.experimental_3d_geom_phi_snap_spin.setValue(0.005)
    self.experimental_3d_geom_phi_snap_spin.setToolTip(
        "If phi < threshold, the cell is snapped to solid (phi=0) during geometry upload.\n"
        "Default is conservative to avoid over-sanitizing valid cut cells."
    )

    self.experimental_3d_geom_area_snap_spin = _find_or_create_double_spin(
        "experimental_3d_geom_area_snap_spin", "3D sanitize area snap min:"
    )
    self.experimental_3d_geom_area_snap_spin.setRange(0.0, 1.0)
    self.experimental_3d_geom_area_snap_spin.setDecimals(6)
    self.experimental_3d_geom_area_snap_spin.setSingleStep(0.001)
    self.experimental_3d_geom_area_snap_spin.setValue(0.01)
    self.experimental_3d_geom_area_snap_spin.setToolTip(
        "If ax/ay/az < threshold, the face-open fraction is snapped to zero during upload.\n"
        "Default is conservative and mainly targets sliver openings."
    )

    self.godunov_mode_combo = _find_or_create_combo(
        "godunov_mode_combo", "GPU solver mode:", param_form
    )
    prev_data = self.godunov_mode_combo.currentData()
    prev_text = self.godunov_mode_combo.currentText()
    self.godunov_mode_combo.blockSignals(True)
    try:
        self.godunov_mode_combo.clear()
        self.godunov_mode_combo.addItem("Current GPU solver", int(GodunovSolverMode.CURRENT_GPU_STEP))
        self.godunov_mode_combo.addItem("Godunov rollout (2nd-order)", int(GodunovSolverMode.GODUNOV_ROLLOUT))
        idx = self.godunov_mode_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.godunov_mode_combo.findText(prev_text)
        if idx < 0:
            idx = 0
        if idx >= 0:
            self.godunov_mode_combo.setCurrentIndex(idx)
    finally:
        self.godunov_mode_combo.blockSignals(False)
    self.godunov_mode_combo.setToolTip(
        "Select the solver implementation used by the GPU path.\n"
        "Current GPU solver: existing production path.\n"
        "Godunov rollout: enables the second-order rollout configuration and\n"
        "keeps the native solver on the migration path for the new FVM mode."
    )

    self.degen_mode_combo = _find_or_create_combo("degen_mode_combo", "Degenerate cell mode:", param_form)
    prev_data = self.degen_mode_combo.currentData()
    prev_text = self.degen_mode_combo.currentText()
    self.degen_mode_combo.blockSignals(True)
    try:
        self.degen_mode_combo.clear()
        for _label, _val in [
            ("None (max_inv_area cap)", 0),
            ("Skip (permanently inactive)", 1),
            ("Repair (neighbor-avg inv_area)", 2),
            ("Merge (redirect flux to owner)", 3),
        ]:
            self.degen_mode_combo.addItem(_label, int(_val))
        idx = self.degen_mode_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.degen_mode_combo.findText(prev_text)
        if idx < 0:
            idx = 0
        if idx >= 0:
            self.degen_mode_combo.setCurrentIndex(idx)
    finally:
        self.degen_mode_combo.blockSignals(False)
    self.degen_mode_combo.setToolTip(
        "Degenerate cell handling mode (cells with area below 1/max_inv_area).\n"
        "None: existing max_inv_area cap in update kernel (default).\n"
        "Skip: permanently exclude degenerate cells from all flux/update.\n"
        "Repair: replace degenerate cell inv_area with neighbor average;\n"
        "  keeps them in physics with sane CFL contribution.\n"
        "Merge: redirect flux accumulation to largest non-degenerate neighbor."
    )

    self.coupling_loop_combo = _find_or_create_combo("coupling_loop_combo", "Coupling loop:", param_form)
    prev_data = self.coupling_loop_combo.currentData()
    prev_text = self.coupling_loop_combo.currentText()
    self.coupling_loop_combo.blockSignals(True)
    try:
        self.coupling_loop_combo.clear()
        self.coupling_loop_combo.addItem("CPU coupling loop (reference)", "cpu")
        self.coupling_loop_combo.addItem("CUDA coupling loop (source assembly)", "cuda")
        idx = self.coupling_loop_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.coupling_loop_combo.findText(prev_text)
        if idx < 0:
            idx = min(1, max(0, self.coupling_loop_combo.count() - 1))
        if idx >= 0:
            self.coupling_loop_combo.setCurrentIndex(idx)
    finally:
        self.coupling_loop_combo.blockSignals(False)
    self.coupling_loop_combo.setToolTip(
        "Select coupling source assembly mode.\n"
        "CPU: Python reference path for drainage/structure source rates.\n"
        "CUDA: uses native CUDA kernel for per-cell source assembly when available;\n"
        "falls back to CPU reference automatically if CUDA binding/device is unavailable."
    )

    self.drainage_solver_mode_combo = _find_or_create_combo(
        "drainage_solver_mode_combo", "Drainage equation set:", param_form
    )
    prev_data = self.drainage_solver_mode_combo.currentData()
    prev_text = self.drainage_solver_mode_combo.currentText()
    self.drainage_solver_mode_combo.blockSignals(True)
    try:
        self.drainage_solver_mode_combo.clear()
        self.drainage_solver_mode_combo.addItem("EGL (Bernoulli + minor losses)", int(0))
        self.drainage_solver_mode_combo.addItem("Diffusion wave", int(1))
        self.drainage_solver_mode_combo.addItem("Dynamic Saint-Venant", int(2))
        idx = self.drainage_solver_mode_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.drainage_solver_mode_combo.findText(prev_text)
        if idx < 0:
            idx = 0
        if idx >= 0:
            self.drainage_solver_mode_combo.setCurrentIndex(idx)
    finally:
        self.drainage_solver_mode_combo.blockSignals(False)
    self.drainage_solver_mode_combo.setToolTip(
        "Drainage 1D equation set.\n"
        "EGL: Bernoulli + Manning + minor losses.\n"
        "Diffusion: slope-driven Manning flow.\n"
        "Dynamic: semi-implicit Saint-Venant momentum update."
    )

    self.drainage_backend_combo = _find_or_create_combo(
        "drainage_backend_combo", "Drainage solver backend:", param_form
    )
    prev_data = self.drainage_backend_combo.currentData()
    prev_text = self.drainage_backend_combo.currentText()
    self.drainage_backend_combo.blockSignals(True)
    try:
        self.drainage_backend_combo.clear()
        self.drainage_backend_combo.addItem("CPU drainage solver (reference)", "cpu")
        self.drainage_backend_combo.addItem("GPU drainage solver (CUDA)", "gpu")
        idx = self.drainage_backend_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.drainage_backend_combo.findText(prev_text)
        if idx < 0:
            idx = min(1, max(0, self.drainage_backend_combo.count() - 1))
        if idx >= 0:
            self.drainage_backend_combo.setCurrentIndex(idx)
    finally:
        self.drainage_backend_combo.blockSignals(False)
    self.drainage_backend_combo.setToolTip(
        "Select drainage network solver backend.\n"
        "CPU: Python reference implementation.\n"
        "GPU: native CUDA drainage solver for EGL/Diffusion/Dynamic modes;\n"
        "falls back to CPU path when CUDA drainage bindings are unavailable."
    )

    self.drainage_gpu_method_combo = _find_or_create_combo(
        "drainage_gpu_method_combo", "Drainage GPU method:", param_form
    )
    prev_data = self.drainage_gpu_method_combo.currentData()
    prev_text = self.drainage_gpu_method_combo.currentText()
    self.drainage_gpu_method_combo.blockSignals(True)
    try:
        self.drainage_gpu_method_combo.clear()
        self.drainage_gpu_method_combo.addItem("Per-step GPU drainage (fast for sparse exchange)", "step")
        self.drainage_gpu_method_combo.addItem("Native iterative GPU drainage (batched substeps)", "iterative")
        idx = self.drainage_gpu_method_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.drainage_gpu_method_combo.findText(prev_text)
        if idx < 0:
            idx = 0
        if idx >= 0:
            self.drainage_gpu_method_combo.setCurrentIndex(idx)
    finally:
        self.drainage_gpu_method_combo.blockSignals(False)
    self.drainage_gpu_method_combo.setToolTip(
        "Select GPU drainage coupling method when drainage backend is GPU.\n"
        "Per-step: calls the GPU drainage step once per substep/iteration from Python.\n"
        "Native iterative: runs substeps and implicit iterations in one native call.\n"
        "Use native iterative for dense/active drainage exchange; per-step can be faster\n"
        "when exchange is sparse or mostly inactive."
    )

    self.drainage_coupling_substeps_spin = _find_or_create_spin(
        "drainage_coupling_substeps_spin", "Drainage substeps:", param_form
    )
    self.drainage_coupling_substeps_spin.setRange(1, 256)
    self.drainage_coupling_substeps_spin.setValue(1)
    self.drainage_coupling_substeps_spin.setToolTip(
        "Fixed number of 1D drainage substeps taken per 2D coupling step.\n"
        "Increase this for stiff drainage networks or dynamic-wave runs."
    )

    self.drainage_max_coupling_substeps_spin = _find_or_create_spin(
        "drainage_max_coupling_substeps_spin", "Drainage max adaptive substeps:", param_form
    )
    self.drainage_max_coupling_substeps_spin.setRange(1, 1024)
    self.drainage_max_coupling_substeps_spin.setValue(64)
    self.drainage_max_coupling_substeps_spin.setToolTip(
        "Maximum adaptive drainage substeps allowed when the 1D stability\n"
        "controller tightens the drainage timestep automatically."
    )

    self.drainage_head_deadband_spin = _find_or_create_double_spin(
        "drainage_head_deadband_spin", "Drainage head deadband:", param_form
    )
    self.drainage_head_deadband_spin.setRange(0.0, 10.0)
    self.drainage_head_deadband_spin.setDecimals(6)
    self.drainage_head_deadband_spin.setValue(1.0e-3)
    self.drainage_head_deadband_spin.setToolTip(
        "Head deadband used before drainage link and inlet exchange updates.\n"
        "Larger values reduce chatter near balanced states."
    )

    self.drainage_dynamic_relaxation_spin = _find_or_create_double_spin(
        "drainage_dynamic_relaxation_spin", "Drainage dynamic relaxation:", param_form
    )
    self.drainage_dynamic_relaxation_spin.setRange(0.0, 1.0)
    self.drainage_dynamic_relaxation_spin.setDecimals(3)
    self.drainage_dynamic_relaxation_spin.setSingleStep(0.05)
    self.drainage_dynamic_relaxation_spin.setValue(1.0)
    self.drainage_dynamic_relaxation_spin.setToolTip(
        "Dynamic-wave flow relaxation factor.\n"
        "1.0 keeps the full update; lower values damp oscillatory link-flow response."
    )

    self.drainage_adaptive_depth_fraction_spin = _find_or_create_double_spin(
        "drainage_adaptive_depth_fraction_spin", "Drainage adaptive depth fraction:", param_form
    )
    self.drainage_adaptive_depth_fraction_spin.setRange(0.001, 1.0)
    self.drainage_adaptive_depth_fraction_spin.setDecimals(3)
    self.drainage_adaptive_depth_fraction_spin.setSingleStep(0.01)
    self.drainage_adaptive_depth_fraction_spin.setValue(0.2)
    self.drainage_adaptive_depth_fraction_spin.setToolTip(
        "Adaptive drainage substepping threshold based on fractional node-depth\n"
        "change per substep. Lower values are more conservative."
    )

    self.drainage_adaptive_wave_courant_spin = _find_or_create_double_spin(
        "drainage_adaptive_wave_courant_spin", "Drainage adaptive wave Courant:", param_form
    )
    self.drainage_adaptive_wave_courant_spin.setRange(0.001, 10.0)
    self.drainage_adaptive_wave_courant_spin.setDecimals(3)
    self.drainage_adaptive_wave_courant_spin.setSingleStep(0.05)
    self.drainage_adaptive_wave_courant_spin.setValue(0.5)
    self.drainage_adaptive_wave_courant_spin.setToolTip(
        "Adaptive drainage substepping target for dynamic-wave links based on\n"
        "wave Courant number. Lower values are more conservative."
    )

    self.drainage_implicit_iters_spin = _find_or_create_spin(
        "drainage_implicit_iters_spin", "Drainage implicit iterations (GPU):", param_form
    )
    self.drainage_implicit_iters_spin.setRange(1, 8)
    self.drainage_implicit_iters_spin.setValue(2)
    self.drainage_implicit_iters_spin.setToolTip(
        "Number of implicit predictor/corrector inner iterations per drainage substep\n"
        "(GPU path only). 1 = explicit single-pass; 2-4 gives better mass conservation\n"
        "at ~linear cost per extra iteration."
    )

    self.drainage_implicit_relax_spin = _find_or_create_double_spin(
        "drainage_implicit_relax_spin", "Drainage implicit relaxation (GPU):", param_form
    )
    self.drainage_implicit_relax_spin.setRange(0.1, 1.0)
    self.drainage_implicit_relax_spin.setDecimals(2)
    self.drainage_implicit_relax_spin.setSingleStep(0.05)
    self.drainage_implicit_relax_spin.setValue(0.5)
    self.drainage_implicit_relax_spin.setToolTip(
        "Relaxation factor for implicit coupling iterates (GPU path only).\n"
        "1.0 = no relaxation (full update); 0.5 damps oscillations between iterates."
    )

    self.gpu_default_lbl = model_tab_page.findChild(QtWidgets.QLabel, "gpu_default_lbl")
    if self.gpu_default_lbl is None:
        self.gpu_default_lbl = QtWidgets.QLabel(
            "GPU is attempted by default when supported by the native backend."
        )
        self.gpu_default_lbl.setObjectName("gpu_default_lbl")
        _ensure_widget_row(self.gpu_default_lbl, param_form)
    self.gpu_default_lbl.setWordWrap(True)

    self.unit_system_lbl = model_tab_page.findChild(QtWidgets.QLabel, "unit_system_lbl")
    if self.unit_system_lbl is None:
        self.unit_system_lbl = QtWidgets.QLabel("Unit system: auto")
        self.unit_system_lbl.setObjectName("unit_system_lbl")
        _ensure_widget_row(self.unit_system_lbl, param_form)
    self.unit_system_lbl.setWordWrap(True)



def _bind_topology_tab_dynamic_controls(self, topology_tab_page: QtWidgets.QWidget, topo_layout: QtWidgets.QGridLayout) -> None:
    def _ensure(widget: QtWidgets.QWidget, row: int, col: int, row_span: int = 1, col_span: int = 1) -> None:
        if topo_layout.indexOf(widget) >= 0:
            return
        topo_layout.addWidget(widget, row, col, row_span, col_span)

    def _find_or_create_combo(name: str, row: int) -> QtWidgets.QComboBox:
        w = topology_tab_page.findChild(QtWidgets.QComboBox, name)
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
        w = topology_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
        if w is None:
            w = QtWidgets.QDoubleSpinBox()
            w.setObjectName(name)
        return w

    def _find_or_create_spin(name: str) -> QtWidgets.QSpinBox:
        w = topology_tab_page.findChild(QtWidgets.QSpinBox, name)
        if w is None:
            w = QtWidgets.QSpinBox()
            w.setObjectName(name)
        return w

    def _find_or_create_line_edit(name: str, text: str) -> QtWidgets.QLineEdit:
        w = topology_tab_page.findChild(QtWidgets.QLineEdit, name)
        if w is None:
            w = QtWidgets.QLineEdit(text)
            w.setObjectName(name)
        if not str(w.text() or "").strip():
            w.setText(text)
        return w

    def _find_or_create_check(name: str, text: str) -> QtWidgets.QCheckBox:
        w = topology_tab_page.findChild(QtWidgets.QCheckBox, name)
        if w is None:
            w = QtWidgets.QCheckBox(text)
            w.setObjectName(name)
        if not str(w.text() or "").strip():
            w.setText(text)
        return w

    def _find_or_create_form_container(name: str, row: int) -> QtWidgets.QFormLayout:
        container = topology_tab_page.findChild(QtWidgets.QWidget, name)
        if container is None:
            container = QtWidgets.QWidget()
            container.setObjectName(name)
        _ensure(container, row, 1)
        layout = container.layout()
        if not isinstance(layout, QtWidgets.QFormLayout):
            layout = QtWidgets.QFormLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        return layout

    def _reconnect(signal: object, callback: Callable[[], None]) -> None:
        try:
            signal.disconnect(callback)
        except Exception:
            pass
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
    _tqmesh_label = "TQMesh (advancing-front, built-in)" if _tqmesh_available() else "TQMesh (build plugin to enable)"
    _set_combo_items(
        self.topo_backend_combo,
        [
            (_gmsh_label, "gmsh"),
            ("Structured (built-in fallback)", "structured"),
            (_tqmesh_label, "tqmesh"),
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

    gmsh_form = _find_or_create_form_container("topo_gmsh_controls_widget", 8)
    quality_form = _find_or_create_form_container("topo_quality_controls_widget", 9)

    self.topo_gmsh_tri_algo_combo = topology_tab_page.findChild(QtWidgets.QComboBox, "topo_gmsh_tri_algo_combo")
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

    self.topo_gmsh_quad_algo_combo = topology_tab_page.findChild(QtWidgets.QComboBox, "topo_gmsh_quad_algo_combo")
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

    self.topo_gmsh_recombine_algo_combo = topology_tab_page.findChild(QtWidgets.QComboBox, "topo_gmsh_recombine_algo_combo")
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

    self.topo_gmsh_smoothing_spin = _find_or_create_spin("topo_gmsh_smoothing_spin")
    self.topo_gmsh_smoothing_spin.setRange(0, 100)
    self.topo_gmsh_smoothing_spin.setValue(5)
    if self.topo_gmsh_smoothing_spin.parent() is None:
        gmsh_form.addRow("Smoothing passes:", self.topo_gmsh_smoothing_spin)

    self.topo_gmsh_optimize_iters_spin = _find_or_create_spin("topo_gmsh_optimize_iters_spin")
    self.topo_gmsh_optimize_iters_spin.setRange(0, 100)
    self.topo_gmsh_optimize_iters_spin.setValue(3)
    if self.topo_gmsh_optimize_iters_spin.parent() is None:
        gmsh_form.addRow("Optimize iterations:", self.topo_gmsh_optimize_iters_spin)

    self.topo_gmsh_verbosity_spin = _find_or_create_spin("topo_gmsh_verbosity_spin")
    self.topo_gmsh_verbosity_spin.setRange(0, 10)
    self.topo_gmsh_verbosity_spin.setValue(1)
    if self.topo_gmsh_verbosity_spin.parent() is None:
        gmsh_form.addRow("Verbosity:", self.topo_gmsh_verbosity_spin)

    self.topo_gmsh_optimize_netgen_chk = _find_or_create_check("topo_gmsh_optimize_netgen_chk", "Enable Netgen optimize")
    if self.topo_gmsh_optimize_netgen_chk.parent() is None:
        gmsh_form.addRow(self.topo_gmsh_optimize_netgen_chk)

    self.topo_gmsh_quality_enable_chk = _find_or_create_check(
        "topo_gmsh_quality_enable_chk", "Enable Gmsh iterative quality loop"
    )
    self.topo_gmsh_quality_enable_chk.setChecked(False)
    if self.topo_gmsh_quality_enable_chk.parent() is None:
        quality_form.addRow(self.topo_gmsh_quality_enable_chk)

    self.topo_gmsh_quality_max_iters_spin = _find_or_create_spin("topo_gmsh_quality_max_iters_spin")
    self.topo_gmsh_quality_max_iters_spin.setRange(1, 50)
    self.topo_gmsh_quality_max_iters_spin.setValue(6)
    if self.topo_gmsh_quality_max_iters_spin.parent() is None:
        quality_form.addRow("Gmsh max attempts:", self.topo_gmsh_quality_max_iters_spin)

    self.topo_gmsh_quality_time_limit_spin = _find_or_create_double_spin("topo_gmsh_quality_time_limit_spin")
    self.topo_gmsh_quality_time_limit_spin.setRange(1.0, 3600.0)
    self.topo_gmsh_quality_time_limit_spin.setDecimals(1)
    self.topo_gmsh_quality_time_limit_spin.setValue(60.0)
    if self.topo_gmsh_quality_time_limit_spin.parent() is None:
        quality_form.addRow("Gmsh time budget (s):", self.topo_gmsh_quality_time_limit_spin)

    self.topo_quality_min_angle_spin = _find_or_create_double_spin("topo_quality_min_angle_spin")
    self.topo_quality_min_angle_spin.setRange(0.0, 89.0)
    self.topo_quality_min_angle_spin.setDecimals(1)
    self.topo_quality_min_angle_spin.setValue(5.0)
    if self.topo_quality_min_angle_spin.parent() is None:
        quality_form.addRow("Min angle (deg):", self.topo_quality_min_angle_spin)

    self.topo_quality_max_aspect_spin = _find_or_create_double_spin("topo_quality_max_aspect_spin")
    self.topo_quality_max_aspect_spin.setRange(1.0, 1.0e4)
    self.topo_quality_max_aspect_spin.setDecimals(2)
    self.topo_quality_max_aspect_spin.setValue(20.0)
    if self.topo_quality_max_aspect_spin.parent() is None:
        quality_form.addRow("Max aspect ratio:", self.topo_quality_max_aspect_spin)

    self.topo_quality_max_non_orth_spin = _find_or_create_double_spin("topo_quality_max_non_orth_spin")
    self.topo_quality_max_non_orth_spin.setRange(1.0, 89.9)
    self.topo_quality_max_non_orth_spin.setDecimals(1)
    self.topo_quality_max_non_orth_spin.setValue(82.0)
    if self.topo_quality_max_non_orth_spin.parent() is None:
        quality_form.addRow("Max non-orthogonality (deg):", self.topo_quality_max_non_orth_spin)

    self.topo_quality_min_area_edit = _find_or_create_line_edit("topo_quality_min_area_edit", "1e-14")
    if self.topo_quality_min_area_edit.parent() is None:
        quality_form.addRow("Min area / bbox area:", self.topo_quality_min_area_edit)

    self.topo_quality_size_scales_edit = _find_or_create_line_edit("topo_quality_size_scales_edit", "1.0,0.9,0.8,0.7")
    if self.topo_quality_size_scales_edit.parent() is None:
        quality_form.addRow("Retry size scales:", self.topo_quality_size_scales_edit)

    self.topo_quality_smooth_increments_edit = _find_or_create_line_edit("topo_quality_smooth_increments_edit", "0,2,4,6")
    if self.topo_quality_smooth_increments_edit.parent() is None:
        quality_form.addRow("Retry smooth increments:", self.topo_quality_smooth_increments_edit)

    self.topo_quality_strict_chk = _find_or_create_check("topo_quality_strict_chk", "Strict quality acceptance")
    if self.topo_quality_strict_chk.parent() is None:
        quality_form.addRow(self.topo_quality_strict_chk)

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
    _reconnect(self.topo_gmsh_quality_enable_chk.toggled, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_quality_max_iters_spin.valueChanged, self._update_topology_control_summary)
    _reconnect(self.topo_gmsh_quality_time_limit_spin.valueChanged, self._update_topology_control_summary)



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
        except Exception:
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
        except Exception:
            return fallback

    for ft in node_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            pt = geom.asPoint()
        except Exception:
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
                except Exception:
                    pass

        area_val = None
        for nm in ("area_m2", "area", "cross_area"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    a_try = float(ft[nm])
                    if a_try > 0.0:
                        area_val = a_try
                        break
                except Exception:
                    pass

        span_val = None
        for nm in ("span", "span_m", "width", "width_m"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    s_try = float(ft[nm])
                    if s_try > 0.0:
                        span_val = s_try
                        break
                except Exception:
                    pass

        rise_val = None
        for nm in ("rise", "rise_m", "height", "height_m"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    r_try = float(ft[nm])
                    if r_try > 0.0:
                        rise_val = r_try
                        break
                except Exception:
                    pass

        equiv_d_val = None
        for nm in ("equiv_diameter_m", "equiv_diameter"):
            if nm in link_fields and ft[nm] not in (None, ""):
                try:
                    eq_try = float(ft[nm])
                    if eq_try > 0.0:
                        equiv_d_val = eq_try
                        break
                except Exception:
                    pass

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

        links.append(
            DrainageLink(
                link_id=link_id,
                from_node_id=from_node,
                to_node_id=to_node,
                link_type=str(ft["link_type"] if "link_type" in link_fields else "conduit").strip() or "conduit",
                length=float(ft["length"]) if "length" in link_fields and ft["length"] not in (None, "") else float(geom.length()),
                roughness_n=float(ft["roughness_n"] if "roughness_n" in link_fields and ft["roughness_n"] not in (None, "") else 0.013),
                diameter=diameter_val,
                max_flow=float(ft["max_flow"]) if "max_flow" in link_fields and ft["max_flow"] not in (None, "") else None,
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
                except Exception:
                    try:
                        c = geom.centroid()
                        pt = c.asPoint() if c is not None and not c.isEmpty() else None
                    except Exception:
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
            "pipe_end", "pipe-end", "daylighted_pipe", "daylighted", "daylight_pipe"
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

    gravity = float(getattr(self, "_gravity", 9.81))
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
        cell_min_z = np.empty(n_cells, dtype=np.float64)
        for i in range(n_cells):
            s, e = int(offsets[i]), int(offsets[i + 1])
            ring = face_nodes_arr[s:e].astype(np.int32)
            face_node[i, : e - s] = ring
            cell_cx[i] = float(np.mean(node_x[ring]))
            cell_cy[i] = float(np.mean(node_y[ring]))
            cell_min_z[i] = float(np.min(node_z[ring]))
    else:
        tri = cell_nodes_tri.reshape(-1, 3).astype(np.int32)
        n_cells = tri.shape[0]
        max_vp = 3
        face_node = tri
        cell_cx = np.mean(node_x[tri], axis=1)
        cell_cy = np.mean(node_y[tri], axis=1)
        cell_min_z = np.min(node_z[tri], axis=1)

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
        except Exception:
            pass

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

        # Face minimum bed elevation
        fz_var = ds.createVariable("face_z", "f8", ("face",))
        fz_var.long_name = "minimum bed elevation at face"
        fz_var.units = len_unit
        fz_var.mesh = "mesh2d"
        fz_var.location = "face"
        fz_var.grid_mapping = "crs"
        fz_var[:] = cell_min_z.astype(np.float64)

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
                wse_arr[ti] = (h_f + cell_min_z[:n_cells]).astype(np.float32)
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



def _on_run(self):
    if self._mesh_data is None:
        self._log("Run aborted: mesh not available after preflight.")
        return
    if SWE2DBackend is None:
        self._log("Run aborted: native backend not available after preflight.")
        return
    if not self._require_run_components(
        [
            ("_run_data_builder", "run data builder"),
            ("_run_options_builder", "run options builder"),
            ("_backend_initializer", "backend initializer"),
            ("_run_finalizer", "run finalizer"),
            ("_run_lifecycle", "run lifecycle"),
        ],
        context_label="Run",
    ):
        return

    self._cancel_requested = False
    self.run_btn.setEnabled(False)
    self.cancel_btn.setEnabled(True)
    self.progress_bar.setValue(0)

    backend = None
    run_id = ""
    run_wallclock_start = ""
    run_perf_start = time.perf_counter()
    run_log_start_idx = len(self._runtime_log_lines)
    try:
        run_input = self._run_data_builder.build()
        node_x = run_input.node_x
        node_y = run_input.node_y
        node_z = run_input.node_z
        cell_nodes = run_input.cell_nodes
        face_offsets = run_input.face_offsets
        face_nodes = run_input.face_nodes
        bc_n0 = run_input.bc_n0
        bc_n1 = run_input.bc_n1
        bc_tp = run_input.bc_tp
        bc_vl = run_input.bc_vl
        side_hydrographs = run_input.side_hydrographs
        edge_hydrographs = run_input.edge_hydrographs
        edge_group_overrides = run_input.edge_group_overrides
        h0 = run_input.h0
        hu0 = run_input.hu0
        hv0 = run_input.hv0
        n_mann_cell = run_input.n_mann_cell

        run_options = self._run_options_builder.build()
        run_duration_s = run_options.run_duration_s
        dt_cfg = run_options.dt_cfg
        adaptive_cfl_dt = run_options.adaptive_cfl_dt
        dt_fixed = run_options.dt_fixed
        dt_request = run_options.dt_request
        reconstruction_mode = run_options.reconstruction_mode
        reconstruction_name = run_options.reconstruction_name
        temporal_scheme = run_options.temporal_scheme
        temporal_scheme_name = run_options.temporal_scheme_name
        godunov_mode = run_options.godunov_mode
        coupling_loop_mode = run_options.coupling_loop_mode
        drainage_solver_backend_mode = run_options.drainage_solver_backend_mode
        drainage_gpu_method_mode = run_options.drainage_gpu_method_mode
        cuda_graphs_enabled = run_options.cuda_graphs_enabled
        experimental_3d_enabled = run_options.experimental_3d_enabled
        model_options = run_options.model_options
        swe3d_env_overrides = run_options.swe3d_env_overrides
        self._three_d_patch_snapshots = []
        self._three_d_patch_last_spec = None
        rain_rate_model = run_options.rain_rate_model
        internal_flow_forcing = run_options.internal_flow_forcing
        cell_source_model = run_options.cell_source_model
        thiessen_forcing = run_options.thiessen_forcing
        pipe_network_cfg = run_options.pipe_network_cfg
        hydraulic_structures_cfg = run_options.hydraulic_structures_cfg

        # Propagate locally-built drainage/structure configs into model_options
        # so that enable_pipe_network_module and enable_hydraulic_structures flags are set correctly.
        if model_options is not None:
            if pipe_network_cfg is not None:
                model_options.pipe_network = pipe_network_cfg
            if hydraulic_structures_cfg is not None:
                model_options.hydraulic_structures = hydraulic_structures_cfg

        if self._model_gpkg_path and os.path.exists(self._model_gpkg_path):
            try:
                self._persist_model_layer_bindings(self._model_gpkg_path)
            except Exception as exc:
                self._log(f"Model coupling metadata persist warning: {exc}")

        coupling_soa = None
        if pack_coupling_soa is not None:
            coupling_soa = pack_coupling_soa(
                n_cells=int(self._mesh_cell_areas().shape[0]),
                pipe_network=pipe_network_cfg,
                hydraulic_structures=hydraulic_structures_cfg,
            )
        coupling_controller = None
        if SWE2DCouplingController is not None and (pipe_network_cfg is not None or hydraulic_structures_cfg is not None):
            drainage_mod = SWE2DUrbanDrainageModule(pipe_network_cfg) if pipe_network_cfg is not None and SWE2DUrbanDrainageModule is not None else None
            if drainage_mod is not None:
                drainage_mod.initialize()
            structures_mod = SWE2DStructureModule(hydraulic_structures_cfg) if hydraulic_structures_cfg is not None and SWE2DStructureModule is not None else None
            coupling_controller = SWE2DCouplingController(
                cell_area_m2=self._mesh_cell_areas(),
                cell_bed_m=self._mesh_cell_min_bed(),
                drainage=drainage_mod,
                structures=structures_mod,
                coupling_loop=coupling_loop_mode,
                drainage_solver_backend=drainage_solver_backend_mode,
                drainage_gpu_method=drainage_gpu_method_mode,
            )
            # GPU-first runtime policy: for legacy saved projects that still
            # carry CPU coupling selections, opportunistically promote to
            # CUDA/GPU coupling when native bindings are available.
            force_cpu_coupling = os.environ.get("BACKWATER_SWE2D_FORCE_CPU_COUPLING", "").strip() == "1"
            if not force_cpu_coupling:
                try:
                    native_mod = coupling_controller._native_cuda_module() if hasattr(coupling_controller, "_native_cuda_module") else None
                except Exception:
                    native_mod = None
                if native_mod is not None and str(coupling_loop_mode).strip().lower() == "cpu":
                    coupling_loop_mode = "cuda"
                    coupling_controller.coupling_loop = "cuda"
                    self._log("Coupling loop auto-promoted: CPU -> CUDA (native CUDA coupling available).")
                if (
                    native_mod is not None
                    and str(drainage_solver_backend_mode).strip().lower() == "cpu"
                    and hasattr(native_mod, "swe2d_gpu_drainage_step")
                    and getattr(coupling_controller, "drainage", None) is not None
                ):
                    drainage_solver_backend_mode = "gpu"
                    coupling_controller.drainage_solver_backend = "gpu"
                    self._log("Drainage backend auto-promoted: CPU -> GPU (native CUDA drainage available).")
        rain_stats_acc = {"rain_mm": 0.0, "excess_mm": 0.0, "samples": 0}

        # Snapshot output interval — clamp to at least 1 s to avoid div-by-zero
        _oi_hr = self._parse_time_hours(self.output_interval_edit.text())
        output_interval_s = max(1.0, _oi_hr * 3600.0)
        _line_oi_hr = self._parse_time_hours(self.line_output_interval_edit.text())
        line_output_interval_s = max(1.0, _line_oi_hr * 3600.0)
        self._snapshot_timesteps = []
        self._line_snapshot_rows = []
        self._line_snapshot_profile_rows = []
        self._coupling_snapshot_rows = []
        _next_snap_t = output_interval_s
        _next_line_snap_t = line_output_interval_s
        _next_coupling_snap_t = line_output_interval_s
        sample_map = self._build_line_sampling_map()
        cell_min_z = self._mesh_cell_min_bed() if sample_map else None
        run_id = datetime.datetime.utcnow().strftime("swe2d_%Y%m%dT%H%M%SZ")
        run_wallclock_start = datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")

        dynamic_bc = bool(np.any((bc_tp == _BC_TS_FLOW) | (bc_tp == _BC_TS_STAGE)) or edge_hydrographs)
        if dynamic_bc:
            self._log("Timeseries BC mode active (flow/stage hydrographs).")

        run_mode_name = "2D"
        if model_options is not None and SWE2DThreeDSolverModel is not None:
            if int(model_options.three_d_solver_model) == int(SWE2DThreeDSolverModel.SINGLE_PHASE_FREE_SURFACE_VOF):
                run_mode_name = "2D + Experimental 3D patch"

        coupling_mode_label = "off"
        if model_options is not None and SWE2DThreeDCouplingMode is not None:
            try:
                cm = int(model_options.coupling_mode)
                if cm == int(SWE2DThreeDCouplingMode.ONE_WAY_2D_TO_3D):
                    coupling_mode_label = "one-way (2D -> 3D)"
                elif cm == int(SWE2DThreeDCouplingMode.TWO_WAY_2D_3D):
                    coupling_mode_label = "two-way (2D <-> 3D)"
            except Exception:
                coupling_mode_label = "off"

        self._log("Starting 2D run...")
        if run_mode_name != "2D":
            self._log(f"Run mode: {run_mode_name} (coupling={coupling_mode_label}).")
        self._log(f"Run wallclock start: {run_wallclock_start}")
        self._log(f"Reconstruction mode: {reconstruction_name}")
        self._log(f"Temporal scheme: {temporal_scheme_name}")
        self._log(
            f"Output intervals: mesh={output_interval_s:.1f}s, sample-lines={line_output_interval_s:.1f}s"
        )
        self._log(
            "Stability controls: "
            f"max_rel_dh={float(self.max_rel_depth_increase_spin.value()):.3f}, "
            f"gpu_diag_sync_steps={int(self.gpu_diag_sync_interval_spin.value())}, "
            f"src_dh_step_cap={float(self.max_source_depth_step_spin.value()):.6e}, "
            f"src_rate_cap={float(self.max_source_rate_spin.value()):.6e}, "
            f"extreme_rain_mode={bool(self.extreme_rain_mode_chk.isChecked())}, "
            f"src_beta={float(self.source_cfl_beta_spin.value()):.3f}, "
            f"src_max_substeps={int(self.source_max_substeps_spin.value())}, "
            f"true_subcycling={bool(self.source_true_subcycling_chk.isChecked())}, "
            f"imex_split={bool(self.source_imex_split_chk.isChecked())}, "
            f"stage_coupled_imex_rk2={bool(getattr(self, 'source_stage_coupled_imex_rk2_chk', None) and self.source_stage_coupled_imex_rk2_chk.isChecked())}, "
            f"shallow_damp_h={float(self.shallow_damping_depth_spin.value()):.6e}, "
            f"depth_cap={float(self.depth_cap_spin.value()):.3f}, "
            f"mom_cap_min={float(self.momentum_cap_min_speed_spin.value()):.3f}, "
            f"mom_cap_mult={float(self.momentum_cap_celerity_mult_spin.value()):.3f}, "
            f"invA_cap={float(self.max_inv_area_spin.value()):.3e}, "
            f"lambda_cap={float(self.cfl_lambda_cap_spin.value()):.3e}"
        )
        if adaptive_cfl_dt:
            self._log(f"Timestep mode: variable CFL (dt_max={dt_cfg:.5f} s)")
        else:
            self._log(f"Timestep mode: fixed dt ({dt_cfg:.5f} s)")
        if float(np.asarray(rain_rate_model, dtype=np.float64)) > 0.0:
            self._log(
                f"Rain-on-grid active: {float(self.rain_rate_spin.value()):.3f} mm/hr "
                f"(applied as {float(np.asarray(rain_rate_model, dtype=np.float64)):.6e} {self._length_unit_name}/s)"
            )
        if thiessen_forcing is not None:
            infil_method = str(getattr(thiessen_forcing, "infiltration_method", "scs_cn") or "scs_cn").lower().strip()
            infil_label = "NRCS CN infiltration"
            if infil_method == "none":
                infil_label = "no infiltration (all rainfall to runoff)"
            self._log(
                "Spatial rainfall forcing active: Thiessen nearest-gage interpolation + "
                f"{infil_label}."
            )
        if cell_source_model is not None:
            self._log(
                f"Internal source/sink forcing active: total_Q={float(np.sum(cell_source_model)):.6f} {self._flow_unit_label()}"
            )
        if internal_flow_forcing is not None:
            ts_count = int(len(internal_flow_forcing.get("dynamic_terms", [])))
            if ts_count > 0:
                self._log(f"Internal flow time-series forcing active: features={ts_count}")
        if coupling_controller is not None:
            self._log(
                "Coupled drainage/structure forcing active: "
                f"drainage={pipe_network_cfg is not None}, structures={hydraulic_structures_cfg is not None}, "
                f"loop={coupling_loop_mode}, drainage_backend={drainage_solver_backend_mode}, "
                f"drainage_gpu_method={drainage_gpu_method_mode}"
            )
            coupling_runtime_mode = "cpu"
            if str(coupling_loop_mode).strip().lower() == "cuda":
                try:
                    mod = coupling_controller._native_cuda_module() if hasattr(coupling_controller, "_native_cuda_module") else None
                except Exception:
                    mod = None
                if mod is not None:
                    coupling_runtime_mode = "cuda"
                else:
                    coupling_runtime_mode = "cpu (cuda requested, fallback active)"
            self._log(f"Coupling runtime mode: {coupling_runtime_mode}")
        self._log(f"CUDA graph replay: {'enabled' if cuda_graphs_enabled else 'disabled'}")
        if coupling_soa is not None:
            dn = coupling_soa.drainage
            ss = coupling_soa.structures
            if dn is not None:
                bad_links = int(np.sum((dn.link_from < 0) | (dn.link_to < 0)))
                bad_inlets = int(np.sum((dn.inlet_cell < 0) | (dn.inlet_node < 0)))
                self._log(
                    "CUDA SoA pack (drainage): "
                    f"nodes={dn.node_x.size}, links={dn.link_from.size}, inlets={dn.inlet_cell.size}, "
                    f"invalid_links={bad_links}, invalid_inlets={bad_inlets}"
                )
            if ss is not None:
                bad_struct = int(np.sum((ss.upstream_cell < 0) | (ss.downstream_cell < 0)))
                self._log(
                    "CUDA SoA pack (structures): "
                    f"count={ss.structure_type.size}, invalid_cell_pairs={bad_struct}"
                )
        def _build_and_initialize_backend() -> SWE2DBackend:
            return self._backend_initializer.build_and_initialize(
                backend_cls=SWE2DBackend,
                swe3d_env_overrides=swe3d_env_overrides,
                dynamic_bc=dynamic_bc,
                node_x=node_x,
                node_y=node_y,
                node_z=node_z,
                cell_nodes=cell_nodes,
                face_offsets=face_offsets,
                face_nodes=face_nodes,
                bc_n0=bc_n0,
                bc_n1=bc_n1,
                bc_tp=bc_tp,
                bc_vl=bc_vl,
                side_hydrographs=side_hydrographs,
                edge_hydrographs=edge_hydrographs,
                h0=h0,
                hu0=hu0,
                hv0=hv0,
                n_mann_cell=n_mann_cell,
                dt_fixed=dt_fixed,
                dt_max=dt_cfg,
                model_options=model_options,
                reconstruction_mode=reconstruction_mode,
                temporal_scheme=temporal_scheme,
                godunov_mode=godunov_mode,
            )

        try:
            backend = _build_and_initialize_backend()
        except Exception as init_exc:
            err_l = str(init_exc).lower()
            is_illegal_mem = "illegal memory access" in err_l
            if cuda_graphs_enabled and is_illegal_mem:
                self._log(
                    "CUDA solver init failed with illegal memory access while graph replay was enabled; "
                    "retrying once with CUDA graph replay disabled."
                )
                cuda_graphs_enabled = False
                os.environ["BACKWATER_ENABLE_CUDA_GRAPHS"] = "0"
                backend = _build_and_initialize_backend()
                self._log("CUDA graph replay fallback at solver init succeeded.")
            else:
                raise

        experimental_3d_runtime = bool(
            model_options is not None
            and SWE2DThreeDSolverModel is not None
            and int(model_options.three_d_solver_model) == int(SWE2DThreeDSolverModel.SINGLE_PHASE_FREE_SURFACE_VOF)
        )
        if experimental_3d_enabled and not experimental_3d_runtime:
            raise RuntimeError(
                "Experimental 3D mode was requested but solver model options did not activate 3D runtime."
            )
        if experimental_3d_runtime and not bool(backend.supports_3d_patch_observation()):
            raise RuntimeError(
                "Experimental 3D mode requires native 3D patch observation APIs; "
                "current native module does not expose them."
            )

        if SWE2DThreeDPatchObserver is None:
            raise RuntimeError("SWE2DThreeDPatchObserver seam is unavailable.")
        _three_d_observer = SWE2DThreeDPatchObserver(backend=backend, runtime_enabled=experimental_3d_runtime)
        _get_3d_patch_stats = _three_d_observer.get_patch_stats
        _get_3d_patch_vof = _three_d_observer.get_patch_vof

        if experimental_3d_runtime:
            try:
                self._apply_3d_patch_face_bc_to_backend(backend)
            except Exception as exc:
                self._log(f"3D face BC upload warning (continuing with env defaults): {exc}")
            stats0 = _get_3d_patch_stats()
            if stats0 is not None:
                try:
                    spec0 = self._build_patch_spec_from_stats(stats0, swe3d_env_overrides)
                    spec0_dict = self._patch_spec_to_dict(spec0)
                    if isinstance(spec0_dict, dict):
                        self._three_d_patch_last_spec = dict(spec0_dict)
                except Exception:
                    self._three_d_patch_last_spec = None
                self._log(
                    "3D patch initialized: "
                    f"nx={int(stats0.get('nx', 0))} ny={int(stats0.get('ny', 0))} nz={int(stats0.get('nz', 0))} "
                    f"dx={float(stats0.get('dx', 0.0)):.3f} dy={float(stats0.get('dy', 0.0)):.3f} dz={float(stats0.get('dz', 0.0)):.3f} "
                    f"cells={int(stats0.get('n_cells', 0))}"
                )
                try:
                    self._upload_experimental_3d_obj_geometry(
                        backend=backend,
                        patch_stats=stats0,
                        swe3d_env_overrides=swe3d_env_overrides,
                        backend_builder=_build_and_initialize_backend,
                        bc_n0=bc_n0,
                        bc_n1=bc_n1,
                        bc_tp=bc_tp,
                        bc_vl=bc_vl,
                    )
                except Exception as exc:
                    self._log(f"3D sub-grid preprocessing failed (run continues): {exc}")
                try:
                    self._initialize_experimental_3d_patch_state(
                        backend=backend,
                        patch_stats=stats0,
                        swe3d_env_overrides=swe3d_env_overrides,
                        bc_n0=bc_n0,
                        bc_n1=bc_n1,
                        bc_tp=bc_tp,
                        bc_vl=bc_vl,
                    )
                except Exception as exc:
                    self._log(f"3D patch initial-state seeding failed (run continues): {exc}")
                try:
                    self._upload_experimental_3d_interface_contract(
                        backend=backend,
                        patch_stats=stats0,
                        bc_n0=bc_n0,
                        bc_n1=bc_n1,
                        bc_tp=bc_tp,
                        coupling_mode=int(model_options.coupling_mode) if model_options is not None else 0,
                    )
                except Exception as exc:
                    raise RuntimeError(f"3D coupling contract setup failed: {exc}")
            else:
                raise RuntimeError(
                    "Experimental 3D mode requested, but native 3D patch stats are unavailable after initialization."
                )

        last_diag = None
        t_accum = 0.0
        i = 0
        last_valid_cmax = float("nan")
        last_valid_wse_res = float("nan")
        # Wall-clock throttle for QApplication.processEvents() – fire at most
        # every _PROCESS_EVENTS_INTERVAL_S seconds regardless of step count.
        # This prevents QGIS canvas repaints from dominating the loop when
        # solver steps are short (e.g. small meshes, fast GPU).
        _PROCESS_EVENTS_INTERVAL_S = 0.10  # 100 ms
        _last_process_events_wall = time.perf_counter()
        timing_totals_ms = {
            "wall": 0.0,
            "step": 0.0,
            "coupling": 0.0,
            "source": 0.0,
            "state": 0.0,
            "bc": 0.0,
            "ui": 0.0,
        }
        timing_samples = 0
        self._log("Step timing diagnostics enabled (ms): wall, step, coupling, source, state, bc, ui.")
        if dynamic_bc and not backend.supports_dynamic_boundary_update():
            raise RuntimeError("Native module does not support dynamic boundary updates. Rebuild hydra_swe2d.")

        native_bc_forcing = False
        native_rain_cn_forcing = False
        if SWE2DRunSetupConfigurator is None:
            raise RuntimeError("SWE2DRunSetupConfigurator seam is unavailable.")
        if SWE2DNativeBoundaryHydrographConfigurator is None:
            raise RuntimeError("SWE2DNativeBoundaryHydrographConfigurator seam is unavailable.")
        run_setup_configurator = SWE2DRunSetupConfigurator()
        native_bc_cfg = SWE2DNativeBoundaryHydrographConfigurator()

        if dynamic_bc and hasattr(backend, "set_boundary_hydrographs_native"):
            try:
                progressive = True
                if hasattr(self, "inflow_progressive_chk") and self.inflow_progressive_chk is not None:
                    progressive = bool(self.inflow_progressive_chk.isChecked())
                node_x = self._mesh_data["node_x"]
                node_y = self._mesh_data["node_y"]
                native_bc_res = native_bc_cfg.configure(
                    backend=backend,
                    bc_n0=bc_n0,
                    bc_n1=bc_n1,
                    bc_tp=bc_tp,
                    side_hydrographs=side_hydrographs,
                    edge_hydrographs=edge_hydrographs,
                    node_x=node_x,
                    node_y=node_y,
                    inflow_q_bc_type=int(_BC_INFLOW_Q),
                    progressive=progressive,
                )
                if bool(native_bc_res.get("native_bc_forcing", False)):
                    native_bc_forcing = True
                    self._log(
                        f"Native BC hydrograph forcing configured for {int(native_bc_res.get('configured_edges', 0))} boundary edges."
                    )
                elif bool(native_bc_res.get("skipped_progressive", False)):
                    self._log("Native BC hydrographs skipped: progressive inflow activation is enabled for flow hydrographs.")
            except Exception as exc:
                self._log(f"Native BC hydrograph forcing unavailable: {exc}")

        if thiessen_forcing is not None and hasattr(backend, "set_rain_cn_forcing_native"):
            try:
                native_rain_res = run_setup_configurator.configure_native_rain_cn_forcing(
                    backend=backend,
                    thiessen_forcing=thiessen_forcing,
                    mm_to_model_depth=float(self._rain_mm_to_model_depth()),
                )
                if bool(native_rain_res.get("configured", False)):
                    native_rain_cn_forcing = True
                    self._log(
                        "Native preprocessed rainfall-excess forcing configured for GPU timestep evaluation "
                        f"(infiltration={str(native_rain_res.get('infiltration_method', 'scs_cn'))}, "
                        f"groups={int(native_rain_res.get('groups', 0))})."
                    )
            except Exception as exc:
                self._log(f"Native rain+CN forcing unavailable: {exc}")

        native_source_injection_mode = hasattr(backend, "set_external_sources_native")
        if native_source_injection_mode:
            try:
                native_src_res = run_setup_configurator.configure_native_source_injection(backend=backend)
                native_source_injection_mode = bool(native_src_res.get("native_source_injection_mode", False))
                if bool(native_src_res.get("configured", False)):
                    self._log("Native external source injection enabled (device-resident coupling path).")
            except Exception as exc:
                native_source_injection_mode = False
                self._log(f"Native external source injection unavailable: {exc}")

        area_model = np.asarray(self._mesh_cell_areas(), dtype=np.float64).ravel()
        n_area = int(area_model.size)
        h0_model = np.asarray(h0, dtype=np.float64).ravel()
        n_store = min(n_area, int(h0_model.size))
        storage_start_model = float(np.sum(h0_model[:n_store] * area_model[:n_store])) if n_store > 0 else 0.0
        source_budget_model = {
            "rain": 0.0,
            "cell": 0.0,
            "coupling": 0.0,
        }

        node_x_bc = self._mesh_data["node_x"]
        node_y_bc = self._mesh_data["node_y"]
        edge_len_bc = np.hypot(node_x_bc[bc_n1] - node_x_bc[bc_n0], node_y_bc[bc_n1] - node_y_bc[bc_n0]).astype(np.float64)
        xmin_bc = float(np.min(node_x_bc)) if node_x_bc.size else 0.0
        xmax_bc = float(np.max(node_x_bc)) if node_x_bc.size else 0.0
        ymin_bc = float(np.min(node_y_bc)) if node_y_bc.size else 0.0
        ymax_bc = float(np.max(node_y_bc)) if node_y_bc.size else 0.0
        mx_bc = 0.5 * (node_x_bc[bc_n0] + node_x_bc[bc_n1]) if bc_n0.size else np.empty(0, dtype=np.float64)
        my_bc = 0.5 * (node_y_bc[bc_n0] + node_y_bc[bc_n1]) if bc_n0.size else np.empty(0, dtype=np.float64)
        if bc_n0.size:
            d_bc = np.vstack([
                np.abs(mx_bc - xmin_bc),
                np.abs(mx_bc - xmax_bc),
                np.abs(my_bc - ymin_bc),
                np.abs(my_bc - ymax_bc),
            ])
            side_idx_bc = np.argmin(d_bc, axis=0)
        else:
            side_idx_bc = np.empty(0, dtype=np.int32)
        side_names_bc = ["left", "right", "bottom", "top"]
        edge_group_labels: List[str] = []
        for ei in range(int(bc_n0.size)):
            if ei in edge_group_overrides:
                edge_group_labels.append(str(edge_group_overrides[ei]))
            else:
                edge_group_labels.append(str(side_names_bc[int(side_idx_bc[ei])]))
        boundary_flux_budget_model: Dict[str, float] = {}

        if SWE2DRuntimeSourceManager is None:
            raise RuntimeError("SWE2DRuntimeSourceManager seam is unavailable.")
        runtime_source_manager = SWE2DRuntimeSourceManager(
            rain_rate_model=rain_rate_model,
            thiessen_forcing=thiessen_forcing,
            native_rain_cn_forcing=native_rain_cn_forcing,
            internal_flow_forcing=internal_flow_forcing,
            rain_stats_acc=rain_stats_acc,
            area_model=area_model,
            edge_len_bc=edge_len_bc,
            edge_group_labels=edge_group_labels,
            inflow_q_bc_type=int(_BC_INFLOW_Q),
            rain_rate_si_to_model_callback=self._rain_rate_si_to_model,
            internal_flow_source_cms_at_time_callback=self._internal_flow_source_cms_at_time,
            flow_si_to_model_callback=self._flow_si_to_model,
        )
        source_budget_model = runtime_source_manager.source_budget_model
        boundary_flux_budget_model = runtime_source_manager.boundary_flux_budget_model
        _accumulate_boundary_flux_volume_model = runtime_source_manager.accumulate_boundary_flux_volume_model
        _accumulate_source_volume_model = runtime_source_manager.accumulate_source_volume_model
        _rain_source_for_window = runtime_source_manager.rain_source_for_window
        _cell_source_model_at_time = runtime_source_manager.cell_source_model_at_time

        stage_coupled_imex_requested = bool(
            hasattr(self, "source_stage_coupled_imex_rk2_chk")
            and self.source_stage_coupled_imex_rk2_chk.isChecked()
        )
        stage_coupled_imex_enabled = False
        stage_res = run_setup_configurator.resolve_stage_coupled_imex(
            requested=stage_coupled_imex_requested,
            coupling_controller=coupling_controller,
            temporal_scheme=temporal_scheme,
            required_temporal_scheme=TemporalScheme.SSP_RK2,
            native_source_injection_mode=native_source_injection_mode,
        )
        stage_coupled_imex_enabled = bool(stage_res.get("enabled", False))
        stage_reasons = list(stage_res.get("reasons", []))
        if stage_coupled_imex_requested:
            if stage_reasons:
                self._log(
                    "Stage-coupled IMEX-RK2 requested but disabled: "
                    + "; ".join(stage_reasons)
                )
            else:
                self._log("Stage-coupled IMEX-RK2 enabled for external coupling sources.")

        if SWE2DRuntimeStepExecutor is None:
            raise RuntimeError("SWE2DRuntimeStepExecutor seam is unavailable.")
        if SWE2DRuntimeReporter is None:
            raise RuntimeError("SWE2DRuntimeReporter seam is unavailable.")
        runtime_step_executor = SWE2DRuntimeStepExecutor()
        runtime_reporter = SWE2DRuntimeReporter()

        loop_result = _execute_run_timestep_loop_runtime_logic(
            wb=self,
            backend=backend,
            runtime_step_executor=runtime_step_executor,
            runtime_reporter=runtime_reporter,
            run_duration_s=run_duration_s,
            t_accum=t_accum,
            i=i,
            last_diag=last_diag,
            last_valid_cmax=last_valid_cmax,
            last_valid_wse_res=last_valid_wse_res,
            dt_cfg=dt_cfg,
            dt_request=dt_request,
            stage_coupled_imex_enabled=stage_coupled_imex_enabled,
            coupling_controller=coupling_controller,
            dynamic_bc=dynamic_bc,
            native_bc_forcing=native_bc_forcing,
            bc_n0=bc_n0,
            bc_n1=bc_n1,
            bc_tp=bc_tp,
            bc_vl=bc_vl,
            side_hydrographs=side_hydrographs,
            edge_hydrographs=edge_hydrographs,
            rain_source_for_window_callback=_rain_source_for_window,
            cell_source_model_at_time_callback=_cell_source_model_at_time,
            accumulate_source_volume_model_callback=_accumulate_source_volume_model,
            native_source_injection_mode=native_source_injection_mode,
            accumulate_boundary_flux_volume_model_callback=_accumulate_boundary_flux_volume_model,
            sample_map=sample_map,
            cell_min_z=cell_min_z,
            experimental_3d_runtime=experimental_3d_runtime,
            timing_totals_ms=timing_totals_ms,
            timing_samples=timing_samples,
            next_snap_t=_next_snap_t,
            next_line_snap_t=_next_line_snap_t,
            next_coupling_snap_t=_next_coupling_snap_t,
            output_interval_s=output_interval_s,
            line_output_interval_s=line_output_interval_s,
            process_events_interval_s=_PROCESS_EVENTS_INTERVAL_S,
            last_process_events_wall=_last_process_events_wall,
            process_events_callback=QtWidgets.QApplication.processEvents,
            get_3d_patch_stats_callback=_get_3d_patch_stats,
            get_3d_patch_vof_callback=_get_3d_patch_vof,
        )
        t_accum = float(loop_result.get("t_accum", t_accum))
        i = int(loop_result.get("i", i))
        last_diag = loop_result.get("last_diag", last_diag)
        last_valid_cmax = float(loop_result.get("last_valid_cmax", last_valid_cmax))
        last_valid_wse_res = float(loop_result.get("last_valid_wse_res", last_valid_wse_res))
        _next_snap_t = float(loop_result.get("next_snap_t", _next_snap_t))
        _next_line_snap_t = float(loop_result.get("next_line_snap_t", _next_line_snap_t))
        _next_coupling_snap_t = float(loop_result.get("next_coupling_snap_t", _next_coupling_snap_t))
        _last_process_events_wall = float(loop_result.get("last_process_events_wall", _last_process_events_wall))
        timing_samples = int(loop_result.get("timing_samples", timing_samples))
        h, hu, hv = backend.get_state()
        if experimental_3d_runtime and not self._three_d_patch_snapshots:
            s3 = _get_3d_patch_stats()
            v3 = _get_3d_patch_vof()
            if s3 is not None and v3 is not None:
                self._append_3d_patch_snapshot(t_accum, s3, v3)
        if native_source_injection_mode:
            try:
                backend.set_external_sources_native(None)
            except Exception:
                pass
        self._result_data = {
            "h": h,
            "hu": hu,
            "hv": hv,
            "n_mann_cell": n_mann_cell.copy() if n_mann_cell is not None else np.full(h.shape, float(self.n_mann_spin.value()), dtype=np.float64),
            "gpu_active": np.array(bool(backend.gpu_active())),
            "last_mass_total": np.array(float(last_diag.get("mass_total", -1.0) if last_diag else -1.0)),
        }

        self._run_finalizer.finalize_and_persist(
            h=h,
            hu=hu,
            hv=hv,
            n_area=n_area,
            area_model=area_model,
            storage_start_model=storage_start_model,
            source_budget_model=source_budget_model,
            run_duration_s=run_duration_s,
            boundary_flux_budget_model=boundary_flux_budget_model,
            run_id=run_id,
            output_interval_s=output_interval_s,
            line_output_interval_s=line_output_interval_s,
            run_perf_start=run_perf_start,
            run_wallclock_start=run_wallclock_start,
            run_log_start_idx=run_log_start_idx,
            thiessen_forcing=thiessen_forcing,
            rain_stats_acc=rain_stats_acc,
        )
    except Exception as exc:
        self._run_lifecycle.handle_run_failure(
            exc,
            lambda msg: QtWidgets.QMessageBox.critical(self, "2D SWE", msg),
        )
    finally:
        self._run_lifecycle.finalize_cleanup(backend)



def _refresh_layer_combos(self):
    if not _HAVE_QGIS_CORE:
        self.layer_status_lbl.setText("QGIS layer API unavailable in this runtime")
        return

    self._project_layer_state_blocked = True
    try:
        keep_nodes = self.nodes_layer_combo.currentData()
        keep_cells = self.cells_layer_combo.currentData()
        keep_terrain = self.terrain_layer_combo.currentData()
        keep_manning = self.manning_layer_combo.currentData() if hasattr(self, "manning_layer_combo") else None
        keep_cn = self.cn_layer_combo.currentData() if hasattr(self, "cn_layer_combo") else None
        keep_rain_gages = self.rain_gage_layer_combo.currentData() if hasattr(self, "rain_gage_layer_combo") else None
        keep_hyetograph = self.hyetograph_layer_combo.currentData() if hasattr(self, "hyetograph_layer_combo") else None
        keep_storm_area = self.storm_area_layer_combo.currentData() if hasattr(self, "storm_area_layer_combo") else None
        keep_topo_nodes = self.topo_nodes_combo.currentData() if hasattr(self, "topo_nodes_combo") else None
        keep_topo_arcs = self.topo_arcs_combo.currentData() if hasattr(self, "topo_arcs_combo") else None
        keep_topo_regions = self.topo_regions_combo.currentData() if hasattr(self, "topo_regions_combo") else None
        keep_topo_constraints = self.topo_constraints_combo.currentData() if hasattr(self, "topo_constraints_combo") else None
        keep_topo_quad_edges = self.topo_quad_edges_combo.currentData() if hasattr(self, "topo_quad_edges_combo") else None
        keep_bc_lines = self.bc_lines_layer_combo.currentData() if hasattr(self, "bc_lines_layer_combo") else None
        keep_internal_flow = self.internal_flow_layer_combo.currentData() if hasattr(self, "internal_flow_layer_combo") else None
        keep_sample_lines = self.sample_lines_layer_combo.currentData() if hasattr(self, "sample_lines_layer_combo") else None
        keep_drain_nodes = self.drain_nodes_layer_combo.currentData() if hasattr(self, "drain_nodes_layer_combo") else None
        keep_drain_links = self.drain_links_layer_combo.currentData() if hasattr(self, "drain_links_layer_combo") else None
        keep_drain_inlets = self.drain_inlets_layer_combo.currentData() if hasattr(self, "drain_inlets_layer_combo") else None
        keep_drain_node_inlets = self.drain_node_inlets_layer_combo.currentData() if hasattr(self, "drain_node_inlets_layer_combo") else None
        keep_structures = self.structures_layer_combo.currentData() if hasattr(self, "structures_layer_combo") else None
        keep_3d_obj_instances = self.experimental_3d_obj_layer_combo.currentData() if hasattr(self, "experimental_3d_obj_layer_combo") else None
        keep_3d_obj_inside_points = self.experimental_3d_obj_inside_points_layer_combo.currentData() if hasattr(self, "experimental_3d_obj_inside_points_layer_combo") else None

        self.nodes_layer_combo.clear()
        self.cells_layer_combo.clear()
        self.terrain_layer_combo.clear()
        if hasattr(self, "manning_layer_combo"):
            self.manning_layer_combo.clear()
            self.manning_layer_combo.addItem("(none)", None)
        if hasattr(self, "cn_layer_combo"):
            self.cn_layer_combo.clear()
            self.cn_layer_combo.addItem("(none)", None)
        if hasattr(self, "rain_gage_layer_combo"):
            self.rain_gage_layer_combo.clear()
            self.rain_gage_layer_combo.addItem("(none)", None)
        if hasattr(self, "hyetograph_layer_combo"):
            self.hyetograph_layer_combo.clear()
            self.hyetograph_layer_combo.addItem("(none)", None)
        if hasattr(self, "storm_area_layer_combo"):
            self.storm_area_layer_combo.clear()
            self.storm_area_layer_combo.addItem("(none)", None)
        if hasattr(self, "sample_lines_layer_combo"):
            self.sample_lines_layer_combo.clear()
            self.sample_lines_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_nodes_layer_combo"):
            self.drain_nodes_layer_combo.clear()
            self.drain_nodes_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_links_layer_combo"):
            self.drain_links_layer_combo.clear()
            self.drain_links_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_inlets_layer_combo"):
            self.drain_inlets_layer_combo.clear()
            self.drain_inlets_layer_combo.addItem("(none)", None)
        if hasattr(self, "drain_node_inlets_layer_combo"):
            self.drain_node_inlets_layer_combo.clear()
            self.drain_node_inlets_layer_combo.addItem("(none)", None)
        if hasattr(self, "structures_layer_combo"):
            self.structures_layer_combo.clear()
            self.structures_layer_combo.addItem("(none)", None)
        if hasattr(self, "experimental_3d_obj_layer_combo"):
            self.experimental_3d_obj_layer_combo.clear()
            self.experimental_3d_obj_layer_combo.addItem("(none)", None)
        if hasattr(self, "experimental_3d_obj_inside_points_layer_combo"):
            self.experimental_3d_obj_inside_points_layer_combo.clear()
            self.experimental_3d_obj_inside_points_layer_combo.addItem("(none)", None)
        if hasattr(self, "topo_nodes_combo"):
            self.topo_nodes_combo.clear()
        if hasattr(self, "topo_arcs_combo"):
            self.topo_arcs_combo.clear()
        if hasattr(self, "topo_regions_combo"):
            self.topo_regions_combo.clear()
        if hasattr(self, "topo_constraints_combo"):
            self.topo_constraints_combo.clear()
            self.topo_constraints_combo.addItem("(none)", None)
        if hasattr(self, "topo_quad_edges_combo"):
            self.topo_quad_edges_combo.clear()
            self.topo_quad_edges_combo.addItem("(none)", None)
        if hasattr(self, "bc_lines_layer_combo"):
            self.bc_lines_layer_combo.clear()
            self.bc_lines_layer_combo.addItem("(none)", None)
        if hasattr(self, "internal_flow_layer_combo"):
            self.internal_flow_layer_combo.clear()
            self.internal_flow_layer_combo.addItem("(none)", None)

        for lyr in self._iter_project_layers():
            try:
                if isinstance(lyr, QgsVectorLayer):
                    self._configure_swe2d_layer_editors(lyr)
                    if hasattr(self, "internal_flow_layer_combo"):
                        self.internal_flow_layer_combo.addItem(lyr.name(), lyr.id())
                    geom_type = lyr.geometryType()
                    if geom_type == QgsWkbTypes.GeometryType.PointGeometry:
                        self.nodes_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "rain_gage_layer_combo"):
                            self.rain_gage_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_nodes_combo"):
                            self.topo_nodes_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_nodes_layer_combo"):
                            self.drain_nodes_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "experimental_3d_obj_layer_combo"):
                            self.experimental_3d_obj_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "experimental_3d_obj_inside_points_layer_combo"):
                            self.experimental_3d_obj_inside_points_layer_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type == QgsWkbTypes.GeometryType.PolygonGeometry:
                        self.cells_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "manning_layer_combo"):
                            self.manning_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "cn_layer_combo"):
                            self.cn_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "storm_area_layer_combo"):
                            self.storm_area_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_regions_combo"):
                            self.topo_regions_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_constraints_combo"):
                            self.topo_constraints_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type in (
                        QgsWkbTypes.GeometryType.UnknownGeometry,
                        getattr(QgsWkbTypes.GeometryType, "NullGeometry", QgsWkbTypes.GeometryType.UnknownGeometry),
                    ):
                        lname = str(lyr.name() or "").lower()
                        if hasattr(self, "hyetograph_layer_combo") and "hyetograph" in lname:
                            self.hyetograph_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_inlets_layer_combo") and "drainage_inlets" in lname:
                            self.drain_inlets_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_node_inlets_layer_combo") and "drainage_node_inlets" in lname:
                            self.drain_node_inlets_layer_combo.addItem(lyr.name(), lyr.id())
                    elif geom_type == QgsWkbTypes.GeometryType.LineGeometry:
                        if hasattr(self, "sample_lines_layer_combo"):
                            self.sample_lines_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_arcs_combo"):
                            self.topo_arcs_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "topo_quad_edges_combo"):
                            self.topo_quad_edges_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "bc_lines_layer_combo"):
                            self.bc_lines_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "drain_links_layer_combo"):
                            self.drain_links_layer_combo.addItem(lyr.name(), lyr.id())
                        if hasattr(self, "structures_layer_combo"):
                            self.structures_layer_combo.addItem(lyr.name(), lyr.id())
                elif isinstance(lyr, QgsRasterLayer):
                    self.terrain_layer_combo.addItem(lyr.name(), lyr.id())
            except Exception:
                continue

        hydro_layer_map = {}
        for lyr in self._iter_project_layers():
            if isinstance(lyr, QgsVectorLayer):
                hydro_layer_map[str(lyr.name())] = str(lyr.name())
        for lyr in self._iter_project_layers():
            if isinstance(lyr, QgsVectorLayer) and "bc_lines" in str(lyr.name()).lower():
                self._set_value_map_editor(lyr, "hydrograph_layer", hydro_layer_map)

        def _restore(combo, keep_id):
            if not keep_id:
                return
            idx = combo.findData(keep_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        _restore(self.nodes_layer_combo, keep_nodes)
        _restore(self.cells_layer_combo, keep_cells)
        _restore(self.terrain_layer_combo, keep_terrain)
        if hasattr(self, "manning_layer_combo"):
            _restore(self.manning_layer_combo, keep_manning)
        if hasattr(self, "cn_layer_combo"):
            _restore(self.cn_layer_combo, keep_cn)
        if hasattr(self, "rain_gage_layer_combo"):
            _restore(self.rain_gage_layer_combo, keep_rain_gages)
        if hasattr(self, "hyetograph_layer_combo"):
            _restore(self.hyetograph_layer_combo, keep_hyetograph)
        if hasattr(self, "storm_area_layer_combo"):
            _restore(self.storm_area_layer_combo, keep_storm_area)
        if hasattr(self, "topo_nodes_combo"):
            _restore(self.topo_nodes_combo, keep_topo_nodes)
        if hasattr(self, "topo_arcs_combo"):
            _restore(self.topo_arcs_combo, keep_topo_arcs)
        if hasattr(self, "topo_regions_combo"):
            _restore(self.topo_regions_combo, keep_topo_regions)
        if hasattr(self, "topo_constraints_combo") and keep_topo_constraints is not None:
            _restore(self.topo_constraints_combo, keep_topo_constraints)
        if hasattr(self, "topo_quad_edges_combo") and keep_topo_quad_edges is not None:
            _restore(self.topo_quad_edges_combo, keep_topo_quad_edges)
        if hasattr(self, "bc_lines_layer_combo") and keep_bc_lines is not None:
            _restore(self.bc_lines_layer_combo, keep_bc_lines)
        if hasattr(self, "internal_flow_layer_combo") and keep_internal_flow is not None:
            _restore(self.internal_flow_layer_combo, keep_internal_flow)
        if hasattr(self, "sample_lines_layer_combo") and keep_sample_lines is not None:
            _restore(self.sample_lines_layer_combo, keep_sample_lines)
        if hasattr(self, "drain_nodes_layer_combo") and keep_drain_nodes is not None:
            _restore(self.drain_nodes_layer_combo, keep_drain_nodes)
        if hasattr(self, "drain_links_layer_combo") and keep_drain_links is not None:
            _restore(self.drain_links_layer_combo, keep_drain_links)
        if hasattr(self, "drain_inlets_layer_combo") and keep_drain_inlets is not None:
            _restore(self.drain_inlets_layer_combo, keep_drain_inlets)
        if hasattr(self, "drain_node_inlets_layer_combo") and keep_drain_node_inlets is not None:
            _restore(self.drain_node_inlets_layer_combo, keep_drain_node_inlets)
        if hasattr(self, "structures_layer_combo") and keep_structures is not None:
            _restore(self.structures_layer_combo, keep_structures)
        if hasattr(self, "experimental_3d_obj_layer_combo") and keep_3d_obj_instances is not None:
            _restore(self.experimental_3d_obj_layer_combo, keep_3d_obj_instances)
        if hasattr(self, "experimental_3d_obj_inside_points_layer_combo") and keep_3d_obj_inside_points is not None:
            _restore(self.experimental_3d_obj_inside_points_layer_combo, keep_3d_obj_inside_points)

        self._update_unit_system_from_crs()
        self._refresh_layer_group_combo()
        self._update_topology_control_summary()
    finally:
        self._project_layer_state_blocked = False

    self._restore_project_layer_bindings()
    self._persist_project_layer_bindings()



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
        cell_min_z = np.empty(n_cells, dtype=np.float64)
        for i in range(n_cells):
            s, e = int(offsets[i]), int(offsets[i + 1])
            ring = face_nodes_arr[s:e].astype(np.int32)
            fp_idx[i, : e - s] = ring
            cell_cx[i] = float(np.mean(node_x[ring]))
            cell_cy[i] = float(np.mean(node_y[ring]))
            cell_min_z[i] = float(np.min(node_z[ring]))
    else:
        tri = cell_nodes_tri.reshape(-1, 3).astype(np.int32)
        n_cells = tri.shape[0]
        fp_idx = tri
        cell_cx = np.mean(node_x[tri], axis=1)
        cell_cy = np.mean(node_y[tri], axis=1)
        cell_min_z = np.min(node_z[tri], axis=1)

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
            except Exception:
                pass
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
        # Minimum bed elevation per cell
        area_grp.create_dataset(
            "Cells Minimum Elevation",
            data=cell_min_z.astype(np.float32),
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
                wse_arr[ti] = (h_f + cell_min_z[:n_cells]).astype(np.float32)
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



def _refresh_velocity_vectors_overlay(self, t_s: float):
    self._velocity_overlay_refresh_token += 1
    refresh_token = int(self._velocity_overlay_refresh_token)
    frame_t0 = time.perf_counter()
    fetch_ms = 0.0
    build_ms = 0.0
    draw_ms = 0.0
    total_vectors = 0
    total_sources = 0
    panel = getattr(self, "_results_panel", None)
    if panel is None or not panel.velocity_overlay_enabled():
        self._clear_velocity_vectors_layers()
        return
    if not _HAVE_QGIS_CORE:
        self._clear_velocity_vectors_layers()
        return

    if not self._velocity_overlay_sources:
        self._clear_velocity_vectors_layers()
        return

    builder = self._get_velocity_vector_builder()
    if builder is None:
        self._clear_velocity_vectors_layers()
        return

    stride = max(1, int(panel.velocity_density_stride()))
    min_speed = max(0.0, float(panel.velocity_min_speed()))

    for source in list(self._velocity_overlay_sources):
        if refresh_token != self._velocity_overlay_refresh_token:
            return
        total_sources += 1
        gpkg_path = str(source.get("gpkg_path", "")).strip()
        run_id = str(source.get("run_id", "")).strip()
        table_name = str(source.get("table_name", "swe2d_mesh_results")).strip() or "swe2d_mesh_results"
        source_key = str(source.get("key", "")).strip()
        if not gpkg_path or not run_id or not source_key or not os.path.exists(gpkg_path):
            continue

        lyr = self._velocity_vectors_layer_for_source(source)
        if lyr is None:
            continue

        cell_to_fid = self._velocity_overlay_feature_ids.get(source_key)
        if cell_to_fid is None:
            cell_to_fid = {}
            self._velocity_overlay_feature_ids[source_key] = cell_to_fid

        dp = lyr.dataProvider()
        if not cell_to_fid:
            try:
                idx_cell = lyr.fields().indexFromName("cell_id")
                if idx_cell >= 0:
                    for f in lyr.getFeatures():
                        try:
                            cid = int(f["cell_id"])
                            cell_to_fid[cid] = int(f.id())
                        except Exception:
                            continue
            except Exception:
                pass

        _tf0 = time.perf_counter()
        snap = builder.load_snapshot(
            gpkg_path,
            run_id,
            float(t_s),
            t_tol=1.0,
            table_name=table_name,
        )
        fetch_ms += (time.perf_counter() - _tf0) * 1000.0
        if snap is None:
            lyr.triggerRepaint()
            continue

        if not self._velocity_overlay_source_mode_logged.get(source_key, False):
            try:
                support = self._velocity_data_support_for_run(gpkg_path, run_id, table_name)
                if str(getattr(snap, "source", "")) == "face_flux_reconstruction":
                    self._log(
                        "Velocity rendering mode: using face-centered reconstruction "
                        f"(run_id={run_id}, table={table_name}, face_table={support.get('face_table')}, "
                        f"face_rows={int(support.get('face_rows', 0))}, cell_rows={int(support.get('cell_rows', 0))})."
                    )
                else:
                    self._log(
                        "Velocity rendering mode: using cell-centered hu/hv "
                        f"(run_id={run_id}, table={table_name}, no usable face rows detected; "
                        f"cell_rows={int(support.get('cell_rows', 0))})."
                    )
            except Exception:
                pass
            self._velocity_overlay_source_mode_logged[source_key] = True

        cell_xy, base_len = self._mesh_cell_centers_for_gpkg(
            gpkg_path,
            run_id=run_id,
            table_name=table_name,
        )
        if not cell_xy:
            lyr.triggerRepaint()
            continue

        _tb0 = time.perf_counter()
        vecs = builder.build_vectors(
            snapshot=snap,
            cell_xy=cell_xy,
            stride=stride,
            min_depth=1.0e-6,
            min_speed=min_speed,
        )
        build_ms += (time.perf_counter() - _tb0) * 1000.0
        if not vecs:
            existing = list(cell_to_fid.values())
            if existing:
                dp.deleteFeatures(existing)
                self._velocity_overlay_feature_ids[source_key] = {}
            lyr.triggerRepaint()
            continue
        total_vectors += int(len(vecs))

        source_color = self._velocity_source_color(source_key)
        idx_speed = lyr.fields().indexFromName("speed")
        idx_u = lyr.fields().indexFromName("u")
        idx_v = lyr.fields().indexFromName("v")
        idx_ang = lyr.fields().indexFromName("angle_deg")
        idx_src = lyr.fields().indexFromName("source")
        idx_color = lyr.fields().indexFromName("color")
        idx_width = lyr.fields().indexFromName("width")

        new_feats = []
        geom_updates = {}
        attr_updates = {}
        seen_cells = set()
        for v in vecs:
            speed = float(v.get("speed", 0.0))
            if speed <= 1.0e-12:
                continue
            cid = int(v.get("cell_id", -1))
            if cid < 0:
                continue
            seen_cells.add(cid)
            dir_u = float(v.get("u", 0.0)) / speed
            dir_v = float(v.get("v", 0.0)) / speed
            line_len = float(base_len) * min(6.0, max(1.0, 1.25 + 1.15 * speed))

            x0 = float(v.get("x", 0.0))
            y0 = float(v.get("y", 0.0))
            x1 = x0 + dir_u * line_len
            y1 = y0 + dir_v * line_len
            geom = QgsGeometry.fromPolylineXY([
                QgsPointXY(x0, y0),
                QgsPointXY(x1, y1),
            ])

            fid = cell_to_fid.get(cid)
            if fid is not None:
                geom_updates[fid] = geom
                updates = {}
                if idx_speed >= 0:
                    updates[idx_speed] = speed
                if idx_u >= 0:
                    updates[idx_u] = float(v.get("u", 0.0))
                if idx_v >= 0:
                    updates[idx_v] = float(v.get("v", 0.0))
                if idx_ang >= 0:
                    updates[idx_ang] = float(v.get("angle_deg", 0.0))
                if idx_src >= 0:
                    updates[idx_src] = str(source.get("label", ""))
                if idx_color >= 0:
                    updates[idx_color] = source_color
                if idx_width >= 0:
                    updates[idx_width] = 0.8
                if updates:
                    attr_updates[fid] = updates
                continue

            feat = QgsFeature(lyr.fields())
            feat.setAttribute("cell_id", cid)
            feat.setAttribute("speed", speed)
            feat.setAttribute("u", float(v.get("u", 0.0)))
            feat.setAttribute("v", float(v.get("v", 0.0)))
            feat.setAttribute("angle_deg", float(v.get("angle_deg", 0.0)))
            feat.setAttribute("source", str(source.get("label", "")))
            feat.setAttribute("color", source_color)
            feat.setAttribute("width", 0.8)
            feat.setGeometry(geom)
            new_feats.append(feat)

        _td0 = time.perf_counter()
        if geom_updates:
            dp.changeGeometryValues(geom_updates)
        if attr_updates:
            dp.changeAttributeValues(attr_updates)
        if new_feats:
            ok, added = dp.addFeatures(new_feats)
            if ok:
                for f in added:
                    try:
                        cid = int(f["cell_id"])
                        cell_to_fid[cid] = int(f.id())
                    except Exception:
                        continue

        stale_cells = [cid for cid in list(cell_to_fid.keys()) if cid not in seen_cells]
        if stale_cells:
            stale_fids = [cell_to_fid[cid] for cid in stale_cells if cid in cell_to_fid]
            if stale_fids:
                dp.deleteFeatures(stale_fids)
            for cid in stale_cells:
                cell_to_fid.pop(cid, None)

        if new_feats or stale_cells:
            lyr.updateExtents()
        lyr.triggerRepaint()
        draw_ms += (time.perf_counter() - _td0) * 1000.0

    iface = getattr(self, "_iface", None)
    if iface is not None and hasattr(iface, "mapCanvas"):
        try:
            iface.mapCanvas().refresh()
        except Exception:
            pass

    self._velocity_overlay_frame_counter += 1
    frame_ms = (time.perf_counter() - frame_t0) * 1000.0
    if (
        self._velocity_overlay_frame_counter % max(1, int(self._velocity_overlay_perf_log_every)) == 0
        or frame_ms > 80.0
    ):
        self._log(
            "Velocity overlay perf: "
            f"frame_ms={frame_ms:.1f}, fetch_ms={fetch_ms:.1f}, build_ms={build_ms:.1f}, draw_ms={draw_ms:.1f}, "
            f"sources={total_sources}, vectors={total_vectors}, stride={stride}"
        )



def _refresh_streamline_traces_overlay(self, t_s: float):
    frame_t0 = time.perf_counter()
    fetch_ms = 0.0
    build_ms = 0.0
    draw_ms = 0.0
    total_traces = 0
    total_sources = 0

    panel = getattr(self, "_results_panel", None)
    if panel is None or not hasattr(panel, "streamline_overlay_enabled"):
        self._clear_streamline_traces_layers()
        return
    if not panel.streamline_overlay_enabled():
        self._clear_streamline_traces_layers()
        return
    if not _HAVE_QGIS_CORE:
        self._clear_streamline_traces_layers()
        return
    if not self._velocity_overlay_sources:
        self._clear_streamline_traces_layers()
        return

    builder = self._get_velocity_vector_builder()
    if builder is None:
        self._clear_streamline_traces_layers()
        return

    seed_count = 48
    max_steps = 30
    step_scale = 0.85
    try:
        seed_count = max(4, int(panel.streamline_seed_count()))
    except Exception:
        pass
    try:
        max_steps = max(4, int(panel.streamline_max_steps()))
    except Exception:
        pass
    try:
        step_scale = max(0.05, float(panel.streamline_step_scale()))
    except Exception:
        pass

    seed_stride = max(1, int(panel.velocity_density_stride()))
    min_speed = max(0.0, float(panel.velocity_min_speed()))

    for source in list(self._velocity_overlay_sources):
        total_sources += 1
        gpkg_path = str(source.get("gpkg_path", "")).strip()
        run_id = str(source.get("run_id", "")).strip()
        table_name = str(source.get("table_name", "swe2d_mesh_results")).strip() or "swe2d_mesh_results"
        source_key = str(source.get("key", "")).strip()
        if not gpkg_path or not run_id or not source_key or not os.path.exists(gpkg_path):
            continue

        lyr = self._streamline_traces_layer_for_source(source)
        if lyr is None:
            continue
        dp = lyr.dataProvider()

        existing_ids = [f.id() for f in lyr.getFeatures()]
        if existing_ids:
            try:
                dp.deleteFeatures(existing_ids)
            except Exception:
                pass

        _tf0 = time.perf_counter()
        snap = builder.load_snapshot(
            gpkg_path,
            run_id,
            float(t_s),
            t_tol=1.0,
            table_name=table_name,
        )
        fetch_ms += (time.perf_counter() - _tf0) * 1000.0
        if snap is None:
            lyr.triggerRepaint()
            continue

        cell_xy, _ = self._mesh_cell_centers_for_gpkg(
            gpkg_path,
            run_id=run_id,
            table_name=table_name,
        )
        if not cell_xy:
            lyr.triggerRepaint()
            continue

        _tb0 = time.perf_counter()
        traces = builder.build_streamline_traces(
            snapshot=snap,
            cell_xy=cell_xy,
            seed_count=seed_count,
            max_steps=max_steps,
            step_len_factor=step_scale,
            min_depth=1.0e-6,
            min_speed=min_speed,
            seed_stride=seed_stride,
        )
        build_ms += (time.perf_counter() - _tb0) * 1000.0
        if not traces:
            lyr.triggerRepaint()
            continue

        source_color = self._velocity_source_color(source_key)
        feats = []
        for tr in traces:
            pts = tr.get("points", [])
            if not isinstance(pts, list) or len(pts) < 2:
                continue
            qpts = []
            for xy in pts:
                try:
                    qpts.append(QgsPointXY(float(xy[0]), float(xy[1])))
                except Exception:
                    continue
            if len(qpts) < 2:
                continue

            mean_speed = float(tr.get("mean_speed", 0.0) or 0.0)
            style = builder.style_from_speed(mean_speed)
            feat = QgsFeature(lyr.fields())
            feat.setAttribute("trace_id", int(tr.get("trace_id", len(feats))))
            feat.setAttribute("speed", mean_speed)
            feat.setAttribute("length", float(tr.get("length", 0.0) or 0.0))
            feat.setAttribute("source", str(source.get("label", "")))
            feat.setAttribute("color", source_color)
            feat.setAttribute("width", float(style.get("width", 0.7) or 0.7))
            feat.setGeometry(QgsGeometry.fromPolylineXY(qpts))
            feats.append(feat)

        _td0 = time.perf_counter()
        if feats:
            try:
                dp.addFeatures(feats)
            except Exception:
                pass
            try:
                lyr.updateExtents()
            except Exception:
                pass
            total_traces += int(len(feats))

        lyr.triggerRepaint()
        draw_ms += (time.perf_counter() - _td0) * 1000.0

    iface = getattr(self, "_iface", None)
    if iface is not None and hasattr(iface, "mapCanvas"):
        try:
            iface.mapCanvas().refresh()
        except Exception:
            pass

    self._streamline_overlay_frame_counter += 1
    frame_ms = (time.perf_counter() - frame_t0) * 1000.0
    if (
        self._streamline_overlay_frame_counter % max(1, int(self._streamline_overlay_perf_log_every)) == 0
        or frame_ms > 100.0
    ):
        self._log(
            "Streamline overlay perf: "
            f"frame_ms={frame_ms:.1f}, fetch_ms={fetch_ms:.1f}, build_ms={build_ms:.1f}, draw_ms={draw_ms:.1f}, "
            f"sources={total_sources}, traces={total_traces}, seeds={seed_count}, steps={max_steps}"
        )

# ------------------------------------------------------------------



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
    except Exception:
        pass

    # Canonical schema: list of (layer_name, memory_uri) pairs.
    # Geometry-less tables use "None?" as the URI prefix.
    layer_specs = [
        ("swe2d_topo_nodes",
         f"Point?crs={crs_auth}&field=node_id:integer"),
        ("swe2d_topo_arcs",
         f"LineString?crs={crs_auth}&field=arc_id:integer&field=node0:integer&field=node1:integer"),
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
         "&field=length:double&field=roughness_n:double&field=coeff:double"
         "&field=cd:double&field=opening:double&field=q_pump:double&field=max_flow:double"),
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



def _bind_run_tab_controls(self, run_tab_page: QtWidgets.QWidget) -> None:
    def _ensure_root_layout() -> QtWidgets.QVBoxLayout:
        layout = run_tab_page.layout()
        if isinstance(layout, QtWidgets.QVBoxLayout):
            return layout
        layout = QtWidgets.QVBoxLayout(run_tab_page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        return layout

    def _ensure_run_group() -> QtWidgets.QGroupBox:
        run_group = run_tab_page.findChild(QtWidgets.QGroupBox, "run_group")
        if run_group is not None:
            return run_group
        run_group = QtWidgets.QGroupBox("Run / Output")
        run_group.setObjectName("run_group")
        root_layout = _ensure_root_layout()
        root_layout.addWidget(run_group)
        return run_group

    def _ensure_run_layout(run_group: QtWidgets.QGroupBox) -> QtWidgets.QVBoxLayout:
        layout = run_group.layout()
        if isinstance(layout, QtWidgets.QVBoxLayout):
            return layout
        layout = QtWidgets.QVBoxLayout(run_group)
        layout.setObjectName("run_layout")
        return layout

    run_group = _ensure_run_group()
    run_layout = _ensure_run_layout(run_group)

    run_row = run_tab_page.findChild(QtWidgets.QHBoxLayout, "run_row_layout")
    if run_row is None:
        run_row = QtWidgets.QHBoxLayout()
        run_row.setObjectName("run_row_layout")
        run_layout.insertLayout(0, run_row)

    self.preview_overrides_btn = run_tab_page.findChild(QtWidgets.QPushButton, "preview_overrides_btn")
    if self.preview_overrides_btn is None:
        self.preview_overrides_btn = QtWidgets.QPushButton("Preview Overrides")
        self.preview_overrides_btn.setObjectName("preview_overrides_btn")
    if run_row.indexOf(self.preview_overrides_btn) < 0:
        run_row.addWidget(self.preview_overrides_btn)

    self.run_btn = run_tab_page.findChild(QtWidgets.QPushButton, "run_btn")
    if self.run_btn is None:
        self.run_btn = QtWidgets.QPushButton("Run 2D Model")
        self.run_btn.setObjectName("run_btn")
    if run_row.indexOf(self.run_btn) < 0:
        run_row.addWidget(self.run_btn)

    self.cancel_btn = run_tab_page.findChild(QtWidgets.QPushButton, "cancel_btn")
    if self.cancel_btn is None:
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancel_btn")
    if run_row.indexOf(self.cancel_btn) < 0:
        run_row.addWidget(self.cancel_btn)

    self.progress_bar = run_tab_page.findChild(QtWidgets.QProgressBar, "progress_bar")
    if self.progress_bar is None:
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        run_layout.addWidget(self.progress_bar)
    elif run_layout.indexOf(self.progress_bar) < 0:
        run_layout.addWidget(self.progress_bar)

    snap_row = run_tab_page.findChild(QtWidgets.QHBoxLayout, "run_snapshot_row_layout")
    if snap_row is None:
        snap_row = QtWidgets.QHBoxLayout()
        snap_row.setObjectName("run_snapshot_row_layout")
        run_layout.addLayout(snap_row)

    output_interval_lbl = run_tab_page.findChild(QtWidgets.QLabel, "output_interval_lbl")
    if output_interval_lbl is None:
        output_interval_lbl = QtWidgets.QLabel("Output interval (hr or HH:MM):")
        output_interval_lbl.setObjectName("output_interval_lbl")
    if snap_row.indexOf(output_interval_lbl) < 0:
        snap_row.addWidget(output_interval_lbl)

    self.output_interval_edit = run_tab_page.findChild(QtWidgets.QLineEdit, "output_interval_edit")
    if self.output_interval_edit is None:
        self.output_interval_edit = QtWidgets.QLineEdit("00:30")
        self.output_interval_edit.setObjectName("output_interval_edit")
    if snap_row.indexOf(self.output_interval_edit) < 0:
        snap_row.addWidget(self.output_interval_edit)

    line_output_interval_lbl = run_tab_page.findChild(QtWidgets.QLabel, "line_output_interval_lbl")
    if line_output_interval_lbl is None:
        line_output_interval_lbl = QtWidgets.QLabel("Line output interval:")
        line_output_interval_lbl.setObjectName("line_output_interval_lbl")
    if snap_row.indexOf(line_output_interval_lbl) < 0:
        snap_row.addWidget(line_output_interval_lbl)

    self.line_output_interval_edit = run_tab_page.findChild(QtWidgets.QLineEdit, "line_output_interval_edit")
    if self.line_output_interval_edit is None:
        self.line_output_interval_edit = QtWidgets.QLineEdit("00:05")
        self.line_output_interval_edit.setObjectName("line_output_interval_edit")
    if snap_row.indexOf(self.line_output_interval_edit) < 0:
        snap_row.addWidget(self.line_output_interval_edit)

    self.snapshot_btn = run_tab_page.findChild(QtWidgets.QPushButton, "snapshot_btn")
    if self.snapshot_btn is None:
        self.snapshot_btn = QtWidgets.QPushButton("Take Snapshot")
        self.snapshot_btn.setObjectName("snapshot_btn")
    if snap_row.indexOf(self.snapshot_btn) < 0:
        snap_row.addWidget(self.snapshot_btn)

    self.cancel_btn.setEnabled(False)

    self.progress_bar.setRange(0, 100)
    self.progress_bar.setValue(0)

    self.output_interval_edit.setMaximumWidth(90)
    if not str(self.output_interval_edit.text() or "").strip():
        self.output_interval_edit.setText("00:30")
    self.output_interval_edit.setToolTip(
        "Interval between captured result snapshots during a run.\n"
        "E.g. 00:30 captures every 30 minutes of simulation time."
    )

    self.line_output_interval_edit.setMaximumWidth(90)
    if not str(self.line_output_interval_edit.text() or "").strip():
        self.line_output_interval_edit.setText("00:05")
    self.line_output_interval_edit.setToolTip(
        "Interval for sampled line time-series output capture.\n"
        "Independent from mesh snapshot interval."
    )

    self.snapshot_btn.setToolTip(
        "Write all captured timesteps up to now to a temporary HEC-RAS HDF5 file.\n"
        "The file path is logged in the message panel."
    )

    for btn, cb in (
        (self.run_btn, self._on_run_requested),
        (self.preview_overrides_btn, self._on_preview_overrides),
        (self.cancel_btn, self._on_cancel),
        (self.snapshot_btn, self._on_snapshot),
    ):
        try:
            btn.clicked.disconnect(cb)
        except Exception:
            pass
        btn.clicked.connect(cb)

    try:
        self.experimental_3d_mode_chk.toggled.disconnect(self._sync_experimental_3d_mode_widgets)
    except Exception:
        pass
    try:
        self.experimental_3d_mode_chk.toggled.connect(self._sync_experimental_3d_mode_widgets)
    except Exception:
        pass
    self._sync_experimental_3d_mode_widgets()



def _bind_map_tab_results_controls(self, map_tab_page: QtWidgets.QWidget, map_results_layout: QtWidgets.QGridLayout) -> None:
    def _find_or_create_check(name: str, text: str) -> QtWidgets.QCheckBox:
        w = map_tab_page.findChild(QtWidgets.QCheckBox, name)
        if w is None:
            w = QtWidgets.QCheckBox(text)
            w.setObjectName(name)
        return w

    def _find_or_create_button(name: str, text: str) -> QtWidgets.QPushButton:
        w = map_tab_page.findChild(QtWidgets.QPushButton, name)
        if w is None:
            w = QtWidgets.QPushButton(text)
            w.setObjectName(name)
        return w

    def _find_or_create_combo(name: str) -> QtWidgets.QComboBox:
        w = map_tab_page.findChild(QtWidgets.QComboBox, name)
        if w is None:
            w = QtWidgets.QComboBox()
            w.setObjectName(name)
        return w

    def _find_or_create_double_spin(name: str) -> QtWidgets.QDoubleSpinBox:
        w = map_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
        if w is None:
            w = QtWidgets.QDoubleSpinBox()
            w.setObjectName(name)
        return w

    self.extended_outputs_chk = _find_or_create_check(
        "extended_outputs_chk",
        "Include extended outputs (momentum, qmag, wet mask, Fr, Manning)",
    )
    self.save_mesh_results_to_gpkg_chk = _find_or_create_check(
        "save_mesh_results_to_gpkg_chk",
        "Save mesh snapshot results to GeoPackage",
    )
    self.save_line_results_to_gpkg_chk = _find_or_create_check(
        "save_line_results_to_gpkg_chk",
        "Save sampled line results to GeoPackage",
    )
    self.save_coupling_results_to_gpkg_chk = _find_or_create_check(
        "save_coupling_results_to_gpkg_chk",
        "Save drainage/structure results to GeoPackage",
    )
    self.save_run_log_to_gpkg_chk = _find_or_create_check(
        "save_run_log_to_gpkg_chk",
        "Save run log to GeoPackage",
    )
    self.open_results_viewer_btn = _find_or_create_button("open_results_viewer_btn", "Open 2D Results Viewer")
    self.open_results_panel_btn = _find_or_create_button("open_results_panel_btn", "Results Panel (multi-run)")
    self.high_perf_canvas_overlay_chk = _find_or_create_check(
        "high_perf_canvas_overlay_chk",
        "Show High-Perf Overlay On Map Canvas",
    )
    self.high_perf_canvas_overlay_field_combo = _find_or_create_combo("high_perf_canvas_overlay_field_combo")
    self.high_perf_canvas_overlay_cmap_combo = _find_or_create_combo("high_perf_canvas_overlay_cmap_combo")
    self.high_perf_canvas_overlay_lock_canvas_chk = _find_or_create_check(
        "high_perf_canvas_overlay_lock_canvas_chk",
        "Lock overlay resolution to current canvas size",
    )
    self.high_perf_canvas_overlay_res_combo = _find_or_create_combo("high_perf_canvas_overlay_res_combo")
    self.high_perf_canvas_overlay_auto_contrast_chk = _find_or_create_check(
        "high_perf_canvas_overlay_auto_contrast_chk",
        "Auto contrast",
    )
    self.high_perf_canvas_overlay_opacity_spin = _find_or_create_double_spin("high_perf_canvas_overlay_opacity_spin")

    if map_results_layout.indexOf(self.extended_outputs_chk) < 0:
        map_results_layout.addWidget(self.extended_outputs_chk, 0, 0, 1, 2)
    if map_results_layout.indexOf(self.save_mesh_results_to_gpkg_chk) < 0:
        map_results_layout.addWidget(self.save_mesh_results_to_gpkg_chk, 1, 0, 1, 2)
    if map_results_layout.indexOf(self.save_line_results_to_gpkg_chk) < 0:
        map_results_layout.addWidget(self.save_line_results_to_gpkg_chk, 2, 0, 1, 2)
    if map_results_layout.indexOf(self.save_coupling_results_to_gpkg_chk) < 0:
        map_results_layout.addWidget(self.save_coupling_results_to_gpkg_chk, 3, 0, 1, 2)
    if map_results_layout.indexOf(self.save_run_log_to_gpkg_chk) < 0:
        map_results_layout.addWidget(self.save_run_log_to_gpkg_chk, 4, 0, 1, 2)
    if map_results_layout.indexOf(self.open_results_viewer_btn) < 0:
        map_results_layout.addWidget(self.open_results_viewer_btn, 5, 0, 1, 2)
    if map_results_layout.indexOf(self.open_results_panel_btn) < 0:
        map_results_layout.addWidget(self.open_results_panel_btn, 6, 0, 1, 2)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_chk) < 0:
        map_results_layout.addWidget(self.high_perf_canvas_overlay_chk, 7, 0, 1, 2)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_field_combo) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay field:"), 8, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_field_combo, 8, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_cmap_combo) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay colormap:"), 9, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_cmap_combo, 9, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_lock_canvas_chk) < 0:
        map_results_layout.addWidget(self.high_perf_canvas_overlay_lock_canvas_chk, 10, 0, 1, 2)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_res_combo) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay resolution:"), 11, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_res_combo, 11, 1)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_auto_contrast_chk) < 0:
        map_results_layout.addWidget(self.high_perf_canvas_overlay_auto_contrast_chk, 12, 0, 1, 2)
    if map_results_layout.indexOf(self.high_perf_canvas_overlay_opacity_spin) < 0:
        map_results_layout.addWidget(QtWidgets.QLabel("High-perf overlay opacity:"), 13, 0)
        map_results_layout.addWidget(self.high_perf_canvas_overlay_opacity_spin, 13, 1)

    self.extended_outputs_chk.setChecked(True)
    self.save_mesh_results_to_gpkg_chk.setChecked(True)
    self.save_line_results_to_gpkg_chk.setChecked(True)
    self.save_coupling_results_to_gpkg_chk.setChecked(True)
    self.save_run_log_to_gpkg_chk.setChecked(True)
    self.open_results_panel_btn.setToolTip("Open the dockable multi-run results panel")

    self.high_perf_canvas_overlay_chk.setChecked(False)
    self.high_perf_canvas_overlay_field_combo.clear()
    self.high_perf_canvas_overlay_field_combo.addItem("Depth", "depth")
    self.high_perf_canvas_overlay_field_combo.addItem("Velocity", "speed")
    self.high_perf_canvas_overlay_field_combo.addItem("Water Surface", "wse")
    self.high_perf_canvas_overlay_cmap_combo.clear()
    self.high_perf_canvas_overlay_cmap_combo.addItem("Turbo", "turbo")
    self.high_perf_canvas_overlay_cmap_combo.addItem("Viridis", "viridis")
    self.high_perf_canvas_overlay_cmap_combo.addItem("Plasma", "plasma")
    self.high_perf_canvas_overlay_cmap_combo.addItem("Gray", "gray")
    self.high_perf_canvas_overlay_res_combo.clear()
    self.high_perf_canvas_overlay_res_combo.addItem("640 x 360", (640, 360))
    self.high_perf_canvas_overlay_res_combo.addItem("960 x 540", (960, 540))
    self.high_perf_canvas_overlay_res_combo.addItem("1280 x 720", (1280, 720))
    self.high_perf_canvas_overlay_res_combo.addItem("1920 x 1080", (1920, 1080))
    self.high_perf_canvas_overlay_res_combo.setCurrentIndex(2)
    self.high_perf_canvas_overlay_lock_canvas_chk.setChecked(True)
    self.high_perf_canvas_overlay_auto_contrast_chk.setChecked(True)
    self.high_perf_canvas_overlay_opacity_spin.setDecimals(2)
    self.high_perf_canvas_overlay_opacity_spin.setRange(0.05, 1.0)
    self.high_perf_canvas_overlay_opacity_spin.setSingleStep(0.05)
    self.high_perf_canvas_overlay_opacity_spin.setValue(0.65)

    for sig_obj, cb in (
        (self.open_results_viewer_btn.clicked, self._open_line_results_viewer),
        (self.open_results_panel_btn.clicked, self._show_results_panel),
        (self.high_perf_canvas_overlay_chk.toggled, self._on_high_perf_canvas_overlay_toggled),
        (self.high_perf_canvas_overlay_field_combo.currentIndexChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_cmap_combo.currentIndexChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_lock_canvas_chk.toggled, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_res_combo.currentIndexChanged, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_auto_contrast_chk.toggled, self._on_high_perf_canvas_overlay_style_changed),
        (self.high_perf_canvas_overlay_opacity_spin.valueChanged, self._on_high_perf_canvas_overlay_style_changed),
    ):
        try:
            sig_obj.disconnect(cb)
        except Exception:
            pass
        sig_obj.connect(cb)

    self._on_high_perf_canvas_overlay_style_changed()



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
                except Exception:
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
        except Exception:
            mesh_layer_name = ""
        finally:
            try:
                conn.close()
            except Exception:
                pass

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
                        except Exception:
                            pass
                    except Exception:
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
            except Exception:
                continue

    # Fallback for current active in-memory mesh if mesh layer was unavailable.
    if not cell_xy and self._mesh_data is not None:
        try:
            cx, cy = self._mesh_cell_centroids()
            n_cells = min(int(cx.size), int(cy.size))
            cell_xy = {i: (float(cx[i]), float(cy[i])) for i in range(n_cells)}
            area = np.asarray(self._mesh_cell_areas(), dtype=np.float64)
            base_len = max(0.05, float(np.sqrt(max(float(np.nanmean(area)), 1.0e-9))))
        except Exception:
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
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS swe2d_line_results_runs (
                run_id TEXT PRIMARY KEY,
                created_utc TEXT,
                mesh_interval_s REAL,
                line_interval_s REAL,
                row_count INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS swe2d_line_results_ts (
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
            "CREATE INDEX IF NOT EXISTS idx_swe2d_line_ts_run_line_t ON swe2d_line_results_ts(run_id, line_id, t_s)"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS swe2d_line_results_profile (
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
            "CREATE INDEX IF NOT EXISTS idx_swe2d_line_prof_run_line_t_s ON swe2d_line_results_profile(run_id, line_id, t_s, station_m)"
        )
        cur.execute("DELETE FROM swe2d_line_results_ts WHERE run_id = ?", (run_id,))
        cur.execute("DELETE FROM swe2d_line_results_profile WHERE run_id = ?", (run_id,))
        cur.execute(
            """
            INSERT OR REPLACE INTO swe2d_line_results_runs
            (run_id, created_utc, mesh_interval_s, line_interval_s, row_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(run_id),
                datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
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
            """
            INSERT OR REPLACE INTO swe2d_line_results_ts
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
                """
                INSERT OR REPLACE INTO swe2d_line_results_profile
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



def _bind_right_pane_controls(self, right_pane: QtWidgets.QWidget) -> None:
    def _ensure_root_layout() -> QtWidgets.QVBoxLayout:
        layout = right_pane.layout()
        if isinstance(layout, QtWidgets.QVBoxLayout):
            return layout
        layout = QtWidgets.QVBoxLayout(right_pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        return layout

    right_layout = _ensure_root_layout()

    view_row = right_pane.findChild(QtWidgets.QHBoxLayout, "view_row_layout")
    if view_row is None:
        view_row = QtWidgets.QHBoxLayout()
        view_row.setObjectName("view_row_layout")
        right_layout.insertLayout(0, view_row)

    view_mode_lbl = right_pane.findChild(QtWidgets.QLabel, "view_mode_lbl")
    if view_mode_lbl is None:
        view_mode_lbl = QtWidgets.QLabel("View:")
        view_mode_lbl.setObjectName("view_mode_lbl")
    if view_row.indexOf(view_mode_lbl) < 0:
        view_row.addWidget(view_mode_lbl)

    self.view_mode_combo = right_pane.findChild(QtWidgets.QComboBox, "view_mode_combo")
    if self.view_mode_combo is None:
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.setObjectName("view_mode_combo")
    if view_row.indexOf(self.view_mode_combo) < 0:
        view_row.addWidget(self.view_mode_combo)
    if view_row.count() < 3:
        view_row.addStretch(1)

    prev_view_text = self.view_mode_combo.currentText()
    self.view_mode_combo.blockSignals(True)
    try:
        self.view_mode_combo.clear()
        self.view_mode_combo.addItems(["Mesh", "Depth", "Velocity magnitude"])
        idx = self.view_mode_combo.findText(prev_view_text)
        if idx < 0:
            idx = 0
        self.view_mode_combo.setCurrentIndex(idx)
    finally:
        self.view_mode_combo.blockSignals(False)
    try:
        self.view_mode_combo.currentIndexChanged.disconnect(self._refresh_plot)
    except Exception:
        pass
    self.view_mode_combo.currentIndexChanged.connect(self._refresh_plot)

    popout_row = right_pane.findChild(QtWidgets.QHBoxLayout, "popout_row_layout")
    if popout_row is None:
        popout_row = QtWidgets.QHBoxLayout()
        popout_row.setObjectName("popout_row_layout")
        right_layout.insertLayout(1, popout_row)

    self.detach_mesh_view_btn = right_pane.findChild(QtWidgets.QPushButton, "detach_mesh_view_btn")
    if self.detach_mesh_view_btn is None:
        self.detach_mesh_view_btn = QtWidgets.QPushButton("Detach Mesh View")
        self.detach_mesh_view_btn.setObjectName("detach_mesh_view_btn")
    if popout_row.indexOf(self.detach_mesh_view_btn) < 0:
        popout_row.addWidget(self.detach_mesh_view_btn)

    self.detach_runtime_log_btn = right_pane.findChild(QtWidgets.QPushButton, "detach_runtime_log_btn")
    if self.detach_runtime_log_btn is None:
        self.detach_runtime_log_btn = QtWidgets.QPushButton("Detach Runtime Log")
        self.detach_runtime_log_btn.setObjectName("detach_runtime_log_btn")
    if popout_row.indexOf(self.detach_runtime_log_btn) < 0:
        popout_row.addWidget(self.detach_runtime_log_btn)
    if popout_row.count() < 3:
        popout_row.addStretch(1)

    for btn, cb in (
        (self.detach_mesh_view_btn, self._open_detached_mesh_view),
        (self.detach_runtime_log_btn, self._open_detached_runtime_log),
    ):
        try:
            btn.clicked.disconnect(cb)
        except Exception:
            pass
        btn.clicked.connect(cb)

    self._right_vertical_split = right_pane.findChild(QtWidgets.QSplitter, "right_vertical_split")
    if self._right_vertical_split is None:
        self._right_vertical_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self._right_vertical_split.setObjectName("right_vertical_split")
    self._right_vertical_split.setOrientation(QtCore.Qt.Orientation.Vertical)
    self._right_vertical_split.setChildrenCollapsible(False)
    if right_layout.indexOf(self._right_vertical_split) < 0:
        right_layout.addWidget(self._right_vertical_split, stretch=1)

    right_plot_host = right_pane.findChild(QtWidgets.QWidget, "right_plot_host")
    if right_plot_host is None:
        right_plot_host = QtWidgets.QWidget()
        right_plot_host.setObjectName("right_plot_host")
    if self._right_vertical_split.indexOf(right_plot_host) < 0:
        self._right_vertical_split.insertWidget(0, right_plot_host)

    plot_layout = right_plot_host.layout()
    if not isinstance(plot_layout, QtWidgets.QVBoxLayout):
        plot_layout = QtWidgets.QVBoxLayout(right_plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
    while plot_layout.count():
        item = plot_layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

    if self._have_mpl:
        self._fig = self._Figure(figsize=(6.4, 4.2), tight_layout=True)
        self._canvas = self._FigureCanvas(self._fig)
        self._canvas.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        try:
            self._canvas.customContextMenuRequested.disconnect()
        except Exception:
            pass
        self._canvas.customContextMenuRequested.connect(
            lambda pos: self._show_panel_detach_menu("mesh", self._canvas.mapToGlobal(pos))
        )
        plot_layout.addWidget(self._canvas)
    else:
        self._fig = None
        self._canvas = None
        no_plot = QtWidgets.QLabel("Matplotlib Qt backend not available; results shown in text log only.")
        no_plot.setWordWrap(True)
        plot_layout.addWidget(no_plot)

    self.log_view = right_pane.findChild(QtWidgets.QPlainTextEdit, "log_view")
    if self.log_view is None:
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setObjectName("log_view")
    if self._right_vertical_split.indexOf(self.log_view) < 0:
        self._right_vertical_split.addWidget(self.log_view)
    self.log_view.setReadOnly(True)
    self.log_view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
    try:
        self.log_view.customContextMenuRequested.disconnect()
    except Exception:
        pass
    self.log_view.customContextMenuRequested.connect(
        lambda pos: self._show_panel_detach_menu("log", self.log_view.mapToGlobal(pos))
    )
    self._right_vertical_split.setSizes([520, 220])



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
    except Exception:
        pass

    nodes = QgsVectorLayer(f"Point?crs={crs_auth}&field=node_id:integer", "swe2d_topo_nodes", "memory")
    arcs = QgsVectorLayer(f"LineString?crs={crs_auth}&field=arc_id:integer&field=node0:integer&field=node1:integer", "swe2d_topo_arcs", "memory")
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
        f"LineString?crs={crs_auth}&field=structure_id:string(64)&field=structure_type:integer&field=crest_elev:double&field=enabled:integer&field=width:double&field=height:double&field=diameter:double&field=length:double&field=roughness_n:double&field=coeff:double&field=cd:double&field=opening:double&field=q_pump:double&field=max_flow:double",
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



def _update_topology_control_summary(self):
    if not hasattr(self, "topo_controls_summary_lbl"):
        return

    backend_name = str(self.topo_backend_combo.currentData() or "structured") if hasattr(self, "topo_backend_combo") else "structured"
    regions_layer = self._combo_layer(self.topo_regions_combo, "vector") if hasattr(self, "topo_regions_combo") else None
    constraints_layer = self._combo_layer(self.topo_constraints_combo, "vector") if hasattr(self, "topo_constraints_combo") else None
    quad_edges_layer = self._combo_layer(self.topo_quad_edges_combo, "vector") if hasattr(self, "topo_quad_edges_combo") else None

    if backend_name == "gmsh":
        backend_hint = (
            "Gmsh: use multiple region polygons for multiblock meshes. "
            "Set region cell_type to 'cartesian' or 'quadrilateral' and populate edge_len_1..4 "
            "for per-edge structured spacing. Opposite edges are matched automatically. "
            "Region interior rings plus empty regions/constraints are meshed as cutout holes."
        )
    elif backend_name == "tqmesh":
        backend_hint = (
            "TQMesh: use multiple region polygons for blockwise target_size and cell_type. "
            "Use quad-edge lines with n_layers, first_height, and growth_rate for transition layers. "
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
        "retry scales={size_scales}, smooth increments={smooth_increments}; "
        "Gmsh loop={gmsh_loop}, attempts={attempts}, budget={budget:.1f}s."
    ).format(
        min_angle=float(self.topo_quality_min_angle_spin.value()) if hasattr(self, "topo_quality_min_angle_spin") else 0.0,
        max_aspect=float(self.topo_quality_max_aspect_spin.value()) if hasattr(self, "topo_quality_max_aspect_spin") else 0.0,
        max_non_orth=float(self.topo_quality_max_non_orth_spin.value()) if hasattr(self, "topo_quality_max_non_orth_spin") else 0.0,
        min_area=str(self.topo_quality_min_area_edit.text()).strip() if hasattr(self, "topo_quality_min_area_edit") else "0",
        strict="on" if getattr(self, "topo_quality_strict_chk", None) is not None and self.topo_quality_strict_chk.isChecked() else "off",
        size_scales=str(self.topo_quality_size_scales_edit.text()).strip() if hasattr(self, "topo_quality_size_scales_edit") else "1.0",
        smooth_increments=str(self.topo_quality_smooth_increments_edit.text()).strip() if hasattr(self, "topo_quality_smooth_increments_edit") else "0",
        gmsh_loop="on" if getattr(self, "topo_gmsh_quality_enable_chk", None) is not None and self.topo_gmsh_quality_enable_chk.isChecked() else "off",
        attempts=int(self.topo_gmsh_quality_max_iters_spin.value()) if hasattr(self, "topo_gmsh_quality_max_iters_spin") else 0,
        budget=float(self.topo_gmsh_quality_time_limit_spin.value()) if hasattr(self, "topo_gmsh_quality_time_limit_spin") else 0.0,
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
                        except Exception:
                            edge_ok = False
                            break
                    if not edge_ok:
                        missing_edge_lengths += 1
                if ctype == "empty":
                    empty_count += 1
                if "target_size" in region_fields and ft["target_size"] not in (None, ""):
                    try:
                        size_values.add(round(float(ft["target_size"]), 6))
                    except Exception:
                        pass
            details.append(f"regions={region_count}")
            if cartesian_count > 0:
                details.append(f"structured-block-regions={cartesian_count}")
            if empty_count > 0:
                details.append(f"empty-regions={empty_count}")
            if len(size_values) > 1:
                details.append(f"multi-block sizes={len(size_values)}")
            if missing_edge_lengths > 0:
                details.append(f"structured regions missing edge_len_1..4={missing_edge_lengths}")
        except Exception:
            pass

    if constraints_layer is not None and getattr(self, "topo_constraints_combo", None) is not None and self.topo_constraints_combo.currentData() is not None:
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
        except Exception:
            pass

    if quad_edges_layer is not None and getattr(self, "topo_quad_edges_combo", None) is not None and self.topo_quad_edges_combo.currentData() is not None:
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
        except Exception:
            pass

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
    except Exception:
        pass

    is_region = "topo_regions" in lname or lname.endswith("swe2d_topo_regions")
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
        self._set_expression_constraint(layer, "node_type", '"node_type" IN (\'junction\',\'outfall\',\'storage\',\'inlet\')')
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
        self._set_expression_constraint(layer, "link_type", '"link_type" IN (\'conduit\',\'lateral_simple\',\'pump\',\'weir\',\'orifice\')')
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
        self._set_expression_constraint(layer, "structure_id", 'length(trim("structure_id")) > 0')
        self._set_expression_constraint(layer, "structure_type", '"structure_type" IN (1,2,3,4,5)')
        self._set_expression_constraint(layer, "enabled", '"enabled" IS NULL OR "enabled" IN (0,1)')



def _bind_model_tab_solver_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
    def _ensure_row(label: str, widget: QtWidgets.QWidget) -> None:
        if param_form.indexOf(widget) >= 0:
            return
        param_form.addRow(label, widget)

    def _find_or_create_combo(name: str, label: str) -> QtWidgets.QComboBox:
        w = model_tab_page.findChild(QtWidgets.QComboBox, name)
        if w is None:
            w = QtWidgets.QComboBox()
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    def _find_or_create_line_edit(name: str, label: str, text: str = "") -> QtWidgets.QLineEdit:
        w = model_tab_page.findChild(QtWidgets.QLineEdit, name)
        if w is None:
            w = QtWidgets.QLineEdit(text)
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    self.internal_flow_layer_combo = _find_or_create_combo("internal_flow_layer_combo", "Internal flow layer:")
    prev_data = self.internal_flow_layer_combo.currentData()
    prev_text = self.internal_flow_layer_combo.currentText()
    self.internal_flow_layer_combo.blockSignals(True)
    try:
        self.internal_flow_layer_combo.clear()
        self.internal_flow_layer_combo.addItem("(none)", None)
        idx = self.internal_flow_layer_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.internal_flow_layer_combo.findText(prev_text)
        if idx >= 0:
            self.internal_flow_layer_combo.setCurrentIndex(idx)
    finally:
        self.internal_flow_layer_combo.blockSignals(False)

    self.internal_flow_field_edit = _find_or_create_line_edit(
        "internal_flow_field_edit", "Internal flow field:", "q_cms"
    )
    if not str(self.internal_flow_field_edit.text() or "").strip():
        self.internal_flow_field_edit.setText("q_cms")
    self.internal_flow_field_edit.setPlaceholderText("field name, e.g. q_cms")

    self.run_time_edit = _find_or_create_line_edit("run_time_edit", "Run duration (hr or HH:MM):")
    self.run_time_edit.setPlaceholderText("decimal hours (e.g. 1.5) or HH:MM (e.g. 01:30)")
    if not str(self.run_time_edit.text() or "").strip():
        self.run_time_edit.setText("1:00")

    self.reconstruction_combo = _find_or_create_combo("reconstruction_combo", "Reconstruction:")
    prev_data = self.reconstruction_combo.currentData()
    prev_text = self.reconstruction_combo.currentText()
    self.reconstruction_combo.blockSignals(True)
    try:
        self.reconstruction_combo.clear()
        for label, value in _RECONSTRUCTION_OPTIONS:
            self.reconstruction_combo.addItem(label, int(value))
        idx = self.reconstruction_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.reconstruction_combo.findText(prev_text)
        if idx < 0:
            idx = min(1, max(0, self.reconstruction_combo.count() - 1))
        if idx >= 0:
            self.reconstruction_combo.setCurrentIndex(idx)
    finally:
        self.reconstruction_combo.blockSignals(False)
    self.reconstruction_combo.setToolTip(
        "Select spatial reconstruction for the native solver.\n"
        "All 2nd-order schemes use Green-Gauss gradient-based TVD reconstruction:\n"
        "  Superbee (MUSCL Fast)  - most aggressive TVD, sharpest fronts\n"
        "  MinMod                 - most conservative, most stable near dry fronts\n"
        "  MC                     - balanced monotonized-central (good default)\n"
        "  Van Leer               - smooth limiter, good for continuous waves\n"
        "Recommend: start with MUSCL MinMod; switch to MC or Van Leer once stable."
    )

    self.temporal_order_combo = _find_or_create_combo("temporal_order_combo", "Temporal discretization:")
    prev_data = self.temporal_order_combo.currentData()
    prev_text = self.temporal_order_combo.currentText()
    self.temporal_order_combo.blockSignals(True)
    try:
        self.temporal_order_combo.clear()
        for label, value in _TEMPORAL_ORDER_OPTIONS:
            self.temporal_order_combo.addItem(label, int(value))
        idx = self.temporal_order_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.temporal_order_combo.findText(prev_text)
        if idx < 0:
            idx = min(1, max(0, self.temporal_order_combo.count() - 1))
        if idx >= 0:
            self.temporal_order_combo.setCurrentIndex(idx)
    finally:
        self.temporal_order_combo.blockSignals(False)
    self.temporal_order_combo.setToolTip(
        "Select temporal integration scheme:\n"
        "  Euler (RK1)  - 1st-order, fastest, use for dry-bed or debugging\n"
        "  RK2 (Heun)   - 2nd-order (default), balanced stability and speed\n"
        "  RK4 (classic) - 4th-order composed path\n"
        "  Graph-safe RK4 - true staged RK4 with CUDA-graph-safe forcing\n"
        "  Graph-safe RK5 - Cash-Karp staged RK5 with CUDA-graph-safe forcing\n"
        "Higher-order schemes are GPU-oriented and may be auto-adjusted by runtime guards."
    )

    self.equation_set_combo = _find_or_create_combo("equation_set_combo", "Equation set:")
    prev_data = self.equation_set_combo.currentData()
    prev_text = self.equation_set_combo.currentText()
    self.equation_set_combo.blockSignals(True)
    try:
        self.equation_set_combo.clear()
        if SWE2DEquationSet is not None:
            self.equation_set_combo.addItem("Hydrostatic 2D (default)", int(SWE2DEquationSet.HYDROSTATIC_2D))
            self.equation_set_combo.addItem("Nonhydrostatic 2D", int(SWE2DEquationSet.NONHYDROSTATIC_2D))
        else:
            self.equation_set_combo.addItem("Hydrostatic 2D (default)", 0)
            self.equation_set_combo.addItem("Nonhydrostatic 2D", 1)
        idx = self.equation_set_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.equation_set_combo.findText(prev_text)
        if idx < 0:
            idx = 0
        if idx >= 0:
            self.equation_set_combo.setCurrentIndex(idx)
    finally:
        self.equation_set_combo.blockSignals(False)
    self.equation_set_combo.setToolTip(
        "Choose the governing equation set for the 2D solver.\n"
        "Hydrostatic 2D keeps the existing shallow-water path.\n"
        "Nonhydrostatic 2D enables the pressure-correction solver and requires GPU."
    )



def _connect_project_workbench_state_signals(self) -> None:
    """Connect workbench widget signals to state persistence callback."""
    widget_specs = [
        ("nx_spin", "valueChanged"),
        ("ny_spin", "valueChanged"),
        ("lx_spin", "valueChanged"),
        ("ly_spin", "valueChanged"),
        ("bed_amp_spin", "valueChanged"),
        ("mesh_layout_combo", "currentIndexChanged"),
        ("h_min_spin", "valueChanged"),
        ("initial_condition_combo", "currentIndexChanged"),
        ("initial_depth_spin", "valueChanged"),
        ("initial_wse_spin", "valueChanged"),
        ("adaptive_cfl_dt_chk", "toggled"),
        ("dt_spin", "valueChanged"),
        ("gpu_diag_sync_interval_spin", "valueChanged"),
        ("enable_cuda_graphs_chk", "toggled"),
        ("max_rel_depth_increase_spin", "valueChanged"),
        ("max_source_depth_step_spin", "valueChanged"),
        ("max_source_rate_spin", "valueChanged"),
        ("extreme_rain_mode_chk", "toggled"),
        ("source_cfl_beta_spin", "valueChanged"),
        ("source_max_substeps_spin", "valueChanged"),
        ("source_true_subcycling_chk", "toggled"),
        ("source_imex_split_chk", "toggled"),
        ("source_stage_coupled_imex_rk2_chk", "toggled"),
        ("shallow_damping_depth_spin", "valueChanged"),
        ("shallow_front_recon_fallback_chk", "toggled"),
        ("front_flux_damping_spin", "valueChanged"),
        ("active_set_hysteresis_chk", "toggled"),
        ("depth_cap_spin", "valueChanged"),
        ("momentum_cap_min_speed_spin", "valueChanged"),
        ("momentum_cap_celerity_mult_spin", "valueChanged"),
        ("max_inv_area_spin", "valueChanged"),
        ("cfl_lambda_cap_spin", "valueChanged"),
        ("rain_rate_spin", "valueChanged"),
        ("cn_default_spin", "valueChanged"),
        ("ia_ratio_spin", "valueChanged"),
        ("use_spatial_rain_cn_chk", "toggled"),
        ("infiltration_method_combo", "currentIndexChanged"),
        ("rain_boundary_buffer_rings_spin", "valueChanged"),
        ("internal_flow_field_edit", "editingFinished"),
        ("run_time_edit", "editingFinished"),
        ("output_interval_edit", "editingFinished"),
        ("line_output_interval_edit", "editingFinished"),
        ("reconstruction_combo", "currentIndexChanged"),
        ("temporal_order_combo", "currentIndexChanged"),
        ("equation_set_combo", "currentIndexChanged"),
        ("experimental_3d_mode_chk", "toggled"),
        ("experimental_3d_coupling_mode_combo", "currentIndexChanged"),
        ("experimental_3d_patch_face_len_x_spin", "valueChanged"),
        ("experimental_3d_patch_face_len_y_spin", "valueChanged"),
        ("experimental_3d_patch_face_len_z_spin", "valueChanged"),
        ("experimental_3d_patch_xmin_edit", "editingFinished"),
        ("experimental_3d_patch_xmax_edit", "editingFinished"),
        ("experimental_3d_patch_ymin_edit", "editingFinished"),
        ("experimental_3d_patch_ymax_edit", "editingFinished"),
        ("experimental_3d_patch_zmin_edit", "editingFinished"),
        ("experimental_3d_patch_zmax_edit", "editingFinished"),
        ("experimental_3d_obj_solids_chk", "toggled"),
        ("experimental_3d_obj_method_combo", "currentIndexChanged"),
        ("experimental_3d_geom_sanitize_chk", "toggled"),
        ("experimental_3d_geom_phi_snap_spin", "valueChanged"),
        ("experimental_3d_geom_area_snap_spin", "valueChanged"),
        ("experimental_3d_obj_layer_combo", "currentIndexChanged"),
        ("experimental_3d_obj_path_field_edit", "editingFinished"),
        ("experimental_3d_obj_default_path_edit", "editingFinished"),
        ("experimental_3d_obj_scale_field_edit", "editingFinished"),
        ("experimental_3d_obj_yaw_field_edit", "editingFinished"),
        ("experimental_3d_obj_z_offset_field_edit", "editingFinished"),
        ("experimental_3d_obj_inside_points_layer_combo", "currentIndexChanged"),
        ("experimental_3d_obj_instance_id_field_edit", "editingFinished"),
        ("experimental_3d_obj_inside_id_field_edit", "editingFinished"),
        ("experimental_3d_obj_inside_z_field_edit", "editingFinished"),
        ("experimental_3d_obj_use_terrain_chk", "toggled"),
        ("experimental_3d_obj_ab_compare_chk", "toggled"),
        ("experimental_3d_obj_ab_probe_steps_spin", "valueChanged"),
        ("experimental_3d_obj_export_obj_chk", "toggled"),
        ("experimental_3d_obj_export_obj_path_edit", "editingFinished"),
        ("degen_mode_combo", "currentIndexChanged"),
        ("coupling_loop_combo", "currentIndexChanged"),
        ("drainage_solver_mode_combo", "currentIndexChanged"),
        ("drainage_backend_combo", "currentIndexChanged"),
        ("drainage_gpu_method_combo", "currentIndexChanged"),
        ("drainage_coupling_substeps_spin", "valueChanged"),
        ("drainage_max_coupling_substeps_spin", "valueChanged"),
        ("drainage_head_deadband_spin", "valueChanged"),
        ("drainage_dynamic_relaxation_spin", "valueChanged"),
        ("drainage_adaptive_depth_fraction_spin", "valueChanged"),
        ("drainage_adaptive_wave_courant_spin", "valueChanged"),
        ("extended_outputs_chk", "toggled"),
        ("save_mesh_results_to_gpkg_chk", "toggled"),
        ("save_line_results_to_gpkg_chk", "toggled"),
        ("save_coupling_results_to_gpkg_chk", "toggled"),
        ("save_run_log_to_gpkg_chk", "toggled"),
        ("topo_backend_combo", "currentIndexChanged"),
        ("topo_default_size_spin", "valueChanged"),
        ("topo_default_cell_type_combo", "currentIndexChanged"),
        ("topo_quality_min_angle_spin", "valueChanged"),
        ("topo_quality_max_aspect_spin", "valueChanged"),
        ("topo_quality_max_non_orth_spin", "valueChanged"),
        ("topo_quality_min_area_edit", "editingFinished"),
        ("topo_quality_size_scales_edit", "editingFinished"),
        ("topo_quality_smooth_increments_edit", "editingFinished"),
        ("topo_quality_strict_chk", "toggled"),
        ("topo_gmsh_quality_enable_chk", "toggled"),
        ("topo_gmsh_quality_max_iters_spin", "valueChanged"),
        ("topo_gmsh_quality_time_limit_spin", "valueChanged"),
        ("topo_gmsh_tri_algo_combo", "currentIndexChanged"),
        ("topo_gmsh_quad_algo_combo", "currentIndexChanged"),
        ("topo_gmsh_recombine_algo_combo", "currentIndexChanged"),
        ("topo_gmsh_smoothing_spin", "valueChanged"),
        ("topo_gmsh_optimize_iters_spin", "valueChanged"),
        ("topo_gmsh_optimize_netgen_chk", "toggled"),
        ("topo_gmsh_verbosity_spin", "valueChanged"),
    ]
    widget_specs.extend(list(getattr(self, "_experimental_3d_bc_signal_specs", []) or []))

    for attr_name, signal_name in widget_specs:
        widget = getattr(self, attr_name, None)
        if widget is None:
            continue
        try:
            signal = getattr(widget, signal_name, None)
            if signal is not None:
                signal.connect(self._persist_project_workbench_state)
        except Exception:
            pass



def _bind_model_tab_core_controls(self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout) -> None:
    def _ensure_row(label: str, widget: QtWidgets.QWidget) -> None:
        if param_form.indexOf(widget) >= 0:
            return
        param_form.addRow(label, widget)

    def _find_or_create_double_spin(name: str, label: str) -> QtWidgets.QDoubleSpinBox:
        w = model_tab_page.findChild(QtWidgets.QDoubleSpinBox, name)
        if w is None:
            w = QtWidgets.QDoubleSpinBox()
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    def _find_or_create_spin(name: str, label: str) -> QtWidgets.QSpinBox:
        w = model_tab_page.findChild(QtWidgets.QSpinBox, name)
        if w is None:
            w = QtWidgets.QSpinBox()
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    def _find_or_create_check(name: str, label: str, text: str) -> QtWidgets.QCheckBox:
        w = model_tab_page.findChild(QtWidgets.QCheckBox, name)
        if w is None:
            w = QtWidgets.QCheckBox(text)
            w.setObjectName(name)
        if not str(w.text() or "").strip():
            w.setText(text)
        _ensure_row(label, w)
        return w

    def _find_or_create_combo(name: str, label: str) -> QtWidgets.QComboBox:
        w = model_tab_page.findChild(QtWidgets.QComboBox, name)
        if w is None:
            w = QtWidgets.QComboBox()
            w.setObjectName(name)
        _ensure_row(label, w)
        return w

    self.n_mann_spin = _find_or_create_double_spin("n_mann_spin", "Manning n:")
    self.n_mann_spin.setRange(0.0, 1.0)
    self.n_mann_spin.setDecimals(5)
    self.n_mann_spin.setValue(0.020)

    self.cfl_spin = _find_or_create_double_spin("cfl_spin", "CFL:")
    self.cfl_spin.setRange(0.01, 0.99)
    self.cfl_spin.setDecimals(3)
    self.cfl_spin.setValue(0.45)

    self.h_min_spin = _find_or_create_double_spin("h_min_spin", "h_min:")
    self.h_min_spin.setRange(1.0e-9, 1.0)
    self.h_min_spin.setDecimals(8)
    self.h_min_spin.setValue(1.0e-6)

    self.initial_condition_combo = _find_or_create_combo("initial_condition_combo", "Initial condition:")
    prev_data = self.initial_condition_combo.currentData()
    prev_text = self.initial_condition_combo.currentText()
    self.initial_condition_combo.blockSignals(True)
    try:
        self.initial_condition_combo.clear()
        self.initial_condition_combo.addItem("Dry start", "dry")
        self.initial_condition_combo.addItem("Uniform depth", "uniform_depth")
        self.initial_condition_combo.addItem("Uniform water surface elevation", "uniform_wse")
        idx = self.initial_condition_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.initial_condition_combo.findText(prev_text)
        if idx < 0:
            idx = self.initial_condition_combo.findData("dry")
        if idx >= 0:
            self.initial_condition_combo.setCurrentIndex(idx)
    finally:
        self.initial_condition_combo.blockSignals(False)
    self.initial_condition_combo.setToolTip(
        "Initial condition source used at run start.\n"
        "Dry start: h=0.\n"
        "Uniform depth: constant initial depth everywhere.\n"
        "Uniform WSE: depth = max(0, WSE - local bed)."
    )

    self.initial_depth_spin = _find_or_create_double_spin("initial_depth_spin", "Initial depth:")
    self.initial_depth_spin.setRange(0.0, 1.0e6)
    self.initial_depth_spin.setDecimals(4)
    self.initial_depth_spin.setValue(0.0)

    self.initial_wse_spin = _find_or_create_double_spin("initial_wse_spin", "Initial WSE:")
    self.initial_wse_spin.setRange(-1.0e6, 1.0e6)
    self.initial_wse_spin.setDecimals(4)
    self.initial_wse_spin.setValue(0.0)

    self.adaptive_cfl_dt_chk = _find_or_create_check(
        "adaptive_cfl_dt_chk", "Variable timestep:", "Enable variable timestep (CFL)"
    )
    self.adaptive_cfl_dt_chk.setChecked(False)
    self.adaptive_cfl_dt_chk.setToolTip(
        "If enabled, runtime dt is selected from CFL each step.\n"
        "The dt field is used as dt_max (upper bound).\n"
        "If disabled, dt is fixed each step."
    )

    self.dt_spin = _find_or_create_double_spin("dt_spin", "dt (fixed or dt_max):")
    self.dt_spin.setRange(1.0e-4, 1.0e6)
    self.dt_spin.setDecimals(5)
    self.dt_spin.setValue(0.05)

    self.gpu_diag_sync_interval_spin = _find_or_create_spin(
        "gpu_diag_sync_interval_spin", "GPU diag sync (steps):"
    )
    self.gpu_diag_sync_interval_spin.setRange(1, 1000000)
    self.gpu_diag_sync_interval_spin.setValue(10)
    self.gpu_diag_sync_interval_spin.setToolTip(
        "GPU host diagnostic sync cadence in computational steps.\n"
        "1 = sync every step (freshest Cmax/WSEres runtime output).\n"
        "Higher values reduce host sync overhead but update diagnostics less often."
    )

    self.enable_cuda_graphs_chk = _find_or_create_check(
        "enable_cuda_graphs_chk", "CUDA graph replay:", "Enable"
    )
    self.enable_cuda_graphs_chk.setChecked(False)
    self.enable_cuda_graphs_chk.setToolTip(
        "Enable CUDA graph capture/replay for the core GPU step kernel chain.\n"
        "Can reduce launch overhead and improve throughput on compatible runs."
    )



def _preview_coupling_configuration(self):
    if self._mesh_data is None:
        QtWidgets.QMessageBox.information(
            self,
            "Coupling Preview",
            "Generate or load a mesh first so cell-based coupling indices can be resolved.",
        )
        return
    pipe_cfg = self._build_pipe_network_config()
    struct_cfg = self._build_hydraulic_structure_config()
    if pipe_cfg is None and struct_cfg is None:
        QtWidgets.QMessageBox.information(
            self,
            "Coupling Preview",
            "No valid drainage or structure layers are configured.",
        )
        return

    lines: List[str] = []

    def _format_id_preview(ids: Sequence[str], limit: int = 10) -> str:
        vals = [str(v) for v in ids if str(v)]
        if not vals:
            return "(none)"
        if len(vals) <= limit:
            return ", ".join(vals)
        return ", ".join(vals[:limit]) + f", ... (+{len(vals) - limit} more)"

    if pipe_cfg is not None:
        lines.append(
            f"Drainage network: nodes={len(pipe_cfg.nodes)}, links={len(pipe_cfg.links)}, inlets={len(pipe_cfg.inlets)}"
        )

        node_by_id = {str(n.node_id): n for n in pipe_cfg.nodes}
        unknown_link_refs: List[str] = []
        unknown_inlet_refs: List[str] = []
        zero_capacity_links: List[str] = []
        near_zero_head_links: List[str] = []
        t0_probably_zero_links: List[str] = []

        for lk in pipe_cfg.links:
            lid = str(lk.link_id)
            n0 = node_by_id.get(str(lk.from_node_id))
            n1 = node_by_id.get(str(lk.to_node_id))
            if n0 is None or n1 is None:
                unknown_link_refs.append(lid)
                continue

            d = float(lk.diameter) if lk.diameter is not None else 0.0
            a = float(lk.metadata.get("area_m2", 0.0) or 0.0)
            eqd = float(lk.metadata.get("equiv_diameter_m", 0.0) or 0.0)
            has_capacity = (d > 0.0) or (a > 0.0) or (eqd > 0.0)
            if not has_capacity:
                zero_capacity_links.append(lid)

            dh0 = float(n0.invert_elev) - float(n1.invert_elev)
            near_zero_head = abs(dh0) <= 1.0e-4
            if near_zero_head:
                near_zero_head_links.append(lid)

            if (not has_capacity) or near_zero_head:
                t0_probably_zero_links.append(lid)

        for inlet in pipe_cfg.inlets:
            if str(inlet.node_id) not in node_by_id:
                unknown_inlet_refs.append(str(inlet.inlet_id))

        lines.append("Coupling sanity report (drainage):")
        lines.append(f"- unknown link node refs: {len(unknown_link_refs)}")
        if unknown_link_refs:
            lines.append(f"  IDs: {_format_id_preview(unknown_link_refs)}")
        lines.append(f"- unknown inlet node refs: {len(unknown_inlet_refs)}")
        if unknown_inlet_refs:
            lines.append(f"  IDs: {_format_id_preview(unknown_inlet_refs)}")
        lines.append(f"- links with zero hydraulic capacity fields: {len(zero_capacity_links)}")
        if zero_capacity_links:
            lines.append(f"  IDs: {_format_id_preview(zero_capacity_links)}")
        lines.append(f"- links with near-zero initial head gradient (|dh0|<=1e-4): {len(near_zero_head_links)}")
        if near_zero_head_links:
            lines.append(f"  IDs: {_format_id_preview(near_zero_head_links)}")
        lines.append(f"- links likely zero-flow at t0 (capacity/head limits): {len(t0_probably_zero_links)}")
    else:
        lines.append("Drainage network: not configured")

    if struct_cfg is not None:
        lines.append(f"Hydraulic structures: count={len(struct_cfg.structures)}")
    else:
        lines.append("Hydraulic structures: not configured")

    if pack_coupling_soa is not None:
        soa = pack_coupling_soa(
            n_cells=int(self._mesh_cell_areas().shape[0]),
            pipe_network=pipe_cfg,
            hydraulic_structures=struct_cfg,
        )
        if soa.drainage is not None:
            dn = soa.drainage
            invalid_links = int(np.sum((dn.link_from < 0) | (dn.link_to < 0)))
            invalid_inlets = int(np.sum((dn.inlet_cell < 0) | (dn.inlet_node < 0)))
            lines.append(
                "Drainage SoA: "
                f"nodes={dn.node_x.size}, links={dn.link_from.size}, inlets={dn.inlet_cell.size}, "
                f"invalid_links={invalid_links}, invalid_inlets={invalid_inlets}"
            )
        if soa.structures is not None:
            ss = soa.structures
            invalid_struct = int(np.sum((ss.upstream_cell < 0) | (ss.downstream_cell < 0)))
            lines.append(
                "Structures SoA: "
                f"count={ss.structure_type.size}, invalid_cell_pairs={invalid_struct}"
            )

    QtWidgets.QMessageBox.information(self, "Coupling Preview", "\n".join(lines))



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

        # For gmsh (process executor), terminate and recreate the pool to
        # ensure stuck native meshing work is not left running.
        if backend_name == "gmsh" and self._topology_mesh_process_pool is not None:
            try:
                self._topology_mesh_process_pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._topology_mesh_process_pool = None

        self.topo_status_lbl.setText(
            f"Topology meshing timed out after {self._topology_mesh_active_timeout_sec:.0f}s "
            f"(backend '{backend_name}')."
        )
        self._log(
            "mesh> timeout "
            f"backend={backend_name} mode={run_mode} elapsed={elapsed:.2f}s "
            f"limit={self._topology_mesh_active_timeout_sec:.0f}s"
        )

        self._set_topology_mesh_busy(False)
        return

    if not fut.done():
        self._topology_mesh_poll_count += 1
        # Emit lightweight runtime heartbeat at ~1 second cadence.
        if self._topology_mesh_poll_count % 8 == 0:
            spinner = "|/-\\"[(self._topology_mesh_poll_count // 8) % 4]
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
            except Exception:
                pass
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
        except Exception:
            continue
        z = 0.0
        if "bed_z" in nodes_layer.fields().names():
            try:
                z = float(ft["bed_z"])
            except Exception:
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

    cell_list: List[int] = []
    for ft in cells_layer.getFeatures():
        n0 = ft["n0"] if "n0" in cells_layer.fields().names() else None
        n1 = ft["n1"] if "n1" in cells_layer.fields().names() else None
        n2 = ft["n2"] if "n2" in cells_layer.fields().names() else None
        if n0 is not None and n1 is not None and n2 is not None:
            try:
                tri_ids = [int(n0), int(n1), int(n2)]
                tri_idx = [id_to_idx[t] for t in tri_ids]
                cell_list.extend(tri_idx)
                continue
            except Exception:
                pass

        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        poly = geom.asPolygon()
        if not poly or not poly[0]:
            continue
        ring = poly[0]
        verts = []
        for p in ring[:-1]:
            key = (round(float(p.x()), 9), round(float(p.y()), 9))
            if key in coord_to_idx:
                verts.append(coord_to_idx[key])
        uniq = []
        for vid in verts:
            if vid not in uniq:
                uniq.append(vid)
        if len(uniq) >= 3:
            cell_list.extend(uniq[:3])

    if len(cell_list) < 3:
        self._log("No valid triangle cells found in selected cells layer.")
        return

    cell_nodes = np.array(cell_list, dtype=np.int32)
    if cell_nodes.size % 3 != 0:
        cell_nodes = cell_nodes[: (cell_nodes.size // 3) * 3]

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
    }
    n_cells = int(cell_nodes.size // 3)
    self.mesh_info_lbl.setText(f"Loaded map mesh: nodes={node_x.size}, cells={n_cells}, triangles={n_cells}")
    self.layer_status_lbl.setText("Mesh loaded from selected map layers.")
    self._log(f"Imported mesh from map layers: nodes={node_x.size}, cells={n_cells}")
    self._result_data = None
    self.view_mode_combo.setCurrentText("Mesh")
    self._refresh_plot()



def _build_line_sampling_map(self) -> List[Dict[str, object]]:
    if self._mesh_data is None or not _HAVE_QGIS_CORE:
        return []
    if not hasattr(self, "sample_lines_layer_combo"):
        return []
    line_layer = self._combo_layer(self.sample_lines_layer_combo, "vector")
    if line_layer is None:
        return []

    fields = set(line_layer.fields().names())
    id_field = "line_id" if "line_id" in fields else None
    name_field = "name" if "name" in fields else None
    enabled_field = "enabled" if "enabled" in fields else None

    cell_polys = self._mesh_cell_polygons()
    if not cell_polys:
        return []
    cell_bboxes = [g.boundingBox() if g is not None and not g.isEmpty() else None for g in cell_polys]

    sample_map: List[Dict[str, object]] = []
    for ft in line_layer.getFeatures():
        geom = ft.geometry()
        if geom is None or geom.isEmpty():
            continue
        try:
            if enabled_field is not None and int(ft[enabled_field]) <= 0:
                continue
        except Exception:
            pass

        line_len = float(geom.length())
        if line_len <= 0.0:
            continue
        try:
            p0 = geom.interpolate(0.0).asPoint()
            p1 = geom.interpolate(max(0.0, line_len - 1.0e-9)).asPoint()
            dx = float(p1.x()) - float(p0.x())
            dy = float(p1.y()) - float(p0.y())
            mag = math.hypot(dx, dy)
            if mag <= 0.0:
                continue
            tx = dx / mag
            ty = dy / mag
            nx = ty
            ny = -tx
        except Exception:
            continue

        try:
            line_id = int(ft[id_field]) if id_field is not None else int(ft.id())
        except Exception:
            line_id = int(ft.id())
        line_name = str(ft[name_field]) if name_field is not None and ft[name_field] not in (None, "") else ""

        line_bbox = geom.boundingBox()
        idx: List[int] = []
        lens: List[float] = []
        station_m: List[float] = []
        for ci, cell_geom in enumerate(cell_polys):
            bb = cell_bboxes[ci]
            if bb is None or not bb.intersects(line_bbox):
                continue
            try:
                inter = cell_geom.intersection(geom)
            except Exception:
                continue
            if inter is None or inter.isEmpty():
                continue
            seg_len = float(inter.length())
            if seg_len <= 0.0:
                continue
            s_loc = float("nan")
            try:
                cgeom = inter.centroid()
                if cgeom is not None and not cgeom.isEmpty():
                    s_loc = float(geom.lineLocatePoint(cgeom))
            except Exception:
                s_loc = float("nan")
            idx.append(ci)
            lens.append(seg_len)
            station_m.append(s_loc)

        if idx:
            ord_idx = np.argsort(np.nan_to_num(np.asarray(station_m, dtype=np.float64), nan=0.0))
            sample_map.append(
                {
                    "line_id": int(line_id),
                    "line_name": line_name,
                    "normal_x": float(nx),
                    "normal_y": float(ny),
                    "cell_idx": np.asarray(idx, dtype=np.int32)[ord_idx],
                    "weights": np.asarray(lens, dtype=np.float64)[ord_idx],
                    "station_m": np.asarray(station_m, dtype=np.float64)[ord_idx],
                }
            )

    if sample_map:
        self._log(f"Sample line mapping ready: {len(sample_map)} line(s).")
    return sample_map



