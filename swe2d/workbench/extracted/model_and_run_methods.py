from __future__ import annotations

# Extracted methods depend on symbols defined in swe2d_workbench_qt.
from swe2d_workbench_qt import *  # type: ignore F401,F403
from swe2d_workbench_qt import (
    _BC_INFLOW_Q,
    _BC_TS_FLOW,
    _BC_TS_STAGE,
    _execute_run_timestep_loop_runtime_logic,
    _RECONSTRUCTION_OPTIONS,
    _TEMPORAL_ORDER_OPTIONS,
)

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

    self.initial_dt_spin = _find_or_create_double_spin("initial_dt_spin", "Initial dt (0 = auto):")
    self.initial_dt_spin.setRange(0.0, 1.0e6)
    self.initial_dt_spin.setDecimals(5)
    self.initial_dt_spin.setValue(0.0)
    self.initial_dt_spin.setToolTip(
        "If > 0, use this dt for the first timestep only.\n"
        "Useful for cold-start stability with CFL adaptive stepping\n"
        "on dry domains where lambda_max=0 causes dt to jump to dt_max.\n"
        "0 = use CFL dt from step 1 (default)."
    )

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

    self.tiny_mode_combo = _find_or_create_combo("tiny_mode_combo", "Tiny mode:")
    if self.tiny_mode_combo.count() == 0:
        self.tiny_mode_combo.addItem("Off (0)", 0)
        self.tiny_mode_combo.addItem("Auto (1)", 1)
        self.tiny_mode_combo.addItem("Fused (2)", 2)
        self.tiny_mode_combo.addItem("Persistent (3)", 3)
    tiny_mode_idx = self.tiny_mode_combo.findData(3)
    if tiny_mode_idx >= 0:
        self.tiny_mode_combo.setCurrentIndex(tiny_mode_idx)
    self.tiny_mode_combo.setToolTip(
        "Tiny-N dispatch mode for low-cell-count/low-wet runs.\n"
        "0=off, 1=auto, 2=fused, 3=persistent.\n"
        "Persistent (3) is recommended for minimizing kernel-launch overhead in GUI runs."
    )

    self.tiny_wet_cell_threshold_spin = _find_or_create_spin(
        "tiny_wet_cell_threshold_spin", "Tiny active/wet cell threshold:"
    )
    self.tiny_wet_cell_threshold_spin.setRange(1, 10000000)
    self.tiny_wet_cell_threshold_spin.setValue(2000)
    self.tiny_wet_cell_threshold_spin.setToolTip(
        "Maximum active/wet cell count used by tiny-mode dispatch gating.\n"
        "Lower values limit tiny-mode activation to smaller wetted domains."
    )

    self.enable_cuda_graphs_chk = _find_or_create_check(
        "enable_cuda_graphs_chk", "CUDA graph replay:", "Enable"
    )
    self.enable_cuda_graphs_chk.setChecked(False)
    self.enable_cuda_graphs_chk.setToolTip(
        "Enable CUDA graph capture/replay for the core GPU step kernel chain.\n"
        "Can reduce launch overhead and improve throughput on compatible runs."
    )

    self.swe2d_perf_mode_chk = _find_or_create_check(
        "swe2d_perf_mode_chk", "SWE2D perf mode:", "Enable"
    )
    self.swe2d_perf_mode_chk.setChecked(False)
    self.swe2d_perf_mode_chk.setToolTip(
        "Toggle BACKWATER_SWE2D_PERF_MODE for this run.\n"
        "When enabled, reduces runtime logging cadence and disables per-step\n"
        "source/boundary forensic accounting to minimize host overhead."
    )





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
        "  Graph-safe RK4 - true staged RK4 with CUDA-graph-safe forcing\n"
        "  Graph-safe RK5 - Cash-Karp staged RK5 with CUDA-graph-safe forcing\n"
        "Higher-order schemes are GPU-oriented and may be auto-adjusted by runtime guards."
    )








def _bind_model_tab_subgrid_drainage_controls(
    self, model_tab_page: QtWidgets.QWidget, param_form: QtWidgets.QFormLayout,
    solver_form: Optional[QtWidgets.QFormLayout] = None,
) -> None:
    _sf = solver_form or param_form  # solver-target items go here, not into the drainage form

    def _ensure_row(label: str, widget: QtWidgets.QWidget, target_form: Optional[QtWidgets.QFormLayout] = None) -> None:
        form = target_form or param_form
        if form.indexOf(widget) >= 0:
            return
        form.addRow(label, widget)

    def _ensure_widget_row(widget: QtWidgets.QWidget, target_form: Optional[QtWidgets.QFormLayout] = None) -> None:
        form = target_form or param_form
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



    self.degen_mode_combo = _find_or_create_combo("degen_mode_combo", "Degenerate cell mode:", _sf)
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

    # GPU-only: solver_backend_combo is removed; always uses CUDA.

    self.coupling_loop_combo = _find_or_create_combo("coupling_loop_combo", "Coupling loop:", _sf)
    prev_data = self.coupling_loop_combo.currentData()
    prev_text = self.coupling_loop_combo.currentText()
    self.coupling_loop_combo.blockSignals(True)
    try:
        self.coupling_loop_combo.clear()
        self.coupling_loop_combo.addItem("CUDA coupling loop (GPU)", "cuda")
        idx = 0
        if idx >= 0:
            self.coupling_loop_combo.setCurrentIndex(idx)
    finally:
        self.coupling_loop_combo.blockSignals(False)
    self.coupling_loop_combo.setToolTip(
        "GPU coupling loop using native CUDA kernel for per-cell source assembly."
    )

    self.culvert_solver_mode_combo = _find_or_create_combo(
        "culvert_solver_mode_combo", "Culvert solver mode:", param_form
    )
    prev_data = self.culvert_solver_mode_combo.currentData()
    prev_text = self.culvert_solver_mode_combo.currentText()
    self.culvert_solver_mode_combo.blockSignals(True)
    try:
        self.culvert_solver_mode_combo.clear()
        self.culvert_solver_mode_combo.addItem("Direct culvert outlet solver (Newton/secant)", int(0))
        self.culvert_solver_mode_combo.addItem("Precomputed culvert lookup table", int(1))
        idx = self.culvert_solver_mode_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.culvert_solver_mode_combo.findText(prev_text)
        if idx < 0:
            idx = 0
        if idx >= 0:
            self.culvert_solver_mode_combo.setCurrentIndex(idx)
    finally:
        self.culvert_solver_mode_combo.blockSignals(False)
    self.culvert_solver_mode_combo.setToolTip(
        "Select the native GPU culvert solver mode.\n"
        "Direct: uses the outlet control solver per structure.\n"
        "Lookup table: uses precomputed Q(hw,tw) tables when available.\n"
        "Requires the native CUDA backend for GPU structure flow evaluation."
    )

    self.culvert_face_flux_chk = _find_or_create_check(
        "culvert_face_flux_chk", "Culvert coupling mode:", "Face-based flux (GPU only)",
        target_form=param_form,
    )
    self.culvert_face_flux_chk.setToolTip(
        "When enabled, culvert flows are applied as face-based fluxes\n"
        "instead of cell-center source/sink terms on the GPU path.\n"
        "This preserves strict mass conservation and transfers momentum\n"
        "proportional to discharge. Only applies to culvert structures.\n"
        "Requires CUDA GPU backend and coupling loop set to CUDA."
    )

    self.bridge_stacked_coupling_mode_combo = _find_or_create_combo(
        "bridge_stacked_coupling_mode_combo", "Bridge stacked coupling mode:", param_form
    )
    prev_data = self.bridge_stacked_coupling_mode_combo.currentData()
    prev_text = self.bridge_stacked_coupling_mode_combo.currentText()
    self.bridge_stacked_coupling_mode_combo.blockSignals(True)
    try:
        self.bridge_stacked_coupling_mode_combo.clear()
        self.bridge_stacked_coupling_mode_combo.addItem(
            "Phase 3 spatial redistribution (recommended)", "phase3_spatial"
        )
        self.bridge_stacked_coupling_mode_combo.addItem(
            "Legacy scalar weighting (backward-compatible)", "legacy_scalar"
        )
        idx = self.bridge_stacked_coupling_mode_combo.findData(prev_data)
        if idx < 0 and prev_text:
            idx = self.bridge_stacked_coupling_mode_combo.findText(prev_text)
        if idx < 0:
            idx = 0
        if idx >= 0:
            self.bridge_stacked_coupling_mode_combo.setCurrentIndex(idx)
    finally:
        self.bridge_stacked_coupling_mode_combo.blockSignals(False)
    self.bridge_stacked_coupling_mode_combo.setToolTip(
        "Select how stacked bridge geometry modifies bridge coupling sources.\n"
        "Phase 3 spatial redistribution: attenuates and redistributes bridge source/sink\n"
        "across stacked corridor cells while preserving total source conservation.\n"
        "Legacy scalar weighting: multiplies helper bridge sources by one plan-scale factor."
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
        self.drainage_backend_combo.addItem("GPU drainage solver (CUDA)", "gpu")
        idx = 0
        if idx >= 0:
            self.drainage_backend_combo.setCurrentIndex(idx)
    finally:
        self.drainage_backend_combo.blockSignals(False)
    self.drainage_backend_combo.setToolTip(
        "GPU drainage solver using native CUDA kernels for EGL/Diffusion/Dynamic modes."
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

    results_row = run_tab_page.findChild(QtWidgets.QHBoxLayout, "run_results_gpkg_row_layout")
    if results_row is None:
        results_row = QtWidgets.QHBoxLayout()
        results_row.setObjectName("run_results_gpkg_row_layout")
        run_layout.addLayout(results_row)

    results_gpkg_lbl = run_tab_page.findChild(QtWidgets.QLabel, "results_gpkg_lbl")
    if results_gpkg_lbl is None:
        results_gpkg_lbl = QtWidgets.QLabel("Results GeoPackage:")
        results_gpkg_lbl.setObjectName("results_gpkg_lbl")
    if results_row.indexOf(results_gpkg_lbl) < 0:
        results_row.addWidget(results_gpkg_lbl)

    self.results_gpkg_path_edit = run_tab_page.findChild(QtWidgets.QLineEdit, "results_gpkg_path_edit")
    if self.results_gpkg_path_edit is None:
        self.results_gpkg_path_edit = QtWidgets.QLineEdit("")
        self.results_gpkg_path_edit.setObjectName("results_gpkg_path_edit")
    self.results_gpkg_path_edit.setPlaceholderText(
        "Optional override (blank = model/sample-line/default fallback)"
    )
    if results_row.indexOf(self.results_gpkg_path_edit) < 0:
        results_row.addWidget(self.results_gpkg_path_edit, stretch=1)

    self.results_gpkg_browse_btn = run_tab_page.findChild(QtWidgets.QPushButton, "results_gpkg_browse_btn")
    if self.results_gpkg_browse_btn is None:
        self.results_gpkg_browse_btn = QtWidgets.QPushButton("Browse...")
        self.results_gpkg_browse_btn.setObjectName("results_gpkg_browse_btn")
    if results_row.indexOf(self.results_gpkg_browse_btn) < 0:
        results_row.addWidget(self.results_gpkg_browse_btn)

    self.load_run_settings_btn = run_tab_page.findChild(QtWidgets.QPushButton, "load_run_settings_btn")
    if self.load_run_settings_btn is None:
        self.load_run_settings_btn = QtWidgets.QPushButton("Load Inputs From Results...")
        self.load_run_settings_btn.setObjectName("load_run_settings_btn")
    if results_row.indexOf(self.load_run_settings_btn) < 0:
        results_row.addWidget(self.load_run_settings_btn)

    table_row = run_tab_page.findChild(QtWidgets.QHBoxLayout, "run_results_table_row_layout")
    if table_row is None:
        table_row = QtWidgets.QHBoxLayout()
        table_row.setObjectName("run_results_table_row_layout")
        run_layout.addLayout(table_row)

    results_table_lbl = run_tab_page.findChild(QtWidgets.QLabel, "results_table_lbl")
    if results_table_lbl is None:
        results_table_lbl = QtWidgets.QLabel("Results table prefix:")
        results_table_lbl.setObjectName("results_table_lbl")
    else:
        results_table_lbl.setText("Results table prefix:")
    if table_row.indexOf(results_table_lbl) < 0:
        table_row.addWidget(results_table_lbl)

    self.results_table_name_edit = run_tab_page.findChild(QtWidgets.QLineEdit, "results_table_name_edit")
    if self.results_table_name_edit is None:
        self.results_table_name_edit = QtWidgets.QLineEdit("")
        self.results_table_name_edit.setObjectName("results_table_name_edit")
    self.results_table_name_edit.setPlaceholderText("Optional prefix, e.g. scenario_a")
    if table_row.indexOf(self.results_table_name_edit) < 0:
        table_row.addWidget(self.results_table_name_edit, stretch=1)

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
    self.results_gpkg_path_edit.setToolTip(
        "Optional output GeoPackage path for all run persistence tables\n"
        "(run logs, mesh snapshots, line/coupling outputs, conservation forensics).\n"
        "Leave blank to use the model GeoPackage fallback chain."
    )
    self.results_gpkg_browse_btn.setToolTip(
        "Choose/create a GeoPackage target used for all persisted run outputs."
    )
    self.load_run_settings_btn.setToolTip(
        "Open run logs from the selected results GeoPackage and apply saved\n"
        "input widget settings from a prior run to the current UI."
    )
    self.results_table_name_edit.setToolTip(
        "Optional prefix prepended to all persisted SWE2D results tables for this run.\n"
        "Example: prefix 'scenario_a' writes tables such as\n"
        "scenario_a_swe2d_mesh_results and scenario_a_swe2d_line_results_ts.\n"
        "Leave blank to use default SWE2D table names."
    )

    for btn, cb in (
        (self.run_btn, self._on_run_requested),
        (self.preview_overrides_btn, self._on_preview_overrides),
        (self.cancel_btn, self._on_cancel),
        (self.snapshot_btn, self._on_snapshot),
        (self.results_gpkg_browse_btn, self._on_select_results_gpkg),
        (self.load_run_settings_btn, self._on_load_run_settings_from_results),
    ):
        try:
            btn.clicked.disconnect(cb)
        except Exception:
            pass
        btn.clicked.connect(cb)





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
        ("initial_dt_spin", "valueChanged"),
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
        ("topo_gmsh_interface_conformance_chk", "toggled"),
        ("topo_gmsh_transverse_interface_centroid_merge_chk", "toggled"),
        ("topo_gmsh_interface_snap_tol_spin", "valueChanged"),
        ("topo_gmsh_interface_reject_near_unshared_chk", "toggled"),
        ("topo_gmsh_interface_reject_tol_spin", "valueChanged"),
    ]

    connected_attrs = set()

    for attr_name, signal_name in widget_specs:
        widget = getattr(self, attr_name, None)
        if widget is None:
            continue
        try:
            signal = getattr(widget, signal_name, None)
            if signal is not None:
                signal.connect(self._persist_project_workbench_state)
                connected_attrs.add(attr_name)
        except Exception:
            pass

    # Fallback auto-wiring: connect all known persistable widget types that may
    # have been added outside the static spec list.
    qspin_cls = getattr(QtWidgets, "QSpinBox")
    qdspin_cls = getattr(QtWidgets, "QDoubleSpinBox")
    qcombo_cls = getattr(QtWidgets, "QComboBox")
    qcheck_cls = getattr(QtWidgets, "QCheckBox")
    qline_cls = getattr(QtWidgets, "QLineEdit")

    auto_specs = []
    for attr_name, widget in vars(self).items():
        if attr_name in connected_attrs:
            continue
        if isinstance(widget, qspin_cls) or isinstance(widget, qdspin_cls):
            auto_specs.append((attr_name, "valueChanged"))
        elif isinstance(widget, qcombo_cls):
            auto_specs.append((attr_name, "currentIndexChanged"))
        elif isinstance(widget, qcheck_cls):
            auto_specs.append((attr_name, "toggled"))
        elif isinstance(widget, qline_cls):
            auto_specs.append((attr_name, "textChanged"))

    for attr_name, signal_name in auto_specs:
        widget = getattr(self, attr_name, None)
        if widget is None:
            continue
        try:
            signal = getattr(widget, signal_name, None)
            if signal is not None:
                signal.connect(self._persist_project_workbench_state)
        except Exception:
            pass





def _on_run(self, request=None):
    if request is None:
        request = getattr(self, "_last_run_request", None)
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
        if request is not None:
            request_run_duration_text = getattr(request, "run_duration_text", None)
            if request_run_duration_text is not None and str(request_run_duration_text).strip():
                try:
                    run_duration_s = max(0.0, self._parse_time_hours(str(request_run_duration_text).strip()) * 3600.0)
                except Exception:
                    pass
        dt_cfg = run_options.dt_cfg
        adaptive_cfl_dt = run_options.adaptive_cfl_dt
        dt_fixed = run_options.dt_fixed
        dt_request = run_options.dt_request
        initial_dt = getattr(run_options, "initial_dt", 0.0)
        reconstruction_mode = run_options.reconstruction_mode
        reconstruction_name = run_options.reconstruction_name
        temporal_scheme = run_options.temporal_scheme
        temporal_scheme_name = run_options.temporal_scheme_name
        solver_backend_mode = str(getattr(run_options, "solver_backend_mode", "gpu")).strip().lower()
        coupling_loop_mode = run_options.coupling_loop_mode
        drainage_solver_backend_mode = run_options.drainage_solver_backend_mode
        drainage_gpu_method_mode = run_options.drainage_gpu_method_mode
        culvert_solver_mode = getattr(run_options, "culvert_solver_mode", 0)
        cuda_graphs_enabled = run_options.cuda_graphs_enabled
        model_options = run_options.model_options
        rain_rate_model = run_options.rain_rate_model
        internal_flow_forcing = run_options.internal_flow_forcing
        cell_source_model = run_options.cell_source_model
        thiessen_forcing = run_options.thiessen_forcing
        pipe_network_cfg = run_options.pipe_network_cfg
        hydraulic_structures_cfg = run_options.hydraulic_structures_cfg
        bridge_stacked_plans = []

        try:
            from swe2d.runtime.bridge_stacked_runtime import build_bridge_stacked_plans_for_runtime
            from swe2d import units as _u

            bridge_stacked_plans = build_bridge_stacked_plans_for_runtime(
                self._mesh_data,
                hydraulic_structures_cfg,
                log_fn=self._log,
            )
        except Exception as exc:
            self._log(f"Bridge stacked-plan mapping warning: {exc}")

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
            # Compute model-to-feet factor: for a foot model (length_scale=3.28)
            # this gives 1.0 (no conversion); for SI (length_scale=1.0) this gives 3.28.
            # _ls = model units per SI meter (e.g. 3.28 for feet, 1.0 for meters)
            # Convert to si_m_per_model (e.g. 0.3048 for feet, 1.0 for meters)
            # which the coupling controller and units.configure() expect.
            _ls = max(1.0e-6, float(self._length_scale_si_to_model()))
            _si_m_per_model = 1.0 / _ls
            _model_to_ft = _u.USC_FT_PER_SI_M * _si_m_per_model
            structures_mod = SWE2DStructureModule(hydraulic_structures_cfg, model_to_ft=_model_to_ft) if hydraulic_structures_cfg is not None and SWE2DStructureModule is not None else None
            coupling_controller = SWE2DCouplingController(
                cell_area=self._mesh_cell_areas(),
                cell_bed=self._mesh_cell_min_bed(),
                drainage=drainage_mod,
                structures=structures_mod,
                drainage_solver_backend=drainage_solver_backend_mode,
                drainage_gpu_method=drainage_gpu_method_mode,
                culvert_solver_mode=culvert_solver_mode,
                bridge_cuda_coupling=bool(run_options.bridge_cuda_coupling),
                bridge_stacked_coupling_mode=str(getattr(run_options, "bridge_stacked_coupling_mode", "phase3_spatial")),
                length_scale_si_to_model=_si_m_per_model,
                culvert_face_flux_mode=str(getattr(run_options, "culvert_face_flux_mode", "off")),
                log_callback=self._log,
            )
            setattr(coupling_controller, "bridge_stacked_plans", bridge_stacked_plans)
            # Provide cell centroids for influence-width redistribution.
            try:
                cx, cy = self._mesh_cell_centroids()
                if hasattr(coupling_controller, "set_cell_centroids"):
                    coupling_controller.set_cell_centroids(cx, cy)
                if hasattr(coupling_controller, "_build_redistribution_data"):
                    coupling_controller._build_redistribution_data()
            except Exception:
                pass
            # GPU-only runtime: all coupling/drainage paths use CUDA.

        rain_stats_acc = {"rain_mm": 0.0, "excess_mm": 0.0, "samples": 0}

        if request is not None:
            self._last_run_request = request

        def _parse_interval_text(text, default_widget_text):
            if text is not None and str(text).strip():
                return self._parse_time_hours(str(text).strip())
            return self._parse_time_hours(str(default_widget_text or ""))

        request_output_interval_text = getattr(request, "output_interval_text", None) if request is not None else None
        request_line_output_interval_text = getattr(request, "line_output_interval_text", None) if request is not None else None

        _oi_hr = _parse_interval_text(
            request_output_interval_text,
            self.output_interval_edit.text() if hasattr(self, "output_interval_edit") else "",
        )
        output_interval_s = max(1.0, _oi_hr * 3600.0)
        _line_oi_hr = _parse_interval_text(
            request_line_output_interval_text,
            self.line_output_interval_edit.text() if hasattr(self, "line_output_interval_edit") else "",
        )
        line_output_interval_s = max(1.0, _line_oi_hr * 3600.0)
        self._snapshot_timesteps = []
        self._snapshot_mesh_fingerprint = self._current_mesh_fingerprint() if hasattr(self, "_current_mesh_fingerprint") else ""
        self._line_snapshot_rows = []
        self._line_snapshot_profile_rows = []
        self._coupling_snapshot_rows = []
        run_span_s = max(float(run_duration_s), 1.0e-9)
        _next_snap_t = min(output_interval_s, run_span_s)
        _next_line_snap_t = min(line_output_interval_s, run_span_s)
        _next_coupling_snap_t = min(line_output_interval_s, run_span_s)
        sample_map = self._build_line_sampling_map()
        cell_solver_z = self._mesh_cell_solver_bed() if sample_map else None
        run_id = datetime.datetime.now().astimezone().strftime("swe2d_%Y%m%dT%H%M%S%z")
        run_wallclock_start = datetime.datetime.now().replace(microsecond=0).isoformat(sep=" ")

        dynamic_bc = bool(np.any((bc_tp == _BC_TS_FLOW) | (bc_tp == _BC_TS_STAGE)) or edge_hydrographs)
        if dynamic_bc:
            self._log("Timeseries BC mode active (flow/stage hydrographs).")

        self._log("Starting 2D run...")
        self._log(f"Run wallclock start: {run_wallclock_start}")
        self._log(f"SWE2D solver backend: {solver_backend_mode}")
        self._log("SWE2D solver: GPU-only mode (CUDA)")
        self._log(f"Reconstruction mode: {reconstruction_name}")
        self._log(f"Temporal scheme: {temporal_scheme_name}")
        self._log(
            "SWE2D perf mode: "
            f"{'enabled' if bool(os.environ.get('BACKWATER_SWE2D_PERF_MODE', '0')) == '1' else 'disabled'}"
        )
        self._log(
            "Tiny-mode config: "
            f"mode={int(getattr(self, 'tiny_mode_combo').currentData()) if getattr(self, 'tiny_mode_combo', None) is not None else 3}, "
            f"wet_cell_threshold={int(getattr(self, 'tiny_wet_cell_threshold_spin').value()) if getattr(self, 'tiny_wet_cell_threshold_spin', None) is not None else 2000}"
        )
        self._log(
            f"Output intervals: mesh={output_interval_s:.1f}s, sample-lines={line_output_interval_s:.1f}s"
        )
        try:
            self.gpu_diag_sync_interval_spin.interpretText()
        except Exception:
            pass
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
        if initial_dt > 0.0:
            self._log(f"Initial dt override: {initial_dt:.5f} s (first step only)")
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
                f"drainage_backend={drainage_solver_backend_mode}, "
                f"drainage_gpu_method={drainage_gpu_method_mode}"
            )
            coupling_runtime_mode = "cuda"
            self._log(f"Coupling runtime mode: {coupling_runtime_mode}")
            if bridge_stacked_plans:
                total_bridge_cells = int(sum(int(p.selected_cells.size) for p in bridge_stacked_plans))
                self._log(
                    "Bridge stacked plans active: "
                    f"count={len(bridge_stacked_plans)}, selected_cells={total_bridge_cells}"
                )
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
                dt_initial=initial_dt,
                reconstruction_mode=reconstruction_mode,
                temporal_scheme=temporal_scheme,
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
        perf_mode = str(os.environ.get("BACKWATER_SWE2D_PERF_MODE", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
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
        if perf_mode:
            self._log(
                "SWE2D perf mode active: reduced runtime logging and disabled per-step source/boundary forensic accounting."
            )
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
                node_z = self._mesh_data["node_z"]
                native_bc_res = native_bc_cfg.configure(
                    backend=backend,
                    bc_n0=bc_n0,
                    bc_n1=bc_n1,
                    bc_tp=bc_tp,
                    side_hydrographs=side_hydrographs,
                    edge_hydrographs=edge_hydrographs,
                    node_x=node_x,
                    node_y=node_y,
                    node_z=node_z,
                    inflow_q_bc_type=int(_BC_INFLOW_Q),
                    progressive=progressive,
                    ts_flow_code=int(_BC_TS_FLOW),
                    ts_stage_code=int(_BC_TS_STAGE),
                )
                if bool(native_bc_res.get("native_bc_forcing", False)):
                    native_bc_forcing = True
                    self._log(
                        f"Native BC hydrograph forcing configured for {int(native_bc_res.get('configured_edges', 0))} boundary edges."
                    )
                    if bool(native_bc_res.get("progressive_uploaded", False)):
                        self._log(
                            f"Progressive inflow data uploaded for {int(native_bc_res.get('n_prog_edges', 0))} edges."
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
        self._log(f"The numbers they go UP! They go UP UP UP!!!")        
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
            enable_source_volume_accounting=(not perf_mode),
            enable_boundary_flux_accounting=(not perf_mode),
            record_source_step_rows=(not perf_mode),
            record_boundary_flux_step_rows=(not perf_mode),
        )
        source_budget_model = runtime_source_manager.source_budget_model
        source_step_rows_model = runtime_source_manager.source_step_rows_model
        boundary_flux_budget_model = runtime_source_manager.boundary_flux_budget_model
        boundary_flux_step_rows_model = runtime_source_manager.boundary_flux_step_rows_model
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
            cell_solver_z=cell_solver_z,
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
            perf_mode=perf_mode,
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
        sim_time_diff = float(t_accum) - float(run_duration_s)
        self._log(
            "Runtime simulated-time check: "
            f"sim_t={float(t_accum):.6f}s, target={float(run_duration_s):.6f}s, "
            f"delta={sim_time_diff:.6e}s"
        )
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
            final_sim_time_s=float(t_accum),
            n_area=n_area,
            area_model=area_model,
            storage_start_model=storage_start_model,
            source_budget_model=source_budget_model,
            source_step_rows_model=source_step_rows_model,
            run_duration_s=run_duration_s,
            boundary_flux_budget_model=boundary_flux_budget_model,
            boundary_flux_step_rows_model=boundary_flux_step_rows_model,
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





