"""Model tab view — owns its own widget references.

QWidget subclass for the Model tab in the Studio workbench.
Holds four pages inside a QToolBox:
  - model_solver_page (core + solver controls)
  - model_rain_page (rain / hydrology controls)
  - model_stability_page (stability controls)
  - model_drain_page (structures & drainage controls)
  - model_run_page (run / output controls)

Each form widget is owned as an instance attribute with a stable
objectName so existing binding code (e.g. ``findChild``) keeps working.
"""
from __future__ import annotations

from typing import List

from qgis.PyQt import QtWidgets

from swe2d.workbench.views.widget_filter_helper import FilterableRowRegistry


class HintLabel(QtWidgets.QLabel):
    """Small italic hint text under a parameter group."""
    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setStyleSheet("color: #888; font-style: italic; padding-left: 12px;")
        self.setWordWrap(True)
class ModelTabView(QtWidgets.QWidget):
    """View for the Model tab.

    Houses six QToolBox pages.  Every widget is created here as a direct
    instance attribute with a stable ``objectName``.

    Solver Parameters page (``model_solver_page``):
        manning_layer_combo (Physics & Friction group),
        n_mann_spin, cfl_spin, h_min_spin,
        initial_condition_combo, initial_depth_spin, initial_wse_spin,
        adaptive_cfl_dt_chk, dt_spin, initial_dt_spin,
        gpu_diag_sync_interval_spin,
        tiny_mode_combo, tiny_wet_cell_threshold_spin,
        enable_cuda_graphs_chk, swe2d_perf_mode_chk,
        internal_flow_layer_combo, internal_flow_field_edit,
        run_time_edit, reconstruction_combo, temporal_order_combo

    Rain / Hydrology page (``model_rain_page``):
        rain_gage_layer_combo, hyetograph_layer_combo (Rainfall Input group),
        rain_rate_spin, use_spatial_rain_cn_chk,
        rain_update_interval_spin, storm_area_layer_combo,
        rain_boundary_buffer_rings_spin,
        infiltration_method_combo, cn_layer_combo (Infiltration group),
        cn_default_spin, ia_ratio_spin,
        max_rel_depth_increase_spin, max_source_depth_step_spin,
        max_source_rate_spin, extreme_rain_mode_chk,
        source_cfl_beta_spin, source_max_substeps_spin,
        source_true_subcycling_chk, source_imex_split_chk

    Stability Controls page (``model_stability_page``):
        shallow_damping_depth_spin, shallow_front_recon_fallback_chk,
        front_flux_damping_spin, active_set_hysteresis_chk,
        depth_cap_spin, momentum_cap_min_speed_spin,
        momentum_cap_celerity_mult_spin, max_inv_area_spin,
        cfl_lambda_cap_spin

    Structures & Drainage page (``model_drain_page``):
        drain_nodes_layer_combo, drain_links_layer_combo,
        drain_inlets_layer_combo, drain_node_inlets_layer_combo,
        structures_layer_combo (Layer Setup group),
        coupling_loop_combo, culvert_solver_mode_combo,
        culvert_face_flux_chk, use_redistribution_chk,
        bridge_stacked_coupling_mode_combo,
        drainage_solver_mode_combo, drainage_gpu_method_combo,
        drainage_coupling_substeps_spin,
        drainage_max_coupling_substeps_spin,
        drainage_head_deadband_spin, drainage_dynamic_relaxation_spin,
        drainage_adaptive_depth_fraction_spin,
        drainage_adaptive_wave_courant_spin,
        drainage_implicit_iters_spin, drainage_implicit_relax_spin,
        gpu_default_lbl, unit_system_lbl

    The following run/output widgets are now created as orphan attributes so
    existing binding code keeps working, but they physically live in the
    dedicated Run dock:

        run_btn, cancel_btn, batch_sim_btn, progress_bar,
        output_interval_edit, line_output_interval_edit,
        preview_overrides_btn, preview_coupling_btn, snapshot_btn,
        results_table_name_edit, results_gpkg_path_edit,
        select_results_gpkg_btn, load_run_settings_btn, save_settings_btn
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the toolbox with all five parameter pages."""
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._param_groups: List[QtWidgets.QGroupBox] = []
        # Filter registry — every widget that should respond to the
        # param_search / show_advanced_chk controls is registered via
        # _add_param_row or directly via _filterable.add(...). This
        # includes the Output-page widgets registered in
        # _build_output_page().
        self._filterable: FilterableRowRegistry = FilterableRowRegistry()

        filter_bar = QtWidgets.QHBoxLayout()
        self.param_search = QtWidgets.QLineEdit()
        self.param_search.setObjectName("param_search")
        self.param_search.setPlaceholderText("Filter parameters…")
        self.param_search.textChanged.connect(self._filter_model_tab)
        self.show_advanced_chk = QtWidgets.QCheckBox("Show advanced parameters")
        self.show_advanced_chk.setObjectName("show_advanced_chk")
        self.show_advanced_chk.setChecked(False)
        self.show_advanced_chk.toggled.connect(self._filter_model_tab)
        filter_bar.addWidget(self.param_search, 1)
        filter_bar.addWidget(self.show_advanced_chk)
        root_layout.addLayout(filter_bar)

        self.model_toolbox = QtWidgets.QToolBox()
        self.model_toolbox.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        root_layout.addWidget(self.model_toolbox)

        self.model_solver_page, self.model_solver_form = self._build_form_page(
            "model_solver_page", "model_solver_form"
        )
        self._build_solver_form_widgets(self.model_solver_form)

        self.model_rain_page, self.model_rain_form = self._build_form_page(
            "model_rain_page", "model_rain_form"
        )
        self._build_rain_form_widgets(self.model_rain_form)

        self.model_stability_page, self.model_stability_form = self._build_form_page(
            "model_stability_page", "model_stability_form"
        )
        self._build_stability_form_widgets(self.model_stability_form)

        self.model_drain_page, self.model_drain_form = self._build_form_page(
            "model_drain_page", "model_drain_form"
        )
        self._build_drain_form_widgets(self.model_drain_form)

        self._build_output_page()

        self.model_toolbox.addItem(self.model_solver_page, "Solver Parameters")
        self.model_toolbox.addItem(self.model_rain_page, "Rain / Hydrology")
        self.model_toolbox.addItem(self.model_stability_page, "Stability Controls")
        self.model_toolbox.addItem(self.model_drain_page, "Structures & Drainage")
        self.model_toolbox.addItem(self.model_output_page, "Output")

        for i in range(self.model_toolbox.count()):
            page = self.model_toolbox.widget(i)
            if page is not None:
                page.setSizePolicy(
                    QtWidgets.QSizePolicy.Preferred,
                    QtWidgets.QSizePolicy.Expanding,
                )

    def _build_form_page(
        self, page_name: str, form_name: str
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        """Create a toolbox page with a form layout."""
        page = QtWidgets.QWidget()
        page.setObjectName(page_name)
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        form = QtWidgets.QFormLayout()
        form.setObjectName(form_name)
        page_layout.addLayout(form)
        return page, form

    def _start_param_group(
        self,
        page_layout: QtWidgets.QFormLayout,
        title: str,
        checkable: bool = False,
        advanced: bool = False,
    ) -> QtWidgets.QFormLayout:
        """Create a titled, collapsible group box and add it to a page."""
        group = QtWidgets.QGroupBox(title)
        group.setObjectName(title.lower().replace(" ", "_").replace("&", "and") + "_group")
        group.setCheckable(checkable)
        if checkable:
            group.setChecked(False)
        if advanced:
            group.setProperty("advanced", True)
        group_layout = QtWidgets.QFormLayout(group)
        group_layout.setObjectName(group.objectName() + "_layout")
        page_layout.addRow(group)
        self._param_groups.append(group)
        return group_layout

    def _add_param_row(
        self,
        group_layout: QtWidgets.QFormLayout,
        label_text: str,
        widget: QtWidgets.QWidget,
        advanced: bool = False,
    ) -> None:
        """Add a labeled widget to a group and register it for filtering."""
        label = QtWidgets.QLabel(label_text)
        group_layout.addRow(label, widget)
        group = group_layout.parentWidget()
        # Register via the shared registry so this row participates in
        # the param_search/show_advanced filter alongside the Output page.
        self._filterable.add(
            widget,
            label_widget=label,
            label_text=label_text,
            tooltip=widget.toolTip() or "",
            group=group if isinstance(group, QtWidgets.QGroupBox) else None,
            advanced=advanced,
        )
        if advanced:
            label.setProperty("advanced", True)

    def _filter_model_tab(self, _value=None) -> None:
        """Show/hide parameter rows based on search text and advanced toggle."""
        self._filterable.apply_filter(
            self.param_search.text(),
            show_advanced=self.show_advanced_chk.isChecked(),
        )

    def _build_run_page_widgets(
        self, run_page: QtWidgets.QWidget, run_page_layout: QtWidgets.QVBoxLayout
    ) -> None:
        """No-op — the Run/Output page has been moved to the Run dock."""
        _ = (run_page, run_page_layout)

    # ------------------------------------------------------------------
    # Output page (storage checkboxes — moved from ResultsToolbox)
    # ------------------------------------------------------------------

    def _build_output_page(self) -> None:
        """Build the Output page with save-to-GPKG controls.

        Uses the same groupbox-based pattern as the other parameter
        pages (Solver Parameters, Rain / Hydrology, etc.) — see
        ``_start_param_group`` and ``_add_param_row`` for details.
        """
        self.model_output_page, layout = self._build_form_page(
            "model_output_page", "model_output_form"
        )

        # ── Output Options (checkboxes that control what's saved) ─────
        form = self._start_param_group(layout, "Output Options")

        def _add_output_checkbox(
            chk: QtWidgets.QCheckBox, label_text: str
        ) -> None:
            """Add a self-describing checkbox to the Output Options group."""
            self._add_param_row(form, label_text, chk)

        self.extended_outputs_chk = QtWidgets.QCheckBox(
            "Include extended outputs (momentum, qmag, wet mask, Fr, Manning)"
        )
        self.extended_outputs_chk.setToolTip(
            "Include extended output fields: momentum components, discharge magnitude, "
            "wet mask, Froude number, and Manning n. Increases result file size."
        )
        self.extended_outputs_chk.setObjectName("extended_outputs_chk")
        self.extended_outputs_chk.setChecked(True)
        _add_output_checkbox(self.extended_outputs_chk, "Extended outputs:")

        self.save_mesh_chk = QtWidgets.QCheckBox("Save mesh results to GPKG")
        self.save_mesh_chk.setToolTip(
            "Save 2D mesh simulation results (depth, velocity, WSE) to the GeoPackage."
        )
        self.save_mesh_chk.setObjectName("save_mesh_chk")
        self.save_mesh_chk.setChecked(True)
        _add_output_checkbox(self.save_mesh_chk, "Save mesh:")

        self.save_line_chk = QtWidgets.QCheckBox("Save line results to GPKG")
        self.save_line_chk.setToolTip(
            "Save sample line (cross-section) results to the GeoPackage."
        )
        self.save_line_chk.setObjectName("save_line_chk")
        self.save_line_chk.setChecked(True)
        _add_output_checkbox(self.save_line_chk, "Save line:")

        self.save_coupling_chk = QtWidgets.QCheckBox("Save coupling results to GPKG")
        self.save_coupling_chk.setToolTip(
            "Save drainage/structure coupling time series results to the GeoPackage."
        )
        self.save_coupling_chk.setObjectName("save_coupling_chk")
        self.save_coupling_chk.setChecked(True)
        _add_output_checkbox(self.save_coupling_chk, "Save coupling:")

        self.save_max_only_chk = QtWidgets.QCheckBox(
            "Save max results only (skip interval snapshots)"
        )
        self.save_max_only_chk.setToolTip(
            "Only save maximum-value results per cell. "
            "Skips interval snapshots to reduce file size."
        )
        self.save_max_only_chk.setObjectName("save_max_only_chk")
        self.save_max_only_chk.setChecked(False)
        _add_output_checkbox(self.save_max_only_chk, "Max only:")

        self.save_log_chk = QtWidgets.QCheckBox("Save run log to GPKG")
        self.save_log_chk.setToolTip(
            "Save the solver run log (diagnostics, timesteps, errors) to the GeoPackage."
        )
        self.save_log_chk.setObjectName("save_log_chk")
        self.save_log_chk.setChecked(True)
        _add_output_checkbox(self.save_log_chk, "Save log:")

        # ── Sampling (sample lines layer) ─────────────────────────────
        form = self._start_param_group(layout, "Sampling")
        self.sample_lines_layer_combo = QtWidgets.QComboBox()
        self.sample_lines_layer_combo.setObjectName("sample_lines_layer_combo")
        self.sample_lines_layer_combo.addItem("(none)", None)
        self.sample_lines_layer_combo.setToolTip(
            "Line layer for sampling flow results along cross-sections during simulation. "
            "Results are saved at the line output interval specified below."
        )
        self._add_param_row(
            form, "Sample lines layer:", self.sample_lines_layer_combo
        )

        # ── Result Storage (intervals + GPKG path + config buttons) ───
        form = self._start_param_group(layout, "Result Storage")

        self.output_interval_edit = QtWidgets.QLineEdit("00:30")
        self.output_interval_edit.setObjectName("output_interval_edit")
        self.output_interval_edit.setToolTip(
            "Time interval between 2D mesh result output writes. "
            "Format: decimal hours (e.g. 0.5) or HH:MM (e.g. 00:30). "
            "Smaller intervals produce larger result files."
        )
        self._add_param_row(
            form, "Output interval (hr or HH:MM):", self.output_interval_edit
        )

        self.line_output_interval_edit = QtWidgets.QLineEdit("00:05")
        self.line_output_interval_edit.setObjectName("line_output_interval_edit")
        self.line_output_interval_edit.setToolTip(
            "Time interval between sample-line (cross-section) result outputs. "
            "Format: decimal hours or HH:MM. Default: 00:05 (5 min)."
        )
        self._add_param_row(
            form, "Line output interval:", self.line_output_interval_edit
        )

        self.results_table_name_edit = QtWidgets.QLineEdit()
        self.results_table_name_edit.setObjectName("results_table_name_edit")
        self.results_table_name_edit.setToolTip(
            "Optional prefix for GeoPackage result table names. "
            "Useful when storing multiple model runs in the same GeoPackage."
        )
        self.results_table_name_edit.setPlaceholderText("optional table prefix")
        self._add_param_row(
            form, "Table prefix:", self.results_table_name_edit
        )

        # Results GPKG row = QLineEdit + Browse button (horizontal layout)
        gpkg_row_widget = QtWidgets.QWidget()
        gpkg_row_layout = QtWidgets.QHBoxLayout(gpkg_row_widget)
        gpkg_row_layout.setContentsMargins(0, 0, 0, 0)
        self.results_gpkg_path_edit = QtWidgets.QLineEdit()
        self.results_gpkg_path_edit.setObjectName("results_gpkg_path_edit")
        self.results_gpkg_path_edit.setToolTip(
            "Path to the output GeoPackage for storing simulation results. "
            "Leave empty to use the model GeoPackage."
        )
        self.results_gpkg_path_edit.setPlaceholderText("GeoPackage path (optional)")
        gpkg_row_widget.setObjectName("results_gpkg_row")
        self._add_param_row(form, "Results GPKG:", gpkg_row_widget)
        self.select_results_gpkg_btn = QtWidgets.QPushButton("Browse…")
        self.select_results_gpkg_btn.setObjectName("select_results_gpkg_btn")
        self.select_results_gpkg_btn.setToolTip(
            "Browse for an existing GeoPackage to store/load simulation results."
        )
        gpkg_row_layout.addWidget(self.results_gpkg_path_edit, 1)
        gpkg_row_layout.addWidget(self.select_results_gpkg_btn)

        # ── Config (preview / load / save — each on its own row) ──────
        form = self._start_param_group(layout, "Config")

        self.preview_overrides_btn = QtWidgets.QPushButton("Preview Overrides")
        self.preview_overrides_btn.setObjectName("preview_overrides_btn")
        self.preview_overrides_btn.setToolTip(
            "Display a summary of all current parameter overrides "
            "before running the simulation."
        )
        self._add_param_row(form, "Preview Overrides:", self.preview_overrides_btn)

        self.preview_coupling_btn = QtWidgets.QPushButton("Preview Coupling")
        self.preview_coupling_btn.setObjectName("preview_coupling_btn")
        self.preview_coupling_btn.setToolTip(
            "Preview the 1D-2D coupling configuration for drainage "
            "and hydraulic structures before running."
        )
        self._add_param_row(form, "Preview Coupling:", self.preview_coupling_btn)

        self.load_run_settings_btn = QtWidgets.QPushButton("Load Config from GPKG…")
        self.load_run_settings_btn.setObjectName("load_run_settings_btn")
        self.load_run_settings_btn.setToolTip(
            "Open a GeoPackage and restore a saved simulation configuration "
            "(all widget values, solver params, and layer references)."
        )
        self._add_param_row(form, "Load Config:", self.load_run_settings_btn)

        self.save_settings_btn = QtWidgets.QPushButton("Save Config to GPKG…")
        self.save_settings_btn.setObjectName("save_settings_btn")
        self.save_settings_btn.setToolTip(
            "Save the current widget configuration to the active GeoPackage "
            "so it can be restored later via Load Config."
        )
        self._add_param_row(form, "Save Config:", self.save_settings_btn)

    def is_extended_outputs(self) -> bool:
        return bool(self.extended_outputs_chk.isChecked())

    def is_save_mesh(self) -> bool:
        return bool(self.save_mesh_chk.isChecked())

    def is_save_line(self) -> bool:
        return bool(self.save_line_chk.isChecked())

    def is_save_coupling(self) -> bool:
        return bool(self.save_coupling_chk.isChecked())

    def is_save_max_only(self) -> bool:
        return bool(self.save_max_only_chk.isChecked())

    def is_save_log(self) -> bool:
        return bool(self.save_log_chk.isChecked())

    def get_storage_checkboxes(self) -> dict:
        """Return storage checkboxes by key."""
        return {
            "extended_outputs": self.extended_outputs_chk,
            "save_mesh": self.save_mesh_chk,
            "save_line": self.save_line_chk,
            "save_coupling": self.save_coupling_chk,
            "save_max_only": self.save_max_only_chk,
            "save_log": self.save_log_chk,
        }

    def collect_storage_params(self) -> dict:
        """Return storage-checkbox parameter values as a flat dict.

        Same schema previously produced by ResultsToolbox.collect_storage_params
        so downstream controllers (run_controller, batch_simulation_dialog)
        don't need to change.
        """
        return {
            "extended_outputs_chk": bool(self.extended_outputs_chk.isChecked()),
            "save_mesh_results_to_gpkg_chk": (
                bool(self.save_mesh_chk.isChecked())
                and not bool(self.save_max_only_chk.isChecked())
            ),
            "save_line_results_to_gpkg_chk": (
                bool(self.save_line_chk.isChecked())
                and not bool(self.save_max_only_chk.isChecked())
            ),
            "save_coupling_results_to_gpkg_chk": (
                bool(self.save_coupling_chk.isChecked())
                and not bool(self.save_max_only_chk.isChecked())
            ),
            "save_max_only_chk": bool(self.save_max_only_chk.isChecked()),
            "save_run_log_to_gpkg_chk": bool(self.save_log_chk.isChecked()),
        }

    def _build_solver_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Solver Parameters page with grouped controls."""
        # Spatial Manning layer (relocated from the removed Layers page)
        self.manning_layer_combo = QtWidgets.QComboBox()
        self.manning_layer_combo.setObjectName("manning_layer_combo")
        self.manning_layer_combo.addItem("(none)", None)
        # -- Time Stepping --
        form = self._start_param_group(param_form, "Time Stepping")
        self.cfl_spin = QtWidgets.QDoubleSpinBox()
        self.cfl_spin.setObjectName("cfl_spin")
        self.cfl_spin.setToolTip(
            "Courant-Friedrichs-Lewy number for explicit timestep control. "
            "Typical range: 0.1–0.8. Lower values improve stability at the "
            "cost of smaller timesteps. Must be < 1.0 for explicit schemes."
        )
        self._add_param_row(form, "CFL:", self.cfl_spin)
        self.cfl_spin.setRange(0.01, 0.99)
        self.cfl_spin.setDecimals(3)
        self.cfl_spin.setValue(0.45)

        self.dt_spin = QtWidgets.QDoubleSpinBox()
        self.dt_spin.setObjectName("dt_spin")
        self.dt_spin.setToolTip(
            "Fixed timestep when variable timestep is disabled. "
            "Acts as dt_max (upper bound) when variable timestep is enabled. "
            "Units: seconds or model time. Range: 0.0001–1e6."
        )
        self._add_param_row(form, "dt (max):", self.dt_spin)
        self.dt_spin.setRange(1.0e-4, 1.0e6)
        self.dt_spin.setDecimals(5)
        self.dt_spin.setValue(0.05)

        self.initial_dt_spin = QtWidgets.QDoubleSpinBox()
        self.initial_dt_spin.setObjectName("initial_dt_spin")
        self.initial_dt_spin.setToolTip(
            "Timestep used for the first step before the adaptive CFL adjusts. "
            "Set to 0 (default) for automatic selection based on mesh size and CFL."
        )
        self._add_param_row(form, "Initial dt:", self.initial_dt_spin)
        self.initial_dt_spin.setRange(0.0, 1.0e6)
        self.initial_dt_spin.setDecimals(5)
        self.initial_dt_spin.setValue(0.0)

        self.adaptive_cfl_dt_chk = QtWidgets.QCheckBox("Enable variable timestep (CFL)")
        self.adaptive_cfl_dt_chk.setObjectName("adaptive_cfl_dt_chk")
        self.adaptive_cfl_dt_chk.setToolTip(
            "When checked, dt is computed adaptively each step based on the CFL condition "
            "and current flow velocity. When unchecked, a fixed dt is used."
        )
        self._add_param_row(form, "Variable timestep:", self.adaptive_cfl_dt_chk)
        self.adaptive_cfl_dt_chk.setChecked(False)

        form.addRow(HintLabel("Adaptive dt uses the CFL condition each step."))

        # -- Boundary Conditions --
        form = self._start_param_group(param_form, "Boundary Conditions")
        self.default_bc_type_combo = QtWidgets.QComboBox()
        self.default_bc_type_combo.setObjectName("default_bc_type_combo")
        self.default_bc_type_combo.setToolTip(
            "Default boundary condition type for all BC line segments. "
            "Per-segment overrides can be set via the BC layer attributes. "
            "Options: Wall (no flux), Inflow Q (discharge), Stage (WSE), "
            "Normal Depth, Timeseries Flow/Stage, Open (zero-gradient), or Reflecting."
        )
        from swe2d.workbench.views.map_tab_view import _BC_OPTIONS
        for label, code in _BC_OPTIONS:
            self.default_bc_type_combo.addItem(label, code)
        self.default_bc_type_combo.setCurrentIndex(2)
        self._add_param_row(form, "Default BC type:", self.default_bc_type_combo)

        # BC lines layer — relocated to row 2 of the Boundary Conditions group
        self.bc_lines_layer_combo = QtWidgets.QComboBox()
        self.bc_lines_layer_combo.setObjectName("bc_lines_layer_combo")
        self.bc_lines_layer_combo.setToolTip(
            "Line layer for boundary condition segments. "
            "Each segment defines a BC type (inflow, stage, normal depth, etc.) "
            "assigned via the default BC type combo or per-segment attributes."
        )
        self.bc_lines_layer_combo.addItem("(none)", None)
        self._add_param_row(form, "BC lines layer:", self.bc_lines_layer_combo)

        self.inflow_progressive_chk = QtWidgets.QCheckBox("Inflow progressive")
        self.inflow_progressive_chk.setObjectName("inflow_progressive_chk")
        self.inflow_progressive_chk.setToolTip(
            "When checked, inflow is ramped up gradually at the start of the simulation "
            "to avoid numerical shock from a sudden full-discharge boundary."
        )
        self.inflow_progressive_chk.setChecked(False)
        self._add_param_row(form, "", self.inflow_progressive_chk)

        self.uniform_inflow_velocity_chk = QtWidgets.QCheckBox("Uniform inflow velocity")
        self.uniform_inflow_velocity_chk.setObjectName("uniform_inflow_velocity_chk")
        self.uniform_inflow_velocity_chk.setToolTip(
            "When checked, inflow boundary cells receive a uniform velocity profile. "
            "Leave unchecked for a more realistic parabolic (shear) velocity distribution."
        )
        self.uniform_inflow_velocity_chk.setChecked(False)
        self._add_param_row(form, "", self.uniform_inflow_velocity_chk)

        # -- Physics & Friction --
        form = self._start_param_group(param_form, "Physics & Friction")
        # Spatial Manning layer — moved from the Layers page
        self.manning_layer_combo.setToolTip(
            "Polygon layer with Manning's n values for spatially varying roughness. "
            "Field must contain a numeric roughness column. Leave empty for uniform n "
            "set in the Model tab."
        )
        self._add_param_row(form, "Manning polygons:", self.manning_layer_combo)
        self.n_mann_spin = QtWidgets.QDoubleSpinBox()
        self.n_mann_spin.setObjectName("n_mann_spin")
        self.n_mann_spin.setToolTip(
            "Manning's roughness coefficient n. Typical values: 0.012 (concrete), "
            "0.020 (gravel), 0.035 (natural stream), 0.050 (floodplain). "
            "Range: 0.0–1.0."
        )
        self._add_param_row(form, "Manning n:", self.n_mann_spin)
        self.n_mann_spin.setRange(0.0, 1.0)
        self.n_mann_spin.setDecimals(5)
        self.n_mann_spin.setValue(0.020)

        self.h_min_spin = QtWidgets.QDoubleSpinBox()
        self.h_min_spin.setObjectName("h_min_spin")
        self.h_min_spin.setToolTip(
            "Minimum water depth threshold (model units). Cells with depth "
            "below h_min are treated as dry and excluded from momentum computation. "
            "Typical: 1e-6 (SI) or 1e-5 (USC). Increase for noisy terrain."
        )
        self._add_param_row(form, "h_min:", self.h_min_spin)
        self.h_min_spin.setRange(1.0e-9, 1.0)
        self.h_min_spin.setDecimals(8)
        self.h_min_spin.setValue(1.0e-6)

        self.internal_flow_layer_combo = QtWidgets.QComboBox()
        self.internal_flow_layer_combo.setObjectName("internal_flow_layer_combo")
        self.internal_flow_layer_combo.setToolTip(
            "Polygon layer defining internal source/sink flow regions. "
            "Each polygon specifies a flow rate (e.g. q_cms) applied as a source term. "
            "Select '(none)' to disable internal flows."
        )
        self._add_param_row(form, "Internal flow layer:", self.internal_flow_layer_combo)
        self.internal_flow_layer_combo.addItem("(none)", None)

        self.internal_flow_field_edit = QtWidgets.QLineEdit()
        self.internal_flow_field_edit.setObjectName("internal_flow_field_edit")
        self.internal_flow_field_edit.setToolTip(
            "Field name in the internal flow layer containing source/sink discharge values. "
            "Default: 'q_cms' (CMS). Positive values = source, negative = sink."
        )
        self.internal_flow_field_edit.setText("q_cms")
        self.internal_flow_field_edit.setPlaceholderText("field name, e.g. q_cms")
        self._add_param_row(form, "Internal flow field:", self.internal_flow_field_edit)

        # -- Numerics --
        form = self._start_param_group(param_form, "Numerics")
        self.reconstruction_combo = QtWidgets.QComboBox()
        self.reconstruction_combo.setObjectName("reconstruction_combo")
        self.reconstruction_combo.setToolTip(
            "Spatial reconstruction scheme for cell-face extrapolation. "
            "First-order (0): fastest, most diffusive. MUSCL variants (1–4) "
            "add second-order accuracy with different slope limiters. "
            "WENO (5–6) use weighted stencils for smooth regions with shock capture."
        )
        for text, data in [
            ("1st-order", 0),
            ("MUSCL + Superbee", 1),
            ("MUSCL + MinMod", 2),
            ("MUSCL + MC", 3),
            ("MUSCL + Van Leer", 4),
            ("WENO3-like", 5),
            ("WENO5", 6),
        ]:
            self.reconstruction_combo.addItem(text, data)
        self._add_param_row(form, "Reconstruction:", self.reconstruction_combo)

        self.temporal_order_combo = QtWidgets.QComboBox()
        self.temporal_order_combo.setObjectName("temporal_order_combo")
        self.temporal_order_combo.setToolTip(
            "Temporal integration (ODE solver) order. "
            "Euler (1): CFL 0.4-0.5 (stable up to 1.0). "
            "SSP-RK2 (2): CFL 0.5-0.8 (default, SSP-stable). "
            "SSP-RK3 (3): CFL 0.5-0.8 (Shu-Osher, SSP-stable). "
            "Classic RK4 (4): NOT SSP — CFL 0.3-0.5 with Superbee/limiters, "
            "up to 0.8 with first-order. Use SSP-RK2/RK3 for higher CFL. "
            "Graph-safe RK4 (5): same as classic RK4. "
            "Graph-safe RK5 (6): CFL 0.5-1.0 (Cash-Karp embedded)."
        )
        for text, data in [
            ("RK1 (Euler)", 1),
            ("RK2 (Heun)", 2),
            ("RK3 (SSP Shu-Osher)", 3),
            ("RK4 (classic)", 4),
            ("RK4 (graph-safe)", 5),
            ("RK5 (graph-safe)", 6),
        ]:
            self.temporal_order_combo.addItem(text, data)
        self._add_param_row(form, "Temporal discretization:", self.temporal_order_combo)

        def _on_temporal_order_changed(idx: int) -> None:
            order = self.temporal_order_combo.itemData(idx)
            tiny_idx = self.tiny_mode_combo.findData(3)
            if tiny_idx >= 0:
                item = self.tiny_mode_combo.model().item(tiny_idx)
                if item is not None:
                    item.setEnabled(order is not None and int(order) < 3)
        self.temporal_order_combo.currentIndexChanged.connect(_on_temporal_order_changed)

        # -- Initial Conditions --
        form = self._start_param_group(param_form, "Initial Conditions", checkable=True)
        self.initial_condition_combo = QtWidgets.QComboBox()
        self.initial_condition_combo.setObjectName("initial_condition_combo")
        self.initial_condition_combo.setToolTip(
            "Initial condition for the entire domain. 'Dry start' sets depth=0 "
            "everywhere. 'Uniform depth' uses a constant depth. 'Uniform WSE' "
            "sets a constant water surface elevation (depth = WSE - bed)."
        )
        self._add_param_row(form, "Initial condition:", self.initial_condition_combo)
        self.initial_condition_combo.addItem("Dry start", "dry")
        self.initial_condition_combo.addItem("Uniform depth", "uniform_depth")
        self.initial_condition_combo.addItem(
            "Uniform water surface elevation", "uniform_wse"
        )
        self.initial_condition_combo.setCurrentIndex(
            self.initial_condition_combo.findData("dry")
        )

        self.initial_depth_spin = QtWidgets.QDoubleSpinBox()
        self.initial_depth_spin.setObjectName("initial_depth_spin")
        self.initial_depth_spin.setToolTip(
            "Uniform initial water depth across the domain (model units). "
            "Only used when 'Initial condition' is 'Uniform depth'. Set to 0 for dry start."
        )
        self._add_param_row(form, "Initial depth:", self.initial_depth_spin)
        self.initial_depth_spin.setRange(0.0, 1.0e6)
        self.initial_depth_spin.setDecimals(4)
        self.initial_depth_spin.setValue(0.0)

        self.initial_wse_spin = QtWidgets.QDoubleSpinBox()
        self.initial_wse_spin.setObjectName("initial_wse_spin")
        self.initial_wse_spin.setToolTip(
            "Uniform initial water surface elevation (model units). "
            "Only used when 'Initial condition' is 'Uniform WSE'. "
            "Depth is computed as WSE minus bed elevation for each cell."
        )
        self._add_param_row(form, "Initial WSE:", self.initial_wse_spin)
        self.initial_wse_spin.setRange(-1.0e6, 1.0e6)
        self.initial_wse_spin.setDecimals(4)
        self.initial_wse_spin.setValue(0.0)

        form.addRow(HintLabel("Dry start uses bed elevation only."))

        # -- Performance --
        form = self._start_param_group(param_form, "Performance", advanced=True)
        self.enable_cuda_graphs_chk = QtWidgets.QCheckBox("Enable")
        self.enable_cuda_graphs_chk.setObjectName("enable_cuda_graphs_chk")
        self.enable_cuda_graphs_chk.setToolTip(
            "Enable CUDA graph replay for solver kernel launches. "
            "Reduces kernel launch overhead by capturing and replaying the "
            "entire kernel graph. Only effective on CUDA 10+ with stable kernel topology."
        )
        self._add_param_row(form, "CUDA graph replay:", self.enable_cuda_graphs_chk, advanced=True)
        self.enable_cuda_graphs_chk.setChecked(False)

        self.swe2d_perf_mode_chk = QtWidgets.QCheckBox("Enable")
        self.swe2d_perf_mode_chk.setObjectName("swe2d_perf_mode_chk")
        self.swe2d_perf_mode_chk.setToolTip(
            "High-performance mode for the SWE2D solver. Enables aggressive "
            "optimizations (kernel fusion, reduced synchronization, stream overlap) "
            "for maximum GPU throughput. May reduce diagnostic granularity."
        )
        self._add_param_row(form, "SWE2D perf mode:", self.swe2d_perf_mode_chk, advanced=True)
        self.swe2d_perf_mode_chk.setChecked(False)

        self.gpu_diag_sync_interval_spin = QtWidgets.QSpinBox()
        self.gpu_diag_sync_interval_spin.setObjectName("gpu_diag_sync_interval_spin")
        self.gpu_diag_sync_interval_spin.setToolTip(
            "Number of solver steps between GPU diagnostics synchronization. "
            "Higher values reduce GPU/CPU sync overhead. Range: 1–1,000,000. "
            "Default: 10 steps."
        )
        self._add_param_row(form, "GPU diag sync (steps):", self.gpu_diag_sync_interval_spin, advanced=True)
        self.gpu_diag_sync_interval_spin.setRange(1, 1000000)
        self.gpu_diag_sync_interval_spin.setValue(10)

        self.tiny_mode_combo = QtWidgets.QComboBox()
        self.tiny_mode_combo.setObjectName("tiny_mode_combo")
        self.tiny_mode_combo.setToolTip(
            "Mode for handling tiny/wet-dry cells. "
            "Off (0): standard treatment. Auto (1): automatic detection. "
            "Fused (2): fused kernel for tiny cells. "
            "Persistent (3): persistent tiny-cell state across steps (default)."
        )
        self._add_param_row(form, "Tiny mode:", self.tiny_mode_combo)
        self.tiny_mode_combo.addItem("Disabled", 0)
        self.tiny_mode_combo.addItem("Auto-detect", 1)
        self.tiny_mode_combo.addItem("Fused", 2)
        self.tiny_mode_combo.addItem("Persistent", 3)
        self.tiny_mode_combo.setCurrentIndex(self.tiny_mode_combo.findData(3))

        self.tiny_wet_cell_threshold_spin = QtWidgets.QSpinBox()
        self.tiny_wet_cell_threshold_spin.setObjectName("tiny_wet_cell_threshold_spin")
        self.tiny_wet_cell_threshold_spin.setToolTip(
            "Maximum number of active/wet cells before tiny-mode optimization engages. "
            "Range: 1–10,000,000. Default: 2000."
        )
        self._add_param_row(
            form, "Tiny active/wet cell threshold:", self.tiny_wet_cell_threshold_spin
        )
        self.tiny_wet_cell_threshold_spin.setRange(1, 10000000)
        self.tiny_wet_cell_threshold_spin.setValue(2000)

        # -- Run Duration --
        form = self._start_param_group(param_form, "Run Duration")
        self.run_time_edit = QtWidgets.QLineEdit()
        self.run_time_edit.setObjectName("run_time_edit")
        self.run_time_edit.setToolTip(
            "Total simulation duration. Enter as decimal hours (e.g. 1.5) "
            "or HH:MM format (e.g. 01:30 for 1 hour 30 min)."
        )
        self.run_time_edit.setText("1:00")
        self.run_time_edit.setPlaceholderText(
            "decimal hours (e.g. 1.5) or HH:MM (e.g. 01:30)"
        )
        self._add_param_row(form, "Run duration:", self.run_time_edit)

        _on_temporal_order_changed(self.temporal_order_combo.currentIndex())

    def _build_rain_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Rain / Hydrology page with grouped rainfall controls."""
        # Layer combos (relocated from the removed Layers page)
        self.rain_gage_layer_combo = QtWidgets.QComboBox()
        self.rain_gage_layer_combo.setObjectName("rain_gage_layer_combo")
        self.rain_gage_layer_combo.addItem("(none)", None)
        self.hyetograph_layer_combo = QtWidgets.QComboBox()
        self.hyetograph_layer_combo.setObjectName("hyetograph_layer_combo")
        self.hyetograph_layer_combo.addItem("(none)", None)
        self.cn_layer_combo = QtWidgets.QComboBox()
        self.cn_layer_combo.setObjectName("cn_layer_combo")
        self.cn_layer_combo.addItem("(none)", None)
        form = self._start_param_group(param_form, "Rainfall Input")
        # Spatial rainfall layers — moved from the Layers page
        self.rain_gage_layer_combo.setToolTip(
            "Point layer defining rain gauge locations. Each gauge should have an ID "
            "matching entries in the hyetograph table layer."
        )
        self._add_param_row(form, "Rain gages (points):", self.rain_gage_layer_combo)
        self.hyetograph_layer_combo.setToolTip(
            "Table layer containing precipitation hyetographs. Columns: time (hours) "
            "and rainfall intensity (mm/hr or in/hr) for each gauge."
        )
        self._add_param_row(form, "Rain hyetographs (table):", self.hyetograph_layer_combo)

        self.rain_rate_spin = QtWidgets.QDoubleSpinBox()
        self.rain_rate_spin.setObjectName("rain_rate_spin")
        self.rain_rate_spin.setToolTip(
            "Uniform rainfall rate in mm/hr applied to the entire domain. "
            "Range: 0–2000 mm/hr. Use spatial rainfall layers for non-uniform "
            "rainfall distribution."
        )
        self._add_param_row(form, "Rain rate:", self.rain_rate_spin)
        self.rain_rate_spin.setRange(0.0, 2000.0)
        self.rain_rate_spin.setDecimals(3)
        self.rain_rate_spin.setValue(0.0)
        self.rain_rate_spin.setSuffix(" mm/hr")

        self.use_spatial_rain_cn_chk = QtWidgets.QCheckBox(
            "Use Thiessen gage rainfall when layers are available"
        )
        self.use_spatial_rain_cn_chk.setObjectName("use_spatial_rain_cn_chk")
        self.use_spatial_rain_cn_chk.setToolTip(
            "When checked and rain gage + hyetograph layers are configured, "
            "Thiessen polygon interpolation is applied for spatially variable "
            "rainfall. Uses CN polygons for spatial infiltration if available."
        )
        self._add_param_row(form, "Spatial rainfall:", self.use_spatial_rain_cn_chk)
        self.use_spatial_rain_cn_chk.setChecked(True)

        self.rain_update_interval_spin = QtWidgets.QSpinBox()
        self.rain_update_interval_spin.setObjectName("rain_update_interval_spin")
        self.rain_update_interval_spin.setToolTip(
            "Interval in seconds for re-evaluating the SCS-CN runoff rate. "
            "Default 60s. Lower values=more responsive but more compute. "
            "Set to 0 to re-evaluate every step (old behavior)."
        )
        self._add_param_row(form, "Rain rate update interval (s):", self.rain_update_interval_spin)
        self.rain_update_interval_spin.setRange(0, 3600)
        self.rain_update_interval_spin.setSingleStep(10)
        self.rain_update_interval_spin.setValue(60)

        self.storm_area_layer_combo = QtWidgets.QComboBox()
        self.storm_area_layer_combo.setObjectName("storm_area_layer_combo")
        self.storm_area_layer_combo.setToolTip(
            "Optional polygon layer defining the storm area extent. "
            "Only cells within storm area polygons receive rainfall. "
            "Select '(none)' to apply rain globally."
        )
        self._add_param_row(form, "Storm area layer (optional):", self.storm_area_layer_combo)
        self.storm_area_layer_combo.addItem("(none)", None)

        self.rain_boundary_buffer_rings_spin = QtWidgets.QSpinBox()
        self.rain_boundary_buffer_rings_spin.setObjectName(
            "rain_boundary_buffer_rings_spin"
        )
        self.rain_boundary_buffer_rings_spin.setToolTip(
            "Number of boundary buffer rings to which rainfall is still applied when "
            "using a storm area layer. Prevents dry boundary artifacts. Range: 0–10."
        )
        self._add_param_row(
            form, "Rain boundary buffer rings:", self.rain_boundary_buffer_rings_spin
        )
        self.rain_boundary_buffer_rings_spin.setRange(0, 10)
        self.rain_boundary_buffer_rings_spin.setValue(1)

        form = self._start_param_group(param_form, "Infiltration")
        self.infiltration_method_combo = QtWidgets.QComboBox()
        self.infiltration_method_combo.setObjectName("infiltration_method_combo")
        self.infiltration_method_combo.setToolTip(
            "Infiltration model for rainfall-runoff computation. "
            "'SCS Curve Number' uses the NRCS runoff curve number method. "
            "'None' skips infiltration entirely."
        )
        self._add_param_row(form, "Infiltration method:", self.infiltration_method_combo)
        self.infiltration_method_combo.addItem("SCS Curve Number", "scs_cn")
        self.infiltration_method_combo.addItem("None (no infiltration)", "none")
        self.infiltration_method_combo.setCurrentIndex(
            self.infiltration_method_combo.findData("scs_cn")
        )
        # CN polygons layer — moved from the Layers page
        self.cn_layer_combo.setToolTip(
            "Polygon layer containing SCS Curve Number values for runoff computation. "
            "Required when infiltration method is SCS Curve Number."
        )
        self._add_param_row(form, "CN polygons:", self.cn_layer_combo)

        self.cn_default_spin = QtWidgets.QDoubleSpinBox()
        self.cn_default_spin.setObjectName("cn_default_spin")
        self.cn_default_spin.setToolTip(
            "Default SCS Curve Number for runoff computation. "
            "CN values range from 1 (high infiltration) to 100 (impervious). "
            "Typical: 75 (residential), 85 (urban). Overridden by CN polygon layer if present."
        )
        self._add_param_row(form, "Default CN:", self.cn_default_spin)
        self.cn_default_spin.setRange(1.0, 100.0)
        self.cn_default_spin.setDecimals(1)
        self.cn_default_spin.setValue(75.0)

        self.ia_ratio_spin = QtWidgets.QDoubleSpinBox()
        self.ia_ratio_spin.setObjectName("ia_ratio_spin")
        self.ia_ratio_spin.setToolTip(
            "SCS initial abstraction ratio Ia/S. Standard SCS value is 0.20. "
            "Lower values (e.g. 0.05) reduce initial abstraction and increase runoff. "
            "Range: 0–1.0."
        )
        self._add_param_row(form, "SCS Ia/S ratio:", self.ia_ratio_spin)
        self.ia_ratio_spin.setRange(0.0, 1.0)
        self.ia_ratio_spin.setDecimals(3)
        self.ia_ratio_spin.setSingleStep(0.01)
        self.ia_ratio_spin.setValue(0.2)

        form = self._start_param_group(param_form, "Source Stability", advanced=True)
        self.max_rel_depth_increase_spin = QtWidgets.QDoubleSpinBox()
        self.max_rel_depth_increase_spin.setObjectName("max_rel_depth_increase_spin")
        self.max_rel_depth_increase_spin.setToolTip(
            "Maximum relative water depth increase per timestep due to source terms. "
            "0 = unlimited. Typical: 2.0 (2x depth per step). Range: 0–1000."
        )
        self._add_param_row(
            form, "Max rel depth increase:", self.max_rel_depth_increase_spin, advanced=True
        )
        self.max_rel_depth_increase_spin.setRange(0.0, 1000.0)
        self.max_rel_depth_increase_spin.setDecimals(3)
        self.max_rel_depth_increase_spin.setValue(2.0)

        self.max_source_depth_step_spin = QtWidgets.QDoubleSpinBox()
        self.max_source_depth_step_spin.setObjectName("max_source_depth_step_spin")
        self.max_source_depth_step_spin.setToolTip(
            "Maximum absolute water depth change per step from source terms (model units). "
            "0 = unlimited. Use to prevent numerical blowup from intense rainfall."
        )
        self._add_param_row(form, "Max source dh/step:", self.max_source_depth_step_spin, advanced=True)
        self.max_source_depth_step_spin.setRange(0.0, 10.0)
        self.max_source_depth_step_spin.setDecimals(6)
        self.max_source_depth_step_spin.setValue(0.0)

        self.max_source_rate_spin = QtWidgets.QDoubleSpinBox()
        self.max_source_rate_spin.setObjectName("max_source_rate_spin")
        self.max_source_rate_spin.setToolTip(
            "Maximum source rate (rainfall intensity) threshold in model "
            "units. Values above this cap are clamped. 0 = no cap. Range: 0–100."
        )
        self._add_param_row(form, "Max source rate:", self.max_source_rate_spin, advanced=True)
        self.max_source_rate_spin.setRange(0.0, 100.0)
        self.max_source_rate_spin.setDecimals(6)
        self.max_source_rate_spin.setValue(0.0)

        self.extreme_rain_mode_chk = QtWidgets.QCheckBox("Enable")
        self.extreme_rain_mode_chk.setObjectName("extreme_rain_mode_chk")
        self.extreme_rain_mode_chk.setToolTip(
            "Enables extreme rainfall handling with enhanced source term "
            "stabilization. Use for high-intensity storms where standard "
            "source treatment may become unstable."
        )
        self._add_param_row(form, "Extreme rain mode:", self.extreme_rain_mode_chk, advanced=True)
        self.extreme_rain_mode_chk.setChecked(False)

        self.source_cfl_beta_spin = QtWidgets.QDoubleSpinBox()
        self.source_cfl_beta_spin.setObjectName("source_cfl_beta_spin")
        self.source_cfl_beta_spin.setToolTip(
            "CFL beta factor for source term sub-stepping. "
            "Lower values → smaller source substeps → more stability. "
            "Range: 0.01–2.0. Default: 0.25."
        )
        self._add_param_row(form, "Source CFL beta:", self.source_cfl_beta_spin, advanced=True)
        self.source_cfl_beta_spin.setRange(0.01, 2.0)
        self.source_cfl_beta_spin.setDecimals(3)
        self.source_cfl_beta_spin.setSingleStep(0.05)
        self.source_cfl_beta_spin.setValue(0.25)

        self.source_max_substeps_spin = QtWidgets.QSpinBox()
        self.source_max_substeps_spin.setObjectName("source_max_substeps_spin")
        self.source_max_substeps_spin.setToolTip(
            "Maximum number of substeps for source term integration. "
            "Higher values allow finer source sub-cycling for stability. "
            "Range: 1–512. Default: 16."
        )
        self._add_param_row(form, "Source max substeps:", self.source_max_substeps_spin, advanced=True)
        self.source_max_substeps_spin.setRange(1, 512)
        self.source_max_substeps_spin.setValue(16)

        self.source_true_subcycling_chk = QtWidgets.QCheckBox("Enable")
        self.source_true_subcycling_chk.setObjectName("source_true_subcycling_chk")
        self.source_true_subcycling_chk.setToolTip(
            "When enabled, source terms are integrated with true sub-cycling "
            "(multiple substeps per hydrodynamic step). When disabled, sources "
            "are integrated with the main timestep."
        )
        self._add_param_row(form, "True source subcycling:", self.source_true_subcycling_chk, advanced=True)
        self.source_true_subcycling_chk.setChecked(False)

        self.source_imex_split_chk = QtWidgets.QCheckBox("Enable")
        self.source_imex_split_chk.setObjectName("source_imex_split_chk")
        self.source_imex_split_chk.setToolTip(
            "Split source terms into Implicit-Explicit (IMEX) components. "
            "Stiff source terms (e.g. friction) are treated implicitly, "
            "non-stiff terms (e.g. rainfall) explicitly."
        )
        self._add_param_row(form, "IMEX source split:", self.source_imex_split_chk, advanced=True)
        self.source_imex_split_chk.setChecked(False)

    def _build_stability_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Stability Controls page with grouped damping/cap controls."""
        form = self._start_param_group(param_form, "Wet/Dry Front")
        self.shallow_damping_depth_spin = QtWidgets.QDoubleSpinBox()
        self.shallow_damping_depth_spin.setObjectName("shallow_damping_depth_spin")
        self.shallow_damping_depth_spin.setToolTip(
            "Depth threshold below which velocity damping is applied to "
            "stabilize wetting/drying fronts. Range: 1e-8–10. Default: 1e-4."
        )
        self._add_param_row(form, "Shallow damping depth:", self.shallow_damping_depth_spin)
        self.shallow_damping_depth_spin.setRange(1.0e-8, 10.0)
        self.shallow_damping_depth_spin.setDecimals(6)
        self.shallow_damping_depth_spin.setValue(1.0e-4)

        self.shallow_front_recon_fallback_chk = QtWidgets.QCheckBox("Enable")
        self.shallow_front_recon_fallback_chk.setObjectName(
            "shallow_front_recon_fallback_chk"
        )
        self.shallow_front_recon_fallback_chk.setToolTip(
            "When enabled, falls back to first-order reconstruction at "
            "shallow wet/dry fronts to prevent overshoot. Recommended: enabled."
        )
        self._add_param_row(
            form, "Shallow-front recon fallback:", self.shallow_front_recon_fallback_chk
        )
        self.shallow_front_recon_fallback_chk.setChecked(True)

        self.front_flux_damping_spin = QtWidgets.QDoubleSpinBox()
        self.front_flux_damping_spin.setObjectName("front_flux_damping_spin")
        self.front_flux_damping_spin.setToolTip(
            "Damping factor applied to fluxes at wet/dry fronts (0–1). "
            "Higher values = more damping = more stability at front. "
            "Default: 0.5. Increase if front oscillations occur."
        )
        self._add_param_row(form, "Front flux damping:", self.front_flux_damping_spin)
        self.front_flux_damping_spin.setRange(0.0, 1.0)
        self.front_flux_damping_spin.setDecimals(2)
        self.front_flux_damping_spin.setSingleStep(0.05)
        self.front_flux_damping_spin.setValue(0.5)

        self.active_set_hysteresis_chk = QtWidgets.QCheckBox("Enable")
        self.active_set_hysteresis_chk.setObjectName("active_set_hysteresis_chk")
        self.active_set_hysteresis_chk.setToolTip(
            "Enable hysteresis in wet/dry cell state transitions. "
            "Prevents cells from flipping between wet/dry every timestep, "
            "improving stability near the dynamic front. Recommended: enabled."
        )
        self._add_param_row(form, "Active-set hysteresis:", self.active_set_hysteresis_chk)
        self.active_set_hysteresis_chk.setChecked(True)

        form = self._start_param_group(param_form, "Capping", advanced=True)
        self.depth_cap_spin = QtWidgets.QDoubleSpinBox()
        self.depth_cap_spin.setObjectName("depth_cap_spin")
        self.depth_cap_spin.setToolTip(
            "Maximum allowable water depth (model units). "
            "Depths exceeding this cap are clamped. "
            "Range: 0.001–1e7. Default: 1e6 (effectively unlimited)."
        )
        self._add_param_row(form, "Depth cap:", self.depth_cap_spin, advanced=True)
        self.depth_cap_spin.setRange(0.001, 1.0e7)
        self.depth_cap_spin.setDecimals(3)
        self.depth_cap_spin.setValue(1.0e6)

        self.momentum_cap_min_speed_spin = QtWidgets.QDoubleSpinBox()
        self.momentum_cap_min_speed_spin.setObjectName("momentum_cap_min_speed_spin")
        self.momentum_cap_min_speed_spin.setToolTip(
            "Minimum flow speed below which momentum capping is inactive. "
            "Prevents capping from affecting low-velocity regions. "
            "Range: 0.1–1e4. Default: 50."
        )
        self._add_param_row(
            form, "Momentum cap min speed:", self.momentum_cap_min_speed_spin, advanced=True
        )
        self.momentum_cap_min_speed_spin.setRange(0.1, 1.0e4)
        self.momentum_cap_min_speed_spin.setDecimals(3)
        self.momentum_cap_min_speed_spin.setValue(50.0)

        self.momentum_cap_celerity_mult_spin = QtWidgets.QDoubleSpinBox()
        self.momentum_cap_celerity_mult_spin.setObjectName(
            "momentum_cap_celerity_mult_spin"
        )
        self.momentum_cap_celerity_mult_spin.setToolTip(
            "Multiplier on wave celerity to determine the momentum cap. "
            "Momentum = min(raw, celerity × mult). "
            "Range: 0.1–1000. Default: 20."
        )
        self._add_param_row(
            form, "Momentum cap celerity mult:", self.momentum_cap_celerity_mult_spin, advanced=True
        )
        self.momentum_cap_celerity_mult_spin.setRange(0.1, 1000.0)
        self.momentum_cap_celerity_mult_spin.setDecimals(3)
        self.momentum_cap_celerity_mult_spin.setValue(20.0)

        form = self._start_param_group(param_form, "Solver Safety", advanced=True)
        self.degen_mode_combo = QtWidgets.QComboBox()
        self.degen_mode_combo.setObjectName("degen_mode_combo")
        self.degen_mode_combo.setToolTip(
            "Mode for handling degenerate cells (area ≤ 0 or invalid topology). "
            "Off (0): treat normally, may cause instability. "
            "Skip (1): permanently exclude degenerate cells (fastest). "
            "Repair (2): replace inv_area with neighbor-average (most robust). "
            "Merge (3): redirect degenerate cell flux to owner cell."
        )
        self._add_param_row(form, "Degenerate cell mode:", self.degen_mode_combo)
        self.degen_mode_combo.addItem("Off (0)", 0)
        self.degen_mode_combo.addItem("Skip (1)", 1)
        self.degen_mode_combo.addItem("Repair (2)", 2)
        self.degen_mode_combo.addItem("Merge (3)", 3)
        self.degen_mode_combo.setCurrentIndex(self.degen_mode_combo.findData(0))

        self.max_inv_area_spin = QtWidgets.QDoubleSpinBox()
        self.max_inv_area_spin.setObjectName("max_inv_area_spin")
        self.max_inv_area_spin.setToolTip(
            "Maximum cell area for determining cell inversion risk. "
            "Large cells with steep water surface gradients may trigger "
            "inverted cell detection. Range: 1–1e12. Default: 1e6."
        )
        self._add_param_row(form, "Max inv area:", self.max_inv_area_spin, advanced=True)
        self.max_inv_area_spin.setRange(1.0, 1.0e12)
        self.max_inv_area_spin.setDecimals(1)
        self.max_inv_area_spin.setValue(1.0e6)

        self.cfl_lambda_cap_spin = QtWidgets.QDoubleSpinBox()
        self.cfl_lambda_cap_spin.setObjectName("cfl_lambda_cap_spin")
        self.cfl_lambda_cap_spin.setToolTip(
            "Maximum eigenvalue (wave speed) used in CFL timestep calculation. "
            "Caps the celerity term to prevent extremely small timesteps "
            "from anomalously high wave speeds. Range: 1–1e12. Default: 1e6."
        )
        self._add_param_row(form, "CFL lambda cap:", self.cfl_lambda_cap_spin, advanced=True)
        self.cfl_lambda_cap_spin.setRange(1.0, 1.0e12)
        self.cfl_lambda_cap_spin.setDecimals(1)
        self.cfl_lambda_cap_spin.setValue(1.0e6)

    def _build_drain_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Structures & Drainage page with grouped coupling controls."""
        # Layer Setup (drainage + structures layers — relocated from removed Layers page)
        for attr in [
            "drain_nodes_layer_combo",
            "drain_links_layer_combo",
            "drain_inlets_layer_combo",
            "drain_node_inlets_layer_combo",
            "structures_layer_combo",
        ]:
            widget = QtWidgets.QComboBox()
            widget.setObjectName(attr)
            setattr(self, attr, widget)
        form = self._start_param_group(param_form, "Layer Setup")
        self.drain_nodes_layer_combo.setToolTip(
            "Point layer for drainage network nodes (manholes, junctions). "
            "Used for 1D-2D coupled drainage simulations."
        )
        self._add_param_row(form, "Drainage nodes layer:", self.drain_nodes_layer_combo)
        self.drain_links_layer_combo.setToolTip(
            "Line layer for drainage network links (pipes, channels). "
            "Connects drain nodes for 1D-2D coupled drainage."
        )
        self._add_param_row(form, "Drainage links layer:", self.drain_links_layer_combo)
        self.drain_inlets_layer_combo.setToolTip(
            "Table layer defining inlet types (grate, curb, combination) "
            "and their hydraulic capture curves."
        )
        self._add_param_row(form, "Drainage inlet types (table):", self.drain_inlets_layer_combo)
        self.drain_node_inlets_layer_combo.setToolTip(
            "Table layer mapping drain nodes to inlet types from the inlet types table. "
            "Defines which inlets are connected to which nodes."
        )
        self._add_param_row(form, "Drainage node-inlets (table):", self.drain_node_inlets_layer_combo)
        self.structures_layer_combo.setToolTip(
            "Line layer for hydraulic structures (weirs, orifices, bridges, culverts, pumps). "
            "Each structure must have a type field and geometry."
        )
        self._add_param_row(form, "Hydraulic structures layer:", self.structures_layer_combo)

        form = self._start_param_group(param_form, "Culvert / Bridge", advanced=True)
        self.coupling_loop_combo = QtWidgets.QComboBox()
        self.coupling_loop_combo.setObjectName("coupling_loop_combo")
        self.coupling_loop_combo.setToolTip(
            "Select the coupling backend for drainage/structure-2D interaction. "
            "'CUDA coupling loop (GPU)' runs the coupling solver on GPU."
        )
        self._add_param_row(form, "Coupling loop:", self.coupling_loop_combo)
        self.coupling_loop_combo.addItem("CUDA coupling loop (GPU)", "cuda")
        self.coupling_loop_combo.setCurrentIndex(0)

        self.culvert_solver_mode_combo = QtWidgets.QComboBox()
        self.culvert_solver_mode_combo.setObjectName("culvert_solver_mode_combo")
        self.culvert_solver_mode_combo.setToolTip(
            "Culvert hydraulics solver method. "
            "'Direct solver' uses Newton/secant iteration at each culvert face. "
            "'Precomputed lookup table' uses interpolated discharge from stored tables."
        )
        self._add_param_row(form, "Culvert solver mode:", self.culvert_solver_mode_combo, advanced=True)
        self.culvert_solver_mode_combo.addItem(
            "Direct culvert outlet solver (Newton/secant)", 0
        )
        self.culvert_solver_mode_combo.addItem(
            "Precomputed culvert lookup table", 1
        )
        self.culvert_solver_mode_combo.setCurrentIndex(0)

        self.culvert_face_flux_chk = QtWidgets.QCheckBox("Face-based flux (GPU only)")
        self.culvert_face_flux_chk.setObjectName("culvert_face_flux_chk")
        self.culvert_face_flux_chk.setToolTip(
            "Use face-based flux coupling for culverts on GPU. "
            "Distributes culvert discharge across the 2D cell face instead "
            "of the whole cell for better spatial resolution."
        )
        self._add_param_row(form, "Culvert coupling mode:", self.culvert_face_flux_chk, advanced=True)

        self.use_redistribution_chk = QtWidgets.QCheckBox("Enable redistribution override")
        self.use_redistribution_chk.setObjectName("use_redistribution_chk")
        self.use_redistribution_chk.setChecked(True)
        self.use_redistribution_chk.setToolTip("When checked, reads per-structure redistribution parameters from the GeoPackage. Uncheck to skip redistribution entirely.")
        self._add_param_row(form, self.use_redistribution_chk.text(), self.use_redistribution_chk, advanced=True)

        self.bridge_stacked_coupling_mode_combo = QtWidgets.QComboBox()
        self.bridge_stacked_coupling_mode_combo.setObjectName(
            "bridge_stacked_coupling_mode_combo"
        )
        self.bridge_stacked_coupling_mode_combo.setToolTip(
            "Spatial redistribution method for bridge stacked coupling. "
            "'Phase 3 spatial redistribution' distributes flow across multiple "
            "cells. 'Legacy scalar weighting' uses a single scalar factor."
        )
        self._add_param_row(
            form, "Bridge stacked coupling mode:", self.bridge_stacked_coupling_mode_combo, advanced=True
        )
        self.bridge_stacked_coupling_mode_combo.addItem(
            "Phase 3 — Spatial", "phase3_spatial"
        )
        self.bridge_stacked_coupling_mode_combo.addItem(
            "Legacy — Scalar", "legacy_scalar"
        )
        self.bridge_stacked_coupling_mode_combo.setCurrentIndex(0)

        form = self._start_param_group(param_form, "Drainage Network — Equation Set", advanced=True)
        self.drainage_solver_mode_combo = QtWidgets.QComboBox()
        self.drainage_solver_mode_combo.setObjectName("drainage_solver_mode_combo")
        self.drainage_solver_mode_combo.setToolTip(
            "Governing equations for 1D drainage network flow. "
            "'EGL' includes Bernoulli + minor losses (recommended). "
            "'Diffusion wave' simplifies to gravity + friction. "
            "'Dynamic Saint-Venant' solves the full 1D momentum equation."
        )
        self._add_param_row(form, "Drainage equation set:", self.drainage_solver_mode_combo, advanced=True)
        self.drainage_solver_mode_combo.addItem("EGL (Bernoulli + minor losses)", 0)
        self.drainage_solver_mode_combo.addItem("Diffusion wave", 1)
        self.drainage_solver_mode_combo.addItem("Dynamic Saint-Venant", 2)
        self.drainage_solver_mode_combo.setCurrentIndex(0)

        self.drainage_gpu_method_combo = QtWidgets.QComboBox()
        self.drainage_gpu_method_combo.setObjectName("drainage_gpu_method_combo")
        self.drainage_gpu_method_combo.setToolTip(
            "GPU execution strategy for drainage coupling. "
            "'Per-step GPU drainage' solves one SWE2D step per drainage substep. "
            "'Native iterative GPU drainage' batches multiple substeps on GPU."
        )
        self._add_param_row(form, "Drainage GPU method:", self.drainage_gpu_method_combo, advanced=True)
        self.drainage_gpu_method_combo.addItem(
            "Per-step GPU drainage (fast for sparse exchange)", "step"
        )
        self.drainage_gpu_method_combo.addItem(
            "Native iterative GPU drainage (batched substeps)", "iterative"
        )
        self.drainage_gpu_method_combo.setCurrentIndex(0)

        form = self._start_param_group(param_form, "Drainage — Substepping", advanced=True)
        self.drainage_coupling_substeps_spin = QtWidgets.QSpinBox()
        self.drainage_coupling_substeps_spin.setObjectName(
            "drainage_coupling_substeps_spin"
        )
        self.drainage_coupling_substeps_spin.setToolTip(
            "Number of drainage substeps per SWE2D timestep. "
            "Range: 1–256. Default: 1. Increase for stiffer drainage systems."
        )
        self._add_param_row(form, "Drainage substeps:", self.drainage_coupling_substeps_spin, advanced=True)
        self.drainage_coupling_substeps_spin.setRange(1, 256)
        self.drainage_coupling_substeps_spin.setValue(1)

        self.drainage_max_coupling_substeps_spin = QtWidgets.QSpinBox()
        self.drainage_max_coupling_substeps_spin.setObjectName(
            "drainage_max_coupling_substeps_spin"
        )
        self.drainage_max_coupling_substeps_spin.setToolTip(
            "Maximum number of adaptive substeps for drainage coupling. "
            "Range: 1–1024. Default: 64."
        )
        self._add_param_row(
            form, "Drainage max adaptive substeps:", self.drainage_max_coupling_substeps_spin, advanced=True
        )
        self.drainage_max_coupling_substeps_spin.setRange(1, 1024)
        self.drainage_max_coupling_substeps_spin.setValue(64)

        form = self._start_param_group(param_form, "Drainage — Stability", advanced=True)
        self.drainage_head_deadband_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_head_deadband_spin.setObjectName("drainage_head_deadband_spin")
        self.drainage_head_deadband_spin.setToolTip(
            "Head (water surface elevation difference) deadband below which "
            "no drainage flow is computed. Prevents oscillation near zero flow. "
            "Range: 0–10. Default: 0.001."
        )
        self._add_param_row(
            form, "Drainage head deadband:", self.drainage_head_deadband_spin, advanced=True
        )
        self.drainage_head_deadband_spin.setRange(0.0, 10.0)
        self.drainage_head_deadband_spin.setDecimals(6)
        self.drainage_head_deadband_spin.setValue(1.0e-3)

        self.drainage_dynamic_relaxation_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_dynamic_relaxation_spin.setObjectName(
            "drainage_dynamic_relaxation_spin"
        )
        self.drainage_dynamic_relaxation_spin.setToolTip(
            "Relaxation factor for the drainage coupling iteration (0–1). "
            "1.0 = no relaxation, 0.5 = 50% under-relaxation. "
            "Lower values improve stability for stiff coupling."
        )
        self._add_param_row(
            form, "Drainage dynamic relaxation:", self.drainage_dynamic_relaxation_spin, advanced=True
        )
        self.drainage_dynamic_relaxation_spin.setRange(0.0, 1.0)
        self.drainage_dynamic_relaxation_spin.setDecimals(3)
        self.drainage_dynamic_relaxation_spin.setSingleStep(0.05)
        self.drainage_dynamic_relaxation_spin.setValue(1.0)

        self.drainage_adaptive_depth_fraction_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_adaptive_depth_fraction_spin.setObjectName(
            "drainage_adaptive_depth_fraction_spin"
        )
        self.drainage_adaptive_depth_fraction_spin.setToolTip(
            "Fraction of cell water depth allowed to be drained per step "
            "when adaptive drainage is active. Range: 0.001–1.0. Default: 0.2."
        )
        self._add_param_row(
            form, "Drainage adaptive depth fraction:", self.drainage_adaptive_depth_fraction_spin, advanced=True
        )
        self.drainage_adaptive_depth_fraction_spin.setRange(0.001, 1.0)
        self.drainage_adaptive_depth_fraction_spin.setDecimals(3)
        self.drainage_adaptive_depth_fraction_spin.setSingleStep(0.01)
        self.drainage_adaptive_depth_fraction_spin.setValue(0.2)

        self.drainage_adaptive_wave_courant_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_adaptive_wave_courant_spin.setObjectName(
            "drainage_adaptive_wave_courant_spin"
        )
        self.drainage_adaptive_wave_courant_spin.setToolTip(
            "Courant number target for adaptive drainage timestep control. "
            "Range: 0.001–10.0. Default: 0.5."
        )
        self._add_param_row(
            form, "Drainage adaptive wave Courant:", self.drainage_adaptive_wave_courant_spin, advanced=True
        )
        self.drainage_adaptive_wave_courant_spin.setRange(0.001, 10.0)
        self.drainage_adaptive_wave_courant_spin.setDecimals(3)
        self.drainage_adaptive_wave_courant_spin.setSingleStep(0.05)
        self.drainage_adaptive_wave_courant_spin.setValue(0.5)

        self.drainage_implicit_iters_spin = QtWidgets.QSpinBox()
        self.drainage_implicit_iters_spin.setObjectName(
            "drainage_implicit_iters_spin"
        )
        self.drainage_implicit_iters_spin.setToolTip(
            "Number of implicit solver iterations for GPU drainage. "
            "Range: 1–8. Default: 2. More iterations improve convergence "
            "at the cost of performance."
        )
        self._add_param_row(
            form, "Drainage implicit iterations (GPU):", self.drainage_implicit_iters_spin, advanced=True
        )
        self.drainage_implicit_iters_spin.setRange(1, 8)
        self.drainage_implicit_iters_spin.setValue(2)

        self.drainage_implicit_relax_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_implicit_relax_spin.setObjectName(
            "drainage_implicit_relax_spin"
        )
        self.drainage_implicit_relax_spin.setToolTip(
            "Relaxation factor for implicit drainage solver on GPU (0.1–1.0). "
            "Default: 0.5. Lower values improve stability."
        )
        self._add_param_row(
            form, "Drainage implicit relaxation (GPU):", self.drainage_implicit_relax_spin, advanced=True
        )
        self.drainage_implicit_relax_spin.setRange(0.1, 1.0)
        self.drainage_implicit_relax_spin.setDecimals(2)
        self.drainage_implicit_relax_spin.setSingleStep(0.05)
        self.drainage_implicit_relax_spin.setValue(0.5)

        self.gpu_default_lbl = QtWidgets.QLabel(
            "GPU is attempted by default when supported by the native backend."
        )
        self.gpu_default_lbl.setObjectName("gpu_default_lbl")
        self.gpu_default_lbl.setWordWrap(True)
        param_form.addRow(self.gpu_default_lbl)

        self.unit_system_lbl = QtWidgets.QLabel("Unit system: (detecting \u2014 open a project or load layers)")
        self.unit_system_lbl.setObjectName("unit_system_lbl")
        self.unit_system_lbl.setWordWrap(True)
        param_form.addRow(self.unit_system_lbl)

    def get_h_min(self) -> float:
        """Minimum water depth threshold."""
        return float(self.h_min_spin.value())

    def get_rain_update_interval_s(self) -> int:
        """Rain rate update interval in seconds."""
        return int(self.rain_update_interval_spin.value())

    def get_run_time_hours(self) -> str:
        """Run duration text (decimal hours or HH:MM)."""
        return str(self.run_time_edit.text())

    def get_run_time_hours_parsed(self) -> float:
        """Run duration parsed as decimal hours."""
        from swe2d.workbench.services.text_parser_service import parse_time_hours
        return parse_time_hours(self.run_time_edit.text())

    def get_n_mann(self) -> float:
        """Manning n roughness coefficient."""
        return float(self.n_mann_spin.value())

    def get_ia_ratio(self) -> float:
        """SCS initial abstraction ratio."""
        return float(self.ia_ratio_spin.value())

    def get_infiltration_method(self) -> str:
        """Selected infiltration method key."""
        return str(self.infiltration_method_combo.currentData() or "scs_cn")

    def get_rain_boundary_buffer_rings(self) -> int:
        """Boundary buffer ring count for rain."""
        return int(self.rain_boundary_buffer_rings_spin.value())

    def get_cn_default(self) -> float:
        """Default SCS curve number."""
        return float(self.cn_default_spin.value())

    def get_drainage_solver_mode(self) -> int:
        """Drainage equation set integer key."""
        return int(self.drainage_solver_mode_combo.currentData())

    def get_drainage_gpu_method(self) -> str:
        """Drainage GPU method key."""
        return str(self.drainage_gpu_method_combo.currentData())

    def get_drainage_coupling_substeps(self) -> int:
        """Number of drainage substeps per SWE2D step."""
        return int(self.drainage_coupling_substeps_spin.value())

    def get_drainage_max_coupling_substeps(self) -> int:
        """Max adaptive substeps for drainage."""
        return int(self.drainage_max_coupling_substeps_spin.value())

    def get_drainage_head_deadband(self) -> float:
        """Head deadband below which no drainage flow."""
        return float(self.drainage_head_deadband_spin.value())

    def get_drainage_dynamic_relaxation(self) -> float:
        """Relaxation factor for drainage coupling."""
        return float(self.drainage_dynamic_relaxation_spin.value())

    def get_drainage_adaptive_depth_fraction(self) -> float:
        """Fraction of cell water depth drainable per step."""
        return float(self.drainage_adaptive_depth_fraction_spin.value())

    def get_drainage_adaptive_wave_courant(self) -> float:
        """Courant target for adaptive drainage."""
        return float(self.drainage_adaptive_wave_courant_spin.value())

    def get_drainage_implicit_iters(self) -> int:
        """Implicit solver iterations for GPU drainage."""
        return int(self.drainage_implicit_iters_spin.value())

    def get_drainage_implicit_relax(self) -> float:
        """Relaxation factor for implicit drainage on GPU."""
        return float(self.drainage_implicit_relax_spin.value())

    def is_inflow_progressive(self) -> bool:
        """Inflow progressive activation checkbox."""
        return bool(self.inflow_progressive_chk.isChecked())

    def get_inflow_progressive_chk(self):
        """Inflow progressive checkbox widget."""
        return self.inflow_progressive_chk

    def get_default_bc_type(self) -> int:
        """Default boundary condition type code."""
        return int(self.default_bc_type_combo.currentData())

    def collect_params(self) -> dict:
        """Return all model parameter values as a flat dict.

        Co-located with the widgets so widget moves only touch this method.
        """
        try:
            self.gpu_diag_sync_interval_spin.interpretText()
        except Exception:
            pass
        return {
            "n_mann_spin": float(self.n_mann_spin.value()),
            "cfl_spin": float(self.cfl_spin.value()),
            "h_min_spin": float(self.h_min_spin.value()),
            "dt_spin": float(self.dt_spin.value()),
            "initial_dt_spin": float(self.initial_dt_spin.value()),
            "adaptive_cfl_dt_chk": bool(self.adaptive_cfl_dt_chk.isChecked()),
            "reconstruction_combo": int(self.reconstruction_combo.currentData()),
            "reconstruction_combo_text": str(self.reconstruction_combo.currentText()).strip(),
            "temporal_order_combo": int(self.temporal_order_combo.currentData()),
            "temporal_order_combo_text": str(self.temporal_order_combo.currentText()).strip(),
            "cfl_lambda_cap_spin": float(self.cfl_lambda_cap_spin.value()),
            "gpu_diag_sync_interval_spin": int(self.gpu_diag_sync_interval_spin.value()),
            "max_rel_depth_increase_spin": float(self.max_rel_depth_increase_spin.value()),
            "max_source_depth_step_spin": float(self.max_source_depth_step_spin.value()),
            "max_source_rate_spin": float(self.max_source_rate_spin.value()),
            "extreme_rain_mode_chk": bool(self.extreme_rain_mode_chk.isChecked()),
            "source_cfl_beta_spin": float(self.source_cfl_beta_spin.value()),
            "source_max_substeps_spin": int(self.source_max_substeps_spin.value()),
            "source_true_subcycling_chk": bool(self.source_true_subcycling_chk.isChecked()),
            "source_imex_split_chk": bool(self.source_imex_split_chk.isChecked()),
            "shallow_damping_depth_spin": float(self.shallow_damping_depth_spin.value()),
            "depth_cap_spin": float(self.depth_cap_spin.value()),
            "momentum_cap_min_speed_spin": float(self.momentum_cap_min_speed_spin.value()),
            "momentum_cap_celerity_mult_spin": float(self.momentum_cap_celerity_mult_spin.value()),
            "max_inv_area_spin": float(self.max_inv_area_spin.value()),
            "rain_rate_spin": float(self.rain_rate_spin.value()),
            "run_time_edit": str(self.run_time_edit.text()),
            "tiny_mode_combo": int(self.tiny_mode_combo.currentData()),
            "tiny_wet_cell_threshold_spin": int(self.tiny_wet_cell_threshold_spin.value()),
            "use_redistribution_chk": bool(self.use_redistribution_chk.isChecked()),
            "gpu_diag_sync_interval_raw": int(self.gpu_diag_sync_interval_spin.value()),
            "swe2d_perf_mode_chk": bool(self.swe2d_perf_mode_chk.isChecked()),
            "culvert_face_flux_chk": bool(self.culvert_face_flux_chk.isChecked()),
            "enable_cuda_graphs_chk": bool(self.enable_cuda_graphs_chk.isChecked()),
            "degen_mode": int(self.degen_mode_combo.currentData()),
            "front_flux_damping_spin": float(self.front_flux_damping_spin.value()),
            "active_set_hysteresis_chk": bool(self.active_set_hysteresis_chk.isChecked()),
            "drainage_gpu_method": str(self.drainage_gpu_method_combo.currentData()),
            "culvert_solver_mode": int(self.culvert_solver_mode_combo.currentData()),
            "bridge_coupling_mode": str(self.bridge_stacked_coupling_mode_combo.currentData()),
        }

    def _on_select_results_gpkg(self) -> None:
        """Open a file dialog and populate the results GeoPackage path."""
        from qgis.PyQt.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Results GeoPackage", "",
            "GeoPackage (*.gpkg);;All files (*)",
        )
        if path:
            self.results_gpkg_path_edit.setText(path)
