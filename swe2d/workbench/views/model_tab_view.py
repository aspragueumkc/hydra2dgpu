"""Model tab view — owns its own widget references.

QWidget subclass for the Model tab in the Studio workbench.
Holds four pages inside a QToolBox:
  - model_solver_page (core + solver controls)
  - model_rain_page (rain / hydrology controls)
  - model_drain_page (structures & drainage controls)
  - model_run_page (run / output controls)

Each form widget is owned as an instance attribute with a stable
objectName so existing binding code (e.g. ``findChild``) keeps working.
"""
from __future__ import annotations

from qgis.PyQt import QtWidgets


class ModelTabView(QtWidgets.QWidget):
    """View for the Model tab.

    Houses five QToolBox pages.  Every widget is created here as a direct
    instance attribute with a stable ``objectName``.

    Solver Parameters page (``model_solver_page``):
        n_mann_spin, cfl_spin, h_min_spin,
        initial_condition_combo, initial_depth_spin, initial_wse_spin,
        adaptive_cfl_dt_chk, dt_spin, initial_dt_spin,
        gpu_diag_sync_interval_spin,
        tiny_mode_combo, tiny_wet_cell_threshold_spin,
        enable_cuda_graphs_chk, swe2d_perf_mode_chk,
        internal_flow_layer_combo, internal_flow_field_edit,
        run_time_edit, reconstruction_combo, temporal_order_combo

    Rain / Hydrology page (``model_rain_page``):
        max_rel_depth_increase_spin, max_source_depth_step_spin,
        max_source_rate_spin, extreme_rain_mode_chk,
        source_cfl_beta_spin, source_max_substeps_spin,
        source_true_subcycling_chk, source_imex_split_chk,
        source_stage_coupled_imex_rk2_chk,
        rain_rate_spin, cn_default_spin, ia_ratio_spin,
        use_spatial_rain_cn_chk, infiltration_method_combo,
        storm_area_layer_combo, rain_boundary_buffer_rings_spin

    Stability Controls page (``model_stability_page``):
        shallow_damping_depth_spin, shallow_front_recon_fallback_chk,
        front_flux_damping_spin, active_set_hysteresis_chk,
        depth_cap_spin, momentum_cap_min_speed_spin,
        momentum_cap_celerity_mult_spin, max_inv_area_spin,
        cfl_lambda_cap_spin

    Structures & Drainage page (``model_drain_page``):
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

    Run / Output page (``model_run_page``):
        run_btn, cancel_btn, progress_bar,
        output_interval_edit, line_output_interval_edit,
        preview_overrides_btn, preview_coupling_btn, snapshot_btn,
        results_table_name_edit, results_gpkg_path_edit,
        select_results_gpkg_btn, load_run_settings_btn
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the toolbox with all five parameter pages."""
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

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

        self.model_run_page, self.run_layout = self._build_run_page()
        self._build_run_page_widgets(self.model_run_page, self.run_layout)

        self.model_toolbox.addItem(self.model_solver_page, "Solver Parameters")
        self.model_toolbox.addItem(self.model_rain_page, "Rain / Hydrology")
        self.model_toolbox.addItem(self.model_stability_page, "Stability Controls")
        self.model_toolbox.addItem(self.model_drain_page, "Structures & Drainage")
        self.model_toolbox.addItem(self.model_run_page, "Run / Output")

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

    def _build_run_page(self) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        """Create the Run/Output toolbox page."""
        run_page = QtWidgets.QWidget()
        run_page.setObjectName("model_run_page")
        run_page_layout = QtWidgets.QVBoxLayout(run_page)
        run_page_layout.setObjectName("run_layout")
        run_page_layout.setContentsMargins(0, 0, 0, 0)
        return run_page, run_page_layout

    def _build_run_page_widgets(
        self, run_page: QtWidgets.QWidget, run_page_layout: QtWidgets.QVBoxLayout
    ) -> None:
        """Populate the Run page with buttons, progress bar, and I/O controls."""
        run_row = QtWidgets.QHBoxLayout()
        run_row.setObjectName("run_row_layout")
        for attr, text, tip in [
            ("run_btn", "Run 2D Model",
             "Start the 2D shallow water simulation with current settings."),
            ("batch_sim_btn", "Batch Simulation...",
             "Open batch simulation dialog for parameter sweeps."),
            ("cancel_btn", "Cancel",
             "Request cancellation of the running simulation. "
             "The solver will stop at the next safe checkpoint."),
        ]:
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName(attr)
            btn.setToolTip(tip)
            setattr(self, attr, btn)
            run_row.addWidget(btn)
        run_page_layout.addLayout(run_row)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setToolTip(
            "Simulation progress indicator. Shows percentage complete "
            "and current timestep information during model execution."
        )
        self.progress_bar.setValue(0)
        run_page_layout.addWidget(self.progress_bar)

        snap_row = QtWidgets.QHBoxLayout()
        snap_row.setObjectName("run_snapshot_row_layout")
        self.output_interval_edit = QtWidgets.QLineEdit("00:30")
        self.output_interval_edit.setObjectName("output_interval_edit")
        self.output_interval_edit.setToolTip(
            "Time interval between 2D mesh result output writes. "
            "Format: decimal hours (e.g. 0.5) or HH:MM (e.g. 00:30). "
            "Smaller intervals produce larger result files."
        )
        self.line_output_interval_edit = QtWidgets.QLineEdit("00:05")
        self.line_output_interval_edit.setObjectName("line_output_interval_edit")
        self.line_output_interval_edit.setToolTip(
            "Time interval between sample-line (cross-section) result outputs. "
            "Format: decimal hours or HH:MM. Default: 00:05 (5 min)."
        )
        snap_row.addWidget(QtWidgets.QLabel("Output interval (hr or HH:MM):"))
        snap_row.addWidget(self.output_interval_edit)
        snap_row.addWidget(QtWidgets.QLabel("Line output interval:"))
        snap_row.addWidget(self.line_output_interval_edit)
        run_page_layout.addLayout(snap_row)

        debug_sep = QtWidgets.QLabel("<b>Debugging</b>")
        run_page_layout.addWidget(debug_sep)

        debug_row = QtWidgets.QHBoxLayout()
        debug_row.setObjectName("run_debug_row_layout")
        for attr, text, tip in [
            ("preview_overrides_btn", "Preview Overrides",
             "Display a summary of all current parameter overrides "
             "before running the simulation."),
            ("preview_coupling_btn", "Preview Drainage/Structure Coupling",
             "Preview the 1D-2D coupling configuration for drainage "
             "and hydraulic structures before running."),
            ("snapshot_btn", "Fetch Device Results",
             "Save the current model state snapshot during a running simulation. "
             "Useful for debugging transient behavior."),
        ]:
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName(attr)
            btn.setToolTip(tip)
            setattr(self, attr, btn)
            debug_row.addWidget(btn)
        run_page_layout.addLayout(debug_row)

        load_row = QtWidgets.QHBoxLayout()
        load_row.setObjectName("load_results_row_layout")
        self.results_table_name_edit = QtWidgets.QLineEdit()
        self.results_table_name_edit.setObjectName("results_table_name_edit")
        self.results_table_name_edit.setToolTip(
            "Optional prefix for GeoPackage result table names. "
            "Useful when storing multiple model runs in the same GeoPackage."
        )
        self.results_table_name_edit.setPlaceholderText("optional table prefix")
        load_row.addWidget(QtWidgets.QLabel("Table prefix:"))
        load_row.addWidget(self.results_table_name_edit)
        self.results_gpkg_path_edit = QtWidgets.QLineEdit()
        self.results_gpkg_path_edit.setObjectName("results_gpkg_path_edit")
        self.results_gpkg_path_edit.setToolTip(
            "Path to the output GeoPackage for storing simulation results. "
            "Leave empty to use the model GeoPackage."
        )
        self.results_gpkg_path_edit.setPlaceholderText("GeoPackage path (optional)")
        load_row.addWidget(QtWidgets.QLabel("GPKG:"))
        load_row.addWidget(self.results_gpkg_path_edit)
        self.select_results_gpkg_btn = QtWidgets.QPushButton("Browse...")
        self.select_results_gpkg_btn.setObjectName("select_results_gpkg_btn")
        self.select_results_gpkg_btn.setToolTip(
            "Browse for an existing GeoPackage to store/load simulation results."
        )
        self.select_results_gpkg_btn.clicked.connect(self._on_select_results_gpkg)
        load_row.addWidget(self.select_results_gpkg_btn)
        self.load_run_settings_btn = QtWidgets.QPushButton("Load Model Config from GPKG...")
        self.load_run_settings_btn.setObjectName("load_run_settings_btn")
        self.load_run_settings_btn.setToolTip(
            "Open a GeoPackage and restore a saved simulation configuration "
            "(all widget values, solver params, and layer references)."
        )
        load_row.addWidget(self.load_run_settings_btn)
        self.save_settings_btn = QtWidgets.QPushButton("Save Config to GPKG...")
        self.save_settings_btn.setObjectName("save_settings_btn")
        self.save_settings_btn.setToolTip(
            "Save the current widget configuration to the active GeoPackage "
            "so it can be restored later via Load Config."
        )
        load_row.addWidget(self.save_settings_btn)
        load_row.addStretch(1)
        run_page_layout.addLayout(load_row)

        run_page_layout.addStretch(1)

    def _build_solver_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Solver Parameters page with numerical controls."""
        self.n_mann_spin = QtWidgets.QDoubleSpinBox()
        self.n_mann_spin.setObjectName("n_mann_spin")
        self.n_mann_spin.setToolTip(
            "Manning's roughness coefficient n. Typical values: 0.012 (concrete), "
            "0.020 (gravel), 0.035 (natural stream), 0.050 (floodplain). "
            "Range: 0.0–1.0."
        )
        param_form.addRow("Manning n:", self.n_mann_spin)
        self.n_mann_spin.setRange(0.0, 1.0)
        self.n_mann_spin.setDecimals(5)
        self.n_mann_spin.setValue(0.020)

        self.cfl_spin = QtWidgets.QDoubleSpinBox()
        self.cfl_spin.setObjectName("cfl_spin")
        self.cfl_spin.setToolTip(
            "Courant-Friedrichs-Lewy number for explicit timestep control. "
            "Typical range: 0.1–0.8. Lower values improve stability at the "
            "cost of smaller timesteps. Must be < 1.0 for explicit schemes."
        )
        param_form.addRow("CFL:", self.cfl_spin)
        self.cfl_spin.setRange(0.01, 0.99)
        self.cfl_spin.setDecimals(3)
        self.cfl_spin.setValue(0.45)

        self.h_min_spin = QtWidgets.QDoubleSpinBox()
        self.h_min_spin.setObjectName("h_min_spin")
        self.h_min_spin.setToolTip(
            "Minimum water depth threshold (model units). Cells with depth "
            "below h_min are treated as dry and excluded from momentum computation. "
            "Typical: 1e-6 (SI) or 1e-5 (USC). Increase for noisy terrain."
        )
        param_form.addRow("h_min:", self.h_min_spin)
        self.h_min_spin.setRange(1.0e-9, 1.0)
        self.h_min_spin.setDecimals(8)
        self.h_min_spin.setValue(1.0e-6)

        self.initial_condition_combo = QtWidgets.QComboBox()
        self.initial_condition_combo.setObjectName("initial_condition_combo")
        self.initial_condition_combo.setToolTip(
            "Initial condition for the entire domain. 'Dry start' sets depth=0 "
            "everywhere. 'Uniform depth' uses a constant depth. 'Uniform WSE' "
            "sets a constant water surface elevation (depth = WSE - bed)."
        )
        param_form.addRow("Initial condition:", self.initial_condition_combo)
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
        param_form.addRow("Initial depth:", self.initial_depth_spin)
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
        param_form.addRow("Initial WSE:", self.initial_wse_spin)
        self.initial_wse_spin.setRange(-1.0e6, 1.0e6)
        self.initial_wse_spin.setDecimals(4)
        self.initial_wse_spin.setValue(0.0)

        self.adaptive_cfl_dt_chk = QtWidgets.QCheckBox("Enable variable timestep (CFL)")
        self.adaptive_cfl_dt_chk.setObjectName("adaptive_cfl_dt_chk")
        self.adaptive_cfl_dt_chk.setToolTip(
            "When checked, dt is computed adaptively each step based on the CFL condition "
            "and current flow velocity. When unchecked, a fixed dt is used."
        )
        param_form.addRow("Variable timestep:", self.adaptive_cfl_dt_chk)
        self.adaptive_cfl_dt_chk.setChecked(False)

        self.dt_spin = QtWidgets.QDoubleSpinBox()
        self.dt_spin.setObjectName("dt_spin")
        self.dt_spin.setToolTip(
            "Fixed timestep when variable timestep is disabled. "
            "Acts as dt_max (upper bound) when variable timestep is enabled. "
            "Units: seconds or model time. Range: 0.0001–1e6."
        )
        param_form.addRow("dt (fixed or dt_max):", self.dt_spin)
        self.dt_spin.setRange(1.0e-4, 1.0e6)
        self.dt_spin.setDecimals(5)
        self.dt_spin.setValue(0.05)

        self.initial_dt_spin = QtWidgets.QDoubleSpinBox()
        self.initial_dt_spin.setObjectName("initial_dt_spin")
        self.initial_dt_spin.setToolTip(
            "Timestep used for the first step before the adaptive CFL adjusts. "
            "Set to 0 (default) for automatic selection based on mesh size and CFL."
        )
        param_form.addRow("Initial dt (0 = auto):", self.initial_dt_spin)
        self.initial_dt_spin.setRange(0.0, 1.0e6)
        self.initial_dt_spin.setDecimals(5)
        self.initial_dt_spin.setValue(0.0)

        self.gpu_diag_sync_interval_spin = QtWidgets.QSpinBox()
        self.gpu_diag_sync_interval_spin.setObjectName("gpu_diag_sync_interval_spin")
        self.gpu_diag_sync_interval_spin.setToolTip(
            "Number of solver steps between GPU diagnostics synchronization. "
            "Higher values reduce GPU/CPU sync overhead. Range: 1–1,000,000. "
            "Default: 10 steps."
        )
        param_form.addRow("GPU diag sync (steps):", self.gpu_diag_sync_interval_spin)
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
        param_form.addRow("Tiny mode:", self.tiny_mode_combo)
        self.tiny_mode_combo.addItem("Off (0)", 0)
        self.tiny_mode_combo.addItem("Auto (1)", 1)
        self.tiny_mode_combo.addItem("Fused (2)", 2)
        self.tiny_mode_combo.addItem("Persistent (3)", 3)
        self.tiny_mode_combo.setCurrentIndex(self.tiny_mode_combo.findData(3))

        self.tiny_wet_cell_threshold_spin = QtWidgets.QSpinBox()
        self.tiny_wet_cell_threshold_spin.setObjectName("tiny_wet_cell_threshold_spin")
        self.tiny_wet_cell_threshold_spin.setToolTip(
            "Maximum number of active/wet cells before tiny-mode optimization engages. "
            "Range: 1–10,000,000. Default: 2000."
        )
        param_form.addRow(
            "Tiny active/wet cell threshold:", self.tiny_wet_cell_threshold_spin
        )
        self.tiny_wet_cell_threshold_spin.setRange(1, 10000000)
        self.tiny_wet_cell_threshold_spin.setValue(2000)

        self.enable_cuda_graphs_chk = QtWidgets.QCheckBox("Enable")
        self.enable_cuda_graphs_chk.setObjectName("enable_cuda_graphs_chk")
        self.enable_cuda_graphs_chk.setToolTip(
            "Enable CUDA graph replay for solver kernel launches. "
            "Reduces kernel launch overhead by capturing and replaying the "
            "entire kernel graph. Only effective on CUDA 10+ with stable kernel topology."
        )
        param_form.addRow("CUDA graph replay:", self.enable_cuda_graphs_chk)
        self.enable_cuda_graphs_chk.setChecked(False)

        self.swe2d_perf_mode_chk = QtWidgets.QCheckBox("Enable")
        self.swe2d_perf_mode_chk.setObjectName("swe2d_perf_mode_chk")
        self.swe2d_perf_mode_chk.setToolTip(
            "High-performance mode for the SWE2D solver. Enables aggressive "
            "optimizations (kernel fusion, reduced synchronization, stream overlap) "
            "for maximum GPU throughput. May reduce diagnostic granularity."
        )
        param_form.addRow("SWE2D perf mode:", self.swe2d_perf_mode_chk)
        self.swe2d_perf_mode_chk.setChecked(False)

        self.internal_flow_layer_combo = QtWidgets.QComboBox()
        self.internal_flow_layer_combo.setObjectName("internal_flow_layer_combo")
        self.internal_flow_layer_combo.setToolTip(
            "Polygon layer defining internal source/sink flow regions. "
            "Each polygon specifies a flow rate (e.g. q_cms) applied as a source term. "
            "Select '(none)' to disable internal flows."
        )
        param_form.addRow("Internal flow layer:", self.internal_flow_layer_combo)
        self.internal_flow_layer_combo.addItem("(none)", None)

        self.internal_flow_field_edit = QtWidgets.QLineEdit()
        self.internal_flow_field_edit.setObjectName("internal_flow_field_edit")
        self.internal_flow_field_edit.setToolTip(
            "Field name in the internal flow layer containing source/sink discharge values. "
            "Default: 'q_cms' (CMS). Positive values = source, negative = sink."
        )
        self.internal_flow_field_edit.setText("q_cms")
        self.internal_flow_field_edit.setPlaceholderText("field name, e.g. q_cms")
        param_form.addRow("Internal flow field:", self.internal_flow_field_edit)

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
        param_form.addRow("Run duration:", self.run_time_edit)

        self.reconstruction_combo = QtWidgets.QComboBox()
        self.reconstruction_combo.setObjectName("reconstruction_combo")
        self.reconstruction_combo.setToolTip(
            "Spatial reconstruction scheme for cell-face extrapolation. "
            "First-order (0): fastest, most diffusive. MUSCL variants (1–4) "
            "add second-order accuracy with different slope limiters. "
            "WENO (5–6) use weighted stencils for smooth regions with shock capture."
        )
        for text, data in [
            ("First-order (baseline)", 0),
            ("MUSCL Fast (high-throughput)", 1),
            ("MUSCL MinMod (robust)", 2),
            ("MUSCL MC (less-diffusive TVD)", 3),
            ("MUSCL Van Leer (smooth TVD)", 4),
            ("WENO3-like (GPU experimental)", 5),
            ("WENO5 (GPU, 3rd-order LSQ)", 6),
        ]:
            self.reconstruction_combo.addItem(text, data)
        param_form.addRow("Reconstruction:", self.reconstruction_combo)

        self.temporal_order_combo = QtWidgets.QComboBox()
        self.temporal_order_combo.setObjectName("temporal_order_combo")
        self.temporal_order_combo.setToolTip(
            "Temporal integration (ODE solver) order. "
            "Euler RK1 (1): first-order, most robust but lowest accuracy. "
            "RK2 Heun (2): second-order, good balance (default). "
            "RK4 (4): classic 4th-order Runge-Kutta. "
            "Graph-safe RK4/RK5 (5/6): staged versions compatible with CUDA graph replay."
        )
        for text, data in [
            ("Euler (RK1, 1st-order)", 1),
            ("RK2 (Heun, 2nd-order, default)", 2),
            ("RK4 (classic, 4th-order)", 4),
            ("Graph-safe RK4 (true staged)", 5),
            ("Graph-safe RK5 (Cash-Karp)", 6),
        ]:
            self.temporal_order_combo.addItem(text, data)
        param_form.addRow("Temporal discretization:", self.temporal_order_combo)

        self.degen_mode_combo = QtWidgets.QComboBox()
        self.degen_mode_combo.setObjectName("degen_mode_combo")
        self.degen_mode_combo.setToolTip(
            "Mode for handling degenerate cells (area ≤ 0 or invalid topology). "
            "Off (0): treat normally, may cause instability. "
            "Skip (1): permanently exclude degenerate cells (fastest). "
            "Repair (2): replace inv_area with neighbor-average (most robust). "
            "Merge (3): redirect degenerate cell flux to owner cell."
        )
        param_form.addRow("Degenerate cell mode:", self.degen_mode_combo)
        self.degen_mode_combo.addItem("Off (0)", 0)
        self.degen_mode_combo.addItem("Skip (1)", 1)
        self.degen_mode_combo.addItem("Repair (2)", 2)
        self.degen_mode_combo.addItem("Merge (3)", 3)
        self.degen_mode_combo.setCurrentIndex(self.degen_mode_combo.findData(0))

    def _build_rain_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Rain / Hydrology page with rainfall and infiltration controls."""
        self.max_rel_depth_increase_spin = QtWidgets.QDoubleSpinBox()
        self.max_rel_depth_increase_spin.setObjectName("max_rel_depth_increase_spin")
        self.max_rel_depth_increase_spin.setToolTip(
            "Maximum relative water depth increase per timestep due to source terms. "
            "0 = unlimited. Typical: 2.0 (2x depth per step). Range: 0–1000."
        )
        param_form.addRow(
            "Max rel depth increase:", self.max_rel_depth_increase_spin
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
        param_form.addRow("Max source dh/step:", self.max_source_depth_step_spin)
        self.max_source_depth_step_spin.setRange(0.0, 10.0)
        self.max_source_depth_step_spin.setDecimals(6)
        self.max_source_depth_step_spin.setValue(0.0)

        self.max_source_rate_spin = QtWidgets.QDoubleSpinBox()
        self.max_source_rate_spin.setObjectName("max_source_rate_spin")
        self.max_source_rate_spin.setToolTip(
            "Maximum source rate (rainfall intensity) threshold in model "
            "units. Values above this cap are clamped. 0 = no cap. Range: 0–100."
        )
        param_form.addRow("Max source rate:", self.max_source_rate_spin)
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
        param_form.addRow("Extreme rain mode:", self.extreme_rain_mode_chk)
        self.extreme_rain_mode_chk.setChecked(False)

        self.source_cfl_beta_spin = QtWidgets.QDoubleSpinBox()
        self.source_cfl_beta_spin.setObjectName("source_cfl_beta_spin")
        self.source_cfl_beta_spin.setToolTip(
            "CFL beta factor for source term sub-stepping. "
            "Lower values → smaller source substeps → more stability. "
            "Range: 0.01–2.0. Default: 0.25."
        )
        param_form.addRow("Source CFL beta:", self.source_cfl_beta_spin)
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
        param_form.addRow("Source max substeps:", self.source_max_substeps_spin)
        self.source_max_substeps_spin.setRange(1, 512)
        self.source_max_substeps_spin.setValue(16)

        self.source_true_subcycling_chk = QtWidgets.QCheckBox("Enable")
        self.source_true_subcycling_chk.setObjectName("source_true_subcycling_chk")
        self.source_true_subcycling_chk.setToolTip(
            "When enabled, source terms are integrated with true sub-cycling "
            "(multiple substeps per hydrodynamic step). When disabled, sources "
            "are integrated with the main timestep."
        )
        param_form.addRow("True source subcycling:", self.source_true_subcycling_chk)
        self.source_true_subcycling_chk.setChecked(False)

        self.source_imex_split_chk = QtWidgets.QCheckBox("Enable")
        self.source_imex_split_chk.setObjectName("source_imex_split_chk")
        self.source_imex_split_chk.setToolTip(
            "Split source terms into Implicit-Explicit (IMEX) components. "
            "Stiff source terms (e.g. friction) are treated implicitly, "
            "non-stiff terms (e.g. rainfall) explicitly."
        )
        param_form.addRow("IMEX source split:", self.source_imex_split_chk)
        self.source_imex_split_chk.setChecked(False)

        self.source_stage_coupled_imex_rk2_chk = QtWidgets.QCheckBox("Enable")
        self.source_stage_coupled_imex_rk2_chk.setObjectName(
            "source_stage_coupled_imex_rk2_chk"
        )
        self.source_stage_coupled_imex_rk2_chk.setToolTip(
            "Stage-coupled IMEX-RK2 integration for sources. "
            "Ties source evaluation to intermediate RK stages for tighter "
            "coupling with the hydrodynamic solver."
        )
        param_form.addRow(
            "Stage-coupled IMEX-RK2 sources:",
            self.source_stage_coupled_imex_rk2_chk,
        )
        self.source_stage_coupled_imex_rk2_chk.setChecked(False)

        self.rain_rate_spin = QtWidgets.QDoubleSpinBox()
        self.rain_rate_spin.setObjectName("rain_rate_spin")
        self.rain_rate_spin.setToolTip(
            "Uniform rainfall rate in mm/hr applied to the entire domain. "
            "Range: 0–2000 mm/hr. Use spatial rainfall layers for non-uniform "
            "rainfall distribution."
        )
        param_form.addRow("Rain rate:", self.rain_rate_spin)
        self.rain_rate_spin.setRange(0.0, 2000.0)
        self.rain_rate_spin.setDecimals(3)
        self.rain_rate_spin.setValue(0.0)
        self.rain_rate_spin.setSuffix(" mm/hr")

        self.cn_default_spin = QtWidgets.QDoubleSpinBox()
        self.cn_default_spin.setObjectName("cn_default_spin")
        self.cn_default_spin.setToolTip(
            "Default SCS Curve Number for runoff computation. "
            "CN values range from 1 (high infiltration) to 100 (impervious). "
            "Typical: 75 (residential), 85 (urban). Overridden by CN polygon layer if present."
        )
        param_form.addRow("Default CN:", self.cn_default_spin)
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
        param_form.addRow("SCS Ia/S ratio:", self.ia_ratio_spin)
        self.ia_ratio_spin.setRange(0.0, 1.0)
        self.ia_ratio_spin.setDecimals(3)
        self.ia_ratio_spin.setSingleStep(0.01)
        self.ia_ratio_spin.setValue(0.2)

        self.use_spatial_rain_cn_chk = QtWidgets.QCheckBox(
            "Use Thiessen gage rainfall when layers are available"
        )
        self.use_spatial_rain_cn_chk.setObjectName("use_spatial_rain_cn_chk")
        self.use_spatial_rain_cn_chk.setToolTip(
            "When checked and rain gage + hyetograph layers are configured, "
            "Thiessen polygon interpolation is applied for spatially variable "
            "rainfall. Uses CN polygons for spatial infiltration if available."
        )
        param_form.addRow("Spatial rainfall:", self.use_spatial_rain_cn_chk)
        self.use_spatial_rain_cn_chk.setChecked(True)

        self.infiltration_method_combo = QtWidgets.QComboBox()
        self.infiltration_method_combo.setObjectName("infiltration_method_combo")
        self.infiltration_method_combo.setToolTip(
            "Infiltration model for rainfall-runoff computation. "
            "'SCS Curve Number' uses the NRCS runoff curve number method. "
            "'None' skips infiltration entirely."
        )
        param_form.addRow("Infiltration method:", self.infiltration_method_combo)
        self.infiltration_method_combo.addItem("SCS Curve Number", "scs_cn")
        self.infiltration_method_combo.addItem("None (no infiltration)", "none")
        self.infiltration_method_combo.setCurrentIndex(
            self.infiltration_method_combo.findData("scs_cn")
        )

        self.storm_area_layer_combo = QtWidgets.QComboBox()
        self.storm_area_layer_combo.setObjectName("storm_area_layer_combo")
        self.storm_area_layer_combo.setToolTip(
            "Optional polygon layer defining the storm area extent. "
            "Only cells within storm area polygons receive rainfall. "
            "Select '(none)' to apply rain globally."
        )
        param_form.addRow("Storm area layer (optional):", self.storm_area_layer_combo)
        self.storm_area_layer_combo.addItem("(none)", None)

        self.rain_boundary_buffer_rings_spin = QtWidgets.QSpinBox()
        self.rain_boundary_buffer_rings_spin.setObjectName(
            "rain_boundary_buffer_rings_spin"
        )
        self.rain_boundary_buffer_rings_spin.setToolTip(
            "Number of boundary buffer rings to which rainfall is still applied when "
            "using a storm area layer. Prevents dry boundary artifacts. Range: 0–10."
        )
        param_form.addRow(
            "Rain boundary buffer rings:", self.rain_boundary_buffer_rings_spin
        )
        self.rain_boundary_buffer_rings_spin.setRange(0, 10)
        self.rain_boundary_buffer_rings_spin.setValue(1)

    def _build_stability_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Stability Controls page with damping and cap parameters."""
        self.shallow_damping_depth_spin = QtWidgets.QDoubleSpinBox()
        self.shallow_damping_depth_spin.setObjectName("shallow_damping_depth_spin")
        self.shallow_damping_depth_spin.setToolTip(
            "Depth threshold below which velocity damping is applied to "
            "stabilize wetting/drying fronts. Range: 1e-8–10. Default: 1e-4."
        )
        param_form.addRow("Shallow damping depth:", self.shallow_damping_depth_spin)
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
        param_form.addRow(
            "Shallow-front recon fallback:", self.shallow_front_recon_fallback_chk
        )
        self.shallow_front_recon_fallback_chk.setChecked(True)

        self.front_flux_damping_spin = QtWidgets.QDoubleSpinBox()
        self.front_flux_damping_spin.setObjectName("front_flux_damping_spin")
        self.front_flux_damping_spin.setToolTip(
            "Damping factor applied to fluxes at wet/dry fronts (0–1). "
            "Higher values = more damping = more stability at front. "
            "Default: 0.5. Increase if front oscillations occur."
        )
        param_form.addRow("Front flux damping:", self.front_flux_damping_spin)
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
        param_form.addRow("Active-set hysteresis:", self.active_set_hysteresis_chk)
        self.active_set_hysteresis_chk.setChecked(True)

        self.depth_cap_spin = QtWidgets.QDoubleSpinBox()
        self.depth_cap_spin.setObjectName("depth_cap_spin")
        self.depth_cap_spin.setToolTip(
            "Maximum allowable water depth (model units). "
            "Depths exceeding this cap are clamped. "
            "Range: 0.001–1e7. Default: 1e6 (effectively unlimited)."
        )
        param_form.addRow("Depth cap:", self.depth_cap_spin)
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
        param_form.addRow(
            "Momentum cap min speed:", self.momentum_cap_min_speed_spin
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
        param_form.addRow(
            "Momentum cap celerity mult:", self.momentum_cap_celerity_mult_spin
        )
        self.momentum_cap_celerity_mult_spin.setRange(0.1, 1000.0)
        self.momentum_cap_celerity_mult_spin.setDecimals(3)
        self.momentum_cap_celerity_mult_spin.setValue(20.0)

        self.max_inv_area_spin = QtWidgets.QDoubleSpinBox()
        self.max_inv_area_spin.setObjectName("max_inv_area_spin")
        self.max_inv_area_spin.setToolTip(
            "Maximum cell area for determining cell inversion risk. "
            "Large cells with steep water surface gradients may trigger "
            "inverted cell detection. Range: 1–1e12. Default: 1e6."
        )
        param_form.addRow("Max inv area:", self.max_inv_area_spin)
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
        param_form.addRow("CFL lambda cap:", self.cfl_lambda_cap_spin)
        self.cfl_lambda_cap_spin.setRange(1.0, 1.0e12)
        self.cfl_lambda_cap_spin.setDecimals(1)
        self.cfl_lambda_cap_spin.setValue(1.0e6)

    def _build_drain_form_widgets(self, param_form: QtWidgets.QFormLayout) -> None:
        """Populate the Structures & Drainage page with coupling controls."""
        self.coupling_loop_combo = QtWidgets.QComboBox()
        self.coupling_loop_combo.setObjectName("coupling_loop_combo")
        self.coupling_loop_combo.setToolTip(
            "Select the coupling backend for drainage/structure-2D interaction. "
            "'CUDA coupling loop (GPU)' runs the coupling solver on GPU."
        )
        param_form.addRow("Coupling loop:", self.coupling_loop_combo)
        self.coupling_loop_combo.addItem("CUDA coupling loop (GPU)", "cuda")
        self.coupling_loop_combo.setCurrentIndex(0)

        self.culvert_solver_mode_combo = QtWidgets.QComboBox()
        self.culvert_solver_mode_combo.setObjectName("culvert_solver_mode_combo")
        self.culvert_solver_mode_combo.setToolTip(
            "Culvert hydraulics solver method. "
            "'Direct solver' uses Newton/secant iteration at each culvert face. "
            "'Precomputed lookup table' uses interpolated discharge from stored tables."
        )
        param_form.addRow("Culvert solver mode:", self.culvert_solver_mode_combo)
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
        param_form.addRow("Culvert coupling mode:", self.culvert_face_flux_chk)

        self.use_redistribution_chk = QtWidgets.QCheckBox("Enable redistribution override")
        self.use_redistribution_chk.setObjectName("use_redistribution_chk")
        self.use_redistribution_chk.setChecked(True)
        self.use_redistribution_chk.setToolTip("When checked, reads per-structure redistribution parameters from the GeoPackage. Uncheck to skip redistribution entirely.")
        param_form.addRow(self.use_redistribution_chk)

        self.bridge_stacked_coupling_mode_combo = QtWidgets.QComboBox()
        self.bridge_stacked_coupling_mode_combo.setObjectName(
            "bridge_stacked_coupling_mode_combo"
        )
        self.bridge_stacked_coupling_mode_combo.setToolTip(
            "Spatial redistribution method for bridge stacked coupling. "
            "'Phase 3 spatial redistribution' distributes flow across multiple "
            "cells. 'Legacy scalar weighting' uses a single scalar factor."
        )
        param_form.addRow(
            "Bridge stacked coupling mode:",
            self.bridge_stacked_coupling_mode_combo,
        )
        self.bridge_stacked_coupling_mode_combo.addItem(
            "Phase 3 spatial redistribution (recommended)", "phase3_spatial"
        )
        self.bridge_stacked_coupling_mode_combo.addItem(
            "Legacy scalar weighting (backward-compatible)", "legacy_scalar"
        )
        self.bridge_stacked_coupling_mode_combo.setCurrentIndex(0)

        self.drainage_solver_mode_combo = QtWidgets.QComboBox()
        self.drainage_solver_mode_combo.setObjectName("drainage_solver_mode_combo")
        self.drainage_solver_mode_combo.setToolTip(
            "Governing equations for 1D drainage network flow. "
            "'EGL' includes Bernoulli + minor losses (recommended). "
            "'Diffusion wave' simplifies to gravity + friction. "
            "'Dynamic Saint-Venant' solves the full 1D momentum equation."
        )
        param_form.addRow("Drainage equation set:", self.drainage_solver_mode_combo)
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
        param_form.addRow("Drainage GPU method:", self.drainage_gpu_method_combo)
        self.drainage_gpu_method_combo.addItem(
            "Per-step GPU drainage (fast for sparse exchange)", "step"
        )
        self.drainage_gpu_method_combo.addItem(
            "Native iterative GPU drainage (batched substeps)", "iterative"
        )
        self.drainage_gpu_method_combo.setCurrentIndex(0)

        self.drainage_coupling_substeps_spin = QtWidgets.QSpinBox()
        self.drainage_coupling_substeps_spin.setObjectName(
            "drainage_coupling_substeps_spin"
        )
        self.drainage_coupling_substeps_spin.setToolTip(
            "Number of drainage substeps per SWE2D timestep. "
            "Range: 1–256. Default: 1. Increase for stiffer drainage systems."
        )
        param_form.addRow("Drainage substeps:", self.drainage_coupling_substeps_spin)
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
        param_form.addRow(
            "Drainage max adaptive substeps:",
            self.drainage_max_coupling_substeps_spin,
        )
        self.drainage_max_coupling_substeps_spin.setRange(1, 1024)
        self.drainage_max_coupling_substeps_spin.setValue(64)

        self.drainage_head_deadband_spin = QtWidgets.QDoubleSpinBox()
        self.drainage_head_deadband_spin.setObjectName("drainage_head_deadband_spin")
        self.drainage_head_deadband_spin.setToolTip(
            "Head (water surface elevation difference) deadband below which "
            "no drainage flow is computed. Prevents oscillation near zero flow. "
            "Range: 0–10. Default: 0.001."
        )
        param_form.addRow(
            "Drainage head deadband:", self.drainage_head_deadband_spin
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
        param_form.addRow(
            "Drainage dynamic relaxation:", self.drainage_dynamic_relaxation_spin
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
        param_form.addRow(
            "Drainage adaptive depth fraction:",
            self.drainage_adaptive_depth_fraction_spin,
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
        param_form.addRow(
            "Drainage adaptive wave Courant:",
            self.drainage_adaptive_wave_courant_spin,
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
        param_form.addRow(
            "Drainage implicit iterations (GPU):", self.drainage_implicit_iters_spin
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
        param_form.addRow(
            "Drainage implicit relaxation (GPU):", self.drainage_implicit_relax_spin
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

    def _on_select_results_gpkg(self) -> None:
        """Open a file dialog and populate the results GeoPackage path."""
        from qgis.PyQt.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Results GeoPackage", "",
            "GeoPackage (*.gpkg);;All files (*)",
        )
        if path:
            self.results_gpkg_path_edit.setText(path)
