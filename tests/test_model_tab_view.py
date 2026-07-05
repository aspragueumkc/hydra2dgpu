"""Tests for ModelTabView."""
import unittest
from qgis.PyQt.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton,
    QSpinBox, QToolBox, QVBoxLayout, QWidget,
)


_app = None


def _ensure_app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])


class TestModelTabView(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def _make_view(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        return ModelTabView()

    def test_view_imports(self):
        from swe2d.workbench.views.model_tab_view import ModelTabView
        self.assertIsNotNone(ModelTabView)

    def test_view_is_qwidget(self):
        view = self._make_view()
        self.assertIsInstance(view, QWidget)

    def test_view_has_solver_form(self):
        view = self._make_view()
        form = view.findChild(QFormLayout, "model_solver_form")
        self.assertIsNotNone(form)

    def test_view_has_rain_form(self):
        view = self._make_view()
        form = view.findChild(QFormLayout, "model_rain_form")
        self.assertIsNotNone(form)

    def test_view_has_drain_form(self):
        view = self._make_view()
        form = view.findChild(QFormLayout, "model_drain_form")
        self.assertIsNotNone(form)

    def test_view_has_solver_page(self):
        view = self._make_view()
        page = view.findChild(QWidget, "model_solver_page")
        self.assertIsNotNone(page)

    def test_view_has_rain_page(self):
        view = self._make_view()
        page = view.findChild(QWidget, "model_rain_page")
        self.assertIsNotNone(page)

    def test_view_has_drain_page(self):
        view = self._make_view()
        page = view.findChild(QWidget, "model_drain_page")
        self.assertIsNotNone(page)

    def test_view_has_run_page(self):
        view = self._make_view()
        page = view.findChild(QWidget, "model_run_page")
        self.assertIsNone(page)

    def test_run_output_widgets_live_on_output_page(self):
        """Output-config widgets (formerly below the Run dock progress
        bar) now live on the Model tab's Output page."""
        view = self._make_view()
        # These are the output-config widgets that moved into ModelTabView:
        for attr in (
            "output_interval_edit", "line_output_interval_edit",
            "results_table_name_edit", "results_gpkg_path_edit",
            "select_results_gpkg_btn", "load_run_settings_btn", "save_settings_btn",
        ):
            with self.subTest(attr=attr):
                self.assertTrue(
                    hasattr(view, attr),
                    f"ModelTabView should now own {attr} on the Output page",
                )

    def test_run_execution_widgets_absent(self):
        """Run/Cancel/Snapshot/Batch/progress widgets stay on RunDockWidget
        (NOT on ModelTabView)."""
        view = self._make_view()
        for attr in ("run_btn", "cancel_btn", "batch_sim_btn", "progress_bar", "snapshot_btn"):
            with self.subTest(attr=attr):
                self.assertFalse(hasattr(view, attr), f"Run control should not be on ModelTabView: {attr}")

    def test_view_has_n_mann_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.n_mann_spin, QDoubleSpinBox)
        self.assertEqual(view.n_mann_spin.objectName(), "n_mann_spin")

    def test_view_has_cfl_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.cfl_spin, QDoubleSpinBox)
        self.assertEqual(view.cfl_spin.objectName(), "cfl_spin")

    def test_view_has_h_min_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.h_min_spin, QDoubleSpinBox)
        self.assertEqual(view.h_min_spin.objectName(), "h_min_spin")

    def test_view_has_initial_condition_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.initial_condition_combo, QComboBox)
        self.assertEqual(view.initial_condition_combo.objectName(), "initial_condition_combo")

    def test_view_has_initial_depth_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.initial_depth_spin, QDoubleSpinBox)
        self.assertEqual(view.initial_depth_spin.objectName(), "initial_depth_spin")

    def test_view_has_initial_wse_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.initial_wse_spin, QDoubleSpinBox)
        self.assertEqual(view.initial_wse_spin.objectName(), "initial_wse_spin")

    def test_view_has_adaptive_cfl_dt_chk(self):
        view = self._make_view()
        self.assertIsInstance(view.adaptive_cfl_dt_chk, QCheckBox)
        self.assertEqual(view.adaptive_cfl_dt_chk.objectName(), "adaptive_cfl_dt_chk")

    def test_view_has_dt_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.dt_spin, QDoubleSpinBox)
        self.assertEqual(view.dt_spin.objectName(), "dt_spin")

    def test_view_has_initial_dt_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.initial_dt_spin, QDoubleSpinBox)
        self.assertEqual(view.initial_dt_spin.objectName(), "initial_dt_spin")

    def test_view_has_gpu_diag_sync_interval_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.gpu_diag_sync_interval_spin, QSpinBox)
        self.assertEqual(view.gpu_diag_sync_interval_spin.objectName(), "gpu_diag_sync_interval_spin")

    def test_view_has_tiny_mode_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.tiny_mode_combo, QComboBox)
        self.assertEqual(view.tiny_mode_combo.objectName(), "tiny_mode_combo")

    def test_view_has_tiny_wet_cell_threshold_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.tiny_wet_cell_threshold_spin, QSpinBox)
        self.assertEqual(view.tiny_wet_cell_threshold_spin.objectName(), "tiny_wet_cell_threshold_spin")

    def test_view_has_enable_cuda_graphs_chk(self):
        view = self._make_view()
        self.assertIsInstance(view.enable_cuda_graphs_chk, QCheckBox)
        self.assertEqual(view.enable_cuda_graphs_chk.objectName(), "enable_cuda_graphs_chk")

    def test_view_has_swe2d_perf_mode_chk(self):
        view = self._make_view()
        self.assertIsInstance(view.swe2d_perf_mode_chk, QCheckBox)
        self.assertEqual(view.swe2d_perf_mode_chk.objectName(), "swe2d_perf_mode_chk")

    def test_view_has_internal_flow_layer_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.internal_flow_layer_combo, QComboBox)
        self.assertEqual(view.internal_flow_layer_combo.objectName(), "internal_flow_layer_combo")

    def test_view_has_internal_flow_field_edit(self):
        view = self._make_view()
        self.assertIsInstance(view.internal_flow_field_edit, QLineEdit)
        self.assertEqual(view.internal_flow_field_edit.objectName(), "internal_flow_field_edit")

    def test_view_has_run_time_edit(self):
        view = self._make_view()
        self.assertIsInstance(view.run_time_edit, QLineEdit)
        self.assertEqual(view.run_time_edit.objectName(), "run_time_edit")

    def test_view_has_reconstruction_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.reconstruction_combo, QComboBox)
        self.assertEqual(view.reconstruction_combo.objectName(), "reconstruction_combo")

    def test_view_has_temporal_order_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.temporal_order_combo, QComboBox)
        self.assertEqual(view.temporal_order_combo.objectName(), "temporal_order_combo")

    def test_view_has_degen_mode_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.degen_mode_combo, QComboBox)
        self.assertEqual(view.degen_mode_combo.objectName(), "degen_mode_combo")
        # Verify all four options are present
        for expected_data in (0, 1, 2, 3):
            self.assertIsNotNone(
                view.degen_mode_combo.findData(expected_data),
                f"Missing degen_mode option with data={expected_data}",
            )

    def test_view_has_max_rel_depth_increase_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.max_rel_depth_increase_spin, QDoubleSpinBox)
        self.assertEqual(view.max_rel_depth_increase_spin.objectName(), "max_rel_depth_increase_spin")

    def test_view_has_max_source_depth_step_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.max_source_depth_step_spin, QDoubleSpinBox)
        self.assertEqual(view.max_source_depth_step_spin.objectName(), "max_source_depth_step_spin")

    def test_view_has_max_source_rate_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.max_source_rate_spin, QDoubleSpinBox)
        self.assertEqual(view.max_source_rate_spin.objectName(), "max_source_rate_spin")

    def test_view_has_extreme_rain_mode_chk(self):
        view = self._make_view()
        self.assertIsInstance(view.extreme_rain_mode_chk, QCheckBox)
        self.assertEqual(view.extreme_rain_mode_chk.objectName(), "extreme_rain_mode_chk")

    def test_view_has_source_cfl_beta_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.source_cfl_beta_spin, QDoubleSpinBox)
        self.assertEqual(view.source_cfl_beta_spin.objectName(), "source_cfl_beta_spin")

    def test_view_has_source_max_substeps_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.source_max_substeps_spin, QSpinBox)
        self.assertEqual(view.source_max_substeps_spin.objectName(), "source_max_substeps_spin")

    def test_view_has_source_true_subcycling_chk(self):
        view = self._make_view()
        self.assertIsInstance(view.source_true_subcycling_chk, QCheckBox)
        self.assertEqual(view.source_true_subcycling_chk.objectName(), "source_true_subcycling_chk")

    def test_view_has_source_imex_split_chk(self):
        view = self._make_view()
        self.assertIsInstance(view.source_imex_split_chk, QCheckBox)
        self.assertEqual(view.source_imex_split_chk.objectName(), "source_imex_split_chk")

    def test_view_has_shallow_damping_depth_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.shallow_damping_depth_spin, QDoubleSpinBox)
        self.assertEqual(view.shallow_damping_depth_spin.objectName(), "shallow_damping_depth_spin")

    def test_view_has_shallow_front_recon_fallback_chk(self):
        view = self._make_view()
        self.assertIsInstance(view.shallow_front_recon_fallback_chk, QCheckBox)
        self.assertEqual(view.shallow_front_recon_fallback_chk.objectName(), "shallow_front_recon_fallback_chk")

    def test_view_has_front_flux_damping_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.front_flux_damping_spin, QDoubleSpinBox)
        self.assertEqual(view.front_flux_damping_spin.objectName(), "front_flux_damping_spin")

    def test_view_has_active_set_hysteresis_chk(self):
        view = self._make_view()
        self.assertIsInstance(view.active_set_hysteresis_chk, QCheckBox)
        self.assertEqual(view.active_set_hysteresis_chk.objectName(), "active_set_hysteresis_chk")

    def test_view_has_depth_cap_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.depth_cap_spin, QDoubleSpinBox)
        self.assertEqual(view.depth_cap_spin.objectName(), "depth_cap_spin")

    def test_view_has_momentum_cap_min_speed_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.momentum_cap_min_speed_spin, QDoubleSpinBox)
        self.assertEqual(view.momentum_cap_min_speed_spin.objectName(), "momentum_cap_min_speed_spin")

    def test_view_has_momentum_cap_celerity_mult_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.momentum_cap_celerity_mult_spin, QDoubleSpinBox)
        self.assertEqual(view.momentum_cap_celerity_mult_spin.objectName(), "momentum_cap_celerity_mult_spin")

    def test_view_has_max_inv_area_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.max_inv_area_spin, QDoubleSpinBox)
        self.assertEqual(view.max_inv_area_spin.objectName(), "max_inv_area_spin")

    def test_view_has_cfl_lambda_cap_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.cfl_lambda_cap_spin, QDoubleSpinBox)
        self.assertEqual(view.cfl_lambda_cap_spin.objectName(), "cfl_lambda_cap_spin")

    def test_view_has_rain_rate_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.rain_rate_spin, QDoubleSpinBox)
        self.assertEqual(view.rain_rate_spin.objectName(), "rain_rate_spin")

    def test_view_has_cn_default_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.cn_default_spin, QDoubleSpinBox)
        self.assertEqual(view.cn_default_spin.objectName(), "cn_default_spin")

    def test_view_has_ia_ratio_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.ia_ratio_spin, QDoubleSpinBox)
        self.assertEqual(view.ia_ratio_spin.objectName(), "ia_ratio_spin")

    def test_view_has_use_spatial_rain_cn_chk(self):
        view = self._make_view()
        self.assertIsInstance(view.use_spatial_rain_cn_chk, QCheckBox)
        self.assertEqual(view.use_spatial_rain_cn_chk.objectName(), "use_spatial_rain_cn_chk")

    def test_view_has_infiltration_method_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.infiltration_method_combo, QComboBox)
        self.assertEqual(view.infiltration_method_combo.objectName(), "infiltration_method_combo")

    def test_view_has_storm_area_layer_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.storm_area_layer_combo, QComboBox)
        self.assertEqual(view.storm_area_layer_combo.objectName(), "storm_area_layer_combo")

    def test_view_has_rain_boundary_buffer_rings_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.rain_boundary_buffer_rings_spin, QSpinBox)
        self.assertEqual(view.rain_boundary_buffer_rings_spin.objectName(), "rain_boundary_buffer_rings_spin")

    def test_view_has_coupling_loop_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.coupling_loop_combo, QComboBox)
        self.assertEqual(view.coupling_loop_combo.objectName(), "coupling_loop_combo")

    def test_view_has_culvert_solver_mode_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.culvert_solver_mode_combo, QComboBox)
        self.assertEqual(view.culvert_solver_mode_combo.objectName(), "culvert_solver_mode_combo")

    def test_view_has_culvert_face_flux_chk(self):
        view = self._make_view()
        self.assertIsInstance(view.culvert_face_flux_chk, QCheckBox)
        self.assertEqual(view.culvert_face_flux_chk.objectName(), "culvert_face_flux_chk")

    def test_view_has_bridge_stacked_coupling_mode_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.bridge_stacked_coupling_mode_combo, QComboBox)
        self.assertEqual(view.bridge_stacked_coupling_mode_combo.objectName(), "bridge_stacked_coupling_mode_combo")

    def test_view_has_drainage_solver_mode_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.drainage_solver_mode_combo, QComboBox)
        self.assertEqual(view.drainage_solver_mode_combo.objectName(), "drainage_solver_mode_combo")

    def test_view_has_drainage_gpu_method_combo(self):
        view = self._make_view()
        self.assertIsInstance(view.drainage_gpu_method_combo, QComboBox)
        self.assertEqual(view.drainage_gpu_method_combo.objectName(), "drainage_gpu_method_combo")

    def test_view_has_drainage_coupling_substeps_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.drainage_coupling_substeps_spin, QSpinBox)
        self.assertEqual(view.drainage_coupling_substeps_spin.objectName(), "drainage_coupling_substeps_spin")

    def test_view_has_drainage_max_coupling_substeps_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.drainage_max_coupling_substeps_spin, QSpinBox)
        self.assertEqual(view.drainage_max_coupling_substeps_spin.objectName(), "drainage_max_coupling_substeps_spin")

    def test_view_has_drainage_head_deadband_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.drainage_head_deadband_spin, QDoubleSpinBox)
        self.assertEqual(view.drainage_head_deadband_spin.objectName(), "drainage_head_deadband_spin")

    def test_view_has_drainage_dynamic_relaxation_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.drainage_dynamic_relaxation_spin, QDoubleSpinBox)
        self.assertEqual(view.drainage_dynamic_relaxation_spin.objectName(), "drainage_dynamic_relaxation_spin")

    def test_view_has_drainage_adaptive_depth_fraction_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.drainage_adaptive_depth_fraction_spin, QDoubleSpinBox)
        self.assertEqual(view.drainage_adaptive_depth_fraction_spin.objectName(), "drainage_adaptive_depth_fraction_spin")

    def test_view_has_drainage_adaptive_wave_courant_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.drainage_adaptive_wave_courant_spin, QDoubleSpinBox)
        self.assertEqual(view.drainage_adaptive_wave_courant_spin.objectName(), "drainage_adaptive_wave_courant_spin")

    def test_view_has_drainage_implicit_iters_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.drainage_implicit_iters_spin, QSpinBox)
        self.assertEqual(view.drainage_implicit_iters_spin.objectName(), "drainage_implicit_iters_spin")

    def test_view_has_drainage_implicit_relax_spin(self):
        view = self._make_view()
        self.assertIsInstance(view.drainage_implicit_relax_spin, QDoubleSpinBox)
        self.assertEqual(view.drainage_implicit_relax_spin.objectName(), "drainage_implicit_relax_spin")

    def test_view_has_gpu_default_lbl(self):
        view = self._make_view()
        self.assertIsInstance(view.gpu_default_lbl, QLabel)
        self.assertEqual(view.gpu_default_lbl.objectName(), "gpu_default_lbl")

    def test_view_has_unit_system_lbl(self):
        view = self._make_view()
        self.assertIsInstance(view.unit_system_lbl, QLabel)
        self.assertEqual(view.unit_system_lbl.objectName(), "unit_system_lbl")

    def test_view_is_standalone(self):
        view = self._make_view()
        view.n_mann_spin.setValue(0.045)
        self.assertEqual(view.n_mann_spin.value(), 0.045)
        view.deleteLater()

    def test_combo_labels_are_human_readable(self):
        view = self._make_view()
        tiny_labels = [view.tiny_mode_combo.itemText(i) for i in range(view.tiny_mode_combo.count())]
        self.assertIn("Persistent", tiny_labels)
        recon_labels = [view.reconstruction_combo.itemText(i) for i in range(view.reconstruction_combo.count())]
        self.assertIn("MUSCL + Superbee", recon_labels)
        bridge_labels = [view.bridge_stacked_coupling_mode_combo.itemText(i)
                         for i in range(view.bridge_stacked_coupling_mode_combo.count())]
        self.assertIn("Phase 3 — Spatial", bridge_labels)

    def test_view_has_toolbox(self):
        view = self._make_view()
        self.assertIsInstance(view.model_toolbox, QToolBox)

    def test_view_toolbox_has_five_pages(self):
        view = self._make_view()
        # Solver, Rain, Stability, Drain, Output (Layers page removed)
        self.assertEqual(view.model_toolbox.count(), 5)

    def test_view_pages_have_expanding_size_policy(self):
        view = self._make_view()
        from qgis.PyQt.QtWidgets import QSizePolicy
        for page_name in ("model_solver_page", "model_rain_page", "model_stability_page", "model_drain_page", "model_output_page"):
            page = view.findChild(QWidget, page_name)
            self.assertIsNotNone(page)
            self.assertEqual(page.sizePolicy().verticalPolicy(), QSizePolicy.Expanding)

    def test_solver_page_has_group_boxes(self):
        view = self._make_view()
        page = view.findChild(QWidget, "model_solver_page")
        groups = page.findChildren(QGroupBox)
        titles = {g.title() for g in groups}
        for expected in ("Time Stepping", "Boundary Conditions", "Physics & Friction", "Initial Conditions", "Run Duration"):
            self.assertIn(expected, titles)

    def test_stability_page_has_group_boxes(self):
        view = self._make_view()
        page = view.findChild(QWidget, "model_stability_page")
        groups = page.findChildren(QGroupBox)
        titles = {g.title() for g in groups}
        for expected in ("Wet/Dry Front", "Capping", "Solver Safety"):
            self.assertIn(expected, titles)

    def test_rain_page_has_group_boxes(self):
        view = self._make_view()
        page = view.findChild(QWidget, "model_rain_page")
        groups = page.findChildren(QGroupBox)
        titles = {g.title() for g in groups}
        for expected in ("Rainfall Input", "Infiltration", "Source Stability"):
            self.assertIn(expected, titles)

    def test_drain_page_has_group_boxes(self):
        view = self._make_view()
        page = view.findChild(QWidget, "model_drain_page")
        groups = page.findChildren(QGroupBox)
        titles = {g.title() for g in groups}
        for expected in ("Culvert / Bridge", "Drainage Network — Equation Set", "Drainage — Substepping", "Drainage — Stability"):
            self.assertIn(expected, titles)

    def test_model_tab_has_search_filter(self):
        view = self._make_view()
        self.assertIsInstance(view.param_search, QLineEdit)
        self.assertEqual(view.param_search.objectName(), "param_search")

    def test_model_tab_has_advanced_toggle(self):
        view = self._make_view()
        self.assertIsInstance(view.show_advanced_chk, QCheckBox)
        self.assertEqual(view.show_advanced_chk.objectName(), "show_advanced_chk")
        self.assertFalse(view.show_advanced_chk.isChecked())

    def test_filter_hides_non_matching_rows(self):
        view = self._make_view()
        view.show()
        view.show_advanced_chk.setChecked(True)
        view.param_search.setText("cfl")
        view._filter_model_tab()
        # Inspect via _filterable.filter_visible (works in headless tests).
        from swe2d.workbench.views.widget_filter_helper import FilterableRowRegistry
        self.assertIsInstance(view._filterable, FilterableRowRegistry)
        # Find at least one row whose search blob contains "cfl"
        cfl_matches = sum(
            1 for _g, _l, w, _a in view._filterable
            if "cfl" in (w.toolTip() or "").lower() or "cfl" in w.objectName().lower()
        )
        self.assertGreater(cfl_matches, 0)
        # And those rows should now report filter_visible=True
        any_visible = False
        for _g, _l, w, _a in view._filterable:
            blob = str(w.property("filter_search_blob") or "")
            if "cfl" in blob and view._filterable.filter_visible(w):
                any_visible = True
                break
        self.assertTrue(any_visible, "Expected at least one CFL row to be filter-visible")

    def test_advanced_toggle_hides_advanced_rows(self):
        view = self._make_view()
        view.show()
        view.param_search.clear()
        view.show_advanced_chk.setChecked(False)
        view._filter_model_tab()
        # For every registered row that was flagged advanced, the filter
        # must report filter_visible=False (hidden).
        for _group, _label, widget, advanced in view._filterable:
            if advanced:
                self.assertFalse(
                    view._filterable.filter_visible(widget),
                    f"Advanced widget {widget.objectName()} should be hidden",
                )

    def test_run_controls_deleted_from_model_tab(self):
        """Run controls moved to RunDockWidget — verify absent from ModelTabView."""
        view = self._make_view()
        for attr in ("run_btn", "cancel_btn", "progress_bar", "batch_sim_btn", "snapshot_btn"):
            with self.subTest(attr=attr):
                self.assertFalse(hasattr(view, attr), f"Orphan attribute should be deleted: {attr}")

    def test_view_has_default_bc_type_combo(self):
        """Boundary conditions group moved from Map tab to Solver Parameters page."""
        view = self._make_view()
        self.assertIsInstance(view.default_bc_type_combo, QComboBox)
        self.assertEqual(view.default_bc_type_combo.objectName(), "default_bc_type_combo")
        # Verify it has BC options
        self.assertGreater(view.default_bc_type_combo.count(), 0)

    def test_view_has_inflow_progressive_chk(self):
        """Boundary conditions group moved from Map tab to Solver Parameters page."""
        view = self._make_view()
        self.assertIsInstance(view.inflow_progressive_chk, QCheckBox)
        self.assertEqual(view.inflow_progressive_chk.objectName(), "inflow_progressive_chk")

    def test_view_has_uniform_inflow_velocity_chk(self):
        """Boundary conditions group moved from Map tab to Solver Parameters page."""
        view = self._make_view()
        self.assertIsInstance(view.uniform_inflow_velocity_chk, QCheckBox)
        self.assertEqual(view.uniform_inflow_velocity_chk.objectName(), "uniform_inflow_velocity_chk")

    def test_solver_page_boundary_conditions_group_exists(self):
        """Boundary Conditions group is in Solver Parameters page after Time Stepping."""
        view = self._make_view()
        page = view.findChild(QWidget, "model_solver_page")
        groups = page.findChildren(QGroupBox)
        titles = {g.title() for g in groups}
        self.assertIn("Boundary Conditions", titles)

    def test_boundary_conditions_methods(self):
        """View protocol methods for BC widgets are implemented."""
        view = self._make_view()
        # is_inflow_progressive
        self.assertTrue(callable(view.is_inflow_progressive))
        self.assertIsInstance(view.is_inflow_progressive(), bool)
        # get_inflow_progressive_chk
        self.assertTrue(callable(view.get_inflow_progressive_chk))
        self.assertIs(view.get_inflow_progressive_chk(), view.inflow_progressive_chk)
        # get_default_bc_type
        self.assertTrue(callable(view.get_default_bc_type))
        self.assertIsInstance(view.get_default_bc_type(), int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
