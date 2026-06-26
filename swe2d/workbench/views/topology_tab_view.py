"""Topology (Topo Mesh) tab view — owns its own widget references.

QWidget subclass for the Topo Mesh tab in the Studio workbench.
Mirrors the structure of _build_topology_tab_page_fallback in studio_dialog.py
so the dialog can delegate tab construction to this view.
"""
from __future__ import annotations

from qgis.PyQt import QtWidgets


class TopologyTabView(QtWidgets.QWidget):
    """View for the Topo Mesh tab.

    Creates and owns:
    - Layer Setup page combos: topo_nodes_combo, topo_arcs_combo,
      topo_regions_combo, topo_constraints_combo, topo_quad_edges_combo
    - Layer Setup button: topo_export_template_btn
    - Controls page combos/spin: topo_backend_combo, topo_default_size_spin,
      topo_default_cell_type_combo
    - Placeholder widget holders: topo_gmsh_controls_widget,
      topo_quality_controls_widget
    - Status label: topo_controls_summary_lbl
    - Action buttons: topo_generate_btn, topo_terminate_btn
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = None  # set by dialog after construction
        self._log_fn = None
        self._combo_layer_fn = None
        self._topo_widgets = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the toolbox with Layer Setup and Controls pages."""
        root_layout = QtWidgets.QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        toolbox = QtWidgets.QToolBox()
        toolbox.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        root_layout.addWidget(toolbox)

        layer_page = QtWidgets.QWidget()
        layer_page.setObjectName("topo_layer_page")
        layer_form = QtWidgets.QFormLayout(layer_page)
        layer_form.setObjectName("topo_layer_form")
        layer_form.setContentsMargins(4, 4, 4, 4)

        self.topo_nodes_combo = QtWidgets.QComboBox()
        self.topo_nodes_combo.setObjectName("topo_nodes_combo")
        self.topo_arcs_combo = QtWidgets.QComboBox()
        self.topo_arcs_combo.setObjectName("topo_arcs_combo")
        self.topo_regions_combo = QtWidgets.QComboBox()
        self.topo_regions_combo.setObjectName("topo_regions_combo")
        self.topo_constraints_combo = QtWidgets.QComboBox()
        self.topo_constraints_combo.setObjectName("topo_constraints_combo")
        self.topo_quad_edges_combo = QtWidgets.QComboBox()
        self.topo_quad_edges_combo.setObjectName("topo_quad_edges_combo")

        for combo, lbl_text in [
            (self.topo_nodes_combo, "Topology nodes layer:"),
            (self.topo_arcs_combo, "Topology arcs layer:"),
            (self.topo_regions_combo, "Topology regions layer:"),
            (self.topo_constraints_combo, "Constraints layer:"),
            (self.topo_quad_edges_combo, "Quad edges / transition layers:"),
        ]:
            layer_form.addRow(QtWidgets.QLabel(lbl_text), combo)

        self.topo_export_template_btn = QtWidgets.QPushButton(
            "Create Topology Template Layers"
        )
        self.topo_export_template_btn.setObjectName("topo_export_template_btn")
        layer_form.addRow(self.topo_export_template_btn)

        toolbox.addItem(layer_page, "Layer Setup")

        ctrl_page = QtWidgets.QWidget()
        ctrl_page.setObjectName("topo_ctrl_page")
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_page)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_form = QtWidgets.QFormLayout()
        ctrl_layout.addLayout(ctrl_form)

        self.topo_backend_combo = QtWidgets.QComboBox()
        self.topo_backend_combo.setObjectName("topo_backend_combo")
        self.topo_default_size_spin = QtWidgets.QDoubleSpinBox()
        self.topo_default_size_spin.setObjectName("topo_default_size_spin")
        self.topo_default_cell_type_combo = QtWidgets.QComboBox()
        self.topo_default_cell_type_combo.setObjectName("topo_default_cell_type_combo")

        for widget, label in [
            (self.topo_backend_combo, "Meshing backend:"),
            (self.topo_default_size_spin, "Default target size:"),
            (self.topo_default_cell_type_combo, "Default cell type:"),
        ]:
            ctrl_form.addRow(QtWidgets.QLabel(label), widget)

        self.topo_gmsh_controls_widget = QtWidgets.QWidget()
        self.topo_gmsh_controls_widget.setObjectName("topo_gmsh_controls_widget")
        self.topo_gmsh_controls_widget.setVisible(False)
        ctrl_layout.addWidget(self.topo_gmsh_controls_widget)

        self.topo_quality_controls_widget = QtWidgets.QWidget()
        self.topo_quality_controls_widget.setObjectName("topo_quality_controls_widget")
        self.topo_quality_controls_widget.setVisible(False)
        ctrl_layout.addWidget(self.topo_quality_controls_widget)

        self.topo_controls_summary_lbl = QtWidgets.QLabel(
            "Topology-layer controls: use multiple region polygons for multiple blocks."
        )
        self.topo_controls_summary_lbl.setObjectName("topo_controls_summary_lbl")
        self.topo_controls_summary_lbl.setWordWrap(True)
        ctrl_layout.addWidget(self.topo_controls_summary_lbl)

        self.topo_generate_btn = QtWidgets.QPushButton("Generate Mesh")
        self.topo_generate_btn.setObjectName("topo_generate_btn")
        self.topo_generate_btn.setEnabled(True)
        ctrl_layout.addWidget(self.topo_generate_btn)

        self.topo_terminate_btn = QtWidgets.QPushButton("Terminate")
        self.topo_terminate_btn.setObjectName("topo_terminate_btn")
        self.topo_terminate_btn.setEnabled(False)
        ctrl_layout.addWidget(self.topo_terminate_btn)

        self.topo_status_lbl = QtWidgets.QLabel(
            "Select regions layer and generate face-centric mesh"
        )
        self.topo_status_lbl.setObjectName("topo_status_lbl")
        self.topo_status_lbl.setWordWrap(True)
        ctrl_layout.addWidget(self.topo_status_lbl)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        ctrl_layout.addWidget(self.progress_bar)

        toolbox.addItem(ctrl_page, "Controls")

        self._populate_gmsh_quality_controls()

    def set_callbacks(self, log_fn=None, combo_layer_fn=None) -> None:
        """Set external callbacks for logging and layer resolution."""
        self._log_fn = log_fn
        self._combo_layer_fn = combo_layer_fn

    def _log(self, msg: str) -> None:
        """Forward a message to the log callback, if set."""
        if self._log_fn is not None:
            self._log_fn(msg)

    def _combo_layer(self, combo, kind: str = "vector"):
        """Resolve a combo's currently selected layer via the callback."""
        if self._combo_layer_fn is not None and combo is not None:
            return self._combo_layer_fn(combo, kind)
        return None

    def _populate_gmsh_quality_controls(self) -> None:
        """Populate gmsh/quality detail widgets inside the placeholder containers."""
        from swe2d.mesh.gmsh_backend import _gmsh_available

        try:
            widgets = _build_topology_tab_controls(self, self, _gmsh_available)
            self._topo_widgets = widgets
            for k, w in widgets.items():
                if isinstance(w, QtWidgets.QWidget) and not hasattr(self, k):
                    setattr(self, k, w)
            gmsh_form = widgets.get("gmsh_form")
            quality_form = widgets.get("quality_form")
            _wire_topology_tab_controls(
                widgets, gmsh_form, quality_form, self.update_control_summary
            )
        except Exception as exc:
            self._log(f"[ERROR] Failed to populate gmsh/quality controls: {exc}")

    def update_topo_status(self, text: str) -> None:
        """Set the status label text (called by topology controller)."""
        lbl = getattr(self, "topo_status_lbl", None)
        if lbl is not None:
            try:
                lbl.setText(str(text))
            except Exception:
                pass

    def update_topo_controls_summary(self, text: str) -> None:
        """Set the controls summary label text."""
        lbl = getattr(self, "topo_controls_summary_lbl", None)
        if lbl is not None:
            try:
                lbl.setText(str(text))
            except Exception:
                pass

    def _find_widget(self, attr: str):
        """Locate a widget by attribute name, checking direct attrs then _topo_widgets dict."""
        """Locate a widget by attribute name, checking direct attrs then _topo_widgets dict."""
        w = getattr(self, attr, None)
        if w is not None:
            return w
        return self._topo_widgets.get(attr)

    def get_topo_widget_value(self, attr: str):
        """Read a widget value (spin, checkbox, combo data, or line edit text) by name."""
        w = self._find_widget(attr)
        if w is None:
            return None
        try:
            if hasattr(w, "currentData"):
                return w.currentData()
            if hasattr(w, "value"):
                return w.value()
            if hasattr(w, "isChecked"):
                return w.isChecked()
            if hasattr(w, "text"):
                return w.text()
        except Exception:
            return None
        return None

    def set_topo_widget_visible(self, attr: str, visible: bool) -> None:
        """Set visibility of a topology widget by attribute name."""
        w = self._find_widget(attr)
        if w is not None:
            try:
                w.setVisible(bool(visible))
            except Exception:
                pass

    def get_topo_combo_data(self, attr: str):
        """Return currentData() of a topology combo by attribute name."""
        w = self._find_widget(attr)
        if w is not None and hasattr(w, "currentData"):
            try:
                return w.currentData()
            except Exception:
                return None
        return None

    def set_mesh_info_text(self, text: str) -> None:
        """Set the mesh info label text (protocol method for controller)."""
        w = getattr(self, "mesh_info_lbl", None)
        if w is not None:
            try:
                w.setText(str(text))
            except RuntimeError:
                pass

    def update_control_summary(self) -> None:
        """Update the topology control summary label from current state.

        Reads widget values directly from the tab view's own attributes —
        no dialog access needed.  Callbacks are used for logging and
        layer resolution.
        """
        from qgis.PyQt import QtWidgets

        log_fn = self._log
        combo_layer_fn = self._combo_layer
        if not hasattr(self, "topo_controls_summary_lbl"):
            return

        def _alive(w):
            """Check if a widget reference is still alive."""
            if w is None:
                return False
            try:
                import sip
                return not sip.isdeleted(w)
            except ImportError:
                return True

        lbl = self.topo_controls_summary_lbl
        if not _alive(lbl):
            return

        def _safe_checked(name: str, default: bool = False) -> bool:
            """Safely read a checkbox state by widget name."""
            w = self._find_widget(name)
            if not _alive(w) or not hasattr(w, "isChecked"):
                return bool(default)
            try:
                return bool(w.isChecked())
            except Exception as e:
                log_fn(f"[ERROR] _safe_checked {name}: {e}")
                return bool(default)

        def _safe_spin_value(name: str, default: float = 0.0) -> float:
            """Safely read a spin box value by widget name."""
            w = self._find_widget(name)
            if not _alive(w) or not hasattr(w, "value"):
                return float(default)
            try:
                return float(w.value())
            except Exception as e:
                log_fn(f"[ERROR] _safe_spin_value {name}: {e}")
                return float(default)

        def _safe_line_text(name: str, default: str = "") -> str:
            """Safely read a line edit text by widget name."""
            w = self._find_widget(name)
            if not _alive(w) or not hasattr(w, "text"):
                return str(default)
            try:
                return str(w.text()).strip()
            except Exception as e:
                log_fn(f"[ERROR] _safe_line_text {name}: {e}")
                return str(default)

        def _safe_combo_data(name: str, default: object = None):
            """Safely read a combo's currentData by widget name."""
            w = self._find_widget(name)
            if not _alive(w) or not hasattr(w, "currentData"):
                return default
            try:
                data = w.currentData()
                return default if data is None else data
            except Exception as e:
                log_fn(f"[ERROR] _safe_combo_data {name}: {e}")
                return default

        backend_name = str(_safe_combo_data("topo_backend_combo", "gmsh") or "gmsh")
        regions_layer = combo_layer_fn(self.topo_regions_combo, "vector") if _alive(self.topo_regions_combo) else None
        constraints_layer = combo_layer_fn(self.topo_constraints_combo, "vector") if _alive(self.topo_constraints_combo) else None
        quad_edges_layer = combo_layer_fn(self.topo_quad_edges_combo, "vector") if _alive(self.topo_quad_edges_combo) else None

        is_gmsh = backend_name == "gmsh"
        if _alive(self.topo_gmsh_controls_widget):
            self.topo_gmsh_controls_widget.setVisible(is_gmsh)
        if _alive(self.topo_quality_controls_widget):
            self.topo_quality_controls_widget.setVisible(is_gmsh)

        gmsh_quality_only = [
            self._find_widget("topo_gmsh_quality_enable_chk"),
            self._find_widget("topo_gmsh_quality_max_iters_spin"),
            self._find_widget("topo_gmsh_quality_time_limit_spin"),
            self._find_widget("topo_gmsh_quality_recombine_topology_passes_edit"),
            self._find_widget("topo_gmsh_quality_recombine_min_quality_edit"),
            self._find_widget("topo_gmsh_quality_random_factors_edit"),
            self._find_widget("topo_gmsh_quality_optimize_methods_edit"),
            self._find_widget("topo_gmsh_algo_switch_on_failure_chk"),
            self._find_widget("topo_gmsh_recombine_node_repositioning_chk"),
            self._find_widget("topo_gmsh_global_recombine_chk"),
        ]
        generic_quality_widgets = [
            self._find_widget("topo_quality_min_angle_spin"),
            self._find_widget("topo_quality_max_aspect_spin"),
            self._find_widget("topo_quality_max_non_orth_spin"),
            self._find_widget("topo_quality_min_area_edit"),
            self._find_widget("topo_quality_size_scales_edit"),
            self._find_widget("topo_quality_smooth_increments_edit"),
            self._find_widget("topo_quality_strict_chk"),
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
            gmsh_loop="on" if _safe_checked("topo_gmsh_quality_enable_chk", False) else "off",
            mesh_size_min=_safe_spin_value("topo_gmsh_mesh_size_min_spin", 0.0),
            edge_tol=_safe_spin_value("topo_gmsh_tolerance_edge_length_spin", 0.0),
            point_refine="on" if _safe_checked("topo_gmsh_mesh_size_from_points_chk", False) else "off",
            attempts=int(round(_safe_spin_value("topo_gmsh_quality_max_iters_spin", 0.0))),
            budget=_safe_spin_value("topo_gmsh_quality_time_limit_spin", 0.0),
        )

        details = []
        if regions_layer is not None:
            try:
                region_fields = set(regions_layer.fields().names())
                region_count = 0
                cartesian_count = 0
                empty_count = 0
                size_values = set()
                missing_edge_lengths = 0
                for ft in regions_layer.getFeatures():
                    rid = int(ft.attribute("region_id") or 0)
                    if rid <= 0:
                        continue
                    region_count += 1
                    ct = str(ft.attribute("cell_type") or "").strip().lower()
                    if ct == "cartesian":
                        cartesian_count += 1
                    elif ct in ("", "default"):
                        empty_count += 1
                    ts = ft.attribute("target_size")
                    if ts is not None:
                        try:
                            size_values.add(float(ts))
                        except (TypeError, ValueError):
                            pass
                    for fld in ("edge_len_1", "edge_len_2", "edge_len_3", "edge_len_4"):
                        val = ft.attribute(fld)
                        if val is None or (isinstance(val, (int, float)) and float(val) <= 0.0):
                            missing_edge_lengths += 1
                            break
                parts = [f"{region_count} regions"]
                if cartesian_count:
                    parts.append(f"{cartesian_count} cartesian")
                if empty_count:
                    parts.append(f"{empty_count} no cell_type")
                if size_values:
                    sizes = sorted(size_values)
                    label = f"target_size={'/'.join(f'{s:.4g}' for s in sizes[:5])}"
                    if len(sizes) > 5:
                        label += f"... ({len(sizes)} unique)"
                    parts.append(label)
                if missing_edge_lengths:
                    parts.append(f"{missing_edge_lengths} missing edge_len")
                details.append(" | ".join(parts))
            except Exception as e:
                log_fn(f"[ERROR] regions layer summary: {e}")

        if constraints_layer is not None:
            try:
                c_count = 0
                for ft in constraints_layer.getFeatures():
                    c_count += 1
                details.append(f"{c_count} constraints")
            except Exception as e:
                log_fn(f"[ERROR] constraints layer summary: {e}")

        if quad_edges_layer is not None:
            try:
                qe_count = 0
                for ft in quad_edges_layer.getFeatures():
                    qe_count += 1
                details.append(f"{qe_count} quad-edge controls")
            except Exception as e:
                log_fn(f"[ERROR] quad_edges layer summary: {e}")

        suffix = " | ".join(details)
        if suffix:
            lbl.setText(f"{backend_hint}{quality_hint} Current layers: {suffix}.")
        else:
            lbl.setText(f"{backend_hint}{quality_hint}")


def _build_topology_tab_controls(
    parent,
    topology_tab_page,
    gmsh_available,
) -> dict:
    """Pure view: create topology tab widgets, return dict of them.

    Returns a dict with all created widgets keyed by object name, plus
    ``"gmsh_form"`` and ``"quality_form"`` QFormLayout entries.
    """
    def _find(name, wtype):
        """Find a widget by name and type in the page or parent."""
        w = topology_tab_page.findChild(wtype, name)
        if w is None:
            w = getattr(parent, name, None)
        return w

    def _set_combo_items(
        combo,
        items,
        default_data=None,
    ) -> None:
        """Populate a combo with label/data items, preserving current selection."""
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

    def _fnc_combo(name):
        """Find or create a QComboBox with the given name."""
        w = _find(name, QtWidgets.QComboBox)
        if w is None:
            w = QtWidgets.QComboBox()
            w.setObjectName(name)
            setattr(parent, name, w)
        return w

    def _fnc_spin(name):
        """Find or create a QDoubleSpinBox with the given name."""
        w = _find(name, QtWidgets.QDoubleSpinBox)
        if w is None:
            w = QtWidgets.QDoubleSpinBox()
            w.setObjectName(name)
            setattr(parent, name, w)
        return w

    widgets = {}

    topo_nodes_combo = _fnc_combo("topo_nodes_combo")
    widgets["topo_nodes_combo"] = topo_nodes_combo
    topo_arcs_combo = _fnc_combo("topo_arcs_combo")
    widgets["topo_arcs_combo"] = topo_arcs_combo
    topo_regions_combo = _fnc_combo("topo_regions_combo")
    widgets["topo_regions_combo"] = topo_regions_combo
    topo_constraints_combo = _fnc_combo("topo_constraints_combo")
    widgets["topo_constraints_combo"] = topo_constraints_combo
    topo_quad_edges_combo = _fnc_combo("topo_quad_edges_combo")
    widgets["topo_quad_edges_combo"] = topo_quad_edges_combo
    topo_backend_combo = _fnc_combo("topo_backend_combo")
    widgets["topo_backend_combo"] = topo_backend_combo
    topo_default_cell_type_combo = _fnc_combo("topo_default_cell_type_combo")
    widgets["topo_default_cell_type_combo"] = topo_default_cell_type_combo
    topo_default_size_spin = _fnc_spin("topo_default_size_spin")
    widgets["topo_default_size_spin"] = topo_default_size_spin

    _set_combo_items(topo_constraints_combo, [("(none)", None)], default_data=None)
    _set_combo_items(topo_quad_edges_combo, [("(none)", None)], default_data=None)

    _gmsh_label = "Gmsh (recommended)" if gmsh_available() else "Gmsh (install: pip install gmsh)"
    _set_combo_items(
        topo_backend_combo,
        [
            (_gmsh_label, "gmsh"),
            ("Structured (built-in fallback)", "structured"),
        ],
        default_data="gmsh",
    )
    _set_combo_items(
        topo_default_cell_type_combo,
        [
            ("triangular", "triangular"),
            ("quadrilateral", "quadrilateral"),
            ("cartesian", "cartesian"),
            ("empty", "empty"),
        ],
        default_data="triangular",
    )

    topo_default_size_spin.setRange(0.01, 1.0e6)
    topo_default_size_spin.setDecimals(3)
    topo_default_size_spin.setValue(20.0)

    # -- Gmsh controls form (finds existing widget or creates fallback) --
    gmsh_form_top = _find("topo_gmsh_controls_widget", QtWidgets.QWidget)
    if gmsh_form_top is None:
        gmsh_form_top = QtWidgets.QWidget()
        gmsh_form_top.setObjectName("topo_gmsh_controls_widget")
    gmsh_form = gmsh_form_top.layout()
    if not isinstance(gmsh_form, QtWidgets.QFormLayout):
        gmsh_form = QtWidgets.QFormLayout(gmsh_form_top)
    gmsh_form.setContentsMargins(0, 0, 0, 0)
    widgets["gmsh_form"] = gmsh_form

    # -- Quality controls form --
    quality_form_top = _find("topo_quality_controls_widget", QtWidgets.QWidget)
    if quality_form_top is None:
        quality_form_top = QtWidgets.QWidget()
        quality_form_top.setObjectName("topo_quality_controls_widget")
    quality_form = quality_form_top.layout()
    if not isinstance(quality_form, QtWidgets.QFormLayout):
        quality_form = QtWidgets.QFormLayout(quality_form_top)
    quality_form.setContentsMargins(0, 0, 0, 0)
    widgets["quality_form"] = quality_form

    # -- Gmsh algorithm combos --
    topo_gmsh_tri_algo_combo = QtWidgets.QComboBox()
    topo_gmsh_tri_algo_combo.setObjectName("topo_gmsh_tri_algo_combo")
    gmsh_form.addRow("Triangle algorithm:", topo_gmsh_tri_algo_combo)
    _set_combo_items(
        topo_gmsh_tri_algo_combo,
        [
            ("Frontal-Delaunay (quality)", 6),
            ("Delaunay (faster)", 5),
        ],
        default_data=6,
    )
    widgets["topo_gmsh_tri_algo_combo"] = topo_gmsh_tri_algo_combo

    topo_gmsh_quad_algo_combo = QtWidgets.QComboBox()
    topo_gmsh_quad_algo_combo.setObjectName("topo_gmsh_quad_algo_combo")
    gmsh_form.addRow("Quadrilateral algorithm:", topo_gmsh_quad_algo_combo)
    _set_combo_items(
        topo_gmsh_quad_algo_combo,
        [
            ("Frontal + Blossom recombine", 6),
            ("Delaunay + Blossom recombine", 5),
            ("Packing of Parallelograms", 9),
        ],
        default_data=6,
    )
    widgets["topo_gmsh_quad_algo_combo"] = topo_gmsh_quad_algo_combo

    topo_gmsh_recombine_algo_combo = QtWidgets.QComboBox()
    topo_gmsh_recombine_algo_combo.setObjectName("topo_gmsh_recombine_algo_combo")
    gmsh_form.addRow("Recombine algorithm:", topo_gmsh_recombine_algo_combo)
    _set_combo_items(
        topo_gmsh_recombine_algo_combo,
        [
            ("Simple", 0),
            ("Blossom", 1),
            ("Simple full-quad", 2),
        ],
        default_data=1,
    )
    widgets["topo_gmsh_recombine_algo_combo"] = topo_gmsh_recombine_algo_combo

    topo_gmsh_global_recombine_chk = QtWidgets.QCheckBox("Apply global recombine pass after mesh generation")
    topo_gmsh_global_recombine_chk.setObjectName("topo_gmsh_global_recombine_chk")
    if not str(topo_gmsh_global_recombine_chk.text() or "").strip():
        topo_gmsh_global_recombine_chk.setText("Apply global recombine pass after mesh generation")
    topo_gmsh_global_recombine_chk.setChecked(False)
    topo_gmsh_global_recombine_chk.setToolTip(
        "If enabled, runs gmsh.model.mesh.recombine() globally after mesh generation. "
        "Default is off to avoid recombining non-quad-targeted regions."
    )
    gmsh_form.addRow(topo_gmsh_global_recombine_chk)
    widgets["topo_gmsh_global_recombine_chk"] = topo_gmsh_global_recombine_chk

    topo_gmsh_quad_full_region_flow_align_chk = QtWidgets.QCheckBox("Gmsh full-region flow-aligned quads")
    topo_gmsh_quad_full_region_flow_align_chk.setObjectName("topo_gmsh_quad_full_region_flow_align_chk")
    if not str(topo_gmsh_quad_full_region_flow_align_chk.text() or "").strip():
        topo_gmsh_quad_full_region_flow_align_chk.setText("Gmsh full-region flow-aligned quads")
    topo_gmsh_quad_full_region_flow_align_chk.setChecked(True)
    topo_gmsh_quad_full_region_flow_align_chk.setToolTip(
        "For quadrilateral/channel-generator regions with complete quad-edge controls, "
        "apply TransfiniteCurve + TransfiniteSurface + Recombine so edge-aligned spacing "
        "propagates across the full region."
    )
    gmsh_form.addRow(topo_gmsh_quad_full_region_flow_align_chk)
    widgets["topo_gmsh_quad_full_region_flow_align_chk"] = topo_gmsh_quad_full_region_flow_align_chk

    topo_gmsh_smoothing_spin = QtWidgets.QSpinBox()
    topo_gmsh_smoothing_spin.setObjectName("topo_gmsh_smoothing_spin")
    topo_gmsh_smoothing_spin.setRange(0, 100)
    topo_gmsh_smoothing_spin.setValue(0)
    gmsh_form.addRow("Smoothing passes:", topo_gmsh_smoothing_spin)
    widgets["topo_gmsh_smoothing_spin"] = topo_gmsh_smoothing_spin

    topo_gmsh_optimize_iters_spin = QtWidgets.QSpinBox()
    topo_gmsh_optimize_iters_spin.setObjectName("topo_gmsh_optimize_iters_spin")
    topo_gmsh_optimize_iters_spin.setRange(0, 100)
    topo_gmsh_optimize_iters_spin.setValue(0)
    gmsh_form.addRow("Optimize iterations:", topo_gmsh_optimize_iters_spin)
    widgets["topo_gmsh_optimize_iters_spin"] = topo_gmsh_optimize_iters_spin

    topo_gmsh_verbosity_spin = QtWidgets.QSpinBox()
    topo_gmsh_verbosity_spin.setObjectName("topo_gmsh_verbosity_spin")
    topo_gmsh_verbosity_spin.setRange(0, 10)
    topo_gmsh_verbosity_spin.setValue(2)
    gmsh_form.addRow("Verbosity:", topo_gmsh_verbosity_spin)
    widgets["topo_gmsh_verbosity_spin"] = topo_gmsh_verbosity_spin

    topo_gmsh_optimize_netgen_chk = QtWidgets.QCheckBox("Enable Netgen optimize")
    topo_gmsh_optimize_netgen_chk.setObjectName("topo_gmsh_optimize_netgen_chk")
    if not str(topo_gmsh_optimize_netgen_chk.text() or "").strip():
        topo_gmsh_optimize_netgen_chk.setText("Enable Netgen optimize")
    gmsh_form.addRow(topo_gmsh_optimize_netgen_chk)
    widgets["topo_gmsh_optimize_netgen_chk"] = topo_gmsh_optimize_netgen_chk

    topo_gmsh_arc_mode_combo = QtWidgets.QComboBox()
    topo_gmsh_arc_mode_combo.setObjectName("topo_gmsh_arc_mode_combo")
    _set_combo_items(
        topo_gmsh_arc_mode_combo,
        [
            ("Hard embed arcs (strict)", "hard_embed"),
            ("Soft arc size hint (non-strict)", "soft_size_hint"),
            ("Disable arc influence", "disabled"),
        ],
        default_data="hard_embed",
    )
    widgets["topo_gmsh_arc_mode_combo"] = topo_gmsh_arc_mode_combo

    topo_gmsh_arc_soft_size_factor_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_arc_soft_size_factor_spin.setObjectName("topo_gmsh_arc_soft_size_factor_spin")
    topo_gmsh_arc_soft_size_factor_spin.setRange(0.05, 1.0)
    topo_gmsh_arc_soft_size_factor_spin.setDecimals(3)
    topo_gmsh_arc_soft_size_factor_spin.setSingleStep(0.05)
    topo_gmsh_arc_soft_size_factor_spin.setValue(0.5)
    topo_gmsh_arc_soft_size_factor_spin.setToolTip(
        "Soft arc mode target-size factor near arcs. Lower values force finer cells along arc corridors."
    )
    widgets["topo_gmsh_arc_soft_size_factor_spin"] = topo_gmsh_arc_soft_size_factor_spin

    topo_gmsh_arc_soft_dist_factor_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_arc_soft_dist_factor_spin.setObjectName("topo_gmsh_arc_soft_dist_factor_spin")
    topo_gmsh_arc_soft_dist_factor_spin.setRange(0.1, 10.0)
    topo_gmsh_arc_soft_dist_factor_spin.setDecimals(3)
    topo_gmsh_arc_soft_dist_factor_spin.setSingleStep(0.1)
    topo_gmsh_arc_soft_dist_factor_spin.setValue(2.0)
    topo_gmsh_arc_soft_dist_factor_spin.setToolTip(
        "Soft arc mode influence distance factor. Higher values widen arc-driven refinement corridors."
    )
    widgets["topo_gmsh_arc_soft_dist_factor_spin"] = topo_gmsh_arc_soft_dist_factor_spin

    topo_gmsh_interface_transition_enable_chk = QtWidgets.QCheckBox("Enable interface transition grading")
    topo_gmsh_interface_transition_enable_chk.setObjectName("topo_gmsh_interface_transition_enable_chk")
    if not str(topo_gmsh_interface_transition_enable_chk.text() or "").strip():
        topo_gmsh_interface_transition_enable_chk.setText("Enable interface transition grading")
    topo_gmsh_interface_transition_enable_chk.setChecked(True)
    topo_gmsh_interface_transition_enable_chk.setToolTip(
        "Apply Distance/Threshold grading near shared interfaces on non-transfinite regions only."
    )
    widgets["topo_gmsh_interface_transition_enable_chk"] = topo_gmsh_interface_transition_enable_chk

    topo_gmsh_interface_transition_dist_factor_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_interface_transition_dist_factor_spin.setObjectName("topo_gmsh_interface_transition_dist_factor_spin")
    topo_gmsh_interface_transition_dist_factor_spin.setRange(0.25, 20.0)
    topo_gmsh_interface_transition_dist_factor_spin.setDecimals(3)
    topo_gmsh_interface_transition_dist_factor_spin.setSingleStep(0.25)
    topo_gmsh_interface_transition_dist_factor_spin.setValue(2.5)
    topo_gmsh_interface_transition_dist_factor_spin.setToolTip(
        "Distance multiplier for interface grading influence width. Higher values widen the transition band."
    )
    widgets["topo_gmsh_interface_transition_dist_factor_spin"] = topo_gmsh_interface_transition_dist_factor_spin

    topo_gmsh_interface_transition_min_ratio_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_interface_transition_min_ratio_spin.setObjectName("topo_gmsh_interface_transition_min_ratio_spin")
    topo_gmsh_interface_transition_min_ratio_spin.setRange(1.0, 10.0)
    topo_gmsh_interface_transition_min_ratio_spin.setDecimals(3)
    topo_gmsh_interface_transition_min_ratio_spin.setSingleStep(0.05)
    topo_gmsh_interface_transition_min_ratio_spin.setValue(1.25)
    topo_gmsh_interface_transition_min_ratio_spin.setToolTip(
        "Only apply interface grading when adjacent region target sizes differ by at least this ratio."
    )
    widgets["topo_gmsh_interface_transition_min_ratio_spin"] = topo_gmsh_interface_transition_min_ratio_spin

    topo_gmsh_interface_conformance_chk = QtWidgets.QCheckBox("Enable transverse interface conformance post-process")
    topo_gmsh_interface_conformance_chk.setObjectName("topo_gmsh_interface_conformance_chk")
    if not str(topo_gmsh_interface_conformance_chk.text() or "").strip():
        topo_gmsh_interface_conformance_chk.setText("Enable transverse interface conformance post-process")
    topo_gmsh_interface_conformance_chk.setChecked(False)
    topo_gmsh_interface_conformance_chk.setToolTip(
        "Snap and weld mixed-interface nodes after Gmsh extraction to enforce shared boundary topology."
    )
    widgets["topo_gmsh_interface_conformance_chk"] = topo_gmsh_interface_conformance_chk

    topo_gmsh_transverse_interface_centroid_merge_chk = QtWidgets.QCheckBox("Use centroid merge for matched transverse interface nodes")
    topo_gmsh_transverse_interface_centroid_merge_chk.setObjectName("topo_gmsh_transverse_interface_centroid_merge_chk")
    if not str(topo_gmsh_transverse_interface_centroid_merge_chk.text() or "").strip():
        topo_gmsh_transverse_interface_centroid_merge_chk.setText("Use centroid merge for matched transverse interface nodes")
    topo_gmsh_transverse_interface_centroid_merge_chk.setChecked(False)
    topo_gmsh_transverse_interface_centroid_merge_chk.setToolTip(
        "Move matched interface-node groups to their centroid before welding instead of one-sided snapping."
    )
    widgets["topo_gmsh_transverse_interface_centroid_merge_chk"] = topo_gmsh_transverse_interface_centroid_merge_chk

    topo_gmsh_interface_snap_tol_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_interface_snap_tol_spin.setObjectName("topo_gmsh_interface_snap_tol_spin")
    topo_gmsh_interface_snap_tol_spin.setRange(1.0e-6, 1.0e5)
    topo_gmsh_interface_snap_tol_spin.setDecimals(6)
    topo_gmsh_interface_snap_tol_spin.setValue(1.0)
    topo_gmsh_interface_snap_tol_spin.setToolTip(
        "Distance tolerance used by transverse interface conformance snapping."
    )
    widgets["topo_gmsh_interface_snap_tol_spin"] = topo_gmsh_interface_snap_tol_spin

    topo_gmsh_interface_reject_near_unshared_chk = QtWidgets.QCheckBox("Reject mixed interfaces with near-coincident unshared nodes")
    topo_gmsh_interface_reject_near_unshared_chk.setObjectName("topo_gmsh_interface_reject_near_unshared_chk")
    if not str(topo_gmsh_interface_reject_near_unshared_chk.text() or "").strip():
        topo_gmsh_interface_reject_near_unshared_chk.setText("Reject mixed interfaces with near-coincident unshared nodes")
    topo_gmsh_interface_reject_near_unshared_chk.setChecked(True)
    topo_gmsh_interface_reject_near_unshared_chk.setToolTip(
        "Fail meshing when a transfinite/tri interface shows hanging-node style near-miss pairs."
    )
    widgets["topo_gmsh_interface_reject_near_unshared_chk"] = topo_gmsh_interface_reject_near_unshared_chk

    topo_gmsh_interface_reject_tol_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_interface_reject_tol_spin.setObjectName("topo_gmsh_interface_reject_tol_spin")
    topo_gmsh_interface_reject_tol_spin.setRange(1.0e-6, 1.0e3)
    topo_gmsh_interface_reject_tol_spin.setDecimals(6)
    topo_gmsh_interface_reject_tol_spin.setValue(1.0e-3)
    topo_gmsh_interface_reject_tol_spin.setToolTip(
        "Tolerance for detecting near-coincident unshared interface nodes (hanging-node signature)."
    )
    widgets["topo_gmsh_interface_reject_tol_spin"] = topo_gmsh_interface_reject_tol_spin

    topo_gmsh_mesh_size_min_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_mesh_size_min_spin.setObjectName("topo_gmsh_mesh_size_min_spin")
    topo_gmsh_mesh_size_min_spin.setRange(0.0, 1.0e6)
    topo_gmsh_mesh_size_min_spin.setDecimals(6)
    topo_gmsh_mesh_size_min_spin.setValue(0.0)
    gmsh_form.addRow("Global min cell size:", topo_gmsh_mesh_size_min_spin)
    widgets["topo_gmsh_mesh_size_min_spin"] = topo_gmsh_mesh_size_min_spin

    topo_gmsh_tolerance_edge_length_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_tolerance_edge_length_spin.setObjectName("topo_gmsh_tolerance_edge_length_spin")
    topo_gmsh_tolerance_edge_length_spin.setRange(0.0, 1.0e6)
    topo_gmsh_tolerance_edge_length_spin.setDecimals(6)
    topo_gmsh_tolerance_edge_length_spin.setValue(0.0)
    gmsh_form.addRow("Ignore edges shorter than:", topo_gmsh_tolerance_edge_length_spin)
    widgets["topo_gmsh_tolerance_edge_length_spin"] = topo_gmsh_tolerance_edge_length_spin

    topo_gmsh_mesh_size_from_points_chk = QtWidgets.QCheckBox("Use region target_size for mesh sizing")
    topo_gmsh_mesh_size_from_points_chk.setObjectName("topo_gmsh_mesh_size_from_points_chk")
    if not str(topo_gmsh_mesh_size_from_points_chk.text() or "").strip():
        topo_gmsh_mesh_size_from_points_chk.setText("Use region target_size for mesh sizing")
    topo_gmsh_mesh_size_from_points_chk.setChecked(True)
    gmsh_form.addRow(topo_gmsh_mesh_size_from_points_chk)
    widgets["topo_gmsh_mesh_size_from_points_chk"] = topo_gmsh_mesh_size_from_points_chk

    topo_gmsh_quality_enable_chk = QtWidgets.QCheckBox("Enable Gmsh iterative quality loop")
    topo_gmsh_quality_enable_chk.setObjectName("topo_gmsh_quality_enable_chk")
    if not str(topo_gmsh_quality_enable_chk.text() or "").strip():
        topo_gmsh_quality_enable_chk.setText("Enable Gmsh iterative quality loop")
    topo_gmsh_quality_enable_chk.setChecked(False)
    widgets["topo_gmsh_quality_enable_chk"] = topo_gmsh_quality_enable_chk

    topo_gmsh_quality_max_iters_spin = QtWidgets.QSpinBox()
    topo_gmsh_quality_max_iters_spin.setObjectName("topo_gmsh_quality_max_iters_spin")
    topo_gmsh_quality_max_iters_spin.setRange(1, 50)
    topo_gmsh_quality_max_iters_spin.setValue(2)
    widgets["topo_gmsh_quality_max_iters_spin"] = topo_gmsh_quality_max_iters_spin

    topo_gmsh_quality_time_limit_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_quality_time_limit_spin.setObjectName("topo_gmsh_quality_time_limit_spin")
    topo_gmsh_quality_time_limit_spin.setRange(1.0, 3600.0)
    topo_gmsh_quality_time_limit_spin.setDecimals(1)
    topo_gmsh_quality_time_limit_spin.setValue(55.0)
    widgets["topo_gmsh_quality_time_limit_spin"] = topo_gmsh_quality_time_limit_spin

    topo_quality_min_angle_spin = QtWidgets.QDoubleSpinBox()
    topo_quality_min_angle_spin.setObjectName("topo_quality_min_angle_spin")
    topo_quality_min_angle_spin.setRange(0.0, 89.0)
    topo_quality_min_angle_spin.setDecimals(1)
    topo_quality_min_angle_spin.setValue(5.0)
    widgets["topo_quality_min_angle_spin"] = topo_quality_min_angle_spin

    topo_quality_max_aspect_spin = QtWidgets.QDoubleSpinBox()
    topo_quality_max_aspect_spin.setObjectName("topo_quality_max_aspect_spin")
    topo_quality_max_aspect_spin.setRange(1.0, 1.0e4)
    topo_quality_max_aspect_spin.setDecimals(2)
    topo_quality_max_aspect_spin.setValue(20.0)
    widgets["topo_quality_max_aspect_spin"] = topo_quality_max_aspect_spin

    topo_quality_max_non_orth_spin = QtWidgets.QDoubleSpinBox()
    topo_quality_max_non_orth_spin.setObjectName("topo_quality_max_non_orth_spin")
    topo_quality_max_non_orth_spin.setRange(1.0, 89.9)
    topo_quality_max_non_orth_spin.setDecimals(1)
    topo_quality_max_non_orth_spin.setValue(82.0)
    widgets["topo_quality_max_non_orth_spin"] = topo_quality_max_non_orth_spin

    topo_quality_min_area_edit = QtWidgets.QLineEdit("1e-14")
    topo_quality_min_area_edit.setObjectName("topo_quality_min_area_edit")
    if not str(topo_quality_min_area_edit.text() or "").strip():
        topo_quality_min_area_edit.setText("1e-14")
    widgets["topo_quality_min_area_edit"] = topo_quality_min_area_edit

    topo_quality_size_scales_edit = QtWidgets.QLineEdit("1.0,0.9,0.8,0.7")
    topo_quality_size_scales_edit.setObjectName("topo_quality_size_scales_edit")
    if not str(topo_quality_size_scales_edit.text() or "").strip():
        topo_quality_size_scales_edit.setText("1.0,0.9,0.8,0.7")
    topo_quality_size_scales_edit.setToolTip(
        "Comma-separated per-attempt size multipliers. Example: 1.0,0.9,0.8"
    )
    widgets["topo_quality_size_scales_edit"] = topo_quality_size_scales_edit

    topo_quality_smooth_increments_edit = QtWidgets.QLineEdit("0,2,4,6")
    topo_quality_smooth_increments_edit.setObjectName("topo_quality_smooth_increments_edit")
    if not str(topo_quality_smooth_increments_edit.text() or "").strip():
        topo_quality_smooth_increments_edit.setText("0,2,4,6")
    topo_quality_smooth_increments_edit.setToolTip(
        "Comma-separated extra smoothing passes added per attempt. Example: 0,2,4,6"
    )
    widgets["topo_quality_smooth_increments_edit"] = topo_quality_smooth_increments_edit

    topo_gmsh_quality_recombine_topology_passes_edit = QtWidgets.QLineEdit("5,12,20")
    topo_gmsh_quality_recombine_topology_passes_edit.setObjectName("topo_gmsh_quality_recombine_topology_passes_edit")
    if not str(topo_gmsh_quality_recombine_topology_passes_edit.text() or "").strip():
        topo_gmsh_quality_recombine_topology_passes_edit.setText("5,12,20")
    topo_gmsh_quality_recombine_topology_passes_edit.setToolTip(
        "Comma-separated topological optimization passes for quad recombination per attempt. "
        "Higher values can improve quad layout but cost more runtime."
    )
    widgets["topo_gmsh_quality_recombine_topology_passes_edit"] = topo_gmsh_quality_recombine_topology_passes_edit

    topo_gmsh_quality_recombine_min_quality_edit = QtWidgets.QLineEdit("0.01,0.03,0.06")
    topo_gmsh_quality_recombine_min_quality_edit.setObjectName("topo_gmsh_quality_recombine_min_quality_edit")
    if not str(topo_gmsh_quality_recombine_min_quality_edit.text() or "").strip():
        topo_gmsh_quality_recombine_min_quality_edit.setText("0.01,0.03,0.06")
    topo_gmsh_quality_recombine_min_quality_edit.setToolTip(
        "Comma-separated minimum acceptable recombined quad quality per attempt. "
        "Typical range: 0.0 to 0.2."
    )
    widgets["topo_gmsh_quality_recombine_min_quality_edit"] = topo_gmsh_quality_recombine_min_quality_edit

    topo_gmsh_quality_random_factors_edit = QtWidgets.QLineEdit("1e-9,1e-7,1e-6")
    topo_gmsh_quality_random_factors_edit.setObjectName("topo_gmsh_quality_random_factors_edit")
    if not str(topo_gmsh_quality_random_factors_edit.text() or "").strip():
        topo_gmsh_quality_random_factors_edit.setText("1e-9,1e-7,1e-6")
    topo_gmsh_quality_random_factors_edit.setToolTip(
        "Comma-separated Mesh.RandomFactor values per attempt. "
        "Use small positive values to perturb deterministic local minima."
    )
    widgets["topo_gmsh_quality_random_factors_edit"] = topo_gmsh_quality_random_factors_edit

    topo_gmsh_quality_optimize_methods_edit = QtWidgets.QLineEdit("Laplace2D,Relocate2D")
    topo_gmsh_quality_optimize_methods_edit.setObjectName("topo_gmsh_quality_optimize_methods_edit")
    if not str(topo_gmsh_quality_optimize_methods_edit.text() or "").strip():
        topo_gmsh_quality_optimize_methods_edit.setText("Laplace2D,Relocate2D")
    topo_gmsh_quality_optimize_methods_edit.setToolTip(
        "Comma-separated gmsh.model.mesh.optimize methods applied each attempt. "
        "Example: Laplace2D,Relocate2D"
    )
    widgets["topo_gmsh_quality_optimize_methods_edit"] = topo_gmsh_quality_optimize_methods_edit

    topo_gmsh_algo_switch_on_failure_chk = QtWidgets.QCheckBox("Gmsh algorithm switch on failure")
    topo_gmsh_algo_switch_on_failure_chk.setObjectName("topo_gmsh_algo_switch_on_failure_chk")
    if not str(topo_gmsh_algo_switch_on_failure_chk.text() or "").strip():
        topo_gmsh_algo_switch_on_failure_chk.setText("Gmsh algorithm switch on failure")
    topo_gmsh_algo_switch_on_failure_chk.setChecked(False)
    topo_gmsh_algo_switch_on_failure_chk.setToolTip(
        "Enable Mesh.AlgorithmSwitchOnFailure. Gmsh may switch 2D algorithms (e.g. to MeshAdapt) on failure."
    )
    widgets["topo_gmsh_algo_switch_on_failure_chk"] = topo_gmsh_algo_switch_on_failure_chk

    topo_gmsh_recombine_node_repositioning_chk = QtWidgets.QCheckBox("Allow recombine node repositioning")
    topo_gmsh_recombine_node_repositioning_chk.setObjectName("topo_gmsh_recombine_node_repositioning_chk")
    if not str(topo_gmsh_recombine_node_repositioning_chk.text() or "").strip():
        topo_gmsh_recombine_node_repositioning_chk.setText("Allow recombine node repositioning")
    topo_gmsh_recombine_node_repositioning_chk.setChecked(True)
    topo_gmsh_recombine_node_repositioning_chk.setToolTip(
        "Enable node repositioning during quad recombination (Mesh.RecombineNodeRepositioning)."
    )
    widgets["topo_gmsh_recombine_node_repositioning_chk"] = topo_gmsh_recombine_node_repositioning_chk

    topo_quality_strict_chk = QtWidgets.QCheckBox("Strict quality acceptance")
    topo_quality_strict_chk.setObjectName("topo_quality_strict_chk")
    if not str(topo_quality_strict_chk.text() or "").strip():
        topo_quality_strict_chk.setText("Strict quality acceptance")
    widgets["topo_quality_strict_chk"] = topo_quality_strict_chk
    quality_form.addRow("Min angle (deg):", topo_quality_min_angle_spin)
    quality_form.addRow("Max aspect ratio:", topo_quality_max_aspect_spin)
    quality_form.addRow("Max non-orthogonal (deg):", topo_quality_max_non_orth_spin)
    quality_form.addRow("Min area rel bbox:", topo_quality_min_area_edit)
    quality_form.addRow(topo_quality_strict_chk)
    quality_form.addRow("Size scales:", topo_quality_size_scales_edit)
    quality_form.addRow("Smooth increments:", topo_quality_smooth_increments_edit)
    quality_form.addRow("Recombine passes:", topo_gmsh_quality_recombine_topology_passes_edit)
    quality_form.addRow("Recombine min quality:", topo_gmsh_quality_recombine_min_quality_edit)
    quality_form.addRow("Random factors:", topo_gmsh_quality_random_factors_edit)
    quality_form.addRow("Optimize methods:", topo_gmsh_quality_optimize_methods_edit)
    quality_form.addRow(topo_gmsh_algo_switch_on_failure_chk)
    quality_form.addRow(topo_gmsh_recombine_node_repositioning_chk)
    quality_form.addRow(topo_gmsh_quality_enable_chk)
    quality_form.addRow("Max iterations:", topo_gmsh_quality_max_iters_spin)
    quality_form.addRow("Time limit (s):", topo_gmsh_quality_time_limit_spin)

    widgets["gmsh_form"] = gmsh_form
    widgets["quality_form"] = quality_form
    widgets["gmsh_form_top"] = gmsh_form_top
    widgets["quality_form_top"] = quality_form_top
    return widgets


def _wire_topology_tab_controls(
    widgets,
    gmsh_form,
    quality_form,
    update_summary_fn,
) -> None:
    """Controller: connect topology tab widget signals."""
    if not widgets:
        return

    def _alive(w):
        """Check if a widget reference is still alive."""
        try:
            _ = w.objectName()
            return True
        except RuntimeError:
            return False

    def _w(sig, handler):
        """Safely disconnect then connect a signal to a handler."""
        try:
            sig.disconnect(handler)
        except (TypeError, RuntimeError):
            pass
        try:
            sig.connect(handler)
        except RuntimeError:
            pass

    # Connect backend/region/constraint/quad-edges combos
    for combo_key in ("topo_backend_combo", "topo_regions_combo",
                       "topo_constraints_combo", "topo_quad_edges_combo"):
        w = widgets.get(combo_key)
        if w is not None and _alive(w) and hasattr(w, "currentIndexChanged"):
            _w(w.currentIndexChanged, update_summary_fn)

    # Connect quality widgets
    for qw_name in ("topo_quality_min_angle_spin", "topo_quality_max_aspect_spin",
                     "topo_quality_max_non_orth_spin"):
        w = widgets.get(qw_name)
        if w is not None and _alive(w) and hasattr(w, "valueChanged"):
            _w(w.valueChanged, update_summary_fn)

    for qw_name in ("topo_quality_min_area_edit", "topo_quality_size_scales_edit",
                     "topo_quality_smooth_increments_edit",
                     "topo_gmsh_quality_recombine_topology_passes_edit",
                     "topo_gmsh_quality_recombine_min_quality_edit",
                     "topo_gmsh_quality_random_factors_edit",
                     "topo_gmsh_quality_optimize_methods_edit"):
        w = widgets.get(qw_name)
        if w is not None and _alive(w) and hasattr(w, "textChanged"):
            _w(w.textChanged, update_summary_fn)

    for chk_name in ("topo_quality_strict_chk", "topo_gmsh_quad_full_region_flow_align_chk",
                      "topo_gmsh_algo_switch_on_failure_chk",
                      "topo_gmsh_recombine_node_repositioning_chk",
                      "topo_gmsh_global_recombine_chk",
                      "topo_gmsh_interface_transition_enable_chk",
                      "topo_gmsh_mesh_size_from_points_chk",
                      "topo_gmsh_quality_enable_chk"):
        w = widgets.get(chk_name)
        if w is not None and _alive(w) and hasattr(w, "toggled"):
            _w(w.toggled, update_summary_fn)

    for combo_name in ("topo_gmsh_arc_mode_combo",):
        w = widgets.get(combo_name)
        if w is not None and _alive(w) and hasattr(w, "currentIndexChanged"):
            _w(w.currentIndexChanged, update_summary_fn)

    for spin_name in ("topo_gmsh_arc_soft_size_factor_spin", "topo_gmsh_arc_soft_dist_factor_spin",
                       "topo_gmsh_interface_transition_dist_factor_spin",
                       "topo_gmsh_interface_transition_min_ratio_spin",
                        "topo_gmsh_mesh_size_min_spin", "topo_gmsh_tolerance_edge_length_spin",
                        "topo_gmsh_quality_max_iters_spin", "topo_gmsh_quality_time_limit_spin"):
        w = widgets.get(spin_name)
        if w is not None and _alive(w) and hasattr(w, "valueChanged"):
            _w(w.valueChanged, update_summary_fn)

    update_summary_fn()



