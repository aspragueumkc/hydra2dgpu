"""Topology (Topo Mesh) tab view — owns its own widget references.

QWidget subclass for the Topo Mesh tab in the Studio workbench.
Mirrors the structure of _build_topology_tab_page_fallback in studio_dialog.py
so the dialog can delegate tab construction to this view.

UI structure (matches ``ModelTabView``):

* A ``QToolBox`` whose pages correspond to functional groups
  (Layer Setup, General, Algorithm, Arcs & Interfaces, Sizing,
  Threading, Transfinite, Quality).
* Each page hosts one or more ``QGroupBox`` sections; each section
  uses a ``QFormLayout`` for its labeled rows.
* A free-text filter (``topo_search``) and an
  "Show advanced parameters" toggle (``topo_show_advanced_chk``)
  hide/show rows via :class:`FilterableRowRegistry`, mirroring
  ``ModelTabView`` so the topology tab behaves like the simulation tab.
"""
from __future__ import annotations

from typing import List

from qgis.PyQt import QtWidgets

from swe2d.workbench.views.widget_filter_helper import FilterableRowRegistry


class TopologyTabView(QtWidgets.QWidget):
    """View for the Topo Mesh tab.

    Creates and owns:
    - Layer Setup page: topo_nodes_combo, topo_arcs_combo,
      topo_regions_combo, topo_constraints_combo, topo_quad_edges_combo,
      topo_elevation_combo
    - General page: topo_backend_combo, topo_default_size_spin,
      topo_default_cell_type_combo, topo_generate_btn, topo_terminate_btn
    - Algorithm page (gmsh): tri/quad/recombine algos, smoothing,
      optimize, verbosity, netgen, global_recombine, flow_align,
      algo_switch_on_failure, recombine_node_repositioning
    - Arcs & Interfaces page (gmsh): arc mode, soft size/dist,
      interface transition, conformance, snap tol, reject controls
    - Sizing page (gmsh): mesh_size_min, tolerance_edge_length,
      mesh_size_from_points
    - Threading page (gmsh): num_threads, max_num_threads_2d
    - Transfinite page (gmsh): harmonize, subset containment, debug
    - Quality page (gmsh): thresholds, retry ladder, loop controls
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

        # ── Filter registry (matches ModelTabView) ─────────────────────
        # Every widget that should respond to topo_search /
        # topo_show_advanced_chk is registered via _add_topo_param_row
        # or directly via _filterable.add(...).
        self._filterable: FilterableRowRegistry = FilterableRowRegistry()
        self._topo_param_groups: List[QtWidgets.QGroupBox] = []

        filter_bar = QtWidgets.QHBoxLayout()
        self.topo_search = QtWidgets.QLineEdit()
        self.topo_search.setObjectName("topo_search")
        self.topo_search.setPlaceholderText("Filter parameters…")
        self.topo_search.textChanged.connect(self._filter_topology_tab)
        self.topo_show_advanced_chk = QtWidgets.QCheckBox("Show advanced parameters")
        self.topo_show_advanced_chk.setObjectName("topo_show_advanced_chk")
        self.topo_show_advanced_chk.setChecked(False)
        self.topo_show_advanced_chk.toggled.connect(self._filter_topology_tab)
        filter_bar.addWidget(self.topo_search, 1)
        filter_bar.addWidget(self.topo_show_advanced_chk)
        root_layout.addLayout(filter_bar)

        self._toolbox = QtWidgets.QToolBox()
        self._toolbox.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        root_layout.addWidget(self._toolbox)

        # ── Layer Setup page ──────────────────────────────────────────
        layer_page, layer_form = self._create_topo_page(
            "topo_layer_page", "topo_layer_form", "Layers"
        )

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
            self._add_topo_param_row(layer_form, lbl_text, combo)

        self.topo_nodes_combo.setToolTip(
            "Point layer containing mesh node coordinates for the topology."
        )
        self.topo_arcs_combo.setToolTip(
            "Line layer defining topological arcs between nodes."
        )
        self.topo_regions_combo.setToolTip(
            "Polygon layer defining mesh regions. "
            "Each region can have a cell_type and target_size for mesh generation."
        )
        self.topo_constraints_combo.setToolTip(
            "Optional polygon layer for mesh refinement constraints. Use '(none)' to disable."
        )
        self.topo_quad_edges_combo.setToolTip(
            "Optional layer for quad-edge transition controls "
            "used in structured mesh regions."
        )

        # ── Elevation source ───────────────────────────────────────────
        # Auto-assign node_z from a single-band raster or PointZ layer when
        # mesh is generated. Selecting "(none)" disables auto-assignment.
        self.topo_elevation_combo = QtWidgets.QComboBox()
        self.topo_elevation_combo.setObjectName("topo_elevation_combo")
        self.topo_elevation_combo.setToolTip(
            "Optional elevation source. If selected, mesh node elevations "
            "(node_z) are auto-populated when 'Generate Mesh' runs:\n"
            "  - Single-band raster: sampled at mesh node coordinates.\n"
            "  - PointZ layer: IDW (inverse-distance-weighted) interpolation "
            "    using the 4 nearest source points."
        )
        self._add_topo_param_row(layer_form, "Elevation source:", self.topo_elevation_combo)

        # -- Import/Export page (top of toolbox, BEFORE Layer Setup) --
        # Moved from MapTabView's "Mesh Setup" page. Widgets retain the
        # same objectNames so existing signal-wiring code keeps working.
        self._build_import_export_page()

        self._toolbox.addItem(layer_page, "Layer Setup")

        # ── General page ─────────────────────────────────────────────
        general_page, general_form = self._create_topo_page(
            "topo_general_page", "topo_general_form", "General"
        )

        self.topo_backend_combo = QtWidgets.QComboBox()
        self.topo_backend_combo.setObjectName("topo_backend_combo")
        self.topo_backend_combo.setToolTip(
            "Meshing engine: Gmsh (recommended) or built-in structured fallback."
        )
        self.topo_default_size_spin = QtWidgets.QDoubleSpinBox()
        self.topo_default_size_spin.setObjectName("topo_default_size_spin")
        self.topo_default_size_spin.setToolTip(
            "Default target element size in model units. "
            "Overridden by per-region target_size if set."
        )
        self.topo_default_cell_type_combo = QtWidgets.QComboBox()
        self.topo_default_cell_type_combo.setObjectName("topo_default_cell_type_combo")
        self.topo_default_cell_type_combo.setToolTip(
            "Default cell type for all regions: triangular, quadrilateral, "
            "cartesian, or empty (void)."
        )

        for widget, label in [
            (self.topo_backend_combo, "Meshing backend:"),
            (self.topo_default_size_spin, "Default target size:"),
            (self.topo_default_cell_type_combo, "Default cell type:"),
        ]:
            self._add_topo_param_row(general_form, label, widget)

        self.topo_generate_btn = QtWidgets.QPushButton("Generate Mesh")
        self.topo_generate_btn.setObjectName("topo_generate_btn")
        self.topo_generate_btn.setToolTip(
            "Start mesh generation with the configured topology and parameters."
        )
        self.topo_generate_btn.setEnabled(True)
        self._add_topo_self_describing_row(
            general_form, self.topo_generate_btn, label_text="Generate Mesh"
        )

        self.topo_terminate_btn = QtWidgets.QPushButton("Terminate")
        self.topo_terminate_btn.setObjectName("topo_terminate_btn")
        self.topo_terminate_btn.setToolTip(
            "Request cancellation of an in-progress mesh generation."
        )
        self.topo_terminate_btn.setEnabled(False)
        self._add_topo_self_describing_row(
            general_form, self.topo_terminate_btn, label_text="Terminate"
        )

        self.topo_status_lbl = QtWidgets.QLabel(
            "Select regions layer and generate face-centric mesh"
        )
        self.topo_status_lbl.setObjectName("topo_status_lbl")
        self.topo_status_lbl.setWordWrap(True)
        general_form.addRow(self.topo_status_lbl)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setObjectName("progress_bar")
        self.progress_bar.setToolTip("Mesh generation progress indicator.")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        general_form.addRow(self.progress_bar)

        self._general_idx = self._toolbox.addItem(general_page, "General")

        # ── Algorithm page ──────────────────────────────────────────
        algo_page, self.topo_algo_form = self._create_topo_page(
            "topo_algo_page", "topo_algo_form", "Algorithm"
        )
        self._algo_idx = self._toolbox.addItem(algo_page, "Algorithm")
        self._toolbox.setItemEnabled(self._algo_idx, False)

        # ── Arcs & Interfaces page ─────────────────────────────────
        arcs_page, self.topo_arcs_form = self._create_topo_page(
            "topo_arcs_page", "topo_arcs_form", "Arcs and Interfaces"
        )
        self._arcs_idx = self._toolbox.addItem(arcs_page, "Arcs and Interfaces")
        self._toolbox.setItemEnabled(self._arcs_idx, False)

        # ── Sizing page ───────────────────────────────────────────
        sizing_page, self.topo_sizing_form = self._create_topo_page(
            "topo_sizing_page", "topo_sizing_form", "Sizing"
        )
        self._sizing_idx = self._toolbox.addItem(sizing_page, "Sizing")
        self._toolbox.setItemEnabled(self._sizing_idx, False)

        # ── Threading page ───────────────────────────────────────
        threading_page, self.topo_threading_form = self._create_topo_page(
            "topo_threading_page", "topo_threading_form", "Threading"
        )
        self._threading_idx = self._toolbox.addItem(threading_page, "Threading")
        self._toolbox.setItemEnabled(self._threading_idx, False)

        # ── Transfinite page ───────────────────────────────────
        transfinite_page, self.topo_transfinite_form = self._create_topo_page(
            "topo_transfinite_page", "topo_transfinite_form", "Transfinite"
        )
        self._transfinite_idx = self._toolbox.addItem(transfinite_page, "Transfinite")
        self._toolbox.setItemEnabled(self._transfinite_idx, False)

        # ── Quality page ───────────────────────────────────────
        quality_page, self.topo_quality_form = self._create_topo_page(
            "topo_quality_page", "topo_quality_form", "Quality"
        )
        self._quality_idx = self._toolbox.addItem(quality_page, "Quality")
        self._toolbox.setItemEnabled(self._quality_idx, False)

        self._gmsh_only_indices = (
            self._algo_idx, self._arcs_idx, self._sizing_idx,
            self._threading_idx, self._transfinite_idx, self._quality_idx,
        )
        self._gmsh_only_base_titles = {
            self._algo_idx: "Algorithm",
            self._arcs_idx: "Arcs and Interfaces",
            self._sizing_idx: "Sizing",
            self._threading_idx: "Threading",
            self._transfinite_idx: "Transfinite",
            self._quality_idx: "Quality",
        }

        self._populate_gmsh_quality_controls()

    # ------------------------------------------------------------------
    # Import/Export page (moved from MapTabView's "Mesh Setup" page)
    # ------------------------------------------------------------------

    def _build_import_export_page(self) -> None:
        """Build the Import/Export page with two combo boxes + Run buttons.

        Replaces the previous 6-button stack with two QComboBox
        selectors ("Import" and "Export"), each backed by a single
        "Run" button. Selecting an action in the combo and clicking
        Run fires the same ``QPushButton.click()`` slot that the old
        stacked buttons did, so the existing signal wiring in
        ``wire_topology_tab_static_signals`` keeps working unchanged.

        The 6 underlying QPushButton objects are still created as
        instance attributes with their original ``objectName``s — they
        just aren't laid out in the form. Tests and the existing
        mesh-I/O wiring (``btn.clicked.connect(cb)``) both find them
        by name.

        Layout (top to bottom):

            [ Import combo     ] [ Run Import ]
            [ Export combo     ] [ Run Export ]
        """
        page = QtWidgets.QWidget()
        page.setObjectName("topo_import_export_page")
        layout = QtWidgets.QFormLayout(page)
        layout.setObjectName("topo_import_export_form")
        layout.setContentsMargins(4, 4, 4, 4)

        # ── Build the 6 buttons (kept as instance attributes for wiring) ──
        btn_specs = [
            ("export_mesh_layers_btn", "Export Mesh To Map Layers"),
            ("export_mesh_ugrid_btn", "Export Mesh To UGRID"),
            ("save_mesh_gpkg_btn", "Save Mesh to GPKG"),
            ("import_mesh_layers_btn", "Load Mesh From Selected Layers"),
            ("export_results_ugrid_btn", "Export Results to UGRID"),
            ("load_mesh_gpkg_btn", "Load Mesh from GPKG..."),
        ]
        for attr, text in btn_specs:
            btn = QtWidgets.QPushButton(text)
            btn.setObjectName(attr)
            setattr(self, attr, btn)

        # ── Tooltips (unchanged from the previous stacked-button layout) ──
        self.export_mesh_layers_btn.setToolTip(
            "Export the current in-memory mesh (nodes + cells) as QGIS map layers. "
            "Creates point and polygon layers in the project for inspection."
        )
        self.import_mesh_layers_btn.setToolTip(
            "Build an in-memory mesh from the currently selected nodes and cells map layers. "
            "Use after editing layer geometry or node elevations externally. "
            "If the topology elevation source combo has a layer selected, "
            "node_z is auto-populated during import."
        )
        self.export_results_ugrid_btn.setToolTip(
            "Export simulation results to UGRID NetCDF format for external visualization."
        )
        self.export_mesh_ugrid_btn.setToolTip(
            "Export the current in-memory mesh geometry to UGRID NetCDF format."
        )
        self.save_mesh_gpkg_btn.setToolTip(
            "Save current mesh to a GeoPackage file."
        )
        self.load_mesh_gpkg_btn.setToolTip(
            "Open a GeoPackage and load a mesh from it."
        )

        # ── Import combo + Run Import button ─────────────────────────────
        # Each item's userData is the corresponding QPushButton. Picking
        # an item + clicking Run fires that button's clicked signal,
        # which the existing controller wiring already handles.
        self.import_combo = QtWidgets.QComboBox()
        self.import_combo.setObjectName("import_combo")
        import_options = [
            ("Load Mesh From Selected Layers", self.import_mesh_layers_btn),
            ("Load Mesh from GPKG...", self.load_mesh_gpkg_btn),
        ]
        for label, btn in import_options:
            self.import_combo.addItem(label, btn)
        self.import_combo.setToolTip(
            "Pick an import action, then click 'Run Import' to execute it."
        )

        self.run_import_btn = QtWidgets.QPushButton("Run Import")
        self.run_import_btn.setObjectName("run_import_btn")
        self.run_import_btn.setToolTip(
            "Execute the import action selected in the Import combo."
        )
        self.run_import_btn.clicked.connect(self._run_selected_import)

        import_row = QtWidgets.QHBoxLayout()
        import_row.setContentsMargins(0, 0, 0, 0)
        import_row.addWidget(self.import_combo, 1)
        import_row.addWidget(self.run_import_btn)
        import_row_widget = QtWidgets.QWidget()
        import_row_widget.setObjectName("import_row_widget")
        import_row_widget.setLayout(import_row)
        layout.addRow("Import:", import_row_widget)

        # ── Export combo + Run Export button ─────────────────────────────
        self.export_combo = QtWidgets.QComboBox()
        self.export_combo.setObjectName("export_combo")
        export_options = [
            ("Export Mesh To Map Layers", self.export_mesh_layers_btn),
            ("Export Mesh To UGRID", self.export_mesh_ugrid_btn),
            ("Save Mesh to GPKG", self.save_mesh_gpkg_btn),
            ("Export Results to UGRID", self.export_results_ugrid_btn),
        ]
        for label, btn in export_options:
            self.export_combo.addItem(label, btn)
        self.export_combo.setToolTip(
            "Pick an export action, then click 'Run Export' to execute it."
        )

        self.run_export_btn = QtWidgets.QPushButton("Run Export")
        self.run_export_btn.setObjectName("run_export_btn")
        self.run_export_btn.setToolTip(
            "Execute the export action selected in the Export combo."
        )
        self.run_export_btn.clicked.connect(self._run_selected_export)

        export_row = QtWidgets.QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.addWidget(self.export_combo, 1)
        export_row.addWidget(self.run_export_btn)
        export_row_widget = QtWidgets.QWidget()
        export_row_widget.setObjectName("export_row_widget")
        export_row_widget.setLayout(export_row)
        layout.addRow("Export:", export_row_widget)

        layout.addItem(
            QtWidgets.QSpacerItem(
                0, 0,
                QtWidgets.QSizePolicy.Minimum,
                QtWidgets.QSizePolicy.Expanding,
            )
        )

        # First page (top) — before Layer Setup
        self._toolbox.insertItem(0, page, "Import/Export")

    # ------------------------------------------------------------------
    # Import/Export combo dispatch
    # ------------------------------------------------------------------

    def _run_selected_import(self) -> None:
        """Click the QPushButton backing the currently-selected Import option."""
        btn = self.import_combo.currentData()
        if btn is None:
            return
        try:
            btn.click()
        except RuntimeError:
            pass

    def _run_selected_export(self) -> None:
        """Click the QPushButton backing the currently-selected Export option."""
        btn = self.export_combo.currentData()
        if btn is None:
            return
        try:
            btn.click()
        except RuntimeError:
            pass

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

    # ------------------------------------------------------------------
    # Group-box / filter helpers (mirrors ModelTabView)
    # ------------------------------------------------------------------

    def _create_topo_page(
        self,
        page_name: str,
        form_name: str,
        group_title: str,
    ) -> tuple[QtWidgets.QWidget, QtWidgets.QFormLayout]:
        """Create a toolbox page hosting a single titled QGroupBox.

        Mirrors ``ModelTabView._build_form_page`` + ``_start_param_group``:
        the page contains one ``QGroupBox`` (the "primary" group, e.g.
        "Layers", "General") whose ``QFormLayout`` is returned for the
        caller to add rows to.
        """
        page = QtWidgets.QWidget()
        page.setObjectName(page_name)
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(4, 4, 4, 4)
        page_layout.setSpacing(4)

        safe_title = group_title.lower().replace(" ", "_").replace("&", "and")
        group = QtWidgets.QGroupBox(group_title)
        group.setObjectName(f"{safe_title}_group")
        group.setCheckable(False)
        group_layout = QtWidgets.QFormLayout(group)
        group_layout.setObjectName(form_name)
        page_layout.addWidget(group)
        self._topo_param_groups.append(group)
        return page, group_layout

    def _start_topo_group(
        self,
        page_form: QtWidgets.QFormLayout,
        title: str,
        *,
        checkable: bool = False,
        advanced: bool = False,
    ) -> QtWidgets.QFormLayout:
        """Wrap a section of a toolbox page in a titled QGroupBox.

        Mirrors ``ModelTabView._start_param_group`` so the topology tab
        has the same visual style as the simulation tab.

        Returns the inner QFormLayout that the caller adds rows to.
        """
        group = QtWidgets.QGroupBox(title)
        safe_title = title.lower().replace(" ", "_").replace("&", "and")
        group.setObjectName(f"{safe_title}_group")
        group.setCheckable(checkable)
        if checkable:
            group.setChecked(False)
        if advanced:
            group.setProperty("advanced", True)
        group_layout = QtWidgets.QFormLayout(group)
        group_layout.setObjectName(f"{safe_title}_layout")
        page_form.addRow(group)
        self._topo_param_groups.append(group)
        return group_layout

    def _add_topo_param_row(
        self,
        group_layout: QtWidgets.QFormLayout,
        label_text: str,
        widget: QtWidgets.QWidget,
        *,
        advanced: bool = False,
    ) -> None:
        """Add a labeled widget to a topology group and register for filtering.

        Mirrors ``ModelTabView._add_param_row``.
        """
        label = QtWidgets.QLabel(label_text)
        group_layout.addRow(label, widget)
        group = group_layout.parentWidget()
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

    def _add_topo_self_describing_row(
        self,
        group_layout: QtWidgets.QFormLayout,
        widget: QtWidgets.QWidget,
        *,
        label_text: str = "",
        advanced: bool = False,
    ) -> None:
        """Register a self-describing widget (e.g. QCheckBox with its own text).

        Use this for QCheckBox / QPushButton rows where the widget already
        carries a descriptive label — we still register it with the
        filter so the search box and advanced toggle can hide it.
        """
        group = group_layout.parentWidget()
        text = label_text or str(widget.text() or widget.objectName() or "")
        self._filterable.add(
            widget,
            label_widget=None,
            label_text=text,
            tooltip=widget.toolTip() or "",
            group=group if isinstance(group, QtWidgets.QGroupBox) else None,
            advanced=advanced,
        )

    def _filter_topology_tab(self, _value=None) -> None:
        """Show/hide topology rows based on search text and advanced toggle."""
        self._filterable.apply_filter(
            self.topo_search.text(),
            show_advanced=self.topo_show_advanced_chk.isChecked(),
        )

    def _populate_gmsh_quality_controls(self) -> dict:
        """Populate gmsh/quality detail widgets inside the placeholder containers.

        Returns the widgets dict so the controller can wire signals.
        """
        try:
            widgets = _build_topology_tab_controls(self, self)
            self._topo_widgets = widgets
            for k, w in widgets.items():
                if isinstance(w, QtWidgets.QWidget) and not hasattr(self, k):
                    setattr(self, k, w)
            self.update_control_summary()
            return widgets
        except Exception as exc:
            self._log(f"[ERROR] Failed to populate gmsh/quality controls: {exc}")
            return {}

    def update_topo_status(self, text: str) -> None:
        """Set the status label text (called by topology controller)."""
        lbl = getattr(self, "topo_status_lbl", None)
        if lbl is not None:
            try:
                lbl.setText(str(text))
            except Exception as _e:
                self._log(f"[ERROR] Exception in topology_tab_view.py: {_e}")

    def _find_widget(self, attr: str):
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
            except Exception as _e:
                self._log(f"[ERROR] Exception in topology_tab_view.py: {_e}")

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
            except RuntimeError as _e:
                self._log(f"[ERROR] RuntimeError in topology_tab_view.py: {_e}")

    def update_control_summary(self) -> None:
        """Toggle Gmsh-only page enable state based on current backend.

        Gmsh-only pages are always enabled; they are hidden in the toolbox
        tab text when the Structured backend is selected.
        """
        backend_combo = getattr(self, "topo_backend_combo", None)
        if backend_combo is None:
            return
        is_gmsh = str(backend_combo.currentData() or "") == "gmsh"
        for idx in getattr(self, "_gmsh_only_indices", []):
            base = self._gmsh_only_base_titles.get(idx, "")
            self._toolbox.setItemEnabled(idx, True)
            self._toolbox.setItemText(
                idx, base if is_gmsh else f"{base} (Gmsh only)"
            )


def _build_topology_tab_controls(
    parent,
    topology_tab_page,
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

    _gmsh_label = "Gmsh (recommended)"
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

    # -- Find per-page forms for organized tabbed layout --
    def _find_form(page_name):
        """Locate the QFormLayout inside a toolbox page's primary QGroupBox.

        Each page hosts a single QGroupBox (the "primary" group created
        by ``_create_topo_page``); this helper unwraps it to expose
        the inner QFormLayout so the existing ``form.addRow(...)`` calls
        keep working unchanged.
        """
        page = _find(page_name, QtWidgets.QWidget)
        if page is None:
            page = QtWidgets.QWidget()
            page.setObjectName(page_name)
        # The page may either be a flat QFormLayout (legacy) or host
        # a QGroupBox whose child is a QFormLayout.
        page_layout = page.layout()
        if isinstance(page_layout, QtWidgets.QFormLayout):
            return page_layout
        # New style: page → QVBoxLayout → QGroupBox → QFormLayout
        group = page.findChild(QtWidgets.QGroupBox)
        if group is not None:
            form = group.layout()
            if isinstance(form, QtWidgets.QFormLayout):
                widgets[page_name] = page
                return form
        # Fallback: legacy behavior — create a QFormLayout on the page
        form = QtWidgets.QFormLayout(page)
        form.setContentsMargins(4, 4, 4, 4)
        widgets[page_name] = page
        return form

    algo_form = _find_form("topo_algo_page")
    arcs_form = _find_form("topo_arcs_page")
    sizing_form = _find_form("topo_sizing_page")
    threading_form = _find_form("topo_threading_page")
    transfinite_form = _find_form("topo_transfinite_page")
    quality_form = _find_form("topo_quality_page")

    def _add_row(form_layout, label_text, widget, *, advanced=False):
        """Add a labeled widget to a page's form and register for filtering.

        Mirrors ``TopologyTabView._add_topo_param_row`` but works as
        a closure inside ``_build_topology_tab_controls`` where we
        don't have direct access to ``self._filterable``. We use
        ``parent`` (the view) to register.
        """
        form_layout.addRow(label_text, widget)
        group = form_layout.parentWidget()
        registry = getattr(parent, "_filterable", None)
        if registry is None:
            return
        registry.add(
            widget,
            label_widget=None,
            label_text=label_text,
            tooltip=widget.toolTip() or "",
            group=group if isinstance(group, QtWidgets.QGroupBox) else None,
            advanced=advanced,
        )

    def _add_self_row(form_layout, widget, *, label_text=None, advanced=False):
        """Add a self-describing widget (no label widget) to the form."""
        form_layout.addRow(widget)
        group = form_layout.parentWidget()
        registry = getattr(parent, "_filterable", None)
        if registry is None:
            return
        text = label_text or str(widget.text() or widget.objectName() or "")
        registry.add(
            widget,
            label_widget=None,
            label_text=text,
            tooltip=widget.toolTip() or "",
            group=group if isinstance(group, QtWidgets.QGroupBox) else None,
            advanced=advanced,
        )

    # -- Gmsh algorithm combos --
    topo_gmsh_tri_algo_combo = QtWidgets.QComboBox()
    topo_gmsh_tri_algo_combo.setObjectName("topo_gmsh_tri_algo_combo")
    topo_gmsh_tri_algo_combo.setToolTip(
        "Gmsh 2D triangle meshing algorithm. "
        "Frontal-Delaunay produces higher quality; Delaunay is faster."
    )
    _add_row(algo_form, 'Triangle algorithm:', topo_gmsh_tri_algo_combo)
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
    topo_gmsh_quad_algo_combo.setToolTip(
        "Gmsh quadrilateral meshing algorithm. "
        "Frontal+Blossom is recommended for quality quads."
    )
    _add_row(algo_form, 'Quadrilateral algorithm:', topo_gmsh_quad_algo_combo)
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
    topo_gmsh_recombine_algo_combo.setToolTip(
        "Algorithm for recombining triangles into quads. "
        "Blossom is highest quality; Simple is fastest."
    )
    _add_row(algo_form, 'Recombine algorithm:', topo_gmsh_recombine_algo_combo)
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
    _add_self_row(algo_form, topo_gmsh_global_recombine_chk)
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
    _add_self_row(algo_form, topo_gmsh_quad_full_region_flow_align_chk)
    widgets["topo_gmsh_quad_full_region_flow_align_chk"] = topo_gmsh_quad_full_region_flow_align_chk

    topo_gmsh_smoothing_spin = QtWidgets.QSpinBox()
    topo_gmsh_smoothing_spin.setObjectName("topo_gmsh_smoothing_spin")
    topo_gmsh_smoothing_spin.setToolTip(
        "Number of Gmsh smoothing passes applied after mesh generation. "
        "0 = no smoothing. Higher values improve element shape quality."
    )
    topo_gmsh_smoothing_spin.setRange(0, 100)
    topo_gmsh_smoothing_spin.setValue(0)
    _add_row(algo_form, 'Smoothing passes:', topo_gmsh_smoothing_spin)
    widgets["topo_gmsh_smoothing_spin"] = topo_gmsh_smoothing_spin

    topo_gmsh_optimize_iters_spin = QtWidgets.QSpinBox()
    topo_gmsh_optimize_iters_spin.setObjectName("topo_gmsh_optimize_iters_spin")
    topo_gmsh_optimize_iters_spin.setToolTip(
        "Number of Gmsh optimization iterations. "
        "Higher values improve mesh quality at the cost of runtime."
    )
    topo_gmsh_optimize_iters_spin.setRange(0, 100)
    topo_gmsh_optimize_iters_spin.setValue(0)
    _add_row(algo_form, 'Optimize iterations:', topo_gmsh_optimize_iters_spin)
    widgets["topo_gmsh_optimize_iters_spin"] = topo_gmsh_optimize_iters_spin

    topo_gmsh_verbosity_spin = QtWidgets.QSpinBox()
    topo_gmsh_verbosity_spin.setObjectName("topo_gmsh_verbosity_spin")
    topo_gmsh_verbosity_spin.setToolTip(
        "Gmsh log verbosity level (0=silent, 10=most verbose). "
        "Default: 2 shows warnings and errors."
    )
    topo_gmsh_verbosity_spin.setRange(0, 10)
    topo_gmsh_verbosity_spin.setValue(2)
    _add_row(algo_form, 'Verbosity:', topo_gmsh_verbosity_spin)
    widgets["topo_gmsh_verbosity_spin"] = topo_gmsh_verbosity_spin

    topo_gmsh_optimize_netgen_chk = QtWidgets.QCheckBox("Enable Netgen optimize")
    topo_gmsh_optimize_netgen_chk.setObjectName("topo_gmsh_optimize_netgen_chk")
    topo_gmsh_optimize_netgen_chk.setToolTip(
        "Run Netgen optimizer after Gmsh mesh generation. "
        "Can improve element quality for some mesh topologies."
    )
    if not str(topo_gmsh_optimize_netgen_chk.text() or "").strip():
        topo_gmsh_optimize_netgen_chk.setText("Enable Netgen optimize")
    _add_self_row(algo_form, topo_gmsh_optimize_netgen_chk)
    widgets["topo_gmsh_optimize_netgen_chk"] = topo_gmsh_optimize_netgen_chk

    topo_gmsh_arc_mode_combo = QtWidgets.QComboBox()
    topo_gmsh_arc_mode_combo.setObjectName("topo_gmsh_arc_mode_combo")
    topo_gmsh_arc_mode_combo.setToolTip(
        "How topology arcs influence the mesh. "
        "'Hard embed' forces nodes on arcs; 'Soft size hint' refines near arcs; "
        "'Disabled' ignores arcs."
    )
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
    _add_row(arcs_form, 'Arc mode:', topo_gmsh_arc_mode_combo)

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
    _add_row(arcs_form, 'Arc soft size factor:', topo_gmsh_arc_soft_size_factor_spin)

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
    _add_row(arcs_form, 'Arc soft dist factor:', topo_gmsh_arc_soft_dist_factor_spin)

    topo_gmsh_interface_transition_enable_chk = QtWidgets.QCheckBox("Enable interface transition grading")
    topo_gmsh_interface_transition_enable_chk.setObjectName("topo_gmsh_interface_transition_enable_chk")
    if not str(topo_gmsh_interface_transition_enable_chk.text() or "").strip():
        topo_gmsh_interface_transition_enable_chk.setText("Enable interface transition grading")
    topo_gmsh_interface_transition_enable_chk.setChecked(True)
    topo_gmsh_interface_transition_enable_chk.setToolTip(
        "Apply Distance/Threshold grading near shared interfaces on non-transfinite regions only."
    )
    widgets["topo_gmsh_interface_transition_enable_chk"] = topo_gmsh_interface_transition_enable_chk
    _add_self_row(arcs_form, topo_gmsh_interface_transition_enable_chk)

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
    _add_row(arcs_form, 'Interface transition dist factor:', topo_gmsh_interface_transition_dist_factor_spin)

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
    _add_row(arcs_form, 'Interface transition min ratio:', topo_gmsh_interface_transition_min_ratio_spin)

    topo_gmsh_interface_conformance_chk = QtWidgets.QCheckBox("Enable transverse interface conformance post-process")
    topo_gmsh_interface_conformance_chk.setObjectName("topo_gmsh_interface_conformance_chk")
    if not str(topo_gmsh_interface_conformance_chk.text() or "").strip():
        topo_gmsh_interface_conformance_chk.setText("Enable transverse interface conformance post-process")
    topo_gmsh_interface_conformance_chk.setChecked(False)
    topo_gmsh_interface_conformance_chk.setToolTip(
        "Snap and weld mixed-interface nodes after Gmsh extraction to enforce shared boundary topology."
    )
    widgets["topo_gmsh_interface_conformance_chk"] = topo_gmsh_interface_conformance_chk
    _add_self_row(arcs_form, topo_gmsh_interface_conformance_chk)

    topo_gmsh_transverse_interface_centroid_merge_chk = QtWidgets.QCheckBox("Use centroid merge for matched transverse interface nodes")
    topo_gmsh_transverse_interface_centroid_merge_chk.setObjectName("topo_gmsh_transverse_interface_centroid_merge_chk")
    if not str(topo_gmsh_transverse_interface_centroid_merge_chk.text() or "").strip():
        topo_gmsh_transverse_interface_centroid_merge_chk.setText("Use centroid merge for matched transverse interface nodes")
    topo_gmsh_transverse_interface_centroid_merge_chk.setChecked(False)
    topo_gmsh_transverse_interface_centroid_merge_chk.setToolTip(
        "Move matched interface-node groups to their centroid before welding instead of one-sided snapping."
    )
    widgets["topo_gmsh_transverse_interface_centroid_merge_chk"] = topo_gmsh_transverse_interface_centroid_merge_chk
    _add_self_row(arcs_form, topo_gmsh_transverse_interface_centroid_merge_chk)

    topo_gmsh_interface_snap_tol_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_interface_snap_tol_spin.setObjectName("topo_gmsh_interface_snap_tol_spin")
    topo_gmsh_interface_snap_tol_spin.setRange(1.0e-6, 1.0e5)
    topo_gmsh_interface_snap_tol_spin.setDecimals(6)
    topo_gmsh_interface_snap_tol_spin.setValue(1.0)
    topo_gmsh_interface_snap_tol_spin.setToolTip(
        "Distance tolerance used by transverse interface conformance snapping."
    )
    widgets["topo_gmsh_interface_snap_tol_spin"] = topo_gmsh_interface_snap_tol_spin
    _add_row(arcs_form, 'Interface snap tolerance:', topo_gmsh_interface_snap_tol_spin)

    topo_gmsh_interface_reject_near_unshared_chk = QtWidgets.QCheckBox("Reject mixed interfaces with near-coincident unshared nodes")
    topo_gmsh_interface_reject_near_unshared_chk.setObjectName("topo_gmsh_interface_reject_near_unshared_chk")
    if not str(topo_gmsh_interface_reject_near_unshared_chk.text() or "").strip():
        topo_gmsh_interface_reject_near_unshared_chk.setText("Reject mixed interfaces with near-coincident unshared nodes")
    topo_gmsh_interface_reject_near_unshared_chk.setChecked(True)
    topo_gmsh_interface_reject_near_unshared_chk.setToolTip(
        "Fail meshing when a transfinite/tri interface shows hanging-node style near-miss pairs."
    )
    widgets["topo_gmsh_interface_reject_near_unshared_chk"] = topo_gmsh_interface_reject_near_unshared_chk
    _add_self_row(arcs_form, topo_gmsh_interface_reject_near_unshared_chk)

    topo_gmsh_interface_reject_tol_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_interface_reject_tol_spin.setObjectName("topo_gmsh_interface_reject_tol_spin")
    topo_gmsh_interface_reject_tol_spin.setRange(1.0e-6, 1.0e3)
    topo_gmsh_interface_reject_tol_spin.setDecimals(6)
    topo_gmsh_interface_reject_tol_spin.setValue(1.0e-3)
    topo_gmsh_interface_reject_tol_spin.setToolTip(
        "Tolerance for detecting near-coincident unshared interface nodes (hanging-node signature)."
    )
    widgets["topo_gmsh_interface_reject_tol_spin"] = topo_gmsh_interface_reject_tol_spin
    _add_row(arcs_form, 'Interface reject tolerance:', topo_gmsh_interface_reject_tol_spin)

    topo_gmsh_mesh_size_min_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_mesh_size_min_spin.setObjectName("topo_gmsh_mesh_size_min_spin")
    topo_gmsh_mesh_size_min_spin.setToolTip(
        "Global minimum mesh element size. "
        "Prevents Gmsh from creating elements smaller than this value."
    )
    topo_gmsh_mesh_size_min_spin.setRange(0.0, 1.0e6)
    topo_gmsh_mesh_size_min_spin.setDecimals(6)
    topo_gmsh_mesh_size_min_spin.setValue(0.0)
    _add_row(sizing_form, 'Global min cell size:', topo_gmsh_mesh_size_min_spin)
    widgets["topo_gmsh_mesh_size_min_spin"] = topo_gmsh_mesh_size_min_spin

    topo_gmsh_tolerance_edge_length_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_tolerance_edge_length_spin.setObjectName("topo_gmsh_tolerance_edge_length_spin")
    topo_gmsh_tolerance_edge_length_spin.setToolTip(
        "Edges shorter than this value are ignored during mesh generation. "
        "Useful for cleaning up tiny geometry artifacts."
    )
    topo_gmsh_tolerance_edge_length_spin.setRange(0.0, 1.0e6)
    topo_gmsh_tolerance_edge_length_spin.setDecimals(6)
    topo_gmsh_tolerance_edge_length_spin.setValue(0.0)
    _add_row(sizing_form, 'Ignore edges shorter than:', topo_gmsh_tolerance_edge_length_spin)
    widgets["topo_gmsh_tolerance_edge_length_spin"] = topo_gmsh_tolerance_edge_length_spin

    topo_gmsh_mesh_size_from_points_chk = QtWidgets.QCheckBox("Use region target_size for mesh sizing")
    topo_gmsh_mesh_size_from_points_chk.setObjectName("topo_gmsh_mesh_size_from_points_chk")
    topo_gmsh_mesh_size_from_points_chk.setToolTip(
        "When checked, per-region target_size values are passed to Gmsh "
        "for spatially varying element sizing."
    )
    if not str(topo_gmsh_mesh_size_from_points_chk.text() or "").strip():
        topo_gmsh_mesh_size_from_points_chk.setText("Use region target_size for mesh sizing")
    topo_gmsh_mesh_size_from_points_chk.setChecked(True)
    _add_self_row(sizing_form, topo_gmsh_mesh_size_from_points_chk)
    widgets["topo_gmsh_mesh_size_from_points_chk"] = topo_gmsh_mesh_size_from_points_chk

    topo_gmsh_quality_enable_chk = QtWidgets.QCheckBox("Enable Gmsh iterative quality loop")
    topo_gmsh_quality_enable_chk.setObjectName("topo_gmsh_quality_enable_chk")
    topo_gmsh_quality_enable_chk.setToolTip(
        "Enable iterative quality improvement loop. "
        "Re-meshes with adjusted parameters when quality thresholds are not met."
    )
    if not str(topo_gmsh_quality_enable_chk.text() or "").strip():
        topo_gmsh_quality_enable_chk.setText("Enable Gmsh iterative quality loop")
    topo_gmsh_quality_enable_chk.setChecked(False)
    widgets["topo_gmsh_quality_enable_chk"] = topo_gmsh_quality_enable_chk

    topo_gmsh_quality_max_iters_spin = QtWidgets.QSpinBox()
    topo_gmsh_quality_max_iters_spin.setObjectName("topo_gmsh_quality_max_iters_spin")
    topo_gmsh_quality_max_iters_spin.setToolTip(
        "Maximum quality improvement iterations before giving up."
    )
    topo_gmsh_quality_max_iters_spin.setRange(1, 50)
    topo_gmsh_quality_max_iters_spin.setValue(2)
    widgets["topo_gmsh_quality_max_iters_spin"] = topo_gmsh_quality_max_iters_spin

    topo_gmsh_quality_time_limit_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_quality_time_limit_spin.setObjectName("topo_gmsh_quality_time_limit_spin")
    topo_gmsh_quality_time_limit_spin.setToolTip(
        "Time budget in seconds for the quality improvement loop."
    )
    topo_gmsh_quality_time_limit_spin.setRange(1.0, 3600.0)
    topo_gmsh_quality_time_limit_spin.setDecimals(1)
    topo_gmsh_quality_time_limit_spin.setValue(55.0)
    widgets["topo_gmsh_quality_time_limit_spin"] = topo_gmsh_quality_time_limit_spin

    topo_quality_min_angle_spin = QtWidgets.QDoubleSpinBox()
    topo_quality_min_angle_spin.setObjectName("topo_quality_min_angle_spin")
    topo_quality_min_angle_spin.setToolTip(
        "Minimum acceptable cell angle in degrees. "
        "Elements with smaller angles fail the quality check."
    )
    topo_quality_min_angle_spin.setRange(0.0, 89.0)
    topo_quality_min_angle_spin.setDecimals(1)
    topo_quality_min_angle_spin.setValue(5.0)
    widgets["topo_quality_min_angle_spin"] = topo_quality_min_angle_spin

    topo_quality_max_aspect_spin = QtWidgets.QDoubleSpinBox()
    topo_quality_max_aspect_spin.setObjectName("topo_quality_max_aspect_spin")
    topo_quality_max_aspect_spin.setToolTip(
        "Maximum acceptable cell aspect ratio. "
        "Higher values allow more stretched elements."
    )
    topo_quality_max_aspect_spin.setRange(1.0, 1.0e4)
    topo_quality_max_aspect_spin.setDecimals(2)
    topo_quality_max_aspect_spin.setValue(20.0)
    widgets["topo_quality_max_aspect_spin"] = topo_quality_max_aspect_spin

    topo_quality_max_non_orth_spin = QtWidgets.QDoubleSpinBox()
    topo_quality_max_non_orth_spin.setObjectName("topo_quality_max_non_orth_spin")
    topo_quality_max_non_orth_spin.setToolTip(
        "Maximum acceptable non-orthogonality angle in degrees. "
        "Higher values allow more non-orthogonal cells."
    )
    topo_quality_max_non_orth_spin.setRange(1.0, 89.9)
    topo_quality_max_non_orth_spin.setDecimals(1)
    topo_quality_max_non_orth_spin.setValue(82.0)
    widgets["topo_quality_max_non_orth_spin"] = topo_quality_max_non_orth_spin

    topo_quality_min_area_edit = QtWidgets.QLineEdit("1e-14")
    topo_quality_min_area_edit.setObjectName("topo_quality_min_area_edit")
    topo_quality_min_area_edit.setToolTip(
        "Minimum cell area relative to bounding box that passes quality check. "
        "Zero-area or degenerate cells are rejected."
    )
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
    _add_self_row(algo_form, topo_gmsh_algo_switch_on_failure_chk)
    _add_self_row(algo_form, topo_gmsh_recombine_node_repositioning_chk)

    topo_quality_strict_chk = QtWidgets.QCheckBox("Strict quality acceptance")
    topo_quality_strict_chk.setObjectName("topo_quality_strict_chk")
    topo_quality_strict_chk.setToolTip(
        "When checked, ALL quality thresholds must be met for mesh acceptance. "
        "When unchecked, a best-effort approach is used."
    )
    if not str(topo_quality_strict_chk.text() or "").strip():
        topo_quality_strict_chk.setText("Strict quality acceptance")
    widgets["topo_quality_strict_chk"] = topo_quality_strict_chk
    _add_row(quality_form, 'Min angle (deg):', topo_quality_min_angle_spin)
    _add_row(quality_form, 'Max aspect ratio:', topo_quality_max_aspect_spin)
    _add_row(quality_form, 'Max non-orthogonal (deg):', topo_quality_max_non_orth_spin)
    _add_row(quality_form, 'Min area rel bbox:', topo_quality_min_area_edit)
    _add_self_row(quality_form, topo_quality_strict_chk)
    _add_row(quality_form, 'Size scales:', topo_quality_size_scales_edit)
    _add_row(quality_form, 'Smooth increments:', topo_quality_smooth_increments_edit)
    _add_row(quality_form, 'Recombine passes:', topo_gmsh_quality_recombine_topology_passes_edit)
    _add_row(quality_form, 'Recombine min quality:', topo_gmsh_quality_recombine_min_quality_edit)
    _add_row(quality_form, 'Random factors:', topo_gmsh_quality_random_factors_edit)
    _add_row(quality_form, 'Optimize methods:', topo_gmsh_quality_optimize_methods_edit)
    _add_self_row(quality_form, topo_gmsh_quality_enable_chk)
    _add_row(quality_form, 'Max iterations:', topo_gmsh_quality_max_iters_spin)
    _add_row(quality_form, 'Time limit (s):', topo_gmsh_quality_time_limit_spin)

    # ── Threading controls ──
    topo_gmsh_num_threads_spin = QtWidgets.QSpinBox()
    topo_gmsh_num_threads_spin.setObjectName("topo_gmsh_num_threads_spin")
    topo_gmsh_num_threads_spin.setRange(1, 64)
    topo_gmsh_num_threads_spin.setValue(1)
    topo_gmsh_num_threads_spin.setToolTip(
        "Number of threads for Gmsh. Higher values speed up meshing on multi-core CPUs. "
        "Respects BACKWATER_GMSH_NUM_THREADS env var as default."
    )
    _add_row(threading_form, 'Num threads:', topo_gmsh_num_threads_spin)
    widgets["topo_gmsh_num_threads_spin"] = topo_gmsh_num_threads_spin

    topo_gmsh_max_num_threads_2d_spin = QtWidgets.QSpinBox()
    topo_gmsh_max_num_threads_2d_spin.setObjectName("topo_gmsh_max_num_threads_2d_spin")
    topo_gmsh_max_num_threads_2d_spin.setRange(0, 64)
    topo_gmsh_max_num_threads_2d_spin.setValue(0)
    topo_gmsh_max_num_threads_2d_spin.setSpecialValueText("Auto")
    topo_gmsh_max_num_threads_2d_spin.setToolTip(
        "Max threads for 2D meshing (0 = auto). "
        "Respects BACKWATER_GMSH_MAX_NUM_THREADS_2D env var as default."
    )
    _add_row(threading_form, 'Max 2D threads:', topo_gmsh_max_num_threads_2d_spin)
    widgets["topo_gmsh_max_num_threads_2d_spin"] = topo_gmsh_max_num_threads_2d_spin

    # ── Transfinite harmonization controls ──
    topo_gmsh_transfinite_shared_interface_harmonize_chk = QtWidgets.QCheckBox(
        "Enable transfinite shared interface harmonization"
    )
    topo_gmsh_transfinite_shared_interface_harmonize_chk.setObjectName(
        "topo_gmsh_transfinite_shared_interface_harmonize_chk"
    )
    if not str(topo_gmsh_transfinite_shared_interface_harmonize_chk.text() or "").strip():
        topo_gmsh_transfinite_shared_interface_harmonize_chk.setText(
            "Enable transfinite shared interface harmonization"
        )
    topo_gmsh_transfinite_shared_interface_harmonize_chk.setChecked(False)
    topo_gmsh_transfinite_shared_interface_harmonize_chk.setToolTip(
        "Harmonize shared interfaces of transfinite regions so opposite-edge "
        "subsets are matched. Helps avoid hanging-node conflicts at shared boundaries."
    )
    _add_self_row(transfinite_form, topo_gmsh_transfinite_shared_interface_harmonize_chk)
    widgets["topo_gmsh_transfinite_shared_interface_harmonize_chk"] = (
        topo_gmsh_transfinite_shared_interface_harmonize_chk
    )

    topo_gmsh_transfinite_opposite_subset_start_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_transfinite_opposite_subset_start_spin.setObjectName(
        "topo_gmsh_transfinite_opposite_subset_start_spin"
    )
    topo_gmsh_transfinite_opposite_subset_start_spin.setRange(0.0, 1.0)
    topo_gmsh_transfinite_opposite_subset_start_spin.setDecimals(4)
    topo_gmsh_transfinite_opposite_subset_start_spin.setSingleStep(0.05)
    topo_gmsh_transfinite_opposite_subset_start_spin.setValue(0.30)
    topo_gmsh_transfinite_opposite_subset_start_spin.setToolTip(
        "Opposite-edge subset start fraction for transfinite interface matching. "
        "Respects BACKWATER_GMSH_TRANSFINITE_OPPOSITE_SUBSET_START env var."
    )
    _add_row(transfinite_form, 'Opposite subset start:', topo_gmsh_transfinite_opposite_subset_start_spin)
    widgets["topo_gmsh_transfinite_opposite_subset_start_spin"] = (
        topo_gmsh_transfinite_opposite_subset_start_spin
    )

    topo_gmsh_transfinite_opposite_subset_end_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_transfinite_opposite_subset_end_spin.setObjectName(
        "topo_gmsh_transfinite_opposite_subset_end_spin"
    )
    topo_gmsh_transfinite_opposite_subset_end_spin.setRange(0.0, 1.0)
    topo_gmsh_transfinite_opposite_subset_end_spin.setDecimals(4)
    topo_gmsh_transfinite_opposite_subset_end_spin.setSingleStep(0.05)
    topo_gmsh_transfinite_opposite_subset_end_spin.setValue(0.70)
    topo_gmsh_transfinite_opposite_subset_end_spin.setToolTip(
        "Opposite-edge subset end fraction for transfinite interface matching. "
        "Respects BACKWATER_GMSH_TRANSFINITE_OPPOSITE_SUBSET_END env var."
    )
    _add_row(transfinite_form, 'Opposite subset end:', topo_gmsh_transfinite_opposite_subset_end_spin)
    widgets["topo_gmsh_transfinite_opposite_subset_end_spin"] = (
        topo_gmsh_transfinite_opposite_subset_end_spin
    )

    topo_gmsh_transfinite_opposite_subset_density_scale_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_transfinite_opposite_subset_density_scale_spin.setObjectName(
        "topo_gmsh_transfinite_opposite_subset_density_scale_spin"
    )
    topo_gmsh_transfinite_opposite_subset_density_scale_spin.setRange(0.05, 5.0)
    topo_gmsh_transfinite_opposite_subset_density_scale_spin.setDecimals(4)
    topo_gmsh_transfinite_opposite_subset_density_scale_spin.setSingleStep(0.05)
    topo_gmsh_transfinite_opposite_subset_density_scale_spin.setValue(0.50)
    topo_gmsh_transfinite_opposite_subset_density_scale_spin.setToolTip(
        "Density scale for opposite-edge subsets. Lower values coarsen subset spacing. "
        "Respects BACKWATER_GMSH_TRANSFINITE_OPPOSITE_SUBSET_DENSITY_SCALE env var."
    )
    _add_row(transfinite_form, 'Opposite density scale:', topo_gmsh_transfinite_opposite_subset_density_scale_spin)
    widgets["topo_gmsh_transfinite_opposite_subset_density_scale_spin"] = (
        topo_gmsh_transfinite_opposite_subset_density_scale_spin
    )

    topo_gmsh_transfinite_interface_debug_chk = QtWidgets.QCheckBox(
        "Enable transfinite interface debug logging"
    )
    topo_gmsh_transfinite_interface_debug_chk.setObjectName(
        "topo_gmsh_transfinite_interface_debug_chk"
    )
    if not str(topo_gmsh_transfinite_interface_debug_chk.text() or "").strip():
        topo_gmsh_transfinite_interface_debug_chk.setText(
            "Enable transfinite interface debug logging"
        )
    topo_gmsh_transfinite_interface_debug_chk.setChecked(False)
    topo_gmsh_transfinite_interface_debug_chk.setToolTip(
        "Enable verbose debug logging for transfinite interface handling. "
        "Respects BACKWATER_GMSH_TRANSFINITE_INTERFACE_DEBUG env var."
    )
    _add_self_row(transfinite_form, topo_gmsh_transfinite_interface_debug_chk)
    widgets["topo_gmsh_transfinite_interface_debug_chk"] = (
        topo_gmsh_transfinite_interface_debug_chk
    )

    topo_gmsh_transfinite_subset_containment_enable_chk = QtWidgets.QCheckBox(
        "Enable subset containment detection"
    )
    topo_gmsh_transfinite_subset_containment_enable_chk.setObjectName(
        "topo_gmsh_transfinite_subset_containment_enable_chk"
    )
    if not str(topo_gmsh_transfinite_subset_containment_enable_chk.text() or "").strip():
        topo_gmsh_transfinite_subset_containment_enable_chk.setText(
            "Enable subset containment detection"
        )
    topo_gmsh_transfinite_subset_containment_enable_chk.setChecked(True)
    topo_gmsh_transfinite_subset_containment_enable_chk.setToolTip(
        "Enable logical detection of subset containment for shared interface matching. "
        "Respects BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_ENABLE env var."
    )
    _add_self_row(transfinite_form, topo_gmsh_transfinite_subset_containment_enable_chk)
    widgets["topo_gmsh_transfinite_subset_containment_enable_chk"] = (
        topo_gmsh_transfinite_subset_containment_enable_chk
    )

    topo_gmsh_transfinite_subset_containment_high_overlap_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_transfinite_subset_containment_high_overlap_spin.setObjectName(
        "topo_gmsh_transfinite_subset_containment_high_overlap_spin"
    )
    topo_gmsh_transfinite_subset_containment_high_overlap_spin.setRange(0.50, 1.0)
    topo_gmsh_transfinite_subset_containment_high_overlap_spin.setDecimals(4)
    topo_gmsh_transfinite_subset_containment_high_overlap_spin.setSingleStep(0.01)
    topo_gmsh_transfinite_subset_containment_high_overlap_spin.setValue(0.95)
    topo_gmsh_transfinite_subset_containment_high_overlap_spin.setToolTip(
        "Threshold above which a subset is considered 'high overlap'. "
        "Respects BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_HIGH_OVERLAP env var."
    )
    _add_row(
        transfinite_form,
        "Subset containment high overlap:",
        topo_gmsh_transfinite_subset_containment_high_overlap_spin,
    )
    widgets["topo_gmsh_transfinite_subset_containment_high_overlap_spin"] = (
        topo_gmsh_transfinite_subset_containment_high_overlap_spin
    )

    topo_gmsh_transfinite_subset_containment_min_overlap_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_transfinite_subset_containment_min_overlap_spin.setObjectName(
        "topo_gmsh_transfinite_subset_containment_min_overlap_spin"
    )
    topo_gmsh_transfinite_subset_containment_min_overlap_spin.setRange(0.0, 1.0)
    topo_gmsh_transfinite_subset_containment_min_overlap_spin.setDecimals(4)
    topo_gmsh_transfinite_subset_containment_min_overlap_spin.setSingleStep(0.01)
    topo_gmsh_transfinite_subset_containment_min_overlap_spin.setValue(0.02)
    topo_gmsh_transfinite_subset_containment_min_overlap_spin.setToolTip(
        "Minimum overlap fraction needed for containment to apply. "
        "Respects BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_MIN_OVERLAP env var."
    )
    _add_row(
        transfinite_form,
        "Subset containment min overlap:",
        topo_gmsh_transfinite_subset_containment_min_overlap_spin,
    )
    widgets["topo_gmsh_transfinite_subset_containment_min_overlap_spin"] = (
        topo_gmsh_transfinite_subset_containment_min_overlap_spin
    )

    topo_gmsh_transfinite_subset_containment_max_length_ratio_spin = QtWidgets.QDoubleSpinBox()
    topo_gmsh_transfinite_subset_containment_max_length_ratio_spin.setObjectName(
        "topo_gmsh_transfinite_subset_containment_max_length_ratio_spin"
    )
    topo_gmsh_transfinite_subset_containment_max_length_ratio_spin.setRange(1.0e-6, 10.0)
    topo_gmsh_transfinite_subset_containment_max_length_ratio_spin.setDecimals(6)
    topo_gmsh_transfinite_subset_containment_max_length_ratio_spin.setSingleStep(0.05)
    topo_gmsh_transfinite_subset_containment_max_length_ratio_spin.setValue(0.35)
    topo_gmsh_transfinite_subset_containment_max_length_ratio_spin.setToolTip(
        "Max edge-length ratio for containment checks. "
        "Respects BACKWATER_GMSH_TRANSFINITE_SUBSET_CONTAINMENT_MAX_LENGTH_RATIO env var."
    )
    _add_row(
        transfinite_form,
        "Subset containment max length ratio:",
        topo_gmsh_transfinite_subset_containment_max_length_ratio_spin,
    )
    widgets["topo_gmsh_transfinite_subset_containment_max_length_ratio_spin"] = (
        topo_gmsh_transfinite_subset_containment_max_length_ratio_spin
    )

    return widgets



