#!/usr/bin/env python3
"""Viewer for sampled SWE2D line results stored in GeoPackage/SQLite."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from qgis.PyQt import QtCore, QtWidgets

from swe2d.results.export_service import export_table_to_csv
from swe2d.results.profile_service import extract_profile_arrays
from swe2d.workbench.dialogs._plot_utils import try_import_matplotlib_qt

logger_wb = logging.getLogger(__name__)


class SWE2DLineResultsViewerDialog(QtWidgets.QDialog):
    """Viewer for sampled SWE2D line results stored in GeoPackage/SQLite."""

    _BASE_COLUMNS = [
        ("t_s", "Time (s)"),
        ("line_id", "Line ID"),
        ("line_name", "Line Name"),
        ("depth_m", "Depth ({L})"),
        ("velocity_ms", "Velocity ({L}/s)"),
        ("wse_m", "Water Surface ({L})"),
        ("bed_m", "Bed ({L})"),
        ("flow_cms", "Flow FV Face ({Q})"),
        ("flow_cell_cms", "Flow Cell ({Q})"),
    ]

    _PLOT_OPTIONS = [
        ("Depth", "depth_m"),
        ("Velocity", "velocity_ms"),
        ("Water Surface", "wse_m"),
        ("Bed", "bed_m"),
        ("Flow FV Face", "flow_cms"),
        ("Flow Cell", "flow_cell_cms"),
    ]

    _PROFILE_OPTIONS = [
        ("Depth", "depth_m"),
        ("Velocity", "velocity_ms"),
        ("Water Surface", "wse_m"),
        ("Bed", "bed_m"),
        ("Normal Flow", "flow_qn"),
        ("Froude", "fr"),
    ]

    _PROFILE_FILL_OPTIONS = [
        ("None", "none"),
        ("Depth", "depth_m"),
        ("Velocity", "velocity_ms"),
        ("Froude", "fr"),
        ("Normal Flow", "flow_qn"),
    ]

    def __init__(
        self,
        ts_records: List[Dict[str, object]],
        profile_records: List[Dict[str, object]],
        run_id: str,
        db_path: str,
        length_unit: str = "",
        flow_unit_label: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("2D Sample Line Results Viewer")
        self.resize(980, 620)

        self._ts_records = list(ts_records)
        self._profile_records = list(profile_records)
        self._run_id = str(run_id)
        self._db_path = str(db_path)
        self._length_unit = str(length_unit).strip() or "m"
        self._flow_unit = str(flow_unit_label).strip() or f"{self._length_unit}3/s"
        l_unit = self._length_unit
        q_unit = self._flow_unit
        self._columns = [(k, lbl.format(L=l_unit, Q=q_unit)) for k, lbl in self._BASE_COLUMNS]
        self._plot_canvas = None
        self._plot_fig = None
        self._mpl_motion_cid = None

        root = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            f"Run ID: {self._run_id}\nSource: {self._db_path}"
        )
        header.setWordWrap(True)
        root.addWidget(header)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Line:"))
        self.line_combo = QtWidgets.QComboBox()
        controls.addWidget(self.line_combo)
        controls.addWidget(QtWidgets.QLabel("View:"))
        self.view_mode_combo = QtWidgets.QComboBox()
        self.view_mode_combo.addItem("Time series", "time")
        self.view_mode_combo.addItem("Profile at timestep", "profile")
        self.view_mode_combo.addItem("WSE + Bed profile", "wse_bed")
        controls.addWidget(self.view_mode_combo)
        controls.addWidget(QtWidgets.QLabel("Variable:"))
        self.metric_combo = QtWidgets.QComboBox()
        for label, key in self._PLOT_OPTIONS:
            self.metric_combo.addItem(label, key)
        controls.addWidget(self.metric_combo)
        controls.addWidget(QtWidgets.QLabel("Profile variable:"))
        self.profile_metric_combo = QtWidgets.QComboBox()
        for label, key in self._PROFILE_OPTIONS:
            self.profile_metric_combo.addItem(label, key)
        controls.addWidget(self.profile_metric_combo)
        controls.addWidget(QtWidgets.QLabel("Timestep:"))
        self.time_combo = QtWidgets.QComboBox()
        controls.addWidget(self.time_combo)
        controls.addWidget(QtWidgets.QLabel("Fill by:"))
        self.fill_metric_combo = QtWidgets.QComboBox()
        for label, key in self._PROFILE_FILL_OPTIONS:
            self.fill_metric_combo.addItem(label, key)
        controls.addWidget(self.fill_metric_combo)
        self.wse_render_lbl = QtWidgets.QLabel("WSE render:")
        controls.addWidget(self.wse_render_lbl)
        self.wse_render_combo = QtWidgets.QComboBox()
        self.wse_render_combo.addItem("Clipped to bed (wet only)", "clipped")
        self.wse_render_combo.addItem("Raw sampled", "raw")
        controls.addWidget(self.wse_render_combo)
        controls.addStretch(1)
        root.addLayout(controls)

        self.table = QtWidgets.QTableWidget()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels([lbl for _, lbl in self._columns])
        self.table.horizontalHeader().setStretchLastSection(True)

        self._have_mpl = False
        FigureCanvas, Figure, _ = try_import_matplotlib_qt()
        if FigureCanvas is not None and Figure is not None:
            self._have_mpl = True
            self._plot_fig = Figure(figsize=(6.8, 3.0), tight_layout=True)
            self._plot_canvas = FigureCanvas(self._plot_fig)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        table_host = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_host)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.table)
        split.addWidget(table_host)

        plot_host = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        if self._have_mpl:
            plot_layout.addWidget(self._plot_canvas)
        else:
            note = QtWidgets.QLabel("Matplotlib backend unavailable; table view only.")
            note.setWordWrap(True)
            plot_layout.addWidget(note)
        split.addWidget(plot_host)
        split.setSizes([380, 220])
        root.addWidget(split, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        export_btn = buttons.addButton("Export Table CSV...", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        export_btn.clicked.connect(self._export_current_table_csv)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._populate_line_combo()
        self._populate_time_combo()
        self._sync_control_visibility()
        self._refresh_table()
        self._refresh_plot()
        self._notify_parent_line_selection()

        self.line_combo.currentIndexChanged.connect(self._refresh_table)
        self.line_combo.currentIndexChanged.connect(self._refresh_plot)
        self.line_combo.currentIndexChanged.connect(self._notify_parent_line_selection)
        self.metric_combo.currentIndexChanged.connect(self._refresh_plot)
        self.profile_metric_combo.currentIndexChanged.connect(self._refresh_plot)
        self.time_combo.currentIndexChanged.connect(self._refresh_table)
        self.time_combo.currentIndexChanged.connect(self._refresh_plot)
        self.fill_metric_combo.currentIndexChanged.connect(self._refresh_plot)
        self.wse_render_combo.currentIndexChanged.connect(self._refresh_plot)
        self.view_mode_combo.currentIndexChanged.connect(self._sync_control_visibility)
        self.view_mode_combo.currentIndexChanged.connect(self._refresh_table)
        self.view_mode_combo.currentIndexChanged.connect(self._refresh_plot)
        self.view_mode_combo.currentIndexChanged.connect(self._notify_parent_line_selection)
        self.finished.connect(self._notify_parent_closed)

        if self._have_mpl and self._plot_canvas is not None:
            try:
                self._mpl_motion_cid = self._plot_canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
            except Exception:
                logger_wb.warning("Exception connecting matplotlib hover event", exc_info=True)
                self._mpl_motion_cid = None

    def _unit_label_for_metric(self, metric: str) -> str:
        """Return the unit label for a given results metric name."""
        m = str(metric or "")
        if m in ("depth_m", "wse_m", "bed_m", "station_m"):
            return self._length_unit
        if m == "velocity_ms":
            return f"{self._length_unit}/s"
        if m in ("flow_cms", "flow_cell_cms", "flow_fv_cms"):
            return self._flow_unit
        if m == "flow_qn":
            return f"{self._length_unit}^2/s"
        return ""

    def _label_with_unit(self, label: str, metric: str) -> str:
        """Append the unit label in parentheses to the given label string."""
        unit = self._unit_label_for_metric(metric)
        return str(label) if not unit else f"{label} ({unit})"

    def _line_filter(self):
        """Return the selected line ID as int, or None for 'All lines'."""
        value = self.line_combo.currentData()
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            self._log("[WARNING] Graceful degradation — Exception returned fallback value")
            return None

    def _selected_time(self) -> Optional[float]:
        """Return the selected timestep value in seconds, or None."""
        value = self.time_combo.currentData()
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            self._log("[WARNING] Graceful degradation — Exception returned fallback value")
            return None

    def _populate_line_combo(self):
        """Populate the line combo from time-series and profile records."""
        self.line_combo.clear()
        self.line_combo.addItem("All lines", None)
        by_line: Dict[int, str] = {}
        for rec in (self._ts_records + self._profile_records):
            try:
                lid = int(rec.get("line_id", -1))
            except (ValueError, TypeError):
                self._log("[WARNING] Skipping item due to Exception")
                continue
            lname = str(rec.get("line_name", "") or "")
            if lid not in by_line:
                by_line[lid] = lname
        for lid in sorted(by_line.keys()):
            label = f"{lid}"
            if by_line[lid]:
                label += f" - {by_line[lid]}"
            self.line_combo.addItem(label, lid)

    def _populate_time_combo(self):
        """Populate the timestep combo from profile or time-series records."""
        self.time_combo.clear()
        ts_vals = sorted({float(r.get("t_s", 0.0)) for r in self._profile_records})
        if not ts_vals:
            ts_vals = sorted({float(r.get("t_s", 0.0)) for r in self._ts_records})
        for t_s in ts_vals:
            self.time_combo.addItem(f"{t_s / 3600.0:.4f} hr", float(t_s))

    def _sync_control_visibility(self):
        """Show/hide controls based on the selected view mode."""
        mode = str(self.view_mode_combo.currentData())
        is_time = (mode == "time")
        is_profile = (mode == "profile")
        is_wse = (mode == "wse_bed")
        self.metric_combo.setVisible(is_time)
        self.profile_metric_combo.setVisible(is_profile)
        self.time_combo.setVisible(is_profile or is_wse)
        self.fill_metric_combo.setVisible(is_wse)
        self.wse_render_lbl.setVisible(is_wse)
        self.wse_render_combo.setVisible(is_wse)

    def _filtered_records(self) -> List[Dict[str, object]]:
        """Return time-series records filtered by the selected line."""
        lid = self._line_filter()
        if lid is None:
            return list(self._ts_records)
        out = []
        for rec in self._ts_records:
            try:
                if int(rec.get("line_id", -1)) == lid:
                    out.append(rec)
            except Exception:
                self._log("[WARNING] Skipping item due to Exception")
                continue
        return out

    def _filtered_profile_records(self) -> List[Dict[str, object]]:
        """Return profile records filtered by the selected line and timestep."""
        lid = self._line_filter()
        t_sel = self._selected_time()
        out = []
        for rec in self._profile_records:
            try:
                if lid is not None and int(rec.get("line_id", -1)) != lid:
                    continue
                if t_sel is not None and abs(float(rec.get("t_s", 0.0)) - t_sel) > 1.0e-9:
                    continue
            except Exception:
                self._log("[WARNING] Skipping item due to Exception")
                continue
            out.append(rec)
        return out

    def _refresh_table(self):
        """Reload the table widget based on the current view mode and filters."""
        def _fmt(v):
            """Format a cell value for display (None→\"\", float→6-decimal, else str)."""
            if v is None:
                return ""
            if isinstance(v, float):
                return f"{v:.6f}" if np.isfinite(v) else ""
            return str(v)

        mode = str(self.view_mode_combo.currentData())
        if mode == "time":
            rows = self._filtered_records()
            rows.sort(key=lambda r: (float(r.get("t_s", 0.0)), int(r.get("line_id", -1))))
            self.table.setColumnCount(len(self._columns))
            self.table.setHorizontalHeaderLabels([lbl for _, lbl in self._columns])
            self.table.setRowCount(len(rows))
            for r, rec in enumerate(rows):
                for c, (key, _) in enumerate(self._columns):
                    val = rec.get(key)
                    txt = _fmt(val)
                    self.table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
            return

        rows = self._filtered_profile_records()
        rows.sort(key=lambda r: float(r.get("station_m", 0.0)))
        cols = [
            ("t_s", "Time (s)"),
            ("line_id", "Line ID"),
            ("line_name", "Line Name"),
            ("station_m", self._label_with_unit("Station", "station_m")),
            ("depth_m", self._columns[3][1]),
            ("velocity_ms", self._columns[4][1]),
            ("wse_m", self._columns[5][1]),
            ("bed_m", self._columns[6][1]),
            ("flow_qn", self._label_with_unit("Normal Flow Density", "flow_qn")),
            ("fr", "Froude"),
        ]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([lbl for _, lbl in cols])
        self.table.setRowCount(len(rows))
        for r, rec in enumerate(rows):
            for c, (key, _) in enumerate(cols):
                val = rec.get(key)
                txt = _fmt(val)
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))

    def _refresh_plot(self):
        """Replot line results for the current view mode and filter selections."""
        if not self._have_mpl or self._plot_fig is None or self._plot_canvas is None:
            return
        mode = str(self.view_mode_combo.currentData())
        self._plot_fig.clear()
        ax = self._plot_fig.add_subplot(111)
        if mode == "time":
            rows = self._filtered_records()
            metric = str(self.metric_combo.currentData())
            if not rows:
                ax.text(0.5, 0.5, "No sampled line results", ha="center", va="center", transform=ax.transAxes)
                self._plot_canvas.draw_idle()
                return
            by_line: Dict[int, List[Tuple[float, float]]] = {}
            name_by_line: Dict[int, str] = {}
            for rec in rows:
                try:
                    lid = int(rec.get("line_id", -1))
                    ts = float(rec.get("t_s", 0.0))
                    vv = float(rec.get(metric, float("nan")))
                except Exception:
                    self._log("[WARNING] Skipping item due to Exception")
                    continue
                if not np.isfinite(vv):
                    continue
                by_line.setdefault(lid, []).append((ts, vv))
                name_by_line[lid] = str(rec.get("line_name", "") or "")
            if not by_line:
                ax.text(0.5, 0.5, "No numeric values to plot", ha="center", va="center", transform=ax.transAxes)
                self._plot_canvas.draw_idle()
                return
            for lid in sorted(by_line.keys()):
                pairs = sorted(by_line[lid], key=lambda x: x[0])
                t_hr = np.asarray([p[0] / 3600.0 for p in pairs], dtype=np.float64)
                vals = np.asarray([p[1] for p in pairs], dtype=np.float64)
                label = f"Line {lid}"
                if name_by_line.get(lid):
                    label += f" ({name_by_line[lid]})"
                ax.plot(t_hr, vals, "-", linewidth=1.8, label=label)
            ax.set_xlabel("Time (hr)")
            ax.set_ylabel(self._label_with_unit(self.metric_combo.currentText(), metric))
            ax.set_title("Sample line time series")
            if len(by_line) > 1:
                ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            self._plot_canvas.draw_idle()
            return

        rows = self._filtered_profile_records()
        if not rows:
            ax.text(0.5, 0.5, "No profile records for selected line/timestep", ha="center", va="center", transform=ax.transAxes)
            self._plot_canvas.draw_idle()
            return
        pa = extract_profile_arrays(rows)
        line_name = str(rows[0].get("line_name", "") or "")
        line_id = int(rows[0].get("line_id", -1))
        t_s = float(rows[0].get("t_s", 0.0))

        if mode == "profile":
            metric = str(self.profile_metric_combo.currentData())
            x = pa["station_m"]
            y = pa.get(metric, np.full_like(x, np.nan))
            ok = np.isfinite(y)
            if np.any(ok):
                ax.plot(x[ok], y[ok], "-", linewidth=1.8)
            ax.set_xlabel(self._label_with_unit("Station", "station_m"))
            ax.set_ylabel(self._label_with_unit(self.profile_metric_combo.currentText(), metric))
            ax.set_title(f"Line {line_id} profile at t={t_s/3600.0:.4f} hr" + (f" ({line_name})" if line_name else ""))
            ax.grid(True, alpha=0.3)
            self._plot_canvas.draw_idle()
            return

        # WSE + bed profile. Rendering is wet-aware and clips WSE to bed for
        # display so dry/near-dry samples do not produce misleading below-bed dips.
        ok = np.isfinite(pa["wse_m"]) & np.isfinite(pa["bed_m"])
        if not np.any(ok):
            ax.text(0.5, 0.5, "No WSE/bed values for selected line/timestep", ha="center", va="center", transform=ax.transAxes)
            self._plot_canvas.draw_idle()
            return

        x_ok = pa["station_m"][ok]
        wse_ok = pa["wse_m"][ok]
        bed_ok = pa["bed_m"][ok]
        depth_ok = pa["depth_m"][ok]
        wet_ok_raw = pa["wet"][ok]
        wet_mask = np.where(np.isfinite(wet_ok_raw), wet_ok_raw > 0.5, depth_ok > 1.0e-9)

        render_mode = str(self.wse_render_combo.currentData()) if hasattr(self, "wse_render_combo") else "clipped"
        wse_phys = np.maximum(wse_ok, bed_ok)
        below_bed_count = int(np.sum(wse_ok < bed_ok))
        if render_mode == "raw":
            fill_mask = np.isfinite(wse_ok) & np.isfinite(bed_ok)
            wse_fill = wse_ok
            wse_plot = wse_ok
            render_note = f"Raw mode: {below_bed_count} sample(s) with WSE < bed"
        else:
            fill_mask = wet_mask
            wse_fill = wse_phys
            wse_plot = np.where(wet_mask, wse_phys, np.nan)
            render_note = f"Display note: clipped {below_bed_count} sample(s) where WSE < bed"
        fill_key = str(self.fill_metric_combo.currentData())

        if fill_key != "none":
            try:
                from matplotlib import cm as mpl_cm, colors as mpl_colors
                fill_vals = pa.get(fill_key, np.full(pa["station_m"].shape[0], np.nan))[ok]
                finite = np.isfinite(fill_vals)
                if np.any(finite):
                    vmin = float(np.nanmin(fill_vals[finite]))
                    vmax = float(np.nanmax(fill_vals[finite]))
                    if vmax <= vmin:
                        vmax = vmin + 1.0
                    norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)
                    cmap = mpl_cm.get_cmap("viridis")
                    for i in range(len(x_ok) - 1):
                        if not (np.isfinite(fill_vals[i]) and np.isfinite(fill_vals[i + 1])):
                            continue
                        if not (fill_mask[i] and fill_mask[i + 1]):
                            continue
                        c_mid = cmap(norm(0.5 * (fill_vals[i] + fill_vals[i + 1])))
                        ax.fill_between(
                            x_ok[i : i + 2],
                            bed_ok[i : i + 2],
                            wse_fill[i : i + 2],
                            color=c_mid,
                            alpha=0.85,
                            linewidth=0.0,
                        )
                    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
                    sm.set_array([])
                    self._plot_fig.colorbar(sm, ax=ax, label=self.fill_metric_combo.currentText())
            except Exception:
                logger_wb.warning("Unhandled Exception", exc_info=True)
                ax.fill_between(x_ok, bed_ok, wse_fill, where=fill_mask, interpolate=True, color="tab:blue", alpha=0.18)
        else:
            ax.fill_between(x_ok, bed_ok, wse_fill, where=fill_mask, interpolate=True, color="tab:blue", alpha=0.18)

        ax.plot(x_ok, bed_ok, "-", color="saddlebrown", linewidth=1.6, label="Bed")
        ax.plot(x_ok, wse_plot, "-", color="royalblue", linewidth=1.8, label="Water Surface")
        if below_bed_count > 0:
            ax.text(
                0.01,
                0.99,
            render_note,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="0.35",
            )
        ax.set_xlabel(self._label_with_unit("Station", "station_m"))
        ax.set_ylabel(self._label_with_unit("Elevation", "wse_m"))
        ax.set_title(f"Line {line_id} WSE + bed at t={t_s/3600.0:.4f} hr" + (f" ({line_name})" if line_name else ""))
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        self._plot_canvas.draw_idle()

    def _table_rows_for_export(self) -> Tuple[List[str], List[List[str]]]:
        """Extract headers and all rows from the current table widget state."""
        headers = [str(self.table.horizontalHeaderItem(i).text()) if self.table.horizontalHeaderItem(i) is not None else f"col_{i}" for i in range(self.table.columnCount())]
        rows: List[List[str]] = []
        for r in range(self.table.rowCount()):
            row = []
            for c in range(self.table.columnCount()):
                it = self.table.item(r, c)
                row.append(str(it.text()) if it is not None else "")
            rows.append(row)
        return headers, rows

    def _export_current_table_csv(self):
        """Export the current table contents to a CSV file via file dialog."""
        default_name = f"line_results_{self._run_id}.csv" if self._run_id else "line_results.csv"
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Current Table to CSV",
            default_name,
            "CSV files (*.csv)",
        )
        if not out_path:
            return
        if not out_path.lower().endswith(".csv"):
            out_path += ".csv"
        headers, rows = self._table_rows_for_export()
        try:
            export_table_to_csv(out_path, headers, rows)
            QtWidgets.QMessageBox.information(self, "Export CSV", f"Exported {len(rows)} row(s) to:\n{out_path}")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Export CSV", f"Failed to export CSV:\n{exc}")

    def _notify_parent_line_selection(self, *_):
        """Notify the parent dialog about the current line selection change."""
        self._notify_parent_hover_station(None)
        p = self.parent()
        if p is None or not hasattr(p, "_on_line_viewer_selection_changed"):
            return
        try:
            p._on_line_viewer_selection_changed(self._line_filter())
        except Exception:
            self._log("[WARNING] Unexpected Exception silently caught — review this handler")

    def _notify_parent_hover_station(self, station_m: Optional[float]):
        """Notify the parent dialog about a hovered station along the line."""
        p = self.parent()
        if p is None or not hasattr(p, "_on_line_viewer_hover_station"):
            return
        try:
            p._on_line_viewer_hover_station(self._line_filter(), station_m)
        except Exception:
            self._log("[WARNING] Unexpected Exception silently caught — review this handler")

    def _notify_parent_closed(self, *_):
        """Notify the parent dialog that this viewer has been closed."""
        self._notify_parent_hover_station(None)
        p = self.parent()
        if p is None or not hasattr(p, "_on_line_viewer_selection_changed"):
            return
        try:
            p._on_line_viewer_selection_changed(None)
        except Exception:
            self._log("[WARNING] Unexpected Exception silently caught — review this handler")

    def _on_plot_hover(self, event):
        """Handle matplotlib hover events to relay station to parent dialog."""
        mode = str(self.view_mode_combo.currentData())
        if mode not in ("profile", "wse_bed"):
            self._notify_parent_hover_station(None)
            return
        if event is None or event.inaxes is None or event.xdata is None:
            self._notify_parent_hover_station(None)
            return
        try:
            station_m = float(event.xdata)
        except (ValueError, TypeError):
            self._log("[WARNING] Unhandled Exception")
            self._notify_parent_hover_station(None)
            return
        if not np.isfinite(station_m):
            self._notify_parent_hover_station(None)
            return
        self._notify_parent_hover_station(station_m)
