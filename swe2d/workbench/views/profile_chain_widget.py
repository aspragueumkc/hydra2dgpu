"""swe2d/workbench/views/profile_chain_widget.py

PyQt5 widget — chain editor for the network profile viewer.

Widgets: toolbar (Pick on Map / Find Path / Add / Reverse / Up / Down /
Remove / Clear), QListWidget of chain links, Save/Load profile buttons,
status bar showing total length + node range.

Signals:
  chain_changed = pyqtSignal(object)   # emits ChainSpec
  pick_requested = pyqtSignal()
"""

from __future__ import annotations

import logging
from typing import List, Optional

from qgis.PyQt import QtCore, QtWidgets

from swe2d import units as _u
from swe2d.workbench.services.drainage_graph_service import (
    DrainageGraph,
    find_chain,
    link_orientation,
)
from swe2d.workbench.services.profile_persistence_service import (
    list_profiles,
    load_profile,
    save_profile,
)
from swe2d.workbench.services.profile_pipeline_service import ChainSpec

logger = logging.getLogger(__name__)


class ProfileChainWidget(QtWidgets.QWidget):
    chain_changed = QtCore.pyqtSignal(object)
    pick_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gpkg_path: str = ""
        self._graph: Optional[DrainageGraph] = None
        self._chain = ChainSpec()
        self._build_ui()

    def set_context(self, gpkg_path: str, graph: DrainageGraph):
        self._gpkg_path = gpkg_path
        self._graph = graph
        self._node_a_combo.clear()
        self._node_b_combo.clear()
        self._node_a_combo.addItems(graph.node_ids)
        self._node_b_combo.addItems(graph.node_ids)

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        tb = QtWidgets.QHBoxLayout()
        self._pick_btn = QtWidgets.QPushButton("Pick on Map")
        self._pick_btn.clicked.connect(self.pick_requested.emit)
        self._find_btn = QtWidgets.QPushButton("Find Shortest Path")
        self._find_btn.clicked.connect(self._on_find_path)
        for w in (self._pick_btn, self._find_btn):
            tb.addWidget(w)
        tb.addStretch(1)
        root.addLayout(tb)

        # Find-path section
        fp = QtWidgets.QGroupBox("Find path by start / end node")
        fp_layout = QtWidgets.QHBoxLayout(fp)
        fp_layout.addWidget(QtWidgets.QLabel("Start:"))
        self._node_a_combo = QtWidgets.QComboBox()
        fp_layout.addWidget(self._node_a_combo)
        fp_layout.addWidget(QtWidgets.QLabel("End:"))
        self._node_b_combo = QtWidgets.QComboBox()
        fp_layout.addWidget(self._node_b_combo)
        root.addWidget(fp)

        # Toolbar 2
        tb2 = QtWidgets.QHBoxLayout()
        for label, slot in [
            ("Reverse", self._on_reverse),
            ("Up", self._on_move_up),
            ("Down", self._on_move_down),
            ("Remove", self._on_remove),
            ("Clear", self._on_clear),
        ]:
            btn = QtWidgets.QPushButton(label)
            btn.clicked.connect(slot)
            tb2.addWidget(btn)
        tb2.addStretch(1)
        root.addLayout(tb2)

        # List of links
        self._list = QtWidgets.QListWidget()
        root.addWidget(self._list, stretch=1)

        # Save / Load
        save_row = QtWidgets.QHBoxLayout()
        self._save_btn = QtWidgets.QPushButton("Save Profile")
        self._save_btn.clicked.connect(self._on_save)
        self._load_combo = QtWidgets.QComboBox()
        self._load_btn = QtWidgets.QPushButton("Load Profile")
        self._load_btn.clicked.connect(self._on_load)
        save_row.addWidget(self._save_btn)
        save_row.addWidget(self._load_combo, stretch=1)
        save_row.addWidget(self._load_btn)
        root.addLayout(save_row)

        # Status
        self._status_lbl = QtWidgets.QLabel("(empty)")
        root.addWidget(self._status_lbl)

    def get_chain(self) -> ChainSpec:
        return self._chain

    def set_chain(self, chain: ChainSpec):
        self._chain = chain
        self._refresh_list()
        self._emit_changed()

    def add_link_id(self, link_id: str):
        if self._graph is None:
            return
        existing = [lid for lid, _ in self._chain.link_specs]
        if link_id in existing:
            return
        # Determine orientation: if this is the first link, default forward.
        reverse = False
        if existing and self._graph is not None:
            last_link_id = existing[-1]
            last_meta_to = self._graph.to_node[last_link_id]
            link_fn = self._graph.from_node[link_id]
            link_tn = self._graph.to_node[link_id]
            if link_fn == last_meta_to:
                reverse = False
            elif link_tn == last_meta_to:
                reverse = True
        spec = self._chain.link_specs + [(link_id, reverse)]
        self.set_chain(ChainSpec(link_specs=spec))

    def _refresh_list(self):
        self._list.clear()
        total_length = 0.0
        for lid, rev in self._chain.link_specs:
            label = f"{'⤴' if rev else '⤵'}  {lid} ({'R' if rev else 'F'})"
            self._list.addItem(label)
            if self._graph and lid in self._graph.from_node:
                meta = _lookup_link_length(self._gpkg_path, lid)
                total_length += meta
        n_links = len(self._chain.link_specs)
        upstream = self._chain.link_specs[0][0] if self._chain.link_specs else "—"
        downstream = self._chain.link_specs[-1][0] if self._chain.link_specs else "—"
        if self._graph:
            if upstream in self._graph.from_node.values():
                pass
        unit = str(_u.length_unit_name()).strip()
        self._status_lbl.setText(
            f"{n_links} link(s) | length ≈ {total_length:.1f} {unit} | "
            f"{self._chain.link_specs[0][0] if self._chain.link_specs else '—'} → "
            f"{self._chain.link_specs[-1][0] if self._chain.link_specs else '—'}"
        )

    def _emit_changed(self):
        self._refresh_list()
        self.chain_changed.emit(self._chain)

    def _on_find_path(self):
        if self._graph is None:
            return
        a = self._node_a_combo.currentText()
        b = self._node_b_combo.currentText()
        link_ids = find_chain(self._graph, a, b)
        if not link_ids:
            QtWidgets.QMessageBox.information(
                self, "Find Path", f"No path from {a} to {b}."
            )
            return
        spec_list = []
        prev_end = a
        for lid in link_ids:
            reverse = not link_orientation(self._graph, lid, prev_end)
            spec_list.append((lid, reverse))
            prev_end = (
                self._graph.to_node[lid] if not reverse else self._graph.from_node[lid]
            )
        self.set_chain(ChainSpec(link_specs=spec_list))

    def _on_reverse(self):
        idx = self._list.currentRow()
        if idx < 0:
            return
        ls = list(self._chain.link_specs)
        lid, rev = ls[idx]
        ls[idx] = (lid, not rev)
        self.set_chain(ChainSpec(link_specs=ls))

    def _on_move_up(self):
        idx = self._list.currentRow()
        if idx <= 0:
            return
        ls = list(self._chain.link_specs)
        ls[idx - 1], ls[idx] = ls[idx], ls[idx - 1]
        self.set_chain(ChainSpec(link_specs=ls))

    def _on_move_down(self):
        idx = self._list.currentRow()
        if idx < 0 or idx >= len(self._chain.link_specs) - 1:
            return
        ls = list(self._chain.link_specs)
        ls[idx + 1], ls[idx] = ls[idx], ls[idx + 1]
        self.set_chain(ChainSpec(link_specs=ls))

    def _on_remove(self):
        idx = self._list.currentRow()
        if idx < 0:
            return
        ls = list(self._chain.link_specs)
        ls.pop(idx)
        self.set_chain(ChainSpec(link_specs=ls))

    def _on_clear(self):
        self.set_chain(ChainSpec(link_specs=[]))

    def _on_save(self):
        if self._chain.is_empty():
            QtWidgets.QMessageBox.information(self, "Save Profile", "Chain is empty.")
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Profile", "Profile name:"
        )
        if not ok or not name.strip():
            return
        try:
            save_profile(self._gpkg_path, name.strip(), self._chain)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Save Profile", str(exc))
            return
        self._refresh_load_combo()

    def _on_load(self):
        pid = self._load_combo.currentData()
        if pid is None:
            return
        try:
            self.set_chain(load_profile(self._gpkg_path, int(pid)))
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Load Profile", str(exc))

    def _refresh_load_combo(self):
        self._load_combo.clear()
        for p in list_profiles(self._gpkg_path):
            self._load_combo.addItem(p["profile_name"], int(p["profile_id"]))


def _lookup_link_length(gpkg_path: str, link_id: str) -> float:
    """Read link length from coupling results (swe2d_baked_coupling, metric='length')."""
    import sqlite3
    import numpy as np
    try:
        conn = sqlite3.connect(f"file:{gpkg_path}?mode=ro", uri=True)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='swe2d_baked_coupling'"
            )
            if cur.fetchone() is None:
                return 0.0
            cur.execute(
                "SELECT values_blob FROM swe2d_baked_coupling "
                "WHERE component='drainage_link' AND object_id=? AND metric='length' "
                "LIMIT 1",
                (link_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                arr = np.frombuffer(row[0], dtype=np.float64)
                return float(arr[0]) if arr.size else 0.0
            return 0.0
        finally:
            conn.close()
    except Exception:
        return 0.0
