"""Characterization tests for ``swe2d.workbench.views.profile_chain_widget``.

Covers ``ProfileChainWidget`` (chain building workflow) and
``_lookup_link_length`` per Task D.2 of
``docs/plans/2026-08-02-gui-test-coverage.md`` and patterns P1+P2 of
``docs/specs/2026-08-02-gui-test-coverage-design.md`` §3-§4.

Fixture strategy: a real GPKG whose drainage layers are written by
``QgsVectorFileWriter`` (the ``TestDrainageGpkgAdapter`` pattern) under the
lowercase table names the production sqlite readers query, plus
``swe2d_baked_coupling`` length rows written by the production
``persist_baked_coupling`` writer.  ``DrainageGraph`` comes from the
production ``load_drainage_graph`` reader.  No mocks.

Network fixture (lengths equal the committed LineString geometry lengths)::

    N1 --L1(100)--> N2 --L2(150)--> N3
                        ^--L3(40)--- N3   (L3: N3 -> N2, points upstream)
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from tests.qgis_real_env import (
    delete_widgets_now,
    ensure_qgis_app,
    grab_non_empty,
    make_temp_model_gpkg,
    requires_qgis,
)

from swe2d.services.gpkg_persistence_service import persist_baked_coupling
from swe2d.workbench.services.drainage_graph_service import load_drainage_graph
from swe2d.workbench.services.profile_persistence_service import save_profile
from swe2d.workbench.services.profile_pipeline_service import ChainSpec
from swe2d.workbench.views.profile_chain_widget import (
    ProfileChainWidget,
    _lookup_link_length,
)

_RUN_ID = "hydra_test_run"

# (link_id, from_node, to_node, geometry length, WKT)
_LINKS = (
    ("L1", "N1", "N2", 100.0, "LINESTRING(0 0, 100 0)"),
    ("L2", "N2", "N3", 150.0, "LINESTRING(100 0, 100 -150)"),
    ("L3", "N3", "N2", 40.0, "LINESTRING(100 -150, 60 -150)"),
)
_NODES = ("N1", "N2", "N3")


def _memory_layer(wkb_name, name, fields):
    from qgis.core import QgsVectorLayer

    layer = QgsVectorLayer(f"{wkb_name}?crs=EPSG:4326", name, "memory")
    if not layer.isValid():
        raise RuntimeError(f"memory provider rejected {wkb_name}")
    layer.dataProvider().addAttributes(fields)
    layer.updateFields()
    return layer


def _write_layer_to_gpkg(layer, gpkg_path, layer_name, *, first):
    from qgis.core import QgsCoordinateTransformContext, QgsVectorFileWriter

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = layer_name
    options.actionOnExistingFile = (
        QgsVectorFileWriter.CreateOrOverwriteFile
        if first
        else QgsVectorFileWriter.CreateOrOverwriteLayer
    )
    err, _new_name, msg, _new_layer = QgsVectorFileWriter.writeAsVectorFormatV3(
        layer, gpkg_path, QgsCoordinateTransformContext(), options
    )
    if err != QgsVectorFileWriter.NoError:
        raise RuntimeError(f"GPKG write of {layer_name} failed: {msg}")


def _persist_lengths(gpkg_path: str) -> None:
    """Coupling length rows via the production writer — the table
    ``_lookup_link_length`` reads (metric='length', values_blob[0])."""
    for lid, fn, tn, length, _wkt in _LINKS:
        persist_baked_coupling(
            gpkg_path=gpkg_path,
            run_id=_RUN_ID,
            component="drainage_link",
            object_id=lid,
            object_name=f"{fn} -> {tn}",
            metric="length",
            times=np.array([0.0], dtype=np.float64),
            values=np.array([length], dtype=np.float64),
        )


def _make_drainage_gpkg(testcase) -> str:
    """Real drainage GPKG via QgsVectorFileWriter + production coupling writer.

    Registers the tmpdir cleanup on *testcase* and returns the file path.
    """
    from qgis.PyQt.QtCore import QVariant
    from qgis.core import QgsFeature, QgsField, QgsGeometry

    tmpdir = tempfile.mkdtemp(prefix="hydra_test_drainage_")
    testcase.addCleanup(shutil.rmtree, tmpdir, True)
    gpkg = os.path.join(tmpdir, "drainage.gpkg")

    node_layer = _memory_layer("Point", "swe2d_drainage_nodes", [
        QgsField("node_id", QVariant.String),
        QgsField("invert_elev", QVariant.Double),
        QgsField("rim_elev", QVariant.Double),
    ])
    for i, nid in enumerate(_NODES):
        feat = QgsFeature(node_layer.fields())
        feat.setGeometry(QgsGeometry.fromWkt(f"POINT({i} {i})"))
        feat.setAttribute("node_id", nid)
        feat.setAttribute("invert_elev", 0.0)
        feat.setAttribute("rim_elev", 1.0)
        node_layer.dataProvider().addFeatures([feat])
    _write_layer_to_gpkg(node_layer, gpkg, "swe2d_drainage_nodes", first=True)

    link_layer = _memory_layer("LineString", "swe2d_drainage_links", [
        QgsField("link_id", QVariant.String),
        QgsField("from_node", QVariant.String),
        QgsField("to_node", QVariant.String),
        QgsField("length", QVariant.Double),
    ])
    for lid, fn, tn, length, wkt in _LINKS:
        geom = QgsGeometry.fromWkt(wkt)
        # Loud self-check: the WKT really encodes the stated length.
        if abs(geom.length() - length) > 1e-9:
            raise ValueError(
                f"fixture WKT for {lid} has length {geom.length()}, expected {length}"
            )
        feat = QgsFeature(link_layer.fields())
        feat.setGeometry(geom)
        feat.setAttribute("link_id", lid)
        feat.setAttribute("from_node", fn)
        feat.setAttribute("to_node", tn)
        feat.setAttribute("length", length)
        link_layer.dataProvider().addFeatures([feat])
    _write_layer_to_gpkg(link_layer, gpkg, "swe2d_drainage_links", first=False)

    _persist_lengths(gpkg)
    return gpkg


@requires_qgis
class TestLookupLinkLength(unittest.TestCase):
    """P1: _lookup_link_length against a real coupling GPKG."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_known_link_returns_exact_length(self):
        gpkg = _make_drainage_gpkg(self)
        self.assertEqual(_lookup_link_length(gpkg, "L1"), 100.0)
        self.assertEqual(_lookup_link_length(gpkg, "L2"), 150.0)
        self.assertEqual(_lookup_link_length(gpkg, "L3"), 40.0)

    def test_missing_link_id_returns_zero(self):
        # Production contract: an unknown object_id yields 0.0 (no row found).
        gpkg = _make_drainage_gpkg(self)
        self.assertEqual(_lookup_link_length(gpkg, "NOPE"), 0.0)

    def test_missing_coupling_table_returns_zero(self):
        # Production contract: no swe2d_baked_coupling table → 0.0.
        with make_temp_model_gpkg() as gpkg:
            self.assertEqual(_lookup_link_length(gpkg, "L1"), 0.0)

    def test_missing_file_returns_zero(self):
        # Production contract: broad except → 0.0 for an unreadable path.
        self.assertEqual(_lookup_link_length("/no/such/file.gpkg", "L1"), 0.0)


@requires_qgis
class TestDrainageGraphOnProductionModelGpkg(unittest.TestCase):
    """Regression: load_drainage_graph vs the production
    model-GPKG creation path (display-case table names)."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def test_load_drainage_graph_loads_on_production_model_gpkg(self):
        # The production model-GPKG creation path
        # (schema_definitions.create_memory_layer + write_memory_layer_to_gpkg)
        # writes drainage tables under display names (SWE2D_Drainage_Links);
        # load_drainage_graph's existence check must be case-insensitive so
        # the graph loads on a production-created model GPKG.
        from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer

        with make_temp_model_gpkg() as gpkg:
            links = QgsVectorLayer(
                f"{gpkg}|layername=swe2d_drainage_links", "links", "ogr"
            )
            self.assertTrue(links.isValid())
            links.startEditing()
            feat = QgsFeature(links.fields())
            feat.setGeometry(QgsGeometry.fromWkt("LINESTRING(0 0, 100 0)"))
            feat.setAttribute("link_id", "L1")
            feat.setAttribute("from_node", "N1")
            feat.setAttribute("to_node", "N2")
            self.assertTrue(links.addFeature(feat))
            self.assertTrue(links.commitChanges())
            del links

            graph = load_drainage_graph(gpkg)
            self.assertEqual(graph.link_ids, ["L1"])
            self.assertEqual(graph.from_node["L1"], "N1")
            self.assertEqual(graph.to_node["L1"], "N2")


@requires_qgis
class TestProfileChainWidget(unittest.TestCase):
    """P2: chain-building workflow on the real widget."""

    @classmethod
    def setUpClass(cls):
        ensure_qgis_app()

    def setUp(self):
        self.gpkg = _make_drainage_gpkg(self)
        # Graph built through the production sqlite reader.
        self.graph = load_drainage_graph(self.gpkg)
        assert self.graph.link_ids == ["L1", "L2", "L3"], (
            f"fixture graph wrong: {self.graph.link_ids}"
        )
        self.widget = ProfileChainWidget()
        self.widget.set_context(self.gpkg, self.graph)

    def tearDown(self):
        delete_widgets_now(self.widget)

    def test_set_context_populates_node_combos(self):
        a_items = [
            self.widget._node_a_combo.itemText(i)
            for i in range(self.widget._node_a_combo.count())
        ]
        b_items = [
            self.widget._node_b_combo.itemText(i)
            for i in range(self.widget._node_b_combo.count())
        ]
        self.assertEqual(a_items, ["N1", "N2", "N3"])
        self.assertEqual(b_items, ["N1", "N2", "N3"])

    def test_add_links_builds_ordered_chain_and_status(self):
        self.widget.add_link_id("L1")
        self.widget.add_link_id("L2")
        chain = self.widget.get_chain()
        self.assertEqual(chain.link_specs, [("L1", False), ("L2", False)])
        self.assertEqual(self.widget._list.count(), 2)
        status = self.widget._status_lbl.text()
        self.assertIn("2 link(s)", status)
        # Total length read back through _lookup_link_length (100 + 150).
        self.assertIn("250.0", status)
        self.assertIn("L1", status)
        self.assertIn("L2", status)

    def test_add_duplicate_link_ignored(self):
        self.widget.add_link_id("L1")
        self.widget.add_link_id("L1")
        self.assertEqual(self.widget.get_chain().link_specs, [("L1", False)])

    def test_add_link_without_context_is_noop(self):
        # Production contract: graph is None → silent no-op.
        bare = ProfileChainWidget()
        try:
            bare.add_link_id("L1")
            self.assertTrue(bare.get_chain().is_empty())
        finally:
            delete_widgets_now(bare)

    def test_orientation_reverse_when_link_points_upstream(self):
        # L1 ends at N2; L3 is N3 -> N2, so its to_node meets the chain end
        # → added as reversed.
        self.widget.add_link_id("L1")
        self.widget.add_link_id("L3")
        self.assertEqual(
            self.widget.get_chain().link_specs,
            [("L1", False), ("L3", True)],
        )

    def test_chain_changed_signal_emitted_with_chainspec(self):
        from qgis.PyQt.QtTest import QSignalSpy

        spy = QSignalSpy(self.widget.chain_changed)
        self.widget.add_link_id("L1")
        self.assertEqual(len(spy), 1)
        emitted = spy[0][0]
        self.assertIsInstance(emitted, ChainSpec)
        self.assertEqual(emitted.link_specs, [("L1", False)])

    def test_reverse_move_remove_clear_slots(self):
        self.widget.add_link_id("L1")
        self.widget.add_link_id("L2")

        # Reverse the first link.
        self.widget._list.setCurrentRow(0)
        self.widget._on_reverse()
        self.assertEqual(
            self.widget.get_chain().link_specs, [("L1", True), ("L2", False)]
        )

        # Move the first link down.
        self.widget._list.setCurrentRow(0)
        self.widget._on_move_down()
        self.assertEqual(
            self.widget.get_chain().link_specs, [("L2", False), ("L1", True)]
        )

        # Move it back up.
        self.widget._list.setCurrentRow(1)
        self.widget._on_move_up()
        self.assertEqual(
            self.widget.get_chain().link_specs, [("L1", True), ("L2", False)]
        )

        # Remove the second link.
        self.widget._list.setCurrentRow(1)
        self.widget._on_remove()
        self.assertEqual(self.widget.get_chain().link_specs, [("L1", True)])

        # Clear.
        self.widget._on_clear()
        self.assertTrue(self.widget.get_chain().is_empty())
        self.assertEqual(self.widget._list.count(), 0)

    def test_find_path_builds_chain(self):
        self.widget._node_a_combo.setCurrentText("N1")
        self.widget._node_b_combo.setCurrentText("N3")
        self.widget._on_find_path()
        self.assertEqual(
            self.widget.get_chain().link_specs,
            [("L1", False), ("L2", False)],
        )

    def test_save_then_load_roundtrip_via_production_service(self):
        # Save through the production persistence service (the widget's
        # _on_save path is modal QInputDialog-driven), refresh the load
        # combo, then drive the real _on_load slot and read back the chain.
        self.widget.add_link_id("L1")
        self.widget.add_link_id("L2")
        original = self.widget.get_chain()
        save_profile(self.gpkg, "fixture-profile", original)

        self.widget._on_clear()
        self.assertTrue(self.widget.get_chain().is_empty())

        self.widget._refresh_load_combo()
        self.assertEqual(self.widget._load_combo.count(), 1)
        self.widget._load_combo.setCurrentIndex(0)
        self.widget._on_load()
        self.assertEqual(
            self.widget.get_chain().link_specs, original.link_specs
        )

    def test_grab_renders_content(self):
        self.widget.add_link_id("L1")
        self.widget.resize(400, 300)
        self.assertTrue(grab_non_empty(self.widget))

    # --- Production findings: invalid link ids --------------------------

    def test_unknown_first_link_is_silently_accepted(self):
        # FINDING: add_link_id validates the new id only against the
        # *existing* chain, never against the graph — an unknown id as the
        # first link is silently accepted into the chain.
        self.widget.add_link_id("BOGUS")
        self.assertEqual(self.widget.get_chain().link_specs, [("BOGUS", False)])

    def test_unknown_link_after_valid_raises_keyerror(self):
        # FINDING: the same unknown id added *after* a valid link crashes
        # with KeyError (graph.from_node[link_id]) — inconsistent with the
        # silent acceptance of an unknown first link.
        self.widget.add_link_id("L1")
        with self.assertRaises(KeyError):
            self.widget.add_link_id("BOGUS")


if __name__ == "__main__":
    unittest.main()
