"""Shared real-QGIS test harness.

Provides a process-global ``QgsApplication`` singleton for headless tests,
a ``@requires_qgis`` unittest skip decorator, and a minimal ``stub_iface()``
for code paths that take a ``QgisInterface``.  No silent fallbacks: if
``qgis.core`` is not importable, ``ensure_qgis_app()`` raises and
``requires_qgis`` skips loudly.
"""

import contextlib
import importlib.util
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

_SKIP_REASON = "real QGIS env required (qgis_stable mamba env)"

_app = None


def _ensure_qgis_python_path() -> None:
    """Re-add the conda env's QGIS bindings to ``sys.path`` when missing.

    The qgis_stable env makes ``qgis`` importable via
    ``etc/conda/activate.d/qgis-activate.sh`` (it sets PYTHONPATH), which is
    silently lost whenever an invocation overrides PYTHONPATH (e.g. the
    common ``PYTHONPATH="$PWD:$PWD/build"`` test pattern).  Without this,
    ``import qgis`` fails with "No module named 'qgis'" and every
    ``@requires_qgis``-gated test errors instead of running.
    """
    for sub in ("python", "python/plugins"):
        path = os.path.join(sys.prefix, "share", "qgis", sub)
        if path not in sys.path and os.path.isdir(path):
            sys.path.insert(0, path)


_ensure_qgis_python_path()


def ensure_qgis_app():
    """Create (once) and return the process-global QgsApplication."""
    global _app
    if _app is not None:
        return _app
    try:
        from qgis.core import QgsApplication
    except ImportError as exc:
        raise ImportError(
            "qgis.core is not importable — run tests inside the "
            "qgis_stable mamba env"
        ) from exc
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    existing = QgsApplication.instance()
    if existing is not None:
        _app = existing
        return _app
    _app = QgsApplication([], False)
    _app.initQgis()
    return _app


def requires_qgis(obj):
    """Skip a unittest class or method when real QGIS is not importable."""
    if importlib.util.find_spec("qgis.core") is None:
        return unittest.skip(_SKIP_REASON)(obj)
    return obj


def stub_iface():
    """Minimal QgisInterface stand-in: mainWindow/messageBar/mapCanvas only."""
    iface = mock.MagicMock(name="QgisInterface")
    iface.mainWindow.return_value = mock.MagicMock(name="mainWindow")
    iface.messageBar.return_value = mock.MagicMock(name="messageBar")
    iface.mapCanvas.return_value = mock.MagicMock(name="mapCanvas")
    return iface


def delete_widgets_now(*widgets):
    """deleteLater() + flush deferred deletions immediately.

    Under the offscreen QPA, ``QApplication.processEvents()`` alone does NOT
    deliver ``DeferredDelete`` events — torn-down widgets stay alive (and
    visible) and leak into subsequent test files.  Use this in tearDown
    instead of a bare ``deleteLater()``.
    """
    from qgis.PyQt.QtCore import QEvent
    from qgis.PyQt.QtWidgets import QApplication
    app = QApplication.instance()
    for w in widgets:
        if w is not None:
            w.deleteLater()
    if app is not None:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()


# ---------------------------------------------------------------------------
# Fixture helpers (spec docs/specs/2026-08-02-gui-test-coverage-design.md §4)
#
# Everything below builds REAL artifacts through the production code paths —
# no hand-rolled SQL, no synthetic Qgs* objects.  All helpers raise loudly on
# misuse; there are no silent fallbacks.
# ---------------------------------------------------------------------------


def make_memory_layer(geometry="Point", fields=(), features=(), crs="EPSG:4326",
                      name="test"):
    """Create a real ``QgsVectorLayer`` memory layer with committed features.

    Args:
        geometry: Memory-provider geometry string — one of ``"Point"``,
            ``"LineString"``, ``"Polygon"``, ``"MultiPoint"``,
            ``"MultiLineString"``, ``"MultiPolygon"``, or ``"None"``.
        fields: Sequence of ``(name, qvariant_type)`` tuples, e.g.
            ``[("id", QVariant.Int), ("depth", QVariant.Double)]``.
        features: Sequence of ``(geometry, attr_tuple)`` where *geometry* is
            a WKT string or a ``QgsGeometry`` and ``attr_tuple`` has exactly
            one value per entry in *fields*.
        crs: CRS authority string, e.g. ``"EPSG:4326"``.
        name: Layer display name.

    Returns:
        The ``QgsVectorLayer`` with all features committed.

    Raises:
        RuntimeError: Unknown geometry type or provider rejected the layer.
        TypeError: A field type is not a ``QVariant.Type``.
        ValueError: Bad WKT, attribute-count mismatch, or commit failure.
    """
    ensure_qgis_app()
    from qgis.PyQt.QtCore import QVariant
    from qgis.core import QgsFeature, QgsField, QgsFields, QgsGeometry, QgsVectorLayer

    layer = QgsVectorLayer(f"{geometry}?crs={crs}", name, "memory")
    if not layer.isValid():
        raise RuntimeError(
            f"memory provider rejected geometry={geometry!r} crs={crs!r}"
        )

    qfields = QgsFields()
    for fname, ftype in fields:
        if not isinstance(ftype, QVariant.Type):
            raise TypeError(
                f"field {fname!r}: type must be a QVariant.Type, got {ftype!r}"
            )
        qfields.append(QgsField(str(fname), ftype))
    layer.dataProvider().addAttributes(qfields)
    layer.updateFields()

    feats = []
    for geom, attrs in features:
        if isinstance(geom, str):
            qgeom = QgsGeometry.fromWkt(geom)
            if qgeom.isNull():
                raise ValueError(f"invalid WKT geometry: {geom!r}")
        elif isinstance(geom, QgsGeometry):
            qgeom = geom
        else:
            raise TypeError(
                f"feature geometry must be WKT str or QgsGeometry, got {type(geom).__name__}"
            )
        if len(attrs) != len(fields):
            raise ValueError(
                f"feature has {len(attrs)} attributes but layer has "
                f"{len(fields)} fields"
            )
        feat = QgsFeature(layer.fields())
        feat.setGeometry(qgeom)
        feat.setAttributes(list(attrs))
        feats.append(feat)
    if feats and not layer.dataProvider().addFeatures(feats):
        raise ValueError(f"provider failed to commit {len(feats)} features")
    layer.updateExtents()
    return layer


@contextlib.contextmanager
def make_temp_model_gpkg(crs="EPSG:4326"):
    """Yield a path to a real HYDRA model GeoPackage (all model layers).

    Built through the exact production creation path used by the Studio
    "Create 2D Model GeoPackage" button:
    ``schema_definitions.create_memory_layer`` for every canonical layer key
    (``swe2d/workbench/services/schema_definitions.py``) written by
    ``lumped_hydrology_service.write_memory_layer_to_gpkg``
    (``swe2d/services/lumped_hydrology_service.py``).  Layers are empty, as in
    production.  The temp directory is removed on exit.
    """
    ensure_qgis_app()
    from swe2d.services.lumped_hydrology_service import write_memory_layer_to_gpkg
    from swe2d.workbench.services.schema_definitions import (
        create_memory_layer,
        get_layer_names,
    )

    tmpdir = tempfile.mkdtemp(prefix="hydra_test_model_")
    path = os.path.join(tmpdir, "model.gpkg")
    try:
        keys = get_layer_names()
        for i, key in enumerate(keys):
            lyr = create_memory_layer(key, crs)
            if not lyr.isValid():
                raise RuntimeError(f"schema_definitions produced invalid layer {key!r}")
            write_memory_layer_to_gpkg(lyr, path, lyr.name(), create_file=(i == 0))
        # Loud verification: the production loader must see every layer.
        from swe2d.workbench.services.model_gpkg_loader_service import (
            load_layers_from_gpkg,
        )
        loaded = load_layers_from_gpkg(path)
        missing = sorted(set(keys) - set(loaded))
        if missing:
            raise RuntimeError(
                f"model GPKG at {path} is not loadable via the production "
                f"loader; missing layers: {missing}"
            )
        yield path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def make_temp_results_gpkg(n_cells=4, n_timesteps=3):
    """Yield a path to a real HYDRA results GeoPackage with one run.

    Written by the production results writer
    ``gpkg_persistence_service.persist_baked_results``
    (``swe2d/services/gpkg_persistence_service.py``).  The run
    (``run_id="hydra_test_run"``, ``mesh_name="hydra_test_mesh"``) contains
    deterministic, spatially varying h/hu/hv snapshots plus GPU-style
    ``max_h/max_hu/max_hv`` tracking so plot/browse dialogs have real data.
    Discover the run via ``collect_baked_runs_from_gpkg``.  The temp
    directory is removed on exit.

    Raises:
        ValueError: ``n_cells`` or ``n_timesteps`` is not positive.
        RuntimeError: The written run is not readable back.
    """
    if n_cells <= 0:
        raise ValueError(f"n_cells must be positive, got {n_cells}")
    if n_timesteps <= 0:
        raise ValueError(f"n_timesteps must be positive, got {n_timesteps}")
    ensure_qgis_app()
    import numpy as np
    from swe2d.services.gpkg_persistence_service import (
        collect_baked_runs_from_gpkg,
        persist_baked_results,
    )

    run_id = "hydra_test_run"
    mesh_name = "hydra_test_mesh"
    tmpdir = tempfile.mkdtemp(prefix="hydra_test_results_")
    path = os.path.join(tmpdir, "results.gpkg")
    try:
        snapshots = []
        for k in range(n_timesteps):
            t_s = float(k) * 10.0
            h = 1.0 + 0.1 * k + np.linspace(0.0, 0.5, n_cells)
            hu = 0.01 * (k + 1) * np.arange(1, n_cells + 1, dtype=np.float64)
            hv = -0.005 * k * np.arange(1, n_cells + 1, dtype=np.float64)
            snapshots.append((t_s, h, hu, hv))
        persist_baked_results(
            gpkg_path=path,
            run_id=run_id,
            mesh_name=mesh_name,
            snapshot_timesteps=snapshots,
            max_tracking={
                "max_h": np.maximum.reduce([s[1] for s in snapshots]),
                "max_hu": np.maximum.reduce([s[2] for s in snapshots]),
                "max_hv": np.maximum.reduce([s[3] for s in snapshots]),
            },
        )
        # Loud readback through the production reader.
        runs = collect_baked_runs_from_gpkg(path)
        if not any(r["run_id"] == run_id for r in runs):
            raise RuntimeError(
                f"persist_baked_results wrote {path} but the production "
                f"reader finds no run {run_id!r}"
            )
        yield path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def grab_non_empty(widget) -> bool:
    """Return True if an offscreen ``widget.grab()`` shows rendered content.

    Grabs the widget to a ``QImage`` and returns True when at least one pixel
    differs from the widget's background-role color (pattern from
    ``tests/test_high_perf_viewer.py``).

    Raises:
        ValueError: *widget* is None or has a zero-size frame.
        RuntimeError: The grab produced a null image.
    """
    if widget is None:
        raise ValueError("grab_non_empty: widget is None")
    ensure_qgis_app()
    if widget.width() <= 0 or widget.height() <= 0:
        raise ValueError(
            f"grab_non_empty: widget has zero size "
            f"({widget.width()}x{widget.height()}); resize it before grabbing"
        )
    image = widget.grab().toImage()
    if image.isNull():
        raise RuntimeError("widget.grab() produced a null QImage")
    bg = widget.palette().color(widget.backgroundRole()).rgb()
    for x in range(image.width()):
        for y in range(image.height()):
            if image.pixel(x, y) != bg:
                return True
    return False
