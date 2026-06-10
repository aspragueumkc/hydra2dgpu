"""
Mock QGIS environment for headless GUI testing.

Call ``install_qgis_mocks()`` before importing any ``swe2d`` module that
imports from ``qgis.core``, ``qgis.gui``, ``qgis.PyQt``, or ``qgis.utils``.

Usage::

    from tests.mocks.qgis_env import install_qgis_mocks
    install_qgis_mocks()

    from swe2d_workbench_qt import SWE2DWorkbenchQtDialog
    from hvac.workbench.project_settings import load_project_json
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, PropertyMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Semi-functional mock class factory
# ═══════════════════════════════════════════════════════════════════════════════

class _MockMeta(type):
    """Metaclass that auto-generates missing attributes as MagicMock children."""
    def __getattr__(cls, name):
        return MagicMock(name=f"{cls.__name__}.{name}")


class MockQgsObject(metaclass=_MockMeta):
    """Base class for mock QGIS objects with auto-generated children.

    Subclasses define only the methods/properties actually used by the codebase.
    All other attribute access returns a MagicMock.
    """
    _mock_attrs: dict = {}

    def __init__(self, *args, **kwargs):
        for k, v in self._mock_attrs.items():
            setattr(self, k, v if not callable(v) else v)

    def __getattr__(self, name):
        if name.startswith('_') or name in ('_mock_attrs',):
            return super().__getattribute__(name)
        return MagicMock(name=f"{type(self).__name__}.{name}")


# ═══════════════════════════════════════════════════════════════════════════════
# QgsPointXY
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgsPointXY(MockQgsObject):
    def __init__(self, x=0.0, y=0.0):
        super().__init__()
        self._x = float(x)
        self._y = float(y)

    def x(self) -> float:
        return self._x

    def y(self) -> float:
        return self._y


# ═══════════════════════════════════════════════════════════════════════════════
# QgsGeometry
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgsGeometry(MockQgsObject):
    _from_cache: dict = {}

    def __init__(self, wkt="", point=None, polyline=None, polygon=None):
        super().__init__()
        self._wkt = wkt
        self._point = point
        self._polyline = polyline or []
        self._polygon = polygon
        self._empty = not bool(wkt or point or polyline or polygon)

    @staticmethod
    def fromPointXY(pt) -> "MockQgsGeometry":
        return MockQgsGeometry(point=pt)

    @staticmethod
    def fromPolylineXY(pts) -> "MockQgsGeometry":
        return MockQgsGeometry(polyline=pts)

    @staticmethod
    def fromPolygonXY(rings) -> "MockQgsGeometry":
        return MockQgsGeometry(polygon=rings)

    @staticmethod
    def fromWkt(wkt: str) -> "MockQgsGeometry":
        return MockQgsGeometry(wkt=wkt)

    def asPoint(self) -> MockQgsPointXY:
        return self._point or MockQgsPointXY()

    def isEmpty(self) -> bool:
        return self._empty

    def contains(self, point) -> bool:
        return True

    def intersects(self, geom) -> bool:
        return True

    def centroid(self) -> "MockQgsGeometry":
        return MockQgsGeometry(point=MockQgsPointXY(0.5, 0.5))

    def area(self) -> float:
        return 1.0

    def length(self) -> float:
        return 1.0

    def distance(self, geom) -> float:
        return 0.0

    def interpolate(self, distance: float) -> "MockQgsGeometry":
        return MockQgsGeometry(point=MockQgsPointXY(distance, 0.0))

    def lineLocatePoint(self, point) -> float:
        return 0.5

    def boundingBox(self) -> MagicMock:
        return MagicMock()

    def wkbType(self) -> int:
        return 1  # PointGeometry


# ═══════════════════════════════════════════════════════════════════════════════
# QgsField
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgsField(MockQgsObject):
    def __init__(self, name="field", field_type=0):
        super().__init__()
        self._name = str(name)
        self._type = int(field_type)

    def name(self) -> str:
        return self._name

    def type(self) -> int:
        return self._type


# ═══════════════════════════════════════════════════════════════════════════════
# QgsFeature
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgsFields(MockQgsObject):
    def __init__(self, field_names=None):
        super().__init__()
        self._names = list(field_names or [])
        self._fields = {n: MockQgsField(n) for n in self._names}
        self._name_to_idx = {n: i for i, n in enumerate(self._names)}

    def names(self) -> list:
        return list(self._names)

    def indexOf(self, name: str) -> int:
        return self._name_to_idx.get(name, -1)

    def indexFromName(self, name: str) -> int:
        return self._name_to_idx.get(name, -1)

    def __iter__(self):
        return iter(list(self._fields.values()))

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(self, idx):
        name = self._names[idx] if idx < len(self._names) else ""
        return self._fields.get(name, MockQgsField(name))

    def toList(self) -> list:
        return list(self._fields.values())


class MockQgsFeature(MockQgsObject):
    _next_id = 0

    def __init__(self, fields=None, attributes: dict = None):
        super().__init__()
        self._fid = MockQgsFeature._next_id
        MockQgsFeature._next_id += 1
        self._fields = fields or MockQgsFields()
        self._geom = MockQgsGeometry()
        self._attrs = dict(attributes or {})

    def id(self) -> int:
        return self._fid

    def geometry(self) -> MockQgsGeometry:
        return self._geom

    def setGeometry(self, geom) -> None:
        self._geom = geom

    def attributes(self) -> list:
        return list(self._attrs.values())

    def __getitem__(self, key):
        return self._attrs.get(key, None)

    def __setitem__(self, key, value):
        self._attrs[key] = value

    def fieldNameIndex(self, name: str) -> int:
        return self._fields.indexOf(name)


# ═══════════════════════════════════════════════════════════════════════════════
# QgsVectorLayer Data Provider (mock)
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgsVectorDataProvider(MockQgsObject):
    def __init__(self):
        super().__init__()
        self._features: list = []
        self._fields: MockQgsFields = MockQgsFields()
        self._uri = "mock:memory"

    def addFeatures(self, features) -> bool:
        self._features.extend(features)
        return True

    def addAttributes(self, fields) -> bool:
        return True

    def changeAttributeValues(self, changes) -> bool:
        return True

    def dataSourceUri(self) -> str:
        return self._uri

    def fields(self) -> MockQgsFields:
        return self._fields


# ═══════════════════════════════════════════════════════════════════════════════
# QgsVectorLayer
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgsVectorLayer(MockQgsObject):
    def __init__(self, path="", name="mock", provider="memory"):
        super().__init__()
        self._path = path
        self._layer_name = name
        self._provider = provider
        self._valid = True
        self._fields = MockQgsFields()
        self._data_provider = MockQgsVectorDataProvider()
        self._features: list = []
        self._editable = False
        self._layer_id = f"mock_layer_{id(self)}"
        self._geom_type = 0

    def isValid(self) -> bool:
        return self._valid

    def name(self) -> str:
        return self._layer_name

    def id(self) -> str:
        return self._layer_id

    def fields(self) -> MockQgsFields:
        return self._fields

    def dataProvider(self) -> MockQgsVectorDataProvider:
        return self._data_provider

    def getFeatures(self):
        return iter(self._features)

    def isEditable(self) -> bool:
        return self._editable

    def startEditing(self) -> bool:
        self._editable = True
        return True

    def commitChanges(self) -> bool:
        self._editable = False
        return True

    def rollBack(self) -> bool:
        self._editable = False
        return True

    def triggerRepaint(self) -> None:
        pass

    def changeAttributeValue(self, fid, idx, value) -> bool:
        return True

    def setDataSource(self, uri, name, provider) -> None:
        self._path = uri
        self._layer_name = name
        self._provider = provider

    def geometryType(self) -> int:
        return self._geom_type

    def setEditFormConfig(self, cfg) -> None:
        pass

    def editFormConfig(self):
        return MagicMock()

    def setFieldAlias(self, idx, alias) -> None:
        pass

    def setEditorWidgetSetup(self, idx, setup) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# QgsRasterLayer
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgsRasterLayer(MockQgsObject):
    def __init__(self, path="", name="mock_raster", provider="gdal"):
        super().__init__()
        self._valid = True
        self._data_provider = MagicMock()
        self._data_provider.sample.return_value = (0.0, True)

    def isValid(self) -> bool:
        return self._valid

    def name(self) -> str:
        return "mock_raster"

    def dataProvider(self) -> MagicMock:
        return self._data_provider


# ═══════════════════════════════════════════════════════════════════════════════
# QgsProject
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgsProject(MockQgsObject):
    _instance = None

    def __init__(self):
        super().__init__()
        self._layers: dict = {}
        self._settings: dict = {}

    @staticmethod
    def instance() -> "MockQgsProject":
        if MockQgsProject._instance is None:
            MockQgsProject._instance = MockQgsProject()
        return MockQgsProject._instance

    def crs(self):
        return MagicMock()

    def fileName(self) -> str:
        return "/mock/project.qgs"

    def mapLayers(self) -> dict:
        return dict(self._layers)

    def mapLayersByName(self, name: str) -> list:
        return [lyr for lyr in self._layers.values() if lyr.name() == name]

    def mapLayer(self, layer_id: str) -> MockQgsVectorLayer | None:
        return self._layers.get(layer_id)

    def addMapLayer(self, layer) -> str:
        self._layers[layer.id()] = layer
        return layer.id()

    def removeMapLayer(self, layer_id) -> None:
        self._layers.pop(layer_id, None)

    def layerTreeRoot(self):
        return MagicMock()

    def readEntry(self, scope, key, default="") -> tuple:
        val = self._settings.get((scope, key), default)
        return (val, True)

    def writeEntry(self, scope, key, value) -> bool:
        self._settings[(scope, key)] = value
        return True

    def transformContext(self):
        return MagicMock()

    def clear(self) -> None:
        self._layers.clear()
        self._settings.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# QgsWkbTypes
# ═══════════════════════════════════════════════════════════════════════════════

class _QgsWkbTypesGeometryType:
    LineGeometry = 1
    PointGeometry = 2
    PolygonGeometry = 3
    UnknownGeometry = 4


class MockQgsWkbTypes(MockQgsObject):
    GeometryType = _QgsWkbTypesGeometryType()

    LineString = 5
    LineString25D = 1001
    LineStringZ = 1002

    @staticmethod
    def geometryType(wkb_type):
        return _QgsWkbTypesGeometryType.PolygonGeometry

    @staticmethod
    def isMultiType(gtype) -> bool:
        return False

    @staticmethod
    def flatType(gtype) -> int:
        return gtype


# ═══════════════════════════════════════════════════════════════════════════════
# QgsEditFormConfig
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgsEditFormConfig(MockQgsObject):
    DragAndDrop = 1
    UiFileLayout = 2

    def __init__(self):
        super().__init__()
        self._layout = 0

    def setLayout(self, layout) -> None:
        self._layout = layout

    def setUiForm(self, path) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# QgsUnitTypes
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgsUnitTypes(MockQgsObject):
    DistanceMeters = 0
    DistanceFeet = 1
    DistanceUSSurveyFeet = 2

    @staticmethod
    def toString(unit) -> str:
        return {0: "meters", 1: "feet", 2: "US feet"}.get(unit, "unknown")


# ═══════════════════════════════════════════════════════════════════════════════
# QgsInterface (qgis.utils.iface)
# ═══════════════════════════════════════════════════════════════════════════════

class MockQgisInterface(MockQgsObject):
    def __init__(self):
        super().__init__()
        self._main_window = MagicMock()
        self._message_bar = MagicMock()
        self._map_canvas = MagicMock()

    def mainWindow(self):
        return self._main_window

    def messageBar(self):
        return self._message_bar

    def mapCanvas(self):
        return self._map_canvas


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level QGIS module install
# ═══════════════════════════════════════════════════════════════════════════════

_INSTALLED = False


def install_qgis_mocks():
    """Install mock ``qgis.*`` and ``PyQt5`` modules into ``sys.modules``.

    Safe to call multiple times (idempotent).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    # ── Ensure PyQt5 is importable ─────────────────────────────────────
    _install_pyqt5_mocks()

    # ── qgis.core ──────────────────────────────────────────────────────
    qgis_core = types.ModuleType("qgis.core")
    qgis_core.QgsProject = MockQgsProject
    qgis_core.QgsVectorLayer = MockQgsVectorLayer
    qgis_core.QgsRasterLayer = MockQgsRasterLayer
    qgis_core.QgsFeature = MockQgsFeature
    qgis_core.QgsFields = MockQgsFields
    qgis_core.QgsGeometry = MockQgsGeometry
    qgis_core.QgsPointXY = MockQgsPointXY
    qgis_core.QgsField = MockQgsField
    qgis_core.QgsWkbTypes = MockQgsWkbTypes
    qgis_core.QgsEditFormConfig = MockQgsEditFormConfig
    qgis_core.QgsUnitTypes = MockQgsUnitTypes
    qgis_core.QgsEditorWidgetSetup = MagicMock  # simple enough
    qgis_core.QgsFieldConstraints = MagicMock
    qgis_core.QgsMeshLayer = MagicMock
    qgis_core.QgsVectorFileWriter = MagicMock
    qgis_core.Qgis = type("Qgis", (), {"Success": 0, "Warning": 1, "Critical": 2})()
    sys.modules["qgis.core"] = qgis_core

    # ── qgis.gui ───────────────────────────────────────────────────────
    qgis_gui = types.ModuleType("qgis.gui")
    qgis_gui.QgsMapTool = MagicMock
    qgis_gui.QgsRubberBand = MagicMock
    qgis_gui.QgsDockWidget = MagicMock
    qgis_gui.QgsMapCanvasItem = MagicMock
    sys.modules["qgis.gui"] = qgis_gui

    # ── qgis.PyQt ──────────────────────────────────────────────────────
    # Redirect to real PyQt5 if available, otherwise mock
    qgis_pyqt = types.ModuleType("qgis.PyQt")
    sys.modules["qgis.PyQt"] = qgis_pyqt
    # Individual sub-modules: delegate to real PyQt5 or mock
    for _sub in ("QtCore", "QtWidgets", "QtGui", "uic"):
        _install_qgis_pyqt_submodule(_sub)

    # ── qgis.utils ─────────────────────────────────────────────────────
    qgis_utils = types.ModuleType("qgis.utils")
    qgis_utils.iface = MockQgisInterface()
    sys.modules["qgis.utils"] = qgis_utils

    # ── External packages that may not be installed ────────────────────
    # culvert_routine: used by swe2d.extensions.drainage_network at import
    # Must provide the exact symbols imported by the codebase
    _install_dummy_module("culvert_routine", [
        "CircularXsect", "RectangularXsect",
        "direct_step_culvert_upstream_energy", "inlet_controlled_flow",
    ])

    # swe2d_high_perf_viewer (standalone in repo root, uses qgis.PyQt)
    if "swe2d_high_perf_viewer" not in sys.modules:
        m = types.ModuleType("swe2d_high_perf_viewer")
        m.render_unstructured_snapshot_image = MagicMock(
            return_value=bytearray(1024))
        sys.modules["swe2d_high_perf_viewer"] = m


def _install_dummy_module(mod_name: str, public_symbols: list = None):
    """Install a dummy module with MagicMock attributes."""
    if mod_name in sys.modules:
        return
    m = types.ModuleType(mod_name)
    if public_symbols:
        for sym in public_symbols:
            setattr(m, sym, MagicMock(name=f"{mod_name}.{sym}"))
    sys.modules[mod_name] = m


def _install_pyqt5_mocks():
    """Ensure PyQt5 modules are importable (use real ones or mock)."""
    try:
        import PyQt5  # noqa: F401
    except ImportError:
        # Create minimal mock PyQt5
        for _mod in ("PyQt5", "PyQt5.QtCore", "PyQt5.QtWidgets", "PyQt5.QtGui", "PyQt5.uic"):
            if _mod not in sys.modules:
                m = types.ModuleType(_mod)
                _Qt = type("Qt", (), {})
                _Qt.WindowModality = type("WindowModality", (), {
                    "NonModal": 0, "WindowModal": 1, "ApplicationModal": 2,
                })()
                _Qt.AlignLeft = 1
                _Qt.AlignRight = 2
                _Qt.AlignCenter = 4
                _Qt.Horizontal = 1
                _Qt.Vertical = 2
                _Qt.WindowFlags = type("WindowFlags", (), {})()
                m.Qt = _Qt
                m.pyqtSignal = MagicMock
                m.QObject = MagicMock
                m.QTimer = lambda *a, **kw: MagicMock(name="QTimer")
                m.QWidget = MagicMock
                # Qt base classes need proper __init__ for MRO
                class _MockQWidget:
                    def __init__(self, parent=None, *args, **kwargs):
                        self._mock_children = {}
                    def __getattr__(self, name):
                        if name.startswith('__'):
                            raise AttributeError(name)
                        return MagicMock(name=f"QWidget.{name}")
                class _MockQDialog(_MockQWidget):
                    def __init__(self, parent=None, *args, **kwargs):
                        super().__init__(parent)
                class _MockQMainWindow(_MockQWidget):
                    def __init__(self, parent=None, *args, **kwargs):
                        super().__init__(parent)
                class _MockQDockWidget(_MockQWidget):
                    def __init__(self, parent=None, *args, **kwargs):
                        super().__init__(parent)
                m.QWidget = _MockQWidget
                m.QDialog = _MockQDialog
                m.QMainWindow = _MockQMainWindow
                m.QDockWidget = _MockQDockWidget
                m.QAction = MagicMock
                m.QMenu = MagicMock
                m.QApplication = MagicMock
                m.QVariant = type("QVariant", (), {})  # old-style
                m.QStyledItemDelegate = type("QStyledItemDelegate", (MagicMock,), {})
                m.QGraphicsScene = MagicMock
                m.QGraphicsView = MagicMock
                m.QGraphicsPolygonItem = MagicMock
                m.QGraphicsLineItem = MagicMock
                m.QGraphicsTextItem = MagicMock
                m.QColor = MagicMock
                m.QPen = MagicMock
                m.QBrush = MagicMock
                m.QPainter = MagicMock
                m.QImage = MagicMock
                m.QPixmap = MagicMock
                m.QFont = MagicMock
                m.QPalette = MagicMock
                m.QSizePolicy = MagicMock
                m.QSpacerItem = MagicMock
                m.QHBoxLayout = MagicMock
                m.QVBoxLayout = MagicMock
                m.QGridLayout = MagicMock
                m.QScrollArea = MagicMock
                m.QSplitter = MagicMock
                m.QTabWidget = MagicMock
                m.QGroupBox = MagicMock
                m.QLabel = MagicMock
                m.QPushButton = MagicMock
                m.QComboBox = MagicMock
                m.QLineEdit = MagicMock
                m.QTextEdit = MagicMock
                m.QSpinBox = MagicMock
                m.QDoubleSpinBox = MagicMock
                m.QCheckBox = MagicMock
                m.QRadioButton = MagicMock
                m.QSlider = MagicMock
                m.QProgressBar = MagicMock
                m.QTableView = MagicMock
                m.QTreeView = MagicMock
                m.QListView = MagicMock
                m.QHeaderView = MagicMock
                m.QFileDialog = MagicMock
                m.QMessageBox = MagicMock
                m.QInputDialog = MagicMock
                m.QMenuBar = MagicMock
                m.QToolBar = MagicMock
                m.QStatusBar = MagicMock
                m.QDockWidget = MagicMock
                m.QToolButton = MagicMock
                m.QFrame = MagicMock
                sys.modules[_mod] = m


def _install_qgis_pyqt_submodule(sub: str):
    """Install ``qgis.PyQt.<sub>`` pointing to real PyQt5 or mock."""
    full = f"qgis.PyQt.{sub}"
    if full in sys.modules:
        return
    # Try the real PyQt5 first
    real_mod = f"PyQt5.{sub}" if sub != "uic" else "PyQt5.uic"
    try:
        __import__(real_mod)
        real = sys.modules[real_mod]
        m = types.ModuleType(full)
        # Copy all public symbols from real module
        for _k, _v in vars(real).items():
            if not _k.startswith("_"):
                setattr(m, _k, _v)
        mod_file = getattr(real, "__file__", None)
        if mod_file:
            m.__file__ = mod_file
        sys.modules[full] = m
    except ImportError:
        # Fall back to mock
        _install_pyqt5_mocks()
        if sys.modules.get(full) is None:
            sys.modules[full] = sys.modules.get(real_mod, MagicMock())
