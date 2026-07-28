---
type: spec
status: complete
created: 2026-07-19
completed: 2026-07-25
---

# Network Profile Viewer — Design Spec

## Overview

Bring SWMM 5.2.4-style longitudinal profile viewing to the SWE2D workbench:
a standalone `NetworkProfileDialog` that lets the user build a chain of drainage
links (by clicking on the QGIS map or by BFS auto-path), then renders a
longitudinal cross-section over the chain at any chosen timestep, with invert,
crown, ground/rim, water surface (HGL), water-filled pipe polygon, and node
"cylinders".

The feature reuses our existing single-link profile plotter primitives
(in `studio_viewer_profile_pg.py:708–872`) and our existing
`swe2d_baked_pipe_cell_ts` per-cell time-series storage (depth, velocity, flow,
head, invert, width/height/shape).

## Architecture

```
NetworkProfileDialog (QDialog)
├── ProfileChainWidget       (left: chain editor)
│   ├── "Pick on Map"  →  activates NetworkProfileMapTool on QGIS canvas
│   ├── "Find Shortest Path"  →  drainage_graph_service.find_chain
│   ├── "Add/Remove/Save/Load"  →  profile_persistence_service
│   └── QListWidget of (link_id, reverse_orientation)
│
├── NetworkProfilePlotWidget (right: matplotlib canvas)
│   ├── invert polyline
│   ├── crown polyline
│   ├── ground/rim polyline (interp between nodes)
│   ├── HGL polyline (water surface)
│   ├── ax.fill_between invert→HGL (water-filled pipe polygon)
│   ├── node "cylinder" Rectangles
│   └── optional secondary variable line/colormap
│
└── Bottom row
    ├── QSlider (timestep index)
    ├── QComboBox (variable: head / depth / velocity / flow / —none—)
    └── [PNG] [CSV] [Options] [Close]

NetworkProfileMapTool (QgsMapTool)
├── first click  → starts the chain at the link's upstream node
├── subsequent click on connected link
│   ├── if (from_node == last_link.to_node): forward orientation
│   ├── elif (to_node == last_link.to_node): reverse orientation
│   └── else: reject with "not connected to last link"
├── double-click / right-click / Escape  →  finished(ChainSpec)
└── emits chain_extended / pick_rejected / finished

Services (pure-Python, zero-Qt)
├── drainage_graph_service
│   ├── load_drainage_graph(gpkg)              → DrainageGraph
│   ├── find_chain(graph, start_node, end_node) → list[link_id]
│   └── link_orientation(graph, link_id, expected_upstream) → bool
│
├── profile_pipeline_service (owns ALL numpy on mesh data)
│   ├── load_pipe_cell_records(gpkg, run_id, link_ids) → dict
│   ├── assemble_chain_profile(gpkg, run_id, chain, graph, t_idx)
│   │                                          → ProfileArrays
│   └── profile_at_variable(profile, metric)   → (values, stations)
│
└── profile_persistence_service
    ├── save_profile / load_profile / list_profiles / delete_profile
    └── backs the new swe2d_profile_chains table

ProfileController (orchestrates the dialog launch)
└── open_network_profile_viewer() — reads GPKG path from view, launches dialog
```

### Layer boundaries (MVP)

| Layer | Files | Touches numpy? | Touches Qt? |
|---|---|---|---|
| Service | `drainage_graph_service.py`, `profile_pipeline_service.py`, `profile_persistence_service.py` | Yes (mesh geometry) | No |
| View | `network_profile_map_tool.py`, `profile_chain_widget.py`, `network_profile_dialog.py`, `network_profile_plot_widget.py`, `profile_options_dialog.py` | No (only uses already-computed numpy arrays + matplotlib) | Yes |
| Controller | `profile_controller.py` | No | No (only reads View protocol methods) |

Enforced by:
- `! grep -q 'from qgis\|from PyQt\|import qgis' swe2d/workbench/services/drainage_graph_service.py swe2d/workbench/services/profile_pipeline_service.py swe2d/workbench/services/profile_persistence_service.py`
- `! grep -q 'np\.' swe2d/workbench/dialogs/network_profile_dialog.py swe2d/workbench/views/profile_chain_widget.py swe2d/workbench/dialogs/network_profile_plot_widget.py swe2d/workbench/views/network_profile_map_tool.py`
  (numpy references only allowed in `_plot_widget.draw_profile()` which receives pre-built arrays)

## Data flow

### User actions

1. User opens **Network Profile Viewer** via menu (`HYDRA2DMenuOpenNetworkProfileAction`).
2. Controller launches `NetworkProfileDialog(gpkg_path, run_id, qgis_iface)`.
3. Dialog pre-loads `DrainageGraph` and `pipe_cell_records` for the chosen run.
4. User clicks **Pick on Map** → `NetworkProfileMapTool` is activated on the QGIS canvas.
5. User clicks the first link on the map; the upstream node of that link is recorded
   as the chain start. Upstream = the link endpoint (from_node or to_node) that has
   the *lower* out-degree in the drainage graph (a tie means "first encountered upstream end").
   The chain starts as a single link with orientation auto-determined.
6. User clicks subsequent connected links on the map; each click:
   - Shares an endpoint with the last link's downstream node, OR
   - The clicked link's `from_node`/`to_node` is the last link's downstream node.
   - Orientation inferred from which endpoint the new link shares.
   - Else: rejected with "not connected to last link" signal.
7. User finishes (double-click / right-click / Escape / explicit Finish button).
8. Slider scrubs time → `assemble_chain_profile(t_idx)` → redraw.
9. Variable combo selects second-line metric → redraw with overlay.
10. Save PNG / CSV / save profile to GPKG.

### Service pipeline (per render)

```
ChainSpec (list of (link_id, reverse))
  + DrainageGraph
  + run_id, timestep_index
  ↓
profile_pipeline_service.assemble_chain_profile(...)
  1. load_pipe_cell_records(gpkg, run_id, [link_ids])
  2. for each (link_id, reverse) in chain:
     a. find link's length, from_node, to_node in DrainageGraph
     b. read cell records for this link, sorted by cell_sub_idx
     c. if reverse: reverse the array
     d. cell stations = (sub_idx + 0.5) * (link_length / n_sub)
        (cumulative: starts at running_offset, increments by link_length)
     e. cell crown = invert + cell_height_for_shape(shape_type)
     f. cell HGL = invert + depth, or read head metric if available
  3. insert node endpoints at link boundaries:
     ground_m[node_idx] = node.rim_elev (drainage_nodes table)
  4. line-interpolate ground_m between node endpoints
  ↓
ProfileArrays — all 1D, length = sum(n_sub)
  ↓
NetworkProfilePlotWidget.draw_profile(profile, variable=...)
  - matplotlib calls only
  - emits Export PNG / CSV actions
```

## Service interfaces

### drainage_graph_service

```python
@dataclass(frozen=True)
class DrainageGraph:
    node_ids: list[str]
    link_ids: list[str]
    from_node: dict[str, str]
    to_node: dict[str, str]
    outgoing: dict[str, list[str]]   # node_id -> downstream link_ids
    incoming: dict[str, list[str]]   # node_id -> upstream link_ids
    both: dict[str, list[str]]       # node_id -> all incident link_ids (undirected)

def load_drainage_graph(gpkg_path: str) -> DrainageGraph: ...
def find_chain(graph: DrainageGraph, start_node: str, end_node: str) -> list[str]: ...
def link_orientation(graph: DrainageGraph, link_id: str, expected_upstream: str) -> bool: ...
```

Notes:
- `find_chain` returns an *ordered* list of link_ids walking downstream; each
  entry has implicit forward-orientation. The map tool / chain widget decide
  per-link whether to reverse.
- Returns `[]` if `start_node == end_node` or no path exists.

### profile_pipeline_service

```python
@dataclass(frozen=True)
class ChainSpec:
    link_specs: list[tuple[str, bool]]   # (link_id, reverse_orientation)
    def cumulative_links(self) -> list[str]: ...
    def is_empty(self) -> bool: ...

@dataclass
class ProfileArrays:
    station_m: np.ndarray
    invert_m: np.ndarray
    crown_m: np.ndarray
    ground_m: np.ndarray
    hgl_m: np.ndarray
    depth_m: np.ndarray
    velocity_ms: np.ndarray
    flow_cms: np.ndarray
    node_stations: list[float]
    node_ids: list[str]
    link_boundaries: list[tuple[int, str]]   # (station_idx, link_id)
    crown_style: str                          # 'circular' | 'rectangular' | 'mixed'

def load_pipe_cell_records(
    gpkg_path: str, run_id: str, link_ids: list[str]
) -> dict[tuple[str, int, str], np.ndarray]: ...

def assemble_chain_profile(
    gpkg_path: str,
    run_id: str,
    chain: ChainSpec,
    graph: DrainageGraph,
    timestep_index: int,
    *,
    crown_offset_m: float | None = None,
) -> ProfileArrays: ...

def profile_at_variable(profile: ProfileArrays, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (values_per_station, station_m). metric ∈ {'depth','velocity','flow','head'}.
    Raises ValueError for unknown metrics."""
    ...
```

Algorithm details:
- `assemble_chain_profile` reads `swe2d_drainage_links.length`/`shape`/etc. and
  `swe2d_drainage_nodes.invert_elev`/`rim_elev`.
- For each link: locate entries in pre-loaded `pipe_cell_records`,
  filter by `cell_sub_idx`, sort ascending.
- Cell-center station within a link:
  `x_local = (sub_idx + 0.5) * (length / n_sub)`,
  added to the running cumulative station.
- Crown by shape:
  - circular: invert + `cell_width`
  - rectangular: invert + `cell_height`
  - other (cell_shape_type ∈ {0, default}): invert + `cell_width` (fallback)
- HGL: prefer `head` metric; otherwise `invert + depth`.
- Ground: insert `node.rim_elev` at each link boundary; line-interp between.

### profile_persistence_service

```python
def save_profile(
    gpkg_path: str,
    profile_name: str,
    chain: ChainSpec,
    run_id: Optional[str] = None,
) -> int:
    """Insert/replace a named profile. Returns profile_id. Raises ValueError on duplicate name unless REPLACE."""

def list_profiles(gpkg_path: str) -> list[dict]:
    """Return list of dicts with profile_id, profile_name, run_id, link_ids, created_utc."""

def load_profile(gpkg_path: str, profile_id: int) -> ChainSpec:
    """Returns ChainSpec reconstructed from comma-separated link_ids."""

def delete_profile(gpkg_path: str, profile_id: int) -> None:
    """No-op if profile_id doesn't exist."""
```

New table `swe2d_profile_chains`:
```sql
CREATE TABLE IF NOT EXISTS swe2d_profile_chains (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT UNIQUE NOT NULL,
    run_id TEXT,
    link_ids TEXT NOT NULL,             -- comma-separated, e.g., "L1,F,L3"
    created_utc TEXT NOT NULL,
    metadata_json TEXT DEFAULT '{}'
);
```

The `link_ids` string uses `L1,F,L3` where `F`/`R` indicates forward/reverse
orientation per link. (Using a separator char between link IDs that cannot
appear in our ID string schema.)

## View interfaces

### NetworkProfileMapTool

```python
class NetworkProfileMapTool(QgsMapTool):
    chain_extended = QtCore.pyqtSignal(object)  # ChainSpec
    chain_cleared  = QtCore.pyqtSignal()
    pick_rejected  = QtCore.pyqtSignal(str, str)  # reason, clicked_feature_id
    finished       = QtCore.pyqtSignal(object)  # ChainSpec
```

UI states:
- Hover: rubber-band highlight over drainage_links
- Selected features: emissive colour highlight (filter applied to layer's renderer)
- Status bar text: "Pick start link → click connected link → press Escape to finish"

### ProfileChainWidget

```python
class ProfileChainWidget(QtWidgets.QWidget):
    chain_changed = QtCore.pyqtSignal(object)  # ChainSpec
    pick_requested = QtCore.pyqtSignal()
```

Widgets:
- Top: run ID label + selectable run combo
- Toolbar: `Pick on Map` / `Find Path` / `Add Link` / `Reverse` / `Up` / `Down` / `Remove` / `Clear` / `Save Profile` / `Load Profile`
- Centre: `QListWidget` showing ordered link rows (with orientation badge)
- Bottom: status label — `'3 links, 412 m total | upstream node N1 → downstream node N5'`

### NetworkProfileDialog

```python
class NetworkProfileDialog(QtWidgets.QDialog):
    def __init__(
        self,
        gpkg_path: str,
        run_id: str | None = None,
        qgis_iface: object | None = None,
        parent=None,
    ):
```

UI:
- Window title: `"Network Profile Viewer — <gpkg basename>"`
- Default size: 1400 × 800
- Layout: `QSplitter(Qt.Horizontal)` with `ProfileChainWidget` on left (default 360 px), `NetworkProfilePlotWidget` on right.
- Bottom bar: time slider, variable combo, export buttons, options, close.
- Caches: `(timestep_index, hash(chain)) -> ProfileArrays` to avoid recomputing unchanged chain/time.

Lifecycle / parent-Qt relationship:
- When "Pick on Map" is pressed: parent dialog hands its map tool activation to `ProfileController` via protocol, then activates `NetworkProfileMapTool` on the QGIS canvas. When finished, returns to previous map tool.
- The dialog never owns the canvas; it requests activation through the controller.

### NetworkProfilePlotWidget

```python
class NetworkProfilePlotWidget(QtWidgets.QWidget):
    def draw_profile(self, profile: ProfileArrays, variable: str = '—none—') -> None: ...
    def export_png(self, path: str) -> None: ...
    def export_csv(self, path: str, profile: ProfileArrays) -> None: ...
    def set_options(self, options: ProfileOptions) -> None: ...
```

Drawing order (back→front):
1. `ax.fill_between(stations, invert, hgl, color=water_color, alpha=0.6)` — water polygon
2. `ax.plot(stations, invert, color=invert_color)` — invert line
3. `ax.plot(stations, crown, color=crown_color)` — crown line
4. `ax.plot(node_stations, node_rim_elevs, color=ground_color)` — ground line
5. `ax.plot(stations, hgl, color=water_color, linewidth=2)` — HGL line
6. `ax.add_patch(Rectangle(...))` × N nodes — node cylinders
7. Optional secondary line: `ax.plot(stations, variable_values, color=accent_color)` — depth/velocity/flow

### ProfileOptionsDialog

```python
@dataclass
class ProfileOptions:
    water_color: str = '#3366CC'
    conduit_color: str = '#5A5A5A'
    invert_color: str = '#2A2A2A'
    crown_color: str = '#888888'
    ground_color: str = '#A0763D'
    ground_line_visible: bool = True
    conduits_only: bool = False
    thick_lines: bool = False
    x_label: str = 'Distance (m)'
    y_label: str = 'Elevation (m)'
    auto_scale: bool = True
    y_min: float = 0.0
    y_max: float = 10.0
    y_inc: float = 1.0
    node_labels_on_top_axis: bool = False
    node_labels_on_plot: bool = True
    arrow_length_px: int = 30
    font_size_pt: int = 8
```

5 tabs (like SWMM `Dproplot`):
1. **Colors** — pickers for water, conduit, invert, crown, ground
2. **Styles** — ground line visible, conduits-only, thick lines, label positions
3. **Axes** — titles (QLineEdit), fonts
4. **Vertical Scale** — auto/manual Y-range with increments
5. **Node Labels** — on top axis / on plot / arrow length / font size

## Test plan

### Unit (no Qt)

```python
# tests/test_drainage_graph_service.py
test_load_empty_graph
test_load_single_link
test_load_two_node_one_link
test_load_three_node_two_link_linear
test_load_branching_network
test_find_chain_same_start_end_returns_empty
test_find_chain_two_link_path
test_find_chain_branch_chooses_shortest
test_find_chain_no_path_returns_empty
test_link_orientation_forward
test_link_orientation_reverse

# tests/test_profile_pipeline_service.py
test_load_pipe_cell_records_returns_correct_keys
test_assemble_single_link_forward
test_assemble_single_link_reverse_reverses_arrays
test_assemble_three_link_chain_with_middle_reversed
test_assemble_chain_at_timestep_index_out_of_range_clamps
test_crown_circular_invert_plus_width
test_crown_rectangular_invert_plus_height
test_ground_interpolation_between_nodes
test_variable_picker_returns_depth
test_variable_picker_returns_velocity
test_variable_picker_returns_flow
test_variable_picker_unknown_metric_raises

# tests/test_profile_persistence_service.py
test_save_profile_creates_table
test_save_profile_replaces_existing
test_list_profiles_returns_in_order
test_load_profile_round_trip
test_load_profile_with_orientation_tokens
test_delete_profile_removes_row
test_delete_profile_unknown_id_no_op

# tests/test_network_profile_plot_widget.py  -- with QT_QPA_PLATFORM=offscreen
test_draw_profile_renders_axes
test_draw_profile_with_depth_variable
test_export_png_creates_file
test_export_csv_has_correct_header
```

### Integration (QGIS-aware)

```python
# tests/test_network_profile_map_tool.py -- requires QGIS app
test_first_click_sets_chain_start
test_connected_second_click_extends_chain_forward
test_connected_second_click_extends_chain_reverse
test_disconnected_click_is_rejected_with_reason
test_double_click_finishes_chain

# tests/test_network_profile_dialog.py -- requires QGIS app
test_dialog_instantiates_with_gpkg
test_dialog_loads_pipe_cell_records_for_run
test_dialog_time_slider_recomputes_profile
test_dialog_variable_combo_changes_overlay
test_dialog_save_profile_persists_to_gpkg

# tests/test_profile_controller.py
test_open_network_profile_viewer_launches_dialog
```

### Existing tests

The following existing tests must remain green (no silent breakage):
- `test_workbench_gpkg_service.py` (gpkg services)
- `test_results_path_audit_fixes.py::TestGpkgExplorerDialogImport`
- `test_map_tab_view.py`
- `test_workbench_tab_views.py`
- `test_workbench_delegation.py::TopologyController.open_model_gpkg_explorer`
- `test_numpy_blob_service.py` (26 tests)
- `test_profile_persistence_service.py` (new)

## Files

| File | Action | Lines (est) |
|---|---|---|
| `swe2d/workbench/services/drainage_graph_service.py` | NEW | ~120 |
| `swe2d/workbench/services/profile_pipeline_service.py` | NEW | ~280 |
| `swe2d/workbench/services/profile_persistence_service.py` | NEW | ~110 |
| `swe2d/workbench/views/network_profile_map_tool.py` | NEW | ~210 |
| `swe2d/workbench/views/profile_chain_widget.py` | NEW | ~230 |
| `swe2d/workbench/dialogs/network_profile_dialog.py` | NEW | ~280 |
| `swe2d/workbench/dialogs/network_profile_plot_widget.py` | NEW | ~190 |
| `swe2d/workbench/dialogs/profile_options_dialog.py` | NEW | ~280 |
| `swe2d/workbench/controllers/profile_controller.py` | NEW | ~50 |
| `swe2d/workbench/views/workbench_main_menu.py` | amend (add menu action) | +5 |
| `swe2d/workbench/views/view_protocols.py` | amend (add `get_active_gpkg_path`, `get_active_run_id`, `get_qgis_iface` to dialog protocol) | +5 |
| `tests/test_drainage_graph_service.py` | NEW | ~150 |
| `tests/test_profile_pipeline_service.py` | NEW | ~250 |
| `tests/test_profile_persistence_service.py` | NEW | ~110 |
| `tests/test_network_profile_map_tool.py` | NEW | ~80 |
| `tests/test_network_profile_dialog.py` | NEW | ~80 |
| `tests/test_network_profile_plot_widget.py` | NEW | ~50 |

**Total: ~2280 lines across 9 new production files, 2 amended files, 6 test files.**

## Deferred decisions

- **Saved-profile orientation encoding**: Decided to store link IDs in the comma-separated
  `link_ids` string with orientation tokens (e.g., `"L1,F,L3,R,L5"`). If a link ID contains
  a comma (very unlikely per current schemas), the operation is rejected with an error.
- **Find Path weightedness**: v1 ships unweighted BFS (matches SWMM). Weighted variant
  by link length deferred to a follow-up feature.
- **Re-using `studio_viewer_profile_pg.py` primitives**: v1 builds a standalone plot
  widget. Refactor to extract shared helpers (invert/crown/water-fill rendering) deferred
  to a separate task — kept out of scope to keep this PR focused.

## Spec coverage checklist

- [x] SWMM-style profile plot (invert, crown, HGL, ground, water polygon, node cylinders)
- [x] Single timestep + slider scrubbing
- [x] Variable picker (depth / velocity / flow / head)
- [x] Map-click chain building
- [x] BFS shortest-path from start to end node
- [x] Saved named profiles in GPKG
- [x] Standalone dialog
- [x] PNG / CSV export
- [x] 5-tab options dialog (Colors / Styles / Axes / Vertical Scale / Node Labels)
- [x] Pure-Python services with numpy math isolated
- [x] View protocol respects MVP architecture
