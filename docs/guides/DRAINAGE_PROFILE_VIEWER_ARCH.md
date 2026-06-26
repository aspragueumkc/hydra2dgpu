# Multi-Link Drainage Network Profile Viewer — Architecture Sketch

## Overview

A new pyqtgraph-based viewer tab that renders a **continuous longitudinal profile**
through multiple connected drainage links. The user selects a sequence of nodes
from a dropdown and the tool traces the links between them, drawing pipe shapes,
water surface, and velocity shading along the full path.

```
Studio Viewer Dock
  ┌──────────────────────────────────────────────────┐
  │ Tab: "Network Profile"                           │
  │                                                   │
  │  Node selection: [__A__] [__B__] [__C__] ... [+] │
  │                                                   │
  │  ┌──────────────────────────────────────────────┐ │
  │  │ pyqtgraph PlotWidget (profile)                │ │
  │  │                                               │ │
  │  │    ┌─────┐       ┌─────┐                     │ │
  │  │    │circ │       │ box │                     │ │
  │  │    │pipe │       │culvt│                     │ │
  │  │   ▒▒▒▒▒▒▒▒▒▒░░░░▒▒▒▒▒▒▒▒░░  ← water fill    │ │
  │  │  ▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒▒▒▒▒▒▒▒▒▒░░  with velocity  │ │
  │  │  ════A══════════B════════C══  ← bed invert   │ │
  │  │  ● A              ● B       ● C ← node labels │ │
  │  └──────────────────────────────────────────────┘ │
  │  [💾 Save] [⚙] [Show data table]                 │
  └──────────────────────────────────────────────────┘
```

## Data Flow

```
GPKG file
  │
  ├─ swe2d_drainage_nodes ─────┐
  ├─ swe2d_drainage_links ─────┤
  └─ swe2d_coupling_results ───┤
                               ▼
               build_network_config_from_gpkg()
                     │
                     ▼
              PipeNetworkConfig
              ├─ nodes: List[DrainageNode]
              ├─ links: List[DrainageLink]
              └─ adjacency: Dict[node_id → List[link_id]]
                     │
                     ▼
           User selects node sequence: [A, B, C, D]
                     │
                     ▼
              trace_path(nodes, node_ids)
              ├─ For each consecutive pair (Ni, Ni+1):
              │   find the DrainageLink connecting them
              ├─ If link not found → show warning, skip
              └─ Return ordered [(link, from_node, to_node), ...]
                     │
                     ▼
              render_profile(pg_plot_widget, path, coupling_data)
              ├─ Compute stationing (cumulative link lengths)
              ├─ Draw pipe shapes (circle/box) to scale
              ├─ Draw bed invert line
              ├─ Draw water surface fill with velocity color shading
              └─ Annotate node names + flow values
```

## New Files Required

### 1. `swe2d/extensions/drainage_gpkg_reader.py` (~150 lines)

Pure Python / SQLite — no Qt. Reads the GPKG tables and builds the network model.

```python
def build_network_config_from_gpkg(
    gpkg_path: str,
) -> PipeNetworkConfig:
    """Read drainage tables from GPKG and build PipeNetworkConfig.

    Queries:
      - swe2d_drainage_nodes (geometry → x,y via GPKG ST_X/ST_Y or parsed WKT)
      - swe2d_drainage_links (geometry → length check, attributes)
      - swe2d_drainage_inlets
      - swe2d_drainage_node_inlets

    Also builds adjacency map:
      adjacency[from_node_id].append(link)
      adjacency[to_node_id].append(link)
    """
```

### 2. `swe2d/workbench/views/studio_viewer_network.py` (~450 lines)

The pyqtgraph widget for the Network Profile tab.

```python
class PGNetworkProfileWidget(QtWidgets.QWidget):
    """Multi-link drainage network longitudinal profile.

    Protocol matches PlotViewWidget / PGTimeSeriesWidget:
      set_data(), refresh(), selected_metric, selected_element_id
    """

    _mode = "Network Profile"

    def __init__(self, parent=None):
        # Data
        self._result_data: Any = None
        self._network_config: Optional[PipeNetworkConfig] = None
        self._path_links: List[LinkPathSegment] = []  # ordered segments

        # UI
        self._plot_widget: pg.PlotWidget
        self._node_list: QtWidgets.QListWidget  # list of selected nodes
        self._add_node_btn: QtWidgets.QPushButton
        self._remove_node_btn: QtWidgets.QPushButton
        self._clear_path_btn: QtWidgets.QPushButton
        self._node_combo: QtWidgets.QComboBox  # available nodes to add
        self._metric_combo: QtWidgets.QComboBox
        self._plot_items: List[pg.PlotDataItem]
        self._hover_label: pg.TextItem

    def _build_ui(self):
        """Build:
        Top bar: node selector combo + Add button + metric combo + settings
        Middle: pyqtgraph PlotWidget with zoom/pan/hover
        Bottom: data table (hidden)
        """

    def _add_node_to_path(self):
        """Add selected node to the path list.
        After adding, try to trace path between consecutive nodes.
        """

    def _trace_path(self) -> List[LinkPathSegment]:
        """Walk the user's node list and find links connecting each pair.

        Returns list of (link, from_node, to_node, cumulative_start_dist).
        If a pair has no direct link, log warning and set segment to None.
        """

    def refresh(self):
        """Re-render the profile with current path + coupling data.

        1. If path changed or node_ids changed → re-trace
        2. Compute stationing (cumulative distance from start)
        3. For each segment:
           a. Draw pipe shape (circle diameter or box rise/span) to scale
           b. Draw bed invert line
           c. Draw water surface fill from coupling data
           d. Color water fill by velocity (lut lookup)
        4. Annotate node names at junctions
        5. Draw flow arrow annotation
        """

    def _draw_pipe_shape(self, link, x_start, x_end, invert_f, invert_t):
        """Draw pipe cross-section shape in profile.

        For circular pipes: draw a rectangle the height of the diameter
        above the invert, with a semicircle cap at top.
        For box culverts: draw rectangle of rise × span proportion.
        For weirs/orifices: draw schematic triangle/gate shape.
        """

    def _draw_water_fill(self, link, x_start, x_end,
                          invert_f, invert_t, depth_f, depth_t,
                          vel_f, vel_t):
        """Draw water surface between node depths with velocity shading.

        Water fill polygon: [inv_f, inv_t, wse_t, wse_f]
        Color is interpolated between vel_f and vel_t using a colormap.
        """
```

### 3. Dataclass for path segments

```python
@dataclass
class LinkPathSegment:
    link: DrainageLink
    from_node: DrainageNode
    to_node: DrainageNode
    dist_start: float = 0.0  # cumulative distance at segment start
    dist_end: float = 0.0    # cumulative distance at segment end
```

## Integration

### Register in `studio_viewer.py`

```python
_TAB_MODES = ["Mesh", "Time Series", "Profile", "Structure",
              "Network", "Network Profile"]

# In _build_ui:
if mode == "Network Profile" and _HAVE_PG:
    widget = PGNetworkProfileWidget()
```

### Wire signals in `studio_results_panel.py`

```python
def on_results_network_profile_changed(dialog) -> None:
    viewer = dialog._studio_viewer
    npw = viewer.plot_widgets.get("Network Profile")
    if npw is not None:
        npw.refresh()
```

## Rendering Detail: Pipe Shapes in Profile

This is the novel part compared to the current flat-line approach:

```
Circular pipe (diameter D):
                  ╔═══╤═══╗
        WSE ──────╨───┼───╨──  water surface
   invert ────────────┼────────
                  ║   │   ║
                  ║   │   ║   D = diameter
                  ║   │   ║
   invert ────────────┼────────
                  ╚═══╧═══╝

   For a circular pipe in profile, the pipe crown is at invert + D.
   The pipe wall is drawn as a thin rectangle with rounded ends.
   Fill between invert and WSE is the water volume.
```

The pipe cross‑section shape is a **side‑view slice** through the pipe centerline.
For the profile (longitudinal section), the pipe appears as:

- **Circular**: A rectangle of height = `diameter` above the invert line.
  If the pipe is flowing partially full, water fills from invert up to WSE.
- **Box culvert**: A rectangle of height = `culvert_rise` and any width
  (constant in profile since we're looking along the axis).
- **Weir**: A trapezoid schematic.

The **velocity color shading** is a linear interpolation between the upstream
and downstream node velocity values, mapped through a pyqtgraph color LUT
onto the water fill polygon vertices.

## Key edge cases

| Case | Handling |
|------|----------|
| No link between consecutive nodes | `_trace_path()` returns a segment with `link=None`, renderer draws a gap + warning label |
| Missing coupling data for a node | Default depth=0, velocity=0, render flat bed |
| Single node in list | Show just that node info, no profile line |
| Backwards order (down→up stream) | Render still works — the profile just slopes opposite |
| Multiple barrels | Draw `barrel_count` pipes stacked vertically |
| Circular pipe vs box culvert | Check `link_type` + `culvert_shape`, dispatch to different draw helpers |
| Animation scrub | `refresh()` called on timestep change — only water fill + velocity colors change, pipe shapes + bed are stable |

## Implementation Sequence

1. **`drainage_gpkg_reader.py`** — read GPKG → `PipeNetworkConfig`, no UI needed
2. **`PGNetworkProfileWidget`** — build UI layout, node list, path tracing
3. **Pipe shape rendering** — circular, box, weir/orifice dispatch
4. **Water fill + velocity color** — coupling data overlay
5. **Wire into viewer** — register tab, connect signals
6. **Polish** — save/export, settings toggles, data table

## Total estimate: ~600 lines, ~3-4 days
