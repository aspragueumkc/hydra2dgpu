"""Shared UI constants — value maps and option tuples for workbench widgets.

Moved from the legacy ``swe2d_workbench_qt`` monolith.  These are plain
data; no Qt imports, no logic.
"""

from __future__ import annotations

# ── Boundary condition constants ────────────────────────────────────────────

BC_INFLOW_Q = 2
BC_TS_FLOW = 102
BC_TS_STAGE = 103

BC_VALUE_MAP = {
    "Wall (zero normal flux)": 1,
    "Inflow Q (total discharge)": 2,
    "Stage (prescribed WSE)": 3,
    "Normal Depth (prescribed depth)": 6,
    "Normal Depth (friction slope Sf)": 7,
    "Timeseries Flow Q": 102,
    "Timeseries Stage": 103,
    "Open (zero-gradient)": 4,
    "Reflecting": 5,
}

BC_OPTIONS = list(BC_VALUE_MAP.items())

# ── Mesh cell type options ─────────────────────────────────────────────────

CELL_TYPE_OPTIONS = [
    "triangular",
    "quadrilateral",
    "cartesian",
    "channel_generator",
    "empty",
]

# ── Spatial reconstruction options ─────────────────────────────────────────

RECONSTRUCTION_OPTIONS = [
    ("First-order (baseline)",          0),
    ("MUSCL Fast (high-throughput)",     1),
    ("MUSCL MinMod (robust)",            2),
    ("MUSCL MC (less-diffusive TVD)",    3),
    ("MUSCL Van Leer (smooth TVD)",      4),
    ("WENO3-like (GPU experimental)",    5),
    ("WENO5 (GPU, 3rd-order LSQ)",        6),
]

# ── Temporal scheme options ────────────────────────────────────────────────

TEMPORAL_ORDER_OPTIONS = [
    ("Euler (RK1, 1st-order)",           1),
    ("RK2 (Heun, 2nd-order, default)",   2),
    ("RK3 (SSP Shu-Osher, 3rd-order)",  3),
    ("RK4 (classic, 4th-order)",         4),
    ("Graph-safe RK4 (true staged)",     5),
    ("Graph-safe RK5 (Cash-Karp)",       6),
]

# ── Structure type map ─────────────────────────────────────────────────────

STRUCTURE_TYPE_VALUE_MAP = {
    "Weir": 1,
    "Culvert": 2,
    "Gate": 3,
    "Bridge": 4,
    "Pump": 5,
}

# ── Drainage node / link / shape maps ──────────────────────────────────────

DRAIN_NODE_TYPE_VALUE_MAP = {
    "Junction": "junction",
    "Outfall": "outfall",
    "Storage": "storage",
    "Inlet": "inlet",
    "Pipe end": "pipe_end",
}

DRAIN_LINK_TYPE_VALUE_MAP = {
    "Conduit": "conduit",
    "Short lateral (simplified)": "lateral_simple",
    "Pump": "pump",
    "Weir": "weir",
    "Orifice": "orifice",
    "Culvert (HDS-5)": "culvert",
}

DRAIN_LINK_SHAPE_VALUE_MAP = {
    "Circular": "circular",
    "Box": "box",
    "Pipe arch": "pipe_arch",
    "Custom area": "custom",
}

# ── Rain / hyetograph maps ────────────────────────────────────────────────

RAIN_GAGE_UNITS_VALUE_MAP = {
    "mm/hr": "mm/hr",
    "in/hr": "in/hr",
    "mm": "mm",
    "in": "in",
}

HYETOGRAPH_VALUE_TYPE_MAP = {
    "Intensity": "intensity",
    "Incremental depth": "incremental",
    "Cumulative depth": "cumulative",
}

HYETOGRAPH_UNITS_VALUE_MAP = {
    "mm/hr": "mm/hr",
    "in/hr": "in/hr",
    "mm": "mm",
    "in": "in",
}

