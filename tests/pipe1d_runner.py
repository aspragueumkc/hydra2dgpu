"""Standalone pipe1d solver runner for validation against SWMM.

Uses a minimal 1-cell 2D mesh as dummy device state, then builds a
pipe network and runs swe2d_pipe1d_step with uploaded cell depths.

Migrated from the legacy per-node solver API to the unified mesh +
cell-state schema (commit a080e61 / ce74f7d):

    swe2d_build_pipe1d_mesh       -> swe2d_build_unified_mesh
    swe2d_pipe1d_upload_node_depth -> swe2d_pipe1d_upload_cell_h
    swe2d_pipe1d_init_area_from_depth -> swe2d_pipe1d_init_cell_area
    swe2d_pipe1d_readback_node_state  -> swe2d_pipe1d_readback_cell_state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Pipe1DConfig:
    link_from: np.ndarray
    link_to: np.ndarray
    link_length: np.ndarray
    link_diameter: np.ndarray
    link_roughness_n: np.ndarray
    link_inlet_loss_k: np.ndarray
    link_outlet_loss_k: np.ndarray
    link_invert_in: np.ndarray
    link_invert_out: np.ndarray
    node_invert: np.ndarray
    node_surface_area: np.ndarray
    node_max_depth: np.ndarray
    max_cell_length: float = 25.0
    link_shape_type: Optional[np.ndarray] = None
    link_width: Optional[np.ndarray] = None
    link_height: Optional[np.ndarray] = None


@dataclass
class Pipe1DResult:
    times: List[float] = field(default_factory=list)
    node_depth: Dict[str, List[float]] = field(default_factory=dict)
    cell_Q: Dict[str, List[float]] = field(default_factory=dict)


class Pipe1DRunner:
    """Run the GPU pipe1d solver in isolation (no 2D coupling).

    Usage::

        runner = Pipe1DRunner()
        runner.build_mesh(cfg)
        runner.set_node_depth([d0, d1, ...])
        runner.step(dt=0.1, substeps=1)
        result = runner.readback()

    NOTE: ``set_node_depth`` now uploads per-cell depths via
    ``swe2d_pipe1d_upload_cell_h`` (the legacy per-node upload binding
    has been removed).  The caller passes per-node depths; this helper
    broadcasts the upstream-end depth to every pipe cell.  The
    ``readback()`` populates ``node_depth`` from per-cell ``cell_depth``
    (cells are aggregated per link; the per-node boundary WSE is no
    longer an explicit quantity in the unified schema).
    """

    def __init__(self):
        self._mod = None
        self._backend = None
        self._dev_ptr = 0
        self._n_nodes = 0
        self._n_links = 0
        self._n_cells = 0
        # link_from / link_to (cached) so set_node_depth can map nodes → cells.
        self._link_from: np.ndarray = np.empty(0, dtype=np.int32)

    def build_mesh(self, cfg: Pipe1DConfig) -> None:
        """Build minimal 2D backend + pipe1d mesh."""
        import hydra_swe2d as _mod
        from swe2d.runtime.backend import SWE2DBackend
        self._mod = _mod

        # Minimal 1-cell 2D mesh (required by swe2d_get_coupling_dev_ptr)
        self._backend = SWE2DBackend()
        node_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        node_y = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        node_z = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        cell_nodes = np.array([0, 1, 2], dtype=np.int32)
        self._backend.build_mesh(node_x, node_y, node_z, cell_nodes)
        self._backend.initialize(
            h0=np.array([0.1], dtype=np.float64),
            hu0=np.zeros(1, dtype=np.float64),
            hv0=np.zeros(1, dtype=np.float64),
            dt_fixed=0.05,
            dt_max=0.05,
        )
        self._dev_ptr = int(_mod.swe2d_get_coupling_dev_ptr())
        self._n_nodes = len(cfg.node_invert)

        # Default shapes
        n_links = len(cfg.link_from)
        shape_type = cfg.link_shape_type
        if shape_type is None:
            shape_type = np.zeros(n_links, dtype=np.int32)
        link_width = cfg.link_width
        if link_width is None:
            link_width = cfg.link_diameter.copy()
        link_height = cfg.link_height
        if link_height is None:
            link_height = cfg.link_diameter.copy()

        # The unified mesh builder derives link_invert_in/out from node_invert
        # automatically, so we drop those plus the inlet/outlet loss kwargs.
        _mod.swe2d_build_unified_mesh(
            dev_ptr=self._dev_ptr,
            n_links=n_links,
            link_from=cfg.link_from,
            link_to=cfg.link_to,
            L=cfg.link_length,
            D=cfg.link_diameter,
            n_mann=cfg.link_roughness_n,
            S0=np.zeros(n_links, dtype=np.float64),
            node_invert=cfg.node_invert,
            mcl=float(cfg.max_cell_length),
            link_shape_type=shape_type,
            link_width=link_width,
            link_height=link_height,
        )

        # Count pipe cells (each link is subdivided).
        self._n_links = n_links
        total_cells = 0
        self._link_from = np.asarray(cfg.link_from, dtype=np.int32).copy()
        self._cells_per_link = np.zeros(n_links, dtype=np.int32)
        for li in range(n_links):
            n_sub = max(1, int(np.ceil(cfg.link_length[li] / cfg.max_cell_length)))
            self._cells_per_link[li] = n_sub
            total_cells += n_sub
        self._n_cells = total_cells

    def set_node_depth(self, depth: np.ndarray) -> None:
        """Upload boundary depths for pipe1d.

        Migrated: the legacy binding ``swe2d_pipe1d_upload_node_depth`` has
        been replaced by ``swe2d_pipe1d_upload_cell_h``.  We broadcast the
        upstream-end node depth to every pipe cell of each link — that
        preserves the spirit of the legacy "set boundary node depths"
        pattern now that per-node boundary state is no longer an
        explicit uploadable quantity.
        """
        depth = np.asarray(depth, dtype=np.float64)
        cell_h = np.zeros(self._n_cells, dtype=np.float64)
        offset = 0
        for li in range(self._n_links):
            n_sub = int(self._cells_per_link[li])
            upstream = int(self._link_from[li])
            if 0 <= upstream < depth.size:
                upstream_depth = float(depth[upstream])
            else:
                upstream_depth = 0.0
            cell_h[offset:offset + n_sub] = upstream_depth
            offset += n_sub
        self._mod.swe2d_pipe1d_upload_cell_h(self._dev_ptr, cell_h)

    def init_area_from_depth(self, default: float = 0.0,
                             h_min: float = 1.0e-4) -> None:
        """Initialize pipe cell areas from current cell depths (used for dry start)."""
        self._mod.swe2d_pipe1d_init_cell_area(self._dev_ptr, h_min)

    def init_full(self) -> None:
        """Set pipe cells to full area (primed start)."""
        self._mod.swe2d_pipe1d_init_full(self._dev_ptr)

    def step(self, dt: float = 0.1, solver_mode: str = "diffusion_wave",
             substeps: int = 1, implicit_iters: int = 2,
             relaxation: float = 0.5, g: float = 9.81,
             k_mann: float = 1.0, h_min: float = 1.0e-4) -> None:
        """Run one pipe1d step."""
        self._mod.swe2d_pipe1d_step(
            self._dev_ptr, dt, solver_mode, substeps,
            implicit_iters, relaxation, g, k_mann, h_min)

    def readback(self) -> Pipe1DResult:
        """Read back current cell depth (per-link) and cell Q (per-link)."""
        state = self._mod.swe2d_pipe1d_readback_cell_state(
            self._dev_ptr, self._n_cells)
        cd = np.zeros(self._n_cells, dtype=np.float64)
        cq = np.zeros(self._n_cells, dtype=np.float64)
        if state:
            if "cell_depth" in state:
                cd = np.asarray(state["cell_depth"], dtype=np.float64)
            if "cell_Q" in state:
                cq = np.asarray(state["cell_Q"], dtype=np.float64)

        # Build per-node depth dict from per-cell depth (use upstream-end
        # cell of each link as the node's representative depth).
        node_depth: Dict[str, List[float]] = {}
        offset = 0
        for ni in range(self._n_nodes):
            node_depth[f"n{ni}"] = [0.0]
        offset = 0
        for li in range(self._n_links):
            n_sub = int(self._cells_per_link[li])
            upstream = int(self._link_from[li])
            if 0 <= upstream < self._n_nodes and n_sub > 0 and offset < len(cd):
                node_depth[f"n{upstream}"] = [float(cd[offset])]
            offset += n_sub
        cell_Q: Dict[str, List[float]] = {}

        # Aggregate cell Q by link (average over sub-cells)
        offset = 0
        for li in range(self._n_links):
            n_sub = int(self._cells_per_link[li])
            if offset + n_sub <= len(cq):
                q = float(np.mean(np.abs(cq[offset:offset + n_sub])))
            else:
                q = 0.0
            cell_Q[f"c{li}"] = [q]
            offset += n_sub

        return Pipe1DResult(
            times=[0.0],
            node_depth=node_depth,
            cell_Q=cell_Q,
        )

    def destroy(self) -> None:
        if self._backend is not None:
            self._backend.destroy()
            self._backend = None
