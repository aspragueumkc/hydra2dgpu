"""Standalone pipe1d solver runner for validation against SWMM.

Uses a minimal 1-cell 2D mesh as dummy device state, then builds a
pipe network and runs swe2d_pipe1d_step with uploaded node depths.
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
    max_cell_length: int = 25
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
    """

    def __init__(self):
        self._mod = None
        self._backend = None
        self._dev_ptr = 0
        self._n_nodes = 0
        self._n_links = 0
        self._n_cells = 0

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

        _mod.swe2d_build_pipe1d_mesh(
            n_links,
            cfg.link_from,
            cfg.link_to,
            cfg.link_length,
            cfg.link_diameter,
            cfg.link_roughness_n,
            cfg.link_inlet_loss_k,
            cfg.link_outlet_loss_k,
            cfg.node_invert,
            cfg.node_surface_area,
            cfg.node_max_depth,
            cfg.link_invert_in,
            cfg.link_invert_out,
            cfg.max_cell_length,
            self._dev_ptr,
            shape_type,
            link_width,
            link_height,
        )

        # Count pipe cells (each link is subdivided)
        self._n_links = n_links
        total_cells = 0
        for li in range(n_links):
            n_sub = max(1, int(np.ceil(cfg.link_length[li] / cfg.max_cell_length)))
            total_cells += n_sub
        self._n_cells = total_cells

    def set_node_depth(self, depth: np.ndarray) -> None:
        """Upload node depths for pipe1d boundary conditions."""
        self._mod.swe2d_pipe1d_upload_node_depth(
            self._dev_ptr, depth.astype(np.float64))

    def init_area_from_depth(self, default: float = 0.0) -> None:
        """Initialize pipe cell areas from current node depths (used for dry start)."""
        self._mod.swe2d_pipe1d_init_area_from_depth(self._dev_ptr)

    def init_full(self) -> None:
        """Set pipe cells to full area (primed start)."""
        self._mod.swe2d_pipe1d_init_full(self._dev_ptr)

    def step(self, dt: float = 0.1, solver_mode: str = "diffusion_wave",
             substeps: int = 1, implicit_iters: int = 2,
             relaxation: float = 0.5, g: float = 9.81) -> None:
        """Run one pipe1d step."""
        self._mod.swe2d_pipe1d_step(
            self._dev_ptr, dt, solver_mode, substeps,
            implicit_iters, relaxation, g)

    def readback(self) -> Pipe1DResult:
        """Read back current node depth and cell Q."""
        state = self._mod.swe2d_pipe1d_readback_node_state(
            self._dev_ptr, self._n_nodes, self._n_cells)
        nd = np.zeros(self._n_nodes, dtype=np.float64)
        cq = np.zeros(self._n_cells, dtype=np.float64)
        if state:
            if "node_depth" in state:
                nd = state["node_depth"]
            if "cell_Q" in state:
                cq = state["cell_Q"]

        # Build dicts
        node_depth: Dict[str, List[float]] = {
            f"n{i}": [float(nd[i])] for i in range(self._n_nodes)}
        cell_Q: Dict[str, List[float]] = {}

        # Aggregate cell Q by link (average over sub-cells)
        cells_per_link = self._n_cells // self._n_links if self._n_links > 0 else self._n_cells
        for li in range(self._n_links):
            start = li * cells_per_link
            end = start + cells_per_link
            if end <= len(cq):
                q = float(np.mean(np.abs(cq[start:end])))
            else:
                q = 0.0
            cell_Q[f"c{li}"] = [q]

        return Pipe1DResult(
            times=[0.0],
            node_depth=node_depth,
            cell_Q=cell_Q,
        )

    def destroy(self) -> None:
        if self._backend is not None:
            self._backend.destroy()
            self._backend = None
