"""Reusable SWMM runner for validating pipe1d solver against SWMM.

Wraps the swmm-toolkit solver API to run models and read back results.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from swmm.toolkit import shared_enum, solver


@dataclass
class SWMMNodeResult:
    depth: float
    head: float
    volume: float
    inflow: float
    overflow: float


@dataclass
class SWMMLinkResult:
    flow: float
    depth: float
    volume: float


class SWMMRunner:
    """Run a SWMM model from an .inp file and read back time-series results."""

    def __init__(self):
        self._file_refs: List[str] = []
        self._n_nodes: int = 0
        self._n_links: int = 0

    def run(self, inp_text: str, max_steps: int = 1000,
            ) -> Tuple[List[float], Dict[str, List[SWMMNodeResult]],
                       Dict[str, List[SWMMLinkResult]]]:
        """Run a SWMM model from input text.

        Args:
            inp_text: Full SWMM .inp file content.
            max_steps: Max timesteps to record.

        Returns:
            (times, node_results, link_results)
        """
        with tempfile.NamedTemporaryFile(suffix='.inp', delete=False) as f:
            inp_path = f.name
            f.write(inp_text.encode('utf-8'))
        rpt_path = inp_path.replace('.inp', '.rpt')
        out_path = inp_path.replace('.inp', '.out')
        self._file_refs = [inp_path, rpt_path, out_path]

        try:
            solver.swmm_open(inp_path, rpt_path, out_path)
            solver.swmm_start(True)

            self._n_nodes = solver.project_get_count(
                shared_enum.ObjectType.NODE)
            self._n_links = solver.project_get_count(
                shared_enum.ObjectType.LINK)

            times: List[float] = []
            node_results: Dict[str, List[SWMMNodeResult]] = {
                solver.project_get_id(shared_enum.ObjectType.NODE, i): []
                for i in range(self._n_nodes)
            }
            link_results: Dict[str, List[SWMMLinkResult]] = {
                solver.project_get_id(shared_enum.ObjectType.LINK, i): []
                for i in range(self._n_links)
            }

            while len(times) < max_steps:
                elapsed = solver.swmm_step()
                if elapsed <= 0.0:
                    break
                times.append(elapsed)
                for ni in range(self._n_nodes):
                    name = solver.project_get_id(
                        shared_enum.ObjectType.NODE, ni)
                    node_results[name].append(self._read_node(ni))
                for li in range(self._n_links):
                    name = solver.project_get_id(
                        shared_enum.ObjectType.LINK, li)
                    link_results[name].append(self._read_link(li))

            solver.swmm_end()
            solver.swmm_close()
            self._cleanup()
            return times, node_results, link_results

        except Exception:
            try:
                solver.swmm_end()
                solver.swmm_close()
            except Exception:
                pass
            self._cleanup()
            raise

    def _read_node(self, idx: int) -> SWMMNodeResult:
        return SWMMNodeResult(
    depth=solver.node_get_result(idx, shared_enum.NodeResult.DEPTH),
    head=solver.node_get_result(idx, shared_enum.NodeResult.HEAD),
    volume=solver.node_get_result(idx, shared_enum.NodeResult.VOLUME),
    inflow=solver.node_get_result(idx, shared_enum.NodeResult.TOTAL_INFLOW),
    overflow=solver.node_get_result(idx, shared_enum.NodeResult.FLOOD),
        )

    def _read_link(self, idx: int) -> SWMMLinkResult:
        return SWMMLinkResult(
            flow=solver.link_get_result(idx, shared_enum.LinkResult.FLOW),
            depth=solver.link_get_result(idx, shared_enum.LinkResult.DEPTH),
            volume=solver.link_get_result(idx, shared_enum.LinkResult.VOLUME),
        )

    def _cleanup(self):
        for p in self._file_refs:
            if os.path.exists(p):
                os.unlink(p)
        self._file_refs = []


def make_drainage_inp(
    *,
    start_date: str = "01/01/2023",
    end_time: str = "02:00:00",
    routing_step_s: float = 5.0,
    junctions: List[Tuple[str, float, float]] = None,
    outfalls: List[Tuple[str, float]] = None,
    conduits: List[Tuple[str, str, str, float, float, float]] = None,
    xsections: List[Tuple[str, str, float]] = None,
    inflows: List[Tuple[str, str]] = None,
    timeseries: List[Tuple[str, float, float]] = None,
) -> str:
    """Build a SWMM .inp file for a simple drainage network."""
    if junctions is None:
        junctions = []
    if outfalls is None:
        outfalls = []
    if conduits is None:
        conduits = []
    if xsections is None:
        xsections = []
    if inflows is None:
        inflows = []
    if timeseries is None:
        timeseries = []

    lines = []
    lines.append("[TITLE]")
    lines.append("pipe1d validation model")
    lines.append("")
    lines.append("[OPTIONS]")
    lines.append("FLOW_UNITS CMS")
    lines.append("FLOW_ROUTING DYNWAVE")
    lines.append(f"START_DATE {start_date}")
    lines.append("START_TIME 00:00:00")
    lines.append(f"REPORT_START_DATE {start_date}")
    lines.append("REPORT_START_TIME 00:00:00")
    lines.append(f"END_DATE {start_date}")
    lines.append(f"END_TIME {end_time}")
    lines.append("SWEEP_START 01/01")
    lines.append("SWEEP_END 12/31")
    lines.append("DRY_DAYS 0")
    lines.append("REPORT_STEP 01:00:00")
    lines.append("WET_STEP 01:00:00")
    lines.append("DRY_STEP 01:00:00")
    lines.append(f"ROUTING_STEP 0:00:{int(routing_step_s):02d}")
    lines.append("")
    lines.append("[JUNCTIONS]")
    lines.append(";;Name            Elev     MaxD    InitD   SurD    Ponded")
    for name, elev, max_depth in junctions:
        lines.append(f"{name:16s} {elev:.3f} {max_depth:.1f} 0 0 0")
    lines.append("")
    lines.append("[OUTFALLS]")
    lines.append(";;Name            Elev    Type    Gated")
    for name, elev in outfalls:
        lines.append(f"{name:16s} {elev:.3f}      FREE    NO")
    lines.append("")
    lines.append("[CONDUITS]")
    lines.append(";;Name            From    To      Len     Mann    InOff   OutOff  InitQ   MaxQ")
    for name, fn, tn, length, n, d in conduits:
        lines.append(f"{name:16s} {fn:8s} {tn:8s} {length:.1f} {n:.3f} 0 0 0 0")
    lines.append("")
    lines.append("[XSECTIONS]")
    lines.append(";;Link            Shape   Geom1   Geom2   Geom3   Geom4   Barrels")
    for link, shape, geom1 in xsections:
        lines.append(f"{link:16s} {shape:8s} {geom1:.3f} 0 0 0 1")
    lines.append("")
    if inflows:
        lines.append("[INFLOWS]")
        lines.append(";;Node            Constituent      Tseries     Mfac  Sfactor")
        for node, ts_name in inflows:
            lines.append(f"{node:16s} FLOW             {ts_name:16s} 1.0   1.0")
        lines.append("")
    if timeseries:
        lines.append("[TIMESERIES]")
        lines.append(";;Name            Time    Value")
        for ts_name, time_h, value in timeseries:
            h = int(time_h)
            m = int((time_h - h) * 60)
            lines.append(f"{ts_name:16s} {h}:{m:02d} {value:.6f}")
        lines.append("")
    lines.append("[REPORT]")
    lines.append("INPUT NO")
    lines.append("NODES ALL")
    lines.append("LINKS ALL")
    lines.append("")
    return "\n".join(lines)
