"""Urban drainage and SWMM-style coupling skeleton for SWE2D."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from swe2d_extensions import (
    CouplingDiagnostics,
    DrainageCouplingEngine,
    DrainageLink,
    DrainageNode,
    DrainageSolverMode,
    InletExchange,
    OutfallExchange,
    PipeNetworkConfig,
    circular_area_from_diameter,
    circular_section_from_depth,
    circular_wet_perimeter_full,
    convert_cell_flows_to_depth_rates,
    compute_orifice_flow,
    compute_weir_flow,
    compute_pipe_manning_capacity_full,
)


def _interp_rating_curve(table: list, wse_m: float) -> float:
    """Linear interpolation on a (WSE_m, Q_m3s) rating table sorted ascending by WSE.

    Returns 0 when wse_m is below the lowest entry; clamps to the highest Q
    when wse_m exceeds the table range.
    """
    if not table:
        return 0.0
    wse_m = float(wse_m)
    if wse_m <= float(table[0][0]):
        return 0.0
    if wse_m >= float(table[-1][0]):
        return float(table[-1][1])
    for i in range(1, len(table)):
        w0, q0 = float(table[i - 1][0]), float(table[i - 1][1])
        w1, q1 = float(table[i][0]), float(table[i][1])
        if w0 <= wse_m <= w1:
            dw = w1 - w0
            return q0 if dw <= 0.0 else q0 + (wse_m - w0) / dw * (q1 - q0)
    return 0.0


class SWE2DUrbanDrainageModule(DrainageCouplingEngine):
    """
    Urban drainage solver: 2D surface <-> 1D pipe-network coupling.

    Three solver equation sets are available via PipeNetworkConfig.solver_mode:

    EGL (0) — Energy-grade-line (Bernoulli + Manning friction + minor losses).
        Models full or partial-flow conduits using the combined resistance formula::

            dH = Q^2 * [n^2*L/(A^2*R_h^(4/3))  +  (K_entry+K_exit)/(2g*A^2)]
            Q  = sqrt(|dH| / C_total) * sign(dH)

        K_entry (default 0.50) and K_exit (default 1.00) can be overridden via
        link.metadata["entry_loss_k"] / ["exit_loss_k"].  Appropriate for
        pressurised storm-drain design; analogous to FHWA HEC-22 outlet-control.

    DIFFUSION (1) — Diffusion-wave: slope-driven Manning flow with partial-flow
        circular-section hydraulic geometry::

            Q = (1/n) * A(y_avg) * R_h(y_avg)^(2/3) * sqrt(|S_w|) * sign(S_w)
            S_w = (H0 - H1) / L

        Better for partially-full gravity sewers and open-channel reaches.
        No minor losses; velocity-head change is neglected.

    DYNAMIC (2) — Full 1D Saint-Venant semi-implicit momentum per link::

            Q^{n+1} = (Q^n + dt * g*A * dH/L)
                      / (1 + dt * g * n^2 * |Q^n| / (A * R_h^(4/3)))

        Link flow is a *state variable* that evolves each sub-step rather than
        being recomputed algebraically.  Captures surge, bore propagation, and
        backwater transients.  Pair with coupling_substeps > 1 for stability.

    Outfall boundary conditions (node_type == "outfall", for nodes not coupled
    to a 2D cell) are governed by DrainageNode.outfall_mode:
        free             — depth reset to 0 each step (freely draining outfall).
        fixed_wse        — head clamped to outfall_fixed_wse each step.
        stage_discharge  — outflow read from outfall_rating_table [(wse, Q), ...].
    """

    def _node_by_id(self, node_id: str) -> DrainageNode:
        idx = self._node_index.get(node_id, -1)
        if idx < 0:
            raise KeyError(f"Unknown drainage node '{node_id}'")
        return self.cfg.nodes[idx]

    def _node_head(self, node: DrainageNode) -> float:
        d = max(0.0, float(self.state.node_depth.get(node.node_id, 0.0)))
        return float(node.invert_elev) + d

    def _head_deadband(self) -> float:
        return max(0.0, float(getattr(self.cfg, "head_deadband_m", 1.0e-3)))

    def _dynamic_flow_relaxation(self) -> float:
        return min(1.0, max(0.0, float(getattr(self.cfg, "dynamic_flow_relaxation", 1.0))))

    def _adaptive_depth_fraction(self) -> float:
        return min(1.0, max(1.0e-3, float(getattr(self.cfg, "adaptive_depth_fraction", 0.2))))

    def _adaptive_wave_courant(self) -> float:
        return max(1.0e-3, float(getattr(self.cfg, "adaptive_wave_courant", 0.5)))

    def _max_adaptive_substeps(self) -> int:
        return max(1, int(getattr(self.cfg, "max_coupling_substeps", 64)))

    def _effective_head_difference(self, head_up: float, head_dn: float) -> float:
        dh = float(head_up) - float(head_dn)
        deadband = self._head_deadband()
        if abs(dh) <= deadband:
            return 0.0
        return math.copysign(abs(dh) - deadband, dh)

    def _node_area_m2(self, node_id: str) -> float:
        return max(1.0, float(self._node_area.get(node_id, 50.0)))

    def _limit_flow_by_volume(self, q_cms: float, available_volume_m3: float, dt_s: float) -> float:
        if q_cms <= 0.0:
            return 0.0
        if dt_s <= 0.0:
            return 0.0
        return min(float(q_cms), max(0.0, float(available_volume_m3)) / float(dt_s))

    def _adaptive_substep_count(self, dt_s: float, solver_mode: DrainageSolverMode) -> int:
        if dt_s <= 0.0 or not self.cfg.nodes:
            return 1

        node_abs_q: Dict[str, float] = {n.node_id: 0.0 for n in self.cfg.nodes}
        dt_limit = float("inf")
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", 9.81)))

        for link in self.cfg.links:
            q_est = abs(float(self.state.link_flow.get(link.link_id, 0.0)))
            node_abs_q[link.from_node_id] = node_abs_q.get(link.from_node_id, 0.0) + q_est
            node_abs_q[link.to_node_id] = node_abs_q.get(link.to_node_id, 0.0) + q_est

            if solver_mode == DrainageSolverMode.DYNAMIC and not self._use_simplified_link_model(link):
                diameter = float(link.diameter or link.metadata.get("diameter", 0.0) or 0.0)
                length = max(1.0, float(link.length or 1.0))
                if diameter > 0.0:
                    wave_celerity = math.sqrt(g * max(1.0e-3, diameter))
                    if wave_celerity > 0.0:
                        dt_limit = min(dt_limit, self._adaptive_wave_courant() * length / wave_celerity)

        for node in self.cfg.nodes:
            q_sum = node_abs_q.get(node.node_id, 0.0)
            if q_sum <= 0.0:
                continue
            area = self._node_area_m2(node.node_id)
            max_depth = max(0.0, float(node.max_depth))
            allowed_depth_change = max(1.0e-2, min(5.0e-2, self._adaptive_depth_fraction() * max(max_depth, 0.1)))
            dt_limit = min(dt_limit, area * allowed_depth_change / q_sum)

        if not math.isfinite(dt_limit) or dt_limit <= 0.0:
            return 1
        return min(self._max_adaptive_substeps(), max(1, int(math.ceil(dt_s / dt_limit))))

    def _estimate_link_flow(self, link: DrainageLink) -> float:
        n0 = self._node_by_id(link.from_node_id)
        n1 = self._node_by_id(link.to_node_id)
        h0 = self._node_head(n0)
        h1 = self._node_head(n1)
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", 9.81)))
        dh = self._effective_head_difference(h0, h1)
        if abs(dh) <= 1.0e-12:
            return 0.0

        diameter = float(link.diameter or link.metadata.get("diameter", 0.0) or 0.0)
        area = float(link.metadata.get("area_m2", 0.0) or 0.0)
        if area <= 0.0 and diameter > 0.0:
            area = circular_area_from_diameter(diameter)
        if area <= 0.0:
            return 0.0

        length = max(1.0, float(link.length or 1.0))
        slope = max(1.0e-6, abs(dh) / length)
        q_orifice = compute_orifice_flow(
            h0,
            h1,
            area,
            discharge_coeff=float(link.metadata.get("cd", 0.75)),
            g=g,
        )
        q_cap = compute_pipe_manning_capacity_full(
            diameter_m=max(diameter, float(link.metadata.get("equiv_diameter_m", 0.0))),
            slope_m_per_m=slope,
            roughness_n=float(link.roughness_n),
        )
        q_mag = abs(q_orifice)
        if q_cap > 0.0:
            q_mag = min(q_mag, q_cap)
        if link.max_flow is not None:
            q_mag = min(q_mag, max(0.0, float(link.max_flow)))
        return q_mag if dh >= 0.0 else -q_mag

    def _use_simplified_link_model(self, link: DrainageLink) -> bool:
        t = str(link.link_type or "").strip().lower()
        if t in {"lateral_simple", "lateral", "short_lateral"}:
            return True
        md = link.metadata or {}
        return bool(md.get("simplified", False) or md.get("ignore_inertia", False))

    # ------------------------------------------------------------------
    # Solver-mode implementations
    # ------------------------------------------------------------------

    def _egl_link_flow(self, link: DrainageLink) -> float:
        """EGL energy-balance flow: Manning friction + entry/exit minor losses.

        Full-pipe section is used when both node heads exceed the pipe crown;
        otherwise the average depth is used with circular partial-flow geometry.
        The resistance formula is::

            dH = Q^2 * [n^2*L/(A^2*R_h^(4/3))  +  (K_e+K_o)/(2g*A^2)]
            Q  = sqrt(|dH| / C_total) * sign(dH)

        Default K_entry = 0.50 (sharp-edged manhole entry),
                K_exit  = 1.00 (pipe discharges into downstream manhole).
        """
        n0 = self._node_by_id(link.from_node_id)
        n1 = self._node_by_id(link.to_node_id)
        h0 = self._node_head(n0)
        h1 = self._node_head(n1)
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", 9.81)))
        dh = self._effective_head_difference(h0, h1)
        if abs(dh) <= 1.0e-12:
            return 0.0

        diameter = float(link.diameter or link.metadata.get("diameter", 0.0) or 0.0)
        if diameter <= 0.0:
            return 0.0
        length = max(1.0, float(link.length or 1.0))
        n_mann = max(1.0e-6, float(link.roughness_n))
        K_e = float(link.metadata.get("entry_loss_k", 0.50))
        K_o = float(link.metadata.get("exit_loss_k", 1.00))

        crown0 = float(n0.invert_elev) + diameter
        crown1 = float(n1.invert_elev) + diameter
        if h0 >= crown0 and h1 >= crown1:
            area = circular_area_from_diameter(diameter)
            perim = circular_wet_perimeter_full(diameter)
        else:
            depth0 = max(0.0, min(h0 - float(n0.invert_elev), diameter))
            depth1 = max(0.0, min(h1 - float(n1.invert_elev), diameter))
            area, perim = circular_section_from_depth(0.5 * (depth0 + depth1), diameter)

        if area <= 0.0 or perim <= 0.0:
            return 0.0
        r_h = area / perim

        C_fric  = (n_mann ** 2 * length) / (area ** 2 * r_h ** (4.0 / 3.0))
        C_minor = (K_e + K_o) / (2.0 * g * area ** 2)
        C_total = C_fric + C_minor
        if C_total <= 0.0:
            return 0.0
        q_egl = math.sqrt(abs(dh) / C_total)

        q_cap = compute_pipe_manning_capacity_full(
            diameter_m=diameter,
            slope_m_per_m=max(1.0e-6, abs(dh) / length),
            roughness_n=n_mann,
        )
        if q_cap > 0.0:
            q_egl = min(q_egl, q_cap)
        if link.max_flow is not None:
            q_egl = min(q_egl, max(0.0, float(link.max_flow)))
        return q_egl if dh >= 0.0 else -q_egl

    def _diffusion_link_flow(self, link: DrainageLink) -> float:
        """Diffusion-wave flow: slope-driven Manning with partial circular geometry.

        The water-surface slope S_w = (H0 - H1) / L drives Manning flow computed
        at the average of the two-end hydraulic sections.  No minor losses;
        suitable for partially-full gravity sewers and open-channel reaches::

            Q = (1/n) * A(y_avg) * R_h(y_avg)^(2/3) * sqrt(S_w) * sign(dH)
        """
        n0 = self._node_by_id(link.from_node_id)
        n1 = self._node_by_id(link.to_node_id)
        h0 = self._node_head(n0)
        h1 = self._node_head(n1)
        dh = self._effective_head_difference(h0, h1)
        if abs(dh) <= 1.0e-12:
            return 0.0

        diameter = float(link.diameter or link.metadata.get("diameter", 0.0) or 0.0)
        if diameter <= 0.0:
            return 0.0
        length = max(1.0, float(link.length or 1.0))
        n_mann = max(1.0e-6, float(link.roughness_n))

        depth0 = max(0.0, min(h0 - float(n0.invert_elev), diameter))
        depth1 = max(0.0, min(h1 - float(n1.invert_elev), diameter))
        area, perim = circular_section_from_depth(0.5 * (depth0 + depth1), diameter)
        if area <= 0.0 or perim <= 0.0:
            return 0.0
        r_h = area / perim
        s_w = abs(dh) / length
        q_diff = (1.0 / n_mann) * area * (r_h ** (2.0 / 3.0)) * math.sqrt(s_w)
        if link.max_flow is not None:
            q_diff = min(q_diff, max(0.0, float(link.max_flow)))
        return q_diff if dh >= 0.0 else -q_diff

    def _dynamic_link_flow_update(self, link: DrainageLink, dt: float) -> float:
        """Semi-implicit Saint-Venant momentum update for one link (one sub-step).

        Integrates the 1D momentum equation with friction treated semi-implicitly
        for stability (analogous to SWMM's dynamic-wave solver)::

            dQ/dt = g*A*(H0 - H1)/L - g*A*S_f(Q)

            Q^{n+1} = (Q^n + dt * g*A * dH/L)
                      / (1 + dt * g * n^2 * |Q^n| / (A * R_h^(4/3)))

        The link's stored flow is the dynamic state variable; it must be
        initialised to 0 (handled by DrainageCouplingEngine.initialize).
        """
        n0 = self._node_by_id(link.from_node_id)
        n1 = self._node_by_id(link.to_node_id)
        h0 = self._node_head(n0)
        h1 = self._node_head(n1)
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", 9.81)))
        dh = self._effective_head_difference(h0, h1)
        dt_s = max(1.0e-9, float(dt))

        diameter = float(link.diameter or link.metadata.get("diameter", 0.0) or 0.0)
        if diameter <= 0.0:
            self.state.link_flow[link.link_id] = 0.0
            return 0.0
        length = max(1.0, float(link.length or 1.0))
        n_mann = max(1.0e-6, float(link.roughness_n))

        depth0 = max(0.0, min(h0 - float(n0.invert_elev), diameter))
        depth1 = max(0.0, min(h1 - float(n1.invert_elev), diameter))
        area, perim = circular_section_from_depth(0.5 * (depth0 + depth1), diameter)
        if area <= 0.0:
            self.state.link_flow[link.link_id] = 0.0
            return 0.0
        r_h = area / perim if perim > 0.0 else 0.0

        Q_old = float(self.state.link_flow.get(link.link_id, 0.0))
        pressure_accel = g * area * dh / length
        if r_h > 0.0 and abs(Q_old) > 0.0:
            # Semi-implicit friction linearisation denominator
            friction_denom = dt_s * g * n_mann ** 2 * abs(Q_old) / (area * r_h ** (4.0 / 3.0))
        else:
            friction_denom = 0.0
        Q_candidate = (Q_old + dt_s * pressure_accel) / (1.0 + friction_denom)
        if link.max_flow is not None:
            q_cap = max(0.0, float(link.max_flow))
            Q_candidate = max(-q_cap, min(q_cap, Q_candidate))
        relax = self._dynamic_flow_relaxation()
        Q_new = (1.0 - relax) * Q_old + relax * Q_candidate
        self.state.link_flow[link.link_id] = Q_new
        return Q_new

    def _apply_outfall_bc(self, dt_sub: float) -> None:
        """Apply boundary conditions for pure-1D outfall nodes each sub-step.

        Only acts on nodes with node_type == "outfall" that are NOT co-located
        with a 2D cell via OutfallExchange (those are handled in exchange_step).
        """
        for node in self.cfg.nodes:
            if node.node_type != "outfall":
                continue
            if node.node_id in self._outfall_exchange_nodes:
                continue  # 2D-coupled outfall: exchange_step manages depth
            mode = getattr(node, "outfall_mode", "free")
            if mode == "fixed_wse":
                fixed_wse = float(getattr(node, "outfall_fixed_wse", 0.0))
                target_d = max(0.0, min(
                    fixed_wse - float(node.invert_elev),
                    float(node.max_depth),
                ))
                self.state.node_depth[node.node_id] = target_d
            elif mode == "stage_discharge":
                table = getattr(node, "outfall_rating_table", [])
                if table:
                    node_head = float(node.invert_elev) + float(
                        self.state.node_depth.get(node.node_id, 0.0)
                    )
                    q_out = _interp_rating_curve(table, node_head)
                    if q_out > 0.0:
                        node_area = max(1.0, float(self._node_area.get(node.node_id, 50.0)))
                        d = float(self.state.node_depth.get(node.node_id, 0.0))
                        self.state.node_depth[node.node_id] = max(
                            0.0, d - q_out * dt_sub / node_area
                        )
            else:
                # free: drain to zero (unlimited downstream capacity)
                self.state.node_depth[node.node_id] = 0.0

    def _step_network_once(
        self, dt_sub: float, solver_mode: DrainageSolverMode
    ) -> Tuple[float, float, float]:
        """One sub-step of the 1D network solve.

        Returns (max_abs_link_flow, max_node_depth, net_node_inflow).
        """
        node_net_q: Dict[str, float] = {n.node_id: 0.0 for n in self.cfg.nodes}
        max_q = 0.0
        if solver_mode == DrainageSolverMode.DYNAMIC:
            for link in self.cfg.links:
                q = self._estimate_link_flow(link) if self._use_simplified_link_model(link) else self._dynamic_link_flow_update(link, dt_sub)
                if self._use_simplified_link_model(link):
                    self.state.link_flow[link.link_id] = q
                node_net_q[link.from_node_id] -= q
                node_net_q[link.to_node_id]   += q
                max_q = max(max_q, abs(q))
        elif solver_mode == DrainageSolverMode.DIFFUSION:
            for link in self.cfg.links:
                q = self._estimate_link_flow(link) if self._use_simplified_link_model(link) else self._diffusion_link_flow(link)
                self.state.link_flow[link.link_id] = q
                node_net_q[link.from_node_id] -= q
                node_net_q[link.to_node_id]   += q
                max_q = max(max_q, abs(q))
        else:
            # EGL (default)
            for link in self.cfg.links:
                q = self._estimate_link_flow(link) if self._use_simplified_link_model(link) else self._egl_link_flow(link)
                self.state.link_flow[link.link_id] = q
                node_net_q[link.from_node_id] -= q
                node_net_q[link.to_node_id]   += q
                max_q = max(max_q, abs(q))

        for node in self.cfg.nodes:
            area = max(1.0, float(self._node_area.get(node.node_id, 50.0)))
            d0 = max(0.0, float(self.state.node_depth.get(node.node_id, 0.0)))
            d1 = d0 + (node_net_q[node.node_id] * dt_sub / area)
            d1 = min(max(0.0, d1), max(0.0, float(node.max_depth)))
            self.state.node_depth[node.node_id] = d1

        self._apply_outfall_bc(dt_sub)

        max_depth = (
            max(float(d) for d in self.state.node_depth.values())
            if self.state.node_depth else 0.0
        )
        return max_q, max_depth, float(sum(node_net_q.values()))

    def solve_network_step(self, dt: float) -> Dict[str, float]:
        """Advance the 1D drainage network by *dt* seconds.

        Dispatches to the equation set selected by PipeNetworkConfig.solver_mode
        and runs ``coupling_substeps`` sub-steps of size ``dt / coupling_substeps``
        to allow the 1D solver to use a finer timestep than the 2D domain without
        requiring GPU sub-stepping (the exchange with the 2D surface only happens
        once per call, after all sub-steps complete).
        """
        dt_s = max(1.0e-6, float(dt))
        if not self._node_index:
            self.initialize()

        solver_mode = getattr(self.cfg, "solver_mode", DrainageSolverMode.EGL)
        substeps = max(
            max(1, int(getattr(self.cfg, "coupling_substeps", 1))),
            self._adaptive_substep_count(dt_s, solver_mode),
        )
        dt_sub = dt_s / substeps

        max_q     = 0.0
        max_depth = 0.0
        net_inflow = 0.0
        for _ in range(substeps):
            mq, md, ni = self._step_network_once(dt_sub, solver_mode)
            max_q      = max(max_q, mq)
            max_depth  = max(max_depth, md)
            net_inflow += ni

        diag = CouplingDiagnostics(
            dt_s=dt_s,
            net_node_inflow=net_inflow,
            max_node_depth=max_depth,
            max_link_flow=max_q,
        )
        return {
            "dt":                  diag.dt_s,
            "substeps_used":       substeps,
            "net_node_inflow":     diag.net_node_inflow,
            "max_node_depth":      diag.max_node_depth,
            "max_link_flow":       diag.max_link_flow,
            # backward-compat aliases
            "net_node_inflow_cms": diag.net_node_inflow,
            "max_node_depth_m":    diag.max_node_depth,
            "max_link_flow_cms":   diag.max_link_flow,
        }

    def exchange_step(
        self,
        dt: float,
        cell_wse: Sequence[float],
        cell_area_m2: Sequence[float] | None = None,
        cell_depth_m: Sequence[float] | None = None,
    ):
        dt_s = max(1.0e-6, float(dt))
        g = max(1.0e-6, float(getattr(self.cfg, "gravity", 9.81)))
        if not self._node_index:
            self.initialize()
        n_cells = len(cell_wse)
        sinks = [0.0] * n_cells
        sources = [0.0] * n_cells
        deadband = self._head_deadband()
        area_arr = None if cell_area_m2 is None else list(cell_area_m2)
        depth_arr = None if cell_depth_m is None else list(cell_depth_m)

        for inlet in self.cfg.inlets:
            ci = int(inlet.cell_id)
            if ci < 0 or ci >= n_cells:
                continue
            try:
                node = self._node_by_id(inlet.node_id)
            except KeyError:
                continue

            wse_surface = float(cell_wse[ci])
            wse_node = self._node_head(node)
            crest = float(inlet.crest_elev)
            capture_head = max(0.0, wse_surface - max(wse_node, crest) - deadband)
            length = max(0.0, float(inlet.length))
            area = max(0.0, float(inlet.area))
            cw = max(0.0, float(inlet.coeff_weir))
            co = max(0.0, float(inlet.coeff_orifice))
            if capture_head > 0.0:
                # Weir before submergence, orifice after submergence.
                submerged_capture = (wse_node > crest)
                if submerged_capture:
                    a_eff = area if area > 0.0 else max(0.0, length) * max(0.01, capture_head)
                    q_capture = compute_orifice_flow(
                        head_up_m=wse_surface,
                        head_down_m=wse_node,
                        area_m2=a_eff,
                        discharge_coeff=co,
                        g=g,
                        max_flow=inlet.max_capture,
                    )
                else:
                    q_capture = compute_weir_flow(
                        upstream_wse_m=wse_surface,
                        downstream_wse_m=wse_node,
                        crest_elev_m=crest,
                        width_m=length,
                        coeff=cw,
                        max_flow=inlet.max_capture,
                    )
                q_capture = max(0.0, q_capture)
                node_area = self._node_area_m2(node.node_id)
                d = float(self.state.node_depth.get(node.node_id, 0.0))
                remaining_node_volume = max(0.0, float(node.max_depth) - d) * node_area
                q_capture = self._limit_flow_by_volume(q_capture, remaining_node_volume, dt_s)
                if area_arr is not None and depth_arr is not None and ci < len(area_arr) and ci < len(depth_arr):
                    available_surface_volume = max(0.0, float(area_arr[ci])) * max(0.0, float(depth_arr[ci]))
                    q_capture = self._limit_flow_by_volume(q_capture, available_surface_volume, dt_s)
                sinks[ci] += q_capture
                self.state.node_depth[node.node_id] = min(
                    float(node.max_depth),
                    max(0.0, d + q_capture * dt_s / node_area),
                )

            surcharge_head = max(0.0, wse_node - max(wse_surface, crest) - deadband)
            if surcharge_head > 0.0:
                submerged_relief = (wse_surface > crest)
                if submerged_relief:
                    a_eff = area if area > 0.0 else max(0.0, length) * max(0.01, surcharge_head)
                    q_relief = compute_orifice_flow(
                        head_up_m=wse_node,
                        head_down_m=wse_surface,
                        area_m2=a_eff,
                        discharge_coeff=co,
                        g=g,
                        max_flow=inlet.max_capture,
                    )
                else:
                    q_relief = compute_weir_flow(
                        upstream_wse_m=wse_node,
                        downstream_wse_m=wse_surface,
                        crest_elev_m=crest,
                        width_m=length,
                        coeff=cw,
                        max_flow=inlet.max_capture,
                    )
                q_relief = max(0.0, q_relief)
                node_area = self._node_area_m2(node.node_id)
                d = float(self.state.node_depth.get(node.node_id, 0.0))
                available_node_volume = max(0.0, d) * node_area
                q_relief = self._limit_flow_by_volume(q_relief, available_node_volume, dt_s)
                sources[ci] += q_relief
                self.state.node_depth[node.node_id] = max(0.0, d - q_relief * dt_s / node_area)

        # --- Outfall exchange ---
        # Two-way coupling at outfall nodes that are located within the 2D mesh:
        #   Surcharge  : network head > surface WSE  -> inject flow into 2D cell
        #   Backwater  : surface WSE > network head  -> drain 2D cell into outfall
        for outfall in self.cfg.outfalls:
            ci = int(outfall.cell_id)
            if ci < 0 or ci >= n_cells:
                continue
            try:
                node = self._node_by_id(outfall.node_id)
            except KeyError:
                continue

            d_pipe = max(0.0, float(outfall.diameter))
            area_pipe = circular_area_from_diameter(d_pipe) if d_pipe > 0.0 else 0.0
            if area_pipe <= 0.0:
                continue

            wse_surface = float(cell_wse[ci])
            zero_storage = bool(getattr(outfall, "zero_storage", False))
            if zero_storage:
                # Daylight outfall: do not use a local node storage bucket during
                # 2D exchange; hold node depth at invert for exchange head.
                self.state.node_depth[node.node_id] = 0.0
                wse_node = float(outfall.invert_elev)
                node_area = 1.0
                d_node = 0.0
            else:
                wse_node = self._node_head(node)
                node_area = max(1.0, float(self._node_area.get(node.node_id, 50.0)))
                d_node = float(self.state.node_depth.get(node.node_id, 0.0))

            if wse_node > wse_surface + deadband and wse_node > float(outfall.invert_elev):
                # Pressurized / surcharge: discharge from network into 2D surface cell
                q_out = compute_orifice_flow(
                    head_up_m=wse_node,
                    head_down_m=wse_surface,
                    area_m2=area_pipe,
                    discharge_coeff=float(outfall.coefficient),
                    g=g,
                    max_flow=outfall.max_flow,
                )
                q_out = max(0.0, q_out)
                if q_out > 0.0:
                    if not zero_storage:
                        available_node_volume = max(0.0, d_node) * node_area
                        q_out = self._limit_flow_by_volume(q_out, available_node_volume, dt_s)
                    sources[ci] += q_out
                    if not zero_storage:
                        self.state.node_depth[node.node_id] = max(
                            0.0, d_node - q_out * dt_s / node_area
                        )
            elif wse_surface > wse_node + deadband and wse_surface > float(outfall.invert_elev):
                # Backwater / submerged outfall: 2D surface drains into outfall node
                q_in = compute_orifice_flow(
                    head_up_m=wse_surface,
                    head_down_m=wse_node,
                    area_m2=area_pipe,
                    discharge_coeff=float(outfall.coefficient),
                    g=g,
                    max_flow=outfall.max_flow,
                )
                q_in = max(0.0, q_in)
                if q_in > 0.0:
                    if not zero_storage:
                        remaining_node_volume = max(0.0, float(node.max_depth) - d_node) * node_area
                        q_in = self._limit_flow_by_volume(q_in, remaining_node_volume, dt_s)
                    if area_arr is not None and depth_arr is not None and ci < len(area_arr) and ci < len(depth_arr):
                        available_surface_volume = max(0.0, float(area_arr[ci])) * max(0.0, float(depth_arr[ci]))
                        q_in = self._limit_flow_by_volume(q_in, available_surface_volume, dt_s)
                    sinks[ci] += q_in
                    if not zero_storage:
                        self.state.node_depth[node.node_id] = min(
                            float(node.max_depth),
                            max(0.0, d_node + q_in * dt_s / node_area),
                        )

        return sinks, sources

    def apply_surface_exchange(
        self,
        dt: float,
        cell_wse: Sequence[float],
        cell_area_m2: Sequence[float] | None = None,
        cell_depth_m: Sequence[float] | None = None,
    ) -> List[float]:
        sinks, sources = self.exchange_step(
            dt=dt,
            cell_wse=cell_wse,
            cell_area_m2=cell_area_m2,
            cell_depth_m=cell_depth_m,
        )
        if not sinks and not sources:
            return [0.0] * len(cell_wse)
        combined = [0.0] * len(cell_wse)
        for i, v in enumerate(sinks):
            if i < len(combined):
                combined[i] -= v
        for i, v in enumerate(sources):
            if i < len(combined):
                combined[i] += v
        return combined

    def surface_exchange_source_rate(
        self,
        dt: float,
        cell_wse: Sequence[float],
        cell_area_m2: Sequence[float],
        cell_depth_m: Sequence[float] | None = None,
    ) -> List[float]:
        """Return per-cell depth-rate sources [m/s] for 2D coupling."""
        net_flow = self.apply_surface_exchange(
            dt=dt,
            cell_wse=cell_wse,
            cell_area_m2=cell_area_m2,
            cell_depth_m=cell_depth_m,
        )
        return convert_cell_flows_to_depth_rates(net_flow, cell_area_m2)


__all__ = [
    "DrainageNode",
    "DrainageLink",
    "DrainageSolverMode",
    "InletExchange",
    "OutfallExchange",
    "PipeNetworkConfig",
    "SWE2DUrbanDrainageModule",
]
