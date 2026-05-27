from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from stacked_bridge_coupling import BridgeLossLaw


@dataclass
class ToyConfig:
    nx: int = 24
    ny: int = 12
    dx_ft: float = 2.0
    dy_ft: float = 2.0
    dt_s: float = 0.05
    steps: int = 160
    g_ft_s2: float = 32.17
    rho_slug_ft3: float = 1.0
    waterline_row: int = 3
    deck_x0: int = 9
    deck_x1: int = 15
    deck_y0: int = 5
    deck_y1: int = 8
    inlet_velocity_start_ft_s: float = 1.0
    inlet_velocity_end_ft_s: float = 4.0
    inlet_ramp_duration_s: float = 4.0
    outlet_velocity_factor: float = 0.92
    loss_k_upstream: float = 0.8
    loss_k_downstream: float = 0.8
    poisson_iters: int = 280
    poisson_omega: float = 1.4
    poisson_tol: float = 1e-7
    video_fps: int = 15
    z_units: int = 24
    pier_width_units: int = 3
    video_output: str = "docs/stacked_bridge_toy_extruded_profile_cross.mp4"
    velocity_quiver_stride: int = 2


def build_geometry(cfg: ToyConfig, include_pier: bool = False) -> Dict[str, np.ndarray]:
    solid = np.zeros((cfg.ny, cfg.nx), dtype=bool)

    # Bed row.
    solid[cfg.ny - 1, :] = True

    # Deck block.
    solid[cfg.deck_y0 : cfg.deck_y1, cfg.deck_x0 : cfg.deck_x1] = True

    if include_pier:
        # Pier in profile view: below deck, across full deck length.
        solid[cfg.deck_y1 : cfg.ny, cfg.deck_x0 : cfg.deck_x1] = True

    rows = np.arange(cfg.ny)[:, None]
    fluid = (~solid) & (rows >= cfg.waterline_row)
    air = (~solid) & (~fluid)

    # Gauge pressure is fixed to atmospheric at free-surface cells only.
    above_air = np.zeros_like(fluid)
    above_air[1:, :] = air[:-1, :]
    free_surface = fluid & (above_air | (rows == cfg.waterline_row))

    underdeck_rows = np.arange(cfg.ny)[:, None] >= cfg.deck_y1
    underdeck_cols = (np.arange(cfg.nx)[None, :] >= cfg.deck_x0) & (
        np.arange(cfg.nx)[None, :] < cfg.deck_x1
    )
    underdeck = fluid & underdeck_rows & underdeck_cols

    overdeck_rows = np.arange(cfg.ny)[:, None] <= (cfg.deck_y0 - 1)
    overdeck = fluid & overdeck_rows & underdeck_cols

    return {
        "solid": solid,
        "fluid": fluid,
        "air": air,
        "free_surface": free_surface,
        "underdeck": underdeck,
        "overdeck": overdeck,
    }


def _rebuild_fluid_from_column_volume(
    col_volume_ft2: np.ndarray,
    solid: np.ndarray,
    cfg: ToyConfig,
) -> Dict[str, np.ndarray]:
    ny, nx = solid.shape
    fluid = np.zeros((ny, nx), dtype=bool)
    fill_frac = np.zeros((ny, nx), dtype=float)
    surface_row = np.full(nx, np.nan, dtype=float)

    for j in range(nx):
        non_solid_rows = [i for i in range(ny - 1, -1, -1) if not solid[i, j]]
        if not non_solid_rows:
            col_volume_ft2[j] = 0.0
            continue

        max_volume = len(non_solid_rows) * cfg.dy_ft
        vol = float(np.clip(col_volume_ft2[j], 0.0, max_volume))
        col_volume_ft2[j] = vol
        remaining = vol

        for i in non_solid_rows:
            if remaining <= 1e-12:
                break
            frac = min(1.0, remaining / cfg.dy_ft)
            fill_frac[i, j] = frac
            fluid[i, j] = True
            remaining -= frac * cfg.dy_ft

        wet_rows = np.where(fill_frac[:, j] > 0.0)[0]
        if wet_rows.size > 0:
            i_top = int(wet_rows[0])
            f_top = float(fill_frac[i_top, j])
            # Cell-centered imshow coordinates: top boundary of row i is i-0.5.
            # If top cell is partially filled, place free surface inside the cell.
            surface_row[j] = i_top + 0.5 - f_top

    air = (~solid) & (~fluid)
    free_surface = np.zeros_like(fluid)
    for j in range(nx):
        wet_rows = np.where(fill_frac[:, j] > 0.0)[0]
        if wet_rows.size > 0:
            free_surface[int(wet_rows[0]), j] = True

    rows = np.arange(ny)[:, None]
    underdeck_rows = rows >= cfg.deck_y1
    underdeck_cols = (np.arange(nx)[None, :] >= cfg.deck_x0) & (
        np.arange(nx)[None, :] < cfg.deck_x1
    )
    underdeck = fluid & underdeck_rows & underdeck_cols

    overdeck_rows = rows <= (cfg.deck_y0 - 1)
    overdeck = fluid & overdeck_rows & underdeck_cols

    col_depth_ft = np.sum(fill_frac, axis=0).astype(float) * cfg.dy_ft

    return {
        "fluid": fluid,
        "air": air,
        "free_surface": free_surface,
        "underdeck": underdeck,
        "overdeck": overdeck,
        "fill_frac": fill_frac,
        "surface_row": surface_row,
        "col_depth_ft": col_depth_ft,
    }


def _inlet_velocity_at_time(cfg: ToyConfig, time_s: float) -> float:
    if cfg.inlet_ramp_duration_s <= 0.0:
        return cfg.inlet_velocity_end_ft_s
    ramp = np.clip(time_s / cfg.inlet_ramp_duration_s, 0.0, 1.0)
    return (1.0 - ramp) * cfg.inlet_velocity_start_ft_s + ramp * cfg.inlet_velocity_end_ft_s


def _pier_bands(cfg: ToyConfig) -> List[tuple[int, int]]:
    # For z=24 and pier=3, this yields [6:9] and [15:18], i.e., 3 equal openings of 6.
    z = int(cfg.z_units)
    w = int(cfg.pier_width_units)
    opening = (z - 2 * w) / 3.0
    o = int(round(opening))
    b1 = (o, o + w)
    b2 = (2 * o + w, 2 * o + 2 * w)
    return [b1, b2]


def _z_is_pier(cfg: ToyConfig) -> np.ndarray:
    zmask = np.zeros(cfg.z_units, dtype=bool)
    for a, b in _pier_bands(cfg):
        zmask[a:b] = True
    return zmask


def _zero_blocked_faces(u: np.ndarray, v: np.ndarray, fluid: np.ndarray) -> None:
    ny, nx = fluid.shape

    # u faces (between left and right cells).
    for i in range(ny):
        u[i, 0] = u[i, 0] if fluid[i, 0] else 0.0
        u[i, nx] = u[i, nx] if fluid[i, nx - 1] else 0.0
        for j in range(1, nx):
            if not (fluid[i, j - 1] and fluid[i, j]):
                u[i, j] = 0.0

    # v faces (between top and bottom cells).
    for j in range(nx):
        v[0, j] = 0.0
        v[ny, j] = 0.0
        for i in range(1, ny):
            if not (fluid[i - 1, j] and fluid[i, j]):
                v[i, j] = 0.0


def _apply_local_losses(
    u: np.ndarray,
    fluid: np.ndarray,
    cfg: ToyConfig,
) -> None:
    # Losses are applied at under-deck entry/exit faces as an equivalent
    # acceleration sink on the face-normal velocity.
    i0 = cfg.deck_y1
    i1 = cfg.ny - 1
    j_up = cfg.deck_x0
    j_dn = cfg.deck_x1
    law = BridgeLossLaw(cfg.loss_k_upstream, cfg.loss_k_downstream)

    for i in range(i0, i1):
        # Upstream deck face.
        if j_up > 0 and j_up < cfg.nx and fluid[i, j_up - 1] and fluid[i, j_up]:
            u[i, j_up] = law.apply_face_velocity(u[i, j_up], cfg.dt_s, cfg.dx_ft, upstream=True)

        # Downstream deck face.
        if j_dn > 0 and j_dn < cfg.nx and fluid[i, j_dn - 1] and fluid[i, j_dn]:
            u[i, j_dn] = law.apply_face_velocity(u[i, j_dn], cfg.dt_s, cfg.dx_ft, upstream=False)


def _divergence(u: np.ndarray, v: np.ndarray, cfg: ToyConfig) -> np.ndarray:
    return (u[:, 1:] - u[:, :-1]) / cfg.dx_ft + (v[1:, :] - v[:-1, :]) / cfg.dy_ft


def _mean_velocity_and_froude(
    u: np.ndarray,
    v: np.ndarray,
    fluid: np.ndarray,
    col_depth_ft: np.ndarray,
    cfg: ToyConfig,
) -> tuple[float, float]:
    uc = 0.5 * (u[:, 1:] + u[:, :-1])
    vc = 0.5 * (v[1:, :] + v[:-1, :])
    speed = np.sqrt(uc * uc + vc * vc)

    if not np.any(fluid):
        return 0.0, 0.0

    mean_vel = float(np.mean(speed[fluid]))

    depth_field = np.broadcast_to(col_depth_ft[None, :], fluid.shape)
    denom = np.sqrt(np.maximum(cfg.g_ft_s2 * depth_field, 1e-12))
    froude = speed / denom
    mean_fr = float(np.mean(froude[fluid]))
    return mean_vel, mean_fr


def _render_frame(
    p: np.ndarray,
    fluid: np.ndarray,
    solid: np.ndarray,
    uc: np.ndarray,
    vc: np.ndarray,
    surface_row: np.ndarray,
    cfg: ToyConfig,
    step: int,
    time_s: float,
    mean_vel: float,
    mean_fr: float,
    mean_p: float,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    p_plot = np.full_like(p, np.nan, dtype=float)
    p_plot[fluid] = p[fluid]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    im = ax.imshow(p_plot, cmap="viridis", interpolation="nearest", origin="upper")

    solid_mask = np.where(solid, 1.0, np.nan)
    ax.imshow(solid_mask, cmap="Greys", interpolation="nearest", origin="upper", alpha=0.7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Gauge pressure (toy units)")

    # Free-surface polyline (column-wise reconstructed).
    x = np.arange(p.shape[1], dtype=float)
    valid_eta = np.isfinite(surface_row)
    if np.any(valid_eta):
        ax.plot(x[valid_eta], surface_row[valid_eta], color="cyan", linewidth=2.0, label="free surface")

    # Velocity vectors (downsampled for readability).
    stride = max(1, int(cfg.velocity_quiver_stride))
    qy, qx = np.mgrid[0 : p.shape[0] : stride, 0 : p.shape[1] : stride]
    uq = uc[::stride, ::stride]
    vq = vc[::stride, ::stride]
    fq = fluid[::stride, ::stride]
    uq = np.where(fq, uq, np.nan)
    vq = np.where(fq, vq, np.nan)
    ax.quiver(
        qx,
        qy,
        uq,
        vq,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=2.5,
        width=0.003,
        alpha=0.8,
    )

    ax.set_title("Stacked-Region Toy: Pressure Shading + Live Metrics")
    ax.set_xlabel("x cell index")
    ax.set_ylabel("y cell index")

    txt = (
        f"step: {step}   t: {time_s:7.2f} s\n"
        f"mean velocity: {mean_vel:8.4f} ft/s\n"
        f"mean Froude:   {mean_fr:8.4f}\n"
        f"mean pressure: {mean_p:8.4f}"
    )
    ax.text(
        0.02,
        0.98,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 6},
    )

    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return frame


def _write_video(frames: List[np.ndarray], output_path: Path, fps: int) -> Path:
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(str(output_path), fps=fps, codec="libx264") as writer:
            for frame in frames:
                writer.append_data(frame)
        return output_path
    except Exception:
        gif_path = output_path.with_suffix(".gif")
        imageio.mimsave(str(gif_path), frames, fps=fps)
        return gif_path


def _solve_pressure(
    rhs: np.ndarray,
    fluid: np.ndarray,
    free_surface: np.ndarray,
    cfg: ToyConfig,
) -> np.ndarray:
    ny, nx = rhs.shape
    p = np.zeros((ny, nx), dtype=float)

    inv_dx2 = 1.0 / (cfg.dx_ft * cfg.dx_ft)
    inv_dy2 = 1.0 / (cfg.dy_ft * cfg.dy_ft)

    for _ in range(cfg.poisson_iters):
        max_delta = 0.0
        for i in range(ny):
            for j in range(nx):
                if not fluid[i, j]:
                    continue
                if free_surface[i, j]:
                    p[i, j] = 0.0
                    continue

                diag = 0.0
                accum = 0.0

                # Left.
                if j - 1 >= 0 and fluid[i, j - 1]:
                    diag += inv_dx2
                    accum += inv_dx2 * p[i, j - 1]
                # Right.
                if j + 1 < nx and fluid[i, j + 1]:
                    diag += inv_dx2
                    accum += inv_dx2 * p[i, j + 1]
                # Up.
                if i - 1 >= 0 and fluid[i - 1, j]:
                    diag += inv_dy2
                    accum += inv_dy2 * p[i - 1, j]
                # Down.
                if i + 1 < ny and fluid[i + 1, j]:
                    diag += inv_dy2
                    accum += inv_dy2 * p[i + 1, j]

                if diag <= 0.0:
                    continue

                p_new = (accum - rhs[i, j]) / diag
                updated = (1.0 - cfg.poisson_omega) * p[i, j] + cfg.poisson_omega * p_new
                max_delta = max(max_delta, abs(updated - p[i, j]))
                p[i, j] = updated

        if max_delta < cfg.poisson_tol:
            break

    p[~fluid] = 0.0
    p[free_surface] = 0.0
    return p


def run_toy_simulation(cfg: ToyConfig, capture_frames: bool = False) -> Dict[str, np.ndarray]:
    geom = build_geometry(cfg, include_pier=False)
    solid = geom["solid"]
    fluid = geom["fluid"].copy()
    free_surface = geom["free_surface"].copy()
    underdeck = geom["underdeck"].copy()
    overdeck = geom["overdeck"].copy()

    ny, nx = fluid.shape
    u = np.zeros((ny, nx + 1), dtype=float)
    v = np.zeros((ny + 1, nx), dtype=float)

    col_volume_ft2 = np.sum(fluid, axis=0).astype(float) * cfg.dy_ft
    col_capacity_ft2 = np.sum(~solid, axis=0).astype(float) * cfg.dy_ft

    div_l2 = np.zeros(cfg.steps, dtype=float)
    underdeck_q = np.zeros(cfg.steps, dtype=float)
    overdeck_q = np.zeros(cfg.steps, dtype=float)
    p_under = np.zeros(cfg.steps, dtype=float)
    p_over = np.zeros(cfg.steps, dtype=float)
    mean_velocity = np.zeros(cfg.steps, dtype=float)
    mean_froude = np.zeros(cfg.steps, dtype=float)
    inlet_velocity_hist = np.zeros(cfg.steps, dtype=float)

    pressure_frames = np.zeros((cfg.steps, ny, nx), dtype=float)
    fluid_frames = np.zeros((cfg.steps, ny, nx), dtype=bool)
    surface_row_frames = np.zeros((cfg.steps, nx), dtype=float)

    frames_rgb: Optional[List[np.ndarray]] = [] if capture_frames else None

    # Use the first/last active face as bookkeeping sections.
    j_under = min(cfg.deck_x0 + 1, nx - 1)
    j_over = max(cfg.deck_x0 - 1, 1)

    for n in range(cfg.steps):
        time_s = n * cfg.dt_s
        inlet_velocity = _inlet_velocity_at_time(cfg, time_s)
        inlet_velocity_hist[n] = inlet_velocity

        # Inlet velocity on the left boundary for wetted cells.
        u[:, 0] = 0.0
        u[fluid[:, 0], 0] = inlet_velocity

        # Controlled outflow to allow visible free-surface transients.
        u[:, nx] = cfg.outlet_velocity_factor * u[:, nx - 1]

        _zero_blocked_faces(u, v, fluid)
        _apply_local_losses(u, fluid, cfg)

        rhs = (cfg.rho_slug_ft3 / cfg.dt_s) * _divergence(u, v, cfg)
        rhs[~fluid] = 0.0
        p = _solve_pressure(rhs, fluid, free_surface, cfg)

        # Projection step for u faces.
        for i in range(ny):
            for j in range(1, nx):
                if fluid[i, j - 1] and fluid[i, j]:
                    gradp = (p[i, j] - p[i, j - 1]) / cfg.dx_ft
                    u[i, j] -= (cfg.dt_s / cfg.rho_slug_ft3) * gradp
                else:
                    u[i, j] = 0.0

        # Projection step for v faces.
        for i in range(1, ny):
            for j in range(nx):
                if fluid[i - 1, j] and fluid[i, j]:
                    gradp = (p[i, j] - p[i - 1, j]) / cfg.dy_ft
                    v[i, j] -= (cfg.dt_s / cfg.rho_slug_ft3) * gradp
                else:
                    v[i, j] = 0.0

        _zero_blocked_faces(u, v, fluid)

        div = _divergence(u, v, cfg)
        div_l2[n] = float(np.sqrt(np.mean((div[fluid]) ** 2)))

        # Update column water volume using depth-integrated x-fluxes and
        # rebuild the moving free surface from the updated column volumes.
        qx_faces = np.sum(u, axis=0) * cfg.dy_ft
        col_volume_ft2 += cfg.dt_s * (qx_faces[:-1] - qx_faces[1:])
        col_volume_ft2 = np.clip(col_volume_ft2, 0.0, col_capacity_ft2)

        dynamic = _rebuild_fluid_from_column_volume(col_volume_ft2, solid, cfg)
        fluid = dynamic["fluid"]
        free_surface = dynamic["free_surface"]
        underdeck = dynamic["underdeck"]
        overdeck = dynamic["overdeck"]
        fill_frac = dynamic["fill_frac"]
        surface_row = dynamic["surface_row"]
        col_depth_ft = dynamic["col_depth_ft"]

        _zero_blocked_faces(u, v, fluid)

        mean_velocity[n], mean_froude[n] = _mean_velocity_and_froude(
            u, v, fluid, col_depth_ft, cfg
        )

        under_rows = np.where(fluid[:, j_under - 1] & fluid[:, j_under])[0]
        over_rows = np.where(fluid[:, j_over - 1] & fluid[:, j_over])[0]

        underdeck_q[n] = float(np.sum(u[under_rows, j_under]) * cfg.dy_ft)
        overdeck_q[n] = float(np.sum(u[over_rows, j_over]) * cfg.dy_ft)

        p_under[n] = float(np.mean(p[underdeck])) if np.any(underdeck) else 0.0
        p_over[n] = float(np.mean(p[overdeck])) if np.any(overdeck) else 0.0

        pressure_frames[n] = p
        fluid_frames[n] = fluid
        surface_row_frames[n] = surface_row

        uc = 0.5 * (u[:, 1:] + u[:, :-1])
        vc = 0.5 * (v[1:, :] + v[:-1, :])

        if capture_frames and frames_rgb is not None:
            mean_p = float(np.mean(p[fluid])) if np.any(fluid) else 0.0
            frames_rgb.append(
                _render_frame(
                    p=p,
                    fluid=fluid,
                    solid=solid,
                    uc=uc,
                    vc=vc,
                    surface_row=surface_row,
                    cfg=cfg,
                    step=n,
                    time_s=time_s,
                    mean_vel=mean_velocity[n],
                    mean_fr=mean_froude[n],
                    mean_p=mean_p,
                )
            )

    out = {
        "div_l2": div_l2,
        "underdeck_discharge": underdeck_q,
        "overdeck_discharge": overdeck_q,
        "underdeck_pressure": p_under,
        "overdeck_pressure": p_over,
        "mean_velocity": mean_velocity,
        "mean_froude": mean_froude,
        "inlet_velocity": inlet_velocity_hist,
        "pressure_frames": pressure_frames,
        "fluid_frames": fluid_frames,
        "surface_row_frames": surface_row_frames,
        "geometry": {
            "solid": solid,
            "fluid": fluid,
            "free_surface": free_surface,
            "underdeck": underdeck,
            "overdeck": overdeck,
        },
    }

    if capture_frames and frames_rgb is not None:
        out["video_frames"] = np.array(frames_rgb, dtype=np.uint8)

    return out


def generate_toy_video(cfg: ToyConfig, output_path: Optional[str] = None) -> Path:
    out = run_toy_simulation(cfg, capture_frames=True)
    frames = [f for f in out["video_frames"]]
    target = Path(output_path or cfg.video_output)
    return _write_video(frames, target, fps=cfg.video_fps)


def run_toy_simulation_with_pier(
    cfg: ToyConfig, include_pier: bool, capture_frames: bool = False
) -> Dict[str, np.ndarray]:
    # Thin wrapper to reuse the solver with either an open or pierced profile.
    geom = build_geometry(cfg, include_pier=include_pier)

    # Copy of run_toy_simulation internals, using the chosen geometry.
    solid = geom["solid"]
    fluid = geom["fluid"].copy()
    free_surface = geom["free_surface"].copy()
    underdeck = geom["underdeck"].copy()
    overdeck = geom["overdeck"].copy()

    ny, nx = fluid.shape
    u = np.zeros((ny, nx + 1), dtype=float)
    v = np.zeros((ny + 1, nx), dtype=float)

    col_volume_ft2 = np.sum(fluid, axis=0).astype(float) * cfg.dy_ft
    col_capacity_ft2 = np.sum(~solid, axis=0).astype(float) * cfg.dy_ft

    div_l2 = np.zeros(cfg.steps, dtype=float)
    underdeck_q = np.zeros(cfg.steps, dtype=float)
    overdeck_q = np.zeros(cfg.steps, dtype=float)
    p_under = np.zeros(cfg.steps, dtype=float)
    p_over = np.zeros(cfg.steps, dtype=float)
    mean_velocity = np.zeros(cfg.steps, dtype=float)
    mean_froude = np.zeros(cfg.steps, dtype=float)
    inlet_velocity_hist = np.zeros(cfg.steps, dtype=float)

    pressure_frames = np.zeros((cfg.steps, ny, nx), dtype=float)
    fluid_frames = np.zeros((cfg.steps, ny, nx), dtype=bool)
    surface_row_frames = np.zeros((cfg.steps, nx), dtype=float)
    uc_frames = np.zeros((cfg.steps, ny, nx), dtype=float)
    vc_frames = np.zeros((cfg.steps, ny, nx), dtype=float)

    frames_rgb: Optional[List[np.ndarray]] = [] if capture_frames else None

    j_under = min(cfg.deck_x0 + 1, nx - 1)
    j_over = max(cfg.deck_x0 - 1, 1)

    for n in range(cfg.steps):
        time_s = n * cfg.dt_s
        inlet_velocity = _inlet_velocity_at_time(cfg, time_s)
        inlet_velocity_hist[n] = inlet_velocity

        u[:, 0] = 0.0
        u[fluid[:, 0], 0] = inlet_velocity
        u[:, nx] = cfg.outlet_velocity_factor * u[:, nx - 1]

        _zero_blocked_faces(u, v, fluid)
        _apply_local_losses(u, fluid, cfg)

        rhs = (cfg.rho_slug_ft3 / cfg.dt_s) * _divergence(u, v, cfg)
        rhs[~fluid] = 0.0
        p = _solve_pressure(rhs, fluid, free_surface, cfg)

        for i in range(ny):
            for j in range(1, nx):
                if fluid[i, j - 1] and fluid[i, j]:
                    gradp = (p[i, j] - p[i, j - 1]) / cfg.dx_ft
                    u[i, j] -= (cfg.dt_s / cfg.rho_slug_ft3) * gradp
                else:
                    u[i, j] = 0.0

        for i in range(1, ny):
            for j in range(nx):
                if fluid[i - 1, j] and fluid[i, j]:
                    gradp = (p[i, j] - p[i - 1, j]) / cfg.dy_ft
                    v[i, j] -= (cfg.dt_s / cfg.rho_slug_ft3) * gradp
                else:
                    v[i, j] = 0.0

        _zero_blocked_faces(u, v, fluid)

        div = _divergence(u, v, cfg)
        div_l2[n] = float(np.sqrt(np.mean((div[fluid]) ** 2))) if np.any(fluid) else 0.0

        qx_faces = np.sum(u, axis=0) * cfg.dy_ft
        col_volume_ft2 += cfg.dt_s * (qx_faces[:-1] - qx_faces[1:])
        col_volume_ft2 = np.clip(col_volume_ft2, 0.0, col_capacity_ft2)

        dynamic = _rebuild_fluid_from_column_volume(col_volume_ft2, solid, cfg)
        fluid = dynamic["fluid"]
        free_surface = dynamic["free_surface"]
        underdeck = dynamic["underdeck"]
        overdeck = dynamic["overdeck"]
        surface_row = dynamic["surface_row"]
        col_depth_ft = dynamic["col_depth_ft"]

        _zero_blocked_faces(u, v, fluid)

        mean_velocity[n], mean_froude[n] = _mean_velocity_and_froude(
            u, v, fluid, col_depth_ft, cfg
        )

        under_rows = np.where(fluid[:, j_under - 1] & fluid[:, j_under])[0]
        over_rows = np.where(fluid[:, j_over - 1] & fluid[:, j_over])[0]
        underdeck_q[n] = float(np.sum(u[under_rows, j_under]) * cfg.dy_ft)
        overdeck_q[n] = float(np.sum(u[over_rows, j_over]) * cfg.dy_ft)

        p_under[n] = float(np.mean(p[underdeck])) if np.any(underdeck) else 0.0
        p_over[n] = float(np.mean(p[overdeck])) if np.any(overdeck) else 0.0

        uc = 0.5 * (u[:, 1:] + u[:, :-1])
        vc = 0.5 * (v[1:, :] + v[:-1, :])

        pressure_frames[n] = p
        fluid_frames[n] = fluid
        surface_row_frames[n] = surface_row
        uc_frames[n] = uc
        vc_frames[n] = vc

        if capture_frames and frames_rgb is not None:
            mean_p = float(np.mean(p[fluid])) if np.any(fluid) else 0.0
            frames_rgb.append(
                _render_frame(
                    p=p,
                    fluid=fluid,
                    solid=solid,
                    uc=uc,
                    vc=vc,
                    surface_row=surface_row,
                    cfg=cfg,
                    step=n,
                    time_s=time_s,
                    mean_vel=mean_velocity[n],
                    mean_fr=mean_froude[n],
                    mean_p=mean_p,
                )
            )

    out = {
        "div_l2": div_l2,
        "underdeck_discharge": underdeck_q,
        "overdeck_discharge": overdeck_q,
        "underdeck_pressure": p_under,
        "overdeck_pressure": p_over,
        "mean_velocity": mean_velocity,
        "mean_froude": mean_froude,
        "inlet_velocity": inlet_velocity_hist,
        "pressure_frames": pressure_frames,
        "fluid_frames": fluid_frames,
        "surface_row_frames": surface_row_frames,
        "uc_frames": uc_frames,
        "vc_frames": vc_frames,
        "geometry": {
            "solid": solid,
            "fluid": fluid,
            "free_surface": free_surface,
            "underdeck": underdeck,
            "overdeck": overdeck,
        },
    }

    if capture_frames and frames_rgb is not None:
        out["video_frames"] = np.array(frames_rgb, dtype=np.uint8)
    return out


def _render_extruded_frame(
    open_data: Dict[str, np.ndarray],
    pier_data: Dict[str, np.ndarray],
    z_is_pier: np.ndarray,
    step: int,
    cfg: ToyConfig,
) -> np.ndarray:
    import matplotlib.pyplot as plt

    p_o = open_data["pressure_frames"][step]
    f_o = open_data["fluid_frames"][step]
    s_o = open_data["geometry"]["solid"]
    u_o = open_data["uc_frames"][step]
    v_o = open_data["vc_frames"][step]
    eta_o = open_data["surface_row_frames"][step]

    p_p = pier_data["pressure_frames"][step]
    f_p = pier_data["fluid_frames"][step]
    s_p = pier_data["geometry"]["solid"]
    eta_p = pier_data["surface_row_frames"][step]

    ny, nx = p_o.shape
    x_center = (cfg.deck_x0 + cfg.deck_x1) // 2

    p_yz = np.zeros((ny, cfg.z_units), dtype=float)
    f_yz = np.zeros((ny, cfg.z_units), dtype=bool)
    s_yz = np.zeros((ny, cfg.z_units), dtype=bool)
    eta_z = np.full(cfg.z_units, np.nan, dtype=float)

    for z in range(cfg.z_units):
        if z_is_pier[z]:
            p_yz[:, z] = p_p[:, x_center]
            f_yz[:, z] = f_p[:, x_center]
            s_yz[:, z] = s_p[:, x_center]
            eta_z[z] = eta_p[x_center]
        else:
            p_yz[:, z] = p_o[:, x_center]
            f_yz[:, z] = f_o[:, x_center]
            s_yz[:, z] = s_o[:, x_center]
            eta_z[z] = eta_o[x_center]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)

    # Left: profile view (unit-width/open slice).
    p0 = np.full_like(p_o, np.nan, dtype=float)
    p0[f_o] = p_o[f_o]
    im0 = ax0.imshow(p0, cmap="viridis", interpolation="nearest", origin="upper")
    ax0.imshow(np.where(s_o, 1.0, np.nan), cmap="Greys", interpolation="nearest", origin="upper", alpha=0.7)

    valid_eta0 = np.isfinite(eta_o)
    if np.any(valid_eta0):
        ax0.plot(np.arange(nx)[valid_eta0], eta_o[valid_eta0], color="cyan", linewidth=2.0)

    stride = max(1, int(cfg.velocity_quiver_stride))
    qy, qx = np.mgrid[0:ny:stride, 0:nx:stride]
    uq = np.where(f_o[::stride, ::stride], u_o[::stride, ::stride], np.nan)
    vq = np.where(f_o[::stride, ::stride], v_o[::stride, ::stride], np.nan)
    ax0.quiver(qx, qy, uq, vq, color="white", angles="xy", scale_units="xy", scale=2.8, width=0.003, alpha=0.8)
    ax0.set_title("Profile View (Unit-Width Opening Slice)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")

    # Right: centered bridge cross section (y-z).
    p1 = np.full_like(p_yz, np.nan, dtype=float)
    p1[f_yz] = p_yz[f_yz]
    im1 = ax1.imshow(p1, cmap="viridis", interpolation="nearest", origin="upper", aspect="auto")
    ax1.imshow(np.where(s_yz, 1.0, np.nan), cmap="Greys", interpolation="nearest", origin="upper", alpha=0.75, aspect="auto")

    valid_eta1 = np.isfinite(eta_z)
    if np.any(valid_eta1):
        ax1.plot(np.arange(cfg.z_units)[valid_eta1], eta_z[valid_eta1], color="cyan", linewidth=2.0)
    ax1.set_title(f"Cross Section at Bridge Center x={x_center}")
    ax1.set_xlabel("z")
    ax1.set_ylabel("y")

    mean_vel = (
        np.mean(open_data["mean_velocity"][step]) * np.mean(~z_is_pier)
        + np.mean(pier_data["mean_velocity"][step]) * np.mean(z_is_pier)
    )
    mean_fr = (
        np.mean(open_data["mean_froude"][step]) * np.mean(~z_is_pier)
        + np.mean(pier_data["mean_froude"][step]) * np.mean(z_is_pier)
    )
    mean_p = (
        np.nanmean(p0) * np.mean(~z_is_pier)
        + np.nanmean(p1) * np.mean(z_is_pier)
    )

    txt = (
        f"step: {step}   t: {step * cfg.dt_s:7.2f} s\n"
        f"mean velocity: {mean_vel:8.4f} ft/s\n"
        f"mean Froude:   {mean_fr:8.4f}\n"
        f"mean pressure: {mean_p:8.4f}"
    )
    fig.text(0.01, 0.98, txt, va="top", ha="left", fontsize=10, color="white", bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 6})

    cbar = fig.colorbar(im1, ax=[ax0, ax1], fraction=0.02, pad=0.02)
    cbar.set_label("Gauge pressure (toy units)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return frame


def generate_extruded_video(cfg: ToyConfig, output_path: Optional[str] = None) -> Path:
    z_is_pier = _z_is_pier(cfg)
    out_open = run_toy_simulation_with_pier(cfg, include_pier=False, capture_frames=False)
    out_pier = run_toy_simulation_with_pier(cfg, include_pier=True, capture_frames=False)

    frames: List[np.ndarray] = []
    for n in range(cfg.steps):
        frames.append(_render_extruded_frame(out_open, out_pier, z_is_pier, n, cfg))

    target = Path(output_path or cfg.video_output)
    return _write_video(frames, target, fps=cfg.video_fps)


if __name__ == "__main__":
    cfg = ToyConfig()
    z_is_pier = _z_is_pier(cfg)
    out_open = run_toy_simulation_with_pier(cfg, include_pier=False, capture_frames=False)
    out_pier = run_toy_simulation_with_pier(cfg, include_pier=True, capture_frames=False)
    video_path = generate_extruded_video(cfg)

    print("Toy stacked-bridge extruded prototype complete")
    print(f"z-units: {cfg.z_units}, pier width: {cfg.pier_width_units}, pier bands: {_pier_bands(cfg)}")
    print(f"pier slices: {int(np.sum(z_is_pier))}, opening slices: {int(np.sum(~z_is_pier))}")
    print(f"open-slice mean velocity [ft/s]: {out_open['mean_velocity'][-1]:.6f}")
    print(f"pier-slice mean velocity [ft/s]: {out_pier['mean_velocity'][-1]:.6f}")
    print(f"inlet velocity start/end [ft/s]: {out_open['inlet_velocity'][0]:.3f} -> {out_open['inlet_velocity'][-1]:.3f}")
    eta0 = np.nanmean(out_open["surface_row_frames"][0])
    eta1 = np.nanmean(out_open["surface_row_frames"][-1])
    print(f"open-slice free-surface row start/end: {eta0:.3f} -> {eta1:.3f}")
    print(f"video output: {video_path}")