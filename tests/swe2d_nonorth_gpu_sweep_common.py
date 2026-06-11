from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from tests._swe2d_test_helpers import (
    _channel_bc_edges,
    _load_module,
    _make_tri_channel_mesh,
    _manning_normal_depth,
)


def run_gpu_nonorth_vs_orth_sweep(
    *,
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    s0: float,
    n_mann: float,
    q_in: float,
    nsteps: int,
    skew_fraction_dx: float,
    artifact_tag: str,
) -> Dict[str, object]:
    mod = _load_module()
    if mod is None:
        raise RuntimeError("hydra_swe2d not built")
    if not bool(mod.swe2d_gpu_available()):
        raise RuntimeError("CUDA GPU not available")

    h_n = _manning_normal_depth(q_in, n_mann, s0)
    n_cells = 2 * int(nx) * int(ny)
    h0 = np.full(n_cells, h_n, dtype=np.float64)

    bc_n0, bc_n1, bc_tp, bc_vl = _channel_bc_edges(nx, ny, q_in, s0)

    node_x_o, node_y_o, node_z_o, cell_nodes_o = _make_tri_channel_mesh(nx, ny, lx, ly, s0, skew_amp=0.0)
    skew_amp = float(skew_fraction_dx) * (float(lx) / float(nx))
    node_x_n, node_y_n, node_z_n, cell_nodes_n = _make_tri_channel_mesh(nx, ny, lx, ly, s0, skew_amp=skew_amp)

    def _run_case(node_x, node_y, node_z, cell_nodes, temporal_order, spatial_scheme, godunov_mode):
        mesh = mod.swe2d_build_mesh(node_x, node_y, node_z, cell_nodes, bc_n0, bc_n1, bc_tp, bc_vl)
        solver = mod.swe2d_create_solver(
            mesh,
            h0.copy(),
            n_mann=float(n_mann),
            cfl=0.45,
            dt_max=0.5,
            temporal_order=int(temporal_order),
            spatial_scheme=int(spatial_scheme),
            godunov_mode=int(godunov_mode),
            use_gpu=True,
        )

        gpu_active = False
        for _ in range(int(nsteps)):
            d = mod.swe2d_step(solver, -1.0)
            gpu_active = gpu_active or bool(d.get("gpu_active", False))

        h, hu, hv = mod.swe2d_get_state(solver)
        mod.swe2d_destroy(solver)
        return h, hu, hv, gpu_active

    rows: List[Dict[str, object]] = []
    for temporal_order in (1, 2):
        for godunov_mode in (0, 1):
            for spatial_scheme in range(7):
                try:
                    h_o, hu_o, hv_o, gpu_o = _run_case(
                        node_x_o, node_y_o, node_z_o, cell_nodes_o,
                        temporal_order, spatial_scheme, godunov_mode,
                    )
                    h_n2, hu_n, hv_n, gpu_n = _run_case(
                        node_x_n, node_y_n, node_z_n, cell_nodes_n,
                        temporal_order, spatial_scheme, godunov_mode,
                    )

                    q_o = float(np.mean(hu_o))
                    q_n = float(np.mean(hu_n))
                    row = {
                        "temporal_order": temporal_order,
                        "godunov_mode": godunov_mode,
                        "spatial_scheme": spatial_scheme,
                        "gpu_active_orth": bool(gpu_o),
                        "gpu_active_nonorth": bool(gpu_n),
                        "q_orth": q_o,
                        "q_nonorth": q_n,
                        "abs_q_diff": float(abs(q_n - q_o)),
                        "rel_q_diff_pct": float(100.0 * abs(q_n - q_o) / max(abs(q_o), 1.0e-12)),
                        "rel_l2_h": float(np.linalg.norm(h_n2 - h_o) / max(np.linalg.norm(h_o), 1.0e-12)),
                        "rel_l2_hu": float(np.linalg.norm(hu_n - hu_o) / max(np.linalg.norm(hu_o), 1.0e-12)),
                        "ok": bool(
                            np.isfinite(h_o).all()
                            and np.isfinite(hu_o).all()
                            and np.isfinite(hv_o).all()
                            and np.isfinite(h_n2).all()
                            and np.isfinite(hu_n).all()
                            and np.isfinite(hv_n).all()
                        ),
                        "error": "",
                    }
                except Exception as exc:
                    row = {
                        "temporal_order": temporal_order,
                        "godunov_mode": godunov_mode,
                        "spatial_scheme": spatial_scheme,
                        "gpu_active_orth": False,
                        "gpu_active_nonorth": False,
                        "q_orth": float("nan"),
                        "q_nonorth": float("nan"),
                        "abs_q_diff": float("nan"),
                        "rel_q_diff_pct": float("nan"),
                        "rel_l2_h": float("nan"),
                        "rel_l2_hu": float("nan"),
                        "ok": False,
                        "error": str(exc),
                    }
                rows.append(row)

    art = Path("tests/artifacts")
    art.mkdir(parents=True, exist_ok=True)
    json_path = art / f"{artifact_tag}.json"
    csv_path = art / f"{artifact_tag}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    ok_rows = [r for r in rows if bool(r["ok"])]
    rel_q = np.array([float(r["rel_q_diff_pct"]) for r in ok_rows], dtype=np.float64) if ok_rows else np.array([], dtype=np.float64)

    return {
        "rows": rows,
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "ok_count": int(len(ok_rows)),
        "total_count": int(len(rows)),
        "all_gpu_active": bool(all(bool(r["gpu_active_orth"]) and bool(r["gpu_active_nonorth"]) for r in rows)),
        "min_rel_q_pct": float(np.min(rel_q)) if rel_q.size else float("nan"),
        "max_rel_q_pct": float(np.max(rel_q)) if rel_q.size else float("nan"),
        "mean_rel_q_pct": float(np.mean(rel_q)) if rel_q.size else float("nan"),
    }
