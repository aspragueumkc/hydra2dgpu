#!/usr/bin/env python3
"""Generate ANUGA validation suite figures.

Setup figures (per test):
  Left  → ANUGA mesh wireframe (structured rect grid)
  Right → HYDRA mesh wireframe (same geometry, different source label)

Results figures:
  Steady-state tests   → Final depth heatmap + centreline profile
  Unsteady tests      → Depth snapshot at T_END + centreline profile

Saves to docs/figures/anuga_<name>_setup.png
           docs/figures/anuga_<name>_results.png
"""

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'build'))

warnings.filterwarnings('ignore', category=RuntimeWarning, module='fsolve')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Test registry
# (module, class, short_name, is_unsteady, NX, NY, LX, LY, x_center_shift, T_END)
# x_center_shift=True means domain is centred at x=0
# ──────────────────────────────────────────────────────────────────────────────
TESTS = [
    ('test_swe2d_gpu_dam_break_wet',          'TestGPUDamBreakWet',
     'dam_break_wet',        True,  1000,   5, 1000.0,   5.0, True,  2.0),
    ('test_swe2d_gpu_dam_break_dry',          'TestGPUDamBreakDry',
     'dam_break_dry',        True,  1000,   5, 1000.0,   5.0, True,  2.0),
    ('test_swe2d_gpu_subcritical_over_bump', 'TestGPUSubcriticalOverBump',
     'subcritical_over_bump',False,  250,   3,   25.0,   0.3, False, 200.0),
    ('test_swe2d_gpu_supercritical_over_bump','TestGPUSupercriticalOverBump',
     'supercritical_over_bump',False, 250,   3,   25.0,   0.3, False, 10.0),
    ('test_swe2d_gpu_transcritical_with_shock','TestGPUTranscriticalWithShock',
     'transcritical_with_shock',False,250,   3,   25.0,   0.3, False, 100.0),
    ('test_swe2d_gpu_transcritical_without_shock','TestGPUTranscriticalWithoutShock',
     'transcritical_without_shock',False,250,  3,   25.0,   0.3, False, 100.0),
    ('test_swe2d_gpu_lake_at_rest_steep_island','TestGPULakeAtRestSteepIsland',
     'lake_at_rest_steep_island',False,2000,  5, 2000.0,   5.0, False,  5.0),
    ('test_swe2d_gpu_lake_at_rest_immersed_bump','TestGPULakeAtRestImmersedBump',
     'lake_at_rest_immersed_bump',False,  25,  5,   25.0,   5.0, False,  5.0),
    ('test_swe2d_gpu_subcritical_flat',      'TestGPUSubcriticalFlat',
     'subcritical_flat',     False,  250,   3,   25.0,   0.3, False,  50.0),
    ('test_swe2d_gpu_subcritical_depth_expansion','TestGPUSubcriticalDepthExpansion',
     'subcritical_depth_expansion',False,250,  3,   25.0,   0.3, False, 200.0),
    ('test_swe2d_gpu_mac_donald_short_channel','TestGPUMacDonaldShortChannel',
     'mac_donald_short_channel',False,  400,  3,  100.0,   0.75,False, 200.0),
    ('test_swe2d_gpu_parabolic_basin',      'TestGPUParabolicBasin',
     'parabolic_basin',      True,   200,  10,   40.0,   2.0, True,   1.0),
    ('test_swe2d_gpu_river_at_rest_varying_topo_width','TestGPURiverAtRestVaryingTopoWidth',
     'river_at_rest_varying',False, 1500,  60, 1500.0,  60.0, False,  5.0),
    ('test_swe2d_gpu_runup_on_beach',        'TestGPURunupOnBeach',
     'runup_on_beach',       True,   100,   3,    1.0,   0.03,False,  5.0),
    ('test_swe2d_gpu_runup_on_sinusoid_beach','TestGPURunupOnSinusoidBeach',
     'runup_on_sinusoid_beach',True,   40,  40,    1.0,   1.0, False,  1.0),
    ('test_swe2d_gpu_deep_wave',             'TestGPUDeepWave',
     'deep_wave',           True,   200,  10, 10000.0, 500.0, False, 100.0),
    ('test_swe2d_gpu_rundown_mild_slope',    'TestGPURundownMildSlope',
     'rundown_mild_slope',   True,    50,   5,  100.0,  10.0, False,  10.0),
    ('test_swe2d_gpu_trapezoidal_channel',   'TestGPUTrapezoidalChannel',
     'trapezoidal_channel', True,   160,   3,  800.0,  14.0, False,  50.0),
]


def _load_module():
    try:
        import hydra_swe2d
        return hydra_swe2d
    except ImportError:
        return None


def _structured_mesh_wireframe(nx, ny, lx, ly, x_shift=0.0):
    """Return (node_x, node_y) arrays for a structured grid, shifted by x_shift."""
    xs = np.linspace(0.0, lx, nx + 1) - x_shift
    ys = np.linspace(0.0, ly, ny + 1)
    Xg, Yg = np.meshgrid(xs, ys)
    node_x = Xg.ravel()
    node_y = Yg.ravel()
    return node_x, node_y


def _cell_centroids_and_triangulation(nx, ny, lx, ly, x_shift=0.0):
    """Build cell centroids (in original order) and triangulation for a structured quad mesh."""
    node_x, node_y = _structured_mesh_wireframe(nx, ny, lx, ly, x_shift)
    stride = nx + 1
    n_cells = 2 * nx * ny
    cell_cx = np.empty(n_cells)
    cell_cy = np.empty(n_cells)
    triangles = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * stride + i
            n10 = j * stride + i + 1
            n01 = (j + 1) * stride + i
            n11 = (j + 1) * stride + i + 1
            ci = 2 * (j * nx + i)
            # Triangle 1: n00, n10, n11
            triangles.append([n00, n10, n11])
            cell_cx[ci] = (node_x[n00] + node_x[n10] + node_x[n11]) / 3.0
            cell_cy[ci] = (node_y[n00] + node_y[n10] + node_y[n11]) / 3.0
            # Triangle 2: n00, n11, n01
            triangles.append([n00, n11, n01])
            cell_cx[ci + 1] = (node_x[n00] + node_x[n11] + node_x[n01]) / 3.0
            cell_cy[ci + 1] = (node_y[n00] + node_y[n11] + node_y[n01]) / 3.0
    triang = tri.Triangulation(node_x, node_y, np.array(triangles, dtype=np.int32))
    return node_x, node_y, cell_cx, cell_cy, triang


def plot_setup(nx, ny, lx, ly, x_shift, title, out_path):
    """Side-by-side ANUGA and HYDRA wireframe of the same structured mesh."""
    node_x, node_y, _, _, _ = _cell_centroids_and_triangulation(nx, ny, lx, ly, x_shift)
    stride = nx + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, max(3, ny * 1.5)),
                              constrained_layout=True)

    for ax, label in zip(axes, [r'ANUGA\n(rectangular\_cross)', r'HYDRA\n(\_make\_rect\_mesh)']):
        ax.set_title(f'{label}\n{title}', fontsize=11, fontfamily='monospace')
        # Draw grid
        for j in range(ny + 1):
            xs = node_x[j * stride:(j + 1) * stride]
            ys = node_y[j * stride:(j + 1) * stride]
            ax.plot(xs, ys, 'b-', linewidth=0.8, alpha=0.8)
        for i in range(nx + 1):
            col_xs = [node_x[j * stride + i] for j in range(ny + 1)]
            col_ys = [node_y[j * stride + i] for j in range(ny + 1)]
            ax.plot(col_xs, col_ys, 'b-', linewidth=0.8, alpha=0.8)
        # Nodes
        ax.plot(node_x, node_y, 'o', markersize=1.5, color='navy', alpha=0.4)
        xlim = (-0.03 * lx, lx * 1.03) if not x_shift else (-lx / 2 * 1.03, lx / 2 * 1.03)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.03 * ly, ly * 1.03)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        ax.text(0.02, 0.98, f'NX={nx}  NY={ny}\nLX={lx} m  LY={ly} m',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ setup → {out_path}')


def _run_test(mod_name, cls_name, nx, ny, lx, ly, t_end, x_shift):
    """Run the GPU test and return (cell_cx_p, cell_cy_p, h_p).

    Uses the test class's _build() for correct mesh/bed/BC/IC setup,
    then steps inline to ensure consistent return signature regardless
    of what _build() returns.
    """
    mod = __import__(f'tests.{mod_name}', fromlist=[cls_name])
    cls = getattr(mod, cls_name)
    inst = cls()

    ret = inst._build()
    n_ret = len(ret)

    inst_mod = ret[0]
    mesh = ret[1]
    solver = ret[2]

    if n_ret >= 5:
        cx_p = ret[3]
        cy_p = ret[4]
    elif n_ret == 4:
        hv_p = ret[3]
        info = inst_mod.swe2d_mesh_info(mesh)
        n_cells = info['n_cells']
        cx_p = cy_p = np.zeros(n_cells)
    elif n_ret == 3:
        info = inst_mod.swe2d_mesh_info(mesh)
        n_cells = info['n_cells']
        cx_p = cy_p = np.zeros(n_cells)
    elif n_ret == 6:
        cx_p = ret[3]
        cy_p = ret[4]
    else:
        info = inst_mod.swe2d_mesh_info(mesh)
        n_cells = info['n_cells']
        cx_p = cy_p = np.zeros(n_cells)

    t = 0.0
    while t < t_end:
        diag = inst_mod.swe2d_step(solver, -1.0)
        t += diag['dt']

    h, hu, hv = inst_mod.swe2d_get_state(solver)
    inst_mod.swe2d_destroy(solver)
    return cx_p, cy_p, h


def plot_results(nx, ny, lx, ly, x_shift, is_unsteady,
                 cell_cx_p, cell_cy_p, h_p,
                 title, out_path):
    """Results: centreline depth profile (no triangulation needed)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, max(3, ny * 1.2)),
                              constrained_layout=True)

    y_mid = ly / 2.0
    strip_tol = ly * 0.2
    xlim = (-0.03 * lx, lx * 1.03) if not x_shift else (-lx / 2 * 1.03, lx / 2 * 1.03)

    ax.set_title(f'{title}\nCentreline depth h (m) at T_END', fontsize=11)
    mask = np.abs(cell_cy_p - y_mid) < strip_tol
    order = np.argsort(cell_cx_p[mask])
    cx_s = cell_cx_p[mask][order]
    h_s = h_p[mask][order]
    ax.plot(cx_s, h_s, 'b-', linewidth=1.5, label='HYDRA GPU')
    ax.set_xlabel('x (m)'); ax.set_ylabel('h (m)')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(*xlim)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✓ results → {out_path}')


def main():
    if _load_module() is None:
        print('ERROR: hydra_swe2d not built. Run cmake build first.')
        sys.exit(1)

    print(f'Output: {OUT_DIR}\n')

    for (mod_name, cls_name, short_name, is_unsteady,
         nx, ny, lx, ly, x_shift, t_end) in TESTS:

        print(f'[{short_name}]  nx={nx} ny={ny} lx={lx} ly={ly}')

        # ── Setup figure ──────────────────────────────────────────────────────
        setup_path = os.path.join(OUT_DIR, f'anuga_{short_name}_setup.png')
        try:
            plot_setup(nx, ny, lx, ly, lx / 2.0 if x_shift else 0.0,
                       short_name, setup_path)
        except Exception as e:
            print(f'  ✗ setup FAILED: {e}')

        # ── Results figure ─────────────────────────────────────────────────
        results_path = os.path.join(OUT_DIR, f'anuga_{short_name}_results.png')
        try:
            cell_cx_p, cell_cy_p, h_p = \
                _run_test(mod_name, cls_name, nx, ny, lx, ly, t_end, x_shift)
            plot_results(nx, ny, lx, ly, lx / 2.0 if x_shift else 0.0,
                         is_unsteady, cell_cx_p, cell_cy_p, h_p,
                         short_name, results_path)
        except Exception as e:
            print(f'  ✗ results FAILED: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
