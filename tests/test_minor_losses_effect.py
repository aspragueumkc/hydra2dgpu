import csv
import os
from pathlib import Path

import pytest

import hydra_1d as bw


ROOT = Path.cwd() / "hec_ras_project"
SECTIONS_META = ROOT / "sections_metadata.csv"
HDF_FILE = ROOT / "test3.p01.hdf"


def _build_sections(rows, contraction=None, expansion=None):
    sections = []
    for r in rows:
        geom = []
        with open(ROOT / r["file"], "r", newline="") as g:
            rdr2 = csv.DictReader(g)
            for row in rdr2:
                try:
                    off = float(row["Offset"])
                    el = float(row["Elevation"])
                except Exception:
                    continue
                geom.append((off, el))

        xs = bw.CrossSection(
            river_station=r["river_station"],
            geometry=geom,
            left_bank_station=float(r["left_bank_station"]),
            right_bank_station=float(r["right_bank_station"]),
            n_lob=float(r["n_lob"]),
            n_ch=float(r["n_ch"]),
            n_rob=float(r["n_rob"]),
            L_lob_to_next=float(r["L_lob_to_next"]),
            L_ch_to_next=float(r["L_ch_to_next"]),
            L_rob_to_next=float(r["L_rob_to_next"]),
        )
        if contraction is not None:
            xs.contraction_coeff = contraction
        if expansion is not None:
            xs.expansion_coeff = expansion
        sections.append(xs)
    return sections


@pytest.mark.skipif(
    not SECTIONS_META.exists() or not HDF_FILE.exists(),
    reason="Optional HEC-RAS comparison fixtures are not present in this workspace",
)
def test_minor_losses_reduction_does_not_raise_wse():
    h5py = pytest.importorskip("h5py")

    rows = []
    with open(SECTIONS_META, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)

    with h5py.File(HDF_FILE, "r") as f:
        ws = f[
            "Results/Steady/Output/Output Blocks/Base Output/Steady Profiles/Cross Sections/Water Surface"
        ][0]
        attrs = f["Geometry/Cross Sections/Attributes"][()]
        rs_order = [a[2].decode("utf-8") for a in attrs]

    q_val = 500.0
    slope = 0.003

    sections_a = _build_sections(rows)
    model_a = bw.ModelInput(
        flow_cfs=q_val,
        flow_change=None,
        boundary_condition="normal_depth",
        boundary_value=slope,
        sections=sections_a,
    )
    results_a = bw.run_hydra_1d(model_a)
    comp_a = {sxn.river_station: st.wse for sxn, st in zip(model_a.sections, results_a)}

    sections_b = _build_sections(rows, contraction=0.0, expansion=0.0)
    model_b = bw.ModelInput(
        flow_cfs=q_val,
        flow_change=None,
        boundary_condition="normal_depth",
        boundary_value=slope,
        sections=sections_b,
    )
    results_b = bw.run_hydra_1d(model_b)
    comp_b = {sxn.river_station: st.wse for sxn, st in zip(model_b.sections, results_b)}

    # Sanity-check fixture alignment while preserving historical comparison flow.
    assert len(rs_order) == len(ws)

    any_higher = False
    for rs in rs_order:
        a = comp_a.get(rs)
        b = comp_b.get(rs)
        if a is None or b is None:
            continue
        if (b - a) > 0:
            any_higher = True
            break

    assert not any_higher
