#!/usr/bin/env python3
"""Tests for section auto-reversal behavior in the 1D solver."""

import hydra_1d as bw


def _build_reversed_order_model() -> bw.ModelInput:
    # Intentionally ordered upstream -> downstream to exercise auto-reversal.
    sections = [
        bw.CrossSection(
            river_station="US_Section",
            geometry=[(0.0, 102.0), (5.0, 101.5), (10.0, 101.0)],
            left_bank_station=0.0,
            right_bank_station=10.0,
            n_lob=0.035,
            n_ch=0.035,
            n_rob=0.035,
            contraction_coeff=0.1,
            expansion_coeff=0.3,
            L_lob_to_next=500.0,
            L_ch_to_next=500.0,
            L_rob_to_next=500.0,
        ),
        bw.CrossSection(
            river_station="Middle_Section",
            geometry=[(0.0, 101.5), (5.0, 101.0), (10.0, 100.5)],
            left_bank_station=0.0,
            right_bank_station=10.0,
            n_lob=0.035,
            n_ch=0.035,
            n_rob=0.035,
            contraction_coeff=0.1,
            expansion_coeff=0.3,
            L_lob_to_next=500.0,
            L_ch_to_next=500.0,
            L_rob_to_next=500.0,
        ),
        bw.CrossSection(
            river_station="DS_Section",
            geometry=[(0.0, 100.0), (5.0, 99.8), (10.0, 99.5)],
            left_bank_station=0.0,
            right_bank_station=10.0,
            n_lob=0.035,
            n_ch=0.035,
            n_rob=0.035,
            contraction_coeff=0.1,
            expansion_coeff=0.3,
        ),
    ]
    return bw.ModelInput(
        flow_cfs=500.0,
        flow_change=None,
        boundary_condition="known_wse",
        boundary_value=101.0,
        sections=sections,
    )


def test_solver_handles_reversed_section_order():
    model = _build_reversed_order_model()
    results = bw.run_hydra_1d(model, solver="py")

    assert len(results) == 3
    wse_values = [st.wse for st in results]

    # With downstream known WSE, solved profile should increase upstream.
    assert wse_values[0] <= wse_values[1] <= wse_values[2]
    assert abs(wse_values[0] - 101.0) < 0.1
