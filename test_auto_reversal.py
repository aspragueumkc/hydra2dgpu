#!/usr/bin/env python3
"""
Comprehensive test demonstrating the section auto-reversal fix.
This mimics what a user would experience in the Qt frontend.
"""

import json
import tempfile
import os
import backwater2 as bw

# Create a realistic 3-section model with sections in REVERSED order (upstream to downstream)
# This is a common mistake when users manually add sections
sections_data = [
    {
        "river_station": "US_Section",
        "geometry": [[0.0, 102.0], [5.0, 101.5], [10.0, 101.0]],
        "left_bank_station": 0.0,
        "right_bank_station": 10.0,
        "n_lob": 0.035,
        "n_ch": 0.035,
        "n_rob": 0.035,
        "contraction_coeff": 0.1,
        "expansion_coeff": 0.3,
        "L_lob_to_next": 500.0,
        "L_ch_to_next": 500.0,
        "L_rob_to_next": 500.0
    },
    {
        "river_station": "Middle_Section",
        "geometry": [[0.0, 101.5], [5.0, 101.0], [10.0, 100.5]],
        "left_bank_station": 0.0,
        "right_bank_station": 10.0,
        "n_lob": 0.035,
        "n_ch": 0.035,
        "n_rob": 0.035,
        "contraction_coeff": 0.1,
        "expansion_coeff": 0.3,
        "L_lob_to_next": 500.0,
        "L_ch_to_next": 500.0,
        "L_rob_to_next": 500.0
    },
    {
        "river_station": "DS_Section",
        "geometry": [[0.0, 100.0], [5.0, 99.8], [10.0, 99.5]],
        "left_bank_station": 0.0,
        "right_bank_station": 10.0,
        "n_lob": 0.035,
        "n_ch": 0.035,
        "n_rob": 0.035,
        "contraction_coeff": 0.1,
        "expansion_coeff": 0.3
    }
]

model_data = {
    "flow_cfs": 500.0,
    "flow_change": None,
    "boundary_condition": "known_wse",
    "boundary_value": 101.0,  # Increased from 100.0 to be more reasonable for the 500 cfs flow
    "sections": sections_data
}

# Save to a temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(model_data, f)
    temp_file = f.name

try:
    print("=" * 70)
    print("BACKWATER SOLVER - SECTION AUTO-REVERSAL FIX TEST")
    print("=" * 70)
    print()
    
    # Load the model
    model = bw.load_input(temp_file)
    
    print("INPUT SECTIONS (in order provided - REVERSED!):")
    print("-" * 70)
    for i, xs in enumerate(model.sections):
        z_min = min(z for _, z in xs.geometry)
        print(f"  [{i}] {xs.river_station:20s} min_bed={z_min:.2f} ft")
    
    print()
    print("RUNNING SOLVER...")
    print("-" * 70)
    
    # Run the solver - it will auto-detect and fix the order
    results = bw.run_backwater(model, solver='py')
    
    print()
    print("RESULTS (after potential auto-reversal):")
    print("-" * 70)
    for i, (xs, st) in enumerate(zip(model.sections, results)):
        print(f"  [{i}] {xs.river_station:20s} WSE={st.wse:8.3f} ft  Depth={st.depth_at_min:6.3f} ft  V={st.V_t:6.3f} ft/s")
    
    print()
    print("ANALYSIS:")
    print("-" * 70)
    wse_values = [st.wse for st in results]
    if wse_values[0] <= wse_values[1] <= wse_values[2]:
        print("✓ CORRECT: Water surface elevations increase from DS to US")
        print("  This indicates the solver is correctly iterating upstream from the boundary condition.")
    else:
        print("✗ ERROR: Water surface elevations do not increase monotonically!")
        print("  Expected WSE to increase from downstream to upstream.")
    
    print()
    print("BOUNDARY CONDITION VERIFICATION:")
    print("-" * 70)
    if abs(wse_values[0] - 101.0) < 0.1:
        print(f"✓ CORRECT: Boundary condition (101.0 ft) applied to first section in result")
        print(f"  Measured WSE at DS = {wse_values[0]:.3f} ft")
    else:
        print(f"✗ ERROR: Boundary condition not applied correctly")
        print(f"  Expected WSE[0] ≈ 101.0, got {wse_values[0]:.3f} ft")
    
    print()
    print("=" * 70)

finally:
    # Clean up
    os.unlink(temp_file)
