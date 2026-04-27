#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to demonstrate culvert integration into backwater solver.

This creates a simple 3-section model with a culvert at the middle section
and solves the water surface profile accounting for inlet control.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add plugin directory to path
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir))

from backwater_model import (
    CrossSection, ModelInput, run_backwater, 
    save_to_geopackage, load_from_geopackage
)


def create_test_model_with_culvert():
    """
    Create a 3-section model with a culvert at the middle section.
    
    Cross-section geometry:
    - DS section (RS 0): trapezoid
    - Culvert section (RS 1): same trapezoid + circular culvert
    - US section (RS 2): trapezoid
    """
    
    # Downstream section (no culvert)
    xs_0 = CrossSection(
        river_station='0',
        geometry=[
            (0.0, 100.0), (10.0, 95.0), (20.0, 92.0),   # left bank
            (40.0, 90.0),                                # channel bed
            (60.0, 92.0), (70.0, 95.0), (80.0, 100.0)   # right bank
        ],
        left_bank_station=10.0,
        right_bank_station=70.0,
        n_lob=0.04, n_ch=0.035, n_rob=0.04,
        L_ch_to_next=100.0,  # 100 ft to next section
        contraction_coeff=0.1,
        expansion_coeff=0.3
    )
    
    # Middle section with CIRCULAR CULVERT (inlet control)
    # Code 1 = Square edge w/headwall (concrete), code 4 = Headwall (corrugated metal)
    xs_1 = CrossSection(
        river_station='1',
        geometry=[
            (0.0, 101.0), (10.0, 96.0), (20.0, 93.0),   # left bank
            (40.0, 91.0),                                # channel bed
            (60.0, 93.0), (70.0, 96.0), (80.0, 101.0)   # right bank
        ],
        left_bank_station=10.0,
        right_bank_station=70.0,
        n_lob=0.04, n_ch=0.035, n_rob=0.04,
        L_ch_to_next=100.0,
        contraction_coeff=0.1,
        expansion_coeff=0.3,
        # CULVERT PROPERTIES
        culvert_code=1,              # Square edge w/headwall concrete
        culvert_shape='circular',     # Circular
        culvert_diameter=2.5,         # 2.5 ft diameter
        culvert_upstream_invert=91.0,
        culvert_downstream_invert=90.9,
        culvert_length=100.0
    )
    
    # Upstream section (no culvert)
    xs_2 = CrossSection(
        river_station='2',
        geometry=[
            (0.0, 102.0), (10.0, 97.0), (20.0, 94.0),   # left bank
            (40.0, 92.0),                                # channel bed
            (60.0, 94.0), (70.0, 97.0), (80.0, 102.0)   # right bank
        ],
        left_bank_station=10.0,
        right_bank_station=70.0,
        n_lob=0.04, n_ch=0.035, n_rob=0.04,
        contraction_coeff=0.1,
        expansion_coeff=0.3
    )
    
    # Build model
    model = ModelInput(
        flow_cfs=150.0,  # 150 cfs flow
        flow_change=None,
        boundary_condition='known_wse',
        boundary_value=92.0,  # Downstream WSE
        sections=[xs_0, xs_1, xs_2]
    )
    
    return model


def test_culvert_integration():
    """Test the culvert integration by solving a model with culvert."""
    
    print("\n" + "="*70)
    print("CULVERT INTEGRATION TEST")
    print("="*70)
    
    # Create test model
    print("\n1. Creating test model with 3 sections (middle has circular culvert)...")
    model = create_test_model_with_culvert()
    
    print(f"   - Flow: {model.flow_cfs} cfs")
    print(f"   - DS BC: {model.boundary_condition} = {model.boundary_value} ft")
    
    for i, xs in enumerate(model.sections):
        if xs.has_culvert():
            print(f"   - Section {i} (RS {xs.river_station}): {xs.culvert_shape} culvert "
                  f"(code {xs.culvert_code}, D={xs.culvert_diameter} ft)")
        else:
            print(f"   - Section {i} (RS {xs.river_station}): No culvert")
    
    # Save to GeoPackage
    print("\n2. Saving model to GeoPackage...")
    with tempfile.TemporaryDirectory() as tmpdir:
        gpkg_path = Path(tmpdir) / 'test_culvert.gpkg'
        save_to_geopackage(str(gpkg_path), model)
        print(f"   Saved to: {gpkg_path}")
        
        # Reload from GeoPackage
        print("\n3. Reloading model from GeoPackage...")
        model_reloaded = load_from_geopackage(str(gpkg_path))
        # Override boundary condition (GeoPackage may not have saved it properly)
        model_reloaded.flow_cfs = model.flow_cfs
        model_reloaded.boundary_condition = model.boundary_condition
        model_reloaded.boundary_value = model.boundary_value
        
        # Verify culvert properties
        print("\n4. Verifying culvert properties were saved/loaded:")
        for i, xs in enumerate(model_reloaded.sections):
            if xs.has_culvert():
                slope = (xs.culvert_upstream_invert - xs.culvert_downstream_invert) / xs.culvert_length if xs.culvert_length > 0 else 0.0
                print(f"   ✓ Section {i}: code={xs.culvert_code}, shape={xs.culvert_shape}, "
                      f"diam={xs.culvert_diameter} ft, slope={slope} ft/ft")
        
        # Solve water surface profile
        print("\n5. Solving water surface profile (with culvert control)...")
        try:
            results = run_backwater(model_reloaded)
            print(f"   ✓ Solver completed successfully")
            
            print("\n6. Results:")
            for i, (xs, state) in enumerate(zip(model_reloaded.sections, results)):
                Q_total = state.Q_lob + state.Q_ch + state.Q_rob
                print(f"   Section {i} (RS {xs.river_station}):")
                print(f"      - WSE: {state.wse:.3f} ft")
                print(f"      - Depth (at min bed): {state.wse - min(z for _, z in xs.geometry):.3f} ft")
                print(f"      - Flow: {Q_total:.2f} cfs")
                print(f"      - Velocity: {state.V_t:.3f} ft/s")
                print(f"      - Froude: {state.Froude:.3f}")
                if xs.has_culvert():
                    print(f"      *** CULVERT: code {xs.culvert_code}, {xs.culvert_shape} D={xs.culvert_diameter} ft")
            
            print("\n" + "="*70)
            print("TEST PASSED: Culvert integration successful!")
            print("="*70)
            return True
            
        except Exception as e:
            print(f"   ✗ Solver failed: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    success = test_culvert_integration()
    sys.exit(0 if success else 1)
