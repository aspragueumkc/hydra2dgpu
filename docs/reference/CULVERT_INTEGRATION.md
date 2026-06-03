#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CULVERT INTEGRATION SUMMARY
===========================

This document describes the integration of culvert_routine.py (FHWA HEC-5 inlet control
equations) into the backwater2.py water surface profile solver.

OVERVIEW
--------
The culvert integration allows individual cross-sections in the backwater model to include
a culvert that controls flow based on the FHWA HEC-5 inlet control equations. When a culvert
is present:

1. The downstream water surface elevation acts as the headwater for the culvert
2. The culvert solver (from culvert_routine.py) computes the maximum flow that can pass
   through the culvert at that headwater depth
3. If the actual flow exceeds the culvert capacity, the flow is reduced to the culvert limit
4. The standard-step solver continues with the reduced flow to compute upstream water levels

CROSS-SECTION PROPERTIES
------------------------
The CrossSection dataclass has been extended with the following culvert properties:

  culvert_code : int
    FHWA culvert type code (1-57, or 0 for no culvert)
    See culvert_routine.py for the complete list of codes

  culvert_shape : str
    Culvert shape: 'circular' or 'rect', or None for no culvert

  culvert_diameter : float
    Diameter for circular culverts (feet)

  culvert_width : float
    Width for rectangular culverts (feet)

  culvert_height : float
    Height for rectangular culverts (feet)

  culvert_slope : float
    Culvert invert slope (ft/ft). Default 0.0 means no slope correction

  has_culvert() : method
    Returns True if culvert_code > 0 and culvert_shape is defined

EXAMPLE USAGE
-------------

1. Create a CrossSection with a culvert:

    xs = CrossSection(
        river_station='100',
        geometry=[...],  # cross-section geometry points
        left_bank_station=10.0,
        right_bank_station=70.0,
        n_lob=0.04, n_ch=0.035, n_rob=0.04,
        
        # Culvert properties
        culvert_code=1,              # Square edge concrete
        culvert_shape='circular',    # Type of culvert
        culvert_diameter=3.0,        # 3 ft diameter
        culvert_slope=0.001          # 0.1% slope
    )

2. Use the culvert in a model:

    model = ModelInput(
        flow_cfs=250.0,
        boundary_condition='known_wse',
        boundary_value=105.5,
        sections=[xs_downstream, xs_with_culvert, xs_upstream]
    )
    
    results = run_backwater(model)

3. The culvert properties are automatically saved/loaded with GeoPackage:

    save_to_geopackage('model.gpkg', model)  # saves culvert columns
    model = load_from_geopackage('model.gpkg')  # loads culvert properties

CULVERT CODES (FHWA HEC-5)
--------------------------
The culvert_code parameter references the FHWA HEC-5 culvert entrance type table:

  Circular Concrete:
    1 = Square edge w/headwall
    2 = Groove end w/headwall
    3 = Groove end projecting
  
  Circular Corrugated Metal Pipe:
    4 = Headwall
    5 = Mitered to slope
    6 = Projecting
  
  Rectangular Box with Flared Wingwalls:
    9 = 30-75 deg wingwall flares
    10 = 90 or 15 deg wingwall flares
    11 = 0 deg wingwall flares (straight sides)
  
  ... (see culvert_routine.py for codes 1-57)

FLOW CONTROL BEHAVIOR
---------------------
When the solver encounters a reach with a culvert at the downstream section:

1. Compute the maximum flow allowed by inlet control at the current headwater depth
   using FHWA equations
2. If computed flow < requested flow, the culvert is restricting
3. Apply the culvert-limited flow to the reach
4. Standard-step solver uses this reduced flow to compute upstream water elevation

Example output from test_culvert_integration.py:

  Section 0 (RS 0): No culvert
      - WSE: 92.000 ft, Flow: 150.00 cfs
  
  Section 1 (RS 1): Circular culvert (code 1, D=2.5 ft)
      - WSE: 92.695 ft, Flow: 150.00 cfs
      - Culvert can pass this flow at this headwater
  
  Section 2 (RS 2): No culvert
      - WSE: 93.695 ft, Flow: 150.00 cfs

GEOPACKAGE PERSISTENCE
----------------------
Culvert properties are stored as columns in the 'cross_sections' layer:

  culvert_code (integer)
  culvert_shape (text)
  culvert_diameter (real)
  culvert_width (real)
  culvert_height (real)
  culvert_slope (real)

Users can edit these properties in QGIS:
- Open the backwater GeoPackage in QGIS
- Toggle edit mode on the cross_sections layer
- Edit the culvert_code and shape columns to specify a culvert
- Set culvert_code=0 to disable a culvert
- Save changes back to the GeoPackage

TECHNICAL DETAILS
-----------------
The apply_culvert_control() function (backwater2.py, line 588):

  1. Checks if downstream section has a culvert (has_culvert() returns True)
  2. Constructs CircularXsect or RectangularXsect from culvert properties
  3. Calls solve_headwater_depth_for_Q() to find required h for target flow
  4. Compares actual headwater depth to required depth
  5. Returns (culvert_controlled_Q, is_restricting) tuple

The culvert check is applied in run_backwater() (line 1620):
  
  if xs_dn.has_culvert():
      Q_dn_controlled, is_restricting = apply_culvert_control(...)
      if is_restricting:
          Q_dn = Q_dn_controlled  # Use reduced flow in energy balance

LIMITATIONS & NOTES
------------------
1. Only inlet control is implemented (outlet control not included)
2. Culvert is treated as a point control at the downstream section
3. No support for compound culverts (multiple barrels in parallel)
4. Culvert slope defaults to 0.0 if not specified
5. Headwater elevation is the water surface at the culvert section
6. Tailwater is not currently used (inlet control dominates at low tailwater)

FUTURE ENHANCEMENTS
-------------------
✅ 1. Add outlet control calculation
   - Implemented in apply_culvert_control() function
   - Compares inlet and outlet control, returns minimum (most restrictive)
   - Outlet control reduces flow when tailwater submergence is high
   
✅ 3. Add tailwater submergence effects
   - Outlet elevation estimated as: z_invert + (yFull * slope)
   - If tailwater above outlet invert, apply submergence factor
   - Fully submerged outlet: Q_outlet = 0.7 * Q_target (conservative)
   - Partially submerged: Q_outlet reduced proportionally to submergence ratio
   
✅ 5. Add Qt widget fields for culvert editing
   - Added 6 culvert property fields to the cross-section form:
     - culvert_code (combobox/input: 1-57 FHWA codes)
     - culvert_shape (text input: 'circular' or 'rect')
     - culvert_diameter (numeric)
     - culvert_width (numeric)
     - culvert_height (numeric)
     - culvert_slope (numeric, ft/ft)
   - Properties are editable in the Qt widget form
   - Changes applied with "Apply Section Changes" button
   - Automatically saved to GeoPackage layer
   
✅ 6. Visualize culvert control zones in plot output
   - Enhanced _plot_results() function to show culvert sections
   - Culvert sections highlighted with red shaded zone (alpha=0.2)
   - Culvert label displayed in center of zone
   - Legend shows culvert type (shape, code, dimensions)
   - Grid added for better readability
   - Improved line weights and styling

IMPLEMENTATION DETAILS
----------------------
The apply_culvert_control() function now returns 3 values:

  (Q_controlled, culvert_is_restricting, control_type)
  
  control_type: 'inlet', 'outlet', or 'none'

When Q_inlet < Q_outlet: Inlet control is dominant (headwater too low)
When Q_outlet < Q_inlet: Outlet control is dominant (tailwater too high)
When both equal Q_target: Culvert does not restrict flow

TEST
----
Run test_culvert_integration.py to verify the culvert solver works correctly:

  python test_culvert_integration.py

This creates a 3-section model with a culvert at the middle section, saves it
to a GeoPackage, reloads it, and solves the water surface profile.

FILES MODIFIED
--------------
- backwater2.py: Core solver integration
  - Added culvert import (line 81)
  - Extended CrossSection dataclass (line 318-348)
  - Added apply_culvert_control() function (line 588)
  - Updated load_from_geopackage() (line 1438)
  - Updated save_to_geopackage() (line 1500)
  - Integrated culvert control in run_backwater() (line 1620)

- test_culvert_integration.py: New test file
  - Comprehensive integration test with GeoPackage I/O
"""

# This is a reference document; no executable code below
