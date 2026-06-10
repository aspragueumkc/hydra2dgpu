#!/usr/bin/env python3
"""
Test script to verify workbench state persistence.

Usage:
  1. Create a NEW test QGIS project
  2. Open the 2D workbench
  3. Set some test values (e.g., nx_spin=50, reconstruction_combo=MUSCL, etc.)
  4. Close the workbench dialog
  5. SAVE the QGIS project
  6. Close and REOPEN the QGIS project
  7. Open the workbench again
  8. Run this test script in QGIS Python console

This script will read directly from QgsProject and show what's stored.
"""
import json
try:
    from qgis.core import QgsProject
    
    proj = QgsProject.instance()
    fname = proj.fileName()
    print(f"\n=== Workbench State Persistence Diagnostic ===")
    print(f"Project: {fname}")
    print(f"Project file exists: {fname and __import__('os').path.exists(fname)}")
    
    # Try to read persisted state
    result = proj.readEntry("Backwater2DWorkbench", "workbench_state_json", "")
    if isinstance(result, tuple):
        raw_json = result[0] if result else ""
    else:
        raw_json = result
    
    if not raw_json:
        print("\n❌ NO PERSISTED STATE FOUND")
        print("   - This could mean:")
        print("   - Workbench state was never saved (run before setting values)")
        print("   - Project was saved before persistence feature was added")
        print("   - writeEntry is failing silently")
    else:
        print(f"\n✓ Found persisted state ({len(raw_json)} bytes)")
        try:
            payload = json.loads(raw_json)
            widgets = payload.get("widgets", {})
            print(f"  Stored widgets: {len(widgets)}")
            print(f"\n  First 10 widgets stored:")
            for name, info in list(widgets.items())[:10]:
                val = info.get("value")
                typ = info.get("type")
                print(f"    {name:40} = {val!r:15} ({typ})")
            if len(widgets) > 10:
                print(f"    ... and {len(widgets)-10} more")
        except Exception as e:
            print(f"  ❌ Failed to parse JSON: {e}")

except ImportError as e:
    print(f"Error: QGIS imports not available: {e}")
    print("This test must be run in QGIS Python console, not standalone terminal")
