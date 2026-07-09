"""Export QML styles from a loaded model GeoPackage to QML/ folder.

Usage: In QGIS Python console:

    exec(open("tools/export_qmls.py").read())

Or from command line with a running QGIS instance.
"""
import os

# Resolve QML directory relative to this script
_QML_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "QML")


def export_qmls():
    from qgis.core import QgsProject
    from swe2d.workbench.services.schema_definitions import get_layer_names

    os.makedirs(_QML_DIR, exist_ok=True)

    known = set(get_layer_names())
    exported = 0
    for lyr in QgsProject.instance().mapLayers().values():
        name = str(lyr.name())
        if name.lower() not in known:
            continue

        out_path = os.path.join(_QML_DIR, f"{name.lower()}.qml")
        ok, err = lyr.exportNamedStyle(out_path)
        if ok:
            print(f"  ✓ {name}  ->  {out_path}")
            exported += 1
        else:
            print(f"  ✗ {name}: {err}")

    if exported == 0:
        print("No model GPKG layers found in the project.")
    else:
        print(f"\nExported {exported} QML files to {_QML_DIR}")


if __name__ == "__main__":
    export_qmls()
