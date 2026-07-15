"""Export QML styles from a loaded model GeoPackage to QML/ folder.

use in qgis python console
"""
import os
import sys


def export_qmls(repo_root=None):
    if repo_root is None:
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        except NameError:
            repo_root = os.path.expanduser("~")

    from qgis.core import QgsProject
    from swe2d.workbench.services.schema_definitions import get_layer_names

    qml_dir = os.path.join(repo_root, "QML")
    os.makedirs(qml_dir, exist_ok=True)

    known = set(get_layer_names())
    exported = 0
    for lyr in QgsProject.instance().mapLayers().values():
        name = str(lyr.name())
        if name.lower() not in known:
            continue
        out_path = os.path.join(qml_dir, f"{name.lower()}.qml")
        err, ok = lyr.saveNamedStyle(out_path)
        if ok:
            print(f"  ok {name}  ->  {out_path}")
            exported += 1
        else:
            print(f"  FAIL {name}: {err}")

    print(f"\nExported {exported} QML files to {qml_dir}")


if __name__ == "__main__":
    export_qmls()
