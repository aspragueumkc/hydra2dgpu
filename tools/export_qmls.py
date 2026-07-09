"""Export QML styles from a loaded model GeoPackage to QML/ folder.

Usage in QGIS Python console:

    exec(open(r"/home/aaron/QGIS_Plugins_dev/public-repo-hydra2dgpu/tools/export_qmls.py").read())

If __file__ errors, run this instead:

    REPO = r"/home/aaron/QGIS_Plugins_dev/public-repo-hydra2dgpu"
    exec(open(os.path.join(REPO, "tools", "export_qmls.py")).read())
    export_qmls(REPO)
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
