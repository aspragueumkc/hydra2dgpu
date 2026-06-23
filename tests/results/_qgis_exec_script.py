# QGIS exec script — launched by run_headless_qgis_tests.sh
# QGIS already initialized itself before running --code.
# We just need our paths.
import os, sys
repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for p in (repo, os.path.join(repo, "build", "Release"), os.path.join(repo, "build")):
    if p not in sys.path: sys.path.insert(0, p)
os.environ['QGIS_PLUGINPATH'] = repo

try:
    exit(0)
    print("QGIS_EXEC_OK")
except Exception as _e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
else:
    sys.exit(0)
