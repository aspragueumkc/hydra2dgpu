#!/usr/bin/env python3
"""Compare dev (qgis_stable + repo) vs production (qgis_clean + installed wheel)
environments side by side. Designed to find install-only mismatches that the
dev workflow masked — numpy versions, missing .py files, missing compiled
extensions, etc.

Output: two JSON dumps plus a flat diff between them, printed to stdout.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

DEV_PY = os.environ.get(
    "HYDRA_DEV_PYTHON",
    "<set HYDRA_DEV_PYTHON to the dev env's python binary, e.g. $CONDA_PREFIX/bin/python>",
)
PROD_PY = os.environ.get(
    "HYDRA_PROD_PYTHON",
    "<set HYDRA_PROD_PYTHON to the production env's python binary>",
)
_PROD_SITE_RAW = os.environ.get(
    "HYDRA_PROD_SITE",
    "<set HYDRA_PROD_SITE to the production site-packages directory>",
)
PROD_SITE = Path(_PROD_SITE_RAW) if not _PROD_SITE_RAW.startswith("<") else Path(_PROD_SITE_RAW)

INNER = r"""
import sys, os, json, importlib, importlib.metadata
from pathlib import Path

def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}

def _pkg_metadata(name):
    try:
        d = importlib.metadata.distribution(name)
        return {"version": d.version, "location": str(d.locate_file(""))}
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}

def _file_list(pkg_name):
    try:
        m = importlib.import_module(pkg_name)
        path = Path(m.__file__).parent
        files = sorted(str(f.relative_to(path)) for f in path.rglob("*") if f.is_file())
        return {"path": str(path), "file_count": len(files), "files": files[:200]}
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}

def _has_symbol(mod_name, sym):
    try:
        m = importlib.import_module(mod_name)
        return {"present": hasattr(m, sym), "value": repr(getattr(m, sym, None))[:120]}
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}

out = {
    "sys_path_first5": sys.path[:5],
    "sys_path_last5":  sys.path[-5:],
    "python": sys.version.split()[0],
    "platform": sys.platform,
    "executable": sys.executable,
    "site_packages": [str(p) for p in sys.path if "site-packages" in str(p)],
    "packages": {
        "numpy":     _pkg_metadata("numpy"),
        "gmsh":      _pkg_metadata("gmsh"),
        "pyqt5":     _pkg_metadata("pyqt5"),
        "qgis":      _pkg_metadata("qgis"),
        "hydra-swe2d": _pkg_metadata("hydra-swe2d"),
        "swe2d":     _pkg_metadata("swe2d"),
    },
    "hydra_swe2d_tree": _file_list("hydra_swe2d"),
    "swe2d_tree":       _file_list("swe2d"),
    "symbols": {
        "hydra_swe2d.swe2d_deserialize_mesh":  _has_symbol("hydra_swe2d", "swe2d_deserialize_mesh"),
        "hydra_swe2d.swe2d_serialize_mesh":    _has_symbol("hydra_swe2d", "swe2d_serialize_mesh"),
        "swe2d.workbench.doc_viewer":           _has_symbol("swe2d.workbench", "doc_viewer"),
        "swe2d.services.mesh_persistence_service": _has_symbol("swe2d.services", "mesh_persistence_service"),
    },
}
print(json.dumps(out, indent=2, default=str))
"""

def dump(label: str, py: str, extra_env=None) -> dict:
    if py.startswith("<"):
        print(
            f"# [{label}] required env var not set: {label.upper()}_PYTHON",
            file=sys.stderr,
        )
        sys.exit(2)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [py, "-c", INNER],
        capture_output=True, text=True, env=env, timeout=60,
    )
    if proc.returncode != 0:
        print(f"# [{label}] exit {proc.returncode}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        sys.exit(2)
    return json.loads(proc.stdout)


def diff(a: dict, b: dict, path: str = "") -> list[str]:
    """Flat key-by-key diff. Only structural — values printed as-is."""
    out = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        sub = f"{path}.{k}" if path else k
        if k not in a:
            out.append(f"+ {sub}: {b[k]!r}")
        elif k not in b:
            out.append(f"- {sub}: {a[k]!r}")
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            out.extend(diff(a[k], b[k], sub))
        elif a[k] != b[k]:
            out.append(f"  {sub}:")
            out.append(f"    dev : {a[k]!r}")
            out.append(f"    prod: {b[k]!r}")
    return out


if __name__ == "__main__":
    dev = dump("dev",  DEV_PY)
    print(f"==== DEV  ({os.environ.get('HYDRA_DEV_LABEL', 'dev env')}) ====")
    print(json.dumps(dev, indent=2))

    # For prod: prepend the wheel site-packages so it sees the installed
    # hydra-swe2d. We use PYTHONPATH (env-only, not exported back to shell).
    prod_env = {"PYTHONPATH": str(PROD_SITE)}
    prod = dump("prod", PROD_PY, extra_env=prod_env)
    print()
    print(f"==== PROD ({os.environ.get('HYDRA_PROD_LABEL', 'prod env + wheel site-packages')}) ====")
    print(json.dumps(prod, indent=2))

    print()
    print("==== DIFF (dev → prod) ====")
    for line in diff(dev, prod):
        print(line)