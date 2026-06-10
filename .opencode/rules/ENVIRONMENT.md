# Default Working Environment

All commands MUST use the `qgis_stable` mamba environment:

```bash
mamba run -n qgis_stable <command>
```

## Activation

```bash
mamba activate qgis_stable
```

## Python

Python 3.12+ is expected in this environment. All dependencies (numpy, gmsh, PyQt5, qgis, etc.) are provided by the environment.

## Key Paths

- Python: `/home/aaron/miniforge3/envs/qgis_stable/bin/python`
- Build dir: `$REPO_ROOT/build/`
