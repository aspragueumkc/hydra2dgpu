# HYDRA2DGPU Distribution Guide

## For End Users (QGIS Plugin Manager)

1. Open QGIS → `Plugins` → `Manage and Install Plugins`.
2. Search for `HYDRA2DGPU` and install it.
3. The first time you open the plugin, it downloads and installs the `hydra-swe2d` backend into `~/.hydra2dgpu/` (or `%USERPROFILE%\.hydra2dgpu\` on Windows).
4. NVIDIA GPU with CUDA support is required (compute capability >= 7.5). CPU-only execution is not supported.

## For Developers (Source Build)

```bash
git clone https://github.com/aspragueumkc/hydra2dgpu
cd hydra2dgpu
mamba create -n hydra python=3.12 numpy gmsh netcdf4 h5py matplotlib
mamba activate hydra
conda install -c nvidia cuda-toolkit=12.4

# Local plugin dev (symlink trick — see AGENTS.md "ENVIRONMENT.md")
ln -s "$(pwd)" ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/hydra2dgpu

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBACKWATER_USE_CUDA=ON
cmake --build build -j$(nproc)

PYTHONPATH="$PWD:$PWD/build" python -m unittest discover -s tests -p "test_*.py"
```

## For QGIS Plugin Maintainers (Submitting Updates)

The release version lives in **one place**: `pyproject.toml::project.version`.
Everything else (the wheel name, `metadata.txt::version=`, and
`installer.py::WHEEL_VERSION`) is derived from it by
`tools/sync_version.py` at release time.

1. Bump `version` in `pyproject.toml`.
2. Append to `## [Unreleased]` in `CHANGELOG.md`.
3. Run `python tools/sync_version.py` to propagate the version to
   `metadata.txt` and `installer.py`. Commit all three files together.
4. Tag the commit: `git tag -a v1.2.0 -m "Release 1.2.0"`.
5. Push the tag. CI:
   - `build-wheels.yml` builds 6 platform wheels (cp310/311/312 × linux/windows) and uploads them as artifacts.
   - `release.yml` runs `tools/sync_version.py`, downloads the wheels,
     builds `dist/HYDRA2DGPU.zip`, and attaches both to the GitHub Release.
6. Upload `dist/HYDRA2DGPU.zip` to https://plugins.qgis.org/ (binary-free zip,
   size asserted <20 MB by `tools/package_plugin.py`).

### Manual rebuild without re-tagging

`release.yml` also accepts `workflow_dispatch`. Pass the existing tag name
(`tag_name=v1.2.0`) and the workflow will re-run `sync_version.py` +
`package_plugin.py` and re-attach the zip to the GitHub Release. Wheels
are **not** rebuilt — run `build-wheels.yml` manually first if you need
fresh wheels.

> **Agent automation:** load `.agents/skills/hydra-release-publish/SKILL.md` for the executable version of this flow (with exact commands, pre-flight checks, artifact verification, and rollback paths).

## Advanced: Pixi Environment

See `pixi.toml` for a fully reproducible QGIS+CUDA dev environment.

```bash
pixi install
pixi run qgis
```
