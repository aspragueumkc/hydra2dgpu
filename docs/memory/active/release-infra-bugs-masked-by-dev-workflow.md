---
type: memory
status: active
created: 2026-07-27
topic: release-infra-bugs-masked-by-dev-workflow
tags: [qgis, infra, session, lesson]
evidence: .worktrees/release-test/qgis_plugin/HYDRA2DGPU/__init__.py:25-32
---
# Release infrastructure bugs are masked by the dev `ln -s` workflow

## Context

This session's local end-to-end install test surfaced **eight real bugs
in the release infra that the dev `ln -s` symlink workflow had silently
masked** — same commit, same code, completely different behavior between
dev and production install.

The dev workflow (`ln -s <repo> …/qgis_stable/.../plugins/hydra2dgpu` +
`PYTHONPATH=<repo>` + cwd at the repo root) automatically puts `swe2d/`
and `hydra_swe2d/` directly on `sys.path` and bypasses every
"first-launch install" code path. Code that runs **only** in a fresh
profile, no `PYTHONPATH`, zip-extracted folder, downloaded wheel — was
never exercised by dev.

## Decision

**Always test release paths from a clean state.** Dev `ln -s` is not a
substitute. Required minimum:

1. Build wheel via `python -m build` (CI form), not just `cmake` in dev.
2. Build zip via `tools/package_plugin.py` — verify root folder matches
   `metadata.txt::name=`.
3. Use a fresh `qgis_clean` conda env (no CUDA, no dev symlinks).
4. Install via QGIS Plugin Manager "Install from ZIP", not `ln -s`.
5. Launch from a non-repo cwd, with `PYTHONPATH` stripped.
6. Watch the install actually fire (dialog, GET, `~/.hydra2dgpu/`
   populates, restart, workbench opens).

If any step is "we always do that manually in dev", that step is where
the next bug lives.

## Bugs found this session

1. **Wheel missing `swe2d/`** (`pyproject.toml` + `CMakeLists.txt`):
   `hydra_swe2d/` C++ extension shipped but Python sources didn't.
   Dev cwd/repo path put `swe2d/` on sys.path; wheel never had to.
2. **Zip missing root folder** (`package_plugin.py`): files at the
   root, no top-level `HYDRA2DGPU/` folder — rejected by QGIS Plugin
   Manager. Dev install is via symlink; folder name irrelevant.
3. **Silent pip failure** (`installer.py:91`):
   `subprocess.run(..., capture_output=True)` swallowed pip errors.
   Dev never triggered the install path, so this never failed.
4. **`_import_all()` deadlock** (`__init__.py:164`): raised
   `ImportError` when `swe2d` wasn't installed yet — install dialog
   can only fire after the plugin loads. Dev always had `swe2d/`.
5. **Case-sensitivity mismatch** (`swe2d/workbench/{startup_state,
   views/doc_viewer}.py`): hard-coded `from hydra2dgpu import …`
   (lowercase); prod folder is `HYDRA2DGPU/` (uppercase, matches
   `metadata.txt::name=`). Dev symlink is lowercase.
6. **Docs missing in zip** (`package_plugin.py` `EXCLUDE_DIRS`):
   excluded all of `docs/`; Help tab couldn't find any guide. Dev
   resolves `<repo>/docs/` directly.
7. **No post-restart path wiring** (`__init__.py` was missing this):
   `installer.py:install()` called `add_env_to_path()` only inside
   the worker thread — path addition died with the thread. Dev never
   restarts.
8. **Case-alias regression** (`__init__.py:189-196`, my own fix for
   #5): aliasing `sys.modules['hydra2dgpu'] = sys.modules['HYDRA2DGPU']`
   shadowed the `hydra_swe2d` C extension. Only manifests in prod.

Bugs #2, #5, #6 would have shipped to `plugins.qgis.org` and been
rejected / silently broken. Bugs #1, #3, #4, #7 would have shipped and
the plugin would have crashed on first launch.

## Open questions

- CI integration test: spin up `qgis_clean`, build wheel + zip, install
  via headless QGIS, assert workbench opens. Needs Xvfb.
- Other untested install-only paths: Windows multiprocessing guard in
  `__init__.py:33-70`, `add_env_to_path()` race mid-install, project
  restore.
- Should `package_plugin.py` add a post-build smoke test that invokes
  `python -c "from <plugin> import PLUGIN_ROOT"` on the built zip?
- Memory CLI bug (separate): `draft` accepts `--text` literally with
  no agent-expansion hook. Filed for later.
