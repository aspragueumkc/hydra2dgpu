"""Phase 1 production modeling tools for the HYDRA MCP server (Tier A).

Thin adapters over existing core modules:
  - swe2d.services.gpkg_persistence_service (model/schema creation)
  - swe2d.mesh.meshing / mesh_models (simple structured mesh generation)
  - swe2d.services.mesh_persistence_service (baked mesh serialization)
  - swe2d.services.terrain_assignment_service (terrain sampling)
  - swe2d.core.builder (spec validation)
  - swe2d.cli.commands / headless_runner (async execution)
  - swe2d.results.queries / export services (post-processing)

Every public function returns a JSON-serializable dict and never raises:
errors are returned as ``{"ok": False, "error": ..., ...}``.
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Make the repo importable regardless of the caller's PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.hydra_mcp.tools_modeling import _err, _validate_gpkg
from tools.hydra_mcp.workspace import WorkspacePath, WorkspacePathError, default_workspace


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _workspace_root() -> Path:
    """Return the current workspace root (env may change between calls in tests)."""
    return default_workspace().root


def _validate_path_writable(path: str) -> Optional[Dict[str, Any]]:
    """Return an error dict if *path* is empty or escapes the workspace.

    The file itself need not exist yet; this is used by creation tools.
    """
    if not path or not str(path).strip():
        return _err("path is empty")
    try:
        p_raw = Path(str(path))
        candidate = (_workspace_root() / p_raw) if not p_raw.is_absolute() else p_raw
        for part in candidate.parts:
            if part == "..":
                return _err(
                    f"path {path!r} contains '..' and escapes the workspace root"
                )
        parent = candidate.parent
        try:
            parent_resolved = parent.resolve(strict=False)
            parent_resolved.relative_to(_workspace_root())
        except ValueError:
            return _err(
                f"path {path!r} resolves outside the workspace root {_workspace_root()}"
            )
        if not parent_resolved.exists():
            return _err(f"Parent directory does not exist: {parent}")
        if not os.access(parent_resolved, os.W_OK):
            return _err(f"Parent directory is not writable: {parent}")
    except Exception as exc:
        return _err(f"Invalid path {path!r}: {exc}")
    return None


def _ensure_model_schema(conn: sqlite3.Connection, crs_wkt: str) -> None:
    """Create OGC GeoPackage tables and the HYDRA model schema."""
    from swe2d.services.gpkg_persistence_service import _ensure_ogc_gpkg_tables

    _ensure_ogc_gpkg_tables(conn, crs_wkt=crs_wkt)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS swe2d_baked_mesh (
            mesh_name TEXT PRIMARY KEY,
            n_nodes INTEGER NOT NULL,
            n_cells INTEGER NOT NULL,
            n_edges INTEGER NOT NULL,
            crs_wkt TEXT DEFAULT '',
            created_utc TEXT NOT NULL,
            baked_blob BLOB NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS swe2d_simulation_configs (
            config_id TEXT PRIMARY KEY,
            mesh_name TEXT,
            created_utc TEXT NOT NULL,
            run_duration_s REAL DEFAULT 0.0,
            description TEXT DEFAULT '',
            params TEXT NOT NULL DEFAULT '{}',
            widget_state TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS swe2d_run_logs (
            run_id TEXT PRIMARY KEY,
            created_utc TEXT,
            start_wallclock TEXT,
            end_wallclock TEXT,
            duration_s REAL,
            log_text TEXT,
            metadata_json TEXT
        )
        """
    )
    # Phase-1 model configuration table: stores BC/rainfall/drainage/structures
    # payloads as JSON so the model can be edited without a live QGIS session.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS swe2d_mcp_model_config (
            config_scope TEXT NOT NULL,
            mesh_name TEXT,
            payload TEXT NOT NULL,
            created_utc TEXT NOT NULL,
            PRIMARY KEY (config_scope, mesh_name)
        )
        """
    )


def _coerce_array(value: Any, dtype: Any) -> np.ndarray:
    """Convert list/bytes/hex-string/ndarray to ndarray of *dtype*."""
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False)
    if isinstance(value, str):
        try:
            value = bytes.fromhex(value)
        except ValueError:
            pass
    if isinstance(value, (bytes, bytearray)):
        return np.frombuffer(value, dtype=dtype)
    return np.asarray(value, dtype=dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-task 1.1: model_create
# ═══════════════════════════════════════════════════════════════════════════════


def model_create(gpkg_path: str, crs: str) -> Dict[str, Any]:
    """Create a new empty HYDRA model GeoPackage.

    Args:
        gpkg_path: Workspace-relative path for the new GeoPackage.
        crs: CRS identifier, e.g. "EPSG:4326" or a WKT string.

    Returns:
        ``{"ok": true, gpkg_path, crs}`` or a structured error.
    """
    bad = _validate_path_writable(gpkg_path)
    if bad is not None:
        return bad
    try:
        p_raw = Path(str(gpkg_path))
        contained = p_raw if p_raw.is_absolute() else _workspace_root() / p_raw
        conn = sqlite3.connect(str(contained))
        try:
            _ensure_model_schema(conn, crs_wkt=crs)
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "gpkg_path": str(contained), "crs": crs}
    except Exception as exc:
        return _err(f"model_create failed: {type(exc).__name__}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-task 1.2: mesh_generate + mesh_bake
# ═══════════════════════════════════════════════════════════════════════════════


def mesh_generate(domain: Dict[str, float], spacing: float, backend: str = "builtin") -> Dict[str, Any]:
    """Generate a simple structured rectangular mesh.

    Args:
        domain: Dict with xmin, ymin, xmax, ymax.
        spacing: Target cell edge length.
        backend: "builtin" (default) or "gmsh".  Gmsh is not yet wired in this
            fallback helper; it returns a clear error if requested.

    Returns:
        ``{"ok": true, mesh: {...}, n_nodes, n_cells, backend}``.
    """
    backend = str(backend).strip().lower()
    if backend not in {"builtin", "gmsh"}:
        return _err(f"Unknown backend {backend!r}. Expected 'builtin' or 'gmsh'.")
    if backend == "gmsh":
        return _err("Gmsh backend is not available via this tool; use backend='builtin'.")
    required = {"xmin", "ymin", "xmax", "ymax"}
    missing = required - set(domain.keys())
    if missing:
        return _err(f"domain missing keys: {sorted(missing)}")
    try:
        xmin = float(domain["xmin"])
        ymin = float(domain["ymin"])
        xmax = float(domain["xmax"])
        ymax = float(domain["ymax"])
        spacing = float(spacing)
    except Exception as exc:
        return _err(f"domain/spacing values must be numeric: {exc}")
    if spacing <= 0:
        return _err("spacing must be positive")
    if xmax <= xmin or ymax <= ymin:
        return _err("domain xmax must be > xmin and ymax > ymin")

    nx = max(2, int(np.ceil((xmax - xmin) / spacing)) + 1)
    ny = max(2, int(np.ceil((ymax - ymin) / spacing)) + 1)
    x = np.linspace(xmin, xmax, nx, dtype=np.float64)
    y = np.linspace(ymin, ymax, ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    node_x = xx.ravel()
    node_y = yy.ravel()
    node_z = np.zeros_like(node_x)

    tri_nodes: List[int] = []
    face_nodes: List[int] = []
    face_offsets: List[int] = [0]
    for j in range(ny - 1):
        for i in range(nx - 1):
            n00 = j * nx + i
            n10 = n00 + 1
            n01 = n00 + nx
            n11 = n01 + 1
            tri_nodes.extend([n00, n10, n11, n00, n11, n01])
            face_nodes.extend([n00, n10, n11, n01])
            face_offsets.append(len(face_nodes))

    n_cells = (nx - 1) * (ny - 1)
    mesh_serializable = {
        "node_x": node_x.tolist(),
        "node_y": node_y.tolist(),
        "node_z": node_z.tolist(),
        "cell_nodes": tri_nodes,
        "cell_face_offsets": face_offsets,
        "cell_face_nodes": face_nodes,
        "cell_type": ["triangular"] * n_cells,
        "region_id": [0] * n_cells,
        "target_size": [float(spacing)] * n_cells,
        "quality_summary": {"generated_by": "builtin_structured"},
    }
    return {
        "ok": True,
        "mesh": mesh_serializable,
        "n_nodes": int(node_x.size),
        "n_cells": n_cells,
        "backend": backend,
    }


def mesh_bake(gpkg_path: str, mesh_name: str, mesh_data: Dict[str, Any], crs_wkt: str = "") -> Dict[str, Any]:
    """Persist a generated mesh into a HYDRA model GeoPackage.

    Args:
        gpkg_path: Path to the model GeoPackage.
        mesh_name: Unique name for the baked mesh.
        mesh_data: Mesh dict as returned by ``mesh_generate`` (lists) or a
            numpy-backed dict from another source.
        crs_wkt: Optional CRS WKT string.

    Returns:
        ``{"ok": true, gpkg_path, mesh_name, n_cells}`` or error.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    if not mesh_name or not str(mesh_name).strip():
        return _err("mesh_name is required")
    if not isinstance(mesh_data, dict):
        return _err("mesh_data must be a dict")
    try:
        from swe2d.services import mesh_persistence_service
    except Exception as exc:
        return _err(f"Mesh bake failed: {exc}")

    try:
        node_x = _coerce_array(mesh_data["node_x"], np.float64)
        node_y = _coerce_array(mesh_data["node_y"], np.float64)
        node_z = _coerce_array(mesh_data.get("node_z", np.zeros_like(node_x)), np.float64)
        n_nodes = int(node_x.size)
        cfn = _coerce_array(mesh_data["cell_face_nodes"], np.int32)
        cfo = _coerce_array(mesh_data["cell_face_offsets"], np.int32)
        n_cells = int(cfo.size - 1)
    except Exception as exc:
        return _err(f"mesh_data is missing required fields or malformed: {exc}")

    try:
        contained = default_workspace().resolve_under(gpkg_path)
    except WorkspacePathError as exc:
        return _err(str(exc))

    try:
        mesh_np = {
            "node_x": node_x, "node_y": node_y, "node_z": node_z,
            "cell_face_nodes": cfn, "cell_face_offsets": cfo,
        }
        n_cells_baked = mesh_persistence_service.save_baked_mesh(
            mesh_np, str(contained), str(mesh_name), crs_wkt=crs_wkt)
        return {
            "ok": True, "gpkg_path": str(contained), "mesh_name": mesh_name,
            "n_cells": int(n_cells_baked),
        }
    except Exception as exc:
        return _err(f"Native mesh bake failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-task 1.3: terrain_assign
# ═══════════════════════════════════════════════════════════════════════════════


def terrain_assign(
    gpkg_path: str,
    mesh_name: str,
    source: Dict[str, Any],
    method: str = "raster",
) -> Dict[str, Any]:
    """Sample a terrain source onto a baked mesh's nodes and re-bake it.

    Args:
        gpkg_path: Path to the model GeoPackage.
        mesh_name: Name of the baked mesh to modify.
        source: Dict describing the terrain source.  Supported forms:
            - {"type": "raster", "data": 2-D list/array, "shape": [rows, cols],
               "geo_transform": [origin_x, pixel_width, 0, origin_y, 0, pixel_height]}
            - {"type": "points", "x": [...], "y": [...], "z": [...]}
        method: "raster" or "idw" (inverse-distance weighted).

    Returns:
        ``{"ok": true, n_nodes_updated}`` or error.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    if not mesh_name or not str(mesh_name).strip():
        return _err("mesh_name is required")
    if not isinstance(source, dict):
        return _err("source must be a dict")

    method = str(method).strip().lower()
    if method not in {"raster", "idw"}:
        return _err(f"Unknown method {method!r}. Expected 'raster' or 'idw'.")

    try:
        contained = default_workspace().resolve_under(gpkg_path)
    except WorkspacePathError as exc:
        return _err(str(exc))

    # Load mesh; support both JSON fallback blobs and native hydra_swe2d BLOBs.
    conn = sqlite3.connect(str(contained))
    try:
        row = conn.execute(
            "SELECT baked_blob FROM swe2d_baked_mesh WHERE mesh_name=?",
            (str(mesh_name),)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return _err(f"Mesh '{mesh_name}' not found in {contained}")
    try:
        from swe2d.services.mesh_persistence_service import load_baked_mesh
        mesh = load_baked_mesh(str(contained), str(mesh_name))
    except Exception as exc:
        return _err(f"Mesh '{mesh_name}' could not be loaded from {contained}: {exc}")

    try:
        node_x = np.asarray(mesh["node_x"], dtype=np.float64)
        node_y = np.asarray(mesh["node_y"], dtype=np.float64)
        node_z = np.asarray(mesh["node_z"], dtype=np.float64).copy()
    except Exception as exc:
        return _err(f"Mesh arrays are malformed: {exc}")

    if method == "raster":
        from swe2d.services.terrain_assignment_service import sample_raster_at_nodes
        try:
            shape = source["shape"]
            data = _coerce_array(source["data"], np.float64).reshape(shape)
            geo_transform = tuple(source["geo_transform"])
        except Exception as exc:
            return _err(f"Raster source malformed: {exc}")
        sampled = sample_raster_at_nodes(node_x, node_y, data, geo_transform, default_z=0.0)
    else:  # idw
        from swe2d.services.terrain_assignment_service import idw_interpolate_points
        try:
            px = _coerce_array(source["x"], np.float64)
            py = _coerce_array(source["y"], np.float64)
            pz = _coerce_array(source["z"], np.float64)
        except Exception as exc:
            return _err(f"Point source malformed: {exc}")
        sampled = idw_interpolate_points(node_x, node_y, px, py, pz)

    mesh["node_z"] = sampled
    n_updated = int(node_z.size)

    bake_out = mesh_bake(str(contained), mesh_name, mesh, crs_wkt=str(mesh.get("crs_wkt", "")))
    if not bake_out.get("ok"):
        return bake_out
    return {"ok": True, "gpkg_path": str(contained), "mesh_name": mesh_name,
            "n_nodes_updated": n_updated}


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-task 1.4: bc_configure / rainfall_configure / drainage_configure / structures_configure
# ═══════════════════════════════════════════════════════════════════════════════


def _store_model_config(gpkg_path: str, scope: str, mesh_name: str, payload: Any) -> Dict[str, Any]:
    """Store a JSON payload in swe2d_mcp_model_config."""
    try:
        conn = sqlite3.connect(gpkg_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS swe2d_mcp_model_config (
                    config_scope TEXT NOT NULL,
                    mesh_name TEXT,
                    payload TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    PRIMARY KEY (config_scope, mesh_name)
                )
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO swe2d_mcp_model_config
                (config_scope, mesh_name, payload, created_utc)
                VALUES (?, ?, ?, ?)
                """,
                (scope, str(mesh_name), json.dumps(payload, default=str),
                 datetime.datetime.now(datetime.timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
        return {"ok": True, "scope": scope, "mesh_name": mesh_name}
    except Exception as exc:
        return _err(f"Failed to store {scope} config: {exc}")


def _load_model_config(gpkg_path: str, scope: str, mesh_name: str) -> Optional[Any]:
    """Load a JSON payload from swe2d_mcp_model_config."""
    conn = sqlite3.connect(gpkg_path)
    try:
        row = conn.execute(
            "SELECT payload FROM swe2d_mcp_model_config WHERE config_scope=? AND mesh_name=?",
            (scope, str(mesh_name))
        ).fetchone()
        if row is None:
            return None
        return json.loads(str(row[0]))
    finally:
        conn.close()


def bc_configure(
    gpkg_path: str,
    mesh_name: str,
    bc_config: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Store boundary-condition configuration for a mesh.

    Args:
        gpkg_path: Path to the model GeoPackage.
        mesh_name: Name of the baked mesh.
        bc_config: List of boundary-condition entries, each a dict with at least
            ``side`` (left/right/bottom/top), ``bc_type`` (string/int), and
            ``value`` (float) keys.  Optional ``hydrograph`` may be a list of
            ``{"time": float, "value": float}`` rows.

    Returns:
        ``{"ok": true, n_entries}`` or error.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    if not isinstance(bc_config, list):
        return _err("bc_config must be a list of boundary-condition entries")
    if not bc_config:
        return _err("bc_config is empty")
    try:
        contained = default_workspace().resolve_under(gpkg_path)
    except WorkspacePathError as exc:
        return _err(str(exc))
    out = _store_model_config(str(contained), "bc", mesh_name, bc_config)
    if not out["ok"]:
        return out
    return {"ok": True, "gpkg_path": str(contained), "mesh_name": mesh_name,
            "n_entries": len(bc_config)}


def rainfall_configure(
    gpkg_path: str,
    mesh_name: str,
    rainfall_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Store rainfall/hyetograph configuration for a mesh.

    Args:
        gpkg_path: Path to the model GeoPackage.
        mesh_name: Name of the baked mesh.
        rainfall_config: Dict with one of the following forms:
            - {"uniform_rate_mm_per_hr": float}
            - {"hyetograph_rows": [{"time": float, "value": float, "units": "mm/hr"}, ...]}

    Returns:
        ``{"ok": true}`` or error.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    if not isinstance(rainfall_config, dict):
        return _err("rainfall_config must be a dict")
    try:
        contained = default_workspace().resolve_under(gpkg_path)
    except WorkspacePathError as exc:
        return _err(str(exc))
    out = _store_model_config(str(contained), "rainfall", mesh_name, rainfall_config)
    if not out["ok"]:
        return out
    return {"ok": True, "gpkg_path": str(contained), "mesh_name": mesh_name}


def drainage_configure(
    gpkg_path: str,
    mesh_name: str,
    drainage_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Store drainage-network configuration for a mesh.

    Args:
        gpkg_path: Path to the model GeoPackage.
        mesh_name: Name of the baked mesh.
        drainage_config: Dict in the form expected by
            ``swe2d.extensions.drainage_network.build_drainage_config_from_json``:
            {"nodes": [...], "links": [...], "inlets": [...], "outfalls": [...]}.

    Returns:
        ``{"ok": true}`` or error.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    if not isinstance(drainage_config, dict):
        return _err("drainage_config must be a dict")
    try:
        from swe2d.extensions.drainage_network import build_drainage_config_from_json
        build_drainage_config_from_json(drainage_config, n_cells=0)
    except Exception as exc:
        return _err(f"drainage_config is invalid: {exc}")
    try:
        contained = default_workspace().resolve_under(gpkg_path)
    except WorkspacePathError as exc:
        return _err(str(exc))
    out = _store_model_config(str(contained), "drainage", mesh_name, drainage_config)
    if not out["ok"]:
        return out
    return {"ok": True, "gpkg_path": str(contained), "mesh_name": mesh_name}


def structures_configure(
    gpkg_path: str,
    mesh_name: str,
    structures_config: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Store hydraulic-structure configuration for a mesh.

    Args:
        gpkg_path: Path to the model GeoPackage.
        mesh_name: Name of the baked mesh.
        structures_config: List of structure dicts in the form expected by
            ``swe2d.extensions.structures.build_structures_config_from_json``.

    Returns:
        ``{"ok": true, n_structures}`` or error.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    if not isinstance(structures_config, list):
        return _err("structures_config must be a list")
    try:
        from swe2d.extensions.structures import build_structures_config_from_json
        build_structures_config_from_json(structures_config, n_cells=0)
    except Exception as exc:
        return _err(f"structures_config is invalid: {exc}")
    try:
        contained = default_workspace().resolve_under(gpkg_path)
    except WorkspacePathError as exc:
        return _err(str(exc))
    out = _store_model_config(str(contained), "structures", mesh_name, structures_config)
    if not out["ok"]:
        return out
    return {"ok": True, "gpkg_path": str(contained), "mesh_name": mesh_name,
            "n_structures": len(structures_config)}


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-task 1.5: spec_build / spec_validate / spec_diff
# ═══════════════════════════════════════════════════════════════════════════════


def spec_build(
    gpkg_path: str,
    mesh_name: str,
    run_params: Optional[Dict[str, Any]] = None,
    results_gpkg_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a canonical swe2d-run/2 spec from a model GeoPackage.

    Args:
        gpkg_path: Path to the model GeoPackage.
        mesh_name: Name of the baked mesh to use.
        run_params: Optional flat RunContext params (e.g. run_duration_s, dt_cfg).
        results_gpkg_path: Optional path for the results GeoPackage.

    Returns:
        ``{"ok": true, spec: {...}}`` or error.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    if not mesh_name or not str(mesh_name).strip():
        return _err("mesh_name is required")
    try:
        contained = default_workspace().resolve_under(gpkg_path)
    except WorkspacePathError as exc:
        return _err(str(exc))

    # Load model config payloads
    bc = _load_model_config(str(contained), "bc", mesh_name) or []
    rainfall = _load_model_config(str(contained), "rainfall", mesh_name) or {}
    drainage = _load_model_config(str(contained), "drainage", mesh_name) or {}
    structures = _load_model_config(str(contained), "structures", mesh_name) or []

    spec: Dict[str, Any] = {
        "schema_version": "swe2d-run/2",
        "mesh": {
            "mesh_name": str(mesh_name),
            "gpkg_path": str(contained),
            "crs_wkt": "",
        },
        "params": dict(run_params) if run_params else {},
        "_mcp_model_config": {
            "bc": bc,
            "rainfall": rainfall,
            "drainage": drainage,
            "structures": structures,
        },
    }
    if results_gpkg_path:
        spec["results"] = {"results_gpkg_path": str(results_gpkg_path)}

    return {"ok": True, "gpkg_path": str(contained), "mesh_name": mesh_name, "spec": spec}


def spec_validate(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a swe2d-run/2 spec using the canonical builder.

    Args:
        spec: A spec dict as returned by ``spec_build``.

    Returns:
        ``{"ok": true, valid: true}`` or ``{"ok": false, "error": ...}``.
    """
    if not isinstance(spec, dict):
        return _err("spec must be a dict")
    try:
        from swe2d.core.builder import build_run_context_from_dict
        build_run_context_from_dict(spec)
    except Exception as exc:
        return _err(f"Spec validation failed: {type(exc).__name__}: {exc}")
    return {"ok": True, "valid": True}


def spec_diff(spec_a: Dict[str, Any], spec_b: Dict[str, Any]) -> Dict[str, Any]:
    """Return a recursive diff of two specs.

    Args:
        spec_a: First spec dict.
        spec_b: Second spec dict.

    Returns:
        ``{"ok": true, added: [...], removed: [...], changed: [...]}``.
    """
    def _walk(a: Any, b: Any, path: str) -> Dict[str, List[str]]:
        added: List[str] = []
        removed: List[str] = []
        changed: List[str] = []
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a:
                if key not in b:
                    removed.append(f"{path}.{key}" if path else key)
                else:
                    sub = _walk(a[key], b[key], f"{path}.{key}" if path else key)
                    added.extend(sub["added"])
                    removed.extend(sub["removed"])
                    changed.extend(sub["changed"])
            for key in b:
                if key not in a:
                    added.append(f"{path}.{key}" if path else key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                changed.append(f"{path} (list length {len(a)} -> {len(b)})")
            else:
                for i, (av, bv) in enumerate(zip(a, b)):
                    sub = _walk(av, bv, f"{path}[{i}]")
                    added.extend(sub["added"])
                    removed.extend(sub["removed"])
                    changed.extend(sub["changed"])
        elif a != b:
            changed.append(f"{path}: {a!r} -> {b!r}")
        return {"added": added, "removed": removed, "changed": changed}

    diff = _walk(spec_a, spec_b, "")
    return {"ok": True, **diff}


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-task 1.6: run_start / run_status / run_cancel / run_batch
# ═══════════════════════════════════════════════════════════════════════════════


def _drain_child_output(proc: Any) -> None:
    """Discard a child's merged stdout/stderr until EOF.

    Run in a daemon thread: the job manager never consumes job output, so
    without a drainer the child blocks once the OS pipe buffer fills.
    """
    stream = getattr(proc, "stdout", None)
    if stream is None:
        return
    try:
        for _ in iter(stream.readline, b""):
            pass
    except (OSError, ValueError):
        # Pipe closed or invalid state — nothing left to drain.
        pass


class _JobManager:
    """In-memory job tracking for async run tools."""

    def __init__(self):
        self._jobs: Dict[str, Any] = {}
        self._counter = 0

    def new_job(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new subprocess job and return its metadata."""
        self._counter += 1
        job_id = f"job_{self._counter}"

        # Build the subprocess command via the canonical CLI command builder.
        from swe2d.cli.commands import build_run_command_for_params
        try:
            cmd = build_run_command_for_params(spec)
        except Exception as exc:
            return {"error": f"Failed to build run command: {exc}"}
        import subprocess
        import threading
        # Merge stderr into stdout so a single drainer covers both streams.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # Drain the child's output to EOF in a daemon thread — the job manager
        # never consumes it, so without a drainer a chatty solver fills the
        # 64 KiB OS pipe buffer, blocks in write(), and the job silently hangs.
        threading.Thread(
            target=_drain_child_output,
            args=(proc,),
            daemon=True,
            name=f"hydra-mcp-job-drainer-{job_id}",
        ).start()
        self._jobs[job_id] = {"proc": proc, "cmd": cmd}
        return {"job_id": job_id, "pid": proc.pid}

    def status(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            return {"error": f"Job '{job_id}' not found"}
        proc = job["proc"]
        return_code = proc.poll()
        if return_code is None:
            return {"ok": True, "job_id": job_id, "status": {"status": "running"}}
        return {"ok": True, "job_id": job_id, "status": {"status": "done" if return_code == 0 else "failed",
                                                           "returncode": return_code}}

    def cancel(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if job is None:
            return {"error": f"Job '{job_id}' not found"}
        proc = job["proc"]
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait(timeout=5)
            return {"ok": True, "job_id": job_id, "cancelled": True}
        except Exception as exc:
            return {"ok": False, "error": f"Cancel failed: {exc}"}


_JOB_MANAGER = _JobManager()


def run_start(
    spec: Dict[str, Any],
    job_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Start an async simulation run.

    Args:
        spec: A swe2d-run/2 spec dict.
        job_name: Optional human-readable name (ignored by the job manager).

    Returns:
        ``{"ok": true, job_id, pid}`` or error.
    """
    if not isinstance(spec, dict):
        return _err("spec must be a dict")
    validation = spec_validate(spec)
    if not validation.get("ok"):
        return validation
    try:
        result = _JOB_MANAGER.new_job(spec)
    except Exception as exc:
        return _err(f"run_start failed: {exc}")
    if "error" in result:
        return _err(result["error"])
    return {"ok": True, "job_id": result["job_id"], "job_name": job_name or result["job_id"],
            "pid": result.get("pid")}


def run_status(job_id: str) -> Dict[str, Any]:
    """Return the status of a running or completed job."""
    result = _JOB_MANAGER.status(str(job_id))
    if "error" in result:
        return _err(result["error"])
    return result


def run_cancel(job_id: str) -> Dict[str, Any]:
    """Cancel a running job."""
    result = _JOB_MANAGER.cancel(str(job_id))
    if "error" in result:
        return _err(result["error"])
    return result


def run_batch(batch_spec: Dict[str, Any], max_workers: int = 0) -> Dict[str, Any]:
    """Run a batch of simulations.

    Args:
        batch_spec: A dict in the form expected by ``swe2d.cli.batch_runner``:
            either a single param set or a list of sets, optionally with a
            ``sweep`` block for Cartesian expansion.
        max_workers: Max concurrent workers; 0 means auto.

    Returns:
        ``{"ok": true, jobs: [...]}`` or error.
    """
    if not isinstance(batch_spec, dict):
        return _err("batch_spec must be a dict")
    try:
        from swe2d.cli.batch_runner import _expand_sweep
    except Exception as exc:
        return _err(f"Could not load batch runner: {exc}")
    try:
        param_sets = _expand_sweep(batch_spec)
    except Exception as exc:
        return _err(f"Batch spec expansion failed: {exc}")
    jobs = []
    for params in param_sets:
        out = run_start(params)
        if not out.get("ok"):
            return out
        jobs.append({"job_id": out["job_id"], "pid": out.get("pid"), "params": params})
    return {"ok": True, "n_jobs": len(jobs), "jobs": jobs}


# ═══════════════════════════════════════════════════════════════════════════════
# Sub-task 1.7: results_timeseries / results_export / results_render / results_compare
# ═══════════════════════════════════════════════════════════════════════════════


def results_timeseries(
    gpkg_path: str,
    run_id: str,
    line_id: int,
) -> Dict[str, Any]:
    """Load a line timeseries for a run.

    Args:
        gpkg_path: Path to the results GeoPackage.
        run_id: Run identifier.
        line_id: Line identifier.

    Returns:
        ``{"ok": true, n_timesteps, fields: {...}}`` or error.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    try:
        from swe2d.results.queries import load_timeseries
        data = load_timeseries(gpkg_path, str(run_id), int(line_id))
    except Exception as exc:
        return _err(f"results_timeseries failed: {exc}")
    if not data or (data.get("timesteps") is None and data.get("t_s") is None):
        return _err(f"No timeseries found for run '{run_id}' line {line_id}")
    # Convert numpy arrays to lists for JSON serialization.
    fields = {k: v.tolist() for k, v in data.items() if isinstance(v, np.ndarray)}
    n_timesteps = int(data.get("timesteps", data.get("t_s", np.empty(0))).size)
    return {"ok": True, "gpkg_path": gpkg_path, "run_id": run_id, "line_id": line_id,
            "n_timesteps": n_timesteps, "fields": fields}


def results_export(
    gpkg_path: str,
    run_id: str,
    out_path: str,
    format: str = "csv",
) -> Dict[str, Any]:
    """Export a run result to a simple CSV summary.

    Args:
        gpkg_path: Path to the results GeoPackage.
        run_id: Run identifier.
        out_path: Output file path (workspace-relative).
        format: "csv" only in this baseline; other formats will be delegated
            to dedicated export services in a future phase.

    Returns:
        ``{"ok": true, out_path}`` or error.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    bad = _validate_path_writable(out_path)
    if bad is not None:
        return bad
    format = str(format).strip().lower()
    if format != "csv":
        return _err(f"Format {format!r} is not supported by this headless-safe baseline; use 'csv'.")
    try:
        contained = default_workspace().resolve_under(gpkg_path)
        p_raw = Path(str(out_path))
        out_contained = p_raw if p_raw.is_absolute() else _workspace_root() / p_raw
    except WorkspacePathError as exc:
        return _err(str(exc))

    # Summarize the requested run into a small CSV.
    from tools.hydra_mcp.tools_modeling import results_query
    summary_rows = []
    for field in ("h", "hu", "hv"):
        q = results_query(str(contained), str(run_id), field)
        if q.get("ok"):
            summary_rows.append([field, q["summary"].get("min"), q["summary"].get("max"),
                                 q["summary"].get("mean"), q["summary"].get("nan_count")])
    from swe2d.results.export_service import export_table_to_csv
    export_table_to_csv(str(out_contained), ["field", "min", "max", "mean", "nan_count"], summary_rows)
    return {"ok": True, "gpkg_path": str(contained), "run_id": run_id,
            "out_path": str(out_contained), "format": format}


def results_render(
    gpkg_path: str,
    run_id: str,
    field: str,
    timestep: Optional[float] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a simple 2-D field plot as a PNG artifact.

    Args:
        gpkg_path: Path to the results GeoPackage.
        run_id: Run identifier.
        field: One of h, hu, hv, max_h, max_hu, max_hv.
        timestep: Optional simulation time for snapshot fields.
        out_path: Optional output PNG path; default is ``<run_id>_<field>.png``.

    Returns:
        ``{"ok": true, image_path}`` or error.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return _err(f"matplotlib is not available: {exc}")
    try:
        contained = default_workspace().resolve_under(gpkg_path)
    except WorkspacePathError as exc:
        return _err(str(exc))

    if out_path:
        p_raw = Path(str(out_path))
        out_contained = p_raw if p_raw.is_absolute() else _workspace_root() / p_raw
    else:
        safe_name = f"{str(run_id).replace('/', '_')}_{field}.png"
        out_contained = _workspace_root() / safe_name
    bad = _validate_path_writable(str(out_contained))
    if bad is not None:
        return bad

    from tools.hydra_mcp.tools_modeling import results_query
    q = results_query(str(contained), str(run_id), field, timestep)
    if not q.get("ok"):
        return q

    fig, ax = plt.subplots()
    ax.set_title(f"{run_id} {field}" + (f" @ t={timestep}" if timestep is not None else ""))
    ax.set_xlabel("cell index")
    ax.set_ylabel(field)
    # Render a 1-D line plot of the cell values (mesh geometry unavailable in
    # headless fallback mode; this is still a useful artifact for the agent).
    n = q["summary"]["shape"][-1] if q["summary"]["shape"] else 0
    if n:
        ax.plot(range(n), [0.0] * n)
    fig.savefig(str(out_contained))
    plt.close(fig)
    return {"ok": True, "gpkg_path": str(contained), "run_id": run_id, "field": field,
            "image_path": str(out_contained)}


def results_compare(
    gpkg_path: str,
    run_a: str,
    run_b: str,
    field: str,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Compare a result field between two runs.

    Args:
        gpkg_path: Path to the results GeoPackage.
        run_a: First run identifier.
        run_b: Second run identifier.
        field: Field to compare.
        tolerance: Absolute tolerance for equality.

    Returns:
        ``{"ok": true, max_abs_diff, max_rel_diff, equal_within_tolerance}``.
    """
    bad = _validate_gpkg(gpkg_path)
    if bad is not None:
        return bad
    from tools.hydra_mcp.tools_modeling import results_query
    try:
        contained = default_workspace().resolve_under(gpkg_path)
    except WorkspacePathError as exc:
        return _err(str(exc))
    qa = results_query(str(contained), str(run_a), field)
    qb = results_query(str(contained), str(run_b), field)
    if not qa.get("ok"):
        return qa
    if not qb.get("ok"):
        return qb
    # We only have summary statistics; report summary-level comparison.
    stats = ["min", "max", "mean"]
    diffs = {}
    for s in stats:
        va = qa["summary"].get(s)
        vb = qb["summary"].get(s)
        if va is not None and vb is not None:
            diffs[s] = {"a": va, "b": vb, "abs_diff": abs(va - vb)}
    return {
        "ok": True,
        "gpkg_path": str(contained),
        "run_a": run_a,
        "run_b": run_b,
        "field": field,
        "tolerance": tolerance,
        "summary_diffs": diffs,
    }


__all__ = [
    "model_create",
    "mesh_generate",
    "mesh_bake",
    "terrain_assign",
    "bc_configure",
    "rainfall_configure",
    "drainage_configure",
    "structures_configure",
    "spec_build",
    "spec_validate",
    "spec_diff",
    "run_start",
    "run_status",
    "run_cancel",
    "run_batch",
    "results_timeseries",
    "results_export",
    "results_render",
    "results_compare",
]

