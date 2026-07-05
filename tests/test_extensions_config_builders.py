"""Tests for the JSON->config builders that used to live in cli/gpkg_adapter.py.

After relocation:
  - build_drainage_config_from_json  -> extensions/drainage_network.py
  - build_structures_config_from_json -> extensions/structures.py
"""

def _drainage_json():
    return {
        "nodes": [{"id": "n1", "x": 0.0, "y": 0.0, "invert": 0.0,
                   "y_max": 5.0, "type": "junction"}],
        "links": [{"id": "l1", "from": "n1", "to": "n1",
                   "length": 100.0, "diameter": 1.0, "roughness": 0.013}],
        "inlets": [],
        "inlet_types": [],
        "node_inlets": [],
        "outfalls": [],
    }


def test_build_drainage_config_importable_from_extensions():
    from swe2d.extensions.drainage_network import build_drainage_config_from_json
    cfg = build_drainage_config_from_json(_drainage_json(), n_cells=10)
    assert cfg is not None
    assert len(cfg.nodes) == 1
    assert len(cfg.links) == 1


def test_build_structures_config_importable_from_extensions():
    from swe2d.extensions.structures import build_structures_config_from_json
    data = {
        "enabled": True,
        "control_interval_s": 2.0,
        "structures": [
            {"id": "s1", "type": "weir", "upstream_cell": 0,
             "downstream_cell": 1, "crest_elev": 5.0},
        ],
    }
    cfg = build_structures_config_from_json(data, n_cells=10)
    assert cfg is not None
    assert cfg.enabled is True
    assert len(cfg.structures) == 1
    assert cfg.control_interval_s == 2.0


def test_build_structures_bare_list_form():
    from swe2d.extensions.structures import build_structures_config_from_json
    data = [{"id": "c1", "type": "culvert", "upstream_cell": 0,
             "downstream_cell": 1, "crest_elev": 2.0,
             "diameter": 0.5, "length": 20.0}]
    cfg = build_structures_config_from_json(data, n_cells=5)
    assert cfg is not None
    assert cfg.enabled is True  # bare-list auto-enables


def test_build_drainage_empty_returns_none():
    from swe2d.extensions.drainage_network import build_drainage_config_from_json
    assert build_drainage_config_from_json(None, n_cells=10) is None
    assert build_drainage_config_from_json({}, n_cells=10) is None


def test_centroid_helper_is_canonical():
    """_compute_cell_centroids in gpkg_adapter should delegate to the canonical."""
    import inspect
    from swe2d.cli import gpkg_adapter
    src = inspect.getsource(gpkg_adapter)
    # After refactor it should import from the canonical location, not define its own loop.
    assert "def _compute_cell_centroids" not in src, (
        "_compute_cell_centroids should be deleted; import canonical instead"
    )
