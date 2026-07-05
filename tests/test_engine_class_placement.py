"""Engine base classes should live next to their concrete subclasses."""


def test_drainage_engine_in_drainage_network_module():
    from swe2d.extensions.drainage_network import DrainageCouplingEngine
    assert DrainageCouplingEngine is not None


def test_structure_engine_in_structures_module():
    from swe2d.extensions.structures import HydraulicStructureEngine
    assert HydraulicStructureEngine is not None


def test_backward_compat_reexport():
    """extension_models still re-exports for any stragglers."""
    from swe2d.extensions.extension_models import DrainageCouplingEngine
    from swe2d.extensions.extension_models import HydraulicStructureEngine
    assert DrainageCouplingEngine is not None
    assert HydraulicStructureEngine is not None
