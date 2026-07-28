import unittest
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

class _PytestStyleWrapper(unittest.TestCase):
    """Auto-generated wrapper for module-level test functions.

    Created by tools/wrap_pytest_style.py so that pytest-style tests
    (def test_* at module level) become visible to `python3 -m unittest`.
    Each module-level test is attached as a staticmethod so it can be
    discovered and run as a unittest TestCase.
    """
__wrapped_funcs = []
for _name, _obj in list(globals().items()):
    if _name.startswith("test_") and callable(_obj) and not isinstance(_obj, type):
        setattr(_PytestStyleWrapper, _name, staticmethod(_obj))
        __wrapped_funcs.append(_name)
for _name in __wrapped_funcs:
    del globals()[_name]
del _name, _obj, __wrapped_funcs
