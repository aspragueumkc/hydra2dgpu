import unittest

from swe2d.extensions.extension_models import HydraulicStructure, HydraulicStructureConfig, StructureType
from swe2d.runtime.run_options_builder import SWE2DRunOptionsBuilder


class TestSWE2DRunOptionsBuilder(unittest.TestCase):
    def test_has_bridge_structures_detects_bridge_entries(self):
        bridge_cfg = HydraulicStructureConfig(
            enabled=True,
            structures=[
                HydraulicStructure(
                    structure_id="B0",
                    structure_type=StructureType.BRIDGE,
                    upstream_cell=0,
                    downstream_cell=1,
                    crest_elev=0.0,
                )
            ],
        )
        other_cfg = HydraulicStructureConfig(
            enabled=True,
            structures=[
                HydraulicStructure(
                    structure_id="W0",
                    structure_type=StructureType.WEIR,
                    upstream_cell=1,
                    downstream_cell=0,
                    crest_elev=0.0,
                )
            ],
        )

        self.assertTrue(SWE2DRunOptionsBuilder._has_bridge_structures(bridge_cfg))
        self.assertFalse(SWE2DRunOptionsBuilder._has_bridge_structures(other_cfg))


if __name__ == "__main__":
    unittest.main()