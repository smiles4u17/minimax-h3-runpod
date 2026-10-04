import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'custom_nodes/h3_runtime'))
from reference_geometry import reference_size


class ReferenceGeometryTests(unittest.TestCase):
    def test_large_portrait_reference_fits_landscape_generation_area(self):
        w, h = reference_size(720, 1280, 864*480)
        self.assertLessEqual(w*h, 864*480)
        self.assertLess(abs(w/h - 720/1280), .04)
        self.assertEqual(w%32, 0)
        self.assertEqual(h%32, 0)

    def test_small_reference_is_not_enlarged(self):
        self.assertEqual(reference_size(608,352,1280*736), (608,352))
