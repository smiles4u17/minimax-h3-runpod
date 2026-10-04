import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from h3_quant_safety import needs_compact_copy, compact_quant_input


class FakeTensor:
    def __init__(self, shape, strides):
        self.shape = shape
        self.strides = strides
    def stride(self):
        return self.strides
    def contiguous(self):
        b,s,h,d = self.shape
        return FakeTensor(self.shape, (s*h*d,h*d,d,1))


class QuantSafetyTests(unittest.TestCase):
    def test_real_h3_long_sequence_fused_buffer_overflows_but_compact_fits(self):
        shape = (1,205000,56,128)
        fused = (205000*21504,21504,128,1)
        self.assertTrue(needs_compact_copy(shape, fused))
        safe = compact_quant_input(FakeTensor(shape, fused))
        self.assertFalse(needs_compact_copy(safe.shape, safe.stride()))

    def test_short_fused_views_keep_zero_copy(self):
        tensor = FakeTensor((1,60000,56,128), (60000*21504,21504,128,1))
        self.assertIs(compact_quant_input(tensor), tensor)

    def test_compact_geometry_beyond_cuda_capacity_fails(self):
        with self.assertRaises(RuntimeError):
            needs_compact_copy((1,700000,56,128), (700000*21504,21504,128,1))
