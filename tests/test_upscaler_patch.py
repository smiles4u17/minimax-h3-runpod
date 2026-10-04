import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from patch_latent_upscaler import patch


class TransferPatchTests(unittest.TestCase):
    def test_latent_and_cached_model_are_ready_on_cpu(self):
        original = 'out = out.to(device="cpu", dtype=orig_dtype, non_blocking=True)\nmodel.to("cpu", non_blocking=True)'
        result = patch(original)
        self.assertNotIn('non_blocking=True', result)
        self.assertEqual(result.count('non_blocking=False'), 2)

    def test_pinned_source_drift_fails_closed(self):
        with self.assertRaises(RuntimeError):
            patch('unexpected upstream source')
