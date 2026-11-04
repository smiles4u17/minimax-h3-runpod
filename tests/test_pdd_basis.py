"""CPU regressions for curve-basis rank loss and rejected-cache reuse."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from safetensors.torch import save_file

spec = importlib.util.spec_from_file_location("pdd_adaln_test", Path(__file__).parents[1] / "custom_nodes/h3_pdd/adaln.py")
adaln = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adaln)


class PDDBasisTests(unittest.TestCase):
    def test_small_valid_basis_direction_is_not_discarded(self):
        generator = torch.Generator().manual_seed(42)
        table = torch.randn(1025, 8, generator=generator)
        table[:, 0] *= 1e-5
        truth = torch.randn(8, 24, generator=generator)
        truth[0] *= 1e5
        grid = 0.3 + table @ truth
        c, v, residual = adaln.solve_basis(table, grid)
        self.assertLess(residual, 1e-5)
        self.assertEqual(c.dtype, torch.float32)
        self.assertEqual(v.dtype, torch.float32)
        torch.testing.assert_close(c + table @ v.T, grid, atol=1e-5, rtol=1e-5)

    def test_failed_fit_cache_is_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "basis.safetensors"
            save_file({"c": torch.zeros(2), "V": torch.zeros(2, 1)}, str(path),
                      metadata={"residual": "0.0957", "source": "rejected"})
            with mock.patch.object(adaln, "_cache_path", return_value=str(path)):
                self.assertIsNone(adaln._read_cache("test"))

    def test_poor_fit_is_not_written_as_cache(self):
        with mock.patch.object(adaln, "_read_cache", return_value=None), \
             mock.patch.object(adaln, "_candidate_grids", return_value=[("wrong", torch.randn(30, 4))]), \
             mock.patch.object(adaln, "_write_cache") as write:
            result = adaln.get_basis(torch.randn(30, 2))
            self.assertGreater(result[2], 5e-3)
            write.assert_not_called()
