import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch


class Nested:
    def __init__(self, tensors):
        self.tensors = tensors


class Model:
    def forward(self, denoise_mask=None):
        pass


class MaskTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('samimate_node', Path(__file__).parents[1] / 'custom_nodes/samimate_h3/__init__.py')
        self.module = importlib.util.module_from_spec(spec)
        self.patch = patch.dict(sys.modules, {
            'comfy.nested_tensor': types.SimpleNamespace(NestedTensor=Nested),
            'comfy.ldm.minimax.model': types.SimpleNamespace(MiniMaxH3Model=Model),
        })
        self.patch.start()
        self.addCleanup(self.patch.stop)
        spec.loader.exec_module(self.module)

    def test_thin_moving_mask_uses_causal_groups_and_token_union(self):
        mask = torch.zeros(22, 64, 64)
        mask[4, 1, 1] = 1
        mask[17, 40, 40] = 1
        result = self.module.causal_mask(mask, 7, 4, 4)
        self.assertEqual(tuple(result.shape), (1, 1, 7, 4, 4))
        self.assertEqual(result[0, 0, 1, :2, :2].sum().item(), 4)
        self.assertEqual(result[0, 0, 5, 2:, 2:].sum().item(), 4)
        self.assertEqual(result.sum().item(), 8)
        with self.assertRaises(ValueError):
            self.module.causal_mask(mask[:-1], 7, 4, 4)

    def test_source_encoded_not_empty_and_padding_does_not_stretch(self):
        target = {'samples': Nested((torch.zeros(1, 24, 7, 4, 4), torch.zeros(1, 32, 2, 37)))}
        source = torch.ones(20, 64, 64, 3)
        mask = torch.zeros_like(source)
        mask[-1] = 1
        def encode(pixels):
            self.assertEqual(len(pixels), 22)
            return torch.full((1, 24, 7, 4, 4), 3.0)
        result, = self.module.SAMimateH3MaskedTarget().encode(target, source, mask, types.SimpleNamespace(encode=encode))
        self.assertEqual(result['samples'].tensors[0].mean().item(), 3)
        self.assertEqual(result['noise_mask'].tensors[0][0, 0, -1].min().item(), 1)
        with self.assertRaises(ValueError):
            self.module.SAMimateH3MaskedTarget().encode(target, source, mask[:-1], types.SimpleNamespace(encode=encode))
