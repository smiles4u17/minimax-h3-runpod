import unittest
from h3_attention_profiles import apply_profile, validate_attention, catalog, SLA_MODEL, SLA_ADAPTER


class AttentionProfileTests(unittest.TestCase):
    def test_profile_only_changes_attention_controls(self):
        source = {'attention_profile': 'sol', 'duration': 15, 'steps': 8, 'latent_upscale': True, 'rtx_upscale': True, 'cache_enabled': True}
        result = apply_profile(source)
        self.assertEqual(result['attention'], 'sol-attn')
        self.assertFalse(result['cache_enabled'])
        for field in ('duration', 'steps', 'latent_upscale', 'rtx_upscale'):
            self.assertEqual(result[field], source[field])
        self.assertEqual(len(catalog()['cards']), 3)

    def test_missing_or_wrong_sparse_weights_never_pass_on_checkbox_alone(self):
        with self.assertRaisesRegex(ValueError, 'paired'):
            validate_attention({'attention': 'sla', 'sparse_trained_weights': True})
        with self.assertRaisesRegex(ValueError, 'VSA'):
            validate_attention({'attention': 'vsa', 'sparse_trained_weights': True})
        valid = {'attention': 'sla', 'task': 'fl2v', 'model': SLA_MODEL,
                 'loras': [{'name': 'H3/'+SLA_ADAPTER, 'strength': 1}], 'sparse_trained_weights': True,
                 'turbo_enabled': False, 'sampler': 'euler', 'scheduler': 'simple', 'steps': 4}
        validate_attention(valid, (8, 9), True)
        for override in ({'task': 'r2v'}, {'model': '10Eros_hybrid.safetensors'}, {'steps': 8}, {'cache_enabled': True}):
            with self.assertRaises(ValueError):
                validate_attention({**valid, **override}, (12, 0), True)
        with self.assertRaisesRegex(ValueError, 'no compiled'):
            validate_attention({'attention': 'sol-attn'}, (8, 9), False)


if __name__ == '__main__':
    unittest.main()
