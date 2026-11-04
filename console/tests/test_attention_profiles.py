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

    def test_generic_sla_preserves_ref2v_and_10eros_settings(self):
        for model in ('minimax_h3_ref2va_pruned_int8_convrot.safetensors',
                      '10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors'):
            source = {'attention_profile': 'sla', 'task': 'r2v', 'model': model,
                      'sampler': 'lcm', 'steps': 8, 'loras': [], 'cache_enabled': True}
            result = apply_profile(source)
            validate_attention(result, (12, 0), True)
            self.assertEqual(result['attention'], 'sla')
            self.assertFalse(result['cache_enabled'])
            for key in ('model', 'sampler', 'steps', 'loras'):
                self.assertEqual(result[key], source[key])

    def test_missing_or_wrong_sparse_weights_never_pass_on_checkbox_alone(self):
        validate_attention({'attention': 'sla', 'sparse_trained_weights': False})
        with self.assertRaisesRegex(ValueError, 'VSA'):
            validate_attention({'attention': 'vsa', 'sparse_trained_weights': True})
        valid = {'attention': 'sla', 'acceleration_recipe': 'sla4', 'task': 'fl2v', 'model': SLA_MODEL,
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
