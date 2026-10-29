import unittest
from h3_acceleration import recipe, PDD_FILES, BASE_MODELS, SLA_ADAPTER


class RecipeTests(unittest.TestCase):
    def test_custom_preserves_lcm_and_user_settings(self):
        source = {'sampler': 'lcm', 'steps': 8, 'turbo_enabled': False,
                  'latent_upscale': True, 'rtx_upscale': True, 'model': 'hybrid.safetensors'}
        self.assertEqual(recipe(source), source)

    def test_pdd_uses_original_heads_and_matching_base(self):
        for task in ('fl2v', 'r2v'):
            result = recipe({'acceleration_recipe': 'pdd8', 'task': task,
                             'sampler': 'lcm', 'attention': 'sage'})
            self.assertEqual(result['model'], BASE_MODELS[task])
            self.assertEqual(result['pdd_lora'], 'H3/' + PDD_FILES[task])
            self.assertEqual(result['sampler'], 'euler')
            self.assertTrue(result['pdd_enabled'])
            self.assertFalse(result['turbo_enabled'])
            self.assertNotIn('loras', result)

    def test_sla_has_trained_shifts_and_sparse_ratio(self):
        result = recipe({'acceleration_recipe': 'sla4', 'task': 'fl2v'})
        self.assertEqual((result['shift_video'], result['shift_audio']), (6., 3.))
        self.assertEqual(result['sparse_keep_percent'], 15.)
        self.assertEqual(result['loras'][-1]['name'], 'H3/' + SLA_ADAPTER)
        self.assertEqual(result['steps'], 4)

    def test_reject_incompatible_modes(self):
        for source in [dict(acceleration_recipe='sla4', task='r2v'),
                       dict(acceleration_recipe='pdd8', attention='sla'),
                       dict(acceleration_recipe='pdd8', second_pass_sigma=4)]:
            with self.assertRaises(ValueError):
                recipe(source)
