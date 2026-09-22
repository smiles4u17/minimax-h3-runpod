"""Regression tests for dated workflow graph routing (no GPU)."""
import unittest
from unittest import mock
from test_handler import handler, asset
import test_handler
from h3_workflow_options import validate_keyframes

class VariantTests(unittest.TestCase):
    def setUp(self):
        test_handler.WorkflowTests.setUp(self)
        for name in ('minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors','minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors'):
            (handler.COMFY_ROOT/'models/loras'/name).touch()
        folder=handler.COMFY_ROOT/'models/latent_upscale_models';folder.mkdir()
        (folder/'minimax_h3_latent_upscaler_3d_bf16.safetensors').touch()
    tearDown = test_handler.WorkflowTests.tearDown
    def build(self, variant='fflf_20260920', **changes):
        data=dict(task='fl2v_20260920' if variant.startswith('fflf') else 'r2v_20260920',
                  workflow_variant=variant, prompt='A toy robot', steps=8, pass1_split=3,
                  photos=[asset('one.png'),asset('two.png'),None,None], sampler='euler',
                  keyframe_positions=['0%','100%','',''], cache_enabled=False,
                  turbo_lora='minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors' if variant.startswith('fflf') else 'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors')
        data.update(changes)
        with mock.patch.object(handler,'_materialize_asset',return_value='fixture.png'):
            return handler.build_preset(data)

    def test_all_sigma_choices_and_final_outputs(self):
        for variant in ('fflf_20260920','ref2v_20260920'):
            for choice in (1,2,3,4):
                graph, meta=self.build(variant,second_pass_sigma=choice)
                self.assertEqual(graph['125']['inputs']['sigmas'],['9300',0])
                self.assertEqual(graph['9301']['inputs']['av_latent'],['125',1])
                self.assertEqual(graph['9308']['inputs']['sigmas'],['9300',1] if choice==4 else ['9307',0])
                if choice!=4:self.assertEqual(len(graph['9307']['inputs']['sigmas'].split(',')),choice+3)
                self.assertEqual(graph['122']['inputs']['samples'],['9308',0])
                self.assertEqual(graph['130']['inputs']['images'],['9310',0])
                self.assertEqual(graph['9311']['inputs']['batch_index'],-1)
                self.assertEqual(graph['9310']['inputs']['resize_type.width'],1920)
                self.assertEqual(graph['9310']['inputs']['resize_type.height'],1080)
                self.assertEqual(meta['workflow_variant'],variant)

    def test_upscale_disabled_removes_both_dependencies(self):
        graph,_=self.build(latent_upscale=False,rtx_upscale=False)
        self.assertNotIn('9303',graph);self.assertNotIn('9310',graph)
        self.assertEqual(graph['130']['inputs']['images'],['122',0])
        self.assertEqual(graph['125']['inputs']['sigmas'],['124',0])

    def test_four_keyframes_and_second_pass_conditioning(self):
        graph,_=self.build(use_multi_image=True,photos=[asset('1'),asset('2'),None,asset('4')],keyframe_positions=['0%','10%','','100%'])
        for node in ('182','9305'):
            self.assertEqual(graph[node]['class_type'],'H3Keyframes')
            self.assertEqual(graph[node]['inputs']['positions'],'0%, 10%, 100%')
            self.assertIn('image_3',graph[node]['inputs'])
            self.assertNotIn('image_4',graph[node]['inputs'])
        self.assertEqual(graph['9305']['inputs']['width'],['9302',0])

    def test_reject_invalid_slots_and_incompatible_references(self):
        for changes in (dict(keyframe_positions=['0%','100%','20%','']),dict(use_multi_image=True,keyframe_positions=['0','100%']),dict(photos=[asset('1'),None,asset('3')]),dict(pass1_split=8),dict(use_multi_image=True,reference_videos=[asset('video')])):
            with self.assertRaises((handler.InputError,ValueError)):self.build(**changes)

    def test_larry_uses_dedicated_sampler_both_passes(self):
        graph,_=self.build(use_larry=True,turbo_lora='H3/minimax_h3_turbo_v4_step600_ema.safetensors')
        first=graph['125']['inputs']['sampler'];self.assertEqual(graph['9308']['inputs']['sampler'],first)
        self.assertIn('Turbo',graph[first[0]]['class_type'])

    def test_legacy_unchanged_and_fail_closed_task(self):
        graph,meta=handler.build_preset(dict(task='t2v',prompt='A toy robot'))
        self.assertEqual(meta['workflow_variant'],'legacy');self.assertNotIn('9312',graph)
        with self.assertRaises(handler.InputError):handler.build_preset(dict(task='fl2v_20260920',prompt='robot'))

    def test_unused_first_photo_can_be_empty_in_multi_image(self):
        graph,_=self.build(use_multi_image=True,photos=[None,None,asset('3'),None],keyframe_positions=['','','15%',''])
        self.assertEqual(graph['182']['inputs']['positions'],'15%')
        self.assertIn('image_1',graph['182']['inputs'])
        self.assertNotIn('image_2',graph['182']['inputs'])

    def test_three_output_modes(self):
        for mode, latent, rtx in [('none',False,False),('latent',True,False),('rtx',False,True)]:
            graph,meta=self.build(output_mode=mode)
            self.assertEqual('9303' in graph,latent)
            self.assertEqual('9310' in graph,rtx)
            self.assertEqual(meta['final_megapixels'],1.0)
            self.assertEqual(graph['130']['inputs']['images'],['9310',0] if rtx else ['122',0])
        with self.assertRaises(handler.InputError):self.build(output_mode='invalid')

    def test_percent_boundaries(self):
        for value in ('0%','10%','15%','100%','99.5%'):validate_keyframes(['image'],[value],True)
        for value in ('-1%','101%','0.5','NaN%',''):
            with self.assertRaises(ValueError):validate_keyframes(['image'],[value],True)
