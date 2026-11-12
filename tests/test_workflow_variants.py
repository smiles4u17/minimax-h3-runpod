"""Regression tests for dated workflow graph routing (no GPU)."""
import unittest
from unittest import mock
from test_handler import handler, asset
import test_handler
from h3_workflow_options import validate_keyframes

class VariantTests(unittest.TestCase):
    def test_legacy_workflows_skip_context_and_use_sol(self):
        for task in ('t2v', 'fl2v', 'r2v'):
            graph, meta = handler.build_preset({'task': task, 'prompt': 'A toy robot',
                'context_windows': True, 'references': [asset('robot.png')],
                'first_frame': asset('robot.png')})
            self.assertNotIn('9180', graph)
            self.assertEqual(graph['122']['inputs']['vae'], ['9181', 0])
            self.assertTrue(meta['context_windows'])

    def test_context_only_reaches_refinement(self):
        for variant in ('fflf_20260920', 'ref2v_20260920'):
            for attention in ('sage', 'native', 'sla', 'sol-attn'):
                graph, meta = self.build(variant, context_windows=True, attention=attention,
                                        latent_upscale=True, diagnostic_frames=True)
                self.assertEqual(graph['126']['inputs']['model'], graph['9180']['inputs']['model'])
                self.assertEqual(graph['124']['inputs']['model'], graph['9180']['inputs']['model'])
                self.assertEqual(graph['9306']['inputs']['model'], ['9180', 0])
                self.assertEqual(graph['9180']['inputs']['context_length'], 124)
                self.assertEqual(graph['9180']['inputs']['context_overlap'], 39)
                self.assertEqual(graph['9180']['inputs']['model'],
                                 ['9160', 0] if attention in ('sla', 'sol-attn') else
                                 ['145', 0] if attention == 'sage' else ['9162', 0])
                for node in graph.values():
                    if node['class_type'] == 'VAEDecode':
                        self.assertEqual(node['inputs']['vae'], ['9181', 0])
                conditioning = '182' if variant.startswith('fflf') else '136'
                self.assertEqual(graph[conditioning]['inputs']['vae'], ['119', 0])
                self.assertEqual(graph['121']['inputs']['vae'], ['120', 0])
                self.assertEqual(meta['video_decoder'], 'sol_int8')

    def test_context_off_original_decoder_and_fp16(self):
        graph, _ = self.build(video_decoder='original')
        self.assertNotIn('9180', graph)
        self.assertNotIn('9181', graph)
        graph, _ = self.build(video_decoder='sol_fp16')
        self.assertEqual(graph['9181']['inputs']['vae_name'], 'minimax_h3_sol_vae_fp16.safetensors')
        with self.assertRaisesRegex(handler.InputError, 'FirstBlockCache'):
            self.build(context_windows=True, cache_enabled=True)
        with self.assertRaisesRegex(handler.InputError, 'boolean'):
            self.build(context_windows='false')

    def test_sla_pairs_sparse_adapter_and_shift_without_larry(self):
        from h3_acceleration import SLA_ADAPTER
        (handler.COMFY_ROOT/'models/loras/H3'/SLA_ADAPTER).touch()
        graph, meta = self.build('fflf_20260920', acceleration_recipe='sla4',
                                latent_upscale=True, rtx_upscale=True)
        self.assertEqual(graph['9160']['inputs']['selection'], 'sla')
        self.assertEqual(graph['9161']['inputs']['shift_video'], 6)
        self.assertEqual(meta['steps'], 4)
        self.assertNotIn('154', graph)
        self.assertIn('9308', graph)
        self.assertIn('9310', graph)

    def test_pdd_full_heads_and_dense_refinement_do_not_use_larry(self):
        for variant, name in [('fflf_20260920', 'MiniMax-H3-FL2VA-Acc-8Step.safetensors'),
                              ('ref2v_20260920', 'MiniMax-H3-Ref2VA-Acc-8Step.safetensors')]:
            (handler.COMFY_ROOT/'models/loras/H3'/name).touch()
            graph, meta = self.build(variant, acceleration_recipe='pdd8',
                                    turbo_enabled=False, use_larry=False, second_pass_sigma=5,
                                    latent_upscale=True, rtx_upscale=True)
            self.assertEqual(graph['9170']['class_type'], 'H3PDDAcceleration')
            self.assertEqual(graph['9170']['inputs']['pdd_lora_name'], 'H3/' + name)
            self.assertTrue(graph['9170']['inputs']['strict_schedule'])
            self.assertEqual(graph['125']['inputs']['sigmas'], ['9170', 1])
            self.assertEqual(graph['126']['inputs']['model'], ['9170', 0])
            self.assertEqual(graph['9306']['inputs']['model'], graph['9170']['inputs']['model'])
            self.assertNotIn('154', graph)
            self.assertEqual(meta['pdd_refinement'], 'dense_base_without_pdd_heads')

    def setUp(self):
        test_handler.WorkflowTests.setUp(self)
        for name in ('minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors','minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors'):
            (handler.COMFY_ROOT/'models/loras'/name).touch()
        folder=handler.COMFY_ROOT/'models/latent_upscale_models';folder.mkdir()
        (folder/'minimax_h3_latent_upscaler_3d_bf16.safetensors').touch()
        (folder/'minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors').touch()
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

    def test_ref2v_nine_references_reach_both_passes(self):
        photos = [asset(f'view-{i}.png') for i in range(9)]
        graph, meta = self.build('ref2v_20260920', photos=photos, keyframe_positions=[],
                                 latent_upscale=True, turbo_enabled=False)
        for i in range(9):
            source = graph['136']['inputs'][f'ref_images.ref_image_{i}']
            self.assertEqual(graph[source[0]]['class_type'], 'LoadImage')
        self.assertEqual(len([key for key in graph['136']['inputs'] if key.startswith('ref_images.')]), 9)
        self.assertTrue(meta['reference_encoding_reused'])
        self.assertEqual(graph['9306']['inputs']['conditioning'], ['136', 0])
        with self.assertRaisesRegex((ValueError, handler.InputError), 'nine'):
            self.build('ref2v_20260920', photos=photos+[None])
        with self.assertRaisesRegex(ValueError, 'four'):
            self.build('fflf_20260920', photos=photos)

    def test_ref2v_empty_slots_keep_filled_reference_order(self):
        graph, _ = self.build('ref2v_20260920', photos=[None, asset('second.png')]+[None]*6+[asset('ninth.png')], keyframe_positions=[])
        refs = [key for key in graph['136']['inputs'] if key.startswith('ref_images.')]
        self.assertEqual(refs, ['ref_images.ref_image_0', 'ref_images.ref_image_1'])

    def test_native_upscaler_preserves_av_and_uses_selected_checkpoint(self):
        for variant in ('fflf_20260920', 'ref2v_20260920'):
            for precision in ('auto', 'bf16', 'fp16', 'fp32'):
                graph, meta = self.build(variant, latent_upscale_precision=precision)
                self.assertEqual(graph['9303']['class_type'], 'LatentUpscaleModelLoader')
                self.assertEqual(graph['9303']['inputs']['model_name'], 'minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')
                self.assertEqual(graph['9304'], {'class_type': 'MiniMaxH3LatentUpscaleWithModel',
                    'inputs': {'model': ['9303', 0], 'samples': ['125', 1],
                               'width': ['9302', 0], 'height': ['9302', 1]}})
                self.assertEqual(meta['latent_upscale_precision'], 'auto')
                self.assertNotIn('9301', graph)
                self.assertFalse(any(n['class_type'] in ('MinimaxH3LatentUpscaler3D',
                    'LTXVSeparateAVLatent', 'LTXVConcatAVLatent') for n in graph.values()))
        graph, _ = self.build(latent_upscale_model='minimax_h3_latent_upscaler_3d_bf16.safetensors')
        self.assertEqual(graph['9303']['inputs']['model_name'], 'minimax_h3_latent_upscaler_3d_bf16.safetensors')
        with self.assertRaisesRegex(ValueError, 'precision'):
            self.build(latent_upscale_precision='fp8')

    def test_all_sigma_choices_and_final_outputs(self):
        for variant in ('fflf_20260920','ref2v_20260920'):
            for choice in (1,2,3,4):
                graph, meta=self.build(variant,second_pass_sigma=choice)
                self.assertEqual(graph['125']['inputs']['sigmas'],['9300',0])
                self.assertEqual(graph['9304']['inputs']['samples'],['125',1])
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
        for changes in (dict(keyframe_positions=['0%','100%','20%','']),dict(use_multi_image=True,keyframe_positions=['0','100%']),dict(photos=[asset('1'),None,asset('3')]),dict(pass1_split=9),dict(pass1_split=8,second_pass_sigma=4),dict(use_multi_image=True,reference_videos=[asset('video')])):
            with self.assertRaises((handler.InputError,ValueError)):self.build(**changes)

    def test_full_first_pass_uses_original_schedule_before_refine(self):
        graph, _ = self.build('ref2v_20260920', pass1_split=8, second_pass_sigma=5)
        self.assertNotIn('9300', graph)
        self.assertEqual(graph['125']['inputs']['sigmas'], ['124', 0])
        self.assertEqual(graph['9304']['inputs']['samples'], ['125', 1])
        self.assertEqual(graph['9308']['inputs']['sigmas'], ['9307', 0])
        self.assertEqual(len(graph['9307']['inputs']['sigmas'].split(',')), 9)

    def test_baked_model_refine_preserves_clean_first_pass_for_both_tabs(self):
        for variant in ('fflf_20260920', 'ref2v_20260920'):
            for choice in (1, 2, 3, 5):
                graph, meta = self.build(variant, turbo_enabled=False, second_pass_sigma=choice)
                self.assertNotIn('9300', graph)
                self.assertEqual(graph['125']['inputs']['sigmas'], ['124', 0])
                self.assertEqual(graph['9304']['inputs']['samples'], ['125', 1])
                sigmas = [float(v) for v in graph['9307']['inputs']['sigmas'].split(',')]
                self.assertEqual(sigmas[0], .45)
                self.assertEqual(sigmas[-1], 0)
                self.assertTrue(all(a > b for a, b in zip(sigmas, sigmas[1:])))
                self.assertEqual(meta['first_pass_steps'], 8)
                self.assertEqual(meta['latent_refine_strategy'], 'clean_low_noise')
                if variant.startswith('ref2v'):
                    self.assertEqual(graph['9306']['inputs']['conditioning'], ['136', 0])
                    self.assertNotIn('9305', graph)
                    self.assertTrue(meta['reference_encoding_reused'])
                else:
                    self.assertEqual(graph['9306']['inputs']['conditioning'], ['9305', 0])
                    self.assertEqual(graph['9305']['inputs']['width'], ['9302', 0])
                    self.assertEqual(graph['9305']['inputs']['height'], ['9302', 1])

    def test_explicit_remaining_schedule_keeps_its_split_without_turbo(self):
        graph, meta = self.build('ref2v_20260920', turbo_enabled=False, second_pass_sigma=4)
        self.assertEqual(graph['125']['inputs']['sigmas'], ['9300', 0])
        self.assertEqual(graph['9308']['inputs']['sigmas'], ['9300', 1])
        self.assertEqual(meta['first_pass_steps'], 3)

    def test_diagnostic_frames_capture_both_refine_boundaries(self):
        graph, _ = self.build('ref2v_20260920', diagnostic_frames=True)
        self.assertEqual(graph['9320']['inputs']['samples'], ['125', 1])
        self.assertEqual(graph['9320']['inputs']['vae'], graph['122']['inputs']['vae'])
        self.assertEqual(graph['9322']['inputs']['images'], ['9321', 0])
        self.assertEqual(graph['9323']['inputs']['image'], ['122', 0])
        self.assertEqual(graph['9324']['inputs']['images'], ['9323', 0])
        self.assertEqual(graph['9325']['inputs']['samples'], ['9304', 0])
        self.assertEqual(graph['9327']['inputs']['images'], ['9326', 0])
        normal, _ = self.build('ref2v_20260920')
        self.assertNotIn('9320', normal)
        self.assertNotIn('9324', normal)

    def test_reference_video_resize_preserves_audio_and_can_be_disabled(self):
        graph, _ = self.build('ref2v_20260920', reference_videos=[asset('clip.mp4')])
        source = graph['136']['inputs']['ref_videos.ref_video_0']
        self.assertEqual(graph[source[0]]['class_type'], 'ImageResizeKJv2')
        self.assertEqual(graph[source[0]]['inputs']['image'], ['8200', 0])
        self.assertEqual(graph[source[0]]['inputs']['width'], ['115', 0])
        self.assertEqual(graph[source[0]]['inputs']['height'], ['115', 1])
        for key, value in dict(upscale_method='nearest-exact', keep_proportion='crop', pad_color='0,0,0',
                              crop_position='center', divisible_by=2, device='cpu').items():
            self.assertEqual(graph[source[0]]['inputs'][key], value)
        self.assertEqual(graph['136']['inputs']['ref_video_audios.ref_video_audio_0'], ['8200', 2])
        original, _ = self.build('ref2v_20260920', reference_videos=[asset('clip.mp4')], reference_resolution='original')
        self.assertEqual(original['136']['inputs']['ref_videos.ref_video_0'], ['8200', 0])
        with self.assertRaises(ValueError):
            self.build('ref2v_20260920', reference_resolution='invalid')

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
        for mode, latent, rtx in [('none',False,False),('latent',True,False),('rtx',True,True)]:
            graph,meta=self.build(output_mode=mode)
            self.assertEqual('9303' in graph,latent)
            self.assertEqual('9310' in graph,rtx)
            if rtx:
                self.assertEqual(graph['122']['inputs']['samples'],['9308',0])
                self.assertEqual(graph['9310']['inputs']['images'],['122',0])
                self.assertEqual(graph['9310']['inputs']['resize_type.width'],1920)
                self.assertEqual(graph['9310']['inputs']['resize_type.height'],1080)
            self.assertEqual(meta['final_megapixels'],1.0)
            self.assertEqual(graph['130']['inputs']['images'],['9310',0] if rtx else ['122',0])
        with self.assertRaises(handler.InputError):self.build(output_mode='invalid')

    def test_no_turbo_and_independent_rtx(self):
        for variant in ('fflf_20260920','ref2v_20260920'):
            for larry in (False,True):
                graph,meta=self.build(variant,turbo_enabled=False,use_larry=larry,latent_upscale=False,rtx_upscale=True)
                self.assertNotIn('152',graph)
                self.assertNotIn('9303',graph)
                sampler=graph['125']['inputs']['sampler'][0]
                self.assertNotIn('Turbo',graph[sampler]['class_type'])
                self.assertFalse(meta['turbo_enabled'])
                self.assertEqual(graph['9310']['inputs']['images'],['122',0])

    def test_percent_boundaries(self):
        for value in ('0%','10%','15%','100%','99.5%'):validate_keyframes(['image'],[value],True)
        for value in ('-1%','101%','0.5','NaN%',''):
            with self.assertRaises(ValueError):validate_keyframes(['image'],[value],True)
