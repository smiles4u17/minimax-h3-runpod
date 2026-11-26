"""Executable adaptation of the user's September 20 two-pass workflows.
Editor Set/Get, bypassers and switches are resolved here, not sent to ComfyUI.
"""
import copy
import re
from h3_workflow_options import first_pass_steps, format_refine_sigmas, workflow_options, validate_keyframes, validate_reference_photos

SIGMAS = {1: "0.9035, 0.6316, 0.3158, 0.0000",
          2: "0.9035, 0.8000, 0.6316, 0.3158, 0.0000",
          3: "0.9231, 0.8780, 0.8000, 0.6316, 0.3158, 0.0000",
          5: "0.9035, 0.8500, 0.8000, 0.7200, 0.6316, 0.5000, 0.3158, 0.1500, 0.0000"}

# A baked model without the separate Turbo adapter needs a clean source and
# a refinement schedule, not the adapter's near-full-noise restart schedule.
REFINE_SIGMAS = {1: "0.4500, 0.3000, 0.1500, 0.0000",
                 2: "0.4500, 0.3375, 0.2250, 0.1125, 0.0000",
                 3: "0.4500, 0.3600, 0.2700, 0.1800, 0.0900, 0.0000",
                 5: "0.4500, 0.39375, 0.3375, 0.28125, 0.2250, 0.16875, 0.1125, 0.05625, 0.0000"}

def crop_input_images(graph, conditioning, first_node_id):
    inputs = graph[conditioning]['inputs']
    image_keys = [key for key in inputs if key in ('first_frame', 'last_frame')
                  or key.startswith('ref_images.ref_image_') or re.fullmatch(r'image_\d+', key)]
    for index, key in enumerate(image_keys):
        source = inputs[key]
        original = graph[source[0]]
        if original['class_type'] == 'ImageResizeKJv2':
            source = original['inputs']['image']
        node_id = str(first_node_id + index)
        graph[node_id] = {'class_type': 'ImageResizeKJv2', 'inputs': {
            'image': source, 'width': inputs['width'], 'height': inputs['height'],
            'upscale_method': 'lanczos', 'keep_proportion': 'crop', 'crop_position': 'center',
            'pad_color': '0, 0, 0', 'divisible_by': 2, 'device': 'cpu',
        }}
        inputs[key] = [node_id, 0]


def apply_variant(graph, payload, materialize, model_name):
    options = workflow_options(payload)
    legacy = options['workflow_variant'] == 'legacy'
    conditioning = '182' if payload['task'] in ('fl2v', 't2v') else '136'
    crop_images = payload.get('crop_input_images') is True and not payload.get('masked_edit')
    if legacy and 'pass1_split_enabled' not in options:
        if crop_images:
            crop_input_images(graph, conditioning, 9500)
        return options
    multi = options['use_multi_image']
    photos = payload.get('photos', [])
    if not legacy:
        positions = (validate_reference_photos(photos) if payload['task'] == 'r2v' and not multi
                     else validate_keyframes(photos, payload.get('keyframe_positions', []), multi))
    def add(n, kind, **inputs):
        graph[n] = {'class_type': kind, 'inputs': inputs}
        return [n, 0]
    if not legacy and graph[conditioning]['class_type'] == 'MiniMaxH3ReferenceToVideo' and options['reference_resolution'] == 'generation':
        for index, (name, source) in enumerate(list(graph[conditioning]['inputs'].items())):
            if name.startswith('ref_videos.'):
                graph[conditioning]['inputs'][name] = add(str(9450+index), 'ImageResizeKJv2', image=source,
                    width=graph[conditioning]['inputs']['width'], height=graph[conditioning]['inputs']['height'],
                    upscale_method='nearest-exact', keep_proportion='crop', pad_color='0,0,0',
                    crop_position='center', divisible_by=2, device='cpu')
        options['reference_resize'] = 'kjv2_nearest_exact_center_crop_cpu'
    if not legacy and multi:
        if payload.get('reference_videos') or payload.get('reference_audios') or payload.get('audio'):
            raise ValueError('Percentage keyframes cannot be combined with video/audio references; disable Use Multi IMG')
        if not any(photos):
            raise ValueError('Multi-image requires at least one photo')
        inputs = {k: copy.deepcopy(graph[conditioning]['inputs'][k]) for k in ('clip','vae','prompt','width','height','length')}
        active = [(i, image) for i, image in enumerate(photos) if image]
        inputs['positions'] = ', '.join(positions[i] for i, _ in active)
        for slot, (original, image) in enumerate(active, 1):
            inputs[f'image_{slot}'] = add(str(9400+slot), 'LoadImage', image=materialize(image, f'photo{original+1}.png'))
        graph[conditioning] = {'class_type': 'H3Keyframes', 'inputs': inputs}
    elif not legacy and payload['task'] == 'fl2v' and any(photos[2:]):
        raise ValueError('Photo 3 and Photo 4 require Use Multi IMG')
    final_latent = ['125', 0]
    upscale = options['latent_upscale']
    if upscale and payload.get('masked_edit'):
        raise ValueError('SAMimate refinement uses the original canvas to preserve the tracked mask')
    if upscale or options['refine_enabled']:
        steps = int(payload.get('steps', 6))
        effective_split = first_pass_steps(options, steps, payload.get('acceleration_recipe'))
        clean_refine = not options['turbo_enabled'] and options['second_pass_sigma'] != 4
        if effective_split < steps:
            sigmas = copy.deepcopy(graph['125']['inputs']['sigmas'])
            add('9300', 'SplitSigmas', sigmas=sigmas, step=effective_split)
            graph['125']['inputs']['sigmas'] = ['9300', 0]
        # With no spatial upscale, SamplerCustomAdvanced carries the original noise mask.
        refine_latent = ['125', 1]
        if upscale:
            add('9302', 'ResolutionSelector', aspect_ratio=payload.get('aspect_ratio','16:9 (Widescreen)'), megapixels=options['final_megapixels'], multiple=32)
            name = model_name({'upscaler':options['latent_upscale_model']}, 'upscaler', 'latent_upscale_models', options['latent_upscale_model'])
            add('9303', 'LatentUpscaleModelLoader', model_name=name)
            refine_latent = add('9304', 'MiniMaxH3LatentUpscaleWithModel', model=['9303', 0],
                samples=refine_latent, width=['9302', 0], height=['9302', 1])
        # Ref2V stores independent reference blocks, not target-shaped hints.
        # Reuse their VAE/Qwen encoding. FL2V/keyframes have spatial masks and
        # still need rebuilding at the refinement resolution.
        reuse_references = graph[conditioning]['class_type'] == 'MiniMaxH3ReferenceToVideo'
        second_conditioning = [conditioning, 0]
        if upscale and not reuse_references:
            graph['9305'] = copy.deepcopy(graph[conditioning])
            graph['9305']['inputs'].update(width=['9302',0], height=['9302',1])
            if crop_images:
                crop_input_images(graph, '9305', 9520)
            second_conditioning = ['9305', 0]
        # FirstBlockCache only belongs to pass one. Share all preceding LoRAs/attention.
        model = graph['126']['inputs']['model']
        if graph[model[0]]['class_type'] == 'H3PDDAcceleration':
            model = graph[model[0]]['inputs']['model']
        if graph[model[0]]['class_type'] == 'MiniMaxH3FirstBlockCache':
            model = graph[model[0]]['inputs']['model']
        if options['context_windows']:
            model = add('9180', 'MiniMaxH3ContextWindows', model=model,
                        context_length=options['context_length'], context_overlap=options['context_overlap'],
                        fuse_method=options['context_fuse_method'], freenoise=options['context_freenoise'],
                        split_conds_to_windows=options['context_split_conds'])
        add('9306', 'BasicGuider', model=model, conditioning=second_conditioning)
        mode = options['second_pass_sigma']
        schedules = REFINE_SIGMAS if clean_refine else SIGMAS
        values = schedules.get(mode)
        if mode == 6:
            count = options['refine_steps']
            values = format_refine_sigmas(options['refine_start_sigma'] * (count - i) / count for i in range(count + 1))
        elif mode == 7:
            values = options['refine_sigmas']
        second_sigmas = ['9300',1] if mode == 4 else add('9307','ManualSigmas',sigmas=values)
        final_latent = add('9308', 'SamplerCustomAdvanced', noise=['129',0], guider=['9306',0],
            sampler=graph['125']['inputs']['sampler'], sigmas=second_sigmas, latent_image=refine_latent)
        graph['121']['inputs']['samples'] = final_latent
        graph['122']['inputs']['samples'] = final_latent
        options.update(first_pass_steps=effective_split,
                       reference_encoding_reused=reuse_references,
                       latent_refine_strategy='clean_low_noise' if clean_refine else 'split_schedule',
                       second_pass_sigmas=values if values is not None else 'remaining')
        if payload.get('pdd_enabled'):
            options['pdd_refinement'] = 'dense_base_without_pdd_heads'
    if crop_images:
        crop_input_images(graph, conditioning, 9500)
    images = ['122',0]
    if options['rtx_upscale']:
        ratio = str(payload.get('aspect_ratio','16:9')).split()[0]
        match = re.fullmatch(r'(\d+):(\d+)',ratio)
        if not match:
            raise ValueError('RTX output needs a numeric aspect ratio such as 16:9')
        w,h = map(int,match.groups())
        if not w or not h: raise ValueError('Invalid aspect ratio')
        width,height = (1920, round(1920*h/w/8)*8) if w>=h else (round(1920*w/h/8)*8,1920)
        images = add('9310','RTXVideoSuperResolution',images=images,resize_type='target dimensions',
            **{'resize_type.width':width,'resize_type.height':height},quality='ULTRA')
    graph['130']['inputs']['images'] = images
    if not legacy:
        last = add('9311','ImageFromBatch',image=images,batch_index=-1,length=1)
        add('9312','SaveImage',images=last,filename_prefix='Last_Frame/H3')
    if payload.get('diagnostic_frames') is True:
        if options['latent_upscale']:
            first_images = add('9320','VAEDecode',samples=['125',1],vae=graph['122']['inputs']['vae'])
            first_last = add('9321','ImageFromBatch',image=first_images,batch_index=-1,length=1)
            add('9322','SaveImage',images=first_last,filename_prefix='Diagnostic/H3_First_Pass')
            learned_images = add('9325','VAEDecode',samples=['9304',0],vae=graph['122']['inputs']['vae'])
            learned_last = add('9326','ImageFromBatch',image=learned_images,batch_index=-1,length=1)
            add('9327','SaveImage',images=learned_last,filename_prefix='Diagnostic/H3_After_Learned_Upscale')
        pre_rtx_last = add('9323','ImageFromBatch',image=['122',0],batch_index=-1,length=1)
        add('9324','SaveImage',images=pre_rtx_last,filename_prefix='Diagnostic/H3_Pre_RTX')
    return options
