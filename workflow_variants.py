"""Executable adaptation of the user's September 20 two-pass workflows.
Editor Set/Get, bypassers and switches are resolved here, not sent to ComfyUI.
"""
import copy
import re
from h3_workflow_options import workflow_options, validate_keyframes

SIGMAS = {1: "0.9035, 0.6316, 0.3158, 0.0000",
          2: "0.9035, 0.8000, 0.6316, 0.3158, 0.0000",
          3: "0.9231, 0.8780, 0.8000, 0.6316, 0.3158, 0.0000"}

def apply_variant(graph, payload, materialize, model_name):
    options = workflow_options(payload)
    if options['workflow_variant'] == 'legacy':
        return options
    multi = options['use_multi_image']
    photos = payload.get('photos', [])
    positions = validate_keyframes(photos, payload.get('keyframe_positions', []), multi)
    conditioning = '182' if payload['task'] == 'fl2v' else '136'
    def add(n, kind, **inputs):
        graph[n] = {'class_type': kind, 'inputs': inputs}
        return [n, 0]
    if multi:
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
    elif payload['task'] == 'fl2v' and any(photos[2:]):
        raise ValueError('Photo 3 and Photo 4 require Use Multi IMG')
    final_latent = ['125', 0]
    if options['latent_upscale']:
        split = options['pass1_split']
        if split >= int(payload.get('steps', 6)):
            raise ValueError('First-pass split must be less than total schedule steps')
        sigmas = copy.deepcopy(graph['125']['inputs']['sigmas'])
        add('9300', 'SplitSigmas', sigmas=sigmas, step=split)
        graph['125']['inputs']['sigmas'] = ['9300', 0]
        add('9301', 'LTXVSeparateAVLatent', av_latent=['125', 1])
        add('9302', 'ResolutionSelector', aspect_ratio=payload.get('aspect_ratio','16:9 (Widescreen)'), megapixels=options['final_megapixels'], multiple=32)
        name = model_name({'upscaler':options['latent_upscale_model']}, 'upscaler', 'latent_upscale_models', options['latent_upscale_model'])
        add('9303', 'MinimaxH3LatentUpscaler3D', latent=['9301',0], model_name=name,
            mode='target dimensions', **{'mode.width':['9302',0], 'mode.height':['9302',1]},
            align=32, enable_temporal_chunking=True, force_unload=True, device='cuda', precision='bf16')
        add('9304', 'LTXVConcatAVLatent', video_latent=['9303',0], audio_latent=['9301',1])
        graph['9305'] = copy.deepcopy(graph[conditioning])
        graph['9305']['inputs'].update(width=['9302',0], height=['9302',1])
        # FirstBlockCache only belongs to pass one. Share all preceding LoRAs/attention.
        model = graph['126']['inputs']['model']
        if graph[model[0]]['class_type'] == 'MiniMaxH3FirstBlockCache':
            model = graph[model[0]]['inputs']['model']
        add('9306', 'BasicGuider', model=model, conditioning=['9305',0])
        mode = options['second_pass_sigma']
        second_sigmas = ['9300',1] if mode == 4 else add('9307','ManualSigmas',sigmas=SIGMAS[mode])
        final_latent = add('9308', 'SamplerCustomAdvanced', noise=['129',0], guider=['9306',0],
            sampler=graph['125']['inputs']['sampler'], sigmas=second_sigmas, latent_image=['9304',0])
        graph['121']['inputs']['samples'] = final_latent
        graph['122']['inputs']['samples'] = final_latent
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
    last = add('9311','ImageFromBatch',image=images,batch_index=-1,length=1)
    add('9312','SaveImage',images=last,filename_prefix='Last_Frame/H3')
    return options
