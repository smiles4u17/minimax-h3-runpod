"""Volume-only model paths and read-only runtime inventory. Never downloads models."""
import os
from pathlib import Path
import folder_paths
from aiohttp import web
from server import PromptServer
import comfy.samplers
import torch

ROOT = Path(os.environ.get('RUNPOD_VOLUME_ROOT', '/runpod-volume')).resolve()
ATTENTION_VALIDATION = None
CATEGORIES = {
    'checkpoints': ['checkpoints'], 'diffusion_models': ['diffusion_models', 'unet'],
    'text_encoders': ['text_encoders', 'clip'], 'vae': ['vae'], 'loras': ['loras'],
    'latent_upscale_models': ['latent_upscale_models'], 'clip_vision': ['clip_vision'], 'upscale_models': ['upscale_models'],
}
for category, subdirs in CATEGORIES.items():
    extensions = folder_paths.folder_names_and_paths.get(category, ([], folder_paths.supported_pt_extensions))[1]
    folder_paths.folder_names_and_paths[category] = ([str(ROOT / 'models' / d) for d in subdirs], extensions)


@PromptServer.instance.routes.get('/h3/runtime')
async def runtime(request):
    paths = {k: folder_paths.get_folder_paths(k) for k in CATEGORIES}
    volume_only = all(Path(p).resolve().is_relative_to(ROOT) for values in paths.values() for p in values)
    return web.json_response({'volume_root': str(ROOT), 'volume_mounted': os.path.ismount(ROOT),
                              'volume_only': volume_only, 'model_paths': paths,
                              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                              'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                              'cuda_capability': list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
                              'attention_validation': ATTENTION_VALIDATION,
                              'samplers': ['h3_turbo', *comfy.samplers.SAMPLER_NAMES],
                              'schedulers': comfy.samplers.SCHEDULER_NAMES})


@PromptServer.instance.routes.post('/h3/resolve-models')
async def resolve_models(request):
    data = await request.json()
    result = []
    for model in data.get('models', []):
        category, name = model['category'], model['name']
        if category not in CATEGORIES:
            return web.json_response({'error': 'Unknown model category'}, status=400)
        path = folder_paths.get_full_path(category, name)
        if not path or not Path(path).resolve().is_relative_to(ROOT):
            return web.json_response({'error': f'Model missing from network volume: {category}/{name}'}, status=400)
        result.append({'category': category, 'name': name, 'path': str(Path(path).resolve()), 'size': Path(path).stat().st_size})
    return web.json_response({'models': result})

from .reference_scale import H3ReferenceVideoScale
NODE_CLASS_MAPPINGS = {'H3ReferenceVideoScale': H3ReferenceVideoScale}


@PromptServer.instance.routes.post('/h3/attention-check')
async def attention_check(request):
    global ATTENTION_VALIDATION
    from .attention_probe import probe, smoke
    data = await request.json()
    mode = data.get('mode', 'address')
    if mode not in {'address', 'smoke'}:
        return web.json_response({'error': 'mode must be address or smoke'}, status=400)
    try:
        result = smoke() if mode == 'smoke' else probe()
        ATTENTION_VALIDATION = result
        return web.json_response(result)
    finally:
        torch.cuda.empty_cache()
