"""Shared attention recipes and compatibility checks; actual GPU wins on worker."""
from pathlib import PurePosixPath

CARDS = {
    'rtx4090': {'label': 'RTX 4090 · 24 GB', 'capability': [8, 9], 'encoder': 'INT8', 'default': 'auto'},
    'rtx5090': {'label': 'RTX 5090 · 32 GB', 'capability': [12, 0], 'encoder': 'NVFP4', 'default': 'auto'},
    'pro4500': {'label': 'RTX PRO 4500 Blackwell · 32 GB', 'capability': [12, 0], 'encoder': 'NVFP4', 'default': 'auto'},
}
SLA_ADAPTER = 'minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors'
SLA_MODEL = 'minimax_h3_fl2va_pruned_int8_convrot.safetensors'
PROFILES = {
    'custom': {'label': 'Custom attention settings', 'attention': None, 'evidence': 'User configuration'},
    'auto': {'label': 'Auto · GPU-adaptive Sage / native', 'attention': 'auto',
             'evidence': 'Supported dense default; not a claim of fastest performance'},
    'sage': {'label': 'Sage · dense baseline', 'attention': 'sage',
             'evidence': 'SM89 and SM120 builds; GPU validation recorded separately'},
    'sol': {'label': 'Sol-Attn · training-free candidate', 'attention': 'sol-attn',
            'evidence': 'Native Comfy kernel; benchmark against dense for this workload'},
    'sla': {'label': 'SLA · paired FL2V weights only', 'attention': 'sla',
            'evidence': 'Requires the converted SLA adapter and standard FL2VA checkpoint',
            'required_model': SLA_MODEL, 'required_adapter': SLA_ADAPTER},
    'vsa': {'label': 'VSA · unavailable (weights not validated)', 'attention': 'vsa',
            'evidence': 'No compatible native-worker VSA checkpoint has been validated', 'disabled': True},
}


def apply_profile(data):
    result = dict(data)
    profile_id = str(data.get('attention_profile') or 'custom')
    if profile_id not in PROFILES:
        raise ValueError('Unknown H3 attention profile')
    profile = PROFILES[profile_id]
    if profile.get('disabled'):
        raise ValueError(profile['evidence'])
    if profile['attention'] is not None:
        result['attention'] = profile['attention']
        result['cache_enabled'] = False
    if profile_id == 'sol':
        result.setdefault('sparse_tau', 1.3)
        result.setdefault('sparse_start_percent', 0.2)
        result.setdefault('sparse_end_percent', 1.0)
    result['attention_profile'] = profile_id
    return result


def validate_attention(data, capability=None, sparse_available=None):
    mode = str(data.get('attention') or 'auto')
    if mode not in {'sol-attn', 'sla', 'vsa'}:
        return
    if data.get('cache_enabled'):
        raise ValueError('Sparse attention cannot be combined with FirstBlockCache')
    if capability is not None and tuple(capability) not in {(8, 9), (12, 0)}:
        raise ValueError('Sparse attention has not been validated for this worker architecture')
    if sparse_available is False:
        raise ValueError('This worker has no compiled Sol/SLA/VSA kernel; choose Auto or Sage')
    if mode == 'vsa':
        raise ValueError(PROFILES['vsa']['evidence'])
    if mode == 'sla':
        if str(data.get('task', '')).startswith(('r2v', 'ref2v')):
            raise ValueError('The registered SLA adapter supports FL2V/T2V, not Ref2V')
        model = data.get('model') or data.get('fl2va_model')
        names = {PurePosixPath(str(item.get('name', '')).replace('\\', '/')).name
                 for item in data.get('loras', []) if isinstance(item, dict) and float(item.get('strength', 1)) > 0}
        if model != SLA_MODEL or SLA_ADAPTER not in names or not data.get('sparse_trained_weights'):
            raise ValueError('SLA requires the standard FL2VA INT8 checkpoint, paired converted SLA adapter, and trained-weights confirmation')
        if data.get('turbo_enabled') or data.get('sampler') != 'euler' or data.get('scheduler') != 'simple' or int(data.get('steps', 0)) != 4:
            raise ValueError('The registered SLA recipe requires 4-step Euler/simple with the separate Turbo adapter disabled')


def catalog():
    return {'cards': CARDS, 'profiles': PROFILES,
            'sources': ['https://docs.comfy.org/tutorials/video/minimax/minimax-h3',
                        'https://github.com/Comfy-Org/ComfyUI/blob/1f641fd9337f0ec4d635a28415a8d25a8d15f753/comfy_extras/nodes_sparse_attention.py'],
            'note': 'Card choice previews compatibility. The allocated worker selects the actual backend and encoder.'}
