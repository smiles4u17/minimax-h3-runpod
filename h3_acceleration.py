"""Explicit H3 acceleration recipes; custom mode preserves the user's sampler."""
SLA_ADAPTER = 'minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors'
PDD_FILES = {'fl2v': 'MiniMax-H3-FL2VA-Acc-8Step.safetensors',
             'r2v': 'MiniMax-H3-Ref2VA-Acc-8Step.safetensors'}
BASE_MODELS = {'fl2v': 'minimax_h3_fl2va_pruned_int8_convrot.safetensors',
               'r2v': 'minimax_h3_ref2va_pruned_int8_convrot.safetensors'}


def recipe(data):
    result = dict(data)
    mode = str(data.get('acceleration_recipe') or 'custom')
    if mode not in {'custom', 'pdd8', 'sla4'}:
        raise ValueError('Unknown H3 acceleration recipe')
    if mode == 'custom':
        return result
    task = 'r2v' if str(data.get('task', '')).startswith(('r2v', 'ref2v')) or data.get('workflow_variant') == 'ref2v_20260920' else 'fl2v'
    if mode == 'sla4' and task == 'r2v':
        raise ValueError('The installed SLA recipe is FL2V/T2V only; use PDD or Sol for Ref2V')
    if str(data.get('attention') or '') in {'sla', 'vsa', 'sol-attn'} and mode == 'pdd8':
        raise ValueError('PDD requires dense attention; select Auto, Sage or Native')
    for item in data.get('loras') or []:
        if not isinstance(item, dict) or item.get('enabled') is False or not float(item.get('strength', 1)):
            continue
        name = str(item.get('name', '')).lower()
        if ('acc-8step' in name or 'turbo' in name) and not (mode == 'sla4' and name.endswith(SLA_ADAPTER.lower())):
            raise ValueError('Disable other acceleration LoRAs before selecting this recipe')
    # Selecting a recipe is the explicit action that applies these training settings.
    result.update(acceleration_recipe=mode, turbo_enabled=False, use_larry=False,
                  turbo_family='none', cache_enabled=False, sampler='euler', scheduler='simple',
                  steps=8 if mode == 'pdd8' else 4, pdd_enabled=mode == 'pdd8',
                  model=BASE_MODELS[task], **{('ref2va_model' if task == 'r2v' else 'fl2va_model'): BASE_MODELS[task]})
    if mode == 'pdd8':
        result.update(pdd_lora='H3/' + PDD_FILES[task], attention_profile='custom', shift_video=12., shift_audio=3.,
                      attention=data.get('attention') or 'auto')
        if result.get('latent_upscale', True) and int(result.get('second_pass_sigma', 2)) == 4:
            raise ValueError('PDD uses an exact complete first pass; select a low-noise refinement sigma option')
        result['pass1_split'] = 8
    else:
        result.update(attention='sla', attention_profile='sla', sparse_trained_weights=True,
                      sparse_keep_percent=15., sparse_start_percent=0., sparse_end_percent=1.,
                      shift_video=6., shift_audio=3.)
        loras = [dict(x) for x in result.get('loras', []) if isinstance(x, dict)
                 and str(x.get('name', '')).replace('\\', '/').split('/')[-1] != SLA_ADAPTER]
        loras.append({'name': 'H3/' + SLA_ADAPTER, 'strength': 1., 'enabled': True})
        result['loras'] = loras
        result['pass1_split'] = 4
        if int(result.get('second_pass_sigma', 2)) == 4:
            raise ValueError('SLA uses a complete four-step first pass; select low-noise refinement')
    return result


def catalog():
    return {'recipes': [
        {'id': 'custom', 'label': 'Custom — keep current sampler and model'},
        {'id': 'pdd8', 'label': 'PDD — 8-step Euler, full displacement heads', 'tasks': ['fl2v', 'r2v', 't2v'],
         'assets': list(PDD_FILES.values()), 'model': BASE_MODELS},
        {'id': 'sla4', 'label': 'SLA — 4-step Euler, paired FL2V weights', 'tasks': ['fl2v', 't2v'],
         'assets': [SLA_ADAPTER], 'model': BASE_MODELS['fl2v']},
    ], 'pdd_refinement': 'Exact PDD first pass; dense base model for low-noise refinement',
        'vsa': 'Requires compatible FastH3 weights; unavailable in this worker'}
