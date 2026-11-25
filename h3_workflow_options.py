"""Shared validation for the September 20 H3 workflow variants."""
import math
import re
from decimal import Decimal

VARIANTS = {"legacy", "fflf_20260920", "ref2v_20260920"}

SOL_VAES = {
    'sol_int8': 'minimax_h3_sol_vae_int8_convrot.safetensors',
    'sol_fp16': 'minimax_h3_sol_vae_fp16.safetensors',
}
CONTEXT_DEFAULTS = {
    'context_length': 124, 'context_overlap': 39, 'context_fuse_method': 'pyramid',
    'context_freenoise': True, 'context_split_conds': False,
}

def runtime_options(data):
    enabled = data.get('context_windows', False)
    if not isinstance(enabled, bool):
        raise ValueError('context_windows must be a boolean')
    decoder = data.get('video_decoder', 'sol_int8')
    if decoder not in (*SOL_VAES, 'original'):
        raise ValueError('video_decoder must be sol_int8, sol_fp16 or original')
    if enabled and data.get('cache_enabled', False):
        raise ValueError('Disable FirstBlockCache when using context windows; its cache is not window-specific')
    result = {'context_windows': enabled, 'video_decoder': decoder}
    if not enabled:
        return result
    context = {key: data.get(key, default) for key, default in CONTEXT_DEFAULTS.items()}
    for key, minimum in (('context_length', 22), ('context_overlap', 5)):
        value = context[key]
        if type(value) is not int or not minimum <= value <= 16384 or (value - 5) % 17:
            raise ValueError(f"{key.replace('_', ' ').capitalize()} must be a whole number of frames in 17-frame increments from {minimum} up to 16384")
    if context['context_overlap'] >= context['context_length']:
        raise ValueError('Context overlap must be shorter than context length')
    if context['context_fuse_method'] not in ('pyramid', 'relative', 'flat', 'overlap-linear'):
        raise ValueError('Unsupported context window blending method')
    for key in ('context_freenoise', 'context_split_conds'):
        if not isinstance(context[key], bool):
            raise ValueError(key + ' must be a boolean')
    return {**result, **context}

def reference_frame_cap(settings, output_frames):
    """Keep an explicit reference trim; never widen it to the output duration."""
    raw = settings.get("frame_load_cap", output_frames)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Reference frame cap must be a positive whole number") from exc
    if isinstance(raw, bool) or not math.isfinite(value) or not value.is_integer() or not 1 <= value <= 3600:
        raise ValueError("Reference frame cap must be a whole number from 1 to 3600")
    return min(int(value), output_frames)

def workflow_options(data):
    runtime = runtime_options(data)
    variant = str(data.get("workflow_variant") or "legacy")
    if variant not in VARIANTS:
        raise ValueError("Unknown H3 workflow variant")
    if variant == "legacy" and "pass1_split_enabled" not in data and not data.get("refine_enabled"):
        return {"workflow_variant": variant, **runtime}
    def flag(name, default):
        value = data.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(name + " must be a boolean")
        return value
    def number(name, default, low, high):
        try:
            value = float(data.get(name, default))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number") from exc
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value
    mode = number("second_pass_sigma", 2, 1, 7)
    split = number("pass1_split", 3 if variant.startswith("fflf") else 6, 1, 1000)
    if not mode.is_integer() or not split.is_integer():
        raise ValueError("Sigma choice and split must be whole numbers")
    reference_resolution = data.get('reference_resolution', 'generation')
    if reference_resolution not in ('generation', 'original'):
        raise ValueError('Reference resolution must be generation or original')
    precision = data.get("latent_upscale_precision", "auto")
    if precision not in ("auto", "bf16", "fp16", "fp32"):
        raise ValueError("Latent upscale precision must be auto, bf16, fp16 or fp32")
    turbo = flag("turbo_enabled", True)
    result = {"latent_upscale_precision": "auto", "workflow_variant": variant, "use_multi_image": flag("use_multi_image", False),
            "latent_upscale": flag("latent_upscale", variant != "legacy"), "rtx_upscale": flag("rtx_upscale", variant != "legacy"),
            "refine_enabled": flag("refine_enabled", False),
            "pass1_split_enabled": flag("pass1_split_enabled", turbo or mode == 4),
            "turbo_enabled": turbo, "use_larry": flag("use_larry", False), "final_megapixels": number("final_megapixels", 1.0, .2, 2),
            "second_pass_sigma": int(mode), "pass1_split": int(split), "reference_resolution": reference_resolution,
            "latent_upscale_model": str(data.get("latent_upscale_model") or "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors")}
    mode = data.get('output_mode')
    if mode is not None:
        if mode not in ('none', 'latent', 'rtx'):
            raise ValueError('Output mode must be none, latent or rtx')
        result.update(output_mode=mode, latent_upscale=mode in ('latent', 'rtx'), rtx_upscale=mode == 'rtx')
    if result["latent_upscale"] or result["refine_enabled"]:
        if result["second_pass_sigma"] == 6:
            steps = number("refine_steps", 4, 1, 1000)
            if not steps.is_integer():
                raise ValueError("Refinement steps must be a whole number")
            result.update(refine_steps=int(steps), refine_start_sigma=number("refine_start_sigma", .45, .000001, 1))
        elif result["second_pass_sigma"] == 7:
            result["refine_sigmas"] = parse_refine_sigmas(data.get("refine_sigmas", ""))
    return {**result, **runtime}


def parse_refine_sigmas(raw):
    if not isinstance(raw, str) or len(raw) > 32000:
        raise ValueError("Enter a comma- or space-separated list of refinement sigmas")
    try:
        sigmas = [float(value) for value in re.split(r"[\s,]+", raw.strip())]
    except ValueError as exc:
        raise ValueError("Enter numeric refinement sigmas") from exc
    if (not 2 <= len(sigmas) <= 1001 or not 0 < sigmas[0] <= 1 or sigmas[-1] != 0
            or not all(a > b for a, b in zip(sigmas, sigmas[1:]))):
        raise ValueError("Refinement sigmas must decrease from at most 1 to 0, with 2 to 1001 values")
    return format_refine_sigmas(sigmas)


def format_refine_sigmas(sigmas):
    # ComfyUI ManualSigmas only recognizes decimal literals.
    return ", ".join(format(Decimal(str(value)), "f") for value in sigmas)


def first_pass_steps(options, steps, recipe="custom"):
    if not (options.get("latent_upscale") or options.get("refine_enabled")):
        return steps
    enabled = options["pass1_split_enabled"]
    if enabled and recipe in ("pdd8", "sla4"):
        raise ValueError("Turn off the first-pass split for the fixed PDD or SLA recipe")
    split = options["pass1_split"] if enabled else steps
    if split > steps:
        raise ValueError("First-pass split cannot exceed total schedule steps")
    if options["second_pass_sigma"] == 4 and split >= steps:
        raise ValueError("Remaining sigmas require an enabled first-pass split below the total step count")
    return split


def validate_reference_photos(images):
    """Ref2V accepts up to nine independent images, without keyframe positions."""
    if not isinstance(images, list) or len(images) > 9:
        raise ValueError("Ref2V supports up to nine reference image slots")
    return [''] * len(images)


def validate_keyframes(images, positions, enabled):
    if not isinstance(images, list) or len(images) > 4:
        raise ValueError("Use up to four photo slots")
    if not isinstance(positions, list) or len(positions) > 4:
        raise ValueError("Keyframes require up to four percentage fields")
    images = images + [None] * (4-len(images))
    positions = positions + [''] * (4-len(positions))
    cleaned = []
    for index, (image, position) in enumerate(zip(images, positions), 1):
        position = str(position or '').strip()
        if not image and position:
            raise ValueError(f"Photo {index} is empty: clear its keyframe percentage")
        if enabled and image:
            if not re.fullmatch(r"(?:100(?:\.0+)?|[0-9]{1,2}(?:\.[0-9]+)?)%", position):
                raise ValueError(f"Photo {index} keyframe must be a percentage from 0% to 100%")
        cleaned.append(position)
    return cleaned


def variant_defaults(options):
    """Defaults shared by direct worker clients and the console API."""
    fflf = options['workflow_variant'].startswith('fflf')
    lora = ('minimax_h3_turbo_v4_step600_ema.safetensors' if options['use_larry'] else
            'minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors' if fflf else
            'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors')
    return {'steps': 4 if fflf else 8, 'megapixels': .2,
            'sampler': 'h3_turbo' if options['use_larry'] and options.get('turbo_enabled', True) else 'euler', 'turbo_lora': 'H3/' + lora}
