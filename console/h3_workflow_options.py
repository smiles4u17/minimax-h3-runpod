"""Shared validation for the September 20 H3 workflow variants."""
import math
import re

VARIANTS = {"legacy", "fflf_20260920", "ref2v_20260920"}

def workflow_options(data):
    variant = str(data.get("workflow_variant") or "legacy")
    if variant not in VARIANTS:
        raise ValueError("Unknown H3 workflow variant")
    if variant == "legacy":
        return {"workflow_variant": variant}
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
    mode = number("second_pass_sigma", 2, 1, 4)
    split = number("pass1_split", 3 if variant.startswith("fflf") else 6, 1, 1000)
    if not mode.is_integer() or not split.is_integer():
        raise ValueError("Sigma choice and split must be whole numbers")
    result = {"workflow_variant": variant, "use_multi_image": flag("use_multi_image", False),
            "latent_upscale": flag("latent_upscale", True), "rtx_upscale": flag("rtx_upscale", True),
            "turbo_enabled": flag("turbo_enabled", True), "use_larry": flag("use_larry", False), "final_megapixels": number("final_megapixels", 1.0, .2, 2),
            "second_pass_sigma": int(mode), "pass1_split": int(split),
            "latent_upscale_model": str(data.get("latent_upscale_model") or "minimax_h3_latent_upscaler_3d_bf16.safetensors")}
    mode = data.get('output_mode')
    if mode is not None:
        if mode not in ('none', 'latent', 'rtx'):
            raise ValueError('Output mode must be none, latent or rtx')
        result.update(output_mode=mode, latent_upscale=mode in ('latent', 'rtx'), rtx_upscale=mode == 'rtx')
    return result


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
