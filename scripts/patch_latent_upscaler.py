"""Require completion of device-to-host copies before returning CPU latents."""
from pathlib import Path


def patch(source: str) -> str:
    old = 'out = out.to(device="cpu", dtype=orig_dtype, non_blocking=True)'
    new = 'out = out.to(device="cpu", dtype=orig_dtype, non_blocking=False)'
    if source.count(old) != 1:
        raise RuntimeError('Pinned latent upscaler CPU transfer changed; review before building')
    return source.replace(old, new, 1)


if __name__ == '__main__':
    path = Path('/comfyui/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler/minimax_h3_latent_upscaler_3d.py')
    path.write_text(patch(path.read_text()))
    print('Patched H3 latent CPU handoff to a synchronized copy')
