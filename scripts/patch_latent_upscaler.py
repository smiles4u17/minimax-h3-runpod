"""Require completion of device-to-host copies before returning CPU latents."""
from pathlib import Path


def patch(source: str) -> str:
    old = 'out = out.to(device="cpu", dtype=orig_dtype, non_blocking=True)'
    new = 'out = out.to(device="cpu", dtype=orig_dtype, non_blocking=False)'
    if source.count(old) != 1:
        raise RuntimeError('Pinned latent upscaler CPU transfer changed; review before building')
    source = source.replace(old, new, 1)
    # The cached CPU model must also be ready before another job reloads it.
    old = 'model.to("cpu", non_blocking=True)'
    if source.count(old) != 1:
        raise RuntimeError('Pinned latent upscaler model offload changed')
    return source.replace(old, 'model.to("cpu", non_blocking=False)', 1)


if __name__ == '__main__':
    path = Path('/comfyui/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler/nodes/minimax_h3_latent_upscaler_3d.py')
    path.write_text(patch(path.read_text()))
    print('Patched H3 latent CPU handoff to a synchronized copy')
