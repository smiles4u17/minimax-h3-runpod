# Native Sol VAE and H3 context windows

The worker pins ComfyUI v0.44.1 (49b61c3cd044fb65acd97a960467a00414694146).
Torch 2.10.0+cu130, the two SageAttention wheels, Triton constraints, custom-node
pins and the existing attention safety patches are retained. Core requires
comfy-aimdo 0.6.0 and comfy-kitchen 0.2.40; build gates check those versions.

H3 and Prompt Book expose independent persisted `video_decoder` selections:
`sol_int8` (default, lower weight VRAM), `sol_fp16`, and `original`.
Both Sol artifacts are pinned with sizes and SHA256 in assets/h3_sol_vae.json.
Sol is connected only to video decode nodes, including refinement diagnostics.
The selected reference encoder and audio VAE retain their existing connections.

`context_windows` is a strict boolean, off by default. When enabled, native
MiniMaxH3ContextWindows is placed after LoRA/attention patches and before the
second-pass guider. The first pass uses the full sequence. Single-pass serverless jobs skip windowing.
Native defaults: 124 real frames, 39 overlap, pyramid fusion, FreeNoise on,
split_conds_to_windows off. FirstBlockCache must be off: its cache is not keyed
by context window. Other attention selections remain independently available.

All legacy and dated T2V, FL2V and Ref2V preset builders use the shared path.
The three downloadable editor templates are single-pass and use Sol decode with full-sequence sampling. Original encoder
loaders remain in place. To roll back decoding, select Original VAE; to roll
back windowed sampling, clear Context windows. No measured speedup is claimed.

Sources:
- https://github.com/Comfy-Org/ComfyUI/blob/v0.44.1/comfy_extras/nodes_context_windows.py
- https://huggingface.co/Comfy-Org/MiniMax-H3/tree/54472221d27c695fca9621b0c3ed52f06c4af707/vae

## Native learned upscaler

The two-pass builder now uses `LatentUpscaleModelLoader` followed by
`MiniMaxH3LatentUpscaleWithModel` directly on the first sampler's denoised AV
latent. The native node returns the combined latent with audio unchanged;
separate/concatenate nodes and the original custom upscaler installation are
removed. The high-resolution guider and refinement schedules are retained.
Existing original BF16 and v1 BF16/FP16/FP32 checkpoints remain selectable.
New selections default to v1 BF16. Native ComfyUI chooses inference precision;
checkpoint storage dtype does not force execution dtype. Saved manual precision
values are accepted for migration and normalized to `auto` in the worker input.

Source: https://github.com/Comfy-Org/ComfyUI/pull/17412
