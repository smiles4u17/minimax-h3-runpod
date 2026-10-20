# Attention configurations

The H3 and Prompt Book pages have an Attention workflow selector and a card
preview. The preview does not pin scheduling: the assigned worker selects its
actual backend and encoder. Auto uses the validated Sage build for SM89/SM120,
with native attention for other architectures. A numerical startup check runs
the production KJ attention path against SDPA in fp16/bf16 before accepting work.
Startup results appear in worker logs, runtime inventory and volume diagnostics.

Separate request configurations are in `static/workflows/h3/attention-profiles/`:
Auto, Sage dense and Sol-Attn candidates for RTX 4090, RTX 5090 and RTX PRO 4500
Blackwell, plus a paired SLA recipe and explicit VSA requirements.

These recipes do not establish a fastest backend for every workload. Dense
Sage is the supported default. Sol-Attn is a training-free, opt-in comparison;
the worker rejects it if the native sparse kernel is unavailable for its card.
Selecting an attention recipe changes attention/cache controls, not duration,
resolution, first-pass steps or upscale settings.

SLA is restricted to the standard FL2VA INT8 checkpoint plus the converted
`minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors` adapter, four
Euler/simple steps, no separate Turbo adapter, and trained-weights confirmation.
Ref2V and hybrid checkpoints are rejected. The UI disables this recipe until
the task/model/adapter match. VSA is disabled: no matching native-worker weights
have been validated. A confirmation checkbox alone cannot bypass these checks.
Sparse modes cannot combine with FirstBlockCache. Native kernels can keep short
sequences, conditioning and some layers dense; execution logs explain fallback.

Sources inspected October 19, 2026:

- [ComfyUI H3 guidance](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)
- [Pinned native sparse implementation](https://github.com/Comfy-Org/ComfyUI/blob/1f641fd9337f0ec4d635a28415a8d25a8d15f753/comfy_extras/nodes_sparse_attention.py)
- [LightX2V H3 recipes](https://github.com/ModelTC/LightX2V/blob/main/scripts/minimax_h3/README.md)
- [Official Comfy Sage wheels](https://github.com/Comfy-Org/wheels/releases/tag/sageattention-latest)

Published speedups from other engines or different workflows are not a benchmark
of this worker. Compare fixed seed, input, shape and schedule before changing
the default or claiming an end-to-end speed improvement.
