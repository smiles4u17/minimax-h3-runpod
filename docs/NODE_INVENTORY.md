# Workflow node inventory

The two API presets use only three custom-node packages:

| Package | Required production nodes |
| --- | --- |
| `kijai/ComfyUI-KJNodes` | `MiniMaxH3MemoryEfficientSageAttentionPatch`, `ImageResizeKJv2` |
| `Larryvrh/ComfyUI-MiniMax-H3-Turbo` | `MiniMaxH3TurboLoRA`, `MiniMaxH3TurboSampler` |
| `Apache0ne/ComfyUI-fasterminimax` | `MiniMaxH3FirstBlockCache` |

Everything else in the production graphs is part of ComfyUI core v0.33.1.

The original UI workflows also contain optional or bypassed editing/post-processing groups. They are deliberately not installed in the lean Serverless image:

| Feature | Package |
| --- | --- |
| Power LoRA loader / group bypass | `rgthree/rgthree-comfy` |
| UI switches | `yolain/ComfyUI-Easy-Use` |
| Legacy video load/combine | `Kosinkadink/ComfyUI-VideoHelperSuite` |
| RIFE interpolation | `Fannovel16/ComfyUI-Frame-Interpolation` |
| SeedVR2 upscaling | `numz/ComfyUI-SeedVR2_VideoUpscaler` / Registry package `ainvfx/ComfyUI-SeedVR2_VideoUpscaler` |
| NVIDIA RTX VSR | `Comfy-Org/Nvidia_RTX_Nodes_ComfyUI` plus `nvidia-vfx` |
| Impact switch | `ltdrdata/ComfyUI-Impact-Pack` |

Those groups do not participate in either exported generation API graph. Running interpolation/upscaling as a second endpoint is recommended because it avoids loading another large model stack into every H3 worker.

The original files are retained under `source_workflows/`. `scripts/build_workflows.py` performs the deterministic conversion into the two serverless presets.
