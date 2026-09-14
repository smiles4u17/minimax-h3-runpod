# H3 LightX2V example workflows

Adapted upstream ComfyUI workflows from ModelTC/Minimax-H3-Turbo, revision 02e26d591f7a04d5d1a074c9566d5dd4f22f6225.
Source: https://github.com/ModelTC/Minimax-H3-Turbo/tree/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/example_workflows

These visual workflows require the upstream ComfyUI dependencies; they are not RunPod API payloads. The console LightX2V presets configure its existing worker graph. FL2V 4-step uses 768p v1.2 (6/3 shifts), FL2V 8-step uses 768p v1.0 (6/3), Ref2V 4-step uses v0.1 (12/3), Ref2V 8-step uses v1.0 768p (12/3). Refresh the model library before applying presets. No identity or style LoRAs are removed by a sampling preset.

The current retained weights are pinned in assets/h3_lightx2v_current.json in the worker repository. Ref2V 4-step v0.1 is still the newest published 4-step version. The downloadable workflows now use the installed H3/ 8-step 768p LoRAs, matching shifts and 8-step schedules. The Ref2V checkpoint is aligned to the installed pruned model. The original T2V example contained a Qwen-image LoRA placeholder; that placeholder is replaced with the FL2V adapter. Render quality is not established by downloading an example.
