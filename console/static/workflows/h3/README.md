# H3 LightX2V example workflows

Unmodified upstream ComfyUI workflows from ModelTC/Minimax-H3-Turbo, revision 02e26d591f7a04d5d1a074c9566d5dd4f22f6225.
Source: https://github.com/ModelTC/Minimax-H3-Turbo/tree/02e26d591f7a04d5d1a074c9566d5dd4f22f6225/example_workflows

These visual workflows require the upstream ComfyUI dependencies; they are not RunPod API payloads. The console LightX2V presets configure its existing worker graph. FL2V 4-step uses 768p v1.0 (6/3 shifts), FL2V 8-step uses legacy v1.0 (12/3), Ref2V 4-step uses v0.1 (12/3), Ref2V 8-step uses v1.0 768p (12/3). Refresh the model library before applying presets. No identity or style LoRAs are removed by a sampling preset.

The new FL2V 4-step v1.1/v1.2 and 8-step 768p weights are catalogued separately; selecting them manually requires checking their author guidance. Render quality is not established by downloading an example.
