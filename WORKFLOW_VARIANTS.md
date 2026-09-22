# September 20 H3 workflows

The console workflow selector keeps Legacy and adds FFLF and Ref2V adaptations of the supplied September 20 graphs. Select a new workflow in Inputs & Subjects, fill photos and prompt, then use Generation & models for the resolution, duration, model loaders, sampling, and upscale options. Model files resolve on the attached network volume.

## API

Use task `fl2v_20260920` with workflow_variant `fflf_20260920`, or task `r2v_20260920` with workflow_variant `ref2v_20260920`. These task names deliberately fail on old images. Legacy task names and defaults are preserved.

`photos` is a list of up to four ordinary worker asset objects (volume_path, URL, or inline data); preserve unused slots as null. `keyframe_positions` is the matching list of percentage strings, with empty strings for unused slots. `use_multi_image: true` selects H3Keyframes. FFLF photos 3 and 4 require it. Video/audio references cannot be combined with keyframes. FFLF without keyframes uses first/last photos; Ref2V without keyframes uses all occupied photos as references. Existing reference_videos/reference_audios controls still work in Ref2V.

With `turbo_enabled: true` (default), `use_larry: true` forces the Larry loader and h3_turbo sampler; false uses the task-matched selectable `turbo_lora` and regular `sampler`. Additional `loras` form one common model chain. Model, CLIP, VAEs, prompt, duration, megapixels, aspect_ratio, seed, attention and cache use the established preset fields.

`latent_upscale` defaults true. `megapixels` controls pass one (console selection initializes 0.2). `final_megapixels` defaults 1.0. `pass1_split` defaults 3 for FFLF / 6 for Ref2V and must be below `steps` when upscale is enabled; console presets initialize 4 / 8 steps. `second_pass_sigma` choices: 1 = 3 steps, 2 = 4 steps, 3 = 5 steps, 4 = the low remainder of the first-pass schedule. The exact manual sigma values are recorded in the sanitized manifests. `latent_upscale_model` defaults minimax_h3_latent_upscaler_3d_bf16.safetensors under models/latent_upscale_models.

`rtx_upscale` defaults true, independent of latent upscale. It targets a 1920-pixel long edge at the selected aspect ratio (1920x1080 at 16:9). Every new workflow saves a final-frame PNG as well as the video. Disable RTX for an unsupported GPU; no silent fallback changes the requested output.

## Graph adaptation and release

The Python builder resolves editor switches, Set/Get helpers and bypass groups into explicit API nodes. Pass one uses high sigmas; its denoised AV latent is separated, video is upscaled, audio is retained, and the AV latent is recombined for pass two. Conditioning is rebuilt at the final dimensions. Both passes share preceding model/LoRA/attention patches; FirstBlockCache is only attached to pass one. Disabling latent upscale removes the second pass entirely.

The manifests under workflows/*_20260920.json record source SHA-256 and relevant node settings; they are not runnable graphs. Original private prompts and media paths are omitted. Pinned custom-node repositories and NVIDIA dependency are installed by the GitHub-connected worker build. Keep h3_workflow_options.py identical to console/h3_workflow_options.py.

Validated with CPU graph/API tests and browser checks, not a GPU render. After publishing, verify the RunPod build, worker image, mounted model inventory, then a short synthetic export. Preserve the existing endpoint volume and GPU configuration. Read and update console/AGENT_ONBOARDING.md at every handoff.

## Console controls
The task selects FFLF or Ref2V automatically. Independent Use latent upscale and RTX switches control their branches; RTX defaults on. RTX follows the second pass when latent is enabled, otherwise original decode. LOWRES SIZE (MP) and FINAL SIZE (MP) sit beside each other (defaults 0.2 and 1). Final MP only applies to latent upscale. No Turbo sets turbo_enabled=false and bypasses the Turbo node and dedicated sampler even if use_larry is remembered true. Additional user LoRAs are preserved. Four photo slots support file drops, file picker, preview and Clear.

Older API clients may still send output_mode: none disables both, latent enables latent only, rtx enables both. This explicit field overrides booleans for compatibility; the new console does not send it. CPU checks cover both tasks with Turbo off and RTX without latent; a GPU export remains unverified.
