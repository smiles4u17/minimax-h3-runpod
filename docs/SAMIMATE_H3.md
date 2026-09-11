# SAMimate masked Ref2VA

The console sends `task: r2v_masked` with normal Ref2VA identity images (1–9), a structured replacement prompt, and `masked_edit: {source, mask, width, height}`. Older workers reject this distinct task instead of ignoring the mask.

The existing R2V template supplies reference conditioning and the authoritative joint latent dimensions. `SAMimateH3MaskedTarget` encodes the source video into that target and attaches a nested video/audio noise mask. The source is not also sent as an ordinary video reference. White means regenerate; black preserves source latents. Masks are conservatively reduced over the H3 causal 1,4,4,4,4 frame groups and 32-pixel spatial tokens. Source and mask lengths must match; only matching trailing frames are repeated to fill the model's 17n+5 grid.

The console starts masked edits with 20-step res_multistep/simple, native attention, no Turbo LoRA and no cache. Existing model/identity LoRA selections remain available. The Dockerfile pins native ComfyUI mask support at `1f641fd9337f0ec4d635a28415a8d25a8d15f753`. The bundled node checks runtime mask support before encoding. Ordinary FL2V/R2V requests retain their existing graph selection.

Source trims are limited to 1–15 seconds. A lossless source plate retains original resolution, frame rate and audio; a synchronized 24fps working copy feeds SAM and H3. The console composites at the plate resolution and rate, clips to its frame count, and restores its soundtrack. Final output includes a lossless FFV1 MKV master and a compressed H.264/AAC MP4 preview. Latent/VAE reconstruction alone does not preserve pixels outside the mask: the final source composite is essential. The hard pixel mask prevents spill into the protected background; exact identity, motion, lighting and edge appearance still require a real GPU render review.

Primary references:

- [MiniMax reference prompt guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
- [ComfyUI H3 implementation](https://github.com/Comfy-Org/ComfyUI/blob/1f641fd9337f0ec4d635a28415a8d25a8d15f753/comfy/ldm/minimax/model.py)
- [Reference-guided masked editing and causal/token mask contract](https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/main/docs/MASKED_EDITING.md)

Validation: worker graph tests cover routing two identity images, clean source-target wiring, old-task rejection, and causal/token mask conversion. Console tests exercise submission contract and a real FFmpeg composite, comparing every protected test pixel and decoded source audio. These are not GPU generation-quality tests.
