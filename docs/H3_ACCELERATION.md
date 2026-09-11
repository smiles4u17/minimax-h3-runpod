# H3 attention and generation recipes

The pinned ComfyUI commit `1f641fd9337f0ec4d635a28415a8d25a8d15f753`
descends from v0.35.0 (`40c4fcdf513a4523e39d54a9d391908af8df8171`). It already
includes the masked velocity correction, cgroup RAM accounting, native sparse
attention and PDD support. Do not downgrade to the release tag to obtain them.
The image build checks Aimdo 0.5.3, Kitchen 0.2.33 and the required source features,
then runs ComfyUI's CPU startup check. These checks do not measure GPU quality.

## Workflows

- `workflows/t2v.json`: ComfyUI API graph for text to video/audio. The first and
  last frame inputs are disconnected. The same FL2VA checkpoint supports this task.
- `examples/t2v-dense.json`: reproducible dense baseline request.
- `examples/t2v-sol-attn.json`: training-free native Sol-Attn comparison request.
- `examples/fl2v-sla-experimental.json`: native SLA comparison using the converted
  FL2V Turbo-SLA LoRA, 15% kept blocks, four Euler steps and shifts 6/3.
  Replace the example first-frame path with an actual mounted-volume file.
  This is a ComfyUI experiment, not demonstrated parity with LightX2V's
  training_euler sampler or its sparse-linear implementation.
- `examples/t2v-pdd8.json`: converted FL2VA PDD LoRA, eight Euler/simple steps,
  shifts 12/3. For reference generation use task `r2v`, supply references and use
  the matching Ref2VA converted LoRA.

## Install optional weights

Installed and SHA256-verified on volume `vgc3ky6r6y` on September 11, 2026: three LightX2V Turbo adapters (FL2V 4/8 step, Ref2V 4 step), both pruned PDD adapters, and the SLA FL2V adapter. The existing Larry v4 EMA adapter was also verified. See `h3_acceleration_installation.json`. No manual pod installation is needed. The temporary download pod was deleted.

For recovery on a different volume, set COMFY_ROOT
to `/runpod-volume` so downloads persist under `/runpod-volume/models`:

```sh
COMFY_ROOT=/runpod-volume python3.12 /opt/minimax-h3/scripts/download_models.py --asset sla-fl2v
COMFY_ROOT=/runpod-volume python3.12 /opt/minimax-h3/scripts/download_models.py --asset pdd-fl2v --asset pdd-r2v
```

These optional downloads pin Hugging Face revisions and do not run on worker
startup. The console also supports importing the converted PDD LoRA: choosing
the PDD preset prepares its pinned URL when the file is absent. Import it in the
LoRA library, refresh, then select the preset again. Original unconverted PDD
weights are not a substitute for the ComfyUI conversion with output-head banks.

## Controls and validation

Native dense remains SAMimate's attention default. Sol-Attn is training-free.
SLA requires matching trained weights; VSA requires FastH3-compatible weights.
The explicit trained-weights control acknowledges this configuration requirement;
it cannot inspect a checkpoint's training history. Sparse attention and
FirstBlockCache cannot be combined through the preset API. Reference/text keys
and target-audio query rows remain exact via `exact_kv_and_rows`.

Native sparse kernels automatically fall back to dense for unsupported GPU
shapes and sequences below 12,288 tokens. Logs are enabled to distinguish actual
sparse execution from fallback. Sparse mode does not guarantee a speedup.
Optional advanced request keys `shift_video` and `shift_audio` configure the
native flow-shift node. PDD fixes these at 12/3 and rejects custom shifts.

Before changing the core/runtime pin, build and test a separate image. Compare
fixed-seed dense and accelerated jobs using the same non-sensitive footage,
reference images, size and duration; inspect identity, motion and audio. Repeat
jobs on both small- and large-VRAM GPUs. Check `RAM limited by cgroup to ...`
when the container actually has a lower cgroup limit than host RAM. Do not infer
correct RAM accounting from an absent log when no such limit is imposed.
For SAMimate also verify the final lossless composite outside the original mask.
Preserve the existing endpoint GPU pools, volume and worker limits.

## Sources checked September 11, 2026

- https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.35.0
- https://github.com/Comfy-Org/ComfyUI/pull/15988
- https://github.com/Comfy-Org/ComfyUI/pull/15908
- https://github.com/Comfy-Org/ComfyUI/blob/v0.35.0/comfy_extras/nodes_sparse_attention.py
- https://huggingface.co/asadkhan89/Minimax-h3-Turbo-SLA
- https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs
- https://huggingface.co/Kijai/MiniMax-H3-experimental

## Step counts and Turbo families

The console and worker accept positive whole-number step counts without the old
4-8 Turbo or 4-50 regular limits. Presets still suggest their trained count;
changing the Turbo family does not clamp or reset a manually chosen count. PDD's
8 steps are now a recommendation too. ComfyUI's own node validation remains in
force; more steps do not imply higher quality for a distilled model.

Select **Larry Turbo** for its dedicated loader/sampler. Select **LightX2V** for
ComfyUI-converted adapters, the standard model-only LoRA loader, and Euler/simple.
The family control chooses a task-matched 4-step file; the file dropdown also
provides LightX2V's FL2V 8-step adapter. FL2V uses flow shifts 6/3; Ref2V uses 12/3.
**Auto from selected file** detects the converted LightX2V filenames. Mismatched
FL2V/Ref2V adapters and Larry's dedicated sampler with LightX2V are rejected.
This uses ComfyUI's native AV implementation, not a separate LightX2V engine.

Source configurations:
https://github.com/ModelTC/LightX2V/tree/main/configs/minimax_h3/dmd

The `console/` directory contains a runnable source snapshot of the local console,
including its UI and tests. It is excluded from the worker Docker build context.
