# ComfyUI-MiniMax-H3-PDD

Standalone native ComfyUI integration for Alibaba PAI's
[MiniMax-H3 PDD acceleration LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs).
It uses the original `.safetensors` files directly; no converted model file is
required.

## Installation

Copy or clone this folder as:

```text
ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-PDD
```

Put the acceleration files in `ComfyUI/models/loras`:

- `MiniMax-H3-FL2VA-Acc-8Step.safetensors`
- `MiniMax-H3-Ref2VA-Acc-8Step.safetensors`

For universal curve/pruned-model support, both explicitly named grid files are
bundled in this custom-node folder:

- `h3_silu_temb_grid_fl2va.safetensors`
- `h3_silu_temb_grid_ref2va.safetensors`

The node discovers the bundled copies automatically, so a normal clone/copy of
the complete repository needs no separate grid download. They are small support
assets, not LoRAs. You may alternatively place them in
`ComfyUI/models/h3_adaln`. An older generic `h3_silu_temb_grid.safetensors` is
no longer needed once both named grids are installed.

Restart ComfyUI. The pack registers two nodes:

- `H3 PDD Acceleration (8-Step)` loads the Alibaba PDD files and supplies their
  exact sigma schedule.
- `H3 AdaLN LoRA Fix (PDD Standalone)` ports incompatible AdaLN patches already
  attached by an ordinary LoRA loader.

This custom node is independent of PlagueKind Nodes. Do not keep an older copy
of `ComfyUI-H3-PDD-Acceleration` nested inside another custom-node pack, because
both copies register the same node ID.

The standalone AdaLN node intentionally uses the unique node ID
`H3PDDStandaloneAdaLNFix`, so it may coexist with PlagueKind's AdaLN node.
Do not put both AdaLN-fix nodes on the same MODEL chain; choose either one.

## Sampling

Wire both outputs from `H3 PDD Acceleration (8-Step)` to `SamplerCustom`:

```text
Load Diffusion Model -> H3 PDD Acceleration (8-Step) -> SamplerCustom MODEL
                                      SIGMAS (8 NFE) -> SamplerCustom SIGMAS
KSamplerSelect: euler                              CFG: 1.0
```

Keep `strict_schedule` enabled. The supplied SIGMAS already cover the complete
denoise-1.0 trajectory; do not place another scheduler between the PDD node and
`SamplerCustom`.

The released PDD checkpoints were distilled at video/audio shifts **12/3**.
The node enforces those shifts by default. Recomputing fusion weights and sigmas
for another shift does not retarget the learned backbone adapters or interval
heads. `allow_experimental_shifts` is an explicit off-distribution escape hatch,
not a quality-preserving conversion.

If the workflow uses `MiniMaxH3SigmaShift`, set it to 12/3 and place it before
the PDD node. Otherwise native H3's default 12/3 settings are used.

## Full and hybrid models

| Loaded model | Primary PDD | Hybrid mode | Secondary PDD |
| --- | --- | --- | --- |
| Full FL2VA | FL2VA | Off | None |
| Full Ref2VA | Ref2VA | Off | None |
| Hybrid b15-49 | FL2VA | b15-49 | Ref2VA |
| Hybrid b20-49 | FL2VA | b20-49 | Ref2VA |
| Hybrid b25-49 | FL2VA | b25-49 | Ref2VA |
| Hybrid b30-49 | FL2VA | b30-49 | Ref2VA |

Hybrid mode mirrors the hybrid checkpoint: FL2VA adapters and output heads are
used everywhere except `adaln_proj` adapters from the selected block through
block 49, which come from Ref2VA. This mapping is structurally faithful but was
not jointly distilled by Alibaba, so it remains experimental.

## Curve/pruned H3 checkpoints

The node internally rebases all 50 dense PDD AdaLN adapters for curve-form H3
checkpoints. It does not register or require a separate AdaLN Fix node.

The separately exposed `H3 AdaLN LoRA Fix (PDD Standalone)` is for non-PDD LoRAs
loaded through ComfyUI's ordinary LoRA loaders. Put it after those loaders. It
uses the same self-contained porting implementation but is not required on the
PDD node's own output.

The rebase needs a matching dense time curve. It discovers one from:

- a matching non-pruned H3 checkpoint in `models/diffusion_models`; or
- one or more `h3_silu_temb_grid*.safetensors` candidates in
  `models/h3_adaln`, `models/loras`, or another ComfyUI model search folder.

Use `h3_silu_temb_grid_fl2va.safetensors` for FL2VA-based curve models and the
b15/b20/b25/b30 hybrids, and `h3_silu_temb_grid_ref2va.safetensors` for
Ref2VA-based curve models. Both may be installed together: the node measures
every candidate against the loaded checkpoint's own table and automatically
uses the best safe fit. Dense models ignore all curve grids. If no candidate
fits safely, the node stops rather than silently dropping the AdaLN adapters.

## What the node does

- maps 362 Diffusers-format rank-64 adapters to native ComfyUI H3 keys;
- applies Q/K/V adapters to the correct slices of ComfyUI's fused QKV weights;
- fuses 32 displacement heads into eight schedule-weighted video and audio
  heads in fp32;
- transfers only the active head to the GPU;
- selects it from the real sigma, not an invocation counter;
- emits and validates the exact eight-NFE sigma schedule.

The files cannot be reduced to ordinary LoRAs for ComfyUI's standard LoRA
loader: their schedule-dependent full output-head banks are essential to PDD.

## Compatibility

- KJNodes' H3 low-VRAM attention and chunk-feedforward execution patches are
  the most sensible combinations to test first.
- PDD was distilled with dense H3 attention. Combining it with SLA or an SLA
  Turbo LoRA is off-distribution and needs visual/audio/reference/sync testing.
- Spectrum changes which evaluations execute and is not currently supported;
  PDD already defines its own eight-evaluation trajectory.

No PDD model weights are included in this repository. The two small bundled
curve grids are support assets derived from the matching MiniMax-H3 time curves.
PDD assets and the reference implementation belong to Alibaba PAI and their
respective authors. This independent ComfyUI integration is MIT-licensed.

## Acknowledgements

- [Alibaba PAI MiniMax-H3 Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)
  for the PDD checkpoints and method.
- [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun) for the publisher's
  reference inference integration.
- [PlagueKind Nodes](https://github.com/PlagueKind/ComfyUI-PlagueKind-Nodes)
  and ComfyUI's native MiniMax-H3 implementation for the surrounding H3
  integration work this project interoperates with.
