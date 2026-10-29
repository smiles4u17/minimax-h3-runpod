"""Native ComfyUI adapter for the MiniMax-H3 Parallel Decoding LoRAs.

Alibaba's files are not ordinary LoRAs.  They contain normal rank-64 updates
for the backbone, plus 32 complete video/audio output heads.  Four adjacent
heads are integrated into one Euler velocity, yielding the released eight-NFE
sampler.  This module maps the Diffusers names to ComfyUI, fuses those heads
once on CPU, and selects the correct fused head from the actual sampler sigma.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from typing import Iterable

import torch
import torch.nn.functional as F

log = logging.getLogger("H3PDD")

EXPECTED_TARGETS = (
    "to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2,adaln_proj.linear"
)
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0
NONE_LORA = "[None]"
QKV_WIDTH = 7168
PDD_MARKER = "_plaguekind_h3_pdd"


def validate_shift_contract(video_shift, audio_shift,
                            allow_experimental_shifts=False):
    """Reject schedules outside the released PDD training shifts by default.

    Reweighting the 32 interval heads and rebuilding the eight sigma boundaries
    is mathematically well-defined for another shift, but it does not retrain
    the backbone adapters or interval-specific heads.  Such a run is therefore
    off-distribution even when its sampler schedule is internally consistent.

    Returns ``True`` when the caller explicitly accepted an experimental shift.
    """
    video_shift = float(video_shift)
    audio_shift = float(audio_shift)
    released = (
        math.isclose(video_shift, DEFAULT_VIDEO_SHIFT, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(audio_shift, DEFAULT_AUDIO_SHIFT,
                         rel_tol=0.0, abs_tol=1e-6)
    )
    if released:
        return False

    message = (
        "Alibaba's released MiniMax-H3 PDD checkpoints were distilled for "
        f"video/audio shifts {DEFAULT_VIDEO_SHIFT:g}/{DEFAULT_AUDIO_SHIFT:g}, "
        f"but this MODEL requests {video_shift:g}/{audio_shift:g}. Rebuilding "
        "the fusion weights and sigmas cannot retarget the learned PDD heads. "
        "Set the active MiniMax-H3 shift to 12/3, or enable "
        "allow_experimental_shifts to run off-distribution."
    )
    if not allow_experimental_shifts:
        raise RuntimeError(message)
    log.warning("[H3PDD] %s", message)
    return True


@dataclass(frozen=True)
class PDDConfig:
    num_steps: int
    block_size: int
    rank: int
    alpha: float
    targets: str

    @property
    def nfe(self):
        return self.num_steps // self.block_size


@dataclass(frozen=True)
class LoRASpec:
    source: str
    target: str
    offset: tuple[int, int, int] | None = None
    transformer_block: int | None = None
    is_adaln: bool = False

    @property
    def down_key(self):
        return self.source + ".lora_down"

    @property
    def up_key(self):
        return self.source + ".lora_up"

    @property
    def patch_key(self):
        return self.target if self.offset is None else (self.target, self.offset)


def shifted_sigma(shift, sigma):
    """MiniMax flow shift, accepting a scalar or tensor."""
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def pdd_sigmas(video_shift=DEFAULT_VIDEO_SHIFT, num_steps=32, block_size=4):
    """The exact descending ComfyUI sigma boundaries for a released PDD file."""
    if block_size < 1 or num_steps < 1 or num_steps % block_size:
        raise ValueError("pdd_num_steps must be divisible by pdd_block_size")
    base = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return shifted_sigma(float(video_shift), base[::block_size]).to(torch.float32)


def pdd_time_grid(shift, num_steps):
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(float(shift), sigma)


def fusion_plans(shift, num_steps, block_size):
    """Return publisher-equivalent per-interval weights, shape [NFE, N]."""
    steps = pdd_time_grid(shift, num_steps).diff()
    plans = torch.zeros(num_steps // block_size, num_steps, dtype=torch.float64)
    for index, start in enumerate(range(0, num_steps, block_size)):
        span = steps[start:start + block_size].sum()
        plans[index, start:start + block_size] = (
            steps[start:start + block_size] / span
        )
    return plans


def fuse_parallel_heads(weight, bias, shift, block_size):
    """Fuse [N,out,in]/[N,out] head banks into [NFE,...] in fp32.

    The four-head slice is converted independently, avoiding a second fp32
    copy of the complete bank during load.  ComfyUI intentionally keeps H3's
    final projections in fp32, so retaining the fused result in fp32 matches
    the native model's output island and is slightly more accurate than
    accumulating the published BF16 tensors in BF16.
    """
    if weight.ndim != 3:
        raise ValueError(f"PDD head weight must be rank 3, got {list(weight.shape)}")
    num_steps = int(weight.shape[0])
    if num_steps % block_size:
        raise ValueError("head count must be divisible by pdd_block_size")
    if bias is not None and tuple(bias.shape) != tuple(weight.shape[:2]):
        raise ValueError(
            f"PDD head bias {list(bias.shape)} does not match weight {list(weight.shape)}"
        )

    plans = fusion_plans(shift, num_steps, block_size)
    fused_w, fused_b = [], []
    for index, start in enumerate(range(0, num_steps, block_size)):
        local = plans[index, start:start + block_size].to(torch.float32)
        w = weight[start:start + block_size].to(torch.float32)
        fused_w.append(torch.einsum("n,noi->oi", local, w).contiguous())
        if bias is not None:
            b = bias[start:start + block_size].to(torch.float32)
            fused_b.append(torch.einsum("n,no->o", local, b).contiguous())
    return torch.stack(fused_w), (torch.stack(fused_b) if fused_b else None)


def iter_lora_specs() -> Iterable[LoRASpec]:
    """All 362 Diffusers-to-Comfy adapter mappings in one PDD checkpoint."""
    for block in range(50):
        src = f"transformer_blocks.{block}"
        dst = f"diffusion_model.blocks.{block}"
        for name, start in (("to_q", 0), ("to_k", QKV_WIDTH),
                            ("to_v", QKV_WIDTH * 2)):
            yield LoRASpec(
                f"{src}.attn.{name}", f"{dst}.attn.qkv_proj.weight",
                (0, start, QKV_WIDTH), transformer_block=block)
        yield LoRASpec(f"{src}.attn.to_out.0",
                       f"{dst}.attn.out_proj.weight",
                       transformer_block=block)
        yield LoRASpec(f"{src}.ff.net.0.proj", f"{dst}.mlp.fc1.weight",
                       transformer_block=block)
        yield LoRASpec(f"{src}.ff.net.2", f"{dst}.mlp.fc2.weight",
                       transformer_block=block)
        yield LoRASpec(f"{src}.adaln_proj.linear",
                       f"{dst}.adaln_proj.linear.weight",
                       transformer_block=block, is_adaln=True)

    for block in range(2):
        src = f"token_refiner.refiner_blocks.{block}"
        dst = f"diffusion_model.token_refiner.blocks.{block}"
        for name, start in (("to_q", 0), ("to_k", QKV_WIDTH),
                            ("to_v", QKV_WIDTH * 2)):
            yield LoRASpec(
                f"{src}.attn.{name}", f"{dst}.attn.qkv_proj.weight",
                (0, start, QKV_WIDTH))
        yield LoRASpec(f"{src}.attn.to_out.0",
                       f"{dst}.attn.out_proj.weight")
        yield LoRASpec(f"{src}.ff.net.0.proj", f"{dst}.mlp.fc1.weight")
        yield LoRASpec(f"{src}.ff.net.2", f"{dst}.mlp.fc2.weight")


def _config(reader, path):
    metadata = reader.metadata() or {}
    missing = [key for key in (
        "pdd_num_steps", "pdd_block_size", "lora_rank", "lora_alpha",
        "lora_targets") if key not in metadata]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)} is not an Alibaba MiniMax-H3 PDD file; "
            f"missing metadata: {', '.join(missing)}"
        )
    config = PDDConfig(
        num_steps=int(metadata["pdd_num_steps"]),
        block_size=int(metadata["pdd_block_size"]),
        rank=int(metadata["lora_rank"]),
        alpha=float(metadata["lora_alpha"]),
        targets=str(metadata["lora_targets"]),
    )
    if config.block_size < 1 or config.num_steps % config.block_size:
        raise ValueError(
            f"Invalid PDD grid={config.num_steps}, block={config.block_size}"
        )
    if config.targets != EXPECTED_TARGETS:
        raise ValueError(
            "This PDD file targets a different architecture: " + config.targets
        )
    return config


def inspect_checkpoint(path):
    """Read metadata and keys without loading the 1.37 GB payload."""
    from safetensors import safe_open

    reader = safe_open(path, framework="pt", device="cpu")
    config = _config(reader, path)
    keys = set(reader.keys())
    specs = tuple(iter_lora_specs())
    required = {"proj_out.weight", "proj_out.bias",
                "audio_proj_out.weight", "audio_proj_out.bias"}
    for spec in specs:
        required.add(spec.down_key)
        required.add(spec.up_key)
    missing = sorted(required - keys)
    if missing:
        preview = ", ".join(missing[:4])
        raise ValueError(
            f"{os.path.basename(path)} is missing {len(missing)} required "
            f"PDD tensor(s), e.g. {preview}"
        )
    return reader, config, specs


def _select_reader(spec, primary, secondary, hybrid_start_block):
    if (secondary is not None and spec.is_adaln
            and spec.transformer_block is not None
            and spec.transformer_block >= hybrid_start_block):
        return secondary
    return primary


def build_lora_patches(primary, secondary, specs, config,
                       hybrid_start_block=None):
    """Build ordinary ComfyUI adapters; only hybrid AdaLN factors are swapped."""
    from comfy.weight_adapter import LoRAAdapter

    patches = {}
    hybrid_count = 0
    for spec in specs:
        reader = _select_reader(
            spec, primary, secondary,
            50 if hybrid_start_block is None else hybrid_start_block)
        down = reader.get_tensor(spec.down_key)
        up = reader.get_tensor(spec.up_key)
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError(f"{spec.source} is not a two-dimensional LoRA")
        if down.shape[0] != config.rank or up.shape[1] != config.rank:
            raise ValueError(
                f"{spec.source} rank is {down.shape[0]}/{up.shape[1]}, "
                f"metadata says {config.rank}"
            )
        patches[spec.patch_key] = LoRAAdapter(
            set(), (up, down, config.alpha, None, None, None))
        if reader is secondary:
            hybrid_count += 1
    return patches, hybrid_count


def _scalar(value):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().reshape(-1)[0].to(torch.float32).cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class PDDHeadController:
    """Own lazy file mappings/fused heads and select heads by real sigma."""

    def __init__(self, readers, video_weight, video_bias, audio_weight,
                 audio_bias, sigmas, video_shift, audio_shift, strict=True):
        # Keep safe_open handles alive: adapter tensors are backed by their maps.
        self.readers = tuple(r for r in readers if r is not None)
        self.video_weight = video_weight.contiguous()
        self.video_bias = video_bias.contiguous() if video_bias is not None else None
        self.audio_weight = audio_weight.contiguous()
        self.audio_bias = audio_bias.contiguous() if audio_bias is not None else None
        self.sigmas = sigmas.detach().to(torch.float32).cpu()
        self.video_shift = float(video_shift)
        self.audio_shift = float(audio_shift)
        self.strict = bool(strict)
        self.index = 0
        self._checked_schedule_id = None

    @property
    def nfe(self):
        return int(self.video_weight.shape[0])

    def _check_shifts(self, transformer_options):
        actual_v = float(transformer_options.get(
            "minimax_h3_sigma_shift_video", self.video_shift))
        actual_a = float(transformer_options.get(
            "minimax_h3_sigma_shift_audio", self.audio_shift))
        if (not math.isclose(actual_v, self.video_shift, abs_tol=1e-6)
                or not math.isclose(actual_a, self.audio_shift, abs_tol=1e-6)):
            raise RuntimeError(
                "H3 PDD was fused for video/audio shifts "
                f"{self.video_shift:g}/{self.audio_shift:g}, but downstream "
                f"model options request {actual_v:g}/{actual_a:g}. Remove the "
                "shift override or put the PDD node after it."
            )

    def _check_full_schedule(self, transformer_options):
        schedule = transformer_options.get("sample_sigmas")
        if schedule is None:
            return
        marker = id(schedule)
        if marker == self._checked_schedule_id:
            return
        try:
            actual = torch.as_tensor(schedule).detach().reshape(-1).to(
                dtype=torch.float32, device="cpu")
        except (RuntimeError, TypeError, ValueError):
            return
        if (actual.shape != self.sigmas.shape
                or not torch.allclose(actual, self.sigmas, rtol=2e-5,
                                      atol=2e-5)):
            if self.strict:
                raise RuntimeError(
                    "H3 PDD requires its exact eight-step SIGMAS output "
                    f"({self.nfe} model evaluations, denoise 1.0). The sampler "
                    "received a different schedule. Connect the SIGMAS output "
                    "of H3 PDD Acceleration directly to SamplerCustom."
                )
            log.warning("[H3PDD] sampler schedule differs from the distilled "
                        "PDD schedule; nearest displacement heads will be used.")
        self._checked_schedule_id = marker

    def select(self, timestep, transformer_options):
        self._check_shifts(transformer_options)
        self._check_full_schedule(transformer_options)
        sigma = _scalar(timestep)
        if sigma is None:
            raise RuntimeError("H3 PDD could not read the current model timestep")
        sigma /= 1000.0
        distances = (self.sigmas[:-1] - sigma).abs()
        index = int(distances.argmin().item())
        if self.strict and float(distances[index]) > 2e-5:
            raise RuntimeError(
                f"H3 PDD received sigma {sigma:.8f}, which is not one of its "
                "eight distilled sampling points. Connect this node's SIGMAS "
                "output directly to SamplerCustom and use Euler/CFG 1.0."
            )
        self.index = index

    @staticmethod
    def _linear(x, bank_w, bank_b, index):
        # Only one fused head visits VRAM at a time (~2 MB video / <1 MB audio).
        # It is a temporary tensor and is not retained between sampling steps.
        weight = bank_w[index].to(device=x.device, dtype=x.dtype,
                                  non_blocking=True)
        bias = None if bank_b is None else bank_b[index].to(
            device=x.device, dtype=x.dtype, non_blocking=True)
        return F.linear(x, weight, bias)

    def video_forward(self, x):
        return self._linear(x, self.video_weight, self.video_bias, self.index)

    def audio_forward(self, x):
        return self._linear(x, self.audio_weight, self.audio_bias, self.index)

    def wrapper(self, executor, x, timestep, context, transformer_options={},
                minimax_payload=None, **kwargs):
        self.select(timestep, transformer_options)
        if minimax_payload is not None:
            kwargs["minimax_payload"] = minimax_payload
        if not isinstance(executor, type) and callable(executor):
            return executor(x, timestep, context,
                            transformer_options=transformer_options, **kwargs)
        return executor.original(x, timestep, context,
                                 transformer_options=transformer_options, **kwargs)


def _diffusion_model(model):
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    if (diffusion is None or not hasattr(diffusion, "blocks")
            or not hasattr(diffusion, "final_layer")
            or not hasattr(diffusion, "sigma_shift_video")):
        raise TypeError("H3 PDD Acceleration requires a native ComfyUI MiniMax-H3 MODEL")
    if len(diffusion.blocks) != 50:
        raise ValueError(f"Expected 50 MiniMax-H3 blocks, found {len(diffusion.blocks)}")
    return diffusion


def _auto_port_curve_adaln(model, required=50):
    """Use the bundled internal basis solver; never silently strip adapters."""
    try:
        from . import adaln_patch as fixer
    except ImportError as exc:
        raise RuntimeError(
            "Curve-form H3 checkpoint detected, but the internal AdaLN rebase "
            "could not be imported. Reinstall the complete "
            "ComfyUI-MiniMax-H3-PDD custom-node folder."
        ) from exc

    fixed, report = fixer.fix_model(model, "port")
    if report.get("effective_mode") != "port" or report.get("ported", 0) < required:
        notes = "; ".join(report.get("notes", [])[:2]) or report.get("status", "unknown")
        raise RuntimeError(
            "The PDD AdaLN factors could not be rebased onto this pruned/curve "
            "H3 checkpoint. Put a matching non-pruned H3 checkpoint in "
            "models/diffusion_models (or matching "
            "h3_silu_temb_grid*.safetensors files in a model search folder), "
            f"then retry. Details: {notes}"
        )
    log.info("[H3PDD] %s", fixer.format_report(report))
    return fixed, report


def apply_pdd(model, primary_path, secondary_path=None,
              hybrid_start_block=None, strict_schedule=True,
              allow_experimental_shifts=False):
    """Return ``(MODEL, SIGMAS, report)`` using original Alibaba files directly."""
    if not os.path.isfile(primary_path):
        raise FileNotFoundError(primary_path)
    if secondary_path is not None and not os.path.isfile(secondary_path):
        raise FileNotFoundError(secondary_path)
    if secondary_path is not None:
        if hybrid_start_block is None or not 0 <= int(hybrid_start_block) <= 49:
            raise ValueError("Hybrid start block must be between 0 and 49")
        hybrid_start_block = int(hybrid_start_block)
    elif hybrid_start_block is not None:
        raise ValueError("A hybrid start block requires a Ref2VA PDD LoRA")

    diffusion = _diffusion_model(model)
    marker = model.model_options.get("transformer_options", {}).get(PDD_MARKER)
    if marker:
        raise RuntimeError("H3 PDD Acceleration is already applied to this MODEL")

    transformer_options = model.model_options.get("transformer_options", {})
    video_shift = float(transformer_options.get(
        "minimax_h3_sigma_shift_video", diffusion.sigma_shift_video))
    audio_shift = float(transformer_options.get(
        "minimax_h3_sigma_shift_audio", diffusion.sigma_shift_audio))
    experimental_shifts = validate_shift_contract(
        video_shift, audio_shift,
        allow_experimental_shifts=bool(allow_experimental_shifts))

    primary, config, specs = inspect_checkpoint(primary_path)
    secondary = secondary_config = None
    try:
        if secondary_path is not None:
            secondary, secondary_config, secondary_specs = inspect_checkpoint(
                secondary_path)
            if secondary_config != config or secondary_specs != specs:
                raise ValueError("Primary and hybrid PDD files have incompatible metadata")

        patches, hybrid_count = build_lora_patches(
            primary, secondary, specs, config, hybrid_start_block)

        vw, vb = fuse_parallel_heads(
            primary.get_tensor("proj_out.weight"),
            primary.get_tensor("proj_out.bias"),
            video_shift, config.block_size)
        aw, ab = fuse_parallel_heads(
            primary.get_tensor("audio_proj_out.weight"),
            primary.get_tensor("audio_proj_out.bias"),
            audio_shift, config.block_size)
        sigmas = pdd_sigmas(
            video_shift, config.num_steps, config.block_size)

        if vw.shape[0] != config.nfe or aw.shape[0] != config.nfe:
            raise ValueError("Fused PDD head count does not match checkpoint metadata")
        video_out = diffusion.final_layer.video_out
        audio_out = diffusion.final_layer.audio_out
        if tuple(vw.shape[1:]) != tuple(video_out.weight.shape):
            raise ValueError(
                f"PDD video heads {list(vw.shape[1:])} do not match model "
                f"{list(video_out.weight.shape)}")
        if tuple(aw.shape[1:]) != tuple(audio_out.weight.shape):
            raise ValueError(
                f"PDD audio heads {list(aw.shape[1:])} do not match model "
                f"{list(audio_out.weight.shape)}")

        patched = model.clone()
        added = patched.add_patches(patches, 1.0)
        if len(added) != len(patches):
            missing = sorted(str(key) for key in set(patches) - set(added))
            raise RuntimeError(
                f"Only {len(added)}/{len(patches)} PDD adapters matched this "
                f"ComfyUI model; first missing key: {missing[0] if missing else '?'}"
            )

        controller = PDDHeadController(
            (primary, secondary), vw, vb, aw, ab, sigmas,
            video_shift, audio_shift, strict_schedule)
        patched.add_object_patch(
            "diffusion_model.final_layer.video_out.forward",
            controller.video_forward)
        patched.add_object_patch(
            "diffusion_model.final_layer.audio_out.forward",
            controller.audio_forward)
        patched.add_wrapper_with_key(
            "diffusion_model", "h3_pdd_acceleration", controller.wrapper)

        to = patched.model_options.get("transformer_options", {}).copy()
        to[PDD_MARKER] = {
            "nfe": config.nfe,
            "video_shift": video_shift,
            "audio_shift": audio_shift,
            "hybrid_start_block": hybrid_start_block,
        }
        patched.model_options["transformer_options"] = to

        adaln_report = None
        if bool(getattr(diffusion, "use_adaln_curves", False)):
            patched, adaln_report = _auto_port_curve_adaln(patched)

        report = {
            "adapters": len(patches),
            "hybrid_adaln": hybrid_count,
            "hybrid_start_block": hybrid_start_block,
            "nfe": config.nfe,
            "grid": config.num_steps,
            "block_size": config.block_size,
            "video_shift": video_shift,
            "audio_shift": audio_shift,
            "experimental_shifts": experimental_shifts,
            "curve_adaln": adaln_report,
        }
        return patched, sigmas, report
    except Exception:
        # If setup fails, no returned controller owns the mappings.
        del primary
        if secondary is not None:
            del secondary
        raise
