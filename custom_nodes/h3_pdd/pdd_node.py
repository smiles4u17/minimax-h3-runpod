"""ComfyUI node schema for MiniMax-H3 PDD acceleration."""

from __future__ import annotations

import logging

import folder_paths
from comfy_api.latest import io

from .pdd import NONE_LORA, apply_pdd

log = logging.getLogger("H3PDD")

HYBRID_MODES = (
    "Off (matching full model)",
    "b15-49",
    "b20-49",
    "b25-49",
    "b30-49",
    "Manual",
)
HYBRID_STARTS = {
    "b15-49": 15,
    "b20-49": 20,
    "b25-49": 25,
    "b30-49": 30,
}


def _lora_names():
    names = list(folder_paths.get_filename_list("loras"))
    return names or ["No LoRAs found"]


class H3PDDAcceleration(io.ComfyNode):
    """Load Alibaba's eight-NFE PDD checkpoint and its exact schedule."""

    @classmethod
    def define_schema(cls):
        names = _lora_names()
        return io.Schema(
            node_id="H3PDDAcceleration",
            display_name="H3 PDD Acceleration (8-Step)",
            category="MiniMax H3/model_patches",
            description=(
                "Loads Alibaba PAI's MiniMax-H3 PDD acceleration file directly, "
                "including its displacement heads, and outputs the exact "
                "eight-step sigma schedule. Use Euler and CFG 1.0. Supports the "
                "FL2VA/Ref2VA hybrid checkpoints by taking only their late "
                "AdaLN adapters from the Ref2VA PDD file."),
            inputs=[
                io.Model.Input("model",
                    tooltip="Native ComfyUI MiniMax-H3 MODEL."),
                io.Combo.Input("pdd_lora_name", options=names,
                    tooltip=(
                        "Alibaba PDD file matching the base: FL2VA for FL2VA "
                        "and hybrid checkpoints; Ref2VA for a full Ref2VA base. "
                        "Copy the original .safetensors into models/loras.")),
                io.Combo.Input("hybrid_mode", options=list(HYBRID_MODES),
                    default="Off (matching full model)",
                    tooltip=(
                        "For smhfacct hybrid checkpoints, select the checkpoint's "
                        "same block boundary. Off loads one matching PDD file. "
                        "Hybrid mode keeps FL2VA adapters/heads everywhere except "
                        "the selected late block AdaLN adapters.")),
                io.Combo.Input("hybrid_ref2va_pdd_lora", options=[NONE_LORA] + names,
                    default=NONE_LORA, optional=True,
                    tooltip=(
                        "Required only in hybrid mode: the Alibaba Ref2VA PDD "
                        "file. It is read only for late AdaLN adapters.")),
                io.Int.Input("manual_hybrid_start_block", default=20, min=0,
                    max=49, step=1, optional=True,
                    tooltip="Used only when Hybrid Mode is Manual."),
                io.Boolean.Input("strict_schedule", default=True, optional=True,
                    label_on="exact schedule required",
                    label_off="allow nearest head (experimental)",
                    tooltip=(
                        "Recommended: reject a sampler schedule that differs "
                        "from the exact eight PDD blocks. Turning this off "
                        "permits experiments but can reduce quality.")),
                io.Boolean.Input("allow_experimental_shifts", default=False,
                    optional=True,
                    label_on="allow non-12/3 shifts (experimental)",
                    label_off="require released 12/3 shifts",
                    tooltip=(
                        "Alibaba's released PDD heads were distilled at video/"
                        "audio shifts 12/3. Leave disabled for production. "
                        "Enabling permits other shifts but cannot retarget the "
                        "learned heads and may substantially reduce quality.")),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Sigmas.Output(display_name="sigmas (8 NFE)"),
            ],
        )

    @classmethod
    def execute(cls, model, pdd_lora_name,
                hybrid_mode="Off (matching full model)",
                hybrid_ref2va_pdd_lora=NONE_LORA,
                manual_hybrid_start_block=20,
                strict_schedule=True,
                allow_experimental_shifts=False) -> io.NodeOutput:
        primary_path = folder_paths.get_full_path_or_raise(
            "loras", pdd_lora_name)

        secondary_path = None
        start = None
        if hybrid_mode != "Off (matching full model)":
            if hybrid_ref2va_pdd_lora in (None, NONE_LORA):
                raise ValueError(
                    "Hybrid mode requires the Ref2VA PDD file in "
                    "hybrid_ref2va_pdd_lora")
            secondary_path = folder_paths.get_full_path_or_raise(
                "loras", hybrid_ref2va_pdd_lora)
            start = (int(manual_hybrid_start_block)
                     if hybrid_mode == "Manual"
                     else HYBRID_STARTS[hybrid_mode])

        patched, sigmas, report = apply_pdd(
            model, primary_path, secondary_path,
            hybrid_start_block=start,
            strict_schedule=bool(strict_schedule),
            allow_experimental_shifts=bool(allow_experimental_shifts))
        log.info(
            "[H3PDD] installed | %d adapters | grid=%d/block=%d -> %d NFE | "
            "shifts=%g/%g%s | hybrid=%s%s | strict_schedule=%s",
            report["adapters"], report["grid"], report["block_size"],
            report["nfe"], report["video_shift"], report["audio_shift"],
            (" (EXPERIMENTAL; released 12/3)"
             if report["experimental_shifts"] else ""),
            hybrid_mode,
            (" (%d Ref2VA AdaLN adapters)" % report["hybrid_adaln"])
            if report["hybrid_adaln"] else "",
            strict_schedule)
        return io.NodeOutput(patched, sigmas)
