"""Standalone node for rebasing MiniMax-H3 AdaLN LoRA patches."""

from __future__ import annotations

import logging

from comfy_api.latest import io

from . import adaln_patch

log = logging.getLogger("H3PDD.AdaLN")

MODES = ("port", "strip", "off")


class H3PDDStandaloneAdaLNFix(io.ComfyNode):
    """Make dense and curve-form H3 LoRAs portable between H3 model variants."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            # Deliberately distinct from PlagueKind's H3AdaLNLoRAFix ID so both
            # packs may be installed without one registration replacing the other.
            node_id="H3PDDStandaloneAdaLNFix",
            display_name="H3 AdaLN LoRA Fix (PDD Standalone)",
            category="MiniMax H3/model_patches",
            description=(
                "Rebases incompatible MiniMax-H3 AdaLN LoRA weights between "
                "dense and curve-form checkpoints. Place it after ordinary LoRA "
                "loaders. FL2VA and Ref2VA grid candidates are detected and the "
                "closest safe match is selected automatically. This node has a "
                "unique ID and can coexist with PlagueKind Nodes."),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="MODEL after the ordinary LoRA loader(s)."),
                io.Combo.Input(
                    "mode", options=list(MODES), default="port",
                    tooltip=(
                        "port: rebase incompatible AdaLN patches. "
                        "strip: remove incompatible patches without replacing "
                        "them. off: pass the model through unchanged.")),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, mode="port") -> io.NodeOutput:
        try:
            patched, report = adaln_patch.fix_model(model, mode)
        except Exception:  # noqa: BLE001 - never take down the workflow
            log.exception(
                "[H3PDD AdaLN] fix failed; passing the model through unchanged")
            return io.NodeOutput(model)

        log.info("[H3PDD AdaLN] %s", adaln_patch.format_report(report))
        for note in report["notes"][:8]:
            log.warning("[H3PDD AdaLN]   %s", note)
        if len(report["notes"]) > 8:
            log.warning("[H3PDD AdaLN]   ... and %d more",
                        len(report["notes"]) - 8)
        return io.NodeOutput(patched)
