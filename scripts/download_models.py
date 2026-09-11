#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui")) / "models"

ASSET_REVISIONS = {"sla-fl2v": "f7c676b328719e281ad0d6e13dce4b54b713f985", "pdd-fl2v": "f4cac997f880e93cf6940af61ee8d58ef31ff7f3", "pdd-r2v": "f4cac997f880e93cf6940af61ee8d58ef31ff7f3"}

ASSETS = {
    "sla-fl2v": ("asadkhan89/Minimax-h3-Turbo-SLA", "minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors", "loras/H3"),
    "pdd-fl2v": ("Kijai/MiniMax-H3-experimental", "loras/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors", "loras"),
    "pdd-r2v": ("Kijai/MiniMax-H3-experimental", "loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors", "loras"),
    "fl2v": (
        "Comfy-Org/MiniMax-H3",
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "diffusion_models",
    ),
    "r2v": (
        "Comfy-Org/MiniMax-H3",
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "diffusion_models",
    ),
    "text-blackwell": (
        "Comfy-Org/MiniMax-H3",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "text_encoders",
    ),
    "text-universal": (
        "Comfy-Org/MiniMax-H3",
        "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        "text_encoders",
    ),
    "video-vae": (
        "Comfy-Org/MiniMax-H3",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae",
    ),
    "audio-vae": (
        "Comfy-Org/MiniMax-H3",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
        "vae",
    ),
    "turbo": (
        "larryvrh/MiniMax-H3-Turbo-Lora",
        "minimax_h3_turbo_v4_step600_ema.safetensors",
        "loras/H3",
    ),
}


def download(asset: str) -> None:
    repo, filename, subdir = ASSETS[asset]
    destination = ROOT / subdir
    destination.mkdir(parents=True, exist_ok=True)
    # Comfy-Org paths already begin with their ComfyUI model directory. Using
    # ROOT avoids creating models/vae/vae or models/diffusion_models/diffusion_models.
    local_dir = ROOT if filename.startswith(f"{subdir}/") else destination
    token = os.environ.get("HF_TOKEN") or None
    hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=local_dir,
        token=token,
        revision=ASSET_REVISIONS.get(asset),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=sorted(ASSETS), action="append", required=True)
    args = parser.parse_args()
    for asset in args.asset:
        download(asset)


if __name__ == "__main__":
    main()
