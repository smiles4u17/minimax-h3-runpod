from __future__ import annotations

import base64
import copy
import ipaddress
import json
import mimetypes
import os
import re
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import runpod


COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
INPUT_DIR = COMFY_ROOT / "input"
OUTPUT_DIR = COMFY_ROOT / "output"
FLAT_OUTPUT_DIR = Path("/runpod-volume/outputs")
VOLUME_ROOT = Path(os.environ.get("RUNPOD_VOLUME_ROOT", "/runpod-volume"))
TEMPLATE_DIR = Path(os.environ.get("WORKFLOW_DIR", "/opt/minimax-h3/workflows"))
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_START_TIMEOUT = int(os.environ.get("COMFY_START_TIMEOUT_SECONDS", "180"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT_SECONDS", "3600"))
MAX_ASSET_BYTES = int(os.environ.get("MAX_ASSET_MB", "250")) * 1024 * 1024
MAX_BASE64_BYTES = int(os.environ.get("MAX_RETURN_BASE64_MB", "6")) * 1024 * 1024
LOW_VRAM_THRESHOLD_GB = float(os.environ.get("H3_LOW_VRAM_THRESHOLD_GB", "48"))

BLACKWELL_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
UNIVERSAL_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
SAGE_CAPABILITIES = {(12, 0)}
SAMPLERS = {"h3_turbo", "res_multistep", "er_sde", "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "deis", "uni_pc"}
SCHEDULERS = {"simple", "beta", "normal", "sgm_uniform", "karras", "exponential", "ddim_uniform", "linear_quadratic", "kl_optimal"}

_COMFY_PROCESS: subprocess.Popen[Any] | None = None
_COMFY_LOCK = threading.Lock()

TASKS = {
    "fl2v": {
        "template": "fl2v.json",
        "cache": "226",
        "conditioning": "182",
        "save": "900",
    },
    "r2v": {
        "template": "r2v.json",
        "cache": "210",
        "conditioning": "136",
        "save": "92",
    },
}


class InputError(ValueError):
    pass


def _load_template(name: str) -> dict[str, Any]:
    return json.loads((TEMPLATE_DIR / name).read_text(encoding="utf-8"))


def _comfy_ready() -> bool:
    try:
        response = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
        return response.status_code < 400
    except requests.RequestException:
        return False


def _ensure_comfy_ready(timeout: int = COMFY_START_TIMEOUT) -> None:
    """Recover ComfyUI if FlashBoot or a process crash left only the handler alive."""
    global _COMFY_PROCESS
    if _comfy_ready():
        return
    with _COMFY_LOCK:
        if _comfy_ready():
            return
        if _COMFY_PROCESS is None or _COMFY_PROCESS.poll() is not None:
            command = [
                sys.executable,
                str(COMFY_ROOT / "main.py"),
                "--listen",
                COMFY_HOST,
                "--port",
                str(COMFY_PORT),
                "--extra-model-paths-config",
                "/opt/minimax-h3/extra_model_paths.yaml",
                *_comfy_launch_args(),
            ]
            print("ComfyUI is unavailable; starting a replacement process.", flush=True)
            _COMFY_PROCESS = subprocess.Popen(command, cwd=str(COMFY_ROOT))

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _comfy_ready():
                print("ComfyUI replacement is ready.", flush=True)
                return
            if _COMFY_PROCESS.poll() is not None:
                raise RuntimeError(f"ComfyUI restart exited with code {_COMFY_PROCESS.returncode}.")
            time.sleep(1)
        raise RuntimeError(f"ComfyUI did not become ready within {timeout} seconds.")


def _gpu_info() -> tuple[str, tuple[int, int] | None]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), tuple(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    return "unknown", None


def _gpu_total_vram_gb() -> float | None:
    configured_mb = os.environ.get("H3_GPU_MEMORY_MB")
    if configured_mb:
        try:
            return float(configured_mb) / 1024
        except ValueError:
            pass
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    except Exception:
        pass
    return None


def _low_vram_worker(total_vram_gb: float | None = None) -> bool:
    mode = os.environ.get("H3_LOW_VRAM_MODE", "auto").strip().lower()
    if mode in {"1", "true", "yes", "on", "force"}:
        return True
    if mode in {"0", "false", "no", "off", "disabled"}:
        return False
    if mode != "auto":
        raise RuntimeError("H3_LOW_VRAM_MODE must be auto, on, or off")
    memory = _gpu_total_vram_gb() if total_vram_gb is None else total_vram_gb
    return memory is not None and memory <= LOW_VRAM_THRESHOLD_GB


def _comfy_launch_args() -> list[str]:
    args = shlex.split(os.environ.get("COMFY_ARGS", ""))
    if not _low_vram_worker():
        return args
    automatic = [
        "--reserve-vram",
        os.environ.get("H3_LOW_VRAM_RESERVE_GB", "1"),
        "--vram-headroom",
        os.environ.get("H3_LOW_VRAM_HEADROOM_GB", "1"),
        "--disable-smart-memory",
        "--cache-none",
    ]
    return [*automatic, *args]


def _model_exists(folder: str, name: str) -> bool:
    candidates = [
        COMFY_ROOT / "models" / folder / name,
        Path("/runpod-volume/models") / folder / name,
    ]
    return any(path.is_file() for path in candidates)


def _select_encoder(capability: tuple[int, int] | None) -> str:
    forced = os.environ.get("TEXT_ENCODER_FILE")
    if forced:
        return forced
    is_blackwell = bool(capability and capability[0] >= 10)
    if is_blackwell and _model_exists("text_encoders", BLACKWELL_ENCODER):
        return BLACKWELL_ENCODER
    if _model_exists("text_encoders", UNIVERSAL_ENCODER):
        return UNIVERSAL_ENCODER
    if _model_exists("text_encoders", BLACKWELL_ENCODER):
        raise RuntimeError(
            "Only the Blackwell NVFP4 text encoder is installed, but this GPU is not "
            "Blackwell. Populate the volume with MODEL_PROFILE=dual or universal."
        )
    raise RuntimeError("No MiniMax H3 text encoder is installed.")


def _sage_available() -> bool:
    try:
        import sageattention  # noqa: F401

        return True
    except Exception:
        return False


def _attention_mode(requested: str, capability: tuple[int, int] | None) -> str:
    configured = (requested or os.environ.get("ATTENTION_MODE", "auto")).lower()
    if configured not in {"auto", "sage", "native", "sla", "sol-attn", "vsa"}:
        raise InputError("attention must be auto, sage, native, sla, sol-attn, or vsa")
    if configured == "sage":
        if capability not in SAGE_CAPABILITIES:
            raise RuntimeError(f"SageAttention is not compiled for CUDA capability {capability}.")
        if not _sage_available():
            raise RuntimeError("SageAttention was requested but is not importable.")
        return "sage"
    if configured in {"native", "sla", "sol-attn", "vsa"}:
        return configured
    return "sage" if capability in SAGE_CAPABILITIES and _sage_available() else "native"


def _normalize_lora_name(name: str) -> str:
    normalized = name.replace("\\", "/").strip("/")
    if not normalized or ".." in Path(normalized).parts or Path(normalized).is_absolute():
        raise InputError(f"Invalid LoRA name: {name!r}")
    return normalized


def _payload_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise InputError(f"Invalid boolean value: {value!r}")


def _model_name(payload: dict[str, Any], key: str, folder: str, fallback: str) -> str:
    name = str(payload.get(key) or fallback).replace("\\", "/").strip("/")
    if not name or ".." in Path(name).parts or Path(name).is_absolute():
        raise InputError(f"Invalid {key} model name: {name!r}")
    if not _model_exists(folder, name):
        raise InputError(f"{key} model is not installed in models/{folder}: {name}")
    return name


def _apply_loras(
    workflow: dict[str, Any],
    loras: list[dict[str, Any]],
    model: list[Any] | None = None,
) -> list[Any]:
    previous: list[Any] = model or ["127", 0]
    for index, item in enumerate(loras):
        if not isinstance(item, dict) or "name" not in item:
            raise InputError("Each LoRA must contain name and may contain strength.")
        name = _normalize_lora_name(str(item["name"]))
        strength = float(item.get("strength", 1.0))
        if not 0 <= strength <= 2:
            raise InputError("LoRA strength must be between 0 and 2.")
        node_id = str(9100 + index)
        workflow[node_id] = {
            "inputs": {"model": previous, "lora_name": name, "strength_model": strength},
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": f"API LoRA {index + 1}"},
        }
        previous = [node_id, 0]
    return previous


def _patch_common(workflow: dict[str, Any], spec: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    gpu_name, capability = _gpu_info()
    total_vram_gb = _gpu_total_vram_gb()
    workflow["138"]["inputs"]["value"] = str(payload.get("prompt", "")).strip()
    if not workflow["138"]["inputs"]["value"]:
        raise InputError("prompt is required")

    duration = float(payload.get("duration", 5.0))
    megapixels = float(payload.get("megapixels", 0.6))
    steps = int(payload.get("steps", 6))
    threshold = float(payload.get("cache_threshold", 0.18))
    if not 1 <= duration <= 15:
        raise InputError("duration must be between 1 and 15 seconds")
    if not 0.2 <= megapixels <= 2.0:
        raise InputError("megapixels must be between 0.2 and 2.0")
    turbo_enabled = _payload_bool(payload.get("turbo_enabled"), default=True)
    sampler = str(payload.get("sampler") or ("h3_turbo" if turbo_enabled else "res_multistep")).lower()
    scheduler = str(payload.get("scheduler") or ("simple" if turbo_enabled else "beta")).lower()
    cache_enabled = _payload_bool(payload.get("cache_enabled"), default=False)
    if sampler not in SAMPLERS:
        raise InputError(f"Unsupported sampler: {sampler}")
    if scheduler not in SCHEDULERS:
        raise InputError(f"Unsupported scheduler: {scheduler}")
    if sampler == "h3_turbo" and not turbo_enabled:
        raise InputError("The dedicated H3 Turbo sampler requires turbo_enabled=true")
    if steps < 1 or isinstance(payload.get("steps"), bool) or float(payload.get("steps", 6)) != steps:
        raise InputError("steps must be a positive whole number")
    if not 0 <= threshold <= 1:
        raise InputError("cache_threshold must be between 0 and 1")

    workflow["132"]["inputs"]["value"] = duration
    workflow["115"]["inputs"]["megapixels"] = megapixels
    workflow["115"]["inputs"]["aspect_ratio"] = str(payload.get("aspect_ratio", "16:9 (Widescreen)"))
    workflow["124"]["inputs"]["steps"] = steps
    seed = int(payload.get("seed", secrets.randbelow(2**63 - 1)))
    if seed < 0:
        seed = secrets.randbelow(2**53)
    workflow["129"]["inputs"]["noise_seed"] = seed
    default_model = workflow["127"]["inputs"]["unet_name"]
    default_video_vae = workflow["119"]["inputs"]["vae_name"]
    default_audio_vae = workflow["120"]["inputs"]["vae_name"]
    selected_encoder = str(payload.get("clip") or "").strip() or _select_encoder(capability)
    workflow["127"]["inputs"]["unet_name"] = _model_name(payload, "model", "diffusion_models", default_model)
    workflow["119"]["inputs"]["vae_name"] = _model_name(payload, "video_vae", "vae", default_video_vae)
    workflow["120"]["inputs"]["vae_name"] = _model_name(payload, "audio_vae", "vae", default_audio_vae)
    workflow["128"]["inputs"]["clip_name"] = _model_name({"clip": selected_encoder}, "clip", "text_encoders", selected_encoder)

    low_vram = _low_vram_worker(total_vram_gb)
    model: list[Any] = ["127", 0]
    low_vram_profile: str | None = None
    if low_vram:
        low_vram_profile = os.environ.get("H3_LOW_VRAM_PROFILE", "minimum_vram")
        if low_vram_profile not in {"balanced", "low_vram", "minimum_vram", "maximum_speed"}:
            raise RuntimeError("H3_LOW_VRAM_PROFILE must be balanced, low_vram, minimum_vram, or maximum_speed")
        workflow["9140"] = {
            "inputs": {
                "model": model,
                "enabled": True,
                "memory_profile": low_vram_profile,
                "custom_chunk_tokens": 8192,
                "block_prefetch": "disable",
                "verbose": True,
            },
            "class_type": "MiniMaxH3LowVRAM",
            "_meta": {"title": f"API H3 low VRAM: {low_vram_profile}"},
        }
        model = ["9140", 0]

    model = _apply_loras(workflow, list(payload.get("loras", [])), model)
    turbo_family = "none"
    if turbo_enabled:
        turbo_name = _normalize_lora_name(str(payload.get("turbo_lora") or "H3/minimax_h3_turbo_v4_step600_ema.safetensors"))
        turbo_strength = float(payload.get("turbo_strength", 1.0))
        if not 0 <= turbo_strength <= 2:
            raise InputError("turbo_strength must be between 0 and 2")
        if not _model_exists("loras", turbo_name):
            raise InputError(f"Turbo LoRA is not installed in models/loras: {turbo_name}")
        lightx = "_comfyui_" in turbo_name.lower() and "minimax_h3_" in turbo_name.lower()
        turbo_family = str(payload.get("turbo_family") or "auto").lower()
        if turbo_family == "auto":
            turbo_family = "lightx2v" if lightx else "larry"
        if turbo_family not in {"larry", "lightx2v"}:
            raise InputError("turbo_family must be auto, larry or lightx2v")
        if lightx and turbo_family == "larry":
            raise InputError("The selected LightX2V LoRA requires the LightX2V loader")
        if turbo_family == "lightx2v":
            is_ref = spec["conditioning"] == "136"
            if ("_fl2v_" in turbo_name and is_ref) or ("_ref2v_" in turbo_name and not is_ref):
                raise InputError("LightX2V LoRA does not match the current FL2V/Ref2V task")
            if sampler == "h3_turbo":
                raise InputError("Choose Euler or another regular sampler for the LightX2V LoRA")
            workflow["152"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": model, "lora_name": turbo_name, "strength_model": turbo_strength}}
            workflow["9162"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
                "model": ["152", 0], "shift_video": 12.0 if is_ref else 6.0, "shift_audio": 3.0}}
            model = ["9162", 0]
        else:
            workflow["152"]["inputs"].update(model=model, lora_name=turbo_name, strength=turbo_strength)
            model = ["152", 0]
    else:
        workflow.pop("152", None)

    if sampler == "h3_turbo":
        workflow["125"]["inputs"]["sampler"] = ["154", 0]
    else:
        workflow["9150"] = {
            "inputs": {"sampler_name": sampler},
            "class_type": "KSamplerSelect",
            "_meta": {"title": f"API sampler: {sampler}"},
        }
        workflow["125"]["inputs"]["sampler"] = ["9150", 0]
        workflow.pop("154", None)
    workflow["124"]["inputs"]["scheduler"] = scheduler

    if "shift_video" in payload or "shift_audio" in payload:
        video_shift = float(payload.get("shift_video", 12.0))
        audio_shift = float(payload.get("shift_audio", 3.0))
        if not (0.01 <= video_shift <= 100 and 0.01 <= audio_shift <= 100):
            raise InputError("H3 flow shifts must be between 0.01 and 100")
        workflow["9161"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": model, "shift_video": video_shift, "shift_audio": audio_shift}}
        model = ["9161", 0]

    if _payload_bool(payload.get("pdd_enabled")):
        expected = "Ref2VA" if spec["conditioning"] == "136" else "FL2VA"
        names = [str(item.get("name", "")) for item in payload.get("loras", [])]
        if not any(expected in name and "Acc-8Step" in name and "comfy" in name for name in names):
            raise InputError(f"PDD requires the converted {expected} Acc-8Step ComfyUI LoRA")
        if "shift_video" in payload or "shift_audio" in payload:
            raise InputError("PDD fixes the flow shifts at 12/3; remove custom shifts")
        if turbo_enabled or sampler != "euler" or scheduler != "simple" or cache_enabled:
            raise InputError("PDD requires Euler/simple, separate Turbo and cache disabled; 8 steps recommended")
        workflow["9161"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {
            "model": model, "shift_video": 12.0, "shift_audio": 3.0}}
        model = ["9161", 0]

    attention = _attention_mode(str(payload.get("attention") or ""), capability)
    if attention in {"native", "sla", "sol-attn", "vsa"}:
        workflow.pop("145", None)
    else:
        workflow["145"]["inputs"]["model"] = model
        model = ["145", 0]

    if attention in {"sla", "sol-attn", "vsa"}:
        if cache_enabled:
            raise InputError("Disable FirstBlockCache when using sparse attention")
        if attention in {"sla", "vsa"} and not _payload_bool(payload.get("sparse_trained_weights")):
            raise InputError("SLA/VSA requires matching trained model weights or LoRA; confirm sparse_trained_weights")
        keep = float(payload.get("sparse_keep_percent", 10.0))
        tau = float(payload.get("sparse_tau", 1.3))
        start = float(payload.get("sparse_start_percent", 0.2))
        end = float(payload.get("sparse_end_percent", 1.0))
        if not (0.5 <= keep <= 95 and 0 <= tau <= 4 and 0 <= start <= end <= 1):
            raise InputError("Invalid sparse attention percentage, threshold, or sampling window")
        inputs = {"model": model, "selection": attention, "start_percent": start,
                  "end_percent": end, "dense_blocks": "", "min_tokens": 12288,
                  "extra_tokens": 256, "sink_conditioning": "exact_kv_and_rows", "verbose": True}
        inputs["selection.tau" if attention == "sol-attn" else "selection.keep_percent"] = tau if attention == "sol-attn" else keep
        workflow["9160"] = {"class_type": "BlockSparseAttention", "inputs": inputs}
        model = ["9160", 0]

    if cache_enabled:
        workflow[spec["cache"]]["inputs"]["threshold"] = threshold
        workflow[spec["cache"]]["inputs"]["model"] = model
        model = [spec["cache"], 0]
    else:
        workflow.pop(spec["cache"], None)
    workflow["124"]["inputs"]["model"] = model
    workflow["126"]["inputs"]["model"] = model

    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(payload.get("filename_prefix", "MiniMax_H3"))).strip("._")
    workflow[spec["save"]]["inputs"]["filename_prefix"] = f"video/{prefix or 'MiniMax_H3'}"
    return {
        "gpu": gpu_name,
        "gpu_vram_gb": round(total_vram_gb, 2) if total_vram_gb is not None else None,
        "compute_capability": capability,
        "low_vram_profile": low_vram_profile,
        "attention": attention,
        "text_encoder": workflow["128"]["inputs"]["clip_name"],
        "model": workflow["127"]["inputs"]["unet_name"],
        "video_vae": workflow["119"]["inputs"]["vae_name"],
        "audio_vae": workflow["120"]["inputs"]["vae_name"],
        "turbo_enabled": turbo_enabled,
        "turbo_family": turbo_family,
        "steps": steps,
        "sampler": sampler,
        "scheduler": scheduler,
        "cache_enabled": cache_enabled,
        "cache_threshold": threshold if cache_enabled else None,
        "loras": [
            {"name": _normalize_lora_name(str(item["name"])), "strength": float(item.get("strength", 1.0))}
            for item in list(payload.get("loras", []))
        ],
        "seed": workflow["129"]["inputs"]["noise_seed"],
    }


def _safe_name(name: str, fallback: str) -> str:
    candidate = Path(name or fallback).name
    if candidate in {"", ".", ".."}:
        candidate = fallback
    return f"{uuid.uuid4().hex[:12]}_{candidate}"


def _validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise InputError("Asset URLs must use HTTPS.")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except socket.gaierror as exc:
        raise InputError(f"Could not resolve asset host: {parsed.hostname}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise InputError("Asset URL resolves to a non-public address.")


def _decode_data(value: str) -> bytes:
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        data = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise InputError("Asset data is not valid base64.") from exc
    if len(data) > MAX_ASSET_BYTES:
        raise InputError("Asset exceeds MAX_ASSET_MB.")
    return data


def _materialize_asset(spec: dict[str, Any], fallback_name: str, *, preserve_name: bool = False) -> str:
    if not isinstance(spec, dict):
        raise InputError("Assets must be objects containing data, url, or volume_path.")
    supplied_name = str(spec.get("name", ""))
    if preserve_name:
        if not supplied_name:
            raise InputError("Raw-workflow assets require a name.")
        filename = Path(supplied_name).name
    else:
        filename = _safe_name(supplied_name, fallback_name)
    destination = INPUT_DIR / filename
    if "data" in spec:
        destination.write_bytes(_decode_data(str(spec["data"])))
    elif "volume_path" in spec:
        root = VOLUME_ROOT.resolve()
        source = Path(str(spec["volume_path"])).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise InputError(f"Asset volume_path is outside the attached network volume: {root}") from exc
        if not source.is_file():
            raise InputError(f"Asset volume_path does not exist: {source}")
        if source.stat().st_size > MAX_ASSET_BYTES:
            raise InputError("Asset exceeds MAX_ASSET_MB.")
        shutil.copy2(source, destination)
    elif "url" in spec:
        url = str(spec["url"])
        _validate_remote_url(url)
        with requests.get(url, stream=True, timeout=(15, 300), allow_redirects=True) as response:
            response.raise_for_status()
            total = 0
            with destination.open("wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_ASSET_BYTES:
                        raise InputError("Asset exceeds MAX_ASSET_MB.")
                    handle.write(chunk)
    else:
        raise InputError("Asset must contain data, url, or volume_path.")
    return filename


def _patch_fl2v(workflow: dict[str, Any], payload: dict[str, Any]) -> None:
    first = payload.get("first_frame")
    if not first:
        raise InputError("first_frame is required for fl2v")
    workflow["139"]["inputs"]["image"] = _materialize_asset(first, "first_frame.png")
    last = payload.get("last_frame")
    if last:
        workflow["901"] = {
            "inputs": {"image": _materialize_asset(last, "last_frame.png")},
            "class_type": "LoadImage",
            "_meta": {"title": "LAST FRAME"},
        }
        workflow["902"] = {
            "inputs": {
                "width": ["115", 0], "height": ["115", 1], "image": ["901", 0],
                "upscale_method": "nearest-exact", "keep_proportion": "resize",
                "pad_color": "0, 0, 0", "crop_position": "center",
                "divisible_by": 2, "device": "cpu",
            },
            "class_type": "ImageResizeKJv2",
            "_meta": {"title": "Resize Last Frame"},
        }
        workflow["182"]["inputs"]["last_frame"] = ["902", 0]


def _output_frame_count(payload: dict[str, Any]) -> int:
    raw = max(5, round(float(payload.get("duration", 7)) * 24))
    return raw + (5 - (raw % 17)) % 17


def _patch_r2v(workflow: dict[str, Any], payload: dict[str, Any]) -> None:
    references = payload.get("references") or []
    videos = payload.get("reference_videos") or []
    audios = payload.get("reference_audios") or ([] if not payload.get("audio") else [payload["audio"]])
    video_audio = payload.get("reference_video_audio") or []
    video_settings = payload.get("reference_video_settings") or []
    if not isinstance(references, list) or len(references) > 9:
        raise InputError("references must contain up to 9 images")
    if not isinstance(videos, list) or len(videos) > 3:
        raise InputError("reference_videos must contain up to 3 videos")
    if not isinstance(audios, list) or len(audios) > 3:
        raise InputError("reference_audios must contain up to 3 audio clips")
    if not isinstance(video_settings, list):
        raise InputError("reference_video_settings must be a list")
    if not references and not videos and not audios:
        raise InputError("r2v requires at least one image, video, or audio reference")
    for index, asset in enumerate(references):
        node_id = str(8000 + index)
        workflow[node_id] = {
            "inputs": {"image": _materialize_asset(asset, f"reference_{index + 1}.png")},
            "class_type": "LoadImage",
            "_meta": {"title": f"REFERENCE {index + 1}"},
        }
        workflow["136"]["inputs"][f"ref_images.ref_image_{index}"] = [node_id, 0]

    for index, asset in enumerate(videos):
        load_id = str(8200 + index)
        settings = video_settings[index] if index < len(video_settings) else {}
        if not isinstance(settings, dict):
            raise InputError(f"reference video {index + 1} settings must be an object")
        force_rate = float(settings.get("force_rate", 24))
        skip_first_frames = int(settings.get("skip_first_frames", settings.get("start_frame", 0)))
        select_every_nth = int(settings.get("select_every_nth", 1))
        if not 0 <= force_rate <= 120:
            raise InputError(f"reference video {index + 1} force_rate must be between 0 and 120")
        if skip_first_frames < 0:
            raise InputError(f"reference video {index + 1} skip_first_frames cannot be negative")
        if not 1 <= select_every_nth <= 1000:
            raise InputError(f"reference video {index + 1} select_every_nth must be between 1 and 1000")
        workflow[load_id] = {
            "inputs": {
                "video": _materialize_asset(asset, f"reference_video_{index + 1}.mp4"),
                "force_rate": force_rate,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": _output_frame_count(payload),
                "skip_first_frames": skip_first_frames,
                "select_every_nth": select_every_nth,
            },
            "class_type": "VHS_LoadVideo",
            "_meta": {"title": f"REFERENCE VIDEO {index + 1}"},
        }
        workflow["136"]["inputs"][f"ref_videos.ref_video_{index}"] = [load_id, 0]
        if index >= len(video_audio) or bool(video_audio[index]):
            workflow["136"]["inputs"][f"ref_video_audios.ref_video_audio_{index}"] = [load_id, 2]

    first_output_audio: list[Any] | None = None
    for index, asset in enumerate(audios):
        load_id = str(8500 + index * 2)
        trim_id = str(8501 + index * 2)
        workflow[load_id] = {
            "inputs": {"audio": _materialize_asset(asset, f"reference_audio_{index + 1}.wav")},
            "class_type": "LoadAudio",
            "_meta": {"title": f"REFERENCE AUDIO {index + 1}"},
        }
        workflow[trim_id] = {
            "inputs": {"start_index": 0.0, "duration": ["132", 0], "audio": [load_id, 0]},
            "class_type": "TrimAudioDuration",
            "_meta": {"title": f"Trim Reference Audio {index + 1}"},
        }
        reference = [trim_id, 0]
        workflow["136"]["inputs"][f"ref_audios.ref_audio_{index}"] = reference
        if first_output_audio is None:
            first_output_audio = reference
    if first_output_audio and bool(payload.get("use_reference_audio_as_output", False)):
        workflow["130"]["inputs"]["audio"] = first_output_audio

    edit = payload.get("masked_edit")
    if edit is not None:
        if not isinstance(edit, dict) or not edit.get("source") or not edit.get("mask") or not references:
            raise InputError("masked_edit requires source video, subject mask, and identity images")
        if videos or audios:
            raise InputError("Masked source is the target latent; supply only identity image references")
        width, height = int(edit.get("width", 0)), int(edit.get("height", 0))
        if min(width, height) < 32 or max(width, height) > 4096 or width % 32 or height % 32:
            raise InputError("Masked H3 canvas must be multiples of 32, between 32 and 4096")
        if width * height > 2_000_000:
            raise InputError("Masked H3 canvas exceeds 2 megapixels")
        workflow["136"]["inputs"].update(width=width, height=height)
        for node_id, key in (("8700", "source"), ("8701", "mask")):
            workflow[node_id] = {"class_type": "VHS_LoadVideo", "inputs": {
                "video": _materialize_asset(edit[key], f"samimate_{key}.mkv"),
                "force_rate": 24, "custom_width": 0, "custom_height": 0,
                "frame_load_cap": _output_frame_count(payload), "skip_first_frames": 0,
                "select_every_nth": 1}}
        workflow["8702"] = {"class_type": "SAMimateH3MaskedTarget", "inputs": {
            "target": ["136", 1], "source": ["8700", 0], "mask_frames": ["8701", 0], "vae": ["119", 0]}}
        workflow["125"]["inputs"]["latent_image"] = ["8702", 0]
        workflow["130"]["inputs"].pop("audio", None)


def build_preset(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    task = str(payload.get("task", "")).lower()
    if task == "r2v_masked":
        if not payload.get("masked_edit"):
            raise InputError("r2v_masked requires masked_edit")
        task = "r2v"
    requested_task = task
    if task == "t2v":
        task = "fl2v"
    if task not in TASKS:
        raise InputError("task must be t2v, fl2v or r2v")
    spec = TASKS[task]
    workflow = _load_template(spec["template"])
    metadata = _patch_common(workflow, spec, payload)
    if requested_task == "t2v":
        for key in ("first_frame", "last_frame"):
            workflow["182"]["inputs"].pop(key, None)
        for node in ("139", "233", "902", "903"):
            workflow.pop(node, None)
    elif task == "fl2v":
        _patch_fl2v(workflow, payload)
    else:
        _patch_r2v(workflow, payload)
    metadata["task"] = requested_task
    return workflow, metadata


def _submit(workflow: dict[str, Any]) -> str:
    request_body = {"prompt": workflow, "client_id": uuid.uuid4().hex}
    try:
        response = requests.post(f"{COMFY_URL}/prompt", json=request_body, timeout=60)
    except requests.ConnectionError:
        _ensure_comfy_ready()
        response = requests.post(f"{COMFY_URL}/prompt", json=request_body, timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"ComfyUI rejected the workflow: {response.text[:4000]}")
    body = response.json()
    if "prompt_id" not in body:
        raise RuntimeError(f"ComfyUI did not return a prompt_id: {body}")
    return str(body["prompt_id"])


def _wait_for_history(prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        response = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        body = response.json()
        if prompt_id in body:
            history = body[prompt_id]
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI execution failed: {status}")
            return history
        time.sleep(1)
    try:
        requests.post(f"{COMFY_URL}/interrupt", timeout=10)
    except requests.RequestException:
        pass
    raise TimeoutError(f"ComfyUI job exceeded {JOB_TIMEOUT} seconds")


def _find_file_descriptors(value: Any) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, dict):
        if isinstance(value.get("filename"), str):
            found.append({
                "filename": value["filename"],
                "subfolder": str(value.get("subfolder", "")),
                "type": str(value.get("type", "output")),
            })
        else:
            for child in value.values():
                found.extend(_find_file_descriptors(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_file_descriptors(child))
    return found


def _descriptor_path(item: dict[str, str]) -> Path:
    roots = {"output": OUTPUT_DIR, "input": INPUT_DIR, "temp": COMFY_ROOT / "temp"}
    root = roots.get(item["type"], OUTPUT_DIR).resolve()
    path = (root / item["subfolder"] / Path(item["filename"]).name).resolve()
    if root not in path.parents and path != root:
        raise RuntimeError("ComfyUI returned an unsafe output path.")
    if not path.is_file():
        raise RuntimeError(f"ComfyUI output is missing: {path.name}")
    return path


def _s3_config() -> dict[str, str] | None:
    bucket = os.environ.get("S3_BUCKET") or os.environ.get("BUCKET_NAME")
    if not bucket:
        return None
    return {
        "bucket": bucket,
        "endpoint": os.environ.get("S3_ENDPOINT_URL") or os.environ.get("BUCKET_ENDPOINT_URL", ""),
        "access": os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("BUCKET_ACCESS_KEY_ID", ""),
        "secret": os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("BUCKET_SECRET_ACCESS_KEY", ""),
        "region": os.environ.get("S3_REGION") or os.environ.get("BUCKET_REGION", "us-east-1"),
        "prefix": os.environ.get("S3_PREFIX", "minimax-h3"),
        "public": os.environ.get("S3_PUBLIC_URL", "").rstrip("/"),
    }


def _upload_s3(path: Path, job_id: str) -> dict[str, Any]:
    import boto3

    config = _s3_config()
    if not config:
        raise RuntimeError("S3 is not configured.")
    kwargs: dict[str, Any] = {"region_name": config["region"]}
    if config["endpoint"]:
        kwargs["endpoint_url"] = config["endpoint"]
    if config["access"]:
        kwargs["aws_access_key_id"] = config["access"]
        kwargs["aws_secret_access_key"] = config["secret"]
    client = boto3.client("s3", **kwargs)
    key = f"{config['prefix'].strip('/')}/{job_id}/{path.name}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    client.upload_file(str(path), config["bucket"], key, ExtraArgs={"ContentType": content_type})
    if config["public"]:
        url = f"{config['public']}/{key}"
    else:
        url = client.generate_presigned_url(
            "get_object", Params={"Bucket": config["bucket"], "Key": key}, ExpiresIn=86400
        )
    return {"filename": path.name, "type": "s3_url", "data": url, "size": path.stat().st_size}


def _deliver(path: Path, payload: dict[str, Any], job_id: str, index: int) -> dict[str, Any]:
    upload_urls = payload.get("output_upload_urls") or []
    if not isinstance(upload_urls, list):
        raise InputError("output_upload_urls must be an array")
    if index < len(upload_urls):
        url = str(upload_urls[index])
        _validate_remote_url(url)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as handle:
            response = requests.put(url, data=handle, headers={"Content-Type": content_type}, timeout=1800)
        response.raise_for_status()
        return {"filename": path.name, "type": "presigned_upload", "size": path.stat().st_size}
    if _s3_config():
        return _upload_s3(path, job_id)
    if path.stat().st_size <= MAX_BASE64_BYTES and not (payload.get("output_layout") == "flat_outputs" and os.environ.get("OUTPUT_VOLUME_DIR")):
        return {
            "filename": path.name,
            "type": "base64",
            "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "size": path.stat().st_size,
        }
    volume_dir = os.environ.get("OUTPUT_VOLUME_DIR")
    if volume_dir:
        flat = payload.get("output_layout") == "flat_outputs"
        target_dir = FLAT_OUTPUT_DIR if flat else Path(volume_dir) / job_id
        target_dir.mkdir(parents=True, exist_ok=True)
        # UUID suffix also prevents collisions if the same job is retried.
        filename = f"{path.stem}_{uuid.uuid4().hex}{path.suffix}" if flat else path.name
        target = target_dir / filename
        shutil.copy2(path, target)
        return {"filename": filename, "type": "volume_path", "data": str(target), "size": path.stat().st_size}
    raise RuntimeError(
        f"Output {path.name} is too large for base64. Configure S3, OUTPUT_VOLUME_DIR, "
        "or provide output_upload_urls."
    )


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input")
    if not isinstance(payload, dict):
        return {"error": "input must be an object"}
    try:
        _ensure_comfy_ready()
        if "workflow" in payload:
            workflow = payload["workflow"]
            if not isinstance(workflow, dict):
                raise InputError("workflow must be an API-format object")
            metadata = {"task": "custom"}
            for index, asset in enumerate(payload.get("assets", [])):
                _materialize_asset(asset, f"asset_{index}", preserve_name=True)
        else:
            workflow, metadata = build_preset(payload)
        prompt_id = _submit(workflow)
        history = _wait_for_history(prompt_id)
        descriptors = _find_file_descriptors(history.get("outputs", {}))
        unique: list[Path] = []
        seen: set[Path] = set()
        for descriptor in descriptors:
            path = _descriptor_path(descriptor)
            if path not in seen:
                seen.add(path)
                unique.append(path)
        if not unique:
            raise RuntimeError("The workflow completed but returned no output files.")
        job_id = str(job.get("id") or prompt_id)
        files = [_deliver(path, payload, job_id, index) for index, path in enumerate(unique)]
        return {"files": files, "metadata": metadata, "prompt_id": prompt_id}
    except InputError as exc:
        return {"error": str(exc), "error_type": "invalid_input"}
    except Exception as exc:
        return {"error": str(exc), "error_type": type(exc).__name__}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
