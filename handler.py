from __future__ import annotations

import base64
import copy
import ipaddress
import json
import mimetypes
import os
import re
import secrets
import shutil
import socket
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
TEMPLATE_DIR = Path(os.environ.get("WORKFLOW_DIR", "/opt/minimax-h3/workflows"))
COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT_SECONDS", "3600"))
MAX_ASSET_BYTES = int(os.environ.get("MAX_ASSET_MB", "250")) * 1024 * 1024
MAX_BASE64_BYTES = int(os.environ.get("MAX_RETURN_BASE64_MB", "6")) * 1024 * 1024

BLACKWELL_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
UNIVERSAL_ENCODER = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

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


def _gpu_info() -> tuple[str, tuple[int, int] | None]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), tuple(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    return "unknown", None


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
    if capability and capability[0] >= 10 and _model_exists("text_encoders", BLACKWELL_ENCODER):
        return BLACKWELL_ENCODER
    if _model_exists("text_encoders", UNIVERSAL_ENCODER):
        return UNIVERSAL_ENCODER
    if _model_exists("text_encoders", BLACKWELL_ENCODER):
        return BLACKWELL_ENCODER
    raise RuntimeError("No MiniMax H3 text encoder is installed.")


def _sage_available() -> bool:
    try:
        import sageattention  # noqa: F401

        return True
    except Exception:
        return False


def _attention_mode(requested: str, capability: tuple[int, int] | None) -> str:
    configured = os.environ.get("ATTENTION_MODE", requested or "auto").lower()
    if configured not in {"auto", "sage", "native"}:
        raise InputError("attention must be auto, sage, or native")
    if configured == "sage":
        if not _sage_available():
            raise RuntimeError("SageAttention was requested but is not importable.")
        return "sage"
    if configured == "native":
        return "native"
    return "sage" if capability and capability[0] >= 8 and _sage_available() else "native"


def _normalize_lora_name(name: str) -> str:
    normalized = name.replace("\\", "/").strip("/")
    if not normalized or ".." in Path(normalized).parts or Path(normalized).is_absolute():
        raise InputError(f"Invalid LoRA name: {name!r}")
    return normalized


def _apply_loras(workflow: dict[str, Any], loras: list[dict[str, Any]]) -> None:
    previous: list[Any] = ["127", 0]
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
    workflow["152"]["inputs"]["model"] = previous


def _patch_common(workflow: dict[str, Any], spec: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    gpu_name, capability = _gpu_info()
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
    if not 4 <= steps <= 8:
        raise InputError("steps must be between 4 and 8 for the Turbo v4 LoRA")
    if not 0 <= threshold <= 1:
        raise InputError("cache_threshold must be between 0 and 1")

    workflow["132"]["inputs"]["value"] = duration
    workflow["115"]["inputs"]["megapixels"] = megapixels
    workflow["115"]["inputs"]["aspect_ratio"] = str(payload.get("aspect_ratio", "16:9 (Widescreen)"))
    workflow["124"]["inputs"]["steps"] = steps
    workflow["129"]["inputs"]["noise_seed"] = int(payload.get("seed", secrets.randbelow(2**63 - 1)))
    workflow[spec["cache"]]["inputs"]["threshold"] = threshold
    workflow["128"]["inputs"]["clip_name"] = _select_encoder(capability)

    attention = _attention_mode(str(payload.get("attention", "auto")), capability)
    if attention == "native":
        upstream = copy.deepcopy(workflow["145"]["inputs"]["model"])
        workflow[spec["cache"]]["inputs"]["model"] = upstream
        workflow.pop("145", None)

    _apply_loras(workflow, list(payload.get("loras", [])))
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(payload.get("filename_prefix", "MiniMax_H3"))).strip("._")
    workflow[spec["save"]]["inputs"]["filename_prefix"] = f"video/{prefix or 'MiniMax_H3'}"
    return {
        "gpu": gpu_name,
        "compute_capability": capability,
        "attention": attention,
        "text_encoder": workflow["128"]["inputs"]["clip_name"],
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
        raise InputError("Assets must be objects containing data or url.")
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
        raise InputError("Asset must contain data or url.")
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


def _patch_r2v(workflow: dict[str, Any], payload: dict[str, Any]) -> None:
    references = payload.get("references")
    if not isinstance(references, list) or not 1 <= len(references) <= 9:
        raise InputError("references must contain 1 to 9 images")
    for index, asset in enumerate(references):
        node_id = str(8000 + index)
        workflow[node_id] = {
            "inputs": {"image": _materialize_asset(asset, f"reference_{index + 1}.png")},
            "class_type": "LoadImage",
            "_meta": {"title": f"REFERENCE {index + 1}"},
        }
        workflow["136"]["inputs"][f"ref_images.ref_image_{index}"] = [node_id, 0]

    audio = payload.get("audio")
    if audio:
        workflow["8500"] = {
            "inputs": {"audio": _materialize_asset(audio, "reference_audio.wav")},
            "class_type": "LoadAudio",
            "_meta": {"title": "REFERENCE AUDIO"},
        }
        workflow["8501"] = {
            "inputs": {"start_time": 0.0, "duration": ["132", 0], "audio": ["8500", 0]},
            "class_type": "TrimAudioDuration",
            "_meta": {"title": "Trim Reference Audio"},
        }
        workflow["136"]["inputs"]["ref_audios.ref_audio_0"] = ["8501", 0]
        if bool(payload.get("use_reference_audio_as_output", False)):
            workflow["130"]["inputs"]["audio"] = ["8501", 0]


def build_preset(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    task = str(payload.get("task", "")).lower()
    if task not in TASKS:
        raise InputError("task must be fl2v or r2v")
    spec = TASKS[task]
    workflow = _load_template(spec["template"])
    metadata = _patch_common(workflow, spec, payload)
    if task == "fl2v":
        _patch_fl2v(workflow, payload)
    else:
        _patch_r2v(workflow, payload)
    metadata["task"] = task
    return workflow, metadata


def _submit(workflow: dict[str, Any]) -> str:
    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": uuid.uuid4().hex},
        timeout=60,
    )
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
    if path.stat().st_size <= MAX_BASE64_BYTES:
        return {
            "filename": path.name,
            "type": "base64",
            "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "size": path.stat().st_size,
        }
    volume_dir = os.environ.get("OUTPUT_VOLUME_DIR")
    if volume_dir:
        target_dir = Path(volume_dir) / job_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)
        return {"filename": path.name, "type": "volume_path", "data": str(target), "size": path.stat().st_size}
    raise RuntimeError(
        f"Output {path.name} is too large for base64. Configure S3, OUTPUT_VOLUME_DIR, "
        "or provide output_upload_urls."
    )


def handler(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("input")
    if not isinstance(payload, dict):
        return {"error": "input must be an object"}
    try:
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
