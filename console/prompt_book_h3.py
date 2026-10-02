"""Prompt book catalog bridge for the MiniMax H3 endpoint.

The book at http://127.0.0.1:8788 supplies the selected stills, source clip,
and baked prompt. Those inputs are passed through the same H3 Ref2V request
path used by the main H3 tab; the worker builds the dated Ref2V graph from
the request and does not need a prompt-book-specific workflow upload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
import imageio_ffmpeg

PROMPT_BOOK_BASE = os.environ.get("PROMPT_BOOK_URL", "http://127.0.0.1:8788").rstrip("/")
WORKFLOW_ROOT = Path(os.environ.get("PROMPT_BOOK_WORKFLOW_DIR", r"Z:\Grok\workflows\sla-bench"))

BETA5_MODEL = "10Eros_Max_h3_TURBO-hybrid_beta5.safetensors"

# Legacy API clients may still supply main-H3 generation settings. The browser
# now sends Prompt Book's own controls, and book references always replace stale
# main-H3 photo/video/prompt fields.
H3_GENERATION_FIELDS = {
    "steps", "sampler", "scheduler", "megapixels", "final_megapixels",
    "latent_upscale", "rtx_upscale", "pass1_split", "second_pass_sigma",
    "latent_upscale_model", "cache_enabled", "cache_threshold", "attention",
    "seed", "seed_random", "turbo_enabled", "use_larry", "turbo_family",
    "turbo_lora", "turbo_strength", "pdd_enabled", "sampling_preset",
    "sparse_keep_percent", "sparse_tau", "sparse_start_percent",
    "sparse_end_percent", "sparse_trained_weights", "ref2va_model",
    "text_encoder", "video_vae", "audio_vae", "clip_projection",
    "advanced_json", "delivery", "max_runtime_seconds",
}

PROMPT_BOOK_WORKFLOWS: dict[str, dict[str, Any]] = {
    "ref2v_4step": {
        "label": "Ref2V Sage+Sol 4-step",
        "filename": "3080ti_best_both_ref2v_4step.json",
        "key": "workflows/prompt-book/ref2v_solsage_bothpasses_4step.json",
        "steps": 4,
        "pass1_split": 3,
        "second_pass_sigma": 2,
    },
    "ref2v_8step": {
        "label": "Ref2V Sage+Sol 8-step",
        "filename": "3080ti_best_both_ref2v_8step.json",
        "key": "workflows/prompt-book/ref2v_solsage_bothpasses_8step.json",
        "steps": 8,
        "pass1_split": 4,
        "second_pass_sigma": 4,
    },
}

# Beta5 already has the turbo bake, so none of these add Larry or a Turbo LoRA.
# 4–8 steps is the useful range for that bake. The second pass is the expensive
# part of a 5090 bill, so Fast keeps a single pass.
PROMPT_BOOK_PRESETS: dict[str, dict[str, Any]] = {
    "ours": {
        "label": "Ours — two-pass",
        "detail": "Local Ref2V shape on beta5. Euler, Sage, 0.2 MP, then a 1 MP latent refine.",
        "steps": 8,
        "latent_upscale": True,
        "megapixels": 0.2,
        "final_megapixels": 1.0,
        "cache_threshold": 0.244,
        "graph": "ref2v_8step",
    },
    "fast5090": {
        "label": "5090 fast",
        "detail": "Shortest 5090 job. Beta5, one pass, no latent refine and no RTX.",
        "steps": 4,
        "latent_upscale": False,
        "megapixels": 0.5,
        "final_megapixels": 0.5,
        "cache_threshold": 0.2,
        "graph": "ref2v_4step",
    },
    "quality5090": {
        "label": "5090 quality",
        "detail": "Eight beta5 steps, then the 1 MP latent refine. Sage, no Larry, no RTX.",
        "steps": 8,
        "latent_upscale": True,
        "megapixels": 0.2,
        "final_megapixels": 1.0,
        "cache_threshold": 0.2,
        "graph": "ref2v_8step",
    },
}

_SAMPLING = json.loads((Path(__file__).resolve().parent / "h3_sampling.json").read_text(encoding="utf-8"))
H3_SAMPLERS = set(_SAMPLING["samplers"])
H3_SCHEDULERS = set(_SAMPLING["schedulers"])

ASPECTS = (
    (1, 1, "1:1 (Square)"),
    (2, 3, "2:3 (Portrait Photo)"),
    (3, 2, "3:2 (Photo)"),
    (3, 4, "3:4 (Portrait Standard)"),
    (4, 3, "4:3 (Standard)"),
    (9, 16, "9:16 (Portrait Widescreen)"),
    (16, 9, "16:9 (Widescreen)"),
    (21, 9, "21:9 (Ultrawide)"),
)


def workflow_spec(workflow_id: str) -> dict[str, Any]:
    spec = PROMPT_BOOK_WORKFLOWS.get(str(workflow_id or "").strip())
    if not spec:
        raise ValueError("Choose the 4-step or 8-step Ref2V workflow")
    return spec


def preset_spec(workflow_id: str) -> dict[str, Any]:
    preset = PROMPT_BOOK_PRESETS.get(str(workflow_id or "ours").strip() or "ours")
    if not preset:
        names = ", ".join(PROMPT_BOOK_PRESETS)
        raise ValueError(f"Choose a workflow: {names}")
    return preset


def parse_steps(value: Any, default: int) -> int:
    raw = default if value in (None, "") else value
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Steps must be a positive whole number up to 1000") from exc
    if isinstance(raw, bool) or not number.is_integer() or not 1 <= number <= 1000:
        raise ValueError("Steps must be a positive whole number up to 1000")
    return int(number)


ATTENTION_MODES = {"auto", "native", "sage", "sol-attn", "sla", "vsa"}
ASPECT_VALUES = {item[2] for item in ASPECTS}


def parse_number(value: Any, default: float, low: float, high: float, label: str, whole: bool = False) -> float:
    raw = default if value in (None, "") else value
    try:
        number = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if isinstance(raw, bool) or number != number or not low <= number <= high or (whole and not number.is_integer()):
        kind = "a whole number" if whole else "a number"
        raise ValueError(f"{label} must be {kind} from {low} to {high}")
    return int(number) if whole else number


def parse_sampling(value: Any, allowed: set[str], default: str, label: str) -> str:
    text = str(default if value in (None, "") else value).strip().lower()
    if text not in allowed:
        raise ValueError(f"Unsupported H3 {label}: {text}")
    return text


def workflow_file(spec: dict[str, Any]) -> Path:
    path = WORKFLOW_ROOT / str(spec["filename"])
    if not path.is_file():
        raise ValueError(f"Workflow file is missing: {path}")
    return path


def workflow_catalog() -> list[dict[str, Any]]:
    items = []
    for workflow_id, spec in PROMPT_BOOK_WORKFLOWS.items():
        path = WORKFLOW_ROOT / str(spec["filename"])
        items.append({
            "id": workflow_id,
            "label": spec["label"],
            "filename": spec["filename"],
            "key": spec["key"],
            "volume_path": "/runpod-volume/" + spec["key"],
            "pod_path": "/workspace/" + spec["key"],
            "steps": spec["steps"],
            "pass1_split": spec["pass1_split"],
            "second_pass_sigma": spec["second_pass_sigma"],
            "local_bytes": path.stat().st_size if path.is_file() else 0,
            "local_ready": path.is_file(),
        })
    return items


def object_size(helper: Any, key: str) -> int | None:
    """Read a network-volume object size. HeadObject is tried first; some gateways only allow GetObject."""
    try:
        meta = helper.call("head_object", Bucket=helper.bucket, Key=key)
        return int(meta.get("ContentLength") or 0)
    except Exception:
        pass
    try:
        response = helper.call("get_object", Bucket=helper.bucket, Key=key, Range="bytes=0-0")
        body = response.get("Body")
        if body is not None:
            try:
                body.close()
            except Exception:
                pass
        header = str(response.get("ContentRange") or response.get("Content-Range") or "")
        if "/" in header:
            return int(header.rsplit("/", 1)[-1])
        if response.get("ContentLength"):
            return int(response["ContentLength"])
    except Exception:
        return None
    return None


def upload_prompt_book_workflows(helper: Any, force: bool = False) -> list[dict[str, Any]]:
    results = []
    for item in workflow_catalog():
        spec = workflow_spec(item["id"])
        path = workflow_file(spec)
        local_size = path.stat().st_size
        remote = None if force else object_size(helper, item["key"])
        uploaded = remote != local_size
        if uploaded:
            helper.upload_to_key(path, item["key"])
            remote = object_size(helper, item["key"])
            if remote != local_size:
                raise ValueError(
                    f"Volume object {item['key']} is {remote} bytes after upload; the local workflow is {local_size} bytes"
                )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        results.append({
            **item,
            "local_bytes": local_size,
            "remote_bytes": remote,
            "on_volume": remote == local_size,
            "uploaded": uploaded,
            "sha256": digest,
        })
    return results


def book_json(path: str, timeout: int = 60) -> Any:
    url = PROMPT_BOOK_BASE + path
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise ValueError(f"Prompt book at {PROMPT_BOOK_BASE} is not reachable. Start it on port 8788.") from exc
    if response.status_code >= 400:
        detail = ""
        try:
            body = response.json()
            detail = str(body.get("error") or "")[:300]
        except Exception:
            detail = response.text[:300]
        raise ValueError(detail or f"Prompt book returned HTTP {response.status_code}")
    return response.json()


def fetch_media(kind: str, name: str) -> tuple[bytes, str]:
    safe = Path(str(name or "")).name
    if kind not in {"subject", "scene"} or not safe or safe != str(name):
        raise ValueError("Unknown prompt book media")
    url = f"{PROMPT_BOOK_BASE}/media/{kind}/{safe}"
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        raise ValueError(f"Prompt book at {PROMPT_BOOK_BASE} is not reachable. Start it on port 8788.") from exc
    if response.status_code >= 400:
        raise ValueError("Prompt book media was not found")
    if len(response.content) > 20 * 1024 * 1024:
        raise ValueError("Prompt book preview is too large")
    ctype = (response.headers.get("Content-Type") or "application/octet-stream").split(";", 1)[0]
    return response.content, ctype


def video_has_audio(path: str) -> bool:
    """Do not wire VHS audio from a silent scene; VHS fails on missing streams."""
    source = Path(path)
    if not source.is_file():
        return True  # Let the shared asset validator report missing files.
    try:
        probe = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(source)],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Could not inspect scene audio: {source.name}") from exc
    details = (probe.stderr or "") + (probe.stdout or "")
    if "Input #0" not in details:
        raise ValueError(f"Could not inspect scene audio: {source.name}")
    return bool(re.search(r"Stream #.*\bAudio:", details))


def flag(data: dict[str, Any], name: str, default: bool) -> bool:
    if name not in data:
        return default
    value = data[name]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def nearest_aspect(width: Any, height: Any) -> str:
    try:
        w = float(width or 0)
        h = float(height or 0)
    except (TypeError, ValueError):
        return "16:9 (Widescreen)"
    if w <= 0 or h <= 0:
        return "16:9 (Widescreen)"
    ratio = w / h
    return min(ASPECTS, key=lambda item: abs(ratio - (item[0] / item[1])))[2]


def trim_window(data: dict[str, Any], scene_duration: Any) -> tuple[float, float]:
    """Return the trim start and selected length in seconds."""
    try:
        total = float(scene_duration or 0)
    except (TypeError, ValueError):
        total = 0
    try:
        start = float(data.get("trim_start") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Trim start must be a number of seconds") from exc
    end_raw = data.get("trim_end")
    if end_raw in (None, "", "full"):
        end = total
    else:
        try:
            end = float(end_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Trim end must be a number of seconds") from exc
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Trim times must be finite numbers")
    if start < 0:
        raise ValueError("Trim start cannot be negative")
    if total > 0:
        if start >= total:
            raise ValueError("Trim start must be before the end of the scene")
        if end_raw not in (None, "", "full"):
            end = min(end, total)
    if end_raw not in (None, "", "full") and end <= start:
        raise ValueError("Trim end must be after trim start")
    if end > start:
        return start, end - start
    return start, total - start if total > start else total


def clamp_duration(seconds: Any) -> tuple[float, bool]:
    try:
        value = float(seconds or 0)
    except (TypeError, ValueError):
        value = 0
    if value < 1:
        return 1.0, False
    if value > 15:
        return 15.0, True
    return round(value, 3), False


def chosen_duration(data: dict[str, Any], start: float, window: float, scene_duration: Any) -> tuple[float, bool]:
    """Output length. An explicit duration wins. Otherwise the trim window is used."""
    explicit = data.get("duration")
    if explicit in (None, ""):
        return clamp_duration(window if window > 0 else scene_duration)
    try:
        requested = float(explicit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Duration must be a number of seconds") from exc
    duration, clamped = clamp_duration(requested)
    try:
        total = float(scene_duration or 0)
    except (TypeError, ValueError):
        total = 0
    if total > 0:
        remaining = max(0.0, total - start)
        if remaining > 0 and duration > remaining:
            duration = 1.0 if remaining < 1 else round(remaining, 3)
            clamped = True
    return duration, clamped


def _card(items: list[dict[str, Any]], card_id: str, label: str) -> dict[str, Any]:
    for item in items:
        if str(item.get("id")) == str(card_id):
            return item
    raise ValueError(f"Unknown {label}: {card_id}")


def photo_paths(subject: dict[str, Any], subject_b: dict[str, Any] | None) -> list[str]:
    ordered: list[str] = []

    def add(path: Any) -> None:
        text = str(path or "").strip()
        if text and text not in ordered:
            ordered.append(text)

    add(subject.get("path"))
    if subject_b:
        add(subject_b.get("path"))
    for person in (subject, subject_b):
        if not person:
            continue
        for image in person.get("images") or []:
            if image.get("primary"):
                continue
            add(image.get("path"))
    if not ordered:
        raise ValueError("The selected subject has no still")
    return ordered[:4]


def slim_catalog(raw: dict[str, Any]) -> dict[str, Any]:
    subject_folders = {str(item.get("id")): item.get("name") or "" for item in raw.get("subject_folders") or []}
    scene_folders = {str(item.get("id")): item.get("name") or "" for item in raw.get("scene_folders") or []}
    subjects = []
    for item in raw.get("subjects") or []:
        subjects.append({
            "id": item.get("id"),
            "label": item.get("label") or item.get("id"),
            "folder": subject_folders.get(str(item.get("folder_id") or ""), ""),
            "pair": bool(item.get("pair")),
            "thumb": Path(str(item.get("thumb") or "")).name,
            "image_count": len(item.get("images") or []),
        })
    videos = []
    for item in raw.get("videos") or []:
        videos.append({
            "id": item.get("id"),
            "label": item.get("label") or item.get("id"),
            "folder": scene_folders.get(str(item.get("folder_id") or ""), ""),
            "category": item.get("category") or "",
            "duration": item.get("duration") or "",
            "duration_sec": item.get("duration_sec") or 0,
            "width": item.get("width") or 0,
            "height": item.get("height") or 0,
            "inspected": bool(item.get("inspected")),
            "act": item.get("act") or "",
        })
    return {
        "subjects": subjects,
        "videos": videos,
        "subject_folders": sorted({item["folder"] for item in subjects if item["folder"]}),
        "scene_folders": sorted({item["folder"] for item in videos if item["folder"]}),
    }


def build_prompt_book_h3_request(
    data: dict[str, Any],
    catalog: dict[str, Any] | None = None,
    baked: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preset = preset_spec(str(data.get("workflow_id") or "ours"))
    steps = parse_steps(data.get("steps"), int(preset["steps"]))
    sampler = parse_sampling(data.get("sampler"), H3_SAMPLERS, "euler", "sampler")
    scheduler = parse_sampling(data.get("scheduler"), H3_SCHEDULERS, "simple", "scheduler")
    if sampler == "h3_turbo":
        sampler = "euler"
    latent = flag(data, "latent_upscale", bool(preset["latent_upscale"])) if "latent_upscale" in data else bool(preset["latent_upscale"])
    rtx = flag(data, "rtx_upscale", False)
    cache_enabled = flag(data, "cache_enabled", True)
    attention = parse_sampling(data.get("attention"), ATTENTION_MODES, "sage", "attention")
    megapixels = parse_number(data.get("megapixels"), float(preset["megapixels"]), 0.2, 2.0, "Low res MP")
    final_megapixels = parse_number(data.get("final_megapixels"), float(preset["final_megapixels"]), 0.2, 2.0, "Latent upscale MP")
    cache_threshold = parse_number(data.get("cache_threshold"), float(preset["cache_threshold"]), 0.0, 1.0, "Cache threshold")
    split = int(parse_number(data.get("pass1_split"), max(1, steps - 1), 1, 1000, "First-pass split", whole=True))
    second_sigma = int(parse_number(data.get("second_pass_sigma"), 2, 1, 5, "Second-pass sigmas", whole=True))
    if latent:
        if split > steps:
            raise ValueError("First-pass split cannot exceed the step count")
        if split == steps and second_sigma == 4:
            raise ValueError("Remaining sigmas require a first-pass split below the step count")
    if attention in {"sol-attn", "sla", "vsa"} and cache_enabled:
        raise ValueError("Turn off FirstBlockCache when using Sol, SLA, or VSA")
    if attention in {"sla", "vsa"} and not flag(data, "sparse_trained_weights", False):
        raise ValueError("SLA and VSA need the trained-weights confirmation")
    aspect = str(data.get("aspect_ratio") or "").strip()
    if aspect not in ASPECT_VALUES:
        aspect = ""
    subject_id = str(data.get("subject_id") or "").strip()
    video_id = str(data.get("video_id") or "").strip()
    if not subject_id or not video_id:
        raise ValueError("Choose one subject card and one scene card")
    threesome = flag(data, "threesome", False)
    subject_b_id = str(data.get("subject_b") or "").strip() if threesome else ""
    if subject_b_id == subject_id:
        subject_b_id = ""
    panties = flag(data, "panties", True)
    sound = flag(data, "sound", True)
    identity_only = flag(data, "identity_only", False)
    bare_breasts = flag(data, "bare_breasts", False)
    test_mode = flag(data, "test_mode", False)
    video_editing = flag(data, "video_editing", True)
    beta = flag(data, "beta", False)
    if catalog is None:
        catalog = book_json("/api/catalog", timeout=120)
    if baked is None:
        baked = book_json("/api/prompt?" + urlencode({
            "subject": subject_id,
            "video": video_id,
            "panties": "1" if panties else "0",
            "sound": "1" if sound else "0",
            "identity_only": "1" if identity_only else "0",
            "bare_breasts": "1" if bare_breasts else "0",
            "use_start_frame": "0",
            "test_mode": "1" if test_mode else "0",
            "video_editing": "1" if video_editing else "0",
            "threesome": "1" if threesome else "0",
            "subject_b": subject_b_id,
            "beta": "1" if beta else "0",
        }))
    subject = _card(list(catalog.get("subjects") or []), subject_id, "subject")
    scene = _card(list(catalog.get("videos") or []), video_id, "video")
    subject_b = _card(list(catalog.get("subjects") or []), subject_b_id, "subject") if subject_b_id else None
    photos = photo_paths(subject, subject_b)
    video = str(scene.get("file") or "").strip()
    if not video:
        raise ValueError("The selected scene has no video file")
    trim_start, trim_length = trim_window(data, scene.get("duration_sec"))
    duration, clamped = chosen_duration(data, trim_start, trim_length, scene.get("duration_sec"))
    trim_frames = math.ceil(trim_start * 24 - 1e-8)
    reference_duration = min(duration, trim_length) if trim_length > 0 else duration
    # Only output latents snap UP to 17k+5. A reference must stay inside the
    # selected interval; the native reference node handles its own DOWN snap.
    frame_cap = math.floor((trim_start + reference_duration) * 24 + 1e-8) - trim_frames
    if frame_cap < 5:
        raise ValueError("Reference trim needs at least five frames at 24 fps")
    prompt = str((baked or {}).get("prompt") or "").strip()
    if flag(data, "use_prompt_override", False):
        override = str(data.get("prompt") or "").strip()
        if override:
            prompt = override
    if not prompt:
        raise ValueError("Prompt book returned an empty prompt")
    slots = (photos + ["", "", "", ""])[:4]
    request: dict[str, Any] = {
        "source_workflow": "prompt_book",
        "task": "r2v",
        "workflow_variant": "ref2v_20260920",
        "prompt": prompt,
        "photo_paths": slots,
        "keyframe_positions": ["", "", "", ""],
        "reference_video_paths": [video],
        "reference_video_audio": [video_has_audio(video)],
        "reference_video_settings": [{
            "force_rate": 24,
            "start_frame": trim_frames,
            "select_every_nth": 1,
            "frame_load_cap": frame_cap,
        }],
        "use_reference_audio_as_output": False,
        "video_vae": "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        "reference_audio_paths": [],
        "use_multi_image": False,
        "latent_upscale": latent,
        "rtx_upscale": rtx,
        "turbo_enabled": False,
        "use_larry": False,
        "attention": attention,
        "sampler": sampler,
        "scheduler": scheduler,
        "cache_enabled": cache_enabled,
        "cache_threshold": cache_threshold,
        "steps": steps,
        "pass1_split": split,
        "second_pass_sigma": second_sigma,
        "diagnostic_frames": flag(data, "diagnostic_frames", False),
        "megapixels": megapixels,
        "final_megapixels": final_megapixels,
        "sparse_keep_percent": parse_number(data.get("sparse_keep_percent"), 10, 0.5, 95, "Sparse keep percent"),
        "sparse_tau": parse_number(data.get("sparse_tau"), 1.3, 0, 4, "Sol threshold"),
        "sparse_start_percent": parse_number(data.get("sparse_start_percent"), 0.2, 0, 1, "Sparse start"),
        "sparse_end_percent": parse_number(data.get("sparse_end_percent"), 1, 0, 1, "Sparse end"),
        "sparse_trained_weights": flag(data, "sparse_trained_weights", False),
        "ref2va_model": str(data.get("ref2va_model") or BETA5_MODEL).strip(),
        "duration": duration,
        "aspect_ratio": aspect or nearest_aspect(scene.get("width"), scene.get("height")),
        "filename_prefix": "H3",
        "seed_random": flag(data, "seed_random", True),
    }
    if request["sparse_start_percent"] > request["sparse_end_percent"]:
        raise ValueError("Sparse start must be at or before sparse end")
    request.update({
        # Use the normal H3 asset path. The endpoint worker mounts the input
        # objects and constructs the selected dated Ref2V graph itself.
        "delivery": "auto",
        # Ordinary workflow LoRAs are validated and normalized by the shared
        # H3 submission path, exactly like LoRAs from the main H3 tab.
        "loras": data.get("loras") or [],
        "advanced_json": "{}",
    })
    inherited = data.get("h3_generation")
    explicit_generation = {key: request[key] for key in (
        "steps", "sampler", "scheduler", "megapixels", "final_megapixels",
        "latent_upscale", "rtx_upscale", "pass1_split", "second_pass_sigma",
        "cache_enabled", "cache_threshold", "attention", "seed_random",
        "diagnostic_frames", "ref2va_model",
    ) if key in data}
    if inherited is not None:
        if not isinstance(inherited, dict):
            raise ValueError("H3 generation settings must be an object")
        request.update({key: inherited[key] for key in H3_GENERATION_FIELDS if key in inherited})
        main_loras = inherited.get("loras") or []
        extra_loras = data.get("loras") or []
        if not isinstance(main_loras, list) or not isinstance(extra_loras, list):
            raise ValueError("H3 LoRAs must be a list")
        loras = list(main_loras)
        def lora_key(item: dict[str, Any]) -> str:
            return str(item.get("name") or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()
        for extra in extra_loras:
            if not isinstance(extra, dict) or not str(extra.get("name") or "").strip():
                raise ValueError("Prompt Book LoRAs need a name")
            loras = [item for item in loras if not isinstance(item, dict) or
                     lora_key(item) != lora_key(extra)]
            loras.append(extra)
        request["loras"] = loras
        request["seed_random"] = inherited.get("seed_random", False)
    request.update(explicit_generation)
    # Prompt Book never sends a separate Turbo LoRA or its dedicated sampler,
    # even if an older caller passed enabled main-H3 settings.
    request["turbo_enabled"] = False
    request["use_larry"] = False
    request["turbo_family"] = "none"
    request["sampling_preset"] = "custom"
    request["pdd_enabled"] = False
    request["advanced_json"] = "{}"
    request.pop("turbo_lora", None)
    request.pop("turbo_strength", None)
    request["sampler"] = parse_sampling(request.get("sampler"), H3_SAMPLERS, "euler", "sampler")
    if request["sampler"] == "h3_turbo":
        request["sampler"] = "euler"
    request["scheduler"] = parse_sampling(request.get("scheduler"), H3_SCHEDULERS, "simple", "scheduler")
    request["steps"] = parse_steps(request.get("steps"), 8)
    request["latent_upscale"] = flag(request, "latent_upscale", False)
    request["rtx_upscale"] = flag(request, "rtx_upscale", False)
    request["pass1_split"] = int(parse_number(request.get("pass1_split"), 6, 1, 1000, "First-pass split", whole=True))
    request["second_pass_sigma"] = int(parse_number(request.get("second_pass_sigma"), 1, 1, 5, "Second-pass sigmas", whole=True))
    if request["latent_upscale"] and request["pass1_split"] > request["steps"]:
        raise ValueError("First-pass split cannot exceed the step count")
    if request["latent_upscale"] and request["pass1_split"] == request["steps"] and request["second_pass_sigma"] == 4:
        raise ValueError("Remaining sigmas require a first-pass split below the step count")
    request["attention"] = parse_sampling(request.get("attention"), ATTENTION_MODES, "sage", "attention")
    request["cache_enabled"] = flag(request, "cache_enabled", False)
    if request["attention"] in {"sol-attn", "sla", "vsa"} and request["cache_enabled"]:
        raise ValueError("Turn off FirstBlockCache when using Sol, SLA, or VSA")
    request["seed_random"] = flag(request, "seed_random", True)
    if not request["seed_random"]:
        request["seed"] = data.get("seed", inherited.get("seed", 0) if inherited is not None else 0)
    if isinstance(data.get("settings"), dict):
        request["settings"] = data["settings"]
    endpoint = str(data.get("h3_endpoint_id") or "").strip()
    if endpoint:
        request["h3_endpoint_id"] = endpoint
    meta = {
        "workflow_id": "main_h3" if inherited is not None else str(data.get("workflow_id") or "ours"),
        "workflow_label": "Prompt Book Ref2V (legacy H3 settings)" if inherited is not None else "Prompt Book Ref2V settings",
        "model": request.get("ref2va_model") or BETA5_MODEL,
        "subject": subject.get("label") or subject_id,
        "scene": scene.get("label") or video_id,
        "duration": duration,
        "duration_clamped": clamped,
        "trim_start": trim_start,
        "trim_end": trim_start + reference_duration,
        "reference_duration": reference_duration,
        "reference_frame_cap": frame_cap,
        "source_duration": scene.get("duration_sec") or 0,
        "aspect_ratio": request["aspect_ratio"],
        "photos": len(photos),
        "steps": request["steps"],
    }
    return request, meta
