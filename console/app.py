from __future__ import annotations

import base64, importlib.util, ipaddress, json, os, re, secrets, shlex, shutil, socket, subprocess, time, uuid, webbrowser, hashlib, urllib.parse, random, sys, math, threading
from pathlib import Path
from typing import Any, Optional

import boto3, imageio_ffmpeg, requests
from botocore.client import Config
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

APP_VERSION = "web-v15.59-h3-turbo-families"
H3_SAMPLERS = {"h3_turbo", "res_multistep", "er_sde", "euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "deis", "uni_pc"}
H3_SCHEDULERS = {"simple", "beta", "normal", "sgm_uniform", "karras", "exponential", "ddim_uniform", "linear_quadratic", "kl_optimal"}
H3_INLINE_FILE_LIMIT_BYTES = 6 * 1024 * 1024
# RunPod rejects request bodies above 10 MiB. Keep enough room for prompts,
# settings, JSON escaping, and the rest of the payload after Base64 expansion.
H3_INLINE_BODY_BUDGET_BYTES = int(7.5 * 1024 * 1024)
APP_NAME = "RunPod Media Console Web"
APP_DIR = Path(__file__).parent
HOME = Path.home()
DATA_DIR = HOME / ".runpod_media_console_web"
DEFAULT_OUTPUT_DIR = APP_DIR / "outputs"
DEFAULT_CACHE_DIR = APP_DIR / "cache"
DEFAULT_SAM_OUTPUT_DIR = APP_DIR / "sam_outputs"
DEFAULT_PIPER_DIR = APP_DIR / "models" / "piper" / "en" / "en_US" / "lessac" / "medium"
DEFAULT_PIPER_MODEL = DEFAULT_PIPER_DIR / "en_US-lessac-medium.onnx"
DEFAULT_PIPER_CONFIG = DEFAULT_PIPER_DIR / "en_US-lessac-medium.onnx.json"
PIPER_MODEL_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_CONFIG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
TTS_SAPI_SCRIPT = APP_DIR / "tts_sapi.ps1"
UPLOAD_DIR = DEFAULT_CACHE_DIR / "uploads"
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
TEMP_DIR = DEFAULT_CACHE_DIR / "temp"
SETTINGS_PATH = DATA_DIR / "settings.json"
PROMPTS_PATH = DATA_DIR / "prompts.json"
H3_SUBJECTS_PATH = DATA_DIR / "h3_subjects.json"
FAVORITES_PATH = DATA_DIR / "favorites.json"
THUMB_DIR = DEFAULT_CACHE_DIR / "thumb_cache"
SAM_DIR = DEFAULT_SAM_OUTPUT_DIR
SAM_HISTORY_PATH = DATA_DIR / "sam_history.json"
PRESETS_PATH = DATA_DIR / "presets.json"
ENDPOINT_PROFILES_PATH = DATA_DIR / "endpoint_profiles.json"
JOB_HISTORY_PATH = DATA_DIR / "job_history.json"
LAN_TOKEN_PATH = DATA_DIR / "lan_access_token.txt"
GENERIC_UPLOAD_MAX_BYTES = 16 * 1024 * 1024 * 1024
SECRET_PATHS = (
    ("runpod_api_key",),
    ("s3_access_key_id",),
    ("s3_secret_access_key",),
    ("sam", "hf_token"),
    ("h3_storage", "s3_access_key_id"),
    ("h3_storage", "s3_secret_access_key"),
    ("model_download", "hf_token"),
    ("model_download", "civitai_token"),
)
JOB_HISTORY_LOCK = threading.Lock()
for d in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, THUMB_DIR, SAM_DIR):
    d.mkdir(parents=True, exist_ok=True)


def defaults() -> dict[str, Any]:
    return {
        "runpod_api_key":"", "infinite_endpoint_id":"", "wan_endpoint_id":"", "h3_endpoint_id":"s1e3i9book2zhh",
        "poll_interval_seconds":4, "timeout_seconds":3600,
        "s3_endpoint_url":"https://s3api-eu-ro-1.runpod.io", "s3_access_key_id":"", "s3_secret_access_key":"",
        "s3_bucket":"", "s3_region":"eu-ro-1", "s3_prefix":"input/runpod-media-console", "runpod_volume_root":"/runpod-volume",
        "delivery_mode":"auto", "browser_s3_storage":"h3", "output_dir":str(DEFAULT_OUTPUT_DIR),
        "cache_dir":str(DEFAULT_CACHE_DIR), "sam_output_dir":str(DEFAULT_SAM_OUTPUT_DIR), "cache_max_age_days":7, "cache_cleanup_enabled":True, "hf_cache_dir":"",
        "sam":{"backend":"local_sam3","device":"cuda","sam3_python":sys.executable,"hf_token":"","external_command":"","mask_fill":"black","make_masked_video":True,"masked_video_frame_cap":180,"video_start":"","video_end":"","source_path":"","mask_path":"","mask_video_path":"","overlay_path":"","cutout_path":"","masked_image_path":"","masked_video_path":""},
        "tts":{"engine":"windows_sapi","voice":"","rate":0,"volume":100,"piper_exe":"","piper_model":"","piper_config":"","output_path":"","inf_output_path":""},
        "infinitetalk":{"input_type":"image","person_count":"single","image_path":"","video_path":"","audio_path":"","audio2_path":"","width":512,"height":512,"max_frame":"","force_offload":True,"network_volume":True,"prompt":"A person talking naturally","advanced_json":"{}","video_start":"","video_end":"","video_frame_cap":"","video_crop":"","audio_start":"","audio_end":"","audio2_start":"","audio2_end":""},
        "wananimate":{"mode":"replace","image_path":"","video_path":"","mask_path":"","masked_video_path":"","mask_pretrimmed":False,"width":0,"height":0,"seed":12345,"fps":16,"cfg":1.0,"steps":6,"pose_estimation":True,"face_detection":True,"mask_editing":False,"network_volume":True,"prompt":"","negative_prompt":"blurry, low quality, distorted","control_points_enabled":False,"advanced_json":"{}","video_start":"","video_end":"","video_frame_cap":"","video_crop":"","delivery_override":"auto"},
        "h3":{"task":"fl2v","first_frame_path":"","last_frame_path":"","reference_paths":[],"reference_video_paths":[],"reference_video_audio":[],"reference_video_settings":[],"reference_audio_paths":[],"reference_subjects":[],"subject_generated_prefix":"","audio_path":"","use_reference_audio_as_output":False,"prompt":"A cinematic shot with natural, stable motion.","duration":2.0,"megapixels":0.2,"steps":6,"seed":123456789,"seed_random":False,"sampling_preset":"stock_turbo","sampler":"h3_turbo","scheduler":"simple","cache_enabled":False,"cache_threshold":0.18,"attention":"auto","aspect_ratio":"16:9 (Widescreen)","filename_prefix":"RunPod_Media_Console_H3","delivery":"auto","fl2va_model":"minimax_h3_fl2va_pruned_int8_convrot.safetensors","ref2va_model":"minimax_h3_ref2va_pruned_int8_convrot.safetensors","text_encoder":"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","video_vae":"minimax_h3_video_vae_fp16.safetensors","audio_vae":"minimax_h3_audio_vae_fp32.safetensors","clip_projection":"","turbo_enabled":True,"turbo_lora":"minimax_h3_turbo_v4_step600_ema.safetensors","turbo_strength":1.0,"loras":[],"advanced_json":"{}"},
        "h3_storage":{"network_volume_id":"vgc3ky6r6y","s3_endpoint_url":"","s3_access_key_id":"","s3_secret_access_key":"","s3_bucket":"vgc3ky6r6y","s3_region":"eu-ro-1","s3_prefix":"minimax-h3","runpod_volume_root":"/runpod-volume","comfyui_pod_id":""},
        "model_download":{"hf_token":"","civitai_token":""},
        "samimate":{"video_path":"","image_path":"","subject_prompt":"person","prompt":"","negative_prompt":"blurry, low quality, distorted","advanced_json":"{}","generation_backend":"wan","mask_frame_cap":0,"mode":"replace","delivery_override":"auto","width":832,"height":480,"seed":12345,"fps":16,"cfg":1.0,"steps":6,"pose_estimation":True,"face_detection":True,"mask_editing":True,"network_volume":True,"video_start":"","video_end":"","video_frame_cap":"","video_crop":"","use_wan_points":False},
    }


def load_json(path: Path, fallback: Any) -> Any:
    try:
        if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    except Exception: pass
    return fallback

def save_json(path: Path, data: Any) -> Any:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data

def append_limited_json(path: Path, item: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
    with JOB_HISTORY_LOCK:
        items = load_json(path, [])
        if not isinstance(items, list):
            items = []
        items.insert(0, item)
        items = items[:limit]
        payload = json.dumps(items, indent=2, ensure_ascii=False)
        last_error: Optional[OSError] = None
        for attempt in range(3):
            temp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                temp_path.write_text(payload, encoding="utf-8")
                os.replace(temp_path, path)
                return items
            except OSError as exc:
                last_error = exc
                temp_path.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1))
        assert last_error is not None
        raise last_error


def merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k,v in (b or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def nested_value(data: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def set_nested_value(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = data
    for key in path[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            child = {}
            current[key] = child
        current = child
    current[path[-1]] = value


def preserve_blank_secrets(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    safe_update = merge({}, update or {})
    clear = {str(item) for item in (safe_update.pop("clear_secrets", []) or [])}
    missing = object()
    for path in SECRET_PATHS:
        dotted = ".".join(path)
        incoming = nested_value(safe_update, path, missing)
        if dotted in clear:
            set_nested_value(safe_update, path, "")
        elif incoming is missing or incoming in (None, ""):
            set_nested_value(safe_update, path, nested_value(current, path, ""))
    return safe_update


def runtime_settings(update: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    current = settings()
    return merge(current, preserve_blank_secrets(current, update or {}))


def public_settings(data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    source = data or settings()
    out = merge({}, source)
    configured: dict[str, bool] = {}
    for path in SECRET_PATHS:
        configured[".".join(path)] = bool(str(nested_value(source, path, "") or "").strip())
        set_nested_value(out, path, "")
    out["secret_fields_configured"] = configured
    return out


def settings() -> dict[str, Any]:
    out = merge(defaults(), load_json(SETTINGS_PATH, {}))
    out.pop("wan22_endpoint_id", None)
    out.pop("wan22", None)
    return out
def save_settings(data: dict[str, Any]) -> dict[str, Any]:
    current = settings()
    out = merge(current, preserve_blank_secrets(current, data))
    SETTINGS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    apply_cache_settings(out)
    return out


def lan_access_token() -> str:
    configured = str(os.environ.get("RUNPOD_MEDIA_CONSOLE_TOKEN") or "").strip()
    if configured:
        return configured
    try:
        existing = LAN_TOKEN_PATH.read_text(encoding="utf-8").strip()
        if len(existing) >= 24:
            return existing
    except Exception:
        pass
    token = secrets.token_urlsafe(32)
    LAN_TOKEN_PATH.write_text(token, encoding="utf-8")
    return token


LAN_ACCESS_TOKEN = lan_access_token()


def is_loopback_client(host: str) -> bool:
    host = str(host or "").strip().split("%", 1)[0]
    if host.lower() in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def request_has_lan_access(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    if is_loopback_client(client_host):
        return True
    authorization = str(request.headers.get("authorization") or "")
    supplied = ""
    if authorization.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(authorization.split(None, 1)[1]).decode("utf-8")
            username, supplied = decoded.split(":", 1)
            if username != "runpod":
                return False
        except Exception:
            return False
    elif authorization.lower().startswith("bearer "):
        supplied = authorization.split(None, 1)[1].strip()
    return bool(supplied) and secrets.compare_digest(supplied, LAN_ACCESS_TOKEN)

def apply_cache_settings(s: Optional[dict[str, Any]] = None) -> None:
    global UPLOAD_DIR, TEMP_DIR, THUMB_DIR, SAM_DIR
    cfg = s or settings()
    root = Path(cfg.get("cache_dir") or DEFAULT_CACHE_DIR).expanduser()
    sam_root = Path(cfg.get("sam_output_dir") or DEFAULT_SAM_OUTPUT_DIR).expanduser()
    UPLOAD_DIR = root / "uploads"
    TEMP_DIR = root / "temp"
    THUMB_DIR = root / "thumb_cache"
    SAM_DIR = sam_root
    for d in (root, UPLOAD_DIR, TEMP_DIR, THUMB_DIR, SAM_DIR):
        d.mkdir(parents=True, exist_ok=True)

def dir_size(path: Path) -> int:
    total = 0
    if not path.exists(): return 0
    for p in path.rglob("*"):
        try:
            if p.is_file(): total += p.stat().st_size
        except Exception:
            pass
    return total

def cleanup_cache(s: Optional[dict[str, Any]] = None, force: bool = False) -> dict[str, Any]:
    cfg = s or settings()
    apply_cache_settings(cfg)
    keep_days = max(float(cfg.get("cache_max_age_days") or 0), 0.0)
    cutoff = time.time() - keep_days * 86400
    roots = [TEMP_DIR, THUMB_DIR, UPLOAD_DIR]
    removed_files = 0
    removed_bytes = 0
    for root in roots:
        if not root.exists(): continue
        for p in sorted(root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            try:
                if p.is_file() and (force or p.stat().st_mtime < cutoff):
                    size = p.stat().st_size
                    p.unlink()
                    removed_files += 1
                    removed_bytes += size
                elif p.is_dir():
                    try: p.rmdir()
                    except OSError: pass
            except Exception:
                pass
    return {"removed_files": removed_files, "removed_bytes": removed_bytes, "removed_mb": round(removed_bytes / (1024 * 1024), 3)}

def cache_info() -> dict[str, Any]:
    cfg = settings()
    apply_cache_settings(cfg)
    parts = {
        "uploads": UPLOAD_DIR,
        "temp": TEMP_DIR,
        "thumb_cache": THUMB_DIR,
    }
    items = {name: {"path": str(path), "size_mb": round(dir_size(path) / (1024 * 1024), 3)} for name, path in parts.items()}
    total = sum(v["size_mb"] for v in items.values())
    return {
        "cache_dir": str(Path(cfg.get("cache_dir") or DEFAULT_CACHE_DIR).expanduser()),
        "sam_output_dir": str(Path(cfg.get("sam_output_dir") or DEFAULT_SAM_OUTPUT_DIR).expanduser()),
        "sam_outputs_mb": round(dir_size(SAM_DIR) / (1024 * 1024), 3),
        "max_age_days": cfg.get("cache_max_age_days", 7),
        "cleanup_enabled": as_bool(cfg.get("cache_cleanup_enabled"), True),
        "total_mb": round(total, 3),
        "items": items,
        "hf_cache_dir": cfg.get("hf_cache_dir") or os.environ.get("HF_HOME") or str(HOME / ".cache" / "huggingface"),
    }

def clean_name(name: str) -> str: return re.sub(r"[^A-Za-z0-9._ -]+", "_", Path(name).name).strip() or f"file_{uuid.uuid4().hex}"
def short_media_name(kind: str, ext: str = ".bin") -> str:
    safe_kind = re.sub(r"[^A-Za-z0-9_-]+", "_", str(kind or "media")).strip("_") or "media"
    suffix = ext if str(ext or "").startswith(".") else f".{ext or 'bin'}"
    return f"{safe_kind}_{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix.lower()}"
def is_url(v: str) -> bool: return str(v).lower().startswith(("http://","https://"))
def is_s3_uri(v: str) -> bool: return str(v).lower().startswith("s3://")
def media_type_for_path(path: str | Path) -> str:
    media_types = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}
    return media_types.get(Path(path).suffix.lower(), "application/octet-stream")

def ffmpeg_bin() -> str: return imageio_ffmpeg.get_ffmpeg_exe()

def has_video_stream(path: str|Path) -> bool:
    p = str(path)
    if not Path(p).exists(): return False
    proc = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", p], capture_output=True, text=True)
    text = (proc.stderr or "") + (proc.stdout or "")
    return bool(re.search(r"Stream #.*Video:", text) or re.search(r"\bVideo:", text))

def duration(path: str|Path) -> float:
    proc = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    text = (proc.stderr or "") + (proc.stdout or "")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    return int(m.group(1))*3600 + int(m.group(2))*60 + float(m.group(3)) if m else 0.0

def video_probe(path: str|Path) -> dict[str, Any]:
    proc = subprocess.run([ffmpeg_bin(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    text = (proc.stderr or "") + (proc.stdout or "")
    info: dict[str, Any] = {}
    stream = re.search(r"Video:[^\n]*", text)
    if stream:
        line = stream.group(0)
        dims = re.search(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", line)
        fps = re.search(r"(\d+(?:\.\d+)?)\s*fps", line)
        if dims:
            info["width"] = int(dims.group(1))
            info["height"] = int(dims.group(2))
        if fps:
            fps_val = float(fps.group(1))
            info["fps"] = int(round(fps_val)) if abs(fps_val - round(fps_val)) < 0.01 else round(fps_val, 3)
    try:
        info["duration_seconds"] = round(duration(path), 3)
    except Exception:
        pass
    return info

def sane_video_fps(fps: Any) -> float:
    try:
        value = float(fps or 0)
    except Exception:
        value = 0.0
    if not math.isfinite(value) or value <= 0:
        return 24.0
    if value > 120:
        return 30.0
    return round(value, 3)

def estimated_video_frames(path: str|Path) -> int:
    info = video_probe(path)
    fps = float(info.get("fps") or 0)
    dur = float(info.get("duration_seconds") or 0)
    return int(round(fps * dur)) if fps > 0 and dur > 0 else 0

def strip_process_noise(text: str, limit: int = 2400) -> str:
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text or "")
    lines = []
    for line in text.replace("\r", "\n").splitlines():
        if "frame loading" in line and "|" in line:
            continue
        if line.strip():
            lines.append(line.rstrip())
    cleaned = "\n".join(lines).strip()
    return cleaned[-limit:] if len(cleaned) > limit else cleaned

def local_media_path(raw: str) -> Path:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Path is empty")
    if is_s3_uri(raw):
        return Path(cached_s3_object_path(raw, media_ext_from_string(raw)))
    if is_url(raw):
        return Path(cached_url(raw, media_ext_from_string(raw)))
    return Path(raw)


def validate_public_http_url(raw: str) -> str:
    value = str(raw or "").strip()
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a valid HTTP or HTTPS URL")
    host = parsed.hostname.rstrip(".")
    if host.lower() == "localhost":
        raise ValueError("Local and private-network URLs are not allowed")
    try:
        addresses = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme.lower() == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as e:
        raise ValueError(f"Could not resolve remote host: {host}") from e
    for item in addresses:
        address = str(item[4][0]).split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise ValueError(f"Remote host resolved to an invalid address: {address}")
        if not ip.is_global:
            raise ValueError("Local and private-network URLs are not allowed")
    return value


def safe_http_get(raw: str, *, max_redirects: int = 5, **kwargs: Any) -> requests.Response:
    current = validate_public_http_url(raw)
    for _ in range(max_redirects + 1):
        response = requests.get(current, allow_redirects=False, **kwargs)
        if response.status_code not in {301, 302, 303, 307, 308}:
            return response
        location = response.headers.get("location")
        response.close()
        if not location:
            raise requests.RequestException("Remote redirect did not include a destination")
        current = validate_public_http_url(urllib.parse.urljoin(current, location))
    raise requests.TooManyRedirects(f"Remote URL exceeded {max_redirects} redirects")


def cached_url(raw: str, suffix: str | None = None) -> str:
    suffix = suffix if suffix and suffix != ".bin" else media_ext_from_string(raw)
    if suffix == ".bin":
        suffix = ".dat"
    key = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    out = TEMP_DIR / f"url_{key}{suffix}"
    if out.exists() and out.stat().st_size > 0:
        return str(out)
    temp = out.with_suffix(out.suffix + ".download")
    try:
        with safe_http_get(raw, stream=True, timeout=(20, 3600)) as response:
            response.raise_for_status()
            copy_stream_limited(response.raw, temp, GENERIC_UPLOAD_MAX_BYTES, "Remote media")
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)
    return str(out)

def audio_probe(path: str|Path) -> dict[str, Any]:
    return {"duration_seconds": round(duration(path), 3)}

def image_probe(path: str|Path) -> dict[str, Any]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return {"width": im.width, "height": im.height, "mode": im.mode}
    except Exception:
        return {}

def media_probe_any(raw: str) -> dict[str, Any]:
    p = local_media_path(raw)
    if not p.exists():
        return {"path": str(p), "exists": False, "kind": "file"}
    kind = media_kind(p)
    info: dict[str, Any] = {"path": str(p), "exists": True, "kind": kind, "size_mb": round(p.stat().st_size / (1024 * 1024), 3)}
    try:
        if kind == "video":
            info.update(video_probe(p))
        elif kind == "audio":
            info.update(audio_probe(p))
        elif kind == "image":
            info.update(image_probe(p))
    except Exception as e:
        info["probe_error"] = str(e)
    return info

def seconds(v: Any) -> Optional[float]:
    if v in (None, ""): return None
    if isinstance(v, (int,float)): return float(v)
    s = str(v).strip()
    if not s: return None
    if s.lower() in {"full", "end", "auto"}: return None
    if ":" not in s: return float(s)
    parts = [float(x) for x in s.split(":")]
    if len(parts)==2: return parts[0]*60 + parts[1]
    if len(parts)==3: return parts[0]*3600 + parts[1]*60 + parts[2]
    raise ValueError(f"Invalid time: {v}")

def normalize_seed(value) -> tuple[int, bool]:
    """Comfy's WanVideoSampler rejects -1. Treat blank/negative as random valid seed."""
    try:
        if value is None or str(value).strip() == "":
            return 12345, False
        seed = int(float(value))
    except Exception:
        seed = 12345
    if seed < 0:
        return random.randint(0, 2**53 - 1), True
    return seed, False

def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n", ""}:
        return False
    return default

apply_cache_settings(settings())
if as_bool(settings().get("cache_cleanup_enabled"), True):
    cleanup_cache(settings())


def align16(value: Any, fallback: int = 16) -> int:
    try:
        n = int(round(float(value)))
    except Exception:
        n = fallback
    n = max(16, n)
    return max(16, n - (n % 16))

def process_video(path: str, start=None, end=None, cap=None, crop=None, target_size: tuple[int, int] | None = None, target_fps: Any = None) -> str:
    if not path or is_url(path): return path
    if is_s3_uri(path):
        path = cached_s3_object_path(path, media_ext_from_string(path))
    src = Path(path)
    if not src.exists(): raise ValueError(f"Video not found: {path}")
    if not has_video_stream(src): raise ValueError(f"Selected file has no video stream: {path}")
    st = seconds(start) or 0.0
    en = seconds(end)
    dur = duration(src)
    trim = bool(st > 0.001 or (en is not None and dur and abs(en-dur)>0.05))
    frame_cap = int(cap or 0)
    if trim: frame_cap = 0
    crop_active = bool(crop and float(crop.get("w",0)) > 1 and float(crop.get("h",0)) > 1)
    resize_w = int(float(crop.get("resize_w", 0) or 0)) if crop else 0
    resize_h = int(float(crop.get("resize_h", 0) or 0)) if crop else 0
    source_info = video_probe(src) if target_size or target_fps not in (None, "", 0, "0") else {}
    if target_size:
        target_w, target_h = target_size
        if int(source_info.get("width") or 0) != int(target_w) or int(source_info.get("height") or 0) != int(target_h):
            resize_w, resize_h = target_size
        else:
            resize_w, resize_h = 0, 0
    resize_active = resize_w > 1 or resize_h > 1
    fps_value = 0.0
    try:
        fps_value = float(target_fps or 0)
    except Exception:
        fps_value = 0.0
    source_fps = float(source_info.get("fps") or 0)
    fps_active = fps_value > 0 and (not source_fps or abs(source_fps - fps_value) > 0.05)
    if not trim and not frame_cap and not crop_active and not resize_active and not fps_active: return path
    out = TEMP_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{src.stem}_edited.mp4"
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error"]
    if st: cmd += ["-ss", str(st)]
    cmd += ["-i", str(src)]
    if en is not None and en > st: cmd += ["-t", str(en-st)]
    filters=[]
    if crop_active:
        x,y,w,h = [int(round(float(crop[k]))) for k in ("x","y","w","h")]
        w -= w % 2; h -= h % 2
        filters.append(f"crop={max(2,w)}:{max(2,h)}:{max(0,x)}:{max(0,y)}")
    if resize_active:
        rw = resize_w - (resize_w % 2) if resize_w > 1 else -2
        rh = resize_h - (resize_h % 2) if resize_h > 1 else -2
        filters.append(f"scale={rw}:{rh}")
    if fps_active:
        filters.append(f"fps={fps_value:g}")
    if filters: cmd += ["-vf", ",".join(filters)]
    if frame_cap: cmd += ["-frames:v", str(frame_cap)]
    cmd += ["-c:v","libx264","-preset","veryfast","-crf","18","-pix_fmt","yuv420p","-an",str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ValueError(f"Video preprocess failed for {src.name}: {detail}")
    if not out.exists() or out.stat().st_size <= 0 or not has_video_stream(out): raise ValueError(f"Preprocess failed: {out}")
    return str(out)

def ensure_video_size(path: str, width: int, height: int) -> str:
    if not path or is_url(path): return path
    info = video_probe(path)
    if int(info.get("width") or 0) == int(width) and int(info.get("height") or 0) == int(height):
        return path
    return process_video(path, target_size=(int(width), int(height)))

def resolve_wan_dimensions(data: dict[str, Any], video_path: str) -> tuple[int, int, dict[str, Any]]:
    info = video_probe(video_path) if video_path and not is_url(video_path) else {}
    width = int(data.get("width") or 0)
    height = int(data.get("height") or 0)
    if width <= 0: width = int(info.get("width") or 832)
    if height <= 0: height = int(info.get("height") or 480)
    resolved = {"source_width": info.get("width"), "source_height": info.get("height")}
    width16, height16 = align16(width, 832), align16(height, 480)
    resolved.update({"requested_width": data.get("width"), "requested_height": data.get("height"), "resolved_width": width16, "resolved_height": height16})
    return width16, height16, resolved

def local_probe_video_path(path: str) -> str:
    raw = (path or "").strip()
    if not raw or is_url(raw):
        return raw
    if is_s3_uri(raw):
        return cached_s3_object_path(raw, media_ext_from_string(raw))
    return raw

def run_media_cmd(cmd: list[str], label: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ValueError(f"{label} failed: {detail}")

def conform_video_duration(path: str, target_duration: float, label: str = "video") -> str:
    if not path or is_url(path) or target_duration <= 0:
        return path
    current = duration(path)
    if current <= 0 or abs(current - target_duration) <= 0.08:
        return path
    src = Path(path)
    out = TEMP_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{src.stem}_{label}_duration.mp4"
    vf = f"tpad=stop_mode=clone:stop_duration={max(0, target_duration-current):.4f},trim=duration={target_duration:.4f},setpts=PTS-STARTPTS"
    proc = subprocess.run([
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "14", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)
    ], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ValueError(f"Could not conform {label} duration to {target_duration:.2f}s: {detail}")
    if not out.exists() or out.stat().st_size <= 0:
        raise ValueError(f"Could not conform {label} duration")
    return str(out)

def process_wan_mask(path: str, target_video_path: str, start=None, end=None, cap=None, crop=None, target_size: tuple[int, int] | None = None, pretrimmed: bool = False) -> str:
    if not path or is_url(path): return path
    if is_s3_uri(path):
        path = cached_s3_object_path(path, media_ext_from_string(path))
    src = Path(path)
    if not src.exists(): raise ValueError(f"mask file not found: {path}")
    kind = source_kind(str(src))
    if kind == "video":
        if pretrimmed:
            processed = process_video(str(src), None, None, None, crop, target_size)
        else:
            processed = process_video(str(src), start, end, cap, crop, target_size)
        return conform_video_duration(processed, duration(target_video_path), "wan_mask")
    if kind != "image":
        raise ValueError(f"WanAnimate mask must be an image or video: {path}")
    from PIL import Image, ImageOps
    width, height = target_size or (0, 0)
    if not width or not height:
        info = video_probe(target_video_path)
        width, height = int(info.get("width") or 0), int(info.get("height") or 0)
    if not width or not height:
        raise ValueError("Could not resolve target size for image mask")
    img = ImageOps.exif_transpose(Image.open(src)).convert("L")
    img = img.resize((int(width), int(height)), Image.Resampling.NEAREST)
    out = TEMP_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{src.stem}_wan_mask.png"
    img.save(out)
    return str(out)

def is_sam_mask_video_path(path: str) -> bool:
    try:
        p = local_media_path(path)
        name = p.name.lower()
        try:
            in_sam_dir = p.resolve().is_relative_to(SAM_DIR.resolve())
        except Exception:
            in_sam_dir = False
        return in_sam_dir or ("sam3_" in name and "mask_video" in name)
    except Exception:
        return False

def wan_mask_is_pretrimmed(data: dict[str, Any], mask_path: str, target_video_path: str | None = None, mask_meta: dict[str, Any] | None = None, video_eff: dict[str, Any] | None = None) -> bool:
    if as_bool(data.get("mask_pretrimmed"), False):
        return True
    if not mask_path or not is_sam_mask_video_path(mask_path):
        return False
    has_trim = bool(data.get("video_start") or data.get("video_end"))
    if not has_trim:
        return False
    try:
        if source_kind(str(local_media_path(mask_path))) != "video":
            return False
    except Exception:
        return False
    mask_dur = float((mask_meta or {}).get("duration_seconds") or 0)
    if mask_dur <= 0:
        try:
            mask_dur = duration(local_media_path(mask_path))
        except Exception:
            mask_dur = 0
    target_dur = float((video_eff or {}).get("effective") or 0)
    if target_dur <= 0 and target_video_path:
        try:
            target_dur = duration(target_video_path)
        except Exception:
            target_dur = 0
    st = seconds(data.get("video_start")) or 0.0
    return bool(mask_dur > 0 and (st >= mask_dur - 0.05 or (target_dur > 0 and mask_dur <= target_dur + 1.0)))

def align_wan_mask_to_video(data: dict[str, Any], processed_video_path: str | None = None) -> dict[str, Any]:
    if processed_video_path:
        video_path = processed_video_path
        width, height, dim_debug = resolve_wan_dimensions(data, video_path)
        video_path = ensure_video_size(video_path, width, height)
    else:
        width, height, dim_debug = resolve_wan_dimensions(data, local_probe_video_path(data.get("video_path","")))
        video_path = process_video(data.get("video_path",""), data.get("video_start"), data.get("video_end"), data.get("video_frame_cap"), data.get("video_crop"), (width, height), data.get("fps") if data.get("source_workflow") == "samimate" else None)
    pretrimmed = wan_mask_is_pretrimmed(data, data.get("mask_path",""), video_path)
    mask_path = process_wan_mask(data.get("mask_path",""), video_path, data.get("video_start"), data.get("video_end"), data.get("video_frame_cap"), data.get("video_crop"), (width, height), pretrimmed)
    mask_info = media_probe_any(mask_path)
    video_info = media_probe_any(video_path)
    return {
        "path": mask_path,
        "url": safe_file_url(mask_path),
        "kind": media_kind(Path(mask_path)),
        "video_path": video_path,
        "mask_probe": mask_info,
        "video_probe": video_info,
        "resolved_width": width,
        "resolved_height": height,
        "dimension_debug": dim_debug,
        "trim": {
            "start": data.get("video_start") or "0",
            "end": data.get("video_end") or "full",
            "frame_cap": "ignored_by_trim" if (data.get("video_start") or data.get("video_end")) else (data.get("video_frame_cap") or "0"),
            "crop": data.get("video_crop") or None,
            "mask_pretrimmed": pretrimmed,
        },
    }

def process_wan_reference_image(path: str) -> str:
    if not path or is_url(path): return path
    if is_s3_uri(path):
        path = cached_s3_object_path(path, media_ext_from_string(path))
    src = Path(path)
    if not src.exists(): raise ValueError(f"image file not found: {path}")
    if source_kind(str(src)) != "image": raise ValueError(f"WanAnimate reference image must be an image: {path}")
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    out = TEMP_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{src.stem}_wan_ref.png"
    img.save(out)
    return str(out)

def process_audio(path: str, start=None, end=None) -> str:
    if not path or is_url(path): return path
    if is_s3_uri(path):
        path = cached_s3_object_path(path, media_ext_from_string(path))
    src = Path(path)
    if not src.exists(): raise ValueError(f"Audio not found: {path}")
    st = seconds(start) or 0.0
    en = seconds(end)
    if en is not None and en <= st: raise ValueError("Audio trim end must be after start")
    # Always normalize before sending to RunPod. Endpoints are much more reliable
    # with plain PCM WAV than mixed MP3/M4A/variable-rate browser uploads.
    out = TEMP_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{src.stem}_normalized.wav"
    cmd = [ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error"]
    if st > 0.001: cmd += ["-ss", str(st)]
    cmd += ["-i", str(src)]
    if en is not None: cmd += ["-to", str(en - st if st > 0.001 else en)]
    cmd += ["-vn", "-map", "0:a:0", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2", str(out)]
    run_media_cmd(cmd, f"Audio preprocess for {src.name}")
    if not out.exists() or out.stat().st_size <= 0: raise ValueError(f"Audio preprocess failed: {out}")
    return str(out)

def tts_output_dir(s: dict[str, Any]) -> Path:
    out = Path(s.get("output_dir") or DEFAULT_OUTPUT_DIR) / "tts"
    out.mkdir(parents=True, exist_ok=True)
    return out

def piper_package_available() -> bool:
    return importlib.util.find_spec("piper") is not None

def piper_default_paths() -> dict[str, str]:
    return {
        "default_model_path": str(DEFAULT_PIPER_MODEL),
        "default_config_path": str(DEFAULT_PIPER_CONFIG),
        "model_url": PIPER_MODEL_URL,
        "config_url": PIPER_CONFIG_URL,
    }

def piper_status(s: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    cfg = (s or settings()).get("tts", {})
    model = Path(cfg.get("piper_model") or DEFAULT_PIPER_MODEL)
    config = Path(cfg.get("piper_config") or (str(model) + ".json"))
    return {
        "package_available": piper_package_available(),
        "configured_model": str(cfg.get("piper_model") or ""),
        "configured_config": str(cfg.get("piper_config") or ""),
        "model_path": str(model),
        "config_path": str(config),
        "model_exists": model.exists(),
        "config_exists": config.exists(),
        **piper_default_paths(),
    }

def download_file(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = out.with_suffix(out.suffix + ".download")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        tmp.replace(out)

def setup_piper_voice(force: bool = False) -> dict[str, Any]:
    if not piper_package_available():
        raise RuntimeError("The piper Python package is not installed in this venv. Run: .\\.venv\\Scripts\\python.exe -m pip install piper-tts")
    if force or not DEFAULT_PIPER_MODEL.exists():
        download_file(PIPER_MODEL_URL, DEFAULT_PIPER_MODEL)
    if force or not DEFAULT_PIPER_CONFIG.exists():
        download_file(PIPER_CONFIG_URL, DEFAULT_PIPER_CONFIG)
    return piper_status({"tts": {"piper_model": str(DEFAULT_PIPER_MODEL), "piper_config": str(DEFAULT_PIPER_CONFIG)}})

def sapi_voices() -> list[dict[str, str]]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.GetInstalledVoices() | ForEach-Object { $i=$_.VoiceInfo; [pscustomobject]@{name=$i.Name; culture=$i.Culture.Name; gender=[string]$i.Gender; age=[string]$i.Age} } | ConvertTo-Json -Compress",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        return []
    raw = (proc.stdout or "").strip()
    if not raw:
        return []
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    return [{"name": str(v.get("name", "")), "culture": str(v.get("culture", "")), "gender": str(v.get("gender", "")), "age": str(v.get("age", ""))} for v in data if v.get("name")]

def run_windows_sapi_tts(text: str, out: Path, voice: str = "", rate: int = 0, volume: int = 100) -> None:
    if not TTS_SAPI_SCRIPT.exists():
        raise RuntimeError(f"SAPI helper not found: {TTS_SAPI_SCRIPT}")
    text_path = TEMP_DIR / f"tts_{int(time.time())}_{uuid.uuid4().hex[:8]}.txt"
    text_path.write_text(text, encoding="utf-8")
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(TTS_SAPI_SCRIPT),
        "-InputTextPath",
        str(text_path),
        "-OutputPath",
        str(out),
        "-Voice",
        voice or "",
        "-Rate",
        str(max(-10, min(10, int(rate or 0)))),
        "-Volume",
        str(max(0, min(100, int(volume or 100)))),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    try:
        text_path.unlink(missing_ok=True)
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "Windows SAPI TTS failed").strip())

def run_piper_tts(text: str, out: Path, cfg: dict[str, Any]) -> None:
    model = cfg.get("piper_model") or (str(DEFAULT_PIPER_MODEL) if DEFAULT_PIPER_MODEL.exists() else "")
    if not model:
        raise ValueError("Piper is not set up yet. Use Settings > Set up Piper voice, or choose a local .onnx voice model.")
    if not Path(model).exists():
        raise ValueError(f"Piper voice model was not found: {model}")
    exe = (cfg.get("piper_exe") or "").strip()
    if exe:
        cmd = [str(exe), "--model", str(model), "--output_file", str(out)]
    else:
        cmd = [sys.executable, "-m", "piper", "--model", str(model), "--output_file", str(out)]
    config = cfg.get("piper_config") or (str(DEFAULT_PIPER_CONFIG) if DEFAULT_PIPER_CONFIG.exists() and str(model) == str(DEFAULT_PIPER_MODEL) else "")
    if config and Path(config).exists():
        cmd += ["--config", str(config)]
    proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "Piper TTS failed").strip())

def generate_tts_audio(data: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    text = str(data.get("text") or "").strip()
    if not text:
        raise ValueError("Text is required")
    if len(text) > 20000:
        raise ValueError("Text is too long; keep it under 20,000 characters")
    tts_s = merge(s.get("tts", {}), data)
    engine = (tts_s.get("engine") or "windows_sapi").strip()
    out = next_output_path(tts_output_dir(s), "tts", ".wav")
    if engine == "piper":
        run_piper_tts(text, out, tts_s)
    elif engine == "windows_sapi":
        run_windows_sapi_tts(text, out, tts_s.get("voice") or "", int(tts_s.get("rate") or 0), int(tts_s.get("volume") or 100))
    else:
        raise ValueError(f"Unsupported TTS engine: {engine}")
    if not out.exists() or out.stat().st_size <= 0:
        raise RuntimeError(f"TTS did not create audio: {out}")
    return {"path": str(out), "url": safe_file_url(out), "kind": "audio", "duration_seconds": duration(out), "engine": engine}

def ext_from_bytes(data: bytes) -> str:
    if len(data)>12 and data[4:8]==b"ftyp": return ".mp4"
    if data.startswith(b"\x89PNG\r\n\x1a\n"): return ".png"
    if data.startswith(b"\xff\xd8\xff"): return ".jpg"
    if data.startswith(b"RIFF") and data[8:12]==b"WAVE": return ".wav"
    return ".bin"

def ext_from_url_or_bytes(url: str, data: bytes) -> str:
    ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if ext in MEDIA_EXT: return ext
    return ext_from_bytes(data)

class S3Helper:
    def __init__(self, s: dict[str, Any]):
        self.cfg = s
        self.bucket=s["s3_bucket"]; self.prefix=s.get("s3_prefix","input/runpod-media-console").strip("/"); self.root=s.get("runpod_volume_root","/runpod-volume").rstrip("/")
        self.client=boto3.client("s3", endpoint_url=s["s3_endpoint_url"], aws_access_key_id=s["s3_access_key_id"], aws_secret_access_key=s["s3_secret_access_key"], region_name=s["s3_region"], config=Config(signature_version="s3v4", connect_timeout=8, read_timeout=120, retries={"total_max_attempts": 2, "mode": "standard"}))
    def refresh_client(self) -> None:
        self.__init__(self.cfg)
    def call(self, method_name: str, *args, **kwargs):
        return getattr(self.client, method_name)(*args, **kwargs)
    def upload(self, path: str|Path, kind: str) -> dict[str,str]:
        p=Path(path); key=f"{self.prefix}/{kind}/{short_media_name(kind, p.suffix or '.bin')}"
        self.call("upload_file", str(p), self.bucket, key)
        return {"s3":f"s3://{self.bucket}/{key}","volume":f"{self.root}/{key}","key":key}
    def upload_to_key(self, path: str|Path, key: str) -> dict[str,str]:
        p=Path(path); clean_key=key.strip("/\\").replace("\\", "/")
        self.call("upload_file", str(p), self.bucket, clean_key)
        return {"s3":f"s3://{self.bucket}/{clean_key}","volume":f"{self.root}/{clean_key}","key":clean_key}
    def download_object(self, bucket: str, key: str, out_path: str|Path) -> str:
        """Download even when an S3-compatible gateway rejects HeadObject.

        boto3's high-level download_file issues HeadObject before GetObject.
        RunPod network-volume S3 can permit the object read while rejecting
        that metadata request, so retry with a direct streaming GetObject.
        """
        out=Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.call("download_file", bucket, key, str(out))
        except Exception as download_error:
            out.unlink(missing_ok=True)
            try:
                response = self.call("get_object", Bucket=bucket, Key=key)
                body = response["Body"]
                with out.open("wb") as handle:
                    shutil.copyfileobj(body, handle)
                try:
                    body.close()
                except Exception:
                    pass
            except Exception:
                out.unlink(missing_ok=True)
                raise download_error
        return str(out)
    def download(self, uri: str, out_dir: str|Path, out_path: str|Path|None = None) -> str:
        rest=uri[5:]; bucket,key=rest.split("/",1); out=Path(out_path) if out_path else Path(out_dir)/Path(key).name
        return self.download_object(bucket, key, out)
    def download_key(self, key: str, out_path: str|Path) -> str:
        return self.download_object(self.bucket, key.lstrip("/"), out_path)
    def presign_key(self, key: str, expires: int = 3600) -> str:
        return self.call("generate_presigned_url", "get_object", Params={"Bucket": self.bucket, "Key": key.lstrip("/")}, ExpiresIn=expires)

def h3_storage_settings(s: dict[str, Any]) -> dict[str, Any]:
    """Return an S3Helper-compatible view for the H3 endpoint's separate volume.

    Blank H3 credentials inherit the account-wide RunPod S3 credentials, while
    the H3 bucket/prefix/root remain independent from the other endpoints.
    """
    out = dict(s)
    h3 = s.get("h3_storage") or {}
    for key in ("s3_endpoint_url", "s3_access_key_id", "s3_secret_access_key", "s3_region"):
        if str(h3.get(key) or "").strip():
            out[key] = h3[key]
    for key, fallback in (
        ("s3_bucket", ""),
        ("s3_prefix", "minimax-h3"),
        ("runpod_volume_root", "/runpod-volume"),
    ):
        out[key] = str(h3.get(key) or fallback).strip()
    return out

def storage_settings_for_endpoint(s: dict[str, Any], endpoint_id: str) -> dict[str, Any]:
    return h3_storage_settings(s) if endpoint_id and endpoint_id == s.get("h3_endpoint_id") else s

def named_storage_settings(s: dict[str, Any], storage: str = "main") -> tuple[str, dict[str, Any]]:
    """Resolve one of the two configured S3 profiles without accepting arbitrary buckets."""
    name = str(storage or "main").strip().lower()
    if name in {"main", "global", "default"}:
        return "main", s
    if name in {"h3", "minimax-h3"}:
        return "h3", h3_storage_settings(s)
    raise ValueError("storage must be main or h3")

def require_named_s3(storage: str = "main", s: Optional[dict[str, Any]] = None) -> tuple[str, S3Helper]:
    name, storage_s = named_storage_settings(s or settings(), storage)
    helper = configured_s3_helper(storage_s)
    if not helper:
        raise ValueError(f"Configure the {name.upper() if name == 'h3' else 'main'} S3 bucket and credentials in Settings first")
    return name, helper

def s3_object_metadata(helper: S3Helper, key: str) -> dict[str, Any]:
    """Retry one transient RunPod S3 metadata miss without creating a long UI stall."""
    try:
        return helper.call("head_object", Bucket=helper.bucket, Key=key)
    except Exception:
        helper.refresh_client()
        time.sleep(0.2)
        return helper.call("head_object", Bucket=helper.bucket, Key=key)

def configured_s3_helper(s: dict[str, Any]) -> Optional[S3Helper]:
    if not all(str(s.get(k) or "").strip() for k in ("s3_bucket", "s3_access_key_id", "s3_secret_access_key")):
        return None
    return S3Helper(s)

H3_LORA_PREFIX = "models/loras/H3"
H3_LORA_MAX_BYTES = 16 * 1024 * 1024 * 1024
H3_MODEL_PREFIXES = {
    "diffusion_models": "models/diffusion_models/",
    "text_encoders": "models/text_encoders/",
    "vaes": "models/vae/",
}
H3_MODEL_DESTINATIONS = {
    "checkpoints": "models/checkpoints",
    "diffusion_models": "models/diffusion_models",
    "text_encoders": "models/text_encoders",
    "clip": "models/clip",
    "clip_vision": "models/clip_vision",
    "vae": "models/vae",
    "loras": "models/loras/H3",
    "controlnet": "models/controlnet",
    "embeddings": "models/embeddings",
    "upscale_models": "models/upscale_models",
}
H3_MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"}
H3_MODEL_MAX_BYTES = 64 * 1024 * 1024 * 1024
# Keep well below RunPod's practical multipart-part ceiling. A 64 GiB model
# produces at most 512 parts at this size, instead of thousands of 16 MiB parts.
H3_MODEL_PART_BYTES = 128 * 1024 * 1024
H3_MODEL_PART_UPLOAD_ATTEMPTS = 5
MODEL_DOWNLOAD_JOBS: dict[str, dict[str, Any]] = {}
MODEL_DOWNLOAD_LOCK = threading.Lock()
RUNPOD_PODS_API = "https://rest.runpod.io/v1/pods"
H3_COMFYUI_TEMPLATE_ID = "2lv7ev3wfp"
H3_COMFYUI_CONFIG_KEY = "runpod-slim/ComfyUI/extra_model_paths.yaml"
H3_COMFYUI_CONFIG_MARKER = "runpod_media_console_h3:"

def h3_lora_name(value: str) -> str:
    name = clean_name(str(value or "").strip())
    if not name.lower().endswith(".safetensors"):
        raise ValueError("H3 LoRAs must use the .safetensors extension")
    return name

def h3_lora_key(name: str) -> str:
    return f"{H3_LORA_PREFIX}/{h3_lora_name(name)}"

def h3_worker_lora_name(name: str) -> str:
    """Return the path ComfyUI exposes from models/loras/H3/."""
    return f"H3/{h3_lora_name(name)}"

def h3_model_name(value: Any, label: str, required: bool = True) -> str:
    name = str(value or "").strip().replace("\\", "/")
    if not name:
        if required:
            raise ValueError(f"{label} is required")
        return ""
    if name.startswith("/") or any(part == ".." for part in name.split("/")):
        raise ValueError(f"{label} must be a model-relative filename")
    return name

def h3_model_role_options(names: list[str], role: str) -> list[str]:
    """Expose every diffusion model, with task-specific filenames listed first."""
    unique = sorted(set(names), key=str.casefold)
    marker = str(role or "").casefold()
    return sorted(unique, key=lambda name: (marker not in name.casefold(), name.casefold()))


def h3_output_frame_count(duration: float) -> int:
    """Match the H3 workflow's 24 fps, 17-frame latent alignment."""
    raw = max(5, round(float(duration) * 24))
    return raw + (5 - (raw % 17)) % 17


def h3_reference_video_settings(value: Any, count: int, duration: float) -> list[dict[str, Any]]:
    if value in (None, ""):
        value = []
    if not isinstance(value, list):
        raise ValueError("H3 R2VA reference video settings must be a list")
    frame_cap = h3_output_frame_count(duration)
    settings: list[dict[str, Any]] = []
    for index in range(count):
        item = value[index] if index < len(value) else {}
        if not isinstance(item, dict):
            raise ValueError(f"H3 R2VA video {index + 1} settings must be an object")
        force_rate = float(item.get("force_rate", 24))
        start_frame = int(item.get("start_frame", item.get("skip_first_frames", 0)))
        select_every_nth = int(item.get("select_every_nth", 1))
        if not 0 <= force_rate <= 120:
            raise ValueError(f"H3 R2VA video {index + 1} force framerate must be between 0 and 120")
        if start_frame < 0:
            raise ValueError(f"H3 R2VA video {index + 1} start frame cannot be negative")
        if not 1 <= select_every_nth <= 1000:
            raise ValueError(f"H3 R2VA video {index + 1} frame stride must be between 1 and 1000")
        settings.append({
            "force_rate": force_rate,
            "skip_first_frames": start_frame,
            "select_every_nth": select_every_nth,
            "frame_load_cap": frame_cap,
        })
    return settings

def require_h3_s3(s: Optional[dict[str, Any]] = None) -> S3Helper:
    helper = configured_s3_helper(h3_storage_settings(s or settings()))
    if not helper:
        raise ValueError("Configure the H3 network-volume S3 bucket and credentials in Settings first")
    return helper


def runpod_pod_request(method: str, path: str = "", *, data: Optional[dict[str, Any]] = None, s: Optional[dict[str, Any]] = None) -> Any:
    current = s or settings()
    api_key = str(current.get("runpod_api_key") or "").strip()
    if not api_key:
        raise ValueError("Configure the RunPod API key in Settings first")
    url = RUNPOD_PODS_API + ("/" + path.strip("/") if path else "")
    response = requests.request(
        method.upper(),
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=data,
        timeout=45,
    )
    if response.status_code >= 400:
        try:
            body = response.json()
            detail = body.get("detail") or body.get("message") or body.get("error") or str(body)
        except Exception:
            detail = response.text[:500]
        raise RuntimeError(f"RunPod Pod API returned HTTP {response.status_code}: {detail}")
    if not response.content:
        return {}
    return response.json()


def h3_comfyui_pod_public(pod: dict[str, Any]) -> dict[str, Any]:
    pod_id = str(pod.get("id") or "")
    status_value = str(pod.get("desiredStatus") or pod.get("status") or "UNKNOWN").upper()
    network_volume = pod.get("networkVolume") if isinstance(pod.get("networkVolume"), dict) else {}
    cost = pod.get("adjustedCostPerHr")
    if cost in (None, ""):
        cost = pod.get("costPerHr")
    return {
        "id": pod_id,
        "name": str(pod.get("name") or "ComfyUI"),
        "status": status_value,
        "image": str(pod.get("imageName") or ""),
        "network_volume_id": str(pod.get("networkVolumeId") or network_volume.get("id") or ""),
        "volume_mount_path": str(pod.get("volumeMountPath") or "/workspace"),
        "gpu": str(((pod.get("gpu") or {}).get("displayName") if isinstance(pod.get("gpu"), dict) else "") or ""),
        "cost_per_hour": float(cost) if cost not in (None, "") else None,
        "url": f"https://{pod_id}-8188.proxy.runpod.net" if pod_id else "",
        "ready": status_value == "RUNNING" and bool(pod_id),
    }


def list_h3_comfyui_pods(s: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    current = s or settings()
    volume_id = str((current.get("h3_storage") or {}).get("network_volume_id") or "").strip()
    response = runpod_pod_request("GET", s=current)
    pods = response if isinstance(response, list) else response.get("data") or response.get("pods") or []
    matches: list[dict[str, Any]] = []
    for pod in pods:
        if not isinstance(pod, dict):
            continue
        public = h3_comfyui_pod_public(pod)
        ports = [str(port).lower() for port in (pod.get("ports") or [])]
        looks_like_comfy = "comfyui" in (public["image"] + " " + public["name"]).lower() or any(port.startswith("8188/") for port in ports)
        if looks_like_comfy and (not volume_id or public["network_volume_id"] == volume_id):
            matches.append(public)
    preferred = str((current.get("h3_storage") or {}).get("comfyui_pod_id") or "").strip()
    return sorted(matches, key=lambda pod: (pod["id"] != preferred, pod["status"] != "RUNNING", pod["name"].casefold()))


def prepare_h3_comfyui_volume(s: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Merge the H3 model search paths into the official ComfyUI volume config."""
    current = s or settings()
    helper = require_h3_s3(current)
    existing = ""
    try:
        obj = helper.client.get_object(Bucket=helper.bucket, Key=H3_COMFYUI_CONFIG_KEY)
        size = int(obj.get("ContentLength") or 0)
        if size > 1024 * 1024:
            raise ValueError("Existing ComfyUI extra_model_paths.yaml is unexpectedly larger than 1 MB")
        existing = obj["Body"].read().decode("utf-8")
        obj["Body"].close()
    except Exception as exc:
        response = getattr(exc, "response", {}) or {}
        code = str((response.get("Error") or {}).get("Code") or "")
        status_code = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        if code not in {"404", "NoSuchKey", "NotFound"} and status_code != 404:
            raise
    block = """\n# Managed by RunPod Media Console. Existing entries above are preserved.\nrunpod_media_console_h3:\n  base_path: /workspace\n  checkpoints: models/checkpoints\n  diffusion_models: models/diffusion_models\n  text_encoders: models/text_encoders\n  clip: models/clip\n  clip_vision: models/clip_vision\n  vae: models/vae\n  loras: |\n    models/loras\n    models/loras/H3\n  controlnet: models/controlnet\n  embeddings: models/embeddings\n  upscale_models: models/upscale_models\n"""
    changed = H3_COMFYUI_CONFIG_MARKER not in existing
    content = (existing.rstrip() + "\n" + block.lstrip()) if existing else block.lstrip()
    if changed:
        helper.client.put_object(
            Bucket=helper.bucket,
            Key=H3_COMFYUI_CONFIG_KEY,
            Body=content.encode("utf-8"),
            ContentType="application/x-yaml",
        )
    return {
        "prepared": True,
        "changed": changed,
        "key": H3_COMFYUI_CONFIG_KEY,
        "volume_path": f"/workspace/{H3_COMFYUI_CONFIG_KEY}",
        "models_path": "/workspace/models",
    }

def h3_lora_exists(s3: S3Helper, key: str) -> bool:
    try:
        # A missing object is expected here, so do not use S3Helper.call's
        # retry loop and turn one 404 into several seconds of apparent delay.
        s3.client.head_object(Bucket=s3.bucket, Key=key)
        return True
    except Exception as e:
        response = getattr(e, "response", {}) or {}
        code = str((response.get("Error") or {}).get("Code") or "")
        status_code = (response.get("ResponseMetadata") or {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status_code == 404:
            return False
        raise

def copy_stream_limited(source: Any, destination: Path, max_bytes: int, label: str = "Upload") -> int:
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"{label} exceeds the {max_bytes / (1024 ** 3):g} GB limit")
            out.write(chunk)
    return size

def h3_lora_result(s3: S3Helper, name: str, size: int = 0) -> dict[str, Any]:
    key = h3_lora_key(name)
    return {
        "name": h3_lora_name(name),
        "key": key,
        "size": int(size or 0),
        "size_mb": round(int(size or 0) / (1024 * 1024), 2),
        "s3_uri": f"s3://{s3.bucket}/{key}",
        "volume_path": f"{s3.root}/{key}",
    }

def list_h3_loras(s3: S3Helper) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": s3.bucket, "Prefix": H3_LORA_PREFIX + "/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.call("list_objects_v2", **kwargs)
        for obj in page.get("Contents") or []:
            key = str(obj.get("Key") or "")
            name = Path(key).name
            if not name.lower().endswith(".safetensors"):
                continue
            item = h3_lora_result(s3, name, int(obj.get("Size") or 0))
            modified = obj.get("LastModified")
            item["modified"] = modified.isoformat() if hasattr(modified, "isoformat") else str(modified or "")
            items.append(item)
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
        if not token:
            break
    return sorted(items, key=lambda item: item["name"].lower())

def list_h3_model_names(s3: S3Helper, prefix: str) -> list[str]:
    names: list[str] = []
    token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": s3.bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        # Avoid retrying a slow or unavailable listing multiple times. The UI can
        # retry explicitly without freezing the rest of the console.
        page = s3.client.list_objects_v2(**kwargs)
        for obj in page.get("Contents") or []:
            key = str(obj.get("Key") or "")
            if key.lower().endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin")):
                names.append(key[len(prefix):] if key.startswith(prefix) else Path(key).name)
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
        if not token:
            break
    return sorted(set(names), key=str.lower)

def h3_model_destination(category: str) -> str:
    clean = str(category or "").strip().lower()
    if clean not in H3_MODEL_DESTINATIONS:
        raise ValueError("Unsupported model category")
    return H3_MODEL_DESTINATIONS[clean]

def h3_remote_model_name(value: str) -> str:
    name = clean_name(str(value or "").strip())
    if Path(name).suffix.lower() not in H3_MODEL_EXTENSIONS:
        raise ValueError("Model filename must end in .safetensors, .ckpt, .pt, .pth, .bin, or .gguf")
    return name

def scoped_download_headers(url: str, download_s: dict[str, Any]) -> dict[str, str]:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        token = str(download_s.get("hf_token") or "").strip()
        return {"Authorization": f"Bearer {token}"} if token else {}
    if host in {"civitai.com", "www.civitai.com"}:
        token = str(download_s.get("civitai_token") or "").strip()
        return {"Authorization": f"Bearer {token}"} if token else {}
    return {}

def scoped_model_get(url: str, download_s: dict[str, Any], **kwargs: Any) -> requests.Response:
    current = validate_public_http_url(url)
    base_headers = dict(kwargs.pop("headers", {}) or {})
    for _ in range(6):
        # Preserve Range and other caller headers across provider/CDN redirects.
        headers = dict(base_headers)
        headers.update(scoped_download_headers(current, download_s))
        response = requests.get(current, headers=headers, allow_redirects=False, **kwargs)
        if response.status_code not in {301, 302, 303, 307, 308}:
            return response
        location = response.headers.get("location")
        response.close()
        if not location:
            raise requests.RequestException("Remote redirect did not include a destination")
        current = validate_public_http_url(urllib.parse.urljoin(current, location))
    raise requests.TooManyRedirects("Remote model URL exceeded 5 redirects")

def resolve_model_download(raw_url: str, download_s: dict[str, Any]) -> tuple[str, str]:
    url = validate_public_http_url(raw_url)
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        if "/blob/" in path:
            path = path.replace("/blob/", "/resolve/", 1)
            url = urllib.parse.urlunparse(parsed._replace(path=path))
        if "/resolve/" not in path:
            raise ValueError("Use a Hugging Face file URL (open the file, then copy its download URL), not only a repository page")
        return url, urllib.parse.unquote(Path(path).name)
    if host in {"civitai.com", "www.civitai.com"} and re.match(r"^/models/\d+", path):
        model_id = re.match(r"^/models/(\d+)", path).group(1)
        query = urllib.parse.parse_qs(parsed.query)
        version_id = str((query.get("modelVersionId") or [""])[0]).strip()
        api_url = f"https://civitai.com/api/v1/model-versions/{version_id}" if version_id else f"https://civitai.com/api/v1/models/{model_id}"
        with scoped_model_get(api_url, download_s, timeout=(20, 120)) as response:
            response.raise_for_status()
            data = response.json()
        if not version_id:
            versions = data.get("modelVersions") or []
            if not versions:
                raise ValueError("Civitai model page did not expose a downloadable version")
            data = versions[0]
        files = data.get("files") or []
        if not files:
            raise ValueError("Civitai model version did not expose a downloadable file")
        item = next((entry for entry in files if entry.get("primary")), files[0])
        return validate_public_http_url(str(item.get("downloadUrl") or "")), str(item.get("name") or "")
    return url, urllib.parse.unquote(Path(path).name)

def model_download_job_update(job_id: str, **values: Any) -> None:
    with MODEL_DOWNLOAD_LOCK:
        job = MODEL_DOWNLOAD_JOBS.get(job_id)
        if job:
            job.update(values, updated_at=time.time())

def upload_model_part(s3: S3Helper, key: str, upload_id: str, part_number: int, body: bytes) -> dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(1, H3_MODEL_PART_UPLOAD_ATTEMPTS + 1):
        try:
            uploaded = s3.client.upload_part(
                Bucket=s3.bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=body,
            )
            return {"ETag": uploaded["ETag"], "PartNumber": part_number}
        except Exception as exc:
            last_error = exc
            if attempt >= H3_MODEL_PART_UPLOAD_ATTEMPTS:
                break
            time.sleep(min(8, 2 ** (attempt - 1)))
    raise RuntimeError(f"S3 part {part_number} failed after {H3_MODEL_PART_UPLOAD_ATTEMPTS} attempts: {last_error}")

def list_model_upload_parts(s3: S3Helper, key: str, upload_id: str) -> Optional[dict[int, dict[str, Any]]]:
    """Return persisted multipart parts, or None when this S3 gateway cannot list them."""
    found: dict[int, dict[str, Any]] = {}
    marker = 0
    try:
        while True:
            response = s3.client.list_parts(
                Bucket=s3.bucket,
                Key=key,
                UploadId=upload_id,
                PartNumberMarker=marker,
            )
            for item in response.get("Parts") or []:
                number = int(item["PartNumber"])
                found[number] = {"ETag": item["ETag"], "PartNumber": number}
            if not response.get("IsTruncated"):
                return found
            marker = int(response.get("NextPartNumberMarker") or max(found, default=marker))
    except Exception:
        return None

def download_model_range(url: str, download_s: dict[str, Any], start: int, length: int) -> bytes:
    end = start + length - 1
    with scoped_model_get(
        url,
        download_s,
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(30, 3600),
    ) as response:
        if response.status_code != 206:
            raise RuntimeError(
                f"Source did not honor byte-range recovery for bytes {start}-{end} "
                f"(HTTP {response.status_code})"
            )
        data = bytearray()
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                data.extend(chunk)
                if len(data) > length:
                    raise RuntimeError(f"Source returned too much data for recovered part at byte {start}")
        if len(data) != length:
            raise RuntimeError(f"Recovered part at byte {start} was {len(data)} bytes; expected {length}")
        return bytes(data)

def verify_and_repair_model_parts(
    s3: S3Helper,
    key: str,
    upload_id: str,
    resolved_url: str,
    download_s: dict[str, Any],
    part_ranges: dict[int, tuple[int, int]],
    original_parts: list[dict[str, Any]],
    job_id: str,
) -> list[dict[str, Any]]:
    expected = set(part_ranges)
    persisted: Optional[dict[int, dict[str, Any]]] = None
    # Give the S3-compatible gateway a short window to expose just-written parts.
    for check in range(3):
        persisted = list_model_upload_parts(s3, key, upload_id)
        if persisted is None or expected.issubset(persisted):
            break
        if check < 2:
            time.sleep(2 ** check)
    if persisted is None:
        return original_parts
    missing = sorted(expected - set(persisted))
    if missing:
        model_download_job_update(
            job_id,
            status="VERIFYING",
            phase=f"Repairing {len(missing)} missing S3 part(s)",
            missing_parts=len(missing),
        )
        for index, number in enumerate(missing, start=1):
            start, length = part_ranges[number]
            body = download_model_range(resolved_url, download_s, start, length)
            persisted[number] = upload_model_part(s3, key, upload_id, number, body)
            model_download_job_update(
                job_id,
                repaired_parts=index,
                missing_parts=len(missing) - index,
            )
        persisted = list_model_upload_parts(s3, key, upload_id)
        if persisted is None:
            raise RuntimeError("S3 part verification became unavailable after recovery")
        still_missing = sorted(expected - set(persisted))
        if still_missing:
            raise RuntimeError(f"S3 gateway still reports {len(still_missing)} missing multipart part(s) after recovery")
    return [persisted[number] for number in sorted(expected)]

def run_model_download_job(job_id: str, request_data: dict[str, Any]) -> None:
    upload_id = ""
    s3: Optional[S3Helper] = None
    key = ""
    try:
        saved = settings()
        download_s = saved.get("model_download") or {}
        s3 = require_h3_s3(saved)
        category = str(request_data.get("category") or "")
        destination = h3_model_destination(category)
        model_download_job_update(job_id, status="RESOLVING", phase="Resolving provider URL")
        resolved_url, hinted_name = resolve_model_download(str(request_data.get("url") or ""), download_s)
        with scoped_model_get(
            resolved_url,
            download_s,
            headers={"Accept-Encoding": "identity"},
            stream=True,
            timeout=(30, 3600),
        ) as response:
            response.raise_for_status()
            disposition = response.headers.get("content-disposition", "")
            match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", disposition, flags=re.I)
            header_name = urllib.parse.unquote(match.group(1).strip()) if match else ""
            name = h3_remote_model_name(str(request_data.get("name") or "") or header_name or hinted_name)
            key = f"{destination}/{name}"
            if not as_bool(request_data.get("overwrite"), False) and h3_lora_exists(s3, key):
                raise FileExistsError(f"{name} already exists in {destination}")
            total = int(response.headers.get("content-length") or 0)
            if str(response.headers.get("content-encoding") or "").lower() not in {"", "identity"}:
                total = 0
            if total > H3_MODEL_MAX_BYTES:
                raise ValueError("Remote model exceeds the 64 GB import limit")
            created = s3.client.create_multipart_upload(Bucket=s3.bucket, Key=key)
            upload_id = str(created["UploadId"])
            parts: list[dict[str, Any]] = []
            downloaded = 0
            model_download_job_update(job_id, status="DOWNLOADING", phase="Streaming into the H3 volume", filename=name, key=key, total_bytes=total)
            part_ranges: dict[int, tuple[int, int]] = {}
            next_offset = 0
            part_number = 0
            for chunk in response.iter_content(chunk_size=H3_MODEL_PART_BYTES):
                if not chunk:
                    continue
                part_number += 1
                downloaded += len(chunk)
                if downloaded > H3_MODEL_MAX_BYTES:
                    raise ValueError("Remote model exceeds the 64 GB import limit")
                parts.append(upload_model_part(s3, key, upload_id, part_number, chunk))
                part_ranges[part_number] = (next_offset, len(chunk))
                next_offset += len(chunk)
                percent = min(99.9, round(downloaded * 100 / total, 1)) if total else None
                model_download_job_update(job_id, downloaded_bytes=downloaded, percent=percent, uploaded_parts=part_number)
            if not parts:
                raise ValueError("Remote model download was empty")
            model_download_job_update(job_id, status="VERIFYING", phase="Verifying S3 multipart inventory")
            parts = verify_and_repair_model_parts(
                s3, key, upload_id, resolved_url, download_s, part_ranges, parts, job_id
            )
            for completion_attempt in range(1, 4):
                try:
                    s3.client.complete_multipart_upload(
                        Bucket=s3.bucket,
                        Key=key,
                        UploadId=upload_id,
                        MultipartUpload={"Parts": parts},
                    )
                    break
                except Exception as exc:
                    if "InvalidPart" not in str(exc) or completion_attempt >= 3:
                        raise
                    model_download_job_update(
                        job_id,
                        status="VERIFYING",
                        phase=f"S3 completion retry {completion_attempt}/2",
                    )
                    time.sleep(2 ** completion_attempt)
                    parts = verify_and_repair_model_parts(
                        s3, key, upload_id, resolved_url, download_s, part_ranges, parts, job_id
                    )
            upload_id = ""
        model_download_job_update(job_id, status="COMPLETED", phase="Model available on H3 volume", percent=100, downloaded_bytes=downloaded, filename=name, key=key, volume_path=f"{s3.root}/{key}", s3_uri=f"s3://{s3.bucket}/{key}")
    except FileExistsError as e:
        model_download_job_update(job_id, status="EXISTS", phase="File already exists", error=str(e), key=key)
    except Exception as e:
        if upload_id and s3 and key:
            try:
                s3.client.abort_multipart_upload(Bucket=s3.bucket, Key=key, UploadId=upload_id)
            except Exception:
                pass
        model_download_job_update(job_id, status="FAILED", phase="Import failed", error=str(e))

def remote_lora_name(response: requests.Response, requested_name: str = "") -> str:
    if str(requested_name or "").strip():
        return h3_lora_name(requested_name)
    disposition = response.headers.get("content-disposition", "")
    match = re.search(r"filename\*?=(?:UTF-8''|\")?([^\";]+)", disposition, flags=re.I)
    candidate = urllib.parse.unquote(match.group(1).strip()) if match else ""
    if not candidate:
        candidate = urllib.parse.unquote(Path(urllib.parse.urlparse(response.url).path).name)
    if not candidate.lower().endswith(".safetensors"):
        raise ValueError("Could not infer a .safetensors filename from this URL; enter a filename beside the URL")
    return h3_lora_name(candidate)

def h3_estimated_inline_bytes(sources: list[str], delivery: str = "auto") -> int:
    """Estimate the Base64 contribution from all local assets in one request."""
    mode = str(delivery or "auto").lower()
    total = 0
    for source in sources:
        raw = str(source or "").strip()
        if not raw or is_url(raw) or is_s3_uri(raw):
            continue
        path = Path(raw)
        if not path.exists() or not path.is_file():
            continue
        size = path.stat().st_size
        if mode == "auto" and size > H3_INLINE_FILE_LIMIT_BYTES:
            continue
        total += 4 * ((size + 2) // 3) + 512
    return total


def h3_effective_delivery(delivery: str, sources: list[str]) -> tuple[str, int]:
    """Choose one safe delivery mode using the combined H3 request size."""
    mode = str(delivery or "auto").lower()
    if mode not in {"auto", "base64", "s3_url", "volume_path"}:
        raise ValueError("H3 delivery must be auto, base64, or network volume")
    estimated = h3_estimated_inline_bytes(sources, mode)
    if mode == "auto" and estimated > H3_INLINE_BODY_BUDGET_BYTES:
        return "volume_path", estimated
    if mode == "base64" and estimated > H3_INLINE_BODY_BUDGET_BYTES:
        raise ValueError(
            "The selected H3 files would make the Base64 request too large for RunPod's 10 MiB limit. "
            "Choose Auto or H3 Network Volume delivery."
        )
    return mode, estimated


def h3_asset_payload(source: str, kind: str, delivery: str, s3: Optional[S3Helper]) -> dict[str, str]:
    """Build an H3 asset, preferring the endpoint's mounted network volume.

    RunPod's network-volume S3 gateway requires authenticated requests and does
    not reliably accept presigned GET URLs.  When an object belongs to the H3
    volume, give the worker its mounted path instead.
    """
    raw = str(source or "").strip()
    if not raw:
        raise ValueError(f"H3 {kind.replace('_', ' ')} is required")
    fallback_name = f"{kind}.bin"
    if is_url(raw):
        name = Path(urllib.parse.urlparse(raw).path).name or fallback_name
        return {"name": clean_name(name), "url": raw}
    if is_s3_uri(raw):
        if not s3:
            raise ValueError("H3 S3 settings are required to use an s3:// input")
        bucket, key = s3_key_from_uri(raw)
        if bucket != s3.bucket:
            raise ValueError(f"H3 input belongs to bucket {bucket}, but H3 storage is configured for {s3.bucket}")
        return {"name": clean_name(Path(key).name or fallback_name), "volume_path": f"{s3.root}/{key.lstrip('/')}"}
    path = Path(raw)
    if not path.exists() or not path.is_file():
        raise ValueError(f"H3 {kind.replace('_', ' ')} file not found: {raw}")
    mode = str(delivery or "auto").lower()
    if mode == "auto":
        mode = "base64" if path.stat().st_size <= H3_INLINE_FILE_LIMIT_BYTES else "volume_path"
    if mode == "base64":
        return {"name": clean_name(path.name or fallback_name), "data": base64.b64encode(path.read_bytes()).decode("ascii")}
    if mode in {"s3_url", "volume_path"}:
        if not s3:
            raise ValueError("H3 input requires network-volume delivery. Configure the H3 network-volume S3 settings, or choose Base64 for a smaller file.")
        uploaded = s3.upload(path, kind)
        return {"name": clean_name(path.name or fallback_name), "volume_path": uploaded["volume"]}
    raise ValueError("H3 delivery must be auto, base64, or network volume")

def h3_path_list(value: Any, maximum: int, label: str) -> list[str]:
    if isinstance(value, str):
        value = [line.strip() for line in value.splitlines() if line.strip()]
    if value is None:
        value = []
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    paths = [str(path).strip() for path in value if str(path).strip()]
    if len(paths) > maximum:
        raise ValueError(f"{label} supports up to {maximum} items")
    return paths

def media_payload(kind: str, source: str, delivery: str, s3: Optional[S3Helper]) -> dict[str,Any]:
    """Create media payload exactly like the old desktop app.

    Desktop behavior:
    - URL input -> <kind>_url
    - auto mode -> base64 for files <= 8 MB, runpod_volume_path for files > 8 MB
    - runpod_volume_path -> upload to S3, then send /runpod-volume/<key>
    - base64 -> send <kind>_base64
    """
    source=(source or "").strip()
    if not source: return {}
    if is_url(source): return {f"{kind}_url": source}
    if is_s3_uri(source):
        mode = delivery or "auto"
        if mode in {"global", "auto"}:
            mode = "runpod_volume_path"
        if mode == "s3_url":
            return {f"{kind}_url": source, f"_{kind}_s3_uri": source}
        if mode == "runpod_volume_path":
            if not s3: raise ValueError("S3 settings required for RunPod volume delivery")
            bucket, key = s3_key_from_uri(source)
            if bucket == s3.bucket:
                return {f"{kind}_path": f"{s3.root}/{key}", f"_{kind}_s3_uri": source}
            source = cached_s3_object_path(source, media_ext_from_string(source))
        elif mode == "base64":
            source = cached_s3_object_path(source, media_ext_from_string(source))
        else:
            raise ValueError(f"Unsupported delivery mode: {delivery}")
    path = Path(source)
    if not path.exists(): raise ValueError(f"{kind} file not found: {source}")

    mode = delivery or "auto"
    if mode == "global":
        mode = "auto"
    if mode == "auto":
        size_mb = path.stat().st_size / (1024 * 1024)
        mode = "runpod_volume_path" if size_mb > 8 else "base64"

    if mode == "base64":
        return {f"{kind}_base64": base64.b64encode(path.read_bytes()).decode("utf-8")}
    if mode == "s3_url":
        if not s3: raise ValueError("S3 settings required for S3 URL delivery")
        up=s3.upload(source, kind)
        return {f"{kind}_url": up["s3"], f"_{kind}_s3_uri": up["s3"]}
    if mode == "runpod_volume_path":
        if not s3: raise ValueError("S3 settings required for RunPod volume delivery")
        up=s3.upload(source, kind)
        return {f"{kind}_path": up["volume"], f"_{kind}_s3_uri": up["s3"]}

    raise ValueError(f"Unsupported delivery mode: {delivery}")

def remove_media_payload_keys(payload: dict[str, Any], kinds: list[str]) -> None:
    for kind in kinds:
        for suffix in ("path", "url", "base64"):
            payload.pop(f"{kind}_{suffix}", None)
        payload.pop(f"_{kind}_s3_uri", None)

def policy(s: dict[str,Any]) -> dict[str,int]:
    secs=max(float(s.get("timeout_seconds") or 3600),900.0); ttl=max(secs*2, secs+3600); return {"executionTimeout":int(secs*1000),"ttl":int(ttl*1000)}

def submit(endpoint: str, key: str, payload: dict[str,Any], s: dict[str,Any]) -> dict[str,Any]:
    try:
        r=requests.post(f"https://api.runpod.ai/v2/{endpoint}/run", headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"}, json={"input":payload,"policy":policy(s)}, timeout=120); r.raise_for_status(); return r.json()
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else None
        body = ""
        if e.response is not None:
            try:
                body = (e.response.text or "").strip()
            except Exception:
                body = ""
        if len(body) > 1800:
            body = body[:1800] + "...[truncated]"
        if code == 404:
            raise HTTPException(502, f"RunPod endpoint not found or not accessible: {endpoint}. Check the endpoint ID in Settings and confirm this API key can access it.") from e
        detail = f"RunPod submit failed: {e}"
        if body:
            detail += f" | response: {body}"
        raise HTTPException(502, detail) from e
    except requests.RequestException as e:
        raise HTTPException(502, f"RunPod submit failed: {e}") from e

def status(endpoint: str, key: str, job_id: str) -> dict[str,Any]:
    try:
        r=requests.get(f"https://api.runpod.ai/v2/{endpoint}/status/{job_id}", headers={"Authorization":f"Bearer {key}"}, timeout=120); r.raise_for_status(); return r.json()
    except requests.RequestException as e:
        raise HTTPException(502, f"RunPod status check failed: {e}") from e

def output_folder_for_endpoint(endpoint_id: str, s: dict[str, Any]) -> tuple[Path, str]:
    base = Path(s.get("output_dir") or OUTPUT_DIR)
    if endpoint_id == s.get("infinite_endpoint_id"):
        return base / "infinite", "infinite"
    if endpoint_id == s.get("wan_endpoint_id"):
        return base / "animate", "animate"
    if endpoint_id == s.get("h3_endpoint_id"):
        return base / "h3", "h3"
    return base / "runs", "run"

def next_output_path(out_dir: Path, prefix: str, ext: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ext if ext.startswith(".") else f".{ext}"
    nums = []
    for p in out_dir.glob(f"{prefix}_*{ext}"):
        m = re.match(rf"{re.escape(prefix)}_(\d+){re.escape(ext)}$", p.name)
        if m:
            nums.append(int(m.group(1)))
    return out_dir / f"{prefix}_{(max(nums) + 1 if nums else 1):04d}{ext}"

def next_numbered_dir(root: Path, prefix: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    nums: list[int] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = re.match(rf"^{re.escape(prefix)}(\d+)$", p.name, flags=re.I)
        if m:
            nums.append(int(m.group(1)))
    out = root / f"{prefix}{(max(nums) + 1 if nums else 1):03d}"
    out.mkdir(parents=True, exist_ok=False)
    return out

def samimate_root_dir() -> Path:
    return Path(settings().get("output_dir") or DEFAULT_OUTPUT_DIR) / "samimate"

def media_item(path: str | Path, label: str = "") -> dict[str, str]:
    p = Path(path)
    item = {"path": str(p), "url": safe_file_url(p), "kind": media_kind(p)}
    if label:
        item["label"] = label
    return item

def copy_media_named(src: str | Path, out_dir: Path, name: str) -> str:
    src_path = Path(resolve_sam_source(str(src)))
    if not src_path.exists() or not src_path.is_file():
        raise ValueError(f"Media file not found: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = src_path.suffix or ".bin"
    safe = clean_name(Path(name).stem or "media")
    dst = out_dir / f"{safe}{suffix}"
    if src_path.resolve() != dst.resolve():
        shutil.copy2(src_path, dst)
    return str(dst)

def media_ext_from_string(value: str) -> str:
    ext = Path(urllib.parse.urlparse(value.split("?", 1)[0]).path).suffix.lower()
    return ext if ext in MEDIA_EXT else ".bin"

def extract_media_references(text: str, s: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    raw = (text or "").strip()
    if not raw:
        return refs
    refs.append(raw)
    refs.extend(re.findall(r"s3://[^\s'\"<>]+", raw, flags=re.I))
    refs.extend(re.findall(r"https?://[^\s'\"<>]+", raw, flags=re.I))
    root = str(s.get("runpod_volume_root") or "/runpod-volume").rstrip("/")
    root_re = re.escape(root).replace("/", r"[\\/]")
    refs.extend(re.findall(root_re + r"[^\s'\"<>]+", raw, flags=re.I))
    # Some endpoints return only the object key or a message containing one.
    refs.extend(re.findall(r"(?:output|outputs|ComfyUI/output|input/runpod-media-console/(?:output|outputs|video|image|audio|mask))[\\/][^\r\n'\"<>]+\.(?:mp4|webm|mov|png|jpe?g|wav|mp3|m4a)", raw, flags=re.I))
    out: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        ref = ref.strip().strip(".,;:)]}")
        if ref and ref not in seen:
            out.append(ref)
            seen.add(ref)
    return out

def s3_output_prefix_candidates(s: dict[str, Any]) -> list[str]:
    candidates = ["output", "outputs", "ComfyUI/output"]
    s3_prefix = str(s.get("s3_prefix") or "").strip("/\\")
    if s3_prefix:
        candidates.extend([
            f"{s3_prefix}/output",
            f"{s3_prefix}/outputs",
            f"{s3_prefix}/ComfyUI/output",
        ])
    out: list[str] = []
    seen: set[str] = set()
    for p in candidates:
        p = p.strip("/\\")
        if p and p not in seen:
            out.append(p)
            seen.add(p)
    return out

def list_s3_objects_paginated(s3: S3Helper, prefix: str, max_objects: int = 5000) -> list[dict[str, Any]]:
    """List nested objects under a prefix, including pages beyond the first 50/1000 keys."""
    found: list[dict[str, Any]] = []
    token: Optional[str] = None
    while len(found) < max_objects:
        kwargs: dict[str, Any] = {
            "Bucket": s3.bucket,
            "Prefix": prefix.rstrip("/") + "/",
            "MaxKeys": min(1000, max_objects - len(found)),
        }
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.call("list_objects_v2", **kwargs)
        found.extend(page.get("Contents") or [])
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
        if not token:
            break
    return found


def download_recent_s3_outputs(s3: Optional[S3Helper], s: dict[str, Any], out_dir: Path, prefix: str, already_saved: int, job_id: str = "", strict_job: bool = False) -> list[str]:
    if not s3 or already_saved:
        return []
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_prefixes = s3_output_prefix_candidates(s)
    search_prefixes: list[str] = []
    if job_id:
        # H3 workers persist to output/h3/<job-id>/<filename>. Search that
        # exact nested folder before considering any unrelated recent output.
        search_prefixes.extend([f"output/h3/{job_id}", f"outputs/h3/{job_id}"])
        search_prefixes.extend(f"{pfx}/{job_id}" for pfx in base_prefixes)
    if not (strict_job and job_id):
        search_prefixes.extend(base_prefixes)
    for pfx in dict.fromkeys(search_prefixes):
        try:
            objects = list_s3_objects_paginated(s3, pfx)
        except Exception:
            continue
        for obj in objects:
            key = str(obj.get("Key") or "")
            if not key or key.endswith("/") or key in seen:
                continue
            if Path(key).suffix.lower() not in MEDIA_EXT:
                continue
            seen.add(key)
            found.append(obj)
    def modified_ts(obj: dict[str, Any]) -> float:
        value = obj.get("LastModified")
        try:
            return float(value.timestamp())
        except Exception:
            return 0.0
    found.sort(key=modified_ts, reverse=True)
    saved: list[str] = []
    for obj in found[:3]:
        key = str(obj.get("Key") or "")
        try:
            saved.append(s3.download_key(key, next_output_path(out_dir, prefix, media_ext_from_string(key))))
        except Exception:
            continue
    return saved

def mirror_saved_outputs_to_s3(saved: list[str], s3: Optional[S3Helper], s: dict[str, Any], prefix: str) -> list[str]:
    if not s3 or not saved:
        return []
    root = str(s.get("s3_output_prefix") or "output").strip("/\\") or "output"
    uploaded: list[str] = []
    for path in saved:
        p = Path(path)
        if not p.exists() or not p.is_file():
            continue
        key = f"{root}/{prefix}/{p.name}"
        try:
            uploaded.append(s3.upload_to_key(p, key)["s3"])
        except Exception:
            continue
    return uploaded

def s3_key_from_volume_path(ref: str, s: dict[str, Any]) -> str:
    root = str(s.get("runpod_volume_root") or "/runpod-volume").rstrip("/").replace("\\", "/")
    normalized = ref.replace("\\", "/")
    if normalized.startswith(root + "/"):
        return normalized[len(root) + 1:]
    return normalized.lstrip("/")

def save_outputs(job_id: str, st: dict[str,Any], s: dict[str,Any], endpoint_id: str = "", target_fps: Any = None) -> list[str]:
    out_dir, prefix = output_folder_for_endpoint(endpoint_id, s); out_dir.mkdir(parents=True, exist_ok=True); saved=[]
    storage_s = storage_settings_for_endpoint(s, endpoint_id)
    errors: list[str] = []
    raw_path = next_output_path(out_dir, f"{prefix}_job", ".json")
    try:
        raw_path.write_text(json.dumps(st.get("output"), indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    try: s3=configured_s3_helper(storage_s)
    except Exception: s3=None
    seen_refs: set[str] = set()
    worker_saved_uris: list[str] = []
    def save_ref(q: str):
        q = str(q or "").strip()
        if not q or q in seen_refs:
            return
        seen_refs.add(q)
        if q.startswith("s3://") and s3:
            try:
                ext = media_ext_from_string(q)
                saved.append(s3.download(q, out_dir, next_output_path(out_dir, prefix, ext)))
            except Exception as e: errors.append(f"S3 URI download failed for {q[:180]}: {e}")
        elif q.lower().startswith(("http://", "https://")) and preview_kind_from_string(q) != "file":
            try:
                r = requests.get(q, timeout=120)
                r.raise_for_status()
                data = r.content
                if len(data) > 64:
                    out = next_output_path(out_dir, prefix, ext_from_url_or_bytes(q, data))
                    out.write_bytes(data)
                    saved.append(str(out))
            except Exception as e: errors.append(f"URL download failed for {q[:180]}: {e}")
        elif (q.startswith("/") or q.startswith("\\")) and s3 and any(q.lower().endswith(ext) for ext in MEDIA_EXT):
            try:
                key = s3_key_from_volume_path(q, storage_s)
                saved.append(s3.download_key(key, next_output_path(out_dir, prefix, media_ext_from_string(q))))
            except Exception as e: errors.append(f"RunPod volume download failed for {q[:180]}: {e}")
        elif any(q.lower().endswith(ext) for ext in MEDIA_EXT) and s3 and not Path(q).exists():
            try:
                saved.append(s3.download_key(q, next_output_path(out_dir, prefix, media_ext_from_string(q))))
            except Exception as e: errors.append(f"S3 key download failed for {q[:180]}: {e}")
        elif len(q) < 500 and Path(q).exists() and Path(q).is_file():
            try:
                src = Path(q)
                out = next_output_path(out_dir, prefix, src.suffix or ".bin")
                shutil.copy2(src, out)
                saved.append(str(out))
            except Exception as e: errors.append(f"Local output copy failed for {q[:180]}: {e}")
        elif len(q)>200:
            try:
                data=base64.b64decode(q.split(",", 1)[-1], validate=True)
                if len(data)>64:
                    out=next_output_path(out_dir, prefix, ext_from_bytes(data)); out.write_bytes(data); saved.append(str(out))
            except Exception as e: errors.append(f"Base64 output decode failed: {e}")
    def save_worker_file(d: dict[str, Any]) -> bool:
        """Save the H3 worker's exact {filename, type, data} file object.

        Handling this before generic Comfy filename probing prevents a valid
        inline result from triggering a series of slow, guaranteed S3 404s.
        """
        name = str(d.get("filename") or "").strip().strip("/\\")
        delivery_type = str(d.get("type") or "").strip().lower()
        data = d.get("data")
        if not name or not any(name.lower().endswith(ext) for ext in MEDIA_EXT) or not isinstance(data, str):
            return False
        before = len(saved)
        if delivery_type == "base64" or (not delivery_type and len(data) > 200):
            try:
                decoded = base64.b64decode(data.split(",", 1)[-1], validate=True)
                if len(decoded) <= 64:
                    raise ValueError("decoded file is empty")
                suffix = Path(name).suffix.lower()
                out = next_output_path(out_dir, prefix, suffix if suffix in MEDIA_EXT else ext_from_bytes(decoded))
                out.write_bytes(decoded)
                saved.append(str(out))
            except Exception as e:
                errors.append(f"Worker base64 output decode failed for {name[:180]}: {e}")
            return True
        if delivery_type in {"s3_url", "url", "volume_path", "path", "s3_uri"} or is_url(data) or is_s3_uri(data) or data.startswith(("/", "\\")):
            save_ref(data)
            if len(saved) > before and s3 and delivery_type in {"volume_path", "path"} and data.startswith(("/", "\\")):
                key = s3_key_from_volume_path(data, storage_s)
                worker_saved_uris.append(f"s3://{s3.bucket}/{key}")
            return len(saved) > before or bool(data)
        return False
    def save_comfy_output(d: dict[str, Any]):
        name = str(d.get("filename") or d.get("file") or "").strip().strip("/\\")
        if not name or not any(name.lower().endswith(ext) for ext in MEDIA_EXT):
            return
        subfolder = str(d.get("subfolder") or "").strip().strip("/\\")
        rel = f"{subfolder}/{name}" if subfolder else name
        candidates = [
            rel,
            f"output/{rel}",
            f"outputs/{rel}",
            f"ComfyUI/output/{rel}",
        ]
        s3_prefix = str(storage_s.get("s3_prefix") or "").strip("/\\")
        if s3_prefix:
            candidates.extend([
                f"{s3_prefix}/output/{rel}",
                f"{s3_prefix}/outputs/{rel}",
                f"{s3_prefix}/ComfyUI/output/{rel}",
            ])
        before = len(saved)
        for candidate in candidates:
            save_ref(candidate.replace("\\", "/"))
            if len(saved) > before:
                return
    def save_structured_output(d: dict[str, Any]):
        for key_name in ("s3_uri", "s3_url", "url", "file_url", "output_url", "path", "file_path", "output_path", "video", "video_path", "audio", "audio_path", "image", "image_path", "key", "s3_key"):
            value = d.get(key_name)
            if isinstance(value, str):
                for q in extract_media_references(value, storage_s):
                    save_ref(q)
        bucket = str(d.get("bucket") or d.get("s3_bucket") or "").strip()
        key = str(d.get("key") or d.get("s3_key") or "").strip()
        if key and any(key.lower().endswith(ext) for ext in MEDIA_EXT):
            save_ref(f"s3://{bucket}/{key}" if bucket else key)
    def walk(x):
        if isinstance(x, dict):
            handled_worker_file = save_worker_file(x)
            if not handled_worker_file:
                save_comfy_output(x)
                save_structured_output(x)
            for key, value in x.items():
                if handled_worker_file and key in {"filename", "type", "data", "size", "mime_type"}:
                    continue
                walk(value)
        elif isinstance(x, list):
            for v in x: walk(v)
        elif isinstance(x, str):
            for q in extract_media_references(x, storage_s):
                save_ref(q)
    walk(st.get("output"))
    saved.extend(download_recent_s3_outputs(
        s3,
        storage_s,
        out_dir,
        prefix,
        len(saved),
        job_id=job_id,
        strict_job=bool(endpoint_id and endpoint_id == s.get("h3_endpoint_id")),
    ))
    uploaded = worker_saved_uris or mirror_saved_outputs_to_s3(saved, s3, storage_s, prefix)
    if uploaded:
        st["_uploaded_s3"] = uploaded
    if errors and not saved:
        st["_download_errors"] = errors[:30]
    return saved

def preview_kind_from_string(value: str) -> str:
    lower = value.split("?", 1)[0].lower()
    ext = Path(urllib.parse.urlparse(lower).path).suffix
    if ext in IMAGE_EXT: return "image"
    if ext in VIDEO_EXT: return "video"
    if ext in AUDIO_EXT: return "audio"
    return "file"

def output_preview_items(st: dict[str, Any], saved: list[str]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for p in saved:
        path = Path(p)
        if path.exists() and path.is_file():
            key = str(path)
            if key not in seen:
                items.append({"path": key, "url": safe_file_url(path), "kind": media_kind(path)})
                seen.add(key)
    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
        elif isinstance(x, str):
            q = x.strip()
            if not q or q in seen: return
            if q.lower().startswith(("http://", "https://")):
                kind = preview_kind_from_string(q)
                if kind != "file":
                    items.append({"path": q, "url": q, "kind": kind})
                    seen.add(q)
            else:
                p = Path(q)
                if p.exists() and p.is_file():
                    key = str(p)
                    if key not in seen:
                        items.append({"path": key, "url": safe_file_url(p), "kind": media_kind(p)})
                        seen.add(key)
    walk(st.get("output"))
    return items

def output_summary(st: dict[str, Any], s: dict[str, Any]) -> dict[str, Any]:
    refs: list[str] = []
    keys: list[str] = []
    def walk(x: Any, path: str = "output") -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                keys.append(f"{path}.{k}")
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            keys.append(f"{path}[{len(x)}]")
            for i, v in enumerate(x[:20]):
                walk(v, f"{path}[{i}]")
        elif isinstance(x, str):
            for ref in extract_media_references(x, s):
                if ref not in refs and len(ref) < 500:
                    refs.append(ref)
    walk(st.get("output"))
    return {"type": type(st.get("output")).__name__, "keys": keys[:40], "refs": refs[:20]}

def redact_for_ui(data: Any) -> Any:
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if any(word in k.lower() for word in ("key", "token", "secret")):
                out[k] = "***" if v else ""
            else:
                out[k] = redact_for_ui(v)
        return out
    if isinstance(data, list):
        return [redact_for_ui(x) for x in data]
    return data

def explain_error_text(text: str) -> list[str]:
    t = text or ""
    hints: list[str] = []
    if "endpoint not found" in t.lower() or "404" in t:
        hints.append("Endpoint ID or API-key access is wrong. Check the selected endpoint profile and RunPod endpoint ID.")
    if "queue_prompt" in t or "HTTP Error 400" in t:
        hints.append("Comfy rejected the workflow before generation. Test Wan Delivery = Base64, then isolate mask/control-points/mode.")
    if "WebSocketConnectionClosedException" in t or "Connection to remote host was lost" in t:
        hints.append("The endpoint worker lost its Comfy websocket. This is usually endpoint-side instability, VRAM pressure, or a crashed workflow.")
    if "No previewable outputs" in t:
        hints.append("The endpoint completed without returning downloadable media references. Check network-volume output and S3/output folder settings.")
    if "S3 settings required" in t:
        hints.append("The selected delivery mode needs complete S3 bucket/access settings, or switch delivery to Base64 for a small test.")
    if "no video stream" in t.lower():
        hints.append("The selected file is not a playable video. Use Repair Media to convert it to browser-compatible H.264 MP4.")
    if "mask" in t.lower() and ("size" in t.lower() or "resolution" in t.lower()):
        hints.append("Mask and source video should match resolution, duration, and frame rate. Run mask repair or regenerate the SAM mask from the same source video.")
    if "lora_name" in t.lower() and "value not in list" in t.lower():
        hints.append("The H3 worker expects LoRAs under the H3 subfolder (H3/filename.safetensors). Refresh the H3 LoRA library and re-add the LoRA before retrying.")
    if "too large for base64" in t.lower():
        hints.append("The worker generated the file but had no persistent output destination. The H3 endpoint template must set OUTPUT_VOLUME_DIR=/runpod-volume/output/h3, then the job must be resubmitted.")
    if not hints and t.strip():
        hints.append("No known pattern matched. Inspect the raw error and compare payload keys, endpoint ID, and media delivery mode.")
    return hints

def record_job_event(item: dict[str, Any]) -> None:
    item = dict(item)
    item.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        append_limited_json(JOB_HISTORY_PATH, item, 120)
    except OSError:
        # Job history is diagnostic bookkeeping. A transient Windows lock or
        # inherited ACL must never turn a successful submit/download into 500.
        return

def job_event_for(job_id: str, target: str = "") -> dict[str, Any]:
    if not job_id:
        return {}
    for item in load_json(JOB_HISTORY_PATH, []):
        if str(item.get("job_id") or "") != str(job_id):
            continue
        if target and item.get("target") != target:
            continue
        return item if isinstance(item, dict) else {}
    return {}

def effective_video_duration(meta: dict[str, Any], start: Any = None, end: Any = None, cap: Any = None) -> dict[str, Any]:
    raw = float(meta.get("duration_seconds") or 0)
    fps = float(meta.get("fps") or 0)
    st = seconds(start) or 0.0
    en = seconds(end)
    has_trim = st > 0.001 or en is not None
    actual_start = min(max(st, 0.0), raw) if raw else max(st, 0.0)
    actual_end = min(en, raw) if en is not None and raw else (en if en is not None else raw)
    if actual_end is None:
        actual_end = raw
    actual = max(0.0, actual_end - actual_start)
    cap_frames = int(cap or 0)
    if has_trim:
        cap_frames = 0
    if cap_frames > 0 and fps > 0:
        actual = min(actual or raw, cap_frames / fps)
    requested = None
    if en is not None:
        requested = max(0.0, en - st)
    elif has_trim and raw:
        requested = max(0.0, raw - st)
    elif cap_frames > 0 and fps > 0:
        requested = cap_frames / fps
    return {"raw": raw, "effective": round(actual, 3), "requested": round(requested, 3) if requested is not None else None, "start": st, "end": en, "cap_frames": cap_frames, "fps": fps}

def validation_result(tab: str, data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    info: dict[str, Any] = {}
    s = runtime_settings(data.get("settings", {}))
    samimate_h3 = tab == "samimate" and str(data.get("generation_backend") or "wan").lower() == "h3"
    endpoint = s.get("h3_endpoint_id") if tab == "h3" or samimate_h3 else (s.get("wan_endpoint_id") if tab in {"wan", "samimate"} else s.get("infinite_endpoint_id"))
    if not s.get("runpod_api_key"):
        errors.append("RunPod API key is missing.")
    if not endpoint:
        errors.append(("MiniMax H3" if tab == "h3" or samimate_h3 else ("WanAnimate" if tab in {"wan", "samimate"} else "InfiniteTalk")) + " endpoint ID is missing.")
    paths: dict[str, str] = {}
    if tab == "inf":
        if data.get("input_type") == "video":
            paths["video"] = data.get("video_path", "")
        else:
            paths["image"] = data.get("image_path", "")
        paths["audio"] = data.get("audio_path", "")
        if data.get("person_count") == "multi":
            paths["audio2"] = data.get("audio2_path", "")
    elif tab == "wan":
        paths = {"image": data.get("image_path", ""), "video": data.get("video_path", ""), "mask": data.get("mask_path", ""), "masked_video": data.get("masked_video_path", "")}
    elif tab == "samimate":
        paths = {"image": data.get("image_path", ""), "video": data.get("video_path", "")}
    elif tab == "h3":
        task = str(data.get("task") or "fl2v").lower()
        turbo_enabled = as_bool(data.get("turbo_enabled"), True)
        sampler = str(data.get("sampler") or ("h3_turbo" if turbo_enabled else "res_multistep")).lower()
        scheduler = str(data.get("scheduler") or ("simple" if turbo_enabled else "beta")).lower()
        cache_enabled = as_bool(data.get("cache_enabled"), False)
        if sampler not in H3_SAMPLERS:
            errors.append("sampler is not supported.")
        if scheduler not in H3_SCHEDULERS:
            errors.append("scheduler is not supported.")
        if sampler == "h3_turbo" and not turbo_enabled:
            errors.append("the dedicated H3 Turbo sampler requires the separate Turbo LoRA.")
        if not configured_s3_helper(h3_storage_settings(s)):
            errors.append("H3 S3 bucket and credentials are required for generated video uploads.")
        if not str(data.get("prompt") or "").strip():
            errors.append("prompt is required.")
        if task == "fl2v":
            paths = {"first_frame": data.get("first_frame_path", ""), "last_frame": data.get("last_frame_path", "")}
        elif task == "r2v":
            try:
                references = h3_path_list(data.get("reference_paths"), 9, "R2VA image references")
                videos = h3_path_list(data.get("reference_video_paths"), 3, "R2VA video references")
                audios = h3_path_list(data.get("reference_audio_paths"), 3, "R2VA audio references")
                legacy_audio = str(data.get("audio_path") or "").strip()
                if legacy_audio and not audios:
                    audios = [legacy_audio]
                if not references and not videos and not audios:
                    errors.append("R2VA requires at least one image, video, or audio reference.")
                paths = {f"picture_{i + 1}": path for i, path in enumerate(references)}
                paths.update({f"reference_video_{i + 1}": path for i, path in enumerate(videos)})
                paths.update({f"reference_audio_{i + 1}": path for i, path in enumerate(audios)})
            except ValueError as e:
                errors.append(str(e) + ".")
                paths = {}
        elif task == "t2v":
            paths = {}
        else:
            errors.append("task must be t2v, fl2v or r2v.")
        for field, minimum, maximum in (("duration", 1, 15), ("megapixels", 0.2, 2), ("cache_threshold", 0, 1)):
            try:
                value = float(data.get(field) if data.get(field) not in (None, "") else (2 if field == "duration" else 0.2 if field == "megapixels" else 6 if field == "steps" else 0.18))
                if not minimum <= value <= maximum:
                    errors.append(f"{field} must be between {minimum} and {maximum}.")
            except (TypeError, ValueError):
                errors.append(f"{field} must be numeric.")
        try:
            validate_h3_steps(data.get("steps", 6))
        except ValueError as e:
            errors.append(str(e))
        required_models = {
            "main model": data.get("ref2va_model") if task == "r2v" else data.get("fl2va_model"),
            "text encoder / CLIP": data.get("text_encoder"),
            "video VAE": data.get("video_vae"),
            "audio VAE": data.get("audio_vae"),
        }
        for label, value in required_models.items():
            try:
                h3_model_name(value, label)
            except ValueError as e:
                errors.append(str(e) + ".")
        if turbo_enabled:
            try:
                h3_worker_lora_name(str(data.get("turbo_lora") or ""))
            except ValueError as e:
                errors.append("Turbo LoRA is required and must be a .safetensors file.")
    for name, raw in paths.items():
        raw = (raw or "").strip()
        if name in {"image", "video", "audio", "first_frame"} and not raw:
            errors.append(f"{name} path is required.")
            continue
        if not raw:
            continue
        try:
            meta = media_probe_any(raw)
            info[name] = meta
            if not meta.get("exists") and not is_url(raw):
                errors.append(f"{name} does not exist: {raw}")
        except Exception as e:
            warnings.append(f"{name} probe failed: {e}")
    video = info.get("video") or {}
    mask = info.get("mask") or {}
    if tab == "wan":
        if not paths.get("mask") and as_bool(data.get("mask_editing"), False):
            errors.append("Mask edit mode is enabled but no subject mask is set.")
        if video and mask and mask.get("kind") == "video":
            if video.get("width") and mask.get("width") and (video["width"], video["height"]) != (mask["width"], mask["height"]):
                warnings.append(f"Video and mask resolution differ: video {video.get('width')}x{video.get('height')}, mask {mask.get('width')}x{mask.get('height')}. The app will align the mask before submit.")
            video_eff = effective_video_duration(video, data.get("video_start"), data.get("video_end"), data.get("video_frame_cap"))
            mask_pretrimmed = wan_mask_is_pretrimmed(data, paths.get("mask", ""), None, mask, video_eff)
            if mask_pretrimmed:
                mask_eff = effective_video_duration(mask, None, None, None)
            else:
                mask_eff = effective_video_duration(mask, data.get("video_start"), data.get("video_end"), data.get("video_frame_cap"))
            info["effective_video_duration"] = video_eff
            info["effective_mask_duration"] = mask_eff
            info["mask_pretrimmed"] = mask_pretrimmed
            if video_eff.get("effective") and not mask_eff.get("effective"):
                warnings.append("Mask video has no readable duration after applying the current Wan trim/frame cap.")
            elif video_eff.get("effective") and mask_eff.get("effective"):
                source_note = "pre-trimmed SAM mask" if mask_pretrimmed else "mask"
                info["mask_duration_note"] = f"Submit will conform the {source_note} to the selected input video window: mask {mask_eff['effective']}s -> video {video_eff['effective']}s."
        if video and (int(data.get("width") or 0) <= 0 or int(data.get("height") or 0) <= 0):
            info["resolved_size"] = {"width": align16(video.get("width") or 832, 832), "height": align16(video.get("height") or 480, 480)}
        if data.get("seed") is not None and int(float(data.get("seed") or 0)) < 0:
            warnings.append("Seed is negative. The app will replace it with a valid random seed.")
    if tab == "samimate":
        cap = int(data.get("samimate_mask_frame_cap") or data.get("mask_frame_cap") or data.get("video_frame_cap") or 0)
        if cap <= 0:
            warnings.append("SAMimate frame cap is 0 or blank; full video processing can be slow.")
        if samimate_h3:
            try:
                duration_value = float(data.get("duration") or 2)
                if not 1 <= duration_value <= 15:
                    errors.append("H3 SAMimate duration must be between 1 and 15 seconds.")
            except (TypeError, ValueError):
                errors.append("H3 SAMimate duration must be numeric.")
            if not configured_s3_helper(h3_storage_settings(s)):
                errors.append("H3 S3 bucket and credentials are required for H3 SAMimate output.")
    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info}

def repair_output_dir(kind: str) -> Path:
    out = Path(settings().get("output_dir") or DEFAULT_OUTPUT_DIR) / "repaired" / kind
    out.mkdir(parents=True, exist_ok=True)
    return out

def repair_media_file(path: str, action: str, width: int = 0, height: int = 0, fps: float = 0) -> dict[str, Any]:
    src = local_media_path(path)
    if not src.exists():
        raise ValueError(f"File not found: {path}")
    kind = media_kind(src)
    if action == "browser_mp4":
        if kind != "video":
            raise ValueError("Browser MP4 repair needs a video file")
        out = next_output_path(repair_output_dir("video"), "repaired", ".mp4")
        run_media_cmd([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(out)], f"Browser MP4 repair for {src.name}")
    elif action == "normalize_fps":
        if kind != "video":
            raise ValueError("FPS normalization needs a video file")
        out = next_output_path(repair_output_dir("video"), "fps", ".mp4")
        target_fps = fps or float(video_probe(src).get("fps") or 16)
        run_media_cmd([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-r", str(target_fps), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out)], f"FPS normalization for {src.name}")
    elif action == "resize":
        if kind != "video":
            raise ValueError("Resize repair needs a video file")
        if not width or not height:
            raise ValueError("Resize needs width and height")
        out = next_output_path(repair_output_dir("video"), "resized", ".mp4")
        run_media_cmd([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vf", f"scale={align16(width)}:{align16(height)}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(out)], f"Resize repair for {src.name}")
    elif action == "binary_mask":
        if kind != "video":
            raise ValueError("Mask repair needs a video mask")
        out = next_output_path(repair_output_dir("mask"), "mask", ".mp4")
        run_media_cmd([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-vf", "format=gray,lutyuv=y='if(gt(val,127),255,0)'", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p", str(out)], f"Binary mask repair for {src.name}")
    else:
        raise ValueError(f"Unsupported repair action: {action}")
    return {"path": str(out), "url": safe_file_url(out), "kind": media_kind(out), "probe": media_probe_any(str(out))}

def image_editor_output_dir() -> Path:
    out = Path(settings().get("output_dir") or DEFAULT_OUTPUT_DIR) / "image_editor"
    out.mkdir(parents=True, exist_ok=True)
    return out

def image_editor_load(path: str):
    from PIL import Image, ImageOps
    src = local_media_path(path)
    if not src.exists():
        raise ValueError(f"Image not found: {path}")
    if src.suffix.lower() not in IMAGE_EXT:
        raise ValueError("Image Editor needs an image file")
    return ImageOps.exif_transpose(Image.open(src)).convert("RGBA"), src

def image_editor_filter(img, name: str, amount: float = 1.0):
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    name = (name or "none").lower()
    amount = float(amount or 1.0)
    if name in {"", "none"}:
        return img
    if name == "grayscale":
        return ImageOps.grayscale(img).convert("RGBA")
    if name == "contrast":
        return ImageEnhance.Contrast(img).enhance(max(0.0, amount))
    if name == "sharp":
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(120 * max(0.1, amount)), threshold=3))
    if name == "warm" or name == "cool":
        r, g, b, a = img.split()
        if name == "warm":
            r = ImageEnhance.Brightness(r).enhance(1.0 + 0.12 * amount)
            b = ImageEnhance.Brightness(b).enhance(max(0.0, 1.0 - 0.10 * amount))
        else:
            b = ImageEnhance.Brightness(b).enhance(1.0 + 0.12 * amount)
            r = ImageEnhance.Brightness(r).enhance(max(0.0, 1.0 - 0.10 * amount))
        return Image.merge("RGBA", (r, g, b, a))
    if name == "vintage":
        gray = ImageOps.grayscale(img)
        sepia = ImageOps.colorize(gray, "#241306", "#f1d5a0").convert("RGBA")
        return Image.blend(img, sepia, min(max(amount, 0.0), 1.0))
    if name == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=max(0.1, amount * 2.0)))
    return img

def mask_from_polygon(size: tuple[int, int], points: list[dict[str, Any]], offset: tuple[int, int] = (0, 0)):
    from PIL import Image, ImageDraw
    mask = Image.new("L", size, 0)
    if len(points or []) < 3:
        return mask
    ox, oy = offset
    poly = [(float(p.get("x", 0) or 0) - ox, float(p.get("y", 0) or 0) - oy) for p in points]
    ImageDraw.Draw(mask).polygon(poly, fill=255)
    return mask

def apply_alpha_mask(img, mask, mode: str, fill: str = "transparent"):
    from PIL import Image, ImageFilter, ImageOps
    mode = (mode or "none").lower()
    fill = (fill or "transparent").lower()
    if mode not in {"keep", "remove"}:
        return img
    alpha = mask if mode == "keep" else ImageOps.invert(mask)
    if fill == "transparent":
        out = img.copy()
        out.putalpha(Image.composite(img.getchannel("A"), Image.new("L", img.size, 0), alpha))
        return out
    bg_color = (255, 255, 255, 255) if fill == "white" else (0, 0, 0, 255)
    if fill == "blur":
        bg = img.filter(ImageFilter.GaussianBlur(radius=14))
    else:
        bg = Image.new("RGBA", img.size, bg_color)
    return Image.composite(img, bg, alpha)

def transform_points_for_crop(points: dict[str, Any], x: int, y: int) -> dict[str, Any]:
    out = {"positive": [], "negative": []}
    for key in ("positive", "negative"):
        for p in (points or {}).get(key, []) or []:
            out[key].append({"x": float(p.get("x", 0) or 0) - x, "y": float(p.get("y", 0) or 0) - y})
    return out

def image_editor_apply(data: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image
    img, src = image_editor_load(data.get("path", ""))
    original_size = img.size
    crop = data.get("crop") or {}
    cx = max(0, int(round(float(crop.get("x", 0) or 0))))
    cy = max(0, int(round(float(crop.get("y", 0) or 0))))
    cw = int(round(float(crop.get("w", 0) or 0)))
    ch = int(round(float(crop.get("h", 0) or 0)))
    if cw > 1 and ch > 1:
        cw = min(cw, img.width - cx)
        ch = min(ch, img.height - cy)
        img = img.crop((cx, cy, cx + cw, cy + ch))
    else:
        cx = cy = 0
    mask = None
    lasso = data.get("lasso") or []
    lasso_mode = (data.get("lasso_mode") or "none").lower()
    if lasso_mode in {"keep", "remove"} and len(lasso) >= 3:
        mask = mask_from_polygon(img.size, lasso, (cx, cy))
    sam = data.get("sam") or {}
    if sam.get("enabled"):
        s = settings()
        sam_s = s.get("sam", {})
        sam_input = TEMP_DIR / f"image_editor_sam_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        img.convert("RGB").save(sam_input)
        sam_points = transform_points_for_crop(sam.get("points") or {}, cx, cy)
        box = dict(sam.get("box") or {})
        if box:
            box["x"] = float(box.get("x", 0) or 0) - cx
            box["y"] = float(box.get("y", 0) or 0) - cy
        runner_data = {
            "text_prompt": sam.get("prompt") or "",
            "prompt_mode": sam.get("prompt_mode") or "all",
            "points": sam_points,
            "box": box,
            "mask_fill": "black",
            "invert": False,
            "hf_token": sam.get("hf_token") or sam_s.get("hf_token", ""),
            "hf_cache_dir": s.get("hf_cache_dir", ""),
        }
        result = run_local_sam3(str(sam_input), str(sam_input), runner_data, sam.get("sam3_python") or sam_s.get("sam3_python") or sys.executable)
        mask_path = result.get("mask_path")
        if mask_path:
            mask = Image.open(mask_path).convert("L").resize(img.size, Image.Resampling.BILINEAR)
    mask_mode = (data.get("mask_mode") or lasso_mode or "none").lower()
    if mask is not None and mask_mode in {"keep", "remove"}:
        img = apply_alpha_mask(img, mask, mask_mode, data.get("mask_fill") or "transparent")
    img = image_editor_filter(img, data.get("filter"), float(data.get("filter_amount") or 1.0))
    scale = max(1.0, min(float(data.get("upscale") or 1.0), 4.0))
    if scale > 1.01:
        img = img.resize((int(round(img.width * scale)), int(round(img.height * scale))), Image.Resampling.LANCZOS)
    out = next_output_path(image_editor_output_dir(), "edit", ".png")
    img.save(out)
    return {
        "path": str(out),
        "url": safe_file_url(out),
        "kind": "image",
        "width": img.width,
        "height": img.height,
        "source_width": original_size[0],
        "source_height": original_size[1],
    }

def recent_outputs(limit: int = 40) -> list[dict[str, Any]]:
    root = Path(settings().get("output_dir") or DEFAULT_OUTPUT_DIR)
    items: list[Path] = []
    if root.exists():
        for p in root.rglob("*"):
            try:
                if p.is_file() and media_kind(p) != "file":
                    items.append(p)
            except Exception:
                pass
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"path": str(p), "url": safe_file_url(p), "kind": media_kind(p), "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)), "size_mb": round(p.stat().st_size/(1024*1024), 3)} for p in items[:limit]]

app = FastAPI(title=APP_NAME)
app.mount("/static", StaticFiles(directory=Path(__file__).parent/"static"), name="static")


@app.middleware("http")
async def require_remote_authentication(request: Request, call_next):
    if request_has_lan_access(request):
        return await call_next(request)
    return JSONResponse(
        status_code=401,
        content={"detail": "LAN access requires the RunPod Media Console username and password shown by run_local_network.bat"},
        headers={"WWW-Authenticate": 'Basic realm="RunPod Media Console"'},
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.exception_handler(subprocess.CalledProcessError)
async def subprocess_error_handler(request, exc: subprocess.CalledProcessError):
    detail = (exc.stderr or exc.stdout or str(exc)).strip() if isinstance(exc.stderr or exc.stdout, str) else str(exc)
    return JSONResponse(status_code=400, content={"detail": detail or str(exc)})

@app.exception_handler(json.JSONDecodeError)
async def json_decode_error_handler(request, exc: json.JSONDecodeError):
    return JSONResponse(status_code=400, content={"detail": f"Invalid JSON: {exc}"})

@app.get("/")
def home(): return FileResponse(Path(__file__).parent/"static"/"index.html")
@app.get("/api/version")
def version(): return {"name":APP_NAME,"version":APP_VERSION}
@app.get("/api/health")
def health(): return {"ok": True, "name": APP_NAME, "version": APP_VERSION}
@app.get("/api/settings")
def get_settings(): return public_settings()
@app.post("/api/settings")
async def post_settings(data: dict[str,Any]): return public_settings(await run_in_threadpool(save_settings, data))
@app.post("/api/settings/use-h3-storage-as-main")
def use_h3_storage_as_main():
    current = settings()
    h3 = h3_storage_settings(current)
    for key in ("s3_endpoint_url", "s3_access_key_id", "s3_secret_access_key", "s3_bucket", "s3_region", "runpod_volume_root"):
        current[key] = h3.get(key, "")
    current["browser_s3_storage"] = "h3"
    return public_settings(save_settings(current))
@app.get("/api/presets")
def get_presets():
    presets = load_json(PRESETS_PATH, {})
    if not isinstance(presets, dict):
        presets = {}
    if presets.pop("wan22", None) is not None:
        save_json(PRESETS_PATH, presets)
    return {"presets": presets}
@app.post("/api/presets")
async def save_preset(data: dict[str, Any]):
    name = clean_name(data.get("name") or "preset")
    tab = data.get("tab") or "wan"
    if tab not in {"inf", "wan", "h3", "samimate"}:
        raise HTTPException(400, "Unsupported preset type.")
    presets = load_json(PRESETS_PATH, {})
    if not isinstance(presets, dict):
        presets = {}
    presets.setdefault(tab, {})[name] = data.get("settings") or {}
    save_json(PRESETS_PATH, presets)
    return {"name": name, "tab": tab, "presets": presets}
@app.delete("/api/presets/{tab}/{name}")
def delete_preset(tab: str, name: str):
    presets = load_json(PRESETS_PATH, {})
    if isinstance(presets, dict):
        presets.get(tab, {}).pop(name, None)
        save_json(PRESETS_PATH, presets)
    return {"presets": presets}
@app.get("/api/endpoint-profiles")
def get_endpoint_profiles():
    profiles = load_json(ENDPOINT_PROFILES_PATH, {})
    if not isinstance(profiles, dict):
        profiles = {}
    clean_profiles: dict[str, Any] = {}
    profile_keys = ["infinite_endpoint_id", "wan_endpoint_id", "h3_endpoint_id", "delivery_mode", "s3_bucket", "s3_prefix", "runpod_volume_root"]
    h3_keys = ["network_volume_id", "s3_endpoint_url", "s3_bucket", "s3_region", "s3_prefix", "runpod_volume_root"]
    for name, raw in profiles.items():
        raw = raw if isinstance(raw, dict) else {}
        profile = {key: raw.get(key, "") for key in profile_keys}
        h3_raw = raw.get("h3_storage") if isinstance(raw.get("h3_storage"), dict) else {}
        profile["h3_storage"] = {key: h3_raw.get(key, "") for key in h3_keys}
        clean_profiles[name] = profile
    if clean_profiles != profiles:
        save_json(ENDPOINT_PROFILES_PATH, clean_profiles)
    return {"profiles": clean_profiles}
@app.post("/api/endpoint-profiles")
async def save_endpoint_profile(data: dict[str, Any]):
    name = clean_name(data.get("name") or "profile")
    profiles = load_json(ENDPOINT_PROFILES_PATH, {})
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[name] = {k: data.get(k, "") for k in ["infinite_endpoint_id", "wan_endpoint_id", "h3_endpoint_id", "delivery_mode", "s3_bucket", "s3_prefix", "runpod_volume_root"]}
    h3_storage = data.get("h3_storage") if isinstance(data.get("h3_storage"), dict) else {}
    profiles[name]["h3_storage"] = {k: h3_storage.get(k, "") for k in ["network_volume_id", "s3_endpoint_url", "s3_bucket", "s3_region", "s3_prefix", "runpod_volume_root"]}
    save_json(ENDPOINT_PROFILES_PATH, profiles)
    return {"name": name, **get_endpoint_profiles()}
@app.delete("/api/endpoint-profiles/{name}")
def delete_endpoint_profile(name: str):
    profiles = load_json(ENDPOINT_PROFILES_PATH, {})
    if isinstance(profiles, dict):
        profiles.pop(name, None)
        save_json(ENDPOINT_PROFILES_PATH, profiles)
    return get_endpoint_profiles()
@app.get("/api/jobs/history")
def get_job_history():
    return {"items": load_json(JOB_HISTORY_PATH, [])}
@app.post("/api/validate/{tab}")
async def validate_tab(tab: str, data: dict[str, Any]):
    return validation_result(tab, data)
@app.post("/api/payload/preview/{tab}")
async def payload_preview(tab: str, data: dict[str, Any]):
    return {"tab": tab, "payload": redact_for_ui(data), "validation": validation_result(tab, data)}
@app.post("/api/media/repair")
async def repair_media(data: dict[str, Any]):
    try:
        return repair_media_file(data.get("path", ""), data.get("action", "browser_mp4"), int(data.get("width") or 0), int(data.get("height") or 0), float(data.get("fps") or 0))
    except Exception as e:
        raise HTTPException(400, str(e))
@app.post("/api/image-editor/apply")
async def post_image_editor_apply(data: dict[str, Any]):
    try:
        return image_editor_apply(data)
    except Exception as e:
        raise HTTPException(400, str(e))
@app.post("/api/wan/align-mask-preview")
async def wan_align_mask_preview(data: dict[str, Any]):
    try:
        if not (data.get("video_path") or "").strip():
            raise HTTPException(400, "Wan reference video is required")
        if not (data.get("mask_path") or "").strip():
            raise HTTPException(400, "Wan subject mask is required")
        return align_wan_mask_to_video(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))
@app.get("/api/outputs/recent")
def get_recent_outputs(limit: int = 40):
    return {"items": recent_outputs(limit)}
@app.post("/api/error/explain")
async def explain_error(data: dict[str, Any]):
    text = data.get("text") or data.get("error") or ""
    return {"hints": explain_error_text(str(text))}
@app.get("/api/cache/info")
def get_cache_info(): return cache_info()
@app.post("/api/cache/cleanup")
async def post_cache_cleanup(data: dict[str,Any] | None = None):
    force = as_bool((data or {}).get("force"), False)
    return {**cleanup_cache(settings(), force=force), "info": cache_info()}
@app.get("/api/tts/voices")
def tts_voices():
    ps = piper_status(settings())
    return {
        "engines": [
            {"id": "windows_sapi", "name": "Windows voices", "available": os.name == "nt", "free": True},
            {"id": "piper", "name": "Piper local model", "available": ps["package_available"] and ps["model_exists"], "free": True},
        ],
        "voices": sapi_voices() if os.name == "nt" else [],
        "output_dir": str(tts_output_dir(settings())),
        "piper": ps,
    }
@app.get("/api/tts/piper/status")
def tts_piper_status():
    return piper_status(settings())
@app.post("/api/tts/piper/setup")
def tts_piper_setup(data: dict[str, Any] | None = None):
    try:
        info = setup_piper_voice(as_bool((data or {}).get("force"), False))
        save_settings({"tts": {"engine": "piper", "piper_model": info["model_path"], "piper_config": info["config_path"]}})
        return {**info, "settings_updated": True}
    except Exception as e:
        raise HTTPException(400, str(e))
@app.post("/api/tts/generate")
def tts_generate(data: dict[str, Any]):
    try:
        return generate_tts_audio(data, settings())
    except Exception as e:
        raise HTTPException(400, str(e))
@app.get("/api/prompts")
def get_prompts(): return load_json(PROMPTS_PATH,{})
@app.post("/api/prompts")
async def put_prompt(data: dict[str,str]):
    name=(data.get("name") or "").strip(); prompt=(data.get("prompt") or "").strip()
    if not name or not prompt: raise HTTPException(400,"name and prompt required")
    p=load_json(PROMPTS_PATH,{}); p[name]=prompt; PROMPTS_PATH.write_text(json.dumps(p,indent=2,ensure_ascii=False), encoding="utf-8"); return p
@app.delete("/api/prompts/{name}")
def delete_prompt(name: str):
    p=load_json(PROMPTS_PATH,{}); p.pop(name,None); PROMPTS_PATH.write_text(json.dumps(p,indent=2,ensure_ascii=False), encoding="utf-8"); return p
@app.get("/api/h3/subjects")
def get_h3_subjects():
    return load_json(H3_SUBJECTS_PATH, {})
@app.post("/api/h3/subjects")
async def put_h3_subject(data: dict[str, str]):
    name = (data.get("name") or "").strip()
    definition = (data.get("definition") or "").strip()
    image_path = (data.get("image_path") or "").strip()
    if not name or not definition:
        raise HTTPException(400, "name and definition required")
    subjects = load_json(H3_SUBJECTS_PATH, {})
    subjects[name] = {"definition": definition, "image_path": image_path}
    H3_SUBJECTS_PATH.write_text(json.dumps(subjects, indent=2, ensure_ascii=False), encoding="utf-8")
    return subjects
@app.delete("/api/h3/subjects/{name}")
def delete_h3_subject(name: str):
    subjects = load_json(H3_SUBJECTS_PATH, {})
    subjects.pop(name, None)
    H3_SUBJECTS_PATH.write_text(json.dumps(subjects, indent=2, ensure_ascii=False), encoding="utf-8")
    return subjects

@app.get("/api/h3/comfyui/status")
def h3_comfyui_status():
    s = settings()
    try:
        pods = list_h3_comfyui_pods(s)
    except Exception as exc:
        raise HTTPException(400, f"Could not read RunPod Pods: {exc}") from exc
    return {
        "pod": pods[0] if pods else None,
        "pods": pods,
        "network_volume_id": str((s.get("h3_storage") or {}).get("network_volume_id") or ""),
        "template_id": H3_COMFYUI_TEMPLATE_ID,
        "template_url": f"https://console.runpod.io/hub/template/{H3_COMFYUI_TEMPLATE_ID}",
        "config_key": H3_COMFYUI_CONFIG_KEY,
    }

@app.post("/api/h3/comfyui/launch")
def h3_comfyui_launch(data: dict[str, Any]):
    if data.get("confirm_cost") is not True:
        raise HTTPException(400, "Confirm GPU billing before starting or creating a ComfyUI Pod")
    s = settings()
    try:
        prepared = prepare_h3_comfyui_volume(s)
        pods = list_h3_comfyui_pods(s)
        requested_id = str(data.get("pod_id") or "").strip()
        pod = next((item for item in pods if item["id"] == requested_id), None) if requested_id else (pods[0] if pods else None)
        created = False
        if pod:
            if pod["status"] != "RUNNING":
                result = runpod_pod_request("POST", f"{pod['id']}/start", s=s)
                pod = h3_comfyui_pod_public(result) if isinstance(result, dict) and result.get("id") else {**pod, "status": "RUNNING", "ready": False}
        else:
            gpu_type = str(data.get("gpu_type") or "NVIDIA GeForce RTX 5090").strip()
            volume_id = str((s.get("h3_storage") or {}).get("network_volume_id") or "").strip()
            if not volume_id:
                raise ValueError("Configure the H3 network volume ID in Settings first")
            result = runpod_pod_request("POST", data={
                "name": "H3 ComfyUI",
                "templateId": H3_COMFYUI_TEMPLATE_ID,
                "computeType": "GPU",
                "gpuTypeIds": [gpu_type],
                "gpuCount": 1,
                "allowedCudaVersions": ["13.0"],
                "dataCenterIds": ["EU-RO-1"],
                "networkVolumeId": volume_id,
                "volumeMountPath": "/workspace",
            }, s=s)
            pod = h3_comfyui_pod_public(result)
            created = True
        return {"pod": pod, "created": created, "prepared": prepared}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"ComfyUI Pod launch failed: {exc}") from exc

@app.post("/api/h3/comfyui/stop")
def h3_comfyui_stop(data: dict[str, Any]):
    if data.get("confirm") is not True:
        raise HTTPException(400, "Confirm before stopping the ComfyUI Pod")
    pod_id = str(data.get("pod_id") or "").strip()
    if not pod_id:
        raise HTTPException(400, "ComfyUI Pod ID is required")
    s = settings()
    try:
        allowed = {pod["id"] for pod in list_h3_comfyui_pods(s)}
        if pod_id not in allowed:
            raise ValueError("That Pod is not a ComfyUI Pod attached to the configured H3 volume")
        result = runpod_pod_request("POST", f"{pod_id}/stop", s=s)
        pod = h3_comfyui_pod_public(result) if isinstance(result, dict) and result.get("id") else {"id": pod_id, "status": "EXITED", "ready": False, "url": f"https://{pod_id}-8188.proxy.runpod.net"}
        return {"pod": pod}
    except Exception as exc:
        raise HTTPException(400, f"Could not stop ComfyUI Pod: {exc}") from exc

@app.post("/api/upload")
async def upload(file: UploadFile=File(...), kind: str=Form("media")):
    safe_kind = str(kind or "media").strip().lower()
    if safe_kind not in {"image", "video", "audio", "media", "mask"}:
        raise HTTPException(400, "Unsupported upload type")
    folder = UPLOAD_DIR / safe_kind
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{clean_name(file.filename or safe_kind)}"
    try:
        size = await run_in_threadpool(copy_stream_limited, file.file, out, GENERIC_UPLOAD_MAX_BYTES, "Media upload")
        if size <= 0:
            raise ValueError("The uploaded file is empty")
        return {"path": str(out), "name": out.name, "url": safe_file_url(out), "size": size}
    except Exception:
        out.unlink(missing_ok=True)
        raise
@app.get("/api/h3/loras")
def h3_loras_list():
    try:
        s3 = require_h3_s3()
        return {"items": list_h3_loras(s3), "prefix": H3_LORA_PREFIX, "bucket": s3.bucket, "volume_root": s3.root}
    except Exception as e:
        raise HTTPException(400, str(e)) from e
@app.get("/api/h3/models")
def h3_models_list():
    try:
        s3 = require_h3_s3()
        diffusion = list_h3_model_names(s3, H3_MODEL_PREFIXES["diffusion_models"])
        encoders = list_h3_model_names(s3, H3_MODEL_PREFIXES["text_encoders"])
        vaes = list_h3_model_names(s3, H3_MODEL_PREFIXES["vaes"])
        loras = [item["name"] for item in list_h3_loras(s3)]
        return {
            "bucket": s3.bucket,
            "volume_root": s3.root,
            "categories": {
                "fl2va": h3_model_role_options(diffusion, "fl2va"),
                "ref2va": h3_model_role_options(diffusion, "ref2va"),
                "diffusion_models": diffusion,
                "text_encoders": encoders,
                "video_vaes": [name for name in vaes if "video" in name.lower()] or vaes,
                "audio_vaes": [name for name in vaes if "audio" in name.lower()] or vaes,
                "vaes": vaes,
                "turbo_loras": [name for name in loras if "turbo" in name.lower()] or loras,
            },
        }
    except Exception as e:
        raise HTTPException(400, f"H3 model library unavailable: {e}") from e
@app.get("/api/h3/model-manager/files")
def h3_model_manager_files(category: str = "diffusion_models"):
    try:
        destination = h3_model_destination(category)
        s3 = require_h3_s3()
        prefix = destination.rstrip("/") + "/"
        items = []
        token = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": s3.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            page = s3.client.list_objects_v2(**kwargs)
            for obj in page.get("Contents") or []:
                key = str(obj.get("Key") or "")
                if Path(key).suffix.lower() not in H3_MODEL_EXTENSIONS:
                    continue
                modified = obj.get("LastModified")
                items.append({
                    "name": key[len(prefix):] if key.startswith(prefix) else Path(key).name,
                    "key": key,
                    "size_mb": round(int(obj.get("Size") or 0) / (1024 * 1024), 2),
                    "modified": modified.isoformat() if hasattr(modified, "isoformat") else str(modified or ""),
                    "volume_path": f"{s3.root}/{key}",
                    "s3_uri": f"s3://{s3.bucket}/{key}",
                })
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
            if not token:
                break
        return {"category": category, "destination": destination, "bucket": s3.bucket, "items": sorted(items, key=lambda item: item["name"].lower())}
    except Exception as e:
        raise HTTPException(400, f"H3 model inventory unavailable: {e}") from e
@app.post("/api/h3/model-manager/import")
def h3_model_manager_import(data: dict[str, Any]):
    try:
        validate_public_http_url(str(data.get("url") or ""))
        h3_model_destination(str(data.get("category") or ""))
        if str(data.get("name") or "").strip():
            h3_remote_model_name(str(data.get("name") or ""))
        job_id = uuid.uuid4().hex
        with MODEL_DOWNLOAD_LOCK:
            MODEL_DOWNLOAD_JOBS[job_id] = {"id": job_id, "status": "QUEUED", "phase": "Queued", "percent": 0, "created_at": time.time(), "updated_at": time.time()}
            if len(MODEL_DOWNLOAD_JOBS) > 50:
                oldest = sorted(MODEL_DOWNLOAD_JOBS.values(), key=lambda item: item.get("created_at", 0))[:-50]
                for item in oldest:
                    MODEL_DOWNLOAD_JOBS.pop(str(item.get("id") or ""), None)
        threading.Thread(target=run_model_download_job, args=(job_id, dict(data)), daemon=True, name=f"h3-model-{job_id[:8]}").start()
        return {"job_id": job_id, "status": "QUEUED"}
    except Exception as e:
        raise HTTPException(400, str(e)) from e
@app.get("/api/h3/model-manager/import/{job_id}")
def h3_model_manager_import_status(job_id: str):
    with MODEL_DOWNLOAD_LOCK:
        job = dict(MODEL_DOWNLOAD_JOBS.get(job_id) or {})
    if not job:
        raise HTTPException(404, "Model import job not found")
    return job
@app.post("/api/h3/loras/upload")
async def h3_lora_upload(file: UploadFile=File(...), overwrite: bool=Form(False)):
    temp_path: Optional[Path] = None
    try:
        s3 = require_h3_s3()
        name = h3_lora_name(file.filename or "")
        key = h3_lora_key(name)
        if not overwrite and await run_in_threadpool(h3_lora_exists, s3, key):
            raise HTTPException(409, f"{name} already exists in the H3 LoRA library")
        temp_path = TEMP_DIR / f"h3_lora_{uuid.uuid4().hex}_{name}"
        size = await run_in_threadpool(copy_stream_limited, file.file, temp_path, H3_LORA_MAX_BYTES, "H3 LoRA")
        if size <= 0:
            raise ValueError("The selected LoRA file is empty")
        await run_in_threadpool(s3.upload_to_key, temp_path, key)
        return {"item": h3_lora_result(s3, name, size), "overwritten": bool(overwrite)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    finally:
        if temp_path:
            try: temp_path.unlink(missing_ok=True)
            except Exception: pass
@app.post("/api/h3/loras/import")
def h3_lora_import(data: dict[str, Any]):
    temp_path: Optional[Path] = None
    try:
        raw_url = str(data.get("url") or "").strip()
        raw_url = validate_public_http_url(raw_url)
        overwrite = as_bool(data.get("overwrite"), False)
        s3 = require_h3_s3()
        with safe_http_get(raw_url, stream=True, timeout=(20, 3600)) as response:
            response.raise_for_status()
            name = remote_lora_name(response, str(data.get("name") or ""))
            key = h3_lora_key(name)
            if not overwrite and h3_lora_exists(s3, key):
                raise HTTPException(409, f"{name} already exists in the H3 LoRA library")
            announced_size = int(response.headers.get("content-length") or 0)
            if announced_size > H3_LORA_MAX_BYTES:
                raise ValueError("H3 LoRA exceeds the 16 GB import limit")
            temp_path = TEMP_DIR / f"h3_lora_{uuid.uuid4().hex}_{name}"
            size = 0
            with temp_path.open("wb") as out:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > H3_LORA_MAX_BYTES:
                        raise ValueError("H3 LoRA exceeds the 16 GB import limit")
                    out.write(chunk)
        if size <= 0:
            raise ValueError("The remote LoRA download was empty")
        s3.upload_to_key(temp_path, key)
        return {"item": h3_lora_result(s3, name, size), "overwritten": bool(overwrite)}
    except HTTPException:
        raise
    except requests.RequestException as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        detail = f"Remote LoRA download failed{f' with HTTP {status_code}' if status_code else ''}"
        raise HTTPException(400, detail) from e
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    finally:
        if temp_path:
            try: temp_path.unlink(missing_ok=True)
            except Exception: pass
@app.get("/api/file")
def file(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file(): raise HTTPException(404,"file not found")
    if p.suffix.lower() not in MEDIA_EXT:
        raise HTTPException(403, "Only supported media files can be served")
    return FileResponse(str(p), media_type=media_type_for_path(p))
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
MEDIA_EXT = IMAGE_EXT | VIDEO_EXT | AUDIO_EXT


def media_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXT: return "image"
    if ext in VIDEO_EXT: return "video"
    if ext in AUDIO_EXT: return "audio"
    return "file"

def media_kind_from_name(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in IMAGE_EXT: return "image"
    if ext in VIDEO_EXT: return "video"
    if ext in AUDIO_EXT: return "audio"
    return "file"


def thumb_cache_path(path: Path) -> Path:
    try:
        st = path.stat()
        raw = f"{path}|{st.st_mtime}|{st.st_size}".encode("utf-8", "ignore")
    except Exception:
        raw = str(path).encode("utf-8", "ignore")
    return THUMB_DIR / f"{hashlib.sha1(raw).hexdigest()}.jpg"


def make_thumbnail(path: Path) -> Optional[Path]:
    kind = media_kind(path)
    if kind not in {"image", "video"}: return None
    out = thumb_cache_path(path)
    if out.exists() and out.stat().st_size > 0: return out
    try:
        from PIL import Image
        if kind == "image":
            img = Image.open(path).convert("RGB")
        else:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            temp = TEMP_DIR / f"thumb_{uuid.uuid4().hex}.jpg"
            subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", "0.2", "-i", str(path), "-frames:v", "1", str(temp)], check=True)
            img = Image.open(temp).convert("RGB")
        img.thumbnail((240, 160))
        img.save(out, quality=84)
        return out
    except Exception:
        return None


def file_info(path: Path) -> dict[str, Any]:
    st = path.stat()
    info = {"name": path.name, "path": str(path), "is_dir": path.is_dir(), "ext": path.suffix.lower(), "kind": "folder" if path.is_dir() else media_kind(path), "size": st.st_size if path.is_file() else None, "size_mb": round(st.st_size/(1024*1024), 3) if path.is_file() else None, "modified": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)), "created": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_ctime))}
    if path.is_file() and media_kind(path) == "image":
        try:
            from PIL import Image
            img = Image.open(path)
            info["width"] = img.width; info["height"] = img.height
        except Exception: pass
    if path.is_file() and media_kind(path) == "video":
        try: info.update(video_probe(path))
        except Exception: pass
    return info

def sort_browser_items(items: list[Any], sort_by: str = "name", sort_dir: str = "asc", is_dir=None) -> list[Any]:
    reverse = str(sort_dir).lower() == "desc"
    def item_get(x, key, default=None):
        if isinstance(x, Path):
            if key == "is_dir":
                return x.is_dir()
            if key == "name":
                return x.name
            if key == "kind":
                return "folder" if x.is_dir() else media_kind(x)
            if key == "modified":
                try: return x.stat().st_mtime
                except Exception: return 0
            if key == "created":
                try: return x.stat().st_ctime
                except Exception: return 0
            if key == "size_mb":
                try: return x.stat().st_size / (1024 * 1024) if x.is_file() else 0
                except Exception: return 0
        return x.get(key, default) if isinstance(x, dict) else getattr(x, key, default)
    def key_func(x):
        name = str(item_get(x, "name", "")).lower()
        if sort_by == "modified":
            return (item_get(x, "modified", "") or "", name)
        if sort_by == "created":
            return (item_get(x, "created", "") or "", name)
        if sort_by == "size":
            return (float(item_get(x, "size_mb", 0) or 0), name)
        if sort_by == "type":
            return (str(item_get(x, "kind", "") or item_get(x, "ext", "")).lower(), name)
        return (name,)
    dirs = [x for x in items if (item_get(x, "is_dir", False) if is_dir is None else is_dir(x))]
    files = [x for x in items if not (item_get(x, "is_dir", False) if is_dir is None else is_dir(x))]
    return sorted(dirs, key=key_func, reverse=reverse) + sorted(files, key=key_func, reverse=reverse)


@app.get("/api/browse")
def browse(path: str="", media_type: str="All", search: str="", recursive: bool=False, sort_by: str="name", sort_dir: str="asc"):
    root=Path(path) if path else HOME
    if root.is_file(): root=root.parent
    if not root.exists(): raise HTTPException(404,"path not found")
    if not root.is_dir(): raise HTTPException(400,"path is not a directory")
    type_map = {"All": None, "Images": IMAGE_EXT, "Videos": VIDEO_EXT, "Audio": AUDIO_EXT}
    allowed = type_map.get(media_type)
    search_l = (search or "").strip().lower()
    items=[]
    try:
        if recursive:
            candidates = [p for p in root.rglob("*") if p.is_file() and (allowed is None or p.suffix.lower() in allowed)]
        else:
            candidates = [p for p in root.iterdir() if p.is_dir()] + [p for p in root.iterdir() if p.is_file() and (allowed is None or p.suffix.lower() in allowed)]
    except Exception:
        candidates = []
    if search_l: candidates = [p for p in candidates if search_l in p.name.lower()]
    candidates = sort_browser_items(candidates, sort_by, sort_dir, is_dir=lambda p: p.is_dir())
    for p in candidates[:750]:
        try:
            info = file_info(p)
            if p.is_file() and info["kind"] in {"image", "video"}: info["thumb_url"] = f"/api/thumb?path={urllib.parse.quote(str(p))}"
            if p.is_file(): info["file_url"] = f"/api/file?path={urllib.parse.quote(str(p))}"
            items.append(info)
        except Exception: continue
    return {"path":str(root),"parent":str(root.parent),"items":items}

@app.get("/api/s3/browse")
def s3_browse(prefix: str = "", media_type: str = "All", search: str = "", recursive: bool = False, sort_by: str = "name", sort_dir: str = "asc", storage: str = "main"):
    s = settings()
    try:
        storage_name, h = require_named_s3(storage, s)
    except Exception as e:
        raise HTTPException(400, f"S3 settings are not complete: {e}")
    clean_prefix = (prefix or "").lstrip("/")
    if clean_prefix and not clean_prefix.endswith("/"):
        clean_prefix += "/"
    type_map = {"All": None, "Images": IMAGE_EXT, "Videos": VIDEO_EXT, "Audio": AUDIO_EXT}
    allowed = type_map.get(media_type)
    search_l = (search or "").strip().lower()
    delimiter = "" if recursive else "/"
    kwargs = {"Bucket": h.bucket, "Prefix": clean_prefix, "MaxKeys": 500}
    if delimiter:
        kwargs["Delimiter"] = delimiter
    try:
        resp = h.call("list_objects_v2", **kwargs)
    except Exception as e:
        raise HTTPException(400, f"S3 browse failed: {e}")
    items = []
    folder_modified: dict[str, str] = {}
    folder_modified_truncated = False
    if not recursive and sort_by == "modified" and resp.get("CommonPrefixes"):
        # CommonPrefixes do not carry timestamps.  Aggregate the newest object
        # timestamp beneath each visible folder only when the user requests a
        # modified-date sort, keeping ordinary browsing fast.
        token: Optional[str] = None
        scanned = 0
        visible_prefixes = {str(cp.get("Prefix") or "") for cp in resp.get("CommonPrefixes", [])}
        while scanned < 5000:
            nested_kwargs: dict[str, Any] = {"Bucket": h.bucket, "Prefix": clean_prefix, "MaxKeys": min(1000, 5000 - scanned)}
            if token:
                nested_kwargs["ContinuationToken"] = token
            nested = h.call("list_objects_v2", **nested_kwargs)
            objects = nested.get("Contents") or []
            scanned += len(objects)
            for obj in objects:
                key = str(obj.get("Key") or "")
                stamp = obj.get("LastModified")
                if not key or not stamp:
                    continue
                for folder_prefix in visible_prefixes:
                    if key.startswith(folder_prefix):
                        value = stamp.isoformat()
                        if value > folder_modified.get(folder_prefix, ""):
                            folder_modified[folder_prefix] = value
                        break
            if not nested.get("IsTruncated"):
                break
            token = nested.get("NextContinuationToken")
            if not token:
                break
        folder_modified_truncated = bool(token and scanned >= 5000)
    for cp in resp.get("CommonPrefixes", []):
        p = cp.get("Prefix", "")
        name = p.rstrip("/").split("/")[-1] or p
        if search_l and search_l not in name.lower(): continue
        items.append({"name": name, "path": p, "key": p, "is_dir": True, "kind": "folder", "ext": "", "size_mb": None, "modified": folder_modified.get(p, ""), "created": "", "source": "s3", "storage": storage_name})
    for obj in resp.get("Contents", []):
        key = obj.get("Key", "")
        if not key or key == clean_prefix: continue
        name = key.rstrip("/").split("/")[-1]
        kind = media_kind_from_name(name)
        ext = Path(name).suffix.lower()
        if allowed is not None and ext not in allowed: continue
        haystack = key.lower()
        if search_l and search_l not in haystack: continue
        items.append({"name": name, "path": f"s3://{h.bucket}/{key}", "key": key, "is_dir": False, "kind": kind, "ext": ext, "size_mb": round((obj.get("Size") or 0)/(1024*1024), 3), "modified": obj.get("LastModified").isoformat() if obj.get("LastModified") else "", "created": "", "source": "s3", "storage": storage_name})
    items = sort_browser_items(items, sort_by, sort_dir)
    parent = ""
    if clean_prefix:
        parts = clean_prefix.rstrip("/").split("/")
        parent = "/".join(parts[:-1])
        if parent: parent += "/"
    return {"source": "s3", "storage": storage_name, "bucket": h.bucket, "path": clean_prefix, "parent": parent, "is_truncated": bool(resp.get("IsTruncated")), "folder_modified_truncated": folder_modified_truncated, "items": items[:500]}

@app.get("/api/s3/details")
def s3_details(key: str, storage: str = "main"):
    s = settings()
    try:
        storage_name, h = require_named_s3(storage, s)
        meta = s3_object_metadata(h, key)
    except Exception as e:
        raise HTTPException(404, f"S3 object not found: {e}")
    kind = media_kind_from_name(key)
    url = f"/api/s3/preview?key={urllib.parse.quote(key)}&storage={storage_name}" if kind != "file" else ""
    return {"source": "s3", "storage": storage_name, "bucket": h.bucket, "key": key, "path": f"s3://{h.bucket}/{key}", "name": Path(key).name, "kind": kind, "size_mb": round((meta.get("ContentLength") or 0)/(1024*1024), 3), "modified": meta.get("LastModified").isoformat() if meta.get("LastModified") else "", "file_url": url}

@app.get("/api/s3/preview")
def s3_preview(key: str, storage: str = "main"):
    key = str(key or "").strip()
    if not key:
        raise HTTPException(400, "S3 key required")
    s = settings()
    try:
        storage_name, h = require_named_s3(storage, s)
    except Exception as e:
        raise HTTPException(400, f"S3 settings are not complete: {e}")
    try:
        meta = s3_object_metadata(h, key)
    except Exception as e:
        raise HTTPException(404, f"S3 object not found: {e}")
    suffix = Path(key).suffix or ".bin"
    safe = hashlib.sha1(f"{storage_name}/{h.bucket}/{key}/{meta.get('ETag','')}/{meta.get('ContentLength','')}".encode("utf-8", "ignore")).hexdigest()
    out_dir = TEMP_DIR / "s3_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{safe}{suffix}"
    expected = int(meta.get("ContentLength") or 0)
    if not out.exists() or (expected and out.stat().st_size != expected):
        h.download_key(key, out)
    return FileResponse(str(out), media_type=media_type_for_path(out), filename=Path(key).name)

@app.post("/api/s3/download")
def s3_download(data: dict[str, Any]):
    key = str(data.get("key") or "").strip()
    if not key:
        raise HTTPException(400, "S3 key required")
    s = settings()
    try:
        storage_name, h = require_named_s3(str(data.get("storage") or "main"), s)
    except Exception as e:
        raise HTTPException(400, f"S3 settings are not complete: {e}")
    out_dir = Path(s.get("output_dir") or DEFAULT_OUTPUT_DIR) / "s3" / storage_name
    out = next_output_path(out_dir, "s3", Path(key).suffix or ".bin")
    try:
        return {"path": h.download_key(key, out), "url": safe_file_url(out), "kind": media_kind(out)}
    except Exception as e:
        raise HTTPException(400, f"S3 download failed: {e}")

@app.post("/api/s3/download-many")
def s3_download_many(data: dict[str, Any]):
    keys_raw = data.get("keys") or []
    if isinstance(keys_raw, str):
        keys_raw = [keys_raw]
    keys: list[str] = []
    seen: set[str] = set()
    for key in keys_raw:
        key = str(key or "").strip()
        if key and key not in seen and not key.endswith("/"):
            keys.append(key)
            seen.add(key)
    if not keys:
        raise HTTPException(400, "At least one S3 key is required")
    s = settings()
    try:
        storage_name, h = require_named_s3(str(data.get("storage") or "main"), s)
    except Exception as e:
        raise HTTPException(400, f"S3 settings are not complete: {e}")
    out_dir = Path(s.get("output_dir") or DEFAULT_OUTPUT_DIR) / "s3" / storage_name
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    total_bytes = 0
    for key in keys:
        try:
            out = next_output_path(out_dir, "s3", Path(key).suffix or ".bin")
            saved = Path(h.download_key(key, out))
            total_bytes += saved.stat().st_size if saved.exists() else 0
            items.append({"path": str(saved), "url": safe_file_url(saved), "kind": media_kind(saved), "key": key})
        except Exception as e:
            errors.append(f"{key}: {e}")
    if not items and errors:
        raise HTTPException(400, "S3 downloads failed: " + "; ".join(errors[:5]))
    return {
        "path": str(out_dir),
        "files": len(items),
        "mb": round(total_bytes / (1024 * 1024), 3),
        "items": items,
        "errors": errors[:20],
    }

@app.post("/api/s3/download-prefix")
def s3_download_prefix(data: dict[str, Any]):
    prefix = str(data.get("prefix") or "").lstrip("/")
    s = settings()
    try:
        storage_name, h = require_named_s3(str(data.get("storage") or "main"), s)
    except Exception as e:
        raise HTTPException(400, f"S3 settings are not complete: {e}")
    out_root = Path(s.get("output_dir") or DEFAULT_OUTPUT_DIR) / "s3" / storage_name / clean_name(prefix.strip("/").replace("/", "_") or h.bucket)
    out_root.mkdir(parents=True, exist_ok=True)
    token = None
    count = 0
    total_bytes = 0
    saved: list[str] = []
    try:
        while True:
            kwargs = {"Bucket": h.bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kwargs["ContinuationToken"] = token
            resp = h.call("list_objects_v2", **kwargs)
            for obj in resp.get("Contents", []):
                key = obj.get("Key", "")
                if not key or key.endswith("/"):
                    continue
                rel = Path(*key[len(prefix):].lstrip("/").split("/")) if prefix and key.startswith(prefix) else Path(*key.split("/"))
                out = out_root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                h.download_key(key, out)
                count += 1
                total_bytes += int(obj.get("Size") or 0)
                if len(saved) < 20:
                    saved.append(str(out))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return {"path": str(out_root), "files": count, "bytes": total_bytes, "mb": round(total_bytes/(1024*1024), 3), "sample": saved}
    except Exception as e:
        raise HTTPException(400, f"S3 folder download failed: {e}")


@app.get("/api/thumb")
def thumb(path: str):
    p = Path(path)
    if not p.exists() or not p.is_file(): raise HTTPException(404, "file not found")
    out = make_thumbnail(p)
    if not out or not out.exists(): raise HTTPException(404, "thumbnail unavailable")
    return FileResponse(str(out), media_type="image/jpeg")


@app.get("/api/media/details")
def media_details(path: str):
    p = Path(path)
    if not p.exists(): raise HTTPException(404, "file not found")
    info = file_info(p)
    if p.is_file():
        info["file_url"] = f"/api/file?path={urllib.parse.quote(str(p))}"
        if info["kind"] in {"image", "video"}: info["thumb_url"] = f"/api/thumb?path={urllib.parse.quote(str(p))}"
    return info

@app.get("/api/media/probe")
def media_probe(path: str):
    raw = str(path or "").strip()
    if not raw:
        raise HTTPException(400, "path required")
    try:
        local = Path(cached_s3_object_path(raw, media_ext_from_string(raw))) if is_s3_uri(raw) else Path(raw)
        if not local.exists() or not local.is_file():
            raise HTTPException(404, "file not found")
        info = file_info(local)
        info["source_path"] = raw
        return info
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"probe failed: {e}")

@app.get("/api/path/siblings")
def path_siblings(path: str="", target_id: str="", kind: str="media"):
    p = Path(path) if path else Path(settings().get("output_dir") or DEFAULT_OUTPUT_DIR)
    root = p.parent if p.suffix else p
    if not root.exists() or not root.is_dir(): return {"folder": str(root), "items": []}
    target = (target_id or "").lower()
    if kind == "image" or "image" in target or "mask" in target:
        allowed = IMAGE_EXT
    elif kind == "video" or "video" in target:
        allowed = VIDEO_EXT
    elif kind == "audio" or "audio" in target:
        allowed = AUDIO_EXT
    else:
        allowed = IMAGE_EXT | VIDEO_EXT
    items = []
    for item in root.iterdir():
        try:
            if not item.is_file() or item.suffix.lower() not in allowed: continue
            info = file_info(item)
            info["file_url"] = safe_file_url(item)
            items.append(info)
        except Exception:
            continue
    items.sort(key=lambda it: it.get("modified", ""), reverse=True)
    return {"folder": str(root), "items": items[:40]}


@app.get("/api/favorites")
def get_favorites():
    favs = load_json(FAVORITES_PATH, [])
    favs = [str(Path(p)) for p in favs if Path(p).exists()]
    return {"favorites": favs}


@app.post("/api/favorites")
async def add_favorite(data: dict[str, Any]):
    folder = data.get("path", "")
    p = Path(folder)
    if not p.exists(): raise HTTPException(404, "folder not found")
    if p.is_file(): p = p.parent
    favs = load_json(FAVORITES_PATH, [])
    favs = [x for x in favs if Path(x).exists()]
    if str(p) not in favs: favs.append(str(p))
    FAVORITES_PATH.write_text(json.dumps(favs, indent=2), encoding="utf-8")
    return {"favorites": favs}


@app.delete("/api/favorites")
async def clear_favorites():
    FAVORITES_PATH.write_text("[]", encoding="utf-8")
    return {"favorites": []}


# ------------------------- SAM Studio / local mask tools -------------------------

def source_kind(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in IMAGE_EXT: return "image"
    if ext in VIDEO_EXT: return "video"
    return "file"


def safe_file_url(path: str | Path) -> str:
    return f"/api/file?path={urllib.parse.quote(str(path))}"


def s3_key_from_uri(ref: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(ref)
    if parsed.scheme.lower() != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {ref}")
    return parsed.netloc, parsed.path.lstrip("/")


def cached_s3_object_path(ref: str, suffix: str | None = None) -> str:
    s = settings()
    h = S3Helper(s)
    bucket, key = s3_key_from_uri(ref)
    try:
        meta = h.call("head_object", Bucket=bucket, Key=key)
    except Exception as e:
        raise ValueError(f"S3 source not found: {e}")
    ext = suffix or Path(key).suffix or ".bin"
    safe = hashlib.sha1(f"{bucket}/{key}/{meta.get('ETag','')}/{meta.get('ContentLength','')}".encode("utf-8", "ignore")).hexdigest()
    out_dir = TEMP_DIR / "s3_preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{safe}{ext}"
    expected = int(meta.get("ContentLength") or 0)
    if not out.exists() or (expected and out.stat().st_size != expected):
        h.call("download_file", bucket, key, str(out))
    return str(out)


def resolve_sam_source(source_path: str) -> str:
    source_path = (source_path or "").strip()
    if source_path.lower().startswith("s3://"):
        return cached_s3_object_path(source_path, Path(urllib.parse.urlparse(source_path).path).suffix)
    return source_path

def prepare_sam_source(source_path: str, data: dict[str, Any]) -> str:
    local = resolve_sam_source(source_path)
    if source_kind(local) != "video":
        return local
    preprocess_fps = data.get("preprocess_fps")
    if data.get("source_workflow") == "samimate" and not preprocess_fps:
        preprocess_fps = data.get("fps")
    return process_video(
        local,
        data.get("video_start"),
        data.get("video_end"),
        None,
        data.get("video_crop"),
        None,
        preprocess_fps,
    )


def sam_record(item: dict[str, Any]) -> None:
    hist = load_json(SAM_HISTORY_PATH, [])
    hist.insert(0, item)
    SAM_HISTORY_PATH.write_text(json.dumps(hist[:80], indent=2), encoding="utf-8")


def extract_sam_frame(source_path: str, frame_time: Any = 0) -> dict[str, Any]:
    source_path = resolve_sam_source(source_path)
    src = Path(source_path)
    if not src.exists(): raise ValueError(f"Source not found: {source_path}")
    kind = source_kind(str(src))
    out = SAM_DIR / f"frame_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    if kind == "image":
        from PIL import Image
        img = Image.open(src).convert("RGB")
        img.save(out)
        return {"frame_path": str(out), "frame_url": safe_file_url(out), "kind": kind, "width": img.width, "height": img.height, "duration_seconds": 0}
    if kind == "video":
        t = seconds(frame_time) or 0.0
        subprocess.run([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-ss", str(max(0,t)), "-i", str(src), "-frames:v", "1", str(out)], check=True)
        if not out.exists() or out.stat().st_size <= 0:
            subprocess.run([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-frames:v", "1", str(out)], check=True)
        from PIL import Image
        img = Image.open(out).convert("RGB")
        return {"frame_path": str(out), "frame_url": safe_file_url(out), "kind": kind, "width": img.width, "height": img.height, "duration_seconds": duration(src)}
    raise ValueError("SAM source must be an image or video")


def clamp_point(p: dict[str, Any], w: int, h: int) -> tuple[float, float]:
    x = max(0.0, min(float(p.get("x", 0)), float(w - 1)))
    y = max(0.0, min(float(p.get("y", 0)), float(h - 1)))
    return x, y


def make_local_test_mask(frame_path: str, data: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image, ImageDraw, ImageFilter, ImageOps
    img = Image.open(frame_path).convert("RGB")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    text_prompt = str(data.get("text_prompt") or "").strip().lower()
    points = data.get("points") or {"positive": [], "negative": []}
    pos = points.get("positive") or []
    neg = points.get("negative") or []
    box = data.get("box") or {}
    radius = max(14, int(min(w, h) * float(data.get("point_radius", 0.045) or 0.045)))
    if box and float(box.get("w", 0) or 0) > 1 and float(box.get("h", 0) or 0) > 1:
        x = max(0, int(float(box.get("x", 0)))); y = max(0, int(float(box.get("y", 0))))
        bw = max(1, int(float(box.get("w", 0)))); bh = max(1, int(float(box.get("h", 0))))
        draw.rectangle([x, y, min(w, x + bw), min(h, y + bh)], fill=255)
    for p in pos:
        x, y = clamp_point(p, w, h)
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=255)
    if not box and not pos and text_prompt:
        rx = int(w * 0.30); ry = int(h * 0.38); cx = int(w * 0.50); cy = int(h * 0.52)
        if any(word in text_prompt for word in ["face", "head"]):
            rx = int(w * 0.18); ry = int(h * 0.18); cy = int(h * 0.30)
        elif any(word in text_prompt for word in ["person", "body", "human", "subject"]):
            rx = int(w * 0.24); ry = int(h * 0.42); cy = int(h * 0.50)
        draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=255)
    if not mask.getbbox():
        draw.ellipse([int(w*.25), int(h*.18), int(w*.75), int(h*.86)], fill=255)
    for p in neg:
        x, y = clamp_point(p, w, h)
        draw.ellipse([x-radius*1.25, y-radius*1.25, x+radius*1.25, y+radius*1.25], fill=0)
    feather = float(data.get("feather", 2) or 0)
    if feather > 0: mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    if str(data.get("invert", "false")).lower() in {"1", "true", "yes", "on"}: mask = ImageOps.invert(mask)
    stamp = f"sam_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    mask_path = SAM_DIR / f"{stamp}_mask.png"; overlay_path = SAM_DIR / f"{stamp}_overlay.png"; cutout_path = SAM_DIR / f"{stamp}_cutout.png"; masked_image_path = SAM_DIR / f"{stamp}_masked.png"
    mask.save(mask_path)
    red = Image.new("RGBA", img.size, (255, 40, 40, 120))
    overlay = Image.composite(red, img.convert("RGBA"), mask)
    overlay = Image.blend(img.convert("RGBA"), overlay, 0.55); overlay.save(overlay_path)
    cutout = img.convert("RGBA"); cutout.putalpha(mask); cutout.save(cutout_path)
    fill = str(data.get("mask_fill") or "black").lower()
    bg = Image.new("RGB", img.size, (0, 0, 0) if fill != "white" else (255, 255, 255))
    Image.composite(img, bg, mask).save(masked_image_path)
    return {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "cutout_path": str(cutout_path),
        "masked_image_path": str(masked_image_path),
        "backend_used": "local_shape_test",
        "is_real_sam3": False,
        "warning": "Local Shape Test is not SAM3. It only draws simple masks from points, boxes, or fallback ellipses.",
    }


def run_external_sam3(frame_path: str, source_path: str, data: dict[str, Any], command_template: str) -> dict[str, Any]:
    if not command_template.strip(): raise ValueError("External SAM3 command is blank. Set a command or use Local Test backend.")
    if "sam3_runner_template.py" in command_template:
        raise ValueError("The bundled sam3_runner_template.py is only an ellipse demo, not real SAM3. Point External SAM3 runner to a real SAM3 script.")
    job_id = f"sam3_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    input_json = TEMP_DIR / f"{job_id}_input.json"; output_json = TEMP_DIR / f"{job_id}_output.json"
    out_dir = Path(data.get("output_dir") or SAM_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(data); payload.update({"frame_path": frame_path, "source_path": source_path, "output_dir": str(out_dir)})
    input_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    cmd = command_template.format(input_json=str(input_json), output_json=str(output_json), frame_path=frame_path, source_path=source_path, output_dir=str(out_dir))
    args = shlex.split(cmd, posix=True)
    if not args:
        raise ValueError("External SAM3 command resolved to an empty command.")
    timeout = max(30, min(7200, int(data.get("sam3_timeout_seconds") or 1800)))
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise ValueError(f"External SAM3 runner timed out after {timeout}s") from e
    if proc.returncode != 0:
        detail = strip_process_noise(proc.stderr or proc.stdout or "")
        raise ValueError(f"External SAM3 runner failed. {detail}")
    if not output_json.exists(): raise ValueError(f"External SAM3 runner did not create output JSON: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8")); result["backend_used"] = "external_sam3"; result["is_real_sam3"] = True; return result


def run_local_sam3(frame_path: str, source_path: str, data: dict[str, Any], python_exe: str) -> dict[str, Any]:
    runner = Path(__file__).parent / "sam3_local_runner.py"
    if not runner.exists():
        raise ValueError(f"SAM3 local runner missing: {runner}")
    py = (python_exe or sys.executable).strip()
    job_id = f"sam3_local_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    input_json = TEMP_DIR / f"{job_id}_input.json"
    output_json = TEMP_DIR / f"{job_id}_output.json"
    out_dir = Path(data.get("output_dir") or SAM_DIR).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload.update({"frame_path": frame_path, "source_path": source_path, "source_kind": source_kind(source_path), "output_dir": str(out_dir)})
    input_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    env = os.environ.copy()
    hf_token = (data.get("hf_token") or "").strip()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    hf_cache_dir = (data.get("hf_cache_dir") or settings().get("hf_cache_dir") or "").strip()
    if hf_cache_dir:
        env["HF_HOME"] = hf_cache_dir
        env["HUGGINGFACE_HUB_CACHE"] = str(Path(hf_cache_dir).expanduser() / "hub")
    frame_cap = int(data.get("video_frame_cap") or 0)
    source_is_video = source_kind(source_path) == "video"
    timeout = int(data.get("sam3_timeout_seconds") or 0)
    if timeout <= 0:
        timeout = 7200 if source_is_video and frame_cap <= 0 else max(900, min(7200, 600 + frame_cap * 12))
    try:
        proc = subprocess.run([py, str(runner), "--input", str(input_json), "--output", str(output_json)], capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        detail = ((e.stderr or "") if isinstance(e.stderr, str) else "") or ((e.stdout or "") if isinstance(e.stdout, str) else "")
        detail = strip_process_noise(detail)
        raise ValueError(
            f"Local SAM3 runner timed out after {timeout}s. "
            "Use a smaller SAM/Wan Frame Cap for a quick test, or set 0 only when you want the full video. "
            + detail
        )
    if proc.returncode != 0:
        detail = strip_process_noise(proc.stderr or proc.stdout or "")
        raise ValueError(f"Local SAM3 runner failed. {detail}")
    if not output_json.exists():
        raise ValueError(f"Local SAM3 runner did not create output JSON: {output_json}")
    result = json.loads(output_json.read_text(encoding="utf-8"))
    result["backend_used"] = result.get("backend_used") or "local_sam3"
    result["is_real_sam3"] = True
    return result


def apply_mask_to_video(source_path: str, mask_path: str, fill: str = "black", frame_cap: Any = None) -> str:
    import cv2, numpy as np
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None: raise ValueError(f"Could not read mask: {mask_path}")
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened(): raise ValueError(f"Could not open video: {source_path}")
    fps = sane_video_fps(cap.get(cv2.CAP_PROP_FPS) or 24); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = (mask_img.astype(np.float32) / 255.0)[..., None]
    bg = np.full((height, width, 3), 255 if str(fill).lower() == "white" else 0, dtype=np.float32)
    stamp = f"sam_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    temp_path = TEMP_DIR / f"{stamp}_masked_video_raw.avi"
    out_path = SAM_DIR / f"{stamp}_masked_video.mp4"
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened(): raise ValueError("Masked video writer could not start")
    limit = int(frame_cap or 0); count = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        writer.write((frame.astype(np.float32) * alpha + bg * (1.0 - alpha)).clip(0, 255).astype(np.uint8))
        count += 1
        if limit and count >= limit: break
    cap.release(); writer.release()
    if not temp_path.exists() or temp_path.stat().st_size <= 0: raise ValueError("Masked video creation failed")
    proc = subprocess.run([
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp_path), "-r", str(fps), "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)
    ], capture_output=True, text=True)
    try: temp_path.unlink(missing_ok=True)
    except Exception: pass
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ValueError(f"Masked video H.264 encode failed: {detail}")
    if not out_path.exists() or out_path.stat().st_size <= 0: raise ValueError("Masked video H.264 encode failed")
    return str(out_path)

def make_mask_video_from_image_mask(source_path: str, mask_path: str, frame_cap: Any = None) -> str:
    import cv2
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None: raise ValueError(f"Could not read mask: {mask_path}")
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened(): raise ValueError(f"Could not open video: {source_path}")
    fps = sane_video_fps(cap.get(cv2.CAP_PROP_FPS) or 24); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_LINEAR)
    mask_frame = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
    stamp = f"sam_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    temp_path = TEMP_DIR / f"{stamp}_mask_video_raw.avi"
    out_path = SAM_DIR / f"{stamp}_mask_video.mp4"
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height), True)
    if not writer.isOpened(): raise ValueError("Mask video writer could not start")
    limit = int(frame_cap or 0); count = 0
    while True:
        ok, _frame = cap.read()
        if not ok: break
        writer.write(mask_frame)
        count += 1
        if limit and count >= limit: break
    cap.release(); writer.release()
    if not temp_path.exists() or temp_path.stat().st_size <= 0: raise ValueError("Mask video creation failed")
    proc = subprocess.run([
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp_path), "-r", str(fps), "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)
    ], capture_output=True, text=True)
    try: temp_path.unlink(missing_ok=True)
    except Exception: pass
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ValueError(f"Mask video H.264 encode failed: {detail}")
    if not out_path.exists() or out_path.stat().st_size <= 0: raise ValueError("Mask video H.264 encode failed")
    return str(out_path)

def invert_mask_video(mask_video_path: str) -> str:
    src = Path(resolve_sam_source(mask_video_path))
    if not src.exists(): raise ValueError(f"Mask video not found: {mask_video_path}")
    out_path = SAM_DIR / f"sam_{int(time.time())}_{uuid.uuid4().hex[:8]}_inverted_mask_video.mp4"
    proc = subprocess.run([
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vf", "negate", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)
    ], capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ValueError(f"Inverted mask video encode failed: {detail}")
    if not out_path.exists() or out_path.stat().st_size <= 0: raise ValueError("Inverted mask video encode failed")
    return str(out_path)

def apply_video_mask_video(source_path: str, mask_video_path: str, out_path: str | Path, keep_white: bool = True) -> str:
    import cv2, numpy as np
    src_path = Path(resolve_sam_source(source_path))
    mask_path = Path(resolve_sam_source(mask_video_path))
    if not src_path.exists(): raise ValueError(f"Source video not found: {source_path}")
    if not mask_path.exists(): raise ValueError(f"Mask video not found: {mask_video_path}")
    cap = cv2.VideoCapture(str(src_path)); mk = cv2.VideoCapture(str(mask_path))
    if not cap.isOpened(): raise ValueError(f"Could not open source video: {source_path}")
    if not mk.isOpened(): raise ValueError(f"Could not open mask video: {mask_video_path}")
    fps = sane_video_fps(cap.get(cv2.CAP_PROP_FPS) or mk.get(cv2.CAP_PROP_FPS) or 24)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or mk.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or mk.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = TEMP_DIR / f"{out_path.stem}_{uuid.uuid4().hex[:8]}_raw.avi"
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height), True)
    if not writer.isOpened(): raise ValueError("Masked preview writer could not start")
    count = 0
    while True:
        ok_src, frame = cap.read(); ok_mk, mask_frame = mk.read()
        if not ok_src or not ok_mk: break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        if mask_frame.shape[1] != width or mask_frame.shape[0] != height:
            mask_frame = cv2.resize(mask_frame, (width, height), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        alpha = gray.astype(np.float32) / 255.0
        if not keep_white:
            alpha = 1.0 - alpha
        out = (frame.astype(np.float32) * alpha[..., None]).clip(0, 255).astype(np.uint8)
        writer.write(out)
        count += 1
    cap.release(); mk.release(); writer.release()
    if count <= 0 or not temp_path.exists() or temp_path.stat().st_size <= 0:
        raise ValueError("Masked preview produced no frames")
    proc = subprocess.run([
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp_path), "-r", str(fps), "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)
    ], capture_output=True, text=True)
    try: temp_path.unlink(missing_ok=True)
    except Exception: pass
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ValueError(f"Masked preview encode failed: {detail}")
    if not out_path.exists() or out_path.stat().st_size <= 0:
        raise ValueError("Masked preview encode failed")
    return str(out_path)

def composite_subject_video(foreground_path: str, background_path: str, mask_video_path: str, mask_selects_background: bool = False, output_dir: str | Path | None = None) -> str:
    import cv2, numpy as np
    fg_path = Path(resolve_sam_source(foreground_path))
    bg_path = Path(resolve_sam_source(background_path))
    mask_path = Path(resolve_sam_source(mask_video_path))
    if not fg_path.exists(): raise ValueError(f"Foreground video not found: {foreground_path}")
    if not bg_path.exists(): raise ValueError(f"Background video not found: {background_path}")
    if not mask_path.exists(): raise ValueError(f"Mask video not found: {mask_video_path}")
    fg = cv2.VideoCapture(str(fg_path)); bg = cv2.VideoCapture(str(bg_path)); mk = cv2.VideoCapture(str(mask_path))
    if not fg.isOpened(): raise ValueError(f"Could not open foreground video: {foreground_path}")
    if not bg.isOpened(): raise ValueError(f"Could not open background video: {background_path}")
    if not mk.isOpened(): raise ValueError(f"Could not open mask video: {mask_video_path}")
    fps = sane_video_fps(fg.get(cv2.CAP_PROP_FPS) or bg.get(cv2.CAP_PROP_FPS) or 24)
    width = int(fg.get(cv2.CAP_PROP_FRAME_WIDTH) or bg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(fg.get(cv2.CAP_PROP_FRAME_HEIGHT) or bg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stamp = f"samimate_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    temp_path = TEMP_DIR / f"{stamp}_raw.avi"
    out_dir = Path(output_dir).expanduser() if output_dir else samimate_root_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "overlayed_output.mp4" if output_dir else next_output_path(out_dir, "samimate", ".mp4")
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height), True)
    if not writer.isOpened(): raise ValueError("SAMimate composite writer could not start")
    count = 0
    while True:
        ok_fg, frame_fg = fg.read(); ok_bg, frame_bg = bg.read(); ok_mk, frame_mk = mk.read()
        if not ok_fg or not ok_bg or not ok_mk: break
        if frame_fg.shape[1] != width or frame_fg.shape[0] != height:
            frame_fg = cv2.resize(frame_fg, (width, height), interpolation=cv2.INTER_LINEAR)
        if frame_bg.shape[1] != width or frame_bg.shape[0] != height:
            frame_bg = cv2.resize(frame_bg, (width, height), interpolation=cv2.INTER_LINEAR)
        if frame_mk.shape[1] != width or frame_mk.shape[0] != height:
            frame_mk = cv2.resize(frame_mk, (width, height), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame_mk, cv2.COLOR_BGR2GRAY)
        alpha = (gray.astype(np.float32) / 255.0)[..., None]
        if mask_selects_background:
            out = (frame_bg.astype(np.float32) * alpha + frame_fg.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        else:
            out = (frame_fg.astype(np.float32) * alpha + frame_bg.astype(np.float32) * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
        writer.write(out); count += 1
    fg.release(); bg.release(); mk.release(); writer.release()
    if count <= 0 or not temp_path.exists() or temp_path.stat().st_size <= 0:
        raise ValueError("SAMimate composite produced no frames")
    proc = subprocess.run([
        ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp_path), "-r", str(fps), "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)
    ], capture_output=True, text=True)
    try: temp_path.unlink(missing_ok=True)
    except Exception: pass
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ValueError(f"SAMimate H.264 encode failed: {detail}")
    if not out_path.exists() or out_path.stat().st_size <= 0: raise ValueError("SAMimate H.264 encode failed")
    return str(out_path)

def prepare_samimate_composite_inputs(data: dict[str, Any]) -> dict[str, Any]:
    foreground = (data.get("foreground_path") or "").strip()
    background = (data.get("background_path") or "").strip()
    subject_mask = (data.get("mask_video_path") or "").strip()
    background_mask = (data.get("inverted_mask_video_path") or "").strip()
    if not foreground: raise HTTPException(400, "foreground_path required")
    if not background: raise HTTPException(400, "background_path required")
    if not subject_mask and not background_mask: raise HTTPException(400, "mask_video_path or inverted_mask_video_path required")
    if data.get("generation_backend") == "h3":
        if not subject_mask:
            raise ValueError("H3 composite requires the original subject mask")
        count = estimated_video_frames(background)
        if duration(foreground) + 1 / 24 < duration(background) or abs(duration(subject_mask) - duration(background)) > 1 / 24 + 0.01:
            raise ValueError("H3 composite source/mask durations differ or generated video is too short")
        return {"foreground": foreground, "background": background, "mask": subject_mask,
                "subject_mask": subject_mask, "background_mask": background_mask,
                "mask_selects_background": False, "frame_count": count}
    fg_info = video_probe(local_media_path(foreground)) if foreground and not is_url(foreground) else {}
    width = int(fg_info.get("width") or data.get("width") or 0)
    height = int(fg_info.get("height") or data.get("height") or 0)
    target_size = (width, height) if width and height else None
    fg_duration = duration(local_media_path(foreground)) if foreground and not is_url(foreground) else 0
    bg = process_video(background, data.get("video_start"), data.get("video_end"), data.get("video_frame_cap"), data.get("video_crop"), target_size, data.get("fps"))
    if fg_duration > 0:
        bg = conform_video_duration(bg, fg_duration, "samimate_background")
    subject_mask_prepared = process_wan_mask(subject_mask, foreground, None, None, None, None, target_size, True) if subject_mask else ""
    background_mask_prepared = process_wan_mask(background_mask, foreground, None, None, None, None, target_size, True) if background_mask else ""
    if background_mask_prepared:
        mask = background_mask_prepared
        mask_selects_background = True
    else:
        mask = subject_mask_prepared
        mask_selects_background = False
    return {
        "foreground": foreground,
        "background": bg,
        "mask": mask,
        "subject_mask": subject_mask_prepared,
        "background_mask": background_mask_prepared,
        "mask_selects_background": mask_selects_background,
        "foreground_probe": media_probe_any(foreground),
        "background_probe": media_probe_any(bg),
        "mask_probe": media_probe_any(mask),
        "trim": {"start": data.get("video_start") or "0", "end": data.get("video_end") or "full", "frame_cap": data.get("video_frame_cap") or "0", "crop": data.get("video_crop") or None},
    }

def discard_sam_still_outputs(result: dict[str, Any]) -> None:
    for key in ("mask_path", "overlay_path", "cutout_path", "masked_image_path"):
        path = result.pop(key, None)
        if path:
            try: Path(path).unlink(missing_ok=True)
            except Exception: pass

def samimate_h3_prompt(references: list[str], subject: str, instructions: str = "") -> str:
    if not 1 <= len(references) <= 9:
        raise ValueError("SAMimate H3 needs 1 to 9 replacement identity images")
    pictures = ", ".join(f"<Picture {i + 1}>" for i in range(len(references)))
    return f"""subject_definitions:
<Subject 1> is the single replacement person shown in {pictures}. All pictures show the same identity; use Picture 1 for the primary appearance and outfit, and the remaining views to resolve facial and body details. Do not create additional people from these views.

summary:
[video editing + reference generation] Replace only the tracked {subject or 'person'} in the source target with <Subject 1>.

retention_analysis:
<Subject 1>: fully_preserved - retain the reference identity, facial proportions, hair, and primary outfit consistently throughout.
Source target: partially_preserved - change only the selected person's appearance. Preserve their exact position, scale, pose, expression, gaze, lip movements, action timing, occlusions, and contacts with objects. Preserve the original camera, framing, cuts, background, other people, and scene lighting.

detailed_description:
Follow every source frame and action in its original order. Replace the masked person with <Subject 1>, matching the original light direction, exposure, color temperature, local shadows, focus, grain, and motion blur. Keep the replacement within the tracked edit area. Do not introduce new actions, cuts, camera motion, props, or people.
Additional replacement instructions: {instructions.strip() or 'Use the reference appearance.'}

overall_soundscape:
The final composite retains the source soundtrack unchanged in timing.

non_diegetic_music:
Retain the source music in the final composite; do not add new music."""


def samimate_frame_count(path: str) -> int:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        count = 0
        while cap.grab():
            count += 1
        if count == 0:
            raise ValueError(f"No readable video frames: {path}")
        return count
    finally:
        cap.release()


@app.post("/api/samimate/h3-prompt")
def preview_samimate_h3_prompt(data: dict[str, Any]):
    try:
        refs = h3_path_list(data.get("reference_paths"), 9, "SAMimate identity images")
        return {"prompt": samimate_h3_prompt(refs, str(data.get("subject_prompt") or "person"), str(data.get("prompt") or ""))}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/samimate/prepare-h3")
def prepare_samimate_h3(data: dict[str, Any]):
    """One authoritative 24fps source for SAM, H3 and the final composite."""
    try:
        source = local_probe_video_path(str(data.get("video_path") or ""))
        info = video_probe(source)
        start = seconds(data.get("video_start")) or 0.0
        end = seconds(data.get("video_end"))
        end = min(end if end is not None else duration(source), duration(source))
        caps = [int(data.get(k) or 0) for k in ("video_frame_cap", "mask_frame_cap")]
        span = end - start
        if any(c < 0 for c in caps):
            raise ValueError("Frame caps cannot be negative")
        if any(caps):
            span = min(span, min(c for c in caps if c > 0) / 24)
        if start < 0 or not 1 <= span <= 15:
            raise ValueError("Select a source trim between 1 and 15 seconds for H3; longer clips are not silently truncated")
        frames = int(round(span * 24))
        crop = data.get("video_crop") or {}
        filters = []
        if crop:
            x, y, w, h = [int(round(float(crop.get(k, 0)))) for k in ("x", "y", "w", "h")]
            if w > 1 and h > 1:
                if min(x, y) < 0 or x + w > int(info["width"]) or y + h > int(info["height"]):
                    raise ValueError("Crop is outside the source video")
                filters.append(f"crop={w}:{h}:{x}:{y}")
        plate = TEMP_DIR / f"samimate_h3_plate_{uuid.uuid4().hex}.mkv"
        out = TEMP_DIR / f"samimate_h3_source_{uuid.uuid4().hex}.mkv"
        cmd = [ffmpeg_bin(), "-y", "-v", "error", "-ss", str(start), "-i", source,
               "-t", str(span), "-map", "0:v:0", "-map", "0:a?"]
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += ["-c:v", "ffv1", "-c:a", "pcm_s16le", str(plate)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode:
            raise ValueError(f"H3 source preparation failed: {proc.stderr}")
        proc = subprocess.run([ffmpeg_bin(), "-y", "-v", "error", "-i", str(plate),
                               "-vf", "fps=24", "-an", "-c:v", "ffv1", str(out)], capture_output=True, text=True)
        if proc.returncode:
            raise ValueError(f"H3 24fps source preparation failed: {proc.stderr}")
        prepared = video_probe(str(out))
        actual = samimate_frame_count(str(out))
        if actual < 24 or actual > 360:
            raise ValueError("Prepared H3 source must contain 24 to 360 frames")
        width = max(32, round(int(data.get("width") or prepared["width"]) / 32) * 32)
        height = max(32, round(int(data.get("height") or prepared["height"]) / 32) * 32)
        if max(width, height) > 4096 or width * height > 2_000_000:
            raise ValueError("Choose an H3 generation canvas of at most 2 megapixels")
        return {"source_path": str(out), "composite_source_path": str(plate), "duration": actual / 24, "frame_count": actual,
                "width": width, "height": height, "fps": 24}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/samimate/run-folder")
async def samimate_run_folder():
    try:
        out_dir = next_numbered_dir(samimate_root_dir(), "SAMimate")
        return {"name": out_dir.name, "path": str(out_dir)}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/sam/frame")
async def sam_frame(data: dict[str, Any]):
    source_path = (data.get("source_path") or "").strip()
    if not source_path: raise HTTPException(400, "source_path required")
    try:
        local_source_path = prepare_sam_source(source_path, data)
        return extract_sam_frame(local_source_path, data.get("frame_time", 0))
    except Exception as e: raise HTTPException(400, str(e))

@app.post("/api/sam/run")
async def sam_run(data: dict[str, Any]):
    source_path = (data.get("source_path") or "").strip()
    if not source_path: raise HTTPException(400, "source_path required")
    try:
        if data.get("source_workflow") == "samimate":
            data["video_frame_cap"] = int(data.get("samimate_mask_frame_cap") or 0)
        local_source_path = prepare_sam_source(source_path, data)
        frame = extract_sam_frame(local_source_path, data.get("frame_time", 0))
        s = settings(); sam_s = s.get("sam", {}); backend = data.get("backend") or sam_s.get("backend", "local_sam3"); command = sam_s.get("external_command", "")
        if backend == "local_sam3" and source_kind(local_source_path) == "video" and data.get("make_masked_video") and int(data.get("video_frame_cap") or 0) <= 0:
            total_frames = estimated_video_frames(local_source_path)
            max_full_frames = int(data.get("sam3_max_full_frames") or 1200)
            if total_frames > max_full_frames:
                raise ValueError(
                    f"Local SAM3 would need to load about {total_frames} frames for this selected video window. "
                    f"That is too large for the local video predictor guard of {max_full_frames} frames. "
                    "Lower SAMimate FPS, set SAM/Wan Frame Cap to a smaller value like 300, or shorten the trim before running SAMimate."
                )
        if backend == "external_sam3":
            result = run_external_sam3(frame["frame_path"], local_source_path, data, command)
        elif backend == "local_sam3":
            runner_data = dict(data)
            runner_data["hf_token"] = runner_data.get("hf_token") or sam_s.get("hf_token", "")
            runner_data["hf_cache_dir"] = runner_data.get("hf_cache_dir") or s.get("hf_cache_dir", "")
            result = run_local_sam3(frame["frame_path"], local_source_path, runner_data, data.get("sam3_python") or sam_s.get("sam3_python") or sys.executable)
        else:
            result = make_local_test_mask(frame["frame_path"], data)
        result.update({"frame_path": frame["frame_path"], "frame_url": frame["frame_url"], "source_path": source_path, "source_kind": frame["kind"]})
        if frame["kind"] == "video" and data.get("make_masked_video"):
            if not result.get("mask_video_path"):
                result["mask_video_path"] = make_mask_video_from_image_mask(local_source_path, result["mask_path"], data.get("video_frame_cap"))
            if result.get("mask_video_path") and not result.get("inverted_mask_video_path"):
                result["inverted_mask_video_path"] = invert_mask_video(result["mask_video_path"])
            if not result.get("masked_video_path"):
                result["masked_video_path"] = apply_mask_to_video(local_source_path, result["mask_path"], data.get("mask_fill", "black"), data.get("video_frame_cap"))
            discard_sam_still_outputs(result)
        for k in ["mask_path", "mask_video_path", "inverted_mask_video_path", "overlay_path", "cutout_path", "masked_image_path", "masked_video_path"]:
            if result.get(k): result[k.replace("_path", "_url")] = safe_file_url(result[k])
        sam_record({"time": time.strftime('%Y-%m-%d %H:%M:%S'), **{k: result.get(k) for k in ["source_path", "mask_video_path", "inverted_mask_video_path", "masked_video_path", "backend_used"]}})
        return result
    except Exception as e: raise HTTPException(400, str(e))

@app.get("/api/sam/history")
def sam_history(): return {"items": load_json(SAM_HISTORY_PATH, [])}
@app.get("/api/sam/health")
def sam_health():
    s = settings().get("sam", {})
    return {
        "ok": True,
        "backend": s.get("backend", "local_sam3"),
        "sam3_python": s.get("sam3_python") or sys.executable,
        "has_hf_token": bool((s.get("hf_token") or "").strip()),
        "has_external_command": bool((s.get("external_command") or "").strip()),
        "local_test_is_real_sam3": False,
        "sam_dir": str(SAM_DIR),
    }

@app.post("/api/run/infinitetalk")
async def run_inf(data: dict[str,Any]):
    s=runtime_settings(data.get("settings",{})); key=s.get("runpod_api_key"); endpoint=s.get("infinite_endpoint_id")
    if not key or not endpoint: raise HTTPException(400,"RunPod key and InfiniteTalk endpoint required")
    if endpoint == "iy7tnhm4e521zk":
        raise HTTPException(400, "InfiniteTalk endpoint is still set to the old default that returned 404. Paste your actual InfiniteTalk RunPod endpoint ID in Settings.")
    delivery=data.get("delivery_mode") or s.get("delivery_mode","runpod_volume_path"); s3=S3Helper(s) if delivery!="base64" else None
    input_type = data.get("input_type","image")
    person_count = data.get("person_count","single")
    if input_type == "video" and not (data.get("video_path") or "").strip():
        raise HTTPException(400, "InfiniteTalk video mode requires a video_path")
    if input_type == "image" and not (data.get("image_path") or "").strip():
        raise HTTPException(400, "InfiniteTalk image mode requires an image_path")
    if not (data.get("audio_path") or "").strip():
        raise HTTPException(400, "InfiniteTalk requires audio_path")
    if person_count == "multi" and not (data.get("audio2_path") or "").strip():
        raise HTTPException(400, "InfiniteTalk multi mode requires audio2_path")
    payload={"input_type":input_type,"person_count":person_count,"prompt":data.get("prompt") or "A person talking naturally","width":int(data.get("width") or 512),"height":int(data.get("height") or 512),"force_offload":as_bool(data.get("force_offload"), True),"network_volume":as_bool(data.get("network_volume"), True)}
    audio_start = seconds(data.get("audio_start")) or 0.0
    audio_end = seconds(data.get("audio_end"))
    audio_len = max(0.0, (audio_end - audio_start) if audio_end is not None else duration(data.get("audio_path","")) - audio_start)
    if as_bool(data.get("sync_to_audio"), False) and input_type == "image" and data.get("max_frame") in (None, "") and audio_len > 0:
        data["max_frame"] = int(max(1, round(audio_len * 24)))
    if data.get("max_frame") not in (None,""): payload["max_frame"]=int(data["max_frame"])
    processed_video = None
    if payload["input_type"]=="video":
        if as_bool(data.get("sync_to_audio"), False) and audio_len > 0:
            vst = seconds(data.get("video_start")) or 0.0
            data["video_end"] = vst + audio_len
            data["video_frame_cap"] = ""
        processed_video=process_video(data.get("video_path",""), data.get("video_start"), data.get("video_end"), data.get("video_frame_cap"), data.get("video_crop"))
        payload.update(media_payload("video",processed_video,delivery,s3))
    else:
        payload.update(media_payload("image", data.get("image_path",""), delivery, s3))
    processed_audio = process_audio(data.get("audio_path",""), data.get("audio_start"), data.get("audio_end"))
    payload.update(media_payload("wav", processed_audio, delivery, s3))
    processed_audio2 = None
    if payload["person_count"]=="multi":
        processed_audio2 = process_audio(data["audio2_path"], data.get("audio2_start"), data.get("audio2_end"))
        payload.update(media_payload("wav_2",processed_audio2,delivery,s3))
    try: payload.update(json.loads(data.get("advanced_json") or "{}"))
    except Exception as e: raise HTTPException(400,f"Invalid advanced JSON: {e}")
    cleared = []
    if payload["input_type"] == "video" and not (data.get("image_path") or "").strip():
        cleared.append("image")
    if payload["input_type"] == "image" and not (data.get("video_path") or "").strip():
        cleared.append("video")
    if not (data.get("audio_path") or "").strip():
        cleared.append("wav")
    if payload["person_count"] != "multi" or not (data.get("audio2_path") or "").strip():
        cleared.append("wav_2")
    remove_media_payload_keys(payload, cleared)
    job=submit(endpoint,key,payload,s)
    debug = {
        "input_type": payload.get("input_type"),
        "person_count": payload.get("person_count"),
        "image_key": next((k for k in ("image_path", "image_url", "image_base64") if k in payload), None),
        "video_key": next((k for k in ("video_path", "video_url", "video_base64") if k in payload), None),
        "audio1_key": next((k for k in ("wav_path", "wav_url", "wav_base64") if k in payload), None),
        "audio2_key": next((k for k in ("wav_2_path", "wav_2_url", "wav_2_base64") if k in payload), None),
        "video_start": data.get("video_start") or "0",
        "video_end": data.get("video_end") or "full",
        "frame_cap": "ignored_by_trim" if (data.get("video_start") or data.get("video_end")) else (data.get("video_frame_cap") or "0"),
        "max_frame": payload.get("max_frame", "none"),
        "delivery_mode": delivery,
        "processed_video": processed_video,
        "audio_start": data.get("audio_start") or "0",
        "audio_end": data.get("audio_end") or "full",
        "audio2_start": data.get("audio2_start") or "0",
        "audio2_end": data.get("audio2_end") or "full",
        "processed_audio": processed_audio,
        "processed_audio2": processed_audio2,
        "sync_to_audio": as_bool(data.get("sync_to_audio"), False),
    }
    record_job_event({"target": "inf", "endpoint_id": endpoint, "job_id": job.get("id"), "status": "SUBMITTED", "payload_keys": sorted(payload.keys()), "debug": debug})
    return {"job":job,"endpoint_id":endpoint,"payload_keys":sorted(payload.keys()),"debug":debug}


def normalize_point_json(raw: Any, fallback: Any) -> tuple[str, Any]:
    """Return both compact JSON string and parsed value."""
    if raw is None or raw == "":
        parsed = fallback
    elif isinstance(raw, str):
        parsed = json.loads(raw) if raw.strip() else fallback
    else:
        parsed = raw
    return json.dumps(parsed, separators=(",", ":")), parsed


def coerce_control_points(data: dict[str, Any]) -> tuple[str, str, str, int, int]:
    """Mirror the old desktop app: send points_store / coordinates / neg_coordinates as JSON strings.

    v9.1 desktop uses visible raw fields and sends the strings directly into payload.
    The only validation is that points_store and coordinates are present, which avoids
    changing the user's payload shape before the endpoint injects it into Comfy.
    """
    points_store_raw = data.get("points_store_raw")
    coordinates_raw = data.get("coordinates_raw")
    neg_raw = data.get("neg_coordinates_raw")

    # Fallback for older web requests that only sent arrays/objects.
    if not points_store_raw and "points_store" in data:
        points_store_raw = data.get("points_store")
    if not coordinates_raw and "coordinates" in data:
        coordinates_raw = data.get("coordinates")
    if not neg_raw and "neg_coordinates" in data:
        neg_raw = data.get("neg_coordinates")

    coords_s, coords = normalize_point_json(coordinates_raw, [])
    neg_s, neg = normalize_point_json(neg_raw, [])
    ps_s, ps = normalize_point_json(points_store_raw, {"positive": coords, "negative": neg})

    if not coords and isinstance(ps, dict):
        coords = ps.get("positive") or []
        neg = ps.get("negative") or neg

    if not coords:
        raise HTTPException(400, "Control points mode needs at least one positive point. Click Add points to payload after placing a point.")
    if not neg:
        # Upstream examples use a dummy negative point even when the user has no real
        # negative selections. The Comfy point-control workflow can reject an empty
        # neg_coordinates field at /prompt.
        neg = [{"x": 0, "y": 0}]

    coords_s = json.dumps(coords, separators=(",", ":"))
    neg_s = json.dumps(neg, separators=(",", ":"))
    ps_s = json.dumps({"positive": coords, "negative": neg}, separators=(",", ":"))

    return ps_s, coords_s, neg_s, len(coords or []), len(neg or [])



@app.post("/api/run/wananimate")
async def run_wan(data: dict[str,Any]):
    s=runtime_settings(data.get("settings",{})); key=s.get("runpod_api_key"); endpoint=s.get("wan_endpoint_id")
    if not key or not endpoint: raise HTTPException(400,"RunPod key and WanAnimate endpoint required")
    delivery=data.get("wan_delivery_override") or data.get("delivery_mode") or s.get("delivery_mode","runpod_volume_path")
    if data.get("source_workflow") == "samimate" and delivery == "auto":
        delivery = data.get("delivery_mode") or s.get("delivery_mode") or "runpod_volume_path"
        if delivery == "auto":
            delivery = "runpod_volume_path"
    if delivery == "global":
        delivery = data.get("delivery_mode") or s.get("delivery_mode","runpod_volume_path")
    s3=S3Helper(s) if delivery!="base64" else None
    if data.get("source_workflow") == "samimate":
        data["video_frame_cap"] = int(data.get("samimate_mask_frame_cap") or 0)
    width, height, dim_debug = resolve_wan_dimensions(data, local_probe_video_path(data.get("video_path","")))
    vp=process_video(data.get("video_path",""), data.get("video_start"), data.get("video_end"), data.get("video_frame_cap"), data.get("video_crop"), (width, height), data.get("fps") if data.get("source_workflow") == "samimate" else None)
    image_path = process_wan_reference_image(data.get("image_path","")) if (data.get("image_path") or "").strip() else ""
    mask_path = ""
    mask_alignment_debug = None
    if (data.get("mask_path") or "").strip():
        mask_alignment = align_wan_mask_to_video(data, vp)
        mask_path = mask_alignment["path"]
        mask_alignment_debug = {k: mask_alignment.get(k) for k in ("mask_probe", "video_probe", "trim", "resolved_width", "resolved_height")}
    masked_video_path = ""
    if (data.get("masked_video_path") or "").strip():
        masked_video_path = process_video(data.get("masked_video_path",""), data.get("video_start"), data.get("video_end"), data.get("video_frame_cap"), data.get("video_crop"), (width, height))
    seed_value, seed_was_randomized = normalize_seed(data.get("seed"))
    payload={"prompt":data.get("prompt") or "","negative_prompt":data.get("negative_prompt") or "blurry, low quality, distorted","mode":data.get("mode") or "replace","pose_estimation":as_bool(data.get("pose_estimation"), True),"face_detection":as_bool(data.get("face_detection"), True),"mask_editing":as_bool(data.get("mask_editing"), False),"network_volume":as_bool(data.get("network_volume"), True),"width":width,"height":height,"seed":seed_value,"fps":int(data.get("fps") or 16),"cfg":float(data.get("cfg") or 1.0),"steps":int(data.get("steps") or 6)}
    payload.update(media_payload("image", image_path, delivery, s3)); payload.update(media_payload("video", vp, delivery, s3))
    if mask_path: payload.update(media_payload("mask", mask_path, delivery, s3))
    if masked_video_path: payload.update(media_payload("masked_video", masked_video_path, delivery, s3))
    point_debug = {"positive_points": 0, "negative_points": 0}
    if data.get("control_points_enabled"):
        points_store_s, coords_s, neg_s, pos_count, neg_count = coerce_control_points(data)
        payload["points_store"] = points_store_s
        payload["coordinates"] = coords_s
        payload["neg_coordinates"] = neg_s
        point_debug = {"positive_points": pos_count, "negative_points": neg_count}
    try: payload.update(json.loads(data.get("advanced_json") or "{}"))
    except Exception as e: raise HTTPException(400,f"Invalid advanced JSON: {e}")
    cleared = []
    if not (data.get("image_path") or "").strip():
        cleared.append("image")
    if not (data.get("video_path") or "").strip():
        cleared.append("video")
    if not (data.get("mask_path") or "").strip():
        cleared.append("mask")
    if not (data.get("masked_video_path") or "").strip():
        cleared.append("masked_video")
    debug = {
        "mode": payload.get("mode"),
        "control_points": bool(data.get("control_points_enabled")),
        "positive_points": point_debug.get("positive_points", 0),
        "negative_points": point_debug.get("negative_points", 0),
        "image_key": next((k for k in ("image_path", "image_url", "image_base64") if k in payload), None),
        "video_key": next((k for k in ("video_path", "video_url", "video_base64") if k in payload), None),
        "masked_video_key": next((k for k in ("masked_video_path", "masked_video_url", "masked_video_base64") if k in payload), None),
        "mask_key": next((k for k in ("mask_path", "mask_url", "mask_base64") if k in payload), None),
        "delivery_mode": delivery,
        "image_value": next((payload.get(k) for k in ("image_path", "image_url", "_image_s3_uri") if k in payload), None),
        "video_value": next((payload.get(k) for k in ("video_path", "video_url", "_video_s3_uri") if k in payload), None),
        "masked_video_value": next((payload.get(k) for k in ("masked_video_path", "masked_video_url", "_masked_video_s3_uri") if k in payload), None),
        "mask_value": next((payload.get(k) for k in ("mask_path", "mask_url", "_mask_s3_uri") if k in payload), None),
        "video_start": data.get("video_start") or "0",
        "video_end": data.get("video_end") or "full",
        "frame_cap": "ignored_by_trim" if (data.get("video_start") or data.get("video_end")) else (data.get("video_frame_cap") or "0"),
        "seed": payload.get("seed"),
        "seed_randomized_from_negative": seed_was_randomized,
        "pose_estimation": payload.get("pose_estimation"),
        "face_detection": payload.get("face_detection"),
        "mask_editing": payload.get("mask_editing"),
        "network_volume": payload.get("network_volume"),
        "resolved_width": width,
        "resolved_height": height,
        "dimension_debug": dim_debug,
        "processed_video": vp,
        "processed_mask": mask_path,
        "mask_alignment": mask_alignment_debug,
        "processed_image": image_path,
    }
    remove_media_payload_keys(payload, cleared)
    debug["payload_keys"] = sorted(payload.keys())
    try:
        job=submit(endpoint,key,payload,s)
    except HTTPException as e:
        record_job_event({"target": "samimate" if data.get("source_workflow") == "samimate" else "wan", "endpoint_id": endpoint, "job_id": None, "status": "SUBMIT_FAILED", "payload_keys": sorted(payload.keys()), "debug": debug, "error": e.detail})
        raise HTTPException(e.status_code, {"message": e.detail, "debug": debug, "payload_keys": sorted(payload.keys())}) from e
    record_job_event({"target": "samimate" if data.get("source_workflow") == "samimate" else "wan", "endpoint_id": endpoint, "job_id": job.get("id"), "status": "SUBMITTED", "payload_keys": sorted(payload.keys()), "debug": debug})
    return {"job": job, "endpoint_id": endpoint, "payload_keys": sorted(payload.keys()), "debug": debug}

def validate_h3_steps(value: Any) -> int:
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or number < 1 or not number.is_integer():
            raise ValueError()
        return int(number)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError("H3 steps must be a positive whole number") from e

@app.post("/api/run/h3")
async def run_h3(data: dict[str, Any]):
    if data.get("source_workflow") == "samimate":
        data = dict(data)
        try:
            refs = h3_path_list(data.get("reference_paths"), 9, "SAMimate identity images")
            if not data.get("source_video_path") or not data.get("mask_path"):
                raise ValueError("SAMimate H3 requires the prepared source and SAM subject mask")
            source_info = video_probe(local_media_path(data["source_video_path"]))
            mask_info = video_probe(local_media_path(data["mask_path"]))
            if any(abs(float(i.get("fps") or 0) - 24) > 0.05 for i in (source_info, mask_info)):
                raise ValueError("SAMimate source and mask must both be prepared at 24 fps")
            source_count = samimate_frame_count(data["source_video_path"])
            if source_count != samimate_frame_count(data["mask_path"]):
                raise ValueError("SAMimate source and tracked mask frame counts differ; rerun segmentation")
            if any("Acc-8Step" in str(item.get("name", "")) for item in data.get("loras", []) if isinstance(item, dict)):
                raise ValueError("Remove the PDD acceleration LoRA before running the SAMimate masked baseline")
            data.update(task="r2v", duration=source_count / 24, reference_video_paths=[], reference_audio_paths=[],
                        audio_path="", use_reference_audio_as_output=False,
                        prompt=samimate_h3_prompt(refs, str(data.get("subject_prompt") or "person"), str(data.get("prompt") or "")))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    s = runtime_settings(data.get("settings", {}))
    key = s.get("runpod_api_key")
    endpoint = data.get("h3_endpoint_id") or s.get("h3_endpoint_id") or "s1e3i9book2zhh"
    if not key or not endpoint:
        raise HTTPException(400, "RunPod key and MiniMax H3 endpoint required")
    task = str(data.get("task") or "fl2v").lower()
    if task not in {"t2v", "fl2v", "r2v"}:
        raise HTTPException(400, "H3 task must be t2v, fl2v or r2v")
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(400, "MiniMax H3 requires a prompt")
    duration_value = float(data.get("duration") or 2.0)
    megapixels_value = float(data.get("megapixels") or 0.2)
    turbo_enabled = as_bool(data.get("turbo_enabled"), True)
    seed_value, seed_was_randomized = normalize_seed(-1 if as_bool(data.get("seed_random"), False) else data.get("seed"))
    sampler = str(data.get("sampler") or ("h3_turbo" if turbo_enabled else "res_multistep")).lower()
    scheduler = str(data.get("scheduler") or ("simple" if turbo_enabled else "beta")).lower()
    cache_enabled = as_bool(data.get("cache_enabled"), False)
    if sampler not in H3_SAMPLERS:
        raise HTTPException(400, f"Unsupported H3 sampler: {sampler}")
    if scheduler not in H3_SCHEDULERS:
        raise HTTPException(400, f"Unsupported H3 scheduler: {scheduler}")
    if sampler == "h3_turbo" and not turbo_enabled:
        raise HTTPException(400, "The dedicated H3 Turbo sampler requires the separate Turbo LoRA")
    try:
        steps_value = validate_h3_steps(data.get("steps", 6 if turbo_enabled else 20))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    cache_value = float(data.get("cache_threshold") or 0.18)
    if not 1 <= duration_value <= 15:
        raise HTTPException(400, "H3 duration must be between 1 and 15 seconds")
    if not 0.2 <= megapixels_value <= 2.0:
        raise HTTPException(400, "H3 megapixels must be between 0.2 and 2.0")

    if not 0 <= cache_value <= 1:
        raise HTTPException(400, "H3 cache threshold must be between 0 and 1")

    storage_s = h3_storage_settings(s)
    try:
        h3_s3 = configured_s3_helper(storage_s)
    except Exception as e:
        raise HTTPException(400, f"Invalid H3 network-volume S3 settings: {e}") from e
    delivery = str(data.get("delivery") or "auto").lower()
    h3_defaults = s.get("h3") or {}
    try:
        fl2va_model = h3_model_name(data.get("fl2va_model") or h3_defaults.get("fl2va_model"), "FL2VA model")
        ref2va_model = h3_model_name(data.get("ref2va_model") or h3_defaults.get("ref2va_model"), "Ref2VA model")
        text_encoder = h3_model_name(data.get("text_encoder") or h3_defaults.get("text_encoder"), "text encoder / CLIP")
        video_vae = h3_model_name(data.get("video_vae") or h3_defaults.get("video_vae"), "video VAE")
        audio_vae = h3_model_name(data.get("audio_vae") or h3_defaults.get("audio_vae"), "audio VAE")
        clip_projection = h3_model_name(data.get("clip_projection") or h3_defaults.get("clip_projection"), "CLIP projection", required=False)
        turbo_lora = h3_worker_lora_name(str(data.get("turbo_lora") or h3_defaults.get("turbo_lora") or "")) if turbo_enabled else ""
        turbo_strength = float(data.get("turbo_strength") if data.get("turbo_strength") is not None else h3_defaults.get("turbo_strength", 1.0))
    except (TypeError, ValueError) as e:
        raise HTTPException(400, f"Invalid H3 model configuration: {e}") from e
    if not 0 <= turbo_strength <= 2:
        raise HTTPException(400, "H3 Turbo strength must be between 0 and 2")
    active_model = ref2va_model if task == "r2v" else fl2va_model
    payload: dict[str, Any] = {
        "task": task,
        "prompt": prompt,
        "duration": duration_value,
        "megapixels": megapixels_value,
        "steps": steps_value,
        "seed": seed_value,
        "sampler": sampler,
        "scheduler": scheduler,
        "cache_enabled": cache_enabled,
        "cache_threshold": cache_value,
        "attention": str(data.get("attention") or "auto").lower(),
        "aspect_ratio": str(data.get("aspect_ratio") or "16:9 (Widescreen)"),
        "filename_prefix": str(data.get("filename_prefix") or "RunPod_Media_Console_H3"),
        "model": active_model,
        "clip": text_encoder,
        "video_vae": video_vae,
        "audio_vae": audio_vae,
        "turbo_enabled": turbo_enabled,
        "turbo_family": str(data.get("turbo_family") or "auto"),
    }
    if payload["attention"] not in {"auto", "native", "sage", "sol-attn", "sla", "vsa"}:
        raise HTTPException(400, "Unsupported H3 attention mode")
    for name, default in (("sparse_keep_percent", 10.0), ("sparse_tau", 1.3),
                          ("sparse_start_percent", 0.2), ("sparse_end_percent", 1.0)):
        try:
            payload[name] = float(data.get(name, default))
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"Invalid {name}") from e
    payload["pdd_enabled"] = as_bool(data.get("pdd_enabled"), False)
    payload["sparse_trained_weights"] = as_bool(data.get("sparse_trained_weights"), False)
    if payload["attention"] in {"sol-attn", "sla", "vsa"}:
        if cache_enabled:
            raise HTTPException(400, "Disable FirstBlockCache when using sparse attention")
        if payload["attention"] in {"sla", "vsa"} and not payload["sparse_trained_weights"]:
            raise HTTPException(400, "Select matching SLA/VSA trained weights and confirm the sparse weights control")
        if not (0.5 <= payload["sparse_keep_percent"] <= 95 and 0 <= payload["sparse_tau"] <= 4
                and 0 <= payload["sparse_start_percent"] <= payload["sparse_end_percent"] <= 1):
            raise HTTPException(400, "Invalid sparse attention settings")
    if payload["attention"] not in {"auto", "native", "sage", "sol-attn", "sla", "vsa"}:
        raise HTTPException(400, "Unsupported H3 attention mode")
    for name, default in (("sparse_keep_percent", 10.0), ("sparse_tau", 1.3),
                          ("sparse_start_percent", 0.2), ("sparse_end_percent", 1.0)):
        try:
            payload[name] = float(data.get(name, default))
        except (ValueError, TypeError) as e:
            raise HTTPException(400, f"Invalid {name}") from e
    payload["sparse_trained_weights"] = as_bool(data.get("sparse_trained_weights"), False)
    if payload["attention"] in {"sol-attn", "sla", "vsa"}:
        if cache_enabled:
            raise HTTPException(400, "Disable FirstBlockCache when using sparse attention")
        if payload["attention"] in {"sla", "vsa"} and not payload["sparse_trained_weights"]:
            raise HTTPException(400, "Select matching SLA/VSA trained weights and confirm the sparse weights control")
        if not (0.5 <= payload["sparse_keep_percent"] <= 95 and 0 <= payload["sparse_tau"] <= 4
                and 0 <= payload["sparse_start_percent"] <= payload["sparse_end_percent"] <= 1):
            raise HTTPException(400, "Invalid sparse attention settings")
    if clip_projection:
        payload["clip_projection"] = clip_projection
    if turbo_enabled:
        payload["turbo_lora"] = turbo_lora
        payload["turbo_strength"] = turbo_strength
    effective_delivery = delivery
    estimated_inline_bytes = 0
    try:
        if task == "fl2v":
            first_frame = str(data.get("first_frame_path") or "").strip()
            last_frame = str(data.get("last_frame_path") or "").strip()
            effective_delivery, estimated_inline_bytes = h3_effective_delivery(delivery, [first_frame, last_frame])
            payload["first_frame"] = h3_asset_payload(first_frame, "first_frame", effective_delivery, h3_s3)
            if last_frame:
                payload["last_frame"] = h3_asset_payload(last_frame, "last_frame", effective_delivery, h3_s3)
        elif task == "r2v":
            references = h3_path_list(data.get("reference_paths"), 9, "H3 R2VA image references")
            videos = h3_path_list(data.get("reference_video_paths"), 3, "H3 R2VA video references")
            audios = h3_path_list(data.get("reference_audio_paths"), 3, "H3 R2VA audio references")
            legacy_audio = str(data.get("audio_path") or "").strip()
            if legacy_audio and not audios:
                audios = [legacy_audio]
            if not references and not videos and not audios:
                raise ValueError("H3 R2VA requires at least one image, video, or audio reference")
            edit_paths = [data["source_video_path"], data["mask_path"]] if data.get("source_workflow") == "samimate" else []
            effective_delivery, estimated_inline_bytes = h3_effective_delivery(delivery, references + videos + audios + edit_paths)
            if edit_paths:
                # Older workers must reject this request, never silently ignore the mask.
                payload["task"] = "r2v_masked"
                payload["masked_edit"] = {
                    "source": h3_asset_payload(edit_paths[0], "samimate_source", effective_delivery, h3_s3),
                    "mask": h3_asset_payload(edit_paths[1], "samimate_mask", effective_delivery, h3_s3),
                    "width": int(data.get("width") or 0), "height": int(data.get("height") or 0),
                }
            payload["references"] = [h3_asset_payload(str(path), f"reference_{i + 1}", effective_delivery, h3_s3) for i, path in enumerate(references)]
            payload["reference_videos"] = [h3_asset_payload(path, f"reference_video_{i + 1}", effective_delivery, h3_s3) for i, path in enumerate(videos)]
            video_audio = data.get("reference_video_audio") or []
            if not isinstance(video_audio, list):
                raise ValueError("H3 R2VA video audio selections must be a list")
            payload["reference_video_audio"] = [as_bool(video_audio[i], True) if i < len(video_audio) else True for i in range(len(videos))]
            payload["reference_video_settings"] = h3_reference_video_settings(data.get("reference_video_settings"), len(videos), duration_value)
            payload["reference_audios"] = [h3_asset_payload(path, f"reference_audio_{i + 1}", effective_delivery, h3_s3) for i, path in enumerate(audios)]
            if audios:
                payload["audio"] = payload["reference_audios"][0]
                payload["use_reference_audio_as_output"] = as_bool(data.get("use_reference_audio_as_output"), False)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    loras = data.get("loras") or []
    if isinstance(loras, str):
        try:
            loras = json.loads(loras) if loras.strip() else []
        except Exception as e:
            raise HTTPException(400, f"Invalid H3 LoRA JSON: {e}") from e
    if not isinstance(loras, list):
        raise HTTPException(400, "H3 LoRAs must be a JSON array")
    clean_loras: list[dict[str, Any]] = []
    for index, item in enumerate(loras):
        if not isinstance(item, dict):
            raise HTTPException(400, f"H3 LoRA {index + 1} must be an object with name and strength")
        try:
            name = h3_lora_name(str(item.get("name") or ""))
            strength = float(item.get("strength", 1.0))
        except Exception as e:
            raise HTTPException(400, f"Invalid H3 LoRA {index + 1}: {e}") from e
        if not 0 <= strength <= 2:
            raise HTTPException(400, f"H3 LoRA strength for {name} must be between 0 and 2")
        clean_loras.append({"name": h3_worker_lora_name(name), "strength": strength})
    loras = clean_loras
    payload["loras"] = loras
    try:
        advanced = json.loads(data.get("advanced_json") or "{}")
        if not isinstance(advanced, dict):
            raise ValueError("Advanced JSON must be an object")
        protected = {
            "task", "prompt", "duration", "megapixels", "steps", "seed", "seed_random", "sampler", "scheduler", "cache_enabled", "cache_threshold",
            "attention", "pdd_enabled", "sparse_keep_percent", "sparse_tau", "sparse_start_percent", "sparse_end_percent", "sparse_trained_weights", "aspect_ratio", "filename_prefix", "model", "clip", "video_vae",
            "audio_vae", "clip_projection", "turbo_enabled", "turbo_family", "turbo_lora", "turbo_strength",
            "loras", "first_frame", "last_frame", "references", "reference_videos", "reference_video_audio", "reference_video_settings", "reference_audios", "audio",
            "use_reference_audio_as_output", "output_upload_urls", "workflow", "output_layout", "masked_edit",
        }
        conflicts = sorted(protected.intersection(advanced))
        if conflicts:
            raise ValueError("Advanced JSON cannot override visible H3 controls: " + ", ".join(conflicts))
        payload.update(advanced)
    except Exception as e:
        raise HTTPException(400, f"Invalid H3 advanced JSON: {e}") from e

    if not h3_s3:
        raise HTTPException(400, "Configure the H3 S3 bucket and credentials in Settings before running so generated MP4 files can be downloaded from the network volume.")

    # RunPod network-volume S3 does not support presigned URLs. Have the worker
    # write the mounted volume and return its exact path for authenticated GET.
    payload["output_layout"] = "flat_outputs"

    def asset_mode(asset: Any) -> str:
        if not isinstance(asset, dict):
            return "none"
        if "volume_path" in asset:
            return "network_volume"
        return "url" if "url" in asset else ("base64" if "data" in asset else "none")
    debug = {
        "task": task,
        "delivery": delivery,
        "effective_delivery": effective_delivery,
        "estimated_inline_mib": round(estimated_inline_bytes / (1024 * 1024), 2),
        "duration": payload.get("duration"),
        "megapixels": payload.get("megapixels"),
        "steps": payload.get("steps"),
        "seed": seed_value,
        "seed_randomized": seed_was_randomized,
        "sampler": sampler,
        "scheduler": scheduler,
        "cache_enabled": cache_enabled,
        "attention": payload.get("attention"),
        "model": active_model,
        "clip": text_encoder,
        "video_vae": video_vae,
        "audio_vae": audio_vae,
        "turbo_enabled": turbo_enabled,
        "turbo_lora": turbo_lora or None,
        "network_volume_id": (s.get("h3_storage") or {}).get("network_volume_id"),
        "storage_bucket": storage_s.get("s3_bucket"),
        "first_frame_mode": asset_mode(payload.get("first_frame")),
        "last_frame_mode": asset_mode(payload.get("last_frame")),
        "reference_modes": [asset_mode(x) for x in payload.get("references", [])],
        "reference_video_modes": [asset_mode(x) for x in payload.get("reference_videos", [])],
        "reference_audio_modes": [asset_mode(x) for x in payload.get("reference_audios", [])],
        "audio_mode": asset_mode(payload.get("audio")),
        "loras": len(loras),
        "output_delivery": "network_volume",
        "output_layout": "flat_outputs",
        "output_volume_dir": "/runpod-volume/outputs",
    }
    try:
        job = submit(endpoint, key, payload, s)
    except HTTPException as e:
        record_job_event({"target": "h3", "endpoint_id": endpoint, "job_id": None, "status": "SUBMIT_FAILED", "payload_keys": sorted(payload.keys()), "debug": debug, "error": e.detail})
        raise HTTPException(e.status_code, {"message": e.detail, "debug": debug, "payload_keys": sorted(payload.keys())}) from e
    record_job_event({"target": "h3", "endpoint_id": endpoint, "job_id": job.get("id"), "status": "SUBMITTED", "payload_keys": sorted(payload.keys()), "debug": debug})
    return {"job": job, "endpoint_id": endpoint, "payload_keys": sorted(payload.keys()), "debug": debug}

@app.get("/api/job/{endpoint_id}/{job_id}")
def job_status(endpoint_id: str, job_id: str):
    s=settings(); key=s.get("runpod_api_key")
    if not key: raise HTTPException(400,"RunPod key missing")
    event_target = "h3" if endpoint_id == s.get("h3_endpoint_id") else ""
    event = job_event_for(job_id, event_target) if event_target else {}
    target_fps = ((event.get("debug") or {}).get("fps") if isinstance(event.get("debug"), dict) else None)
    st=status(endpoint_id,key,job_id)
    output_error = st.get("output", {}).get("error") if isinstance(st.get("output"), dict) else None
    if st.get("status") == "COMPLETED" and output_error:
        st["status"] = "FAILED"
        st["error"] = str(output_error)
    saved=save_outputs(job_id, st, s, endpoint_id, target_fps=target_fps) if st.get("status")=="COMPLETED" else []
    saved_items = output_preview_items(st, saved) if st.get("status")=="COMPLETED" else []
    parsed_error = None
    if isinstance(st.get("error"), str):
        try:
            parsed_error = json.loads(st["error"])
        except Exception:
            parsed_error = None
    if st.get("status") in {"COMPLETED", "FAILED"}:
        record_job_event({"target": "status", "endpoint_id": endpoint_id, "job_id": job_id, "status": st.get("status"), "saved": saved, "error": st.get("error"), "parsed_error": parsed_error, "hints": explain_error_text(str(st.get("error") or parsed_error or ""))})
    return {"status":st,"saved":saved,"saved_items":saved_items,"uploaded_s3":st.get("_uploaded_s3", []),"download_errors":st.get("_download_errors", []),"parsed_error":parsed_error,"error_hints":explain_error_text(str(st.get("error") or parsed_error or "")),"output_summary":output_summary(st, s)}

def composite_h3_subject(foreground: str, background: str, mask: str, run_dir: Path) -> tuple[str, str]:
    """Composite at the source canvas; never pass preserved pixels through MJPEG."""
    info = video_probe(background)
    width, height = int(info["width"]), int(info["height"])
    frames = samimate_frame_count(background)
    fps = float(info.get("fps") or 24)
    master = run_dir / "06_composite_lossless.mkv"
    preview = run_dir / "overlayed_output.mp4"
    graph = (f"[0:v]format=gbrp,settb=AVTB,setpts=N/({fps}*TB)[bg];"
             f"[1:v]fps={fps},scale={width}:{height},format=gbrp,settb=AVTB,setpts=N/({fps}*TB)[fg];"
             f"[2:v]fps={fps},scale={width}:{height}:flags=neighbor,format=gray,lut=y='if(gte(val,128),255,0)',format=gbrp,settb=AVTB,setpts=N/({fps}*TB)[mask];"
             f"[bg][fg][mask]maskedmerge,trim=end_frame={frames},setpts=N/({fps}*TB)[out]")
    cmd = [ffmpeg_bin(), "-y", "-v", "error", "-i", background, "-i", foreground, "-i", mask,
           "-filter_complex", graph, "-map", "[out]", "-map", "0:a?", "-r", str(fps),
           "-c:v", "ffv1", "-c:a", "copy", str(master)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        raise ValueError(f"H3 lossless composite failed: {proc.stderr}")
    proc = subprocess.run([ffmpeg_bin(), "-y", "-v", "error", "-i", str(master),
                           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
                           "-movflags", "+faststart", str(preview)], capture_output=True, text=True)
    if proc.returncode:
        raise ValueError(f"H3 preview encode failed: {proc.stderr}")
    return str(master), str(preview)


@app.post("/api/samimate/composite")
async def samimate_composite(data: dict[str, Any]):
    try:
        run_dir = Path(data.get("run_dir") or data.get("output_dir") or next_numbered_dir(samimate_root_dir(), "SAMimate")).expanduser()
        run_dir.mkdir(parents=True, exist_ok=True)
        prepared = prepare_samimate_composite_inputs(data)
        items: list[dict[str, str]] = []
        normal_mask = copy_media_named(data.get("mask_video_path") or prepared.get("subject_mask"), run_dir, "01_normal_video_mask")
        inverted_mask = copy_media_named(data.get("inverted_mask_video_path") or prepared.get("background_mask"), run_dir, "02_inverted_video_mask")
        items.append(media_item(normal_mask, "Normal video mask"))
        items.append(media_item(inverted_mask, "Inverted video mask"))
        subject_mask_for_preview = prepared.get("subject_mask") or normal_mask
        white_removed = apply_video_mask_video(prepared["background"], subject_mask_for_preview, run_dir / "03_video_with_white_removed.mp4", keep_white=False)
        black_removed = apply_video_mask_video(prepared["background"], subject_mask_for_preview, run_dir / "04_video_with_black_removed.mp4", keep_white=True)
        items.append(media_item(white_removed, "Video with white removed"))
        items.append(media_item(black_removed, "Video with black removed"))
        is_h3 = data.get("generation_backend") == "h3"
        wan_output = copy_media_named(prepared["foreground"], run_dir, "05_h3_output" if is_h3 else "05_wan_animate_output")
        items.append(media_item(wan_output, "H3 output" if is_h3 else "Wan Animate output"))
        if is_h3:
            master, out = composite_h3_subject(wan_output, prepared["background"], prepared["mask"], run_dir)
            items.append(media_item(master, "Lossless composite master"))
        else:
            out = composite_subject_video(wan_output, prepared["background"], prepared["mask"], prepared["mask_selects_background"], run_dir)
        items.append(media_item(out, "Overlayed output"))
        debug = dict(prepared)
        debug["run_dir"] = str(run_dir)
        return {"path": out, "url": safe_file_url(out), "kind": "video", "items": items, "run_dir": str(run_dir), "debug": debug}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))

if __name__ == "__main__":
    import uvicorn
    webbrowser.open("http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765)
