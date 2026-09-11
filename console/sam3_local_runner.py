"""Local SAM3 image/video runner for RunPod Media Console Web.

This script expects to run inside a Python environment where Meta's sam3 package,
PyTorch, CUDA dependencies, and Hugging Face checkpoint access are already set up.
For images it masks the selected frame. For videos with ``make_masked_video`` it
uses SAM3 video propagation and writes a masked video from per-frame masks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def sane_fps(fps) -> float:
    try:
        value = float(fps or 0)
    except Exception:
        value = 0.0
    if not np.isfinite(value) or value <= 0:
        return 24.0
    # Some variable-frame-rate inputs report unusable OpenCV timebases such as
    # 116129/500, which OpenCV's mp4v writer cannot represent. Keep common
    # rates intact and clamp pathological values to a browser/Wan-safe rate.
    if value > 120:
        return 30.0
    return round(value, 3)


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_numpy_masks(masks):
    if masks is None:
        return np.zeros((0, 1, 1), dtype=bool)
    try:
        import torch
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().float().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(masks)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4:
        arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))
    if arr.size == 0:
        return np.zeros((0, 1, 1), dtype=bool)
    return arr > 0.5


def scores_mask(scores, count: int, threshold: float):
    if scores is None or count <= 0:
        return np.ones((count,), dtype=bool)
    try:
        import torch
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().float().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(scores).reshape(-1)
    if arr.size < count:
        return np.ones((count,), dtype=bool)
    return arr[:count] >= threshold


def normalized_box(box: dict, w: int, h: int):
    x = float(box.get("x", 0) or 0)
    y = float(box.get("y", 0) or 0)
    bw = float(box.get("w", 0) or 0)
    bh = float(box.get("h", 0) or 0)
    if bw <= 1 or bh <= 1:
        return None
    cx = (x + bw / 2.0) / max(1, w)
    cy = (y + bh / 2.0) / max(1, h)
    return [cx, cy, bw / max(1, w), bh / max(1, h)]


def normalized_xywh_box(box: dict, w: int, h: int):
    x = float(box.get("x", 0) or 0)
    y = float(box.get("y", 0) or 0)
    bw = float(box.get("w", 0) or 0)
    bh = float(box.get("h", 0) or 0)
    if bw <= 1 or bh <= 1:
        return None
    return [
        max(0.0, min(1.0, x / max(1, w))),
        max(0.0, min(1.0, y / max(1, h))),
        max(0.0, min(1.0, bw / max(1, w))),
        max(0.0, min(1.0, bh / max(1, h))),
    ]


def box_from_points(points: dict, w: int, h: int):
    pos = points.get("positive") or []
    if not pos:
        return None
    xs = [float(p.get("x", 0) or 0) for p in pos]
    ys = [float(p.get("y", 0) or 0) for p in pos]
    pad = max(24.0, min(w, h) * 0.08)
    x1 = max(0.0, min(xs) - pad)
    y1 = max(0.0, min(ys) - pad)
    x2 = min(float(w), max(xs) + pad)
    y2 = min(float(h), max(ys) + pad)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return [(x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h, (x2 - x1) / w, (y2 - y1) / h]


def xywh_box_from_points(points: dict, w: int, h: int):
    pos = points.get("positive") or []
    if not pos:
        return None
    xs = [float(p.get("x", 0) or 0) for p in pos]
    ys = [float(p.get("y", 0) or 0) for p in pos]
    pad = max(24.0, min(w, h) * 0.08)
    x1 = max(0.0, min(xs) - pad)
    y1 = max(0.0, min(ys) - pad)
    x2 = min(float(w), max(xs) + pad)
    y2 = min(float(h), max(ys) + pad)
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]


def save_outputs(image: Image.Image, mask: Image.Image, data: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"sam3_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    mask_path = out_dir / f"{stamp}_mask.png"
    overlay_path = out_dir / f"{stamp}_overlay.png"
    cutout_path = out_dir / f"{stamp}_cutout.png"
    masked_path = out_dir / f"{stamp}_masked.png"

    feather = float(data.get("feather", 0) or 0)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    if str(data.get("invert", "false")).lower() in {"1", "true", "yes", "on"}:
        mask = ImageOps.invert(mask)

    mask.save(mask_path)
    rgba = image.convert("RGBA")
    red = Image.new("RGBA", image.size, (255, 40, 40, 120))
    overlay = Image.blend(rgba, Image.composite(red, rgba, mask), 0.55)
    overlay.save(overlay_path)
    cutout = rgba.copy()
    cutout.putalpha(mask)
    cutout.save(cutout_path)
    fill = str(data.get("mask_fill") or "black").lower()
    bg = Image.new("RGB", image.size, (255, 255, 255) if fill == "white" else (0, 0, 0))
    Image.composite(image.convert("RGB"), bg, mask).save(masked_path)
    return {
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "cutout_path": str(cutout_path),
        "masked_image_path": str(masked_path),
    }


def frame_index_for_video(path: str, frame_time, cap=None) -> tuple[int, float, int, int, int]:
    import cv2

    vid = cv2.VideoCapture(str(path))
    if not vid.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = sane_fps(vid.get(cv2.CAP_PROP_FPS) or 24.0)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    vid.release()
    try:
        t = float(frame_time or 0)
    except Exception:
        t = 0.0
    idx = int(round(max(0.0, t) * fps))
    limit = int(cap or 0)
    if limit > 0:
        total = min(total or limit, limit)
    if total > 0:
        idx = max(0, min(idx, total - 1))
    return idx, fps, total, width, height


def prepare_video_for_sam3(source_path: str, frame_cap: int, out_dir: Path) -> str:
    """SAM3 loads video frames into memory, so cap the source before session start."""
    if frame_cap <= 0:
        return source_path
    import imageio_ffmpeg
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    clipped = out_dir / f"sam3_work_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
    fps = sane_fps(frame_index_for_video(source_path, 0, None)[1])
    proc = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source_path), "-frames:v", str(frame_cap), "-an",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(clipped)
    ], capture_output=True, text=True)
    if proc.returncode != 0 or not clipped.exists() or clipped.stat().st_size <= 0:
        raise RuntimeError((proc.stderr or proc.stdout or "Could not prepare capped SAM3 video").strip())
    return str(clipped)


def output_to_mask(outputs, width: int, height: int) -> Image.Image:
    masks = None
    if isinstance(outputs, dict):
        masks = outputs.get("out_binary_masks")
        if masks is None:
            masks = outputs.get("masks")
    else:
        masks = getattr(outputs, "out_binary_masks", None) or getattr(outputs, "masks", None)
    arr = to_numpy_masks(masks)
    if arr.shape[0] == 0:
        return Image.new("L", (width, height), 0)
    combined = np.any(arr, axis=0)
    img = Image.fromarray((combined.astype(np.uint8) * 255), mode="L")
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.BILINEAR)
    return img


def write_masked_video(source_path: str, outputs_per_frame: dict, data: dict, out_dir: Path, fps: float, total: int, width: int, height: int) -> str:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"sam3_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    temp_path = out_dir / f"{stamp}_masked_video_raw.avi"
    out_path = out_dir / f"{stamp}_masked_video.mp4"
    writer_fps = sane_fps(fps)
    fill = str(data.get("mask_fill") or "black").lower()
    bg_value = 255 if fill == "white" else 0
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"MJPG"), writer_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Masked video writer could not start")
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open video: {source_path}")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if total and count >= total:
            break
        if count in outputs_per_frame:
            frame_mask = output_to_mask(outputs_per_frame[count], width, height)
        else:
            frame_mask = Image.new("L", (width, height), 0)
        mask = np.asarray(frame_mask, dtype=np.float32) / 255.0
        alpha = mask[..., None]
        bg = np.full((height, width, 3), bg_value, dtype=np.float32)
        writer.write((frame.astype(np.float32) * alpha + bg * (1.0 - alpha)).clip(0, 255).astype(np.uint8))
        count += 1
    cap.release()
    writer.release()
    if count <= 0 or not temp_path.exists() or temp_path.stat().st_size <= 0:
        raise RuntimeError("SAM3 masked video creation failed")
    import imageio_ffmpeg
    import subprocess

    proc = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp_path), "-r", str(writer_fps), "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)
    ], capture_output=True, text=True)
    try:
        temp_path.unlink(missing_ok=True)
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "Masked video encode failed").strip())
    return str(out_path)


def write_mask_video(outputs_per_frame: dict, data: dict, out_dir: Path, fps: float, total: int, width: int, height: int) -> str:
    """Write a Wan/Comfy-compatible mask video.

    WanAnimate's character_mask input is a MASK stream, so this is grayscale:
    white = selected character/subject, black = background. No alpha channel.
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"sam3_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    temp_path = out_dir / f"{stamp}_mask_video_raw.avi"
    out_path = out_dir / f"{stamp}_mask_video.mp4"
    writer_fps = sane_fps(fps)
    writer = cv2.VideoWriter(str(temp_path), cv2.VideoWriter_fourcc(*"MJPG"), writer_fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError("Mask video writer could not start")
    invert = str(data.get("invert", "false")).lower() in {"1", "true", "yes", "on"}
    count = 0
    for idx in range(max(0, int(total or 0))):
        if idx in outputs_per_frame:
            frame_mask = output_to_mask(outputs_per_frame[idx], width, height)
        else:
            frame_mask = Image.new("L", (width, height), 0)
        if invert:
            frame_mask = ImageOps.invert(frame_mask)
        mask = np.asarray(frame_mask, dtype=np.uint8)
        frame = np.repeat(mask[..., None], 3, axis=2)
        writer.write(frame)
        count += 1
    writer.release()
    if count <= 0 or not temp_path.exists() or temp_path.stat().st_size <= 0:
        raise RuntimeError("SAM3 mask video creation failed")
    import imageio_ffmpeg
    import subprocess

    proc = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(temp_path), "-r", str(writer_fps), "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out_path)
    ], capture_output=True, text=True)
    try:
        temp_path.unlink(missing_ok=True)
    except Exception:
        pass
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "Mask video encode failed").strip())
    return str(out_path)


def run_video_sam3(data: dict, out_dir: Path) -> dict:
    import torch
    from sam3.model_builder import build_sam3_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 video segmentation requires CUDA in the local SAM3 environment.")
    source_path = data["source_path"]
    frame_cap = int(data.get("video_frame_cap") or 0)
    work_source_path = prepare_video_for_sam3(source_path, frame_cap, out_dir)
    image = Image.open(data["frame_path"]).convert("RGB")
    w, h = image.size
    text_prompt = (data.get("text_prompt") or "").strip()
    points = data.get("points") or {"positive": [], "negative": []}
    box = data.get("box") or {}
    frame_idx, fps, total, video_w, video_h = frame_index_for_video(work_source_path, data.get("frame_time", 0), frame_cap)
    video_w = video_w or w
    video_h = video_h or h

    prompt = {"type": "add_prompt", "frame_index": frame_idx}
    if text_prompt:
        prompt["text"] = text_prompt
    geom = normalized_xywh_box(box, w, h)
    if geom is None and points.get("positive"):
        geom = xywh_box_from_points(points, w, h)
    if geom is not None:
        prompt["bounding_boxes"] = [geom]
        prompt["bounding_box_labels"] = [1]
    if not text_prompt and geom is None:
        raise RuntimeError("SAM3 video needs a text prompt, a box, or positive points.")

    predictor = build_sam3_video_predictor()
    session_id = None
    outputs_per_frame = {}
    try:
        response = predictor.handle_request({"type": "start_session", "resource_path": work_source_path})
        session_id = response["session_id"]
        prompt["session_id"] = session_id
        response = predictor.handle_request(prompt)
        prompted_outputs = response.get("outputs", {})
        outputs_per_frame[int(response.get("frame_index", frame_idx))] = prompted_outputs
        stream_request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both",
            "start_frame_index": frame_idx,
        }
        if total:
            stream_request["max_frame_num_to_track"] = total
        for item in predictor.handle_stream_request(stream_request):
            outputs_per_frame[int(item["frame_index"])] = item.get("outputs", {})
    finally:
        if session_id:
            try:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
            except Exception:
                pass

    expected_frames = total or 0
    if expected_frames and len(outputs_per_frame) <= 1:
        raise RuntimeError(
            "SAM3 video propagation returned only the prompt frame. "
            "Refusing to create a static-mask video; lower the frame cap or check the SAM3 video predictor setup."
        )
    first_outputs = outputs_per_frame.get(frame_idx) or next(iter(outputs_per_frame.values()), prompted_outputs)
    first_mask = output_to_mask(first_outputs, w, h)
    result = save_outputs(image, first_mask, data, out_dir)
    result["mask_video_path"] = write_mask_video(outputs_per_frame, data, out_dir, fps, total, video_w, video_h)
    result["masked_video_path"] = write_masked_video(work_source_path, outputs_per_frame, data, out_dir, fps, total, video_w, video_h)
    result.update({
        "backend_used": "local_sam3_video",
        "is_real_sam3": True,
        "mask_count": int(len(outputs_per_frame)),
        "text_prompt": text_prompt,
        "video_frame_count": int(len(outputs_per_frame)),
        "prompt_frame_index": int(frame_idx),
        "sam3_work_source_path": work_source_path if work_source_path != source_path else "",
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = load_json(args.input)

    try:
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as exc:
        raise RuntimeError(
            "Could not import Meta SAM3. Install facebookresearch/sam3 in this Python environment, "
            "install PyTorch with CUDA, and authenticate Hugging Face access to the SAM3 checkpoint."
        ) from exc

    image = Image.open(data["frame_path"]).convert("RGB")
    w, h = image.size
    text_prompt = (data.get("text_prompt") or "").strip()
    prompt_mode = data.get("prompt_mode") or "text"
    points = data.get("points") or {"positive": [], "negative": []}
    box = data.get("box") or {}
    score_threshold = float(data.get("score_threshold", 0.0) or 0.0)

    if data.get("source_kind") == "video" and data.get("make_masked_video"):
        result = run_video_sam3(data, Path(data.get("output_dir") or Path(args.output).parent))
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(device=device)
    model.eval()
    processor = Sam3Processor(model)

    autocast_enabled = device == "cuda"
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            state = processor.set_image(image)

            output = None
            if text_prompt and prompt_mode in {"text", "text_points", "text_box", "all"}:
                output = processor.set_text_prompt(state=state, prompt=text_prompt)
            elif text_prompt:
                output = processor.set_text_prompt(state=state, prompt=text_prompt)

            geom = normalized_box(box, w, h)
            if geom is None and points.get("positive"):
                geom = box_from_points(points, w, h)
            if geom is not None and hasattr(processor, "add_geometric_prompt"):
                output = processor.add_geometric_prompt(box=geom, label=True, state=state)
            if output is None:
                if not text_prompt:
                    raise RuntimeError("SAM3 needs a text prompt, a box, or positive points to create a mask.")
                output = processor.set_text_prompt(state=state, prompt=text_prompt)

    masks = to_numpy_masks(output.get("masks") if isinstance(output, dict) else getattr(output, "masks", None))
    scores = output.get("scores") if isinstance(output, dict) else getattr(output, "scores", None)
    if masks.shape[0] == 0:
        raise RuntimeError(f"SAM3 found no masks for prompt: {text_prompt or 'visual prompt'}")
    keep = scores_mask(scores, masks.shape[0], score_threshold)
    if keep.any():
        masks = masks[keep]
    combined = np.any(masks, axis=0)
    if combined.shape != (h, w):
        combined_img = Image.fromarray((combined.astype(np.uint8) * 255), mode="L").resize((w, h), Image.Resampling.BILINEAR)
    else:
        combined_img = Image.fromarray((combined.astype(np.uint8) * 255), mode="L")

    result = save_outputs(image, combined_img, data, Path(data.get("output_dir") or Path(args.output).parent))
    result.update({
        "backend_used": "local_sam3",
        "is_real_sam3": True,
        "mask_count": int(masks.shape[0]),
        "text_prompt": text_prompt,
    })
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
