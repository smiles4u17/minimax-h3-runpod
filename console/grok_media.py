"""Bounded, labelled attachment context for the Grok prompt enhancer."""
import base64
import io
import math
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageOps
import requests

MAX_IMAGE_BYTES = 16 * 1024 * 1024
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tif', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.oga', '.weba', '.aac', '.opus', '.wma'}


def attachments(data):
    """Use the same active task/slots as H3; do not send stale hidden inputs."""
    if data.get('use_attached_media') is False or data.get('task') == 't2v':
        return []
    ref = str(data.get('task', 'fl2v')).startswith(('r2v', 'ref2v'))
    photos = data.get('photo_paths')
    if photos is None:
        photos = data.get('reference_paths', []) if ref else [data.get('first_frame_path'), data.get('last_frame_path')]
    if not isinstance(photos, list) or len(photos) > 9:
        raise ValueError('Enhancer images must be a list of at most nine paths.')
    result = []
    if not ref and not data.get('use_multi_image'):
        photos = photos[:2]
    for index, path in enumerate(photos):
        if path:
            result.append({'kind': 'image', 'label': f'<Picture {index + 1}>', 'path': path})
    if not ref:
        # Photos 3/4 are only active when percentage keyframes are enabled.
        return result
    for field, kind, label in [('reference_video_paths', 'video', 'Video'), ('reference_audio_paths', 'audio', 'Audio')]:
        values = data.get(field) or []
        if not isinstance(values, list) or len(values) > 3:
            raise ValueError(f'Enhancer {kind} references must be a list of at most three paths.')
        for index, path in enumerate(values):
            if path:
                options = (data.get('reference_video_settings') or [])
                audio_flags = data.get('reference_video_audio') or []
                if not isinstance(options, list) or not isinstance(audio_flags, list):
                    raise ValueError('Video attachment settings must be lists.')
                if kind == 'video' and index < len(options) and not isinstance(options[index], dict):
                    raise ValueError('Video attachment options must be objects.')
                result.append({'kind': kind, 'label': f'<{label} {index + 1}>', 'path': path,
                               'options': options[index] if kind == 'video' and index < len(options) else {},
                               'audio': kind == 'audio' or (index < len(audio_flags) and audio_flags[index] is True)})
    return result


def image_part(raw):
    with Image.open(io.BytesIO(raw)) as source:
        if source.width * source.height > 40000000:
            raise ValueError('Image dimensions too large')
        image = ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((1024, 1024))
        output = io.BytesIO()
        image.save(output, 'JPEG', quality=85)
    return {'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode('ascii')}


def clip_window(item, info, duration):
    options = item.get('options') or {}
    fps = float(options.get('force_rate') or info.get('fps') or 24)
    start = float(options.get('skip_first_frames', options.get('start_frame', 0))) / fps
    stride = float(options.get('select_every_nth') or 1)
    span = float(options.get('frame_load_cap') or (duration * fps)) * stride / fps
    if not all(math.isfinite(x) for x in (fps, start, span, stride)) or fps <= 0 or start < 0 or span <= 0 or stride < 1:
        raise ValueError('Invalid reference clip window')
    total = float(info.get('duration_seconds') or 0)
    if total:
        span = min(span, total - start)
    if span <= 0:
        raise ValueError('Reference starts beyond end of clip')
    return start, min(span, 60.)


def ffmpeg_run(binary, path, options):
    result = subprocess.run([binary, '-nostdin', '-v', 'error', '-threads', '1',
                             '-protocol_whitelist', 'file,pipe', *options[:2], '-i', str(path), *options[2:]],
                            capture_output=True, timeout=30)
    if result.returncode:
        raise ValueError('Media decoding failed')
    return result.stdout


def transcribe(binary, path, start, span, key):
    with tempfile.TemporaryDirectory(prefix='grok-audio-') as folder:
        output = Path(folder) / 'reference.wav'
        ffmpeg_run(binary, path, ['-ss', str(start), '-t', str(span), '-vn', '-ac', '1', '-ar', '16000', '-y', str(output)])
        with output.open('rb') as audio:
            response = requests.post('https://api.x.ai/v1/stt', headers={'Authorization': f'Bearer {key}'},
                                     data={'model': 'grok-voice-transcribe-2.0'},
                                     files={'file': ('reference.wav', audio, 'audio/wav')},
                                     timeout=(10, 90), allow_redirects=False)
        if response.status_code != 200:
            raise ValueError('Audio transcription unavailable')
        text = response.json().get('text')
        if not isinstance(text, str):
            raise ValueError('Audio transcription unavailable')
        return text[:12000]


def prepare(data, settings, resolve, ffmpeg, probe):
    content, reviewed, warnings = [], [], []
    budget = 0
    try:
        duration = float(data.get('duration') or 7)
    except (TypeError, ValueError):
        raise ValueError('Invalid enhancer duration') from None
    if not math.isfinite(duration) or not 0 < duration <= 60:
        raise ValueError('Enhancer duration must be between 0 and 60 seconds.')

    def add_image(part, label):
        nonlocal budget
        size = len(part['image_url'])
        if budget + size > MAX_IMAGE_BYTES:
            raise ValueError('Image budget exceeded')
        budget += size
        content.extend([{'type': 'input_text', 'text': label}, part])

    for item in attachments(data):
        label, kind = item['label'], item['kind']
        try:
            source = item['path']
            if not isinstance(source, str) or len(source) > 8192:
                raise ValueError('Invalid attachment path')
            try:
                path = Path(resolve(source))
            except Exception:
                warnings.append(f'{label}: attachment unavailable; not reviewed.')
                continue
            suffixes = IMAGE_EXTENSIONS if kind == 'image' else VIDEO_EXTENSIONS if kind == 'video' else AUDIO_EXTENSIONS
            if not path.is_file() or path.suffix.lower() not in suffixes:
                raise ValueError('Attachment unavailable or too large')
            if kind == 'image':
                if path.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError('Image too large')
                add_image(image_part(path.read_bytes()), label + ' — attached image')
                reviewed.append(label + ': image')
                continue
            start, span = clip_window(item, probe(path), duration)
            if kind == 'video':
                count = 0
                for index in range(6):
                    second = start + max(0., span - 0.1) * index / 5
                    try:
                        raw = ffmpeg_run(ffmpeg, path, ['-ss', str(second), '-frames:v', '1', '-vf',
                            'scale=1024:1024:force_original_aspect_ratio=decrease', '-f', 'image2pipe', '-vcodec', 'mjpeg', 'pipe:1'])
                        add_image(image_part(raw), f'{label} — sampled frame at source {second:.2f}s')
                        count += 1
                    except (ValueError, OSError, subprocess.SubprocessError):
                        continue
                if count:
                    reviewed.append(f'{label}: {count} sampled frames ({start:.2f}–{start + span:.2f}s)')
                else:
                    warnings.append(f'{label}: video frames unavailable; not visually reviewed.')
            if item.get('audio'):
                try:
                    text = transcribe(ffmpeg, path, start, span, str(settings.get('grok_api_key') or ''))
                    content.append({'type': 'input_text', 'text': f'{label} — speech transcript ({start:.2f}–{start + span:.2f}s):\n{text or "No speech recognized."}\nTranscription does not describe music, ambient sound, or non-speech vocalizations.'})
                    reviewed.append(label + ': speech transcription')
                except (ValueError, OSError, requests.RequestException, subprocess.SubprocessError):
                    warnings.append(f'{label}: audio transcription unavailable; audio not reviewed.')
        except (ValueError, TypeError, OSError, requests.RequestException, subprocess.SubprocessError, Image.DecompressionBombError):
            warnings.append(f'{label}: attachment unavailable; not reviewed.')
    return {'content': content, 'reviewed': reviewed, 'warnings': warnings}
