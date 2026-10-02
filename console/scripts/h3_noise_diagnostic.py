"""Explicitly submit small, neutral Ref2V comparisons through the console."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prompt_book_h3


def fixtures(duration=2, reference_mp=0.2):
    folder = ROOT / 'cache' / 'h3_noise_diagnosis'
    folder.mkdir(parents=True, exist_ok=True)
    still = folder / f'neutral_reference_{reference_mp:g}mp.png'
    video = folder / f'neutral_reference_{reference_mp:g}mp_{duration:g}s.mp4'
    scale = (reference_mp / 0.2) ** 0.5
    width, height = (max(32, round(axis * scale / 32) * 32) for axis in (608, 352))
    if not still.exists():
        image = Image.new('RGB', (608, 352), '#cad3df')
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 240, 608, 352), fill='#ece5d4')
        draw.ellipse((100, 253, 262, 291), fill='#96958e')
        draw.polygon([(125, 160), (221, 160), (243, 139), (148, 139)], fill='#ee7974')
        draw.polygon([(221, 160), (243, 139), (243, 240), (221, 262)], fill='#9b3538')
        draw.rectangle((125, 160, 221, 262), fill='#d54348')
        draw.ellipse((365, 250, 510, 280), fill='#96958e')
        draw.ellipse((371, 153, 487, 269), fill='#2872b4')
        draw.ellipse((384, 164, 419, 192), fill='#9fcff2')
        image.resize((width, height)).save(still)
    if not video.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                        '-loop', '1', '-i', str(still), '-t', str(duration), '-r', '24',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(video)], check=True)
    return folder, still, video, width, height


def submit(label, lora, strength, latent, duration, reference_mp=0.2,
           steps=8, split=6, sigma_choice=1, attention='sage',
           source_duration=None, trim_start=0, reference_length=None):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', label):
        raise ValueError('Use a short alphanumeric test label, not a path')
    if not 1 <= duration <= 15 or not 0.2 <= reference_mp <= 2:
        raise ValueError('Use a 1-15 second output and 0.2-2 MP reference')
    source_duration = source_duration or duration
    folder, still, video, width, height = fixtures(source_duration, reference_mp)
    catalog = {'subjects': [{'id': 'neutral', 'label': 'Neutral test', 'path': str(still)}],
               'videos': [{'id': 'neutral', 'label': 'Neutral test', 'file': str(video),
                           'duration_sec': source_duration, 'width': width, 'height': height}]}
    request, _ = prompt_book_h3.build_prompt_book_h3_request(
        {'subject_id': 'neutral', 'video_id': 'neutral', 'duration': duration,
         'trim_start': trim_start,
         'trim_end': trim_start + (reference_length or duration),
         'steps': steps, 'sampler': 'euler', 'scheduler': 'simple', 'seed_random': False,
         'seed': 3907608799868968, 'attention': attention, 'cache_enabled': False,
         'megapixels': 0.2, 'latent_upscale': latent, 'final_megapixels': 0.9,
         'rtx_upscale': False, 'pass1_split': split, 'second_pass_sigma': sigma_choice,
         'diagnostic_frames': True,
         'loras': [{'name': lora, 'strength': strength}] if lora else []},
        catalog=catalog,
        baked={'prompt': 'A red cube and a blue sphere sit on a pale tabletop against a light blue wall. '
                         'Preserve the objects and composition from <Picture 1> and <Video 1>. '
                         'The camera stays fixed. Soft daylight, a still life with stable shapes.'})
    request['filename_prefix'] = 'H3_Diagnostic_' + label
    response = requests.post('http://127.0.0.1:8765/api/run/h3', json=request, timeout=180)
    response.raise_for_status()
    result = response.json()
    job_id = result['job']['id']
    manifest = folder / (label + '.json')
    manifest.write_text(json.dumps({'label': label, 'job_id': job_id,
                                   'endpoint_id': result['endpoint_id'],
                                   'request': request, 'debug': result['debug']}, indent=2), encoding='utf-8')
    print(json.dumps({'label': label, 'job_id': job_id, 'latent': latent,
                      'lora': lora, 'strength': strength, 'manifest': str(manifest)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', required=True, help='Test label; submits a paid GPU job')
    parser.add_argument('--lora', default='')
    parser.add_argument('--strength', type=float, default=1)
    parser.add_argument('--latent', action='store_true')
    parser.add_argument('--duration', type=float, default=2)
    parser.add_argument('--reference-mp', type=float, default=0.2)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--split', type=int, default=6)
    parser.add_argument('--sigma-choice', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--attention', choices=['sage', 'native'], default='sage')
    parser.add_argument('--source-duration', type=float)
    parser.add_argument('--trim-start', type=float, default=0)
    parser.add_argument('--reference-length', type=float)
    args = parser.parse_args()
    submit(args.submit, args.lora, args.strength, args.latent, args.duration,
           args.reference_mp, args.steps, args.split, args.sigma_choice, args.attention,
           args.source_duration, args.trim_start, args.reference_length)
