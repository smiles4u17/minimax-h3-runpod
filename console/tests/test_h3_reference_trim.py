"""Actual media trimming checks with synthetic colors and non-speech tones."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import app


class ReferenceTrimTests(unittest.TestCase):
    def test_console_preserves_reference_cap(self):
        result = app.h3_reference_video_settings([{'start_frame': 48, 'frame_load_cap': 48}], 1, 15)
        self.assertEqual(result[0]['frame_load_cap'], 48)
        self.assertEqual(result[0]['skip_first_frames'], 48)
        self.assertEqual(app.h3_reference_video_settings([], 1, 2)[0]['frame_load_cap'], 56)
        for cap in (0, -1, 2.5, True, 'inf', 3601):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                app.h3_reference_video_settings([{'frame_load_cap': cap}], 1, 15)

    def test_preview_does_not_trim_or_touch_media(self):
        token = app.H3_PREVIEW.set(True)
        try:
            with mock.patch.object(app.subprocess, 'run', side_effect=AssertionError('No preview IO')):
                path, options, debug = app.h3_prepare_reference_clip('not-real.mp4', {
                    'force_rate': 24, 'skip_first_frames': 48,
                    'select_every_nth': 1, 'frame_load_cap': 48})
            self.assertEqual(path, 'not-real.mp4')
            self.assertEqual(options['skip_first_frames'], 48)
            self.assertFalse(debug['pretrimmed'])
        finally:
            app.H3_PREVIEW.reset(token)

    def test_physical_trim_excludes_both_outside_video_and_audio(self):
        app.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=app.TEMP_DIR) as directory:
            folder = Path(directory)
            source = folder / 'three_segments.mp4'
            command = [app.ffmpeg_bin(), '-hide_banner', '-loglevel', 'error', '-y']
            for color, frequency in [('red', 200), ('green', 400), ('blue', 800)]:
                command += ['-f', 'lavfi', '-i', f'color=c={color}:s=64x64:r=30:d=2',
                            '-f', 'lavfi', '-i', f'sine=frequency={frequency}:duration=2']
            command += ['-filter_complex', '[0:v][1:a][2:v][3:a][4:v][5:a]concat=n=3:v=1:a=1[v][a]',
                        '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', str(source)]
            subprocess.run(command, capture_output=True, check=True, timeout=60)
            options = {'force_rate': 24, 'skip_first_frames': 48, 'select_every_nth': 1, 'frame_load_cap': 48}
            with mock.patch.object(app, 'TEMP_DIR', folder):
                clip, normalized, debug = app.h3_prepare_reference_clip(str(source), options)
                again, _, _ = app.h3_prepare_reference_clip(str(source), options)
            self.assertEqual(clip, again)
            self.assertEqual(normalized['skip_first_frames'], 0)
            self.assertTrue(debug['pretrimmed'])
            self.assertAlmostEqual(debug['uploaded_duration_seconds'], 2, delta=0.05)
            video = subprocess.run([app.ffmpeg_bin(), '-v', 'error', '-i', clip,
                '-map', '0:v:0', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True)
            pixels = np.frombuffer(video.stdout, np.uint8).reshape(-1, 64, 64, 3)
            self.assertEqual(len(pixels), 48)
            self.assertGreater(pixels[..., 1].mean(), 100)
            self.assertLess(pixels[..., [0, 2]].mean(), 10)
            audio = subprocess.run([app.ffmpeg_bin(), '-v', 'error', '-i', clip,
                '-map', '0:a:0', '-f', 'f32le', '-ac', '1', '-ar', '8000', '-'], capture_output=True, check=True)
            samples = np.frombuffer(audio.stdout, '<f4')
            self.assertGreater(len(samples), 15000)
            self.assertLessEqual(len(samples), 16400)
            peak = np.fft.rfftfreq(len(samples), 1 / 8000)[np.abs(np.fft.rfft(samples)).argmax()]
            self.assertAlmostEqual(peak, 400, delta=2)
