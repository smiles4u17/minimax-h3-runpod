import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
import app


class SAMimateH3Tests(unittest.TestCase):
    def test_prompt_groups_all_images_as_one_identity(self):
        prompt = app.samimate_h3_prompt(['a.png', 'b.png', 'c.png'], 'person on the left', 'Keep the red jacket')
        self.assertIn('<Picture 1>, <Picture 2>, <Picture 3>', prompt)
        self.assertNotIn('<Subject 2>', prompt)
        self.assertNotIn('<Video 1>', prompt)
        self.assertIn('person on the left', prompt)
        self.assertIn('Keep the red jacket', prompt)
        for n in (0, 10):
            with self.assertRaises(ValueError):
                app.samimate_h3_prompt(['x'] * n, 'person')

    def test_real_preparation_and_lossless_composite_preserve_background_and_audio(self):
        for fps in (24, 30):
            with self.subTest(source_fps=fps):
                self.check_composite(fps)

    def check_composite(self, fps):
        with tempfile.TemporaryDirectory(dir=app.TEMP_DIR) as tmp:
            root = Path(tmp)
            def make(name, source, audio=False):
                path = root / name
                cmd = [app.ffmpeg_bin(), '-y', '-v', 'error', '-f', 'lavfi', '-i', source]
                if audio:
                    cmd += ['-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000']
                cmd += ['-t', '2', '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(path)]
                subprocess.run(cmd, check=True, capture_output=True)
                return str(path)
            source = make('source.mkv', f'testsrc2=size=96x64:rate={fps}', True)
            foreground = make('foreground.mkv', 'color=red:size=96x64:rate=24')
            mask = make('mask.mkv', 'color=black:size=96x64:rate=24,drawbox=x=32:y=0:w=32:h=64:color=white:t=fill')
            prepared = app.prepare_samimate_h3({'video_path': source, 'video_start': 0.5, 'video_end': 1.5, 'width': 96, 'height': 64})
            self.assertEqual(prepared['frame_count'], 24)
            master, preview = app.composite_h3_subject(foreground, prepared['composite_source_path'], mask, root)
            def frames(path):
                raw = subprocess.check_output([app.ffmpeg_bin(), '-v', 'error', '-i', path, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'])
                return np.frombuffer(raw, dtype=np.uint8).reshape(-1, 64, 96, 3)
            original, result = frames(prepared['composite_source_path']), frames(master)
            self.assertEqual(len(result), fps)
            np.testing.assert_array_equal(result[:, :, :32], original[:, :, :32])
            np.testing.assert_array_equal(result[:, :, 64:], original[:, :, 64:])
            self.assertGreater(int(result[0, 20, 48, 0]), 200)
            def audio(path):
                return subprocess.check_output([app.ffmpeg_bin(), '-v', 'error', '-i', path, '-map', '0:a:0', '-f', 's16le', '-'])
            self.assertEqual(audio(master), audio(prepared['composite_source_path']))
            self.assertTrue(Path(preview).is_file())
            Path(prepared['source_path']).unlink()
            Path(prepared['composite_source_path']).unlink()


if __name__ == '__main__':
    unittest.main()
