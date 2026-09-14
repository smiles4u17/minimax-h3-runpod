import asyncio
from unittest import mock
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
import app


class SAMimateH3Tests(unittest.TestCase):
    def test_mask_cache_reuses_and_invalidates_source_and_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.mp4';source.write_bytes(b'video one')
            call_count = []
            def prepare(data):
                n = len(call_count)
                a = root / f'source{n}.mkv';b = root / f'plate{n}.mkv'
                a.write_bytes(b'source');b.write_bytes(b'plate')
                return {'source_path':str(a),'composite_source_path':str(b),'frame_count':48,'duration':2}
            async def segment(data):
                call_count.append(data)
                folder = Path(data['output_dir'])
                mask=folder/'mask.mp4';inv=folder/'inv.mp4';mask.write_bytes(b'mask');inv.write_bytes(b'inv')
                return {'mask_video_path':str(mask),'inverted_mask_video_path':str(inv)}
            data={'video_path':str(source),'video_start':'0','segmentation':{'text_prompt':'person'}}
            with (mock.patch.object(app,'samimate_mask_root',return_value=root),
                  mock.patch.object(app,'local_probe_video_path',side_effect=lambda p:p),
                  mock.patch.object(app,'prepare_samimate_h3',side_effect=prepare),
                  mock.patch.object(app,'sam_run',side_effect=segment)):
                a=asyncio.run(app.prepare_samimate_masks(data))
                b=asyncio.run(app.prepare_samimate_masks(dict(data,prompt='new prompt',steps=8,width=1024)))
                self.assertTrue(b['reused']);self.assertEqual(a['key'],b['key']);self.assertEqual(len(call_count),1)
                source.write_bytes(b'video two')
                c=asyncio.run(app.prepare_samimate_masks(data));self.assertFalse(c['reused'])
                d=asyncio.run(app.prepare_samimate_masks(dict(data,segmentation={'text_prompt':'face'})))
                self.assertNotEqual(c['key'],d['key']);self.assertEqual(len(call_count),3)
                Path(d['masks']['mask_video_path']).unlink()
                self.assertIsNone(app.samimate_cached_masks(d['key']))

    def test_real_mask_pair_and_cached_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'neutral.mkv';cache=root/'cache';cache.mkdir();sam=root/'sam';sam.mkdir()
            subprocess.run([app.ffmpeg_bin(),'-y','-v','error','-f','lavfi','-i','testsrc2=size=96x64:rate=24',
                            '-t','2','-c:v','ffv1',str(source)],check=True,capture_output=True)
            request={'video_path':str(source),'segmentation':{'backend':'local_test','text_prompt':'person'}}
            with (mock.patch.object(app,'samimate_mask_root',return_value=cache),
                  mock.patch.object(app,'TEMP_DIR',root),mock.patch.object(app,'SAM_DIR',sam),
                  mock.patch.object(app,'sam_record')):
                result=asyncio.run(app.prepare_samimate_masks(request))
                again=asyncio.run(app.prepare_samimate_masks(request))
                self.assertTrue(again['reused'])
                for key in ('mask_video_path','inverted_mask_video_path'):
                    self.assertEqual(app.samimate_frame_count(result['masks'][key]),48)
                    self.assertAlmostEqual(app.video_probe(result['masks'][key])['fps'],24)
                def first_frame(path):
                    return np.frombuffer(subprocess.check_output([app.ffmpeg_bin(),'-v','error','-i',path,
                          '-frames:v','1','-f','rawvideo','-pix_fmt','gray','-']),dtype=np.uint8).astype(int)
                total=first_frame(result['masks']['mask_video_path'])+first_frame(result['masks']['inverted_mask_video_path'])
                self.assertLess(np.mean(np.abs(total-255)),3)

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
