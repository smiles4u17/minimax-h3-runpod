import base64
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import imageio_ffmpeg
import grok_media
from grok_enhancer import enhance_prompt


class GrokMediaTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def image(self):
        path = self.root / 'picture.png'
        Image.new('RGB', (40, 30), 'blue').save(path)
        return path

    def test_active_slots_preserve_labels_and_ignore_hidden_references(self):
        data = {'task': 'fl2v', 'photo_paths': ['', 'last.png', 'unused.png'],
                'reference_video_paths': ['stale.mp4']}
        result = grok_media.attachments(data)
        self.assertEqual([(x['label'], x['path']) for x in result], [('<Picture 2>', 'last.png')])
        self.assertEqual(grok_media.attachments(dict(data, task='t2v')), [])
        self.assertEqual(grok_media.attachments(dict(data, use_attached_media=False)), [])
        self.assertEqual(len(grok_media.attachments(dict(data, use_multi_image=True))), 2)

    def test_missing_media_does_not_drop_available_image(self):
        media = grok_media.prepare({'task': 'fl2v', 'photo_paths': [str(self.image()), 'missing.png']},
                                  {}, str, 'unused', lambda _: {})
        self.assertEqual(media['reviewed'], ['<Picture 1>: image'])
        self.assertIn('<Picture 2>', media['warnings'][0])
        payload = media['content'][1]
        raw = base64.b64decode(payload['image_url'].split(',')[1])
        with Image.open(io.BytesIO(raw)) as decoded:
            self.assertEqual(decoded.size, (40, 30))
            self.assertEqual(decoded.format, 'JPEG')

    def test_selected_clip_window_uses_frame_start_stride_and_cap(self):
        item = {'options': {'force_rate': 24, 'start_frame': 48, 'frame_load_cap': 24, 'select_every_nth': 2}}
        self.assertEqual(grok_media.clip_window(item, {'duration_seconds': 20}, 7), (2, 2))
        self.assertEqual(grok_media.clip_window(item, {'duration_seconds': 3}, 7), (2, 1))

    def test_real_video_frames_and_audio_clip_are_prepared_without_provider_calls(self):
        binary = imageio_ffmpeg.get_ffmpeg_exe()
        video = self.root / 'reference.mp4'
        subprocess.run([binary, '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=64x64:r=24',
                        '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=16000', '-t', '2',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(video)],
                       check=True, capture_output=True, timeout=30)
        captured = []
        def post(*args, **kwargs):
            captured.append(kwargs['files']['file'][1].read())
            return Mock(status_code=200, json=lambda: {'text': 'Test transcript.'})
        with patch('grok_media.requests.post', side_effect=post) as request:
            media = grok_media.prepare({'task': 'r2v', 'reference_video_paths': [str(video)],
                'reference_video_audio': [True], 'reference_video_settings': [
                    {'force_rate': 24, 'start_frame': 12, 'frame_load_cap': 24}]},
                {'grok_api_key': 'private-key'}, str, binary, lambda _: {'duration_seconds': 2, 'fps': 24})
        self.assertEqual(len([x for x in media['content'] if x['type'] == 'input_image']), 6)
        self.assertIn('0.50–1.50s', media['reviewed'][0])
        self.assertEqual(media['warnings'], [])
        self.assertIn('speech transcription', media['reviewed'][1])
        self.assertTrue(captured[0].startswith(b'RIFF'))
        self.assertLess(len(captured[0]), 40000)
        request.assert_called_once()
        self.assertEqual(request.call_args.args[0], 'https://api.x.ai/v1/stt')

    def test_audio_failure_is_reported_not_fabricated(self):
        path = self.root / 'reference.wav'
        path.write_bytes(b'fixture')
        with patch('grok_media.transcribe', side_effect=ValueError('private provider body')):
            result = grok_media.prepare({'task': 'r2v', 'reference_audio_paths': [str(path)]},
                                       {}, str, 'unused', lambda _: {})
        self.assertEqual(result['reviewed'], [])
        self.assertIn('audio not reviewed', result['warnings'][0])
        self.assertNotIn('private', str(result))

    def test_validated_request_passes_multimodal_content_and_reports_coverage(self):
        response = Mock(status_code=200, json=lambda: {'status': 'completed', 'output': [
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'A blue robot.'}]}]})
        prepare = Mock(return_value={'content': [{'type': 'input_image', 'image_url': 'data:image/jpeg;base64,YQ=='}],
                                      'reviewed': ['<Picture 1>: image'], 'warnings': ['<Audio 1>: unavailable']})
        with patch('grok_enhancer.requests.post', return_value=response) as post:
            result = enhance_prompt({'prompt': 'Robot'}, {'grok_api_key': 'key'}, prepare)
        body = post.call_args.kwargs['json']
        self.assertEqual(body['input'][0]['content'][1]['type'], 'input_image')
        self.assertEqual(result['media_reviewed'], ['<Picture 1>: image'])
        self.assertEqual(result['media_warnings'], ['<Audio 1>: unavailable'])
        self.assertFalse(body['store'])
        prepare.reset_mock()
        with self.assertRaises(ValueError):
            enhance_prompt({'prompt': 'Robot'}, {}, prepare)
        prepare.assert_not_called()

    def test_resolver_failure_is_skipped_without_exposing_details(self):
        resolve = Mock(side_effect=RuntimeError('credentials/internal path'))
        result = grok_media.prepare({'photo_paths': ['private.png']}, {}, resolve, 'unused', lambda _: {})
        self.assertEqual(result['reviewed'], [])
        self.assertNotIn('credentials', str(result))


if __name__ == '__main__':
    unittest.main()
