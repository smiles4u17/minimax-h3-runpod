import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests
from fastapi import HTTPException
import app


class GenerationLogTests(unittest.TestCase):
    def test_request_saved_before_submit_and_job_receipt_is_linked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {'prompt': 'A quiet landscape.', 'seed': 123,
                       'reference_videos': [{'base64': 'c29tZW1lZGlh'}]}
            source_token = app.SUBMISSION_SOURCE.set({'reference_video_paths': ['D:/clip.mp4']})
            def send(*args, **kwargs):
                saved = list(root.glob('*_request.json'))
                self.assertEqual(len(saved), 1)
                self.assertEqual(json.loads(saved[0].read_text()), kwargs['json'])
                response = mock.Mock()
                response.json.return_value = {'id': 'job-123', 'status': 'IN_QUEUE'}
                return response
            try:
                with mock.patch.object(app, 'GENERATION_LOG_DIR', root), mock.patch.object(app.requests, 'post', side_effect=send):
                    job = app.submit('endpoint', 'private-api-key', payload, {})
            finally:
                app.SUBMISSION_SOURCE.reset(source_token)
            self.assertEqual(job['id'], 'job-123')
            text = next(root.glob('*.txt')).read_text()
            for value in ('A quiet landscape.', '123', 'job-123', 'D:/clip.mp4', 'SUBMITTED'):
                self.assertIn(value, text)
            self.assertNotIn('c29tZW1lZGlh', text)
            self.assertNotIn('private-api-key', ''.join(p.read_text() for p in root.iterdir()))
            self.assertEqual(json.loads(next(root.glob('*_receipt.json')).read_text())['job'], job)

    def test_failed_archive_prevents_paid_submission(self):
        with mock.patch.object(app, 'SubmissionLog', side_effect=OSError('disk full')), mock.patch.object(app.requests, 'post') as send:
            with self.assertRaises(HTTPException) as caught:
                app.submit('endpoint', 'key', {'prompt': 'Landscape'}, {})
            self.assertEqual(caught.exception.status_code, 507)
            send.assert_not_called()

    def test_timeout_keeps_inputs_without_claiming_definite_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(app, 'GENERATION_LOG_DIR', root), mock.patch.object(app.requests, 'post', side_effect=requests.Timeout('connection timed out')):
                with self.assertRaises(HTTPException):
                    app.submit('endpoint', 'key', {'prompt': 'Landscape', 'seed': 42}, {})
            receipt = json.loads(next(root.glob('*_receipt.json')).read_text())
            self.assertEqual(receipt['status'], 'SUBMIT_UNCONFIRMED')
            self.assertEqual(json.loads(next(root.glob('*_request.json')).read_text())['input']['seed'], 42)


if __name__ == '__main__':
    unittest.main()
