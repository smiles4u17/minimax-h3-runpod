import asyncio
import threading
import unittest
from unittest import mock
import app


class AsyncSubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_wait_keeps_event_loop_live_and_preserves_source_context(self):
        entered, release = threading.Event(), threading.Event()
        sources = []
        def submit(*args):
            sources.append(app.SUBMISSION_SOURCE.get())
            entered.set()
            if not release.wait(2):
                raise AssertionError('Provider wait blocked the event loop')
            return {'id': 'test-job'}
        async def other_request():
            for _ in range(200):
                if entered.is_set():
                    release.set()
                    return
                await asyncio.sleep(0.005)
            raise AssertionError('Submission did not start')
        data = {'task': 't2v', 'prompt': 'A toy robot', 'turbo_enabled': False,
                'steps': 8, 'sampler': 'euler', 'source_workflow': 'async-regression',
                'settings': {'runpod_api_key': 'key', 'h3_endpoint_id': 'test'}}
        with (mock.patch.object(app, 'configured_s3_helper', return_value=object()),
              mock.patch.object(app, 'submit', side_effect=submit),
              mock.patch.object(app, 'record_job_event')):
            result, _ = await asyncio.gather(app.run_h3(data), other_request())
        self.assertEqual(result['job']['id'], 'test-job')
        self.assertEqual(sources[0]['source_workflow'], 'async-regression')


if __name__ == '__main__':
    unittest.main()
