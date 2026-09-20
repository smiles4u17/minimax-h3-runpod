import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).parents[1]))
from telemetry import JobTelemetry, is_oom, redact


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock = mock.patch('telemetry.time.monotonic', return_value=0)
        self.now = self.clock.start()
        self.m = JobTelemetry({'id': 'test-job'}, self.temp.name, 'http://localhost:8188', 120, 30)
        self.m.publish = mock.Mock()

    def tearDown(self):
        self.clock.stop()
        self.temp.cleanup()

    def test_progress_is_prompt_scoped_and_stage_specific(self):
        self.m.prompt_id='ours'
        self.m.consume({'type':'progress','data':{'prompt_id':'other','value':4,'max':8}})
        self.assertNotIn('step', self.m.state)
        self.now.return_value=20
        self.m.consume({'type':'progress','data':{'prompt_id':'ours','value':4,'max':8}})
        self.assertEqual(self.m.state['step'],4)
        self.assertEqual(self.m.last_activity,20)
        self.m.consume({'type':'executing','data':{'node':'VAE'}})
        self.assertNotIn('step',self.m.state)

    def test_heartbeat_does_not_extend_inactivity(self):
        self.now.return_value=31
        with self.assertRaisesRegex(TimeoutError,'no progress'):
            self.m.check()

    def test_activity_extends_idle_but_not_maximum(self):
        self.now.return_value=25
        self.m.stage('loading_model')
        self.now.return_value=40
        self.m.check()
        self.m.last_activity=119
        self.now.return_value=121
        with self.assertRaisesRegex(TimeoutError,'maximum runtime'):
            self.m.check()

    def test_oom_is_immediate(self):
        self.m.consume({'type':'execution_error','data':{'exception_type':'torch.OutOfMemoryError','exception_message':'CUDA out of memory'}})
        self.assertTrue(self.m.state['oom'])
        with self.assertRaises(RuntimeError): self.m.check()

    def test_durable_logs_are_bounded_and_secrets_redacted(self):
        self.m.log_path=Path(self.temp.name)/'comfy.log'
        self.m.log_path.write_text('\n'.join(['hello']*150+['Bearer abc-secret https://site/a?token=secret']))
        self.m._read_logs()
        self.assertEqual(len(self.m.state['logs']),100)
        self.assertNotIn('abc-secret',''.join(self.m.state['logs']))
        self.assertNotIn('token=secret',''.join(self.m.state['logs']))
        JobTelemetry.publish(self.m,force=True)
        self.assertEqual(json.loads(self.m.path.read_text())['job_id'],'test-job')

    def test_oom_classification_does_not_guess_from_sigabrt(self):
        self.assertTrue(is_oom('torch.OutOfMemoryError'))
        self.assertFalse(is_oom('process exited -6'))


if __name__=='__main__': unittest.main()
