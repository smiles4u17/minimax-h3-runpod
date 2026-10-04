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

    def test_repeated_progress_and_poll_logs_do_not_extend_inactivity(self):
        self.m.stage('sampling', step=1, total_steps=8)
        self.now.return_value = 31
        self.m.stage('sampling', step=1, total_steps=8)
        self.m.log_path = Path(self.temp.name) / 'poll.log'
        self.m.log_path.write_text('GET /history 200\nstatus update\n')
        self.m._read_logs()
        with self.assertRaisesRegex(TimeoutError, 'no progress'):
            self.m.check()

    def test_provider_heartbeat_is_once_per_minute_with_compact_payload(self):
        import types
        progress = mock.Mock()
        self.m.state['metadata'] = {'large': 'details'}
        with mock.patch.dict(sys.modules, {'runpod': types.SimpleNamespace(serverless=types.SimpleNamespace(progress_update=progress))}), mock.patch('builtins.print'):
            for tick in (0, 15, 30, 45, 60):
                self.now.return_value = tick
                self.m._publish(force=True)
            self.assertEqual(progress.call_count, 2)
            self.assertNotIn('metadata', progress.call_args.args[1])
            self.assertIn('metadata', json.loads(self.m.path.read_text()))

    def test_blocked_handler_watchdog_interrupts_and_exits(self):
        self.now.return_value = 121
        self.m.stop_event = mock.Mock()
        self.m.stop_event.wait.return_value = False
        with mock.patch('telemetry.requests.post') as interrupt, mock.patch('telemetry.os._exit') as terminate:
            self.m._watchdog()
        interrupt.assert_called_once()
        terminate.assert_called_once_with(1)
        self.assertEqual(self.m.state['stage'], 'watchdog_expired')
        with self.assertRaisesRegex(TimeoutError, 'maximum runtime'):
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

    def test_fatal_allocator_log_fails_without_waiting_for_idle(self):
        self.m.log_path = Path(self.temp.name) / 'comfy.log'
        self.m.log_path.write_text("terminate called after throwing an instance of 'c10::AcceleratorError'\n  what(): CUDA error: invalid argument\nFatal Python error: Aborted\n")
        self.m._read_logs()
        self.assertTrue(self.m.state['fatal_process_error'])
        self.assertFalse(self.m.state.get('oom', False))
        with self.assertRaisesRegex(RuntimeError, 'fatal CUDA/process'):
            self.m.check()


if __name__=='__main__': unittest.main()
