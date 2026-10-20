import importlib.util
from pathlib import Path
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('gpu_gate', Path(__file__).parents[1] / 'scripts/verify_gpu_attention.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GpuGateTests(unittest.TestCase):
    def test_both_supported_architectures_require_real_success(self):
        for capability in ([8, 9], [12, 0]):
            session = mock.Mock()
            session.get.return_value.json.return_value = {'cuda_capability': capability}
            session.post.return_value.json.return_value = {'passed': True}
            self.assertTrue(gate.verify('http://localhost', session)['passed'])
            session.post.return_value.json.return_value = {'passed': False}
            with self.assertRaisesRegex(RuntimeError, 'GPU correctness'):
                gate.verify('http://localhost', session)

    def test_other_architectures_do_not_attempt_an_unsupported_kernel(self):
        session = mock.Mock()
        session.get.return_value.json.return_value = {'cuda_capability': [8, 6]}
        self.assertEqual(gate.verify('http://localhost', session)['attention'], 'native')
        session.post.assert_not_called()


if __name__ == '__main__':
    unittest.main()
