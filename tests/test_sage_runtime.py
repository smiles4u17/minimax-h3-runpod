import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('sage_runtime', Path(__file__).parents[1] / 'scripts/select_sage_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class SageRuntimeTests(unittest.TestCase):
    def test_only_ada_shadows_the_proven_blackwell_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, 'SM89 runtime is missing'):
                runtime.package_path((8, 9), root)
            (root / 'sageattention').mkdir()
            (root / 'sageattention/__init__.py').touch()
            self.assertEqual(runtime.package_path((8, 9), root), str(root))
            for capability in ((12, 0), (8, 6), (10, 0)):
                self.assertEqual(runtime.package_path(capability, root), '')


if __name__ == '__main__':
    unittest.main()
