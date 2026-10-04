import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from output_naming import copy_flat_output


class NamingTests(unittest.TestCase):
    def test_concurrent_workers_and_retries_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'H3_00001_.mp4'
            source.write_bytes(b'complete-video')
            out = root / 'outputs'
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: copy_flat_output(source, out), range(24)))
            self.assertEqual(len({p.name for p in results}), 24)
            self.assertEqual({p.read_bytes() for p in results}, {b'complete-video'})
            self.assertEqual(copy_flat_output(source, out).name, 'H3_0025.mp4')

    def test_existing_legacy_names_and_custom_diagnostic_prefixes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / 'outputs'
            out.mkdir()
            (out / ('H3_00418__' + 'a'*32 + '.mp4')).write_bytes(b'old')
            source = root / 'H3_00001_.mp4'
            source.write_bytes(b'new')
            self.assertEqual(copy_flat_output(source, out).name, 'H3_0419.mp4')
            source = root / 'CloudStages_FirstPass_00001_.png'
            source.write_bytes(b'frame')
            self.assertEqual(copy_flat_output(source, out).name, 'CloudStages_FirstPass_0001.png')
