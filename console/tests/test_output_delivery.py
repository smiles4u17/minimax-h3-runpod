import concurrent.futures
import tempfile
from pathlib import Path
import unittest
from unittest import mock
import app


class OutputDeliveryTests(unittest.TestCase):
    def test_concurrent_polls_download_once_and_missing_file_can_recover(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'output.mp4'
            def download(*args):
                target.write_bytes(b'rendered media')
                return [str(target)]
            settings = {'output_dir': str(root)}
            with mock.patch.object(app, 'DATA_DIR', root), mock.patch.object(app, '_save_outputs', side_effect=download) as save:
                def poll(_):
                    return app.save_outputs('job', {'output': {'files': ['clip.mp4']}}, settings, 'endpoint')
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(poll, range(2)))
                self.assertEqual(results, [[str(target)]] * 2)
                self.assertEqual(save.call_count, 1)
                target.unlink()
                poll(0)
                self.assertEqual(save.call_count, 2)

    def test_failed_atomic_write_preserves_existing_document(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'settings.json'
            target.write_text('{"valid": true}')
            with mock.patch.object(app.os, 'replace', side_effect=OSError('locked')):
                with self.assertRaises(OSError):
                    app.save_json(target, {'new': True})
            self.assertEqual(target.read_text(), '{"valid": true}')
            self.assertEqual(list(Path(directory).iterdir()), [target])


if __name__ == '__main__':
    unittest.main()
