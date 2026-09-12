import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from output_sync import OutputSync


class Storage:
    bucket = "test-volume"

    def __init__(self):
        self.objects = {}
        self.downloads = []
        self.fail = False
        self.pages = []

    def add(self, key, data, when=None, etag=None):
        self.objects[key] = (data, {"Key": key, "Size": len(data),
            "ETag": etag or hashlib.md5(data).hexdigest(),
            "LastModified": when or datetime.now().astimezone()})

    def call(self, method, **kwargs):
        self.pages.append(kwargs)
        matching = [obj for key, (_, obj) in self.objects.items() if key.startswith(kwargs["Prefix"])]
        offset = int(kwargs.get("ContinuationToken", 0))
        # Tiny pages exercise continuation even with a small fixture.
        return {"Contents": matching[offset:offset+2], "IsTruncated": offset+2 < len(matching),
                "NextContinuationToken": str(offset+2)}

    def download_key(self, key, path):
        self.downloads.append(key)
        Path(path).write_bytes(b"partial" if self.fail else self.objects[key][0])


class OutputSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.storage = Storage()
        self.sync = OutputSync(self.root / "state.json",
            lambda: [("H3", self.storage, ["outputs"], "test")],
            lambda: self.root / "local", {".mp4", ".png"})

    def state(self):
        return next(iter(json.loads(self.sync.state_path.read_text())["stores"].values()))

    def test_initial_today_only_pagination_exclusions_and_dedup(self):
        midnight = datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
        self.storage.add("outputs/old.mp4", b"old", midnight-timedelta(seconds=1))
        self.storage.add("outputs/a.mp4", b"one")
        self.storage.add("outputs/b.mp4", b"two")
        self.storage.add("outputs/_setup/test.png", b"setup")
        self.storage.add("inputs/ref.png", b"input")
        self.storage.add("outputs/model.bin", b"model")
        local = self.root / "local"
        local.mkdir()
        (local / "h3_0024.mp4").write_bytes(b"one")
        self.sync.run()
        self.assertEqual(self.storage.downloads, ["outputs/b.mp4"])
        self.assertEqual(self.sync.snapshot()["downloaded"], 1)
        self.assertEqual(self.sync.snapshot()["existing"], 1)
        self.assertGreater(len(self.storage.pages), 1)

    def test_resume_new_additions_and_recover_deleted_known_output(self):
        self.storage.add("outputs/a.mp4", b"one")
        self.sync.run()
        first = Path(self.state()["files"]["outputs/a.mp4"]["path"])
        first.unlink()
        self.storage.add("outputs/b.mp4", b"second")
        self.storage.add("outputs/old.mp4", b"old", datetime.now().astimezone()-timedelta(days=1))
        # New instance simulates a server restart and reuses its durable checkpoint.
        restarted = OutputSync(self.sync.state_path, self.sync.sources, self.sync.output_root, {".mp4"})
        restarted.run()
        self.assertTrue(first.exists())
        self.assertNotIn("outputs/old.mp4", self.storage.downloads)
        self.assertIn("outputs/b.mp4", self.storage.downloads)
        self.assertEqual(restarted.snapshot()["downloaded"], 2)

    def test_failure_keeps_checkpoint_cleans_partial_and_retries(self):
        self.storage.add("outputs/a.mp4", b"complete output")
        self.storage.fail = True
        self.sync.run()
        since = self.state()["since"]
        self.assertTrue(self.sync.snapshot()["errors"])
        self.assertEqual(list((self.root / "local").rglob("*.part")), [])
        self.assertFalse(self.state()["files"])
        self.storage.fail = False
        self.sync.run()
        self.assertEqual(self.sync.snapshot()["downloaded"], 1)
        self.assertGreaterEqual(self.state()["since"], since)

    def test_multipart_etag_matches_numbered_local_file_by_content(self):
        local = self.root / "local"
        local.mkdir()
        (local / "h3_0024.mp4").write_bytes(b"same file")
        self.storage.add("outputs/a.mp4", b"same file", etag="multipart-2")
        self.sync.run()
        self.assertEqual(self.sync.snapshot()["downloaded"], 0)
        self.assertEqual(len(list(local.rglob("*.mp4"))), 1)

    def test_file_arriving_during_listing_is_found_next_run(self):
        self.sync.run()
        cutoff = self.state()["since"]
        arrival = datetime.fromtimestamp(cutoff+1).astimezone()
        self.storage.add("outputs/late.mp4", b"late", arrival)
        self.sync.run()
        self.assertEqual(self.storage.downloads, ["outputs/late.mp4"])

    def test_concurrent_start_coalesces(self):
        with patch("output_sync.threading.Thread") as thread:
            self.sync.start()
            self.sync.start()
        self.assertEqual(thread.call_count, 1)


if __name__ == "__main__":
    unittest.main()
