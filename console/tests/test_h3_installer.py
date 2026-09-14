import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from botocore.exceptions import ClientError

spec = importlib.util.spec_from_file_location("h3_installer", Path(__file__).parents[1] / "scripts/install_h3_acceleration.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class H3InstallerTests(unittest.TestCase):
    def test_gateway_without_metadata_verifies_actual_stored_bytes(self):
        for stored in (b"model-bytes", b"wrong-bytes"):
            with self.subTest(stored=stored), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = b"model-bytes"
                manifest = root / "models.json"
                manifest.write_text(json.dumps([{"key": "models/loras/test.safetensors", "size": len(source),
                    "sha256": hashlib.sha256(source).hexdigest(), "repo": "fixture/model", "revision": "pinned", "file": "test.safetensors"}]))
                client = mock.MagicMock()
                missing = ClientError({"Error": {"Code": "404"}}, "HeadObject")
                client.head_object.side_effect = [missing, {"ContentLength": len(source), "Metadata": {}}]
                client.create_multipart_upload.return_value = {"UploadId": "fixture"}
                client.upload_part.return_value = {"ETag": "etag"}
                body = mock.Mock()
                body.iter_chunks.return_value = [stored]
                client.get_object.return_value = {"Body": body}
                client.abort_multipart_upload.side_effect = ClientError({"Error": {"Code": "NoSuchUpload"}}, "AbortMultipartUpload")
                response = mock.MagicMock()
                response.__enter__.return_value = response
                response.iter_content.return_value = [source]
                state = root / "state.json"
                with (mock.patch.object(installer, "MANIFEST", manifest), mock.patch.object(installer, "STATE", state),
                      mock.patch.object(installer.app, "require_h3_s3", return_value=SimpleNamespace(client=client, bucket="test")),
                      mock.patch.object(installer.app, "settings", return_value={}),
                      mock.patch.object(installer.app, "scoped_model_get", return_value=response)):
                    if stored == source:
                        installer.main()
                        self.assertEqual(json.loads(state.read_text())["models/loras/test.safetensors"]["status"], "VERIFIED")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "Stored object SHA256"):
                            installer.main()
                        self.assertEqual(json.loads(state.read_text())["models/loras/test.safetensors"]["status"], "FAILED")
                body.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
