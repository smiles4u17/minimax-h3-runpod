import io
import json
import math
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

os.environ.setdefault("RUNPOD_MEDIA_CONSOLE_TOKEN", "test-console-token")

from fastapi.testclient import TestClient

import app as media_console


class SecurityAndH3Tests(unittest.TestCase):
    def test_h3_steps_above_training_threshold_survive_submission(self):
        captured = []
        data = {"settings": {"runpod_api_key": "key", "h3_endpoint_id": "test-h3"},
                "task": "t2v", "prompt": "A toy robot waves", "steps": 160,
                "turbo_enabled": True, "turbo_family": "lightx2v", "sampler": "euler",
                "turbo_lora": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"}
        with (mock.patch.object(media_console, "configured_s3_helper", return_value=object()),
              mock.patch.object(media_console, "record_job_event"),
              mock.patch.object(media_console, "submit", side_effect=lambda e,k,p,s: captured.append(p) or {"id": "test"})):
            response = self.client.post("/api/run/h3", json=data)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured[0]["steps"], 160)
        self.assertEqual(captured[0]["turbo_family"], "lightx2v")
        for steps in (0, -1, True, 1.5, "nan", "inf"):
            with self.assertRaises(ValueError):
                media_console.validate_h3_steps(steps)

    def test_text_only_submission_excludes_saved_reference_assets(self):
        captured = []
        data = {"settings": {"runpod_api_key": "key", "h3_endpoint_id": "test-h3"},
                "task": "t2v", "prompt": "A toy robot waves", "first_frame_path": "stale.png",
                "reference_paths": ["stale-ref.png"], "attention": "sol-attn", "sparse_tau": 1.1,
                "turbo_enabled": False, "sampler": "euler", "steps": 20}
        with (mock.patch.object(media_console, "configured_s3_helper", return_value=object()),
              mock.patch.object(media_console, "h3_asset_payload", side_effect=AssertionError("stale input")),
              mock.patch.object(media_console, "record_job_event"),
              mock.patch.object(media_console, "submit", side_effect=lambda e,k,p,s: captured.append(p) or {"id": "test"})):
            response = self.client.post("/api/run/h3", json=data)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured[0]["task"], "t2v")
        self.assertEqual(captured[0]["sparse_tau"], 1.1)
        self.assertNotIn("first_frame", captured[0])
        self.assertNotIn("references", captured[0])
        data.update(attention="sla", sparse_trained_weights=False)
        response = self.client.post("/api/run/h3", json=data)
        self.assertEqual(response.status_code, 400)

    def test_samimate_submission_builds_masked_contract_and_prompt(self):
        captured = []
        data = {
            "settings": {"runpod_api_key": "key", "h3_endpoint_id": "test-h3",
                         "s3_endpoint_url": "https://s3.example", "s3_access_key_id": "access",
                         "s3_secret_access_key": "secret", "h3_storage": {"s3_bucket": "bucket", "s3_region": "test"}},
            "source_workflow": "samimate", "task": "fl2v", "prompt": "Keep the red jacket",
            "subject_prompt": "woman on the left", "source_video_path": "source.mkv", "mask_path": "mask.mp4",
            "reference_paths": ["front.png", "side.png"], "reference_video_paths": ["stale.mp4"],
            "duration": 15, "width": 832, "height": 480,
            "fl2va_model": "fl.safetensors", "ref2va_model": "ref.safetensors",
            "text_encoder": "clip.safetensors", "video_vae": "video.safetensors", "audio_vae": "audio.safetensors",
            "turbo_enabled": False, "sampler": "res_multistep", "steps": 20,
        }
        with (mock.patch.object(media_console, "video_probe", return_value={"fps": 24}),
              mock.patch.object(media_console, "samimate_frame_count", return_value=48),
              mock.patch.object(media_console, "h3_asset_payload", side_effect=lambda path, *args: {"name": path, "data": "mock"}),
              mock.patch.object(media_console, "record_job_event"),
              mock.patch.object(media_console, "submit", side_effect=lambda e, k, p, s: captured.append(p) or {"id": "test"})):
            response = self.client.post("/api/run/h3", json=data)
        self.assertEqual(response.status_code, 200, response.text)
        payload = captured[0]
        self.assertEqual(payload['task'], 'r2v_masked')
        self.assertEqual(payload['duration'], 2)
        self.assertEqual(payload['model'], 'ref.safetensors')
        self.assertEqual(payload['masked_edit']['mask']['name'], 'mask.mp4')
        self.assertEqual(payload['reference_videos'], [])
        self.assertIn('<Picture 1>, <Picture 2>', payload['prompt'])
        self.assertIn('woman on the left', payload['prompt'])

    def setUp(self):
        startup = mock.patch.object(media_console.OUTPUT_SYNC, "start")
        startup.start()
        self.addCleanup(startup.stop)
        temp_parent = media_console.APP_DIR / "cache" / "temp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=temp_parent)
        root = Path(self.temp_dir.name)
        self.original_paths = (
            media_console.SETTINGS_PATH,
            media_console.PRESETS_PATH,
            media_console.ENDPOINT_PROFILES_PATH,
            media_console.PROMPTS_PATH,
            media_console.H3_SUBJECTS_PATH,
        )
        media_console.SETTINGS_PATH = root / "settings.json"
        media_console.PRESETS_PATH = root / "presets.json"
        media_console.ENDPOINT_PROFILES_PATH = root / "endpoint_profiles.json"
        media_console.PROMPTS_PATH = root / "prompts.json"
        media_console.H3_SUBJECTS_PATH = root / "h3_subjects.json"
        self.client = TestClient(
            media_console.app,
            headers={"Authorization": "Bearer test-console-token"},
        )

    def tearDown(self):
        self.client.close()
        (
            media_console.SETTINGS_PATH,
            media_console.PRESETS_PATH,
            media_console.ENDPOINT_PROFILES_PATH,
            media_console.PROMPTS_PATH,
            media_console.H3_SUBJECTS_PATH,
        ) = self.original_paths
        self.temp_dir.cleanup()

    def test_settings_are_redacted_and_blank_secrets_are_preserved(self):
        raw = media_console.defaults()
        raw["runpod_api_key"] = "runpod-secret"
        raw["s3_secret_access_key"] = "s3-secret"
        raw["model_download"] = {"hf_token": "hf-secret", "civitai_token": "civitai-secret"}
        raw["wan22_endpoint_id"] = "removed"
        raw["wan22"] = {"prompt": "removed"}
        media_console.SETTINGS_PATH.write_text(json.dumps(raw), encoding="utf-8")

        response = self.client.get("/api/settings")
        self.assertEqual(response.status_code, 200)
        public = response.json()
        self.assertEqual(public["runpod_api_key"], "")
        self.assertEqual(public["s3_secret_access_key"], "")
        self.assertEqual(public["model_download"]["hf_token"], "")
        self.assertEqual(public["model_download"]["civitai_token"], "")
        self.assertTrue(public["secret_fields_configured"]["runpod_api_key"])
        self.assertTrue(public["secret_fields_configured"]["model_download.hf_token"])
        self.assertNotIn("wan22", public)
        self.assertNotIn("wan22_endpoint_id", public)

        saved = self.client.post("/api/settings", json={"runpod_api_key": "", "timeout_seconds": 1234})
        self.assertEqual(saved.status_code, 200)
        on_disk = json.loads(media_console.SETTINGS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["runpod_api_key"], "runpod-secret")
        self.assertEqual(on_disk["timeout_seconds"], 1234)
        self.assertNotIn("wan22", on_disk)

    def test_legacy_wan22_presets_and_profile_secrets_are_removed(self):
        media_console.PRESETS_PATH.write_text(json.dumps({"wan22": {"old": {}}, "wan": {"good": {}}}), encoding="utf-8")
        profile = {
            "wan22_endpoint_id": "removed",
            "h3_storage": {
                "network_volume_id": "volume",
                "s3_access_key_id": "access-secret",
                "s3_secret_access_key": "secret-secret",
                "s3_bucket": "bucket",
            },
        }
        media_console.ENDPOINT_PROFILES_PATH.write_text(json.dumps({"old": profile}), encoding="utf-8")

        presets = self.client.get("/api/presets").json()["presets"]
        profiles = self.client.get("/api/endpoint-profiles").json()["profiles"]
        self.assertNotIn("wan22", presets)
        self.assertNotIn("wan22_endpoint_id", profiles["old"])
        self.assertNotIn("s3_access_key_id", profiles["old"]["h3_storage"])
        self.assertNotIn("s3_secret_access_key", profiles["old"]["h3_storage"])

    def test_h3_validation_uses_h3_endpoint_and_requires_first_frame_once(self):
        payload = {
            "settings": {
                "runpod_api_key": "key",
                "h3_endpoint_id": "h3-endpoint",
                "s3_endpoint_url": "https://s3.example",
                "s3_access_key_id": "access",
                "s3_secret_access_key": "secret",
                "h3_storage": {"s3_bucket": "h3-bucket", "s3_region": "test-1"},
            },
            "task": "fl2v",
            "prompt": "test",
            "first_frame_path": "",
            "duration": 2,
            "megapixels": 0.2,
            "steps": 6,
            "cache_threshold": 0.18,
        }
        result = self.client.post("/api/validate/h3", json=payload)
        self.assertEqual(result.status_code, 200)
        errors = result.json()["errors"]
        self.assertEqual(errors.count("first_frame path is required."), 1)
        self.assertFalse(any("InfiniteTalk endpoint" in error for error in errors))

    def test_private_remote_lora_urls_are_rejected(self):
        with self.assertRaises(ValueError):
            media_console.validate_public_http_url("http://127.0.0.1/model.safetensors")

    def test_non_loopback_requests_require_the_lan_token(self):
        with TestClient(media_console.app, client=("192.168.1.50", 50000)) as unauthenticated:
            response = unauthenticated.get("/api/health")
        self.assertEqual(response.status_code, 401)
        self.assertIn("Basic", response.headers.get("www-authenticate", ""))

    def test_named_storage_profiles_keep_main_and_h3_buckets_separate(self):
        raw = media_console.defaults()
        raw.update({"s3_bucket": "old-main", "s3_access_key_id": "access", "s3_secret_access_key": "secret"})
        raw["h3_storage"]["s3_bucket"] = "new-h3"
        main_name, main = media_console.named_storage_settings(raw, "main")
        h3_name, h3 = media_console.named_storage_settings(raw, "h3")
        self.assertEqual((main_name, main["s3_bucket"]), ("main", "old-main"))
        self.assertEqual((h3_name, h3["s3_bucket"]), ("h3", "new-h3"))
        self.assertEqual(h3["s3_access_key_id"], "access")
        with self.assertRaises(ValueError):
            media_console.named_storage_settings(raw, "arbitrary-bucket")

    def test_h3_payload_uses_worker_lora_path_and_model_turbo_overrides(self):
        captured = {}

        def fake_submit(endpoint, key, payload, settings):
            captured.update(payload)
            return {"id": "test-job"}

        payload = {
            "settings": {
                "runpod_api_key": "key",
                "h3_endpoint_id": "h3-endpoint",
                "s3_endpoint_url": "https://s3.example",
                "s3_access_key_id": "access",
                "s3_secret_access_key": "secret",
                "h3_storage": {"s3_bucket": "h3-bucket", "s3_region": "test-1"},
            },
            "task": "r2v",
            "prompt": "test",
            "reference_paths": ["reference.png"],
            "reference_video_paths": ["motion.mp4"],
            "reference_video_audio": [True],
            "reference_video_settings": [{"force_rate": 30, "start_frame": 12, "select_every_nth": 2}],
            "reference_audio_paths": ["voice.wav", "music.wav"],
            "duration": 2,
            "megapixels": 0.2,
            "steps": 6,
            "cache_threshold": 0.18,
            "fl2va_model": "fl.safetensors",
            "ref2va_model": "ref.safetensors",
            "text_encoder": "clip.safetensors",
            "video_vae": "video.safetensors",
            "audio_vae": "audio.safetensors",
            "turbo_enabled": True,
            "turbo_lora": "turbo.safetensors",
            "turbo_strength": 0.9,
            "loras": [{"name": "creative.safetensors", "strength": 0.75}],
        }
        with (
            mock.patch.object(media_console, "submit", side_effect=fake_submit),
            mock.patch.object(media_console, "h3_asset_payload", return_value={"data": "mock"}),
            mock.patch.object(media_console, "record_job_event"),
        ):
            response = self.client.post("/api/run/h3", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["model"], "ref.safetensors")
        self.assertEqual(captured["clip"], "clip.safetensors")
        self.assertEqual(captured["turbo_lora"], "H3/turbo.safetensors")
        self.assertEqual(captured["loras"], [{"name": "H3/creative.safetensors", "strength": 0.75}])
        self.assertEqual(len(captured["references"]), 1)
        self.assertEqual(len(captured["reference_videos"]), 1)
        self.assertEqual(captured["reference_video_audio"], [True])
        self.assertEqual(captured["reference_video_settings"], [{
            "force_rate": 30.0,
            "skip_first_frames": 12,
            "select_every_nth": 2,
            "frame_load_cap": 56,
        }])
        self.assertEqual(len(captured["reference_audios"]), 2)
        self.assertNotIn("output_upload_urls", captured)
        self.assertEqual(captured["output_layout"], "flat_outputs")
        debug = response.json()["debug"]
        self.assertEqual(debug["output_delivery"], "network_volume")
        self.assertEqual(debug["output_volume_dir"], "/runpod-volume/outputs")
        self.assertNotIn("X-Amz", json.dumps(debug))
        self.assertFalse(captured["cache_enabled"])

    def test_model_manager_rewrites_huggingface_blob_urls(self):
        url, name = media_console.resolve_model_download(
            "https://huggingface.co/example/repo/blob/main/models/demo.safetensors",
            {},
        )
        self.assertEqual(url, "https://huggingface.co/example/repo/resolve/main/models/demo.safetensors")
        self.assertEqual(name, "demo.safetensors")
        self.assertEqual(media_console.h3_model_destination("text_encoders"), "models/text_encoders")

    def test_h3_role_dropdowns_include_hybrid_diffusion_models(self):
        names = [
            "minimax_h3_fl2va_pruned.safetensors",
            "minimax_h3_ref2va_pruned.safetensors",
            "10Eros_Max_h3_TURBO-hybrid_beta4.safetensors",
        ]
        fl2va = media_console.h3_model_role_options(names, "fl2va")
        ref2va = media_console.h3_model_role_options(names, "ref2va")
        self.assertEqual(set(fl2va), set(names))
        self.assertEqual(set(ref2va), set(names))
        self.assertEqual(fl2va[0], "minimax_h3_fl2va_pruned.safetensors")
        self.assertEqual(ref2va[0], "minimax_h3_ref2va_pruned.safetensors")

    def test_model_manager_streams_multipart_directly_to_h3_volume(self):
        class FakeResponse:
            headers = {"content-length": "6", "content-disposition": 'attachment; filename="demo.safetensors"'}
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def raise_for_status(self): return None
            def iter_content(self, chunk_size): return iter((b"abc", b"def"))

        class FakeClient:
            def __init__(self): self.completed = None
            def create_multipart_upload(self, **kwargs): return {"UploadId": "upload-1"}
            def upload_part(self, **kwargs): return {"ETag": f"etag-{kwargs['PartNumber']}"}
            def complete_multipart_upload(self, **kwargs): self.completed = kwargs
            def abort_multipart_upload(self, **kwargs): raise AssertionError("upload should not abort")

        fake_s3 = mock.Mock()
        fake_s3.bucket = "h3-bucket"
        fake_s3.root = "/runpod-volume"
        fake_s3.client = FakeClient()
        job_id = "model-job"
        media_console.MODEL_DOWNLOAD_JOBS[job_id] = {"id": job_id}
        with (
            mock.patch.object(media_console, "settings", return_value={"model_download": {}}),
            mock.patch.object(media_console, "require_h3_s3", return_value=fake_s3),
            mock.patch.object(media_console, "resolve_model_download", return_value=("https://example.com/demo.safetensors", "demo.safetensors")),
            mock.patch.object(media_console, "scoped_model_get", return_value=FakeResponse()),
        ):
            media_console.run_model_download_job(job_id, {"url": "https://example.com/demo.safetensors", "category": "diffusion_models", "overwrite": True})
        job = media_console.MODEL_DOWNLOAD_JOBS.pop(job_id)
        self.assertEqual(job["status"], "COMPLETED")
        self.assertEqual(job["key"], "models/diffusion_models/demo.safetensors")
        self.assertEqual(job["downloaded_bytes"], 6)
        self.assertEqual(len(fake_s3.client.completed["MultipartUpload"]["Parts"]), 2)

    def test_model_manager_repairs_parts_missing_from_s3_inventory(self):
        class FakeClient:
            def __init__(self):
                self.repaired = False

            def list_parts(self, **kwargs):
                parts = [{"PartNumber": 2, "ETag": "etag-2"}]
                if self.repaired:
                    parts.insert(0, {"PartNumber": 1, "ETag": "etag-repaired-1"})
                return {"Parts": parts, "IsTruncated": False}

            def upload_part(self, **kwargs):
                self.repaired = True
                self.uploaded = kwargs
                return {"ETag": "etag-repaired-1"}

        fake_s3 = mock.Mock()
        fake_s3.bucket = "h3-bucket"
        fake_s3.client = FakeClient()
        job_id = "repair-job"
        media_console.MODEL_DOWNLOAD_JOBS[job_id] = {"id": job_id}
        with (
            mock.patch.object(media_console.time, "sleep"),
            mock.patch.object(media_console, "download_model_range", return_value=b"abc") as download_range,
        ):
            parts = media_console.verify_and_repair_model_parts(
                fake_s3,
                "models/diffusion_models/demo.safetensors",
                "upload-1",
                "https://example.com/demo.safetensors",
                {},
                {1: (0, 3), 2: (3, 3)},
                [{"PartNumber": 1, "ETag": "etag-1"}, {"PartNumber": 2, "ETag": "etag-2"}],
                job_id,
            )
        self.assertEqual([item["PartNumber"] for item in parts], [1, 2])
        self.assertEqual(fake_s3.client.uploaded["PartNumber"], 1)
        download_range.assert_called_once_with("https://example.com/demo.safetensors", {}, 0, 3)
        media_console.MODEL_DOWNLOAD_JOBS.pop(job_id)

    def test_large_model_part_size_stays_below_gateway_part_ceiling(self):
        max_parts = math.ceil(media_console.H3_MODEL_MAX_BYTES / media_console.H3_MODEL_PART_BYTES)
        self.assertLessEqual(max_parts, 512)

    def test_h3_turbo_off_is_authoritative(self):
        captured = {}

        def fake_submit(endpoint, key, payload, settings):
            captured.update(payload)
            return {"id": "test-job"}

        payload = {
            "settings": {
                "runpod_api_key": "key",
                "h3_endpoint_id": "h3-endpoint",
                "s3_endpoint_url": "https://s3.example",
                "s3_access_key_id": "access",
                "s3_secret_access_key": "secret",
                "h3_storage": {"s3_bucket": "h3-bucket", "s3_region": "test-1"},
            },
            "task": "fl2v", "prompt": "test", "first_frame_path": "first.png",
            "duration": 2, "megapixels": 0.2, "steps": 20, "cache_threshold": 0.18,
            "fl2va_model": "fl.safetensors", "ref2va_model": "ref.safetensors",
            "text_encoder": "clip.safetensors", "video_vae": "video.safetensors",
            "audio_vae": "audio.safetensors", "turbo_enabled": False,
            "sampler": "er_sde", "scheduler": "simple", "cache_enabled": False,
            "advanced_json": "{}",
        }
        with (
            mock.patch.object(media_console, "submit", side_effect=fake_submit),
            mock.patch.object(media_console, "h3_asset_payload", return_value={"data": "mock"}),
            mock.patch.object(media_console, "record_job_event"),
        ):
            response = self.client.post("/api/run/h3", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(captured["turbo_enabled"])
        self.assertEqual(captured["steps"], 20)
        self.assertEqual(captured["sampler"], "er_sde")
        self.assertEqual(captured["scheduler"], "simple")
        self.assertFalse(captured["cache_enabled"])
        self.assertNotIn("turbo_lora", captured)
        self.assertNotIn("turbo_strength", captured)

        payload["advanced_json"] = '{"turbo_enabled": true}'
        with mock.patch.object(media_console, "h3_asset_payload", return_value={"data": "mock"}):
            rejected = self.client.post("/api/run/h3", json=payload)
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("cannot override visible H3 controls", rejected.text)

        payload["advanced_json"] = '{"sampler": "euler"}'
        with mock.patch.object(media_console, "h3_asset_payload", return_value={"data": "mock"}):
            rejected = self.client.post("/api/run/h3", json=payload)
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("cannot override visible H3 controls", rejected.text)

    def test_h3_random_seed_and_negative_seed_are_normalized(self):
        captured = []

        def fake_submit(endpoint, key, payload, settings):
            captured.append(payload.copy())
            return {"id": f"test-job-{len(captured)}"}

        base = {
            "settings": {
                "runpod_api_key": "key", "h3_endpoint_id": "h3-endpoint",
                "s3_endpoint_url": "https://s3.example", "s3_access_key_id": "access",
                "s3_secret_access_key": "secret",
                "h3_storage": {"s3_bucket": "h3-bucket", "s3_region": "test-1"},
            },
            "task": "fl2v", "prompt": "test", "first_frame_path": "first.png",
            "duration": 2, "megapixels": 0.2, "steps": 6, "cache_threshold": 0.18,
            "fl2va_model": "fl.safetensors", "ref2va_model": "ref.safetensors",
            "text_encoder": "clip.safetensors", "video_vae": "video.safetensors",
            "audio_vae": "audio.safetensors", "turbo_enabled": False,
            "sampler": "er_sde", "scheduler": "simple", "cache_enabled": False,
            "advanced_json": "{}",
        }
        with (
            mock.patch.object(media_console, "submit", side_effect=fake_submit),
            mock.patch.object(media_console, "h3_asset_payload", return_value={"data": "mock"}),
            mock.patch.object(media_console, "record_job_event"),
            mock.patch.object(media_console.random, "randint", side_effect=[111, 222]),
        ):
            first = self.client.post("/api/run/h3", json={**base, "seed": -1})
            second = self.client.post("/api/run/h3", json={**base, "seed": 123, "seed_random": True})
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(captured[0]["seed"], 111)
        self.assertEqual(captured[1]["seed"], 222)
        self.assertGreaterEqual(captured[0]["seed"], 0)

    def test_identical_h3_submissions_are_each_submitted(self):
        payload = {
            "settings": {
                "runpod_api_key": "key", "h3_endpoint_id": "h3-endpoint",
                "s3_endpoint_url": "https://s3.example", "s3_access_key_id": "access",
                "s3_secret_access_key": "secret",
                "h3_storage": {"s3_bucket": "h3-bucket", "s3_region": "test-1"},
            },
            "task": "fl2v", "prompt": "deduplicate me", "first_frame_path": "first.png",
            "duration": 2, "megapixels": 0.2, "steps": 6, "cache_threshold": 0.18,
            "fl2va_model": "fl.safetensors", "ref2va_model": "ref.safetensors",
            "text_encoder": "clip.safetensors", "video_vae": "video.safetensors",
            "audio_vae": "audio.safetensors", "turbo_enabled": True,
            "turbo_lora": "turbo.safetensors", "advanced_json": "{}",
        }
        with (
            mock.patch.object(media_console, "submit", side_effect=[{"id": "one-job"}, {"id": "two-job"}]) as submit,
            mock.patch.object(media_console, "h3_asset_payload", return_value={"data": "mock"}),
            mock.patch.object(media_console, "record_job_event"),
        ):
            first = self.client.post("/api/run/h3", json=payload)
            second = self.client.post("/api/run/h3", json=payload)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json()["job"]["id"], "one-job")
        self.assertEqual(second.json()["job"]["id"], "two-job")
        self.assertNotIn("deduplicated", second.json())
        self.assertEqual(submit.call_count, 2)

    def test_h3_s3_asset_uses_mounted_volume_path_not_presigned_url(self):
        s3 = mock.Mock(bucket="h3-bucket", root="/runpod-volume")
        result = media_console.h3_asset_payload(
            "s3://h3-bucket/outputs/reference.mp4", "reference_video_1", "auto", s3
        )
        self.assertEqual(result, {
            "name": "reference.mp4",
            "volume_path": "/runpod-volume/outputs/reference.mp4",
        })
        s3.presign_key.assert_not_called()

    def test_h3_auto_delivery_budgets_all_base64_assets_together(self):
        root = Path(self.temp_dir.name)
        first = root / "first.png"
        second = root / "second.png"
        video = root / "motion.mp4"
        for path in (first, second, video):
            path.write_bytes(b"x" * (2 * 1024 * 1024))

        mode, estimated = media_console.h3_effective_delivery(
            "auto", [str(first), str(second), str(video)]
        )

        self.assertEqual(mode, "volume_path")
        self.assertGreater(estimated, media_console.H3_INLINE_BODY_BUDGET_BYTES)

    def test_h3_auto_delivery_keeps_small_combined_request_inline(self):
        root = Path(self.temp_dir.name)
        first = root / "first.png"
        second = root / "second.png"
        first.write_bytes(b"x" * 1024)
        second.write_bytes(b"y" * 1024)

        mode, estimated = media_console.h3_effective_delivery("auto", [str(first), str(second)])

        self.assertEqual(mode, "auto")
        self.assertLess(estimated, media_console.H3_INLINE_BODY_BUDGET_BYTES)

    def test_h3_explicit_base64_rejects_oversized_combined_request(self):
        root = Path(self.temp_dir.name)
        first = root / "first.png"
        second = root / "second.png"
        for path in (first, second):
            path.write_bytes(b"x" * (3 * 1024 * 1024))

        with self.assertRaisesRegex(ValueError, "10 MiB limit"):
            media_console.h3_effective_delivery("base64", [str(first), str(second)])

    def test_h3_run_auto_converts_combined_references_to_volume_paths(self):
        root = Path(self.temp_dir.name)
        sources = [root / "first.png", root / "second.png", root / "motion.mp4"]
        for path in sources:
            path.write_bytes(b"x" * (2 * 1024 * 1024))
        captured = {}
        fake_s3 = mock.Mock(bucket="h3-bucket", root="/runpod-volume")
        fake_s3.upload.side_effect = lambda path, kind: {
            "volume": f"/runpod-volume/input/{kind}/{Path(path).name}"
        }

        def fake_submit(endpoint, key, payload, settings):
            captured.update(payload)
            return {"id": "volume-job"}

        request = {
            "settings": {
                "runpod_api_key": "key",
                "h3_endpoint_id": "h3-endpoint",
                "h3_storage": {"s3_bucket": "h3-bucket", "s3_region": "test-1"},
            },
            "task": "r2v",
            "prompt": "test",
            "reference_paths": [str(sources[0]), str(sources[1])],
            "reference_video_paths": [str(sources[2])],
            "duration": 2,
            "megapixels": 0.2,
            "steps": 6,
            "turbo_enabled": True,
            "delivery": "auto",
        }
        with (
            mock.patch.object(media_console, "configured_s3_helper", return_value=fake_s3),
            mock.patch.object(media_console, "submit", side_effect=fake_submit),
            mock.patch.object(media_console, "record_job_event"),
        ):
            response = self.client.post("/api/run/h3", json=request)

        self.assertEqual(response.status_code, 200, response.text)
        assets = captured["references"] + captured["reference_videos"]
        self.assertEqual(len(assets), 3)
        self.assertTrue(all("volume_path" in asset for asset in assets))
        self.assertTrue(all("data" not in asset for asset in assets))
        self.assertEqual(response.json()["debug"]["effective_delivery"], "volume_path")
        self.assertEqual(response.json()["debug"]["reference_modes"], ["network_volume", "network_volume"])
        self.assertEqual(response.json()["debug"]["reference_video_modes"], ["network_volume"])

    def test_h3_lora_path_error_has_actionable_hint(self):
        hints = media_console.explain_error_text("lora_name value not in list")
        self.assertTrue(any("H3/filename.safetensors" in hint for hint in hints))

    def test_h3_subject_book_crud_and_optional_image(self):
        created = self.client.post(
            "/api/h3/subjects",
            json={
                "name": "Blue Jacket Lead",
                "definition": "the woman with short dark hair and a blue jacket",
                "image_path": "C:/references/lead.png",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        self.assertEqual(created.json()["Blue Jacket Lead"]["image_path"], "C:/references/lead.png")
        loaded = self.client.get("/api/h3/subjects")
        self.assertEqual(loaded.status_code, 200, loaded.text)
        self.assertEqual(
            loaded.json()["Blue Jacket Lead"]["definition"],
            "the woman with short dark hair and a blue jacket",
        )
        deleted = self.client.delete("/api/h3/subjects/Blue%20Jacket%20Lead")
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertNotIn("Blue Jacket Lead", deleted.json())

    def test_h3_subject_book_rejects_blank_definition(self):
        response = self.client.post("/api/h3/subjects", json={"name": "Empty", "definition": ""})
        self.assertEqual(response.status_code, 400)
        self.assertIn("name and definition required", response.text)

    def test_h3_ui_has_collapsible_sections_prompt_run_and_subject_builder(self):
        html = (media_console.APP_DIR / "static" / "index.html").read_text(encoding="utf-8")
        js = (media_console.APP_DIR / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn('data-h3-section="prompt"', html)
        self.assertIn('id="h3_run_button"', html)
        self.assertIn('id="h3_prompt_pick"', html)
        self.assertIn('id="h3_reference_grid"', html)
        self.assertIn('id="h3_reference_video_grid"', html)
        self.assertIn('id="h3_reference_audio_grid"', html)
        self.assertIn('FL2VA — first/last frame', html)
        self.assertIn('id="h3_comfyui_launch"', html)
        self.assertIn('id="h3_quality_notice"', html)
        self.assertIn("subject_definitions:\\n", js)
        self.assertIn("function toggleH3InputPreviews", js)
        self.assertIn("function h3ReferenceVideoPaths", js)
        self.assertIn("function h3SubjectTag", js)
        self.assertIn("function launchH3ComfyUI", js)
        self.assertIn("function applyH3CleanBaseline", js)
        self.assertNotIn("H3_SUBMITTING", js)
        self.assertIn("JOB_TIMERS", js)
        self.assertIn("sendBrowserSelectionToH3", js)
        self.assertIn("function setupH3RecentInputs", js)
        self.assertIn("function togglePathRecent", js)
        self.assertIn("Recent ▾", js)
        self.assertIn('id="samimate_generation_backend"', html)
        self.assertIn("samimateH3Payload", js)
        self.assertIn("sortBrowserFoldersByModified", js)
        self.assertIn("Apply sampling preset (keep LoRAs)", html)
        self.assertNotIn("Clean baseline (remove extra LoRAs)", html)
        self.assertNotIn("setH3Loras([],false)", js)

    def test_large_h3_output_error_has_volume_destination_hint(self):
        hints = media_console.explain_error_text("Output result.mp4 is too large for base64")
        self.assertTrue(any("OUTPUT_VOLUME_DIR=/runpod-volume/output/h3" in hint for hint in hints))

    def test_h3_volume_output_uses_exact_s3_key_without_mirroring(self):
        class FakeS3:
            bucket = "h3-bucket"
            def download_key(self, key, out_path):
                Path(out_path).write_bytes(b"video")
                return str(out_path)

        root = Path(self.temp_dir.name)
        settings = media_console.defaults()
        settings["output_dir"] = str(root / "outputs")
        settings["h3_endpoint_id"] = "h3-endpoint"
        status = {"output": {"files": [{"filename": "worker_name.mp4", "type": "volume_path", "data": "/runpod-volume/output/h3/job-1/worker_name.mp4", "size": 5}]}}
        with (
            mock.patch.object(media_console, "configured_s3_helper", return_value=FakeS3()),
            mock.patch.object(media_console, "download_recent_s3_outputs", return_value=[]),
        ):
            saved = media_console.save_outputs(
                "job-1",
                status,
                settings,
                "h3-endpoint",
            )
        self.assertEqual(len(saved), 1)
        self.assertEqual(Path(saved[0]).read_bytes(), b"video")
        self.assertEqual(status["_uploaded_s3"], ["s3://h3-bucket/output/h3/job-1/worker_name.mp4"])

    def test_h3_volume_downloads_exact_flat_key_without_mirroring(self):
        s3 = mock.Mock(bucket="h3-bucket")
        def download(key, path):
            Path(path).write_bytes(b"video")
            return str(path)
        s3.download_key.side_effect = download
        settings = media_console.defaults()
        settings.update(output_dir=str(Path(self.temp_dir.name) / "outputs"), h3_endpoint_id="h3-endpoint")
        status = {"output": {"files": [{"filename": "h3_unique.mp4", "type": "volume_path", "data": "/runpod-volume/outputs/h3_unique.mp4", "size": 5}]}}
        with (
            mock.patch.object(media_console, "configured_s3_helper", return_value=s3),
            mock.patch.object(media_console, "download_recent_s3_outputs", return_value=[]),
            mock.patch.object(media_console, "mirror_saved_outputs_to_s3") as mirror,
        ):
            saved = media_console.save_outputs("job", status, settings, "h3-endpoint")
        self.assertEqual(len(saved), 1)
        self.assertEqual(s3.download_key.call_args.args[0], "outputs/h3_unique.mp4")
        mirror.assert_not_called()
        self.assertEqual(status["_uploaded_s3"], ["s3://h3-bucket/outputs/h3_unique.mp4"])

    def test_h3_output_fallback_searches_nested_job_folder_and_paginates(self):
        class FakeS3:
            bucket = "h3-bucket"
            def __init__(self):
                self.calls = []
            def call(self, method, **kwargs):
                self.calls.append(kwargs)
                if kwargs.get("Prefix") != "output/h3/job-9/":
                    return {"Contents": []}
                if "ContinuationToken" not in kwargs:
                    return {
                        "Contents": [{"Key": "output/h3/job-9/readme.txt"}],
                        "IsTruncated": True,
                        "NextContinuationToken": "page-2",
                    }
                return {
                    "Contents": [{"Key": "output/h3/job-9/final.mp4"}],
                    "IsTruncated": False,
                }
            def download_key(self, key, out_path):
                self.downloaded = key
                Path(out_path).write_bytes(b"video")
                return str(out_path)

        fake = FakeS3()
        saved = media_console.download_recent_s3_outputs(
            fake,
            media_console.h3_storage_settings(media_console.defaults()),
            Path(self.temp_dir.name),
            "h3",
            0,
            job_id="job-9",
            strict_job=True,
        )
        self.assertEqual(len(saved), 1)
        self.assertEqual(fake.downloaded, "output/h3/job-9/final.mp4")
        self.assertTrue(any(call.get("ContinuationToken") == "page-2" for call in fake.calls))
        self.assertTrue(all("job-9/" in call.get("Prefix", "") for call in fake.calls))

    def test_h3_comfyui_status_and_launch_reuse_volume_pod(self):
        current = media_console.defaults()
        current["runpod_api_key"] = "runpod-secret"
        media_console.SETTINGS_PATH.write_text(json.dumps(current), encoding="utf-8")
        pod = {
            "id": "pod-1", "name": "H3 ComfyUI", "desiredStatus": "EXITED",
            "imageName": "runpod/comfyui:cuda13.0", "networkVolumeId": "vgc3ky6r6y",
            "volumeMountPath": "/workspace", "ports": ["8188/http"], "costPerHr": 0.99,
        }
        with mock.patch.object(media_console, "runpod_pod_request", return_value=[pod]):
            status = self.client.get("/api/h3/comfyui/status")
        self.assertEqual(status.status_code, 200, status.text)
        self.assertEqual(status.json()["pod"]["id"], "pod-1")
        self.assertNotIn("env", status.json()["pod"])

        missing_confirmation = self.client.post("/api/h3/comfyui/launch", json={})
        self.assertEqual(missing_confirmation.status_code, 400)
        with (
            mock.patch.object(media_console, "prepare_h3_comfyui_volume", return_value={"prepared": True}),
            mock.patch.object(media_console, "list_h3_comfyui_pods", return_value=[media_console.h3_comfyui_pod_public(pod)]),
            mock.patch.object(media_console, "runpod_pod_request", return_value={}) as pod_api,
        ):
            launched = self.client.post("/api/h3/comfyui/launch", json={"confirm_cost": True, "pod_id": "pod-1"})
        self.assertEqual(launched.status_code, 200, launched.text)
        self.assertFalse(launched.json()["created"])
        pod_api.assert_called_once_with("POST", "pod-1/start", s=mock.ANY)

    def test_job_history_write_failure_never_breaks_job_result(self):
        with mock.patch.object(media_console, "append_limited_json", side_effect=PermissionError("locked")):
            media_console.record_job_event({"target": "status", "job_id": "job-locked", "status": "COMPLETED"})

    def test_browser_all_lists_non_media_files_locally_and_in_s3(self):
        root = Path(self.temp_dir.name)
        (root / "workflow.json").write_text("{}", encoding="utf-8")
        (root / "preview.mp4").write_bytes(b"video")
        local_all = self.client.get("/api/browse", params={"path": str(root), "media_type": "All"}).json()
        local_videos = self.client.get("/api/browse", params={"path": str(root), "media_type": "Videos"}).json()
        self.assertIn("workflow.json", {item["name"] for item in local_all["items"]})
        self.assertNotIn("workflow.json", {item["name"] for item in local_videos["items"]})

        helper = mock.Mock()
        helper.bucket = "h3-bucket"
        helper.call.return_value = {
            "Contents": [
                {"Key": "output/h3/workflow.json", "Size": 2},
                {"Key": "output/h3/preview.mp4", "Size": 5},
            ]
        }
        with mock.patch.object(media_console, "require_named_s3", return_value=("h3", helper)):
            s3_all = self.client.get("/api/s3/browse", params={"prefix": "output/h3", "media_type": "All", "storage": "h3"}).json()
            s3_videos = self.client.get("/api/s3/browse", params={"prefix": "output/h3", "media_type": "Videos", "storage": "h3"}).json()
        self.assertIn("workflow.json", {item["name"] for item in s3_all["items"]})
        self.assertNotIn("workflow.json", {item["name"] for item in s3_videos["items"]})

    def test_s3_folders_sort_by_newest_nested_object(self):
        helper = mock.Mock(bucket="h3-bucket")
        helper.call.side_effect = [
            {
                "CommonPrefixes": [{"Prefix": "older/"}, {"Prefix": "newer/"}],
                "Contents": [],
                "IsTruncated": False,
            },
            {
                "Contents": [
                    {"Key": "older/a.mp4", "Size": 1, "LastModified": datetime(2026, 1, 1, tzinfo=timezone.utc)},
                    {"Key": "newer/b.mp4", "Size": 1, "LastModified": datetime(2026, 2, 1, tzinfo=timezone.utc)},
                ],
                "IsTruncated": False,
            },
        ]
        with mock.patch.object(media_console, "require_named_s3", return_value=("h3", helper)):
            result = self.client.get(
                "/api/s3/browse",
                params={"sort_by": "modified", "sort_dir": "desc", "storage": "h3"},
            )
        self.assertEqual(result.status_code, 200, result.text)
        folders = [item for item in result.json()["items"] if item["is_dir"]]
        self.assertEqual([item["name"] for item in folders], ["newer", "older"])
        self.assertTrue(folders[0]["modified"].startswith("2026-02-01"))

    def test_s3_metadata_retries_one_transient_failure(self):
        helper = mock.Mock()
        helper.bucket = "bucket"
        helper.call.side_effect = [RuntimeError("transient"), {"ContentLength": 123}]
        with mock.patch.object(media_console.time, "sleep"):
            result = media_console.s3_object_metadata(helper, "output/file.mp4")
        self.assertEqual(result["ContentLength"], 123)
        helper.refresh_client.assert_called_once_with()

    def test_s3_download_falls_back_to_get_object_when_head_is_forbidden(self):
        class FakeClient:
            def __init__(self):
                self.get_calls = 0

            def download_file(self, _bucket, _key, _path):
                raise RuntimeError("HeadObject forbidden")

            def get_object(self, **kwargs):
                self.get_calls += 1
                self.kwargs = kwargs
                return {"Body": io.BytesIO(b"direct object bytes")}

        helper = object.__new__(media_console.S3Helper)
        helper.bucket = "h3-bucket"
        helper.client = FakeClient()
        out = Path(self.temp_dir.name) / "downloaded.bin"

        result = helper.download_key("output/h3/test.bin", out)

        self.assertEqual(result, str(out))
        self.assertEqual(out.read_bytes(), b"direct object bytes")
        self.assertEqual(helper.client.get_calls, 1)
        self.assertEqual(helper.client.kwargs, {"Bucket": "h3-bucket", "Key": "output/h3/test.bin"})


if __name__ == "__main__":
    unittest.main()
