import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("RUNPOD_MEDIA_CONSOLE_TOKEN", "test-console-token")

from fastapi.testclient import TestClient

import app as media_console
import prompt_book_h3


class PromptBookH3Tests(unittest.TestCase):
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
            prompt_book_h3.WORKFLOW_ROOT,
        )
        media_console.SETTINGS_PATH = root / "settings.json"
        media_console.PRESETS_PATH = root / "presets.json"
        media_console.ENDPOINT_PROFILES_PATH = root / "endpoint_profiles.json"
        media_console.PROMPTS_PATH = root / "prompts.json"
        media_console.H3_SUBJECTS_PATH = root / "h3_subjects.json"
        self.workflow_root = root / "workflows"
        self.workflow_root.mkdir()
        prompt_book_h3.WORKFLOW_ROOT = self.workflow_root
        self.client = TestClient(media_console.app, headers={"Authorization": "Bearer test-console-token"})

    def tearDown(self):
        self.client.close()
        (
            media_console.SETTINGS_PATH,
            media_console.PRESETS_PATH,
            media_console.ENDPOINT_PROFILES_PATH,
            media_console.PROMPTS_PATH,
            media_console.H3_SUBJECTS_PATH,
            prompt_book_h3.WORKFLOW_ROOT,
        ) = self.original_paths
        self.temp_dir.cleanup()

    def catalog(self):
        return {
            "subjects": [
                {"id": "ada", "label": "Ada", "path": r"C:\stills\ada.jpg", "images": [
                    {"path": r"C:\stills\ada.jpg", "primary": True},
                    {"path": r"C:\stills\ada-2.jpg", "primary": False},
                ]},
                {"id": "bea", "label": "Bea", "path": r"C:\stills\bea.jpg", "images": []},
            ],
            "videos": [
                {"id": "clip", "label": "Clip", "file": r"C:\clips\clip.mp4", "duration_sec": 42, "width": 720, "height": 1280},
            ],
        }

    def test_request_uses_four_step_ref2v_and_clamps_duration(self):
        request, meta = prompt_book_h3.build_prompt_book_h3_request(
            {"workflow_id": "ours", "steps": 4, "sampler": "res_multistep", "scheduler": "beta", "subject_id": "ada", "video_id": "clip", "threesome": True, "subject_b": "bea", "loras": [{"name": "creative.safetensors", "strength": 0.75}]},
            catalog=self.catalog(),
            baked={"prompt": "Completely replace <Subject 3> with <Subject 1>."},
        )
        self.assertEqual(request["task"], "r2v")
        self.assertEqual(request["workflow_variant"], "ref2v_20260920")
        self.assertEqual(request["steps"], 4)
        self.assertEqual(request["pass1_split"], 3)
        self.assertEqual(request["second_pass_sigma"], 2)
        self.assertFalse(request["use_larry"])
        self.assertFalse(request["turbo_enabled"])
        self.assertEqual(request["sampler"], "res_multistep")
        self.assertEqual(request["scheduler"], "beta")
        self.assertEqual(request["ref2va_model"], prompt_book_h3.BETA5_MODEL)
        self.assertFalse(request["rtx_upscale"])
        self.assertTrue(request["latent_upscale"])
        self.assertEqual(request["attention"], "sage")
        self.assertEqual(request["megapixels"], 0.2)
        self.assertEqual(request["final_megapixels"], 1.0)
        self.assertEqual(request["duration"], 15.0)
        self.assertTrue(meta["duration_clamped"])
        self.assertEqual(request["aspect_ratio"], "9:16 (Portrait Widescreen)")
        self.assertEqual(request["photo_paths"][:3], [r"C:\stills\ada.jpg", r"C:\stills\bea.jpg", r"C:\stills\ada-2.jpg"])
        self.assertEqual(request["reference_video_paths"], [r"C:\clips\clip.mp4"])
        self.assertFalse(request["use_multi_image"])
        self.assertEqual(request["loras"], [{"name": "creative.safetensors", "strength": 0.75}])
        self.assertEqual(request["advanced_json"], "{}")
        tuned, _ = prompt_book_h3.build_prompt_book_h3_request(
            {"workflow_id": "ours", "steps": 6, "subject_id": "ada", "video_id": "clip",
             "megapixels": 0.3, "final_megapixels": 1.2, "latent_upscale": True, "rtx_upscale": True,
             "pass1_split": 3, "second_pass_sigma": 3, "cache_enabled": False, "attention": "native",
             "aspect_ratio": "16:9 (Widescreen)"},
            catalog=self.catalog(),
            baked={"prompt": "Completely replace <Subject 3> with <Subject 1>."},
        )
        self.assertEqual(tuned["megapixels"], 0.3)
        self.assertEqual(tuned["final_megapixels"], 1.2)
        self.assertTrue(tuned["rtx_upscale"])
        self.assertEqual(tuned["pass1_split"], 3)
        self.assertEqual(tuned["second_pass_sigma"], 3)
        self.assertFalse(tuned["cache_enabled"])
        self.assertEqual(tuned["attention"], "native")
        self.assertEqual(tuned["aspect_ratio"], "16:9 (Widescreen)")
        trimmed, _ = prompt_book_h3.build_prompt_book_h3_request(
            {"workflow_id": "ours", "subject_id": "ada", "video_id": "clip", "trim_start": 2, "trim_end": 10},
            catalog=self.catalog(),
            baked={"prompt": "Completely replace <Subject 3> with <Subject 1>."},
        )
        self.assertEqual(trimmed["duration"], 8)
        self.assertEqual(trimmed["reference_video_settings"][0]["start_frame"], 48)
        self.assertEqual(trimmed["reference_video_settings"][0]["force_rate"], 24)
        self.assertEqual(trimmed["reference_video_audio"], [True])
        self.assertEqual(trimmed["audio_vae"], "minimax_h3_audio_vae_fp32.safetensors")
        self.assertFalse(trimmed["use_reference_audio_as_output"])
        explicit, explicit_meta = prompt_book_h3.build_prompt_book_h3_request(
            {"workflow_id": "ours", "subject_id": "ada", "video_id": "clip", "duration": 5, "trim_start": 2},
            catalog=self.catalog(),
            baked={"prompt": "Completely replace <Subject 3> with <Subject 1>."},
        )
        self.assertEqual(explicit["duration"], 5)
        self.assertFalse(explicit_meta["duration_clamped"])
        self.assertEqual(explicit["reference_video_settings"][0]["start_frame"], 48)
        too_long, too_long_meta = prompt_book_h3.build_prompt_book_h3_request(
            {"workflow_id": "ours", "subject_id": "ada", "video_id": "clip", "duration": 20},
            catalog=self.catalog(),
            baked={"prompt": "Completely replace <Subject 3> with <Subject 1>."},
        )
        self.assertEqual(too_long["duration"], 15.0)
        self.assertTrue(too_long_meta["duration_clamped"])

    def test_eight_step_and_prompt_override(self):
        request, meta = prompt_book_h3.build_prompt_book_h3_request(
            {
                "workflow_id": "quality5090",
                "subject_id": "ada",
                "video_id": "clip",
                "use_prompt_override": True,
                "prompt": "Edited prompt",
            },
            catalog={
                "subjects": self.catalog()["subjects"][:1],
                "videos": [{**self.catalog()["videos"][0], "duration_sec": 6.2, "width": 1920, "height": 1080}],
            },
            baked={"prompt": "Baked"},
        )
        self.assertEqual(request["prompt"], "Edited prompt")
        self.assertEqual(request["steps"], 8)
        self.assertEqual(request["pass1_split"], 7)
        self.assertEqual(request["second_pass_sigma"], 2)
        self.assertFalse(request["turbo_enabled"])
        self.assertEqual(request["duration"], 6.2)
        self.assertFalse(meta["duration_clamped"])
        capped, capped_meta = prompt_book_h3.build_prompt_book_h3_request(
            {"workflow_id": "quality5090", "subject_id": "ada", "video_id": "clip", "duration": 12},
            catalog={
                "subjects": self.catalog()["subjects"][:1],
                "videos": [{**self.catalog()["videos"][0], "duration_sec": 6.2, "width": 1920, "height": 1080}],
            },
            baked={"prompt": "Baked"},
        )
        self.assertEqual(capped["duration"], 6.2)
        self.assertTrue(capped_meta["duration_clamped"])
        self.assertEqual(request["aspect_ratio"], "16:9 (Widescreen)")
        self.assertEqual(request["photo_paths"][0], r"C:\stills\ada.jpg")
        self.assertEqual(request["photo_paths"][1], r"C:\stills\ada-2.jpg")

    def test_full_first_pass_and_diagnostic_capture_are_explicit(self):
        base = {"workflow_id": "ours", "subject_id": "ada", "video_id": "clip",
                "pass1_split": 8, "second_pass_sigma": 5, "diagnostic_frames": True}
        request, _ = prompt_book_h3.build_prompt_book_h3_request(
            base, catalog=self.catalog(), baked={"prompt": "Test prompt"})
        self.assertEqual(request["pass1_split"], 8)
        self.assertEqual(request["second_pass_sigma"], 5)
        self.assertTrue(request["diagnostic_frames"])
        for changes in ({"pass1_split": 9}, {"second_pass_sigma": 4}):
            with self.assertRaises(ValueError):
                prompt_book_h3.build_prompt_book_h3_request(
                    {**base, **changes}, catalog=self.catalog(), baked={"prompt": "Test prompt"})

    def test_silent_scene_does_not_request_missing_vhs_audio(self):
        scene = Path(self.temp_dir.name) / "silent.mp4"
        scene.write_bytes(b"fixture")
        catalog = self.catalog()
        catalog["videos"][0]["file"] = str(scene)
        probe = mock.Mock(stdout="", stderr="Input #0, mov, from 'silent.mp4':\n  Stream #0:0: Video: h264")
        with mock.patch.object(prompt_book_h3.subprocess, "run", return_value=probe):
            request, _ = prompt_book_h3.build_prompt_book_h3_request(
                {"subject_id": "ada", "video_id": "clip"}, catalog=catalog, baked={"prompt": "Test prompt"})
        self.assertEqual(request["reference_video_audio"], [False])

    def test_main_h3_generation_is_inherited_without_stale_references(self):
        main = {
            "task": "r2v", "workflow_variant": "ref2v_20260920",
            "steps": 8, "sampler": "res_multistep", "scheduler": "beta",
            "ref2va_model": "my-working-ref2v.safetensors", "text_encoder": "encoder.safetensors",
            "video_vae": "my-video-vae.safetensors", "audio_vae": "my-audio-vae.safetensors",
            "megapixels": 0.4, "final_megapixels": 0.9,
            "latent_upscale": True, "rtx_upscale": False,
            "output_mode": "rtx",  # obsolete setting must not override the switches
            "pass1_split": 8, "second_pass_sigma": 5,
            "turbo_enabled": False, "use_larry": False, "cache_enabled": False,
            "attention": "native", "seed_random": False, "seed": 12345,
            "loras": [{"name": "first.safetensors", "strength": 0.5},
                      {"name": "H3/shared.safetensors", "strength": 0.7}],
            "prompt": "Stale H3 prompt", "photo_paths": ["stale.jpg"],
            "reference_video_paths": ["stale.mp4"],
        }
        request, meta = prompt_book_h3.build_prompt_book_h3_request(
            {"subject_id": "ada", "video_id": "clip", "h3_generation": main,
             "loras": [{"name": "SHARED.safetensors", "strength": 0.9},
                       {"name": "extra.safetensors", "strength": 0.8}]},
            catalog=self.catalog(), baked={"prompt": "Baked from Prompt Book"})
        for field in ("steps", "sampler", "scheduler", "ref2va_model", "text_encoder",
                      "video_vae", "audio_vae", "megapixels", "final_megapixels",
                      "latent_upscale", "rtx_upscale", "pass1_split", "second_pass_sigma",
                      "turbo_enabled", "use_larry", "cache_enabled", "attention", "seed"):
            self.assertEqual(request[field], main[field], field)
        self.assertEqual(request["prompt"], "Baked from Prompt Book")
        self.assertNotIn("output_mode", request)
        self.assertEqual(request["photo_paths"][0], r"C:\stills\ada.jpg")
        self.assertEqual(request["reference_video_paths"], [r"C:\clips\clip.mp4"])
        self.assertEqual(len(request["loras"]), 3)
        self.assertEqual(request["loras"][1], {"name": "SHARED.safetensors", "strength": 0.9})
        self.assertEqual(meta["model"], "my-working-ref2v.safetensors")
        self.assertEqual(meta["workflow_id"], "main_h3")
        other_tab, _ = prompt_book_h3.build_prompt_book_h3_request(
            {"subject_id": "ada", "video_id": "clip",
             "h3_generation": {**main, "task": "fl2v"}},
            catalog=self.catalog(), baked={"prompt": "Test"})
        self.assertEqual(other_tab["task"], "r2v")
        self.assertEqual(other_tab["workflow_variant"], "ref2v_20260920")

    def test_own_controls_override_legacy_main_h3_settings(self):
        request, _ = prompt_book_h3.build_prompt_book_h3_request(
            {"subject_id": "ada", "video_id": "clip", "steps": 20,
             "sampler": "res_multistep", "scheduler": "karras",
             "pass1_split": 14, "second_pass_sigma": 5, "latent_upscale": True,
             "ref2va_model": "selected-ref2v.safetensors", "seed_random": False,
             "seed": 42, "loras": [{"name": "book.safetensors", "strength": 0.7}],
             "h3_generation": {"task": "fl2v", "steps": 4, "sampler": "h3_turbo",
                               "scheduler": "simple", "turbo_enabled": True,
                               "turbo_lora": "turbo.safetensors", "seed": 99}},
            catalog=self.catalog(), baked={"prompt": "Book prompt"})
        self.assertEqual(request["task"], "r2v")
        self.assertEqual(request["workflow_variant"], "ref2v_20260920")
        self.assertEqual((request["steps"], request["sampler"], request["scheduler"]),
                         (20, "res_multistep", "karras"))
        self.assertEqual((request["pass1_split"], request["second_pass_sigma"]), (14, 5))
        self.assertEqual(request["ref2va_model"], "selected-ref2v.safetensors")
        self.assertEqual(request["seed"], 42)
        self.assertFalse(request["turbo_enabled"])
        self.assertNotIn("turbo_lora", request)
        self.assertEqual(request["loras"], [{"name": "book.safetensors", "strength": 0.7}])

    def test_all_catalog_sampling_choices_are_accepted(self):
        for sampler in sorted(prompt_book_h3.H3_SAMPLERS - {"h3_turbo"}):
            request, _ = prompt_book_h3.build_prompt_book_h3_request(
                {"subject_id": "ada", "video_id": "clip", "sampler": sampler},
                catalog=self.catalog(), baked={"prompt": "Book prompt"})
            self.assertEqual(request["sampler"], sampler)
        for scheduler in sorted(prompt_book_h3.H3_SCHEDULERS):
            request, _ = prompt_book_h3.build_prompt_book_h3_request(
                {"subject_id": "ada", "video_id": "clip", "scheduler": scheduler},
                catalog=self.catalog(), baked={"prompt": "Book prompt"})
            self.assertEqual(request["scheduler"], scheduler)

    def test_upload_skips_matching_volume_object_and_verifies_size(self):
        for spec in prompt_book_h3.PROMPT_BOOK_WORKFLOWS.values():
            (self.workflow_root / spec["filename"]).write_text('{"nodes":[]}', encoding="utf-8")
        stored = {}

        class Helper:
            bucket = "vgc3ky6r6y"

            def call(self, name, **kwargs):
                if name == "head_object":
                    size = stored.get(kwargs["Key"])
                    if size is None:
                        raise RuntimeError("missing")
                    return {"ContentLength": size}
                raise AssertionError(name)

            def upload_to_key(self, path, key):
                stored[key] = Path(path).stat().st_size
                return {"key": key}

        first = prompt_book_h3.upload_prompt_book_workflows(Helper())
        self.assertEqual(len(stored), 2)
        self.assertTrue(all(item["uploaded"] and item["on_volume"] for item in first))
        second = prompt_book_h3.upload_prompt_book_workflows(Helper())
        self.assertTrue(all(not item["uploaded"] and item["on_volume"] for item in second))

    def test_run_route_submits_ref2v_without_calling_the_network(self):
        captured = {}

        def fake_submit(endpoint, key, payload, settings):
            captured.update(payload)
            return {"id": "prompt-book-job"}

        def fake_book(path, timeout=60):
            if path.startswith("/api/catalog"):
                return self.catalog()
            if path.startswith("/api/prompt"):
                return {"prompt": "Baked from the book."}
            raise AssertionError(path)

        settings = {
            "runpod_api_key": "key",
            "h3_endpoint_id": "h3-endpoint",
            "s3_endpoint_url": "https://s3.example",
            "s3_access_key_id": "access",
            "s3_secret_access_key": "secret",
            "h3_storage": {"s3_bucket": "h3-bucket", "s3_region": "test-1"},
            "h3": {
                "fl2va_model": "fl.safetensors",
                "ref2va_model": "ref.safetensors",
                "text_encoder": "clip.safetensors",
                "video_vae": "video.safetensors",
                "audio_vae": "audio.safetensors",
                "turbo_lora": "minimax_h3_turbo_v4_step600_ema.safetensors",
            },
        }
        with (
            mock.patch.object(prompt_book_h3, "book_json", side_effect=fake_book),
            mock.patch.object(prompt_book_h3, "upload_prompt_book_workflows") as upload,
            mock.patch.object(media_console, "submit", side_effect=fake_submit),
            mock.patch.object(media_console, "h3_asset_payload", return_value={"volume_path": "/runpod-volume/input/mock"}),
            mock.patch.object(media_console, "record_job_event"),
        ):
            response = self.client.post("/api/run/prompt-book", json={
                "settings": settings,
                "workflow_id": "fast5090",
                "steps": 4,
                "subject_id": "ada",
                "video_id": "clip",
                "loras": [{"name": "creative.safetensors", "strength": 0.75}],
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(captured["task"], "r2v_20260920")
        self.assertEqual(captured["steps"], 4)
        self.assertEqual(captured["model"], prompt_book_h3.BETA5_MODEL)
        self.assertFalse(captured["turbo_enabled"])
        self.assertEqual(captured["sampler"], "euler")
        self.assertFalse(captured["latent_upscale"])
        self.assertEqual(len(captured["references"]), 2)
        self.assertEqual(len(captured["reference_videos"]), 1)
        self.assertEqual(captured["loras"], [{"name": "H3/creative.safetensors", "strength": 0.75}])
        self.assertFalse(captured["rtx_upscale"])
        self.assertNotIn("volume_path", response.json()["prompt_book"])
        self.assertNotIn("secret", json.dumps(response.json()["prompt_book"]))
        upload.assert_not_called()

    def test_prompt_book_reuses_main_h3_worker_generation_payload(self):
        submitted = []
        def fake_submit(endpoint, key, payload, settings):
            submitted.append(payload)
            return {"id": f"job-{len(submitted)}"}

        def fake_book(path, timeout=60):
            return self.catalog() if path.startswith("/api/catalog") else {"prompt": "Book prompt"}

        settings = {
            "runpod_api_key": "key", "h3_endpoint_id": "h3-endpoint",
            "s3_endpoint_url": "https://s3.example", "s3_access_key_id": "access",
            "s3_secret_access_key": "secret",
            "h3_storage": {"s3_bucket": "h3-bucket", "s3_region": "test-1"},
            "h3": {"ref2va_model": "ref.safetensors", "text_encoder": "clip.safetensors",
                   "video_vae": "video.safetensors", "audio_vae": "audio.safetensors"},
        }
        main = {
            "settings": settings, "task": "r2v", "workflow_variant": "ref2v_20260920",
            "use_multi_image": False, "use_larry": False, "turbo_enabled": False,
            "ref2va_model": "ref.safetensors", "text_encoder": "clip.safetensors",
            "video_vae": "video.safetensors", "audio_vae": "audio.safetensors",
            "sampler": "euler", "scheduler": "simple", "steps": 8, "pass1_split": 7,
            "second_pass_sigma": 2, "latent_upscale": True, "rtx_upscale": False,
            "megapixels": 0.2, "final_megapixels": 1.0,
            "attention": "sage", "cache_enabled": True, "cache_threshold": 0.18,
            "seed_random": False, "seed": 12345, "duration": 2,
            "aspect_ratio": "9:16 (Portrait Widescreen)", "filename_prefix": "H3",
            "delivery": "auto", "advanced_json": "{}",
            "photo_paths": [r"C:\stills\ada.jpg", r"C:\stills\ada-2.jpg", "", ""],
            "keyframe_positions": ["", "", "", ""],
            "reference_video_paths": [r"C:\clips\clip.mp4"],
            "reference_video_audio": [False],
            "reference_video_settings": [{"force_rate": 24, "start_frame": 0,
                                          "select_every_nth": 1, "frame_load_cap": 56}],
            "reference_audio_paths": [], "use_reference_audio_as_output": False,
            "prompt": "Book prompt", "loras": [{"name": "main.safetensors", "strength": 0.6}],
        }
        with (
            mock.patch.object(prompt_book_h3, "book_json", side_effect=fake_book),
            mock.patch.object(prompt_book_h3, "video_has_audio", return_value=False),
            mock.patch.object(media_console, "submit", side_effect=fake_submit),
            mock.patch.object(media_console, "h3_asset_payload", return_value={"volume_path": "/runpod-volume/input/mock"}),
            mock.patch.object(media_console, "record_job_event"),
        ):
            normal = self.client.post("/api/run/h3", json=main)
            book = self.client.post("/api/run/prompt-book", json={
                "settings": settings, "subject_id": "ada", "video_id": "clip",
                "h3_generation": main, "duration": 2,
                "aspect_ratio": "9:16 (Portrait Widescreen)",
                "loras": [{"name": "extra.safetensors", "strength": 0.8}],
            })
        self.assertEqual(normal.status_code, 200, normal.text)
        self.assertEqual(book.status_code, 200, book.text)
        self.assertEqual(len(submitted), 2)
        for field in ("task", "workflow_variant", "model", "clip", "video_vae", "audio_vae",
                      "sampler", "scheduler", "steps", "pass1_split", "second_pass_sigma",
                      "latent_upscale", "rtx_upscale", "megapixels", "final_megapixels",
                      "attention", "cache_enabled", "cache_threshold", "seed", "turbo_enabled",
                      "references", "reference_videos", "reference_video_audio",
                      "reference_video_settings", "photos", "keyframe_positions"):
            self.assertEqual(submitted[1].get(field), submitted[0].get(field), field)
        self.assertEqual(submitted[1]["loras"], [
            {"name": "H3/main.safetensors", "strength": 0.6},
            {"name": "H3/extra.safetensors", "strength": 0.8},
        ])

    def test_beta5_separate_turbo_is_rejected_for_h3_but_disabled_for_prompt_book(self):
        settings = {
            "runpod_api_key": "key", "h3_endpoint_id": "h3-endpoint",
            "s3_endpoint_url": "https://s3.example", "s3_access_key_id": "access",
            "s3_secret_access_key": "secret",
            "h3_storage": {"s3_bucket": "h3-bucket", "s3_region": "test-1"},
            "h3": {"text_encoder": "clip.safetensors", "video_vae": "video.safetensors",
                   "audio_vae": "audio.safetensors"},
        }
        generation = {
            "task": "r2v", "workflow_variant": "ref2v_20260920",
            "ref2va_model": prompt_book_h3.BETA5_MODEL,
            "turbo_enabled": True, "use_larry": True,
            "turbo_family": "larry", "sampler": "h3_turbo",
            "turbo_lora": "minimax_h3_turbo_v4_step600_ema.safetensors",
            "prompt": "Test", "photo_paths": [r"C:\stills\ada.jpg"],
        }
        with (
            mock.patch.object(prompt_book_h3, "book_json", side_effect=lambda path, timeout=60:
                              self.catalog() if path.startswith("/api/catalog") else {"prompt": "Book prompt"}),
            mock.patch.object(media_console, "h3_asset_payload", return_value={"volume_path": "/runpod-volume/input/mock"}) as asset,
            mock.patch.object(media_console, "submit", return_value={"id": "book-job"}) as submit,
            mock.patch.object(media_console, "record_job_event"),
        ):
            normal = self.client.post("/api/run/h3", json={**generation, "settings": settings})
            book = self.client.post("/api/run/prompt-book", json={
                "settings": settings, "subject_id": "ada", "video_id": "clip",
                "h3_generation": generation,
                "sampler": "res_multistep", "scheduler": "karras", "steps": 20,
                "latent_upscale": True, "pass1_split": 14, "second_pass_sigma": 5,
            })
        self.assertEqual(normal.status_code, 400, normal.text)
        self.assertIn("already accelerated", normal.text)
        self.assertEqual(book.status_code, 200, book.text)
        self.assertTrue(asset.called)
        self.assertEqual(submit.call_count, 1)
        payload = submit.call_args.args[2]
        self.assertFalse(payload["turbo_enabled"])
        self.assertEqual((payload["sampler"], payload["scheduler"], payload["steps"]),
                         ("res_multistep", "karras", 20))
        self.assertEqual((payload["pass1_split"], payload["second_pass_sigma"]), (14, 5))
        self.assertTrue(payload["latent_upscale"])
        self.assertNotIn("turbo_lora", payload)


if __name__ == "__main__":
    unittest.main()
