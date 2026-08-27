from __future__ import annotations

import base64
import importlib.util
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path


sys.modules.setdefault("runpod", types.SimpleNamespace(serverless=types.SimpleNamespace(start=lambda _: None)))
sys.modules.setdefault(
    "requests",
    types.SimpleNamespace(
        get=lambda *args, **kwargs: None,
        post=lambda *args, **kwargs: None,
        put=lambda *args, **kwargs: None,
        RequestException=Exception,
    ),
)
SPEC = importlib.util.spec_from_file_location("h3_handler", Path(__file__).parents[1] / "handler.py")
handler = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(handler)


def asset(name: str) -> dict[str, str]:
    return {"name": name, "data": base64.b64encode(b"test-asset").decode("ascii")}


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        handler.COMFY_ROOT = root
        handler.INPUT_DIR = root / "input"
        handler.OUTPUT_DIR = root / "output"
        handler.TEMPLATE_DIR = Path(__file__).parents[1] / "workflows"
        handler.INPUT_DIR.mkdir(parents=True)
        handler.OUTPUT_DIR.mkdir(parents=True)
        encoder = root / "models" / "text_encoders" / handler.BLACKWELL_ENCODER
        encoder.parent.mkdir(parents=True)
        encoder.touch()
        for folder, names in {
            "diffusion_models": [
                "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
                "alternate_h3.safetensors",
            ],
            "vae": [
                "minimax_h3_video_vae_fp16.safetensors",
                "minimax_h3_audio_vae_fp32.safetensors",
                "alternate_video_vae.safetensors",
            ],
            "loras": ["H3/minimax_h3_turbo_v4_step600_ema.safetensors"],
        }.items():
            for name in names:
                path = root / "models" / folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        handler._gpu_info = lambda: ("RTX 5090", (12, 0))
        handler._sage_available = lambda: True

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fl2v_native_last_frame_and_lora(self) -> None:
        workflow, metadata = handler.build_preset({
            "task": "fl2v",
            "prompt": "test prompt",
            "first_frame": asset("first.png"),
            "last_frame": asset("last.png"),
            "attention": "native",
            "loras": [{"name": "H3/test.safetensors", "strength": 0.75}],
        })
        self.assertNotIn("145", workflow)
        self.assertEqual(workflow["226"]["inputs"]["model"], ["152", 0])
        self.assertEqual(workflow["152"]["inputs"]["model"], ["9100", 0])
        self.assertEqual(workflow["182"]["inputs"]["last_frame"], ["902", 0])
        self.assertEqual(workflow["124"]["inputs"]["steps"], 6)
        self.assertEqual(metadata["attention"], "native")

    def test_turbo_disabled_uses_native_sampler_and_selected_models(self) -> None:
        workflow, metadata = handler.build_preset({
            "task": "fl2v",
            "prompt": "test prompt",
            "first_frame": asset("first.png"),
            "turbo_enabled": False,
            "steps": 20,
            "model": "alternate_h3.safetensors",
            "video_vae": "alternate_video_vae.safetensors",
        })
        self.assertNotIn("152", workflow)
        self.assertNotIn("154", workflow)
        self.assertEqual(workflow["125"]["inputs"]["sampler"], ["9150", 0])
        self.assertEqual(workflow["124"]["inputs"]["scheduler"], "beta")
        self.assertEqual(workflow["124"]["inputs"]["steps"], 20)
        self.assertEqual(workflow["127"]["inputs"]["unet_name"], "alternate_h3.safetensors")
        self.assertEqual(workflow["119"]["inputs"]["vae_name"], "alternate_video_vae.safetensors")
        self.assertFalse(metadata["turbo_enabled"])

    def test_r2v_references_and_audio(self) -> None:
        workflow, metadata = handler.build_preset({
            "task": "r2v",
            "prompt": "test prompt",
            "references": [asset("one.png"), asset("two.png")],
            "audio": asset("voice.wav"),
            "use_reference_audio_as_output": True,
        })
        self.assertEqual(workflow["136"]["inputs"]["ref_images.ref_image_0"], ["8000", 0])
        self.assertEqual(workflow["136"]["inputs"]["ref_images.ref_image_1"], ["8001", 0])
        self.assertEqual(workflow["136"]["inputs"]["ref_audios.ref_audio_0"], ["8501", 0])
        self.assertEqual(workflow["130"]["inputs"]["audio"], ["8501", 0])
        self.assertEqual(metadata["task"], "r2v")

    def test_finds_video_descriptors(self) -> None:
        value = {"92": {"videos": [{"filename": "clip.mp4", "type": "output"}]}}
        self.assertEqual(handler._find_file_descriptors(value)[0]["filename"], "clip.mp4")

    def test_auto_attention_uses_sage_only_on_sm120(self) -> None:
        with mock.patch.dict(handler.os.environ, {"ATTENTION_MODE": "auto"}):
            self.assertEqual(handler._attention_mode("auto", (8, 9)), "native")
            self.assertEqual(handler._attention_mode("auto", (10, 0)), "native")
            self.assertEqual(handler._attention_mode("auto", (12, 0)), "sage")

    def test_blackwell_only_encoder_rejected_on_older_gpu(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "MODEL_PROFILE=dual or universal"):
            handler._select_encoder((8, 9))


if __name__ == "__main__":
    unittest.main()
