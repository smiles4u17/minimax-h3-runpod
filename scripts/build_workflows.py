#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


BLACKWELL_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, workflow: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(workflow, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prune(workflow: dict, removed: set[str]) -> dict:
    result = {str(k): copy.deepcopy(v) for k, v in workflow.items() if str(k) not in removed}
    for node in result.values():
        inputs = node.get("inputs", {})
        for name in list(inputs):
            value = inputs[name]
            if isinstance(value, list) and value and str(value[0]) in removed:
                del inputs[name]
    return result


def set_common(workflow: dict, cache_id: str, turbo_id: str = "152") -> None:
    workflow["124"]["inputs"].update(scheduler="simple", steps=6, denoise=1)
    workflow["125"]["inputs"]["sampler"] = ["154", 0]
    workflow["128"]["inputs"]["clip_name"] = BLACKWELL_ENCODER
    workflow[turbo_id]["inputs"]["model"] = ["127", 0]
    workflow[cache_id]["inputs"].update(
        threshold=0.18,
        warmup_steps=1,
        max_consecutive_reuse=1,
        start_percent=0.0,
        end_percent=1.0,
        downsample_factor=1,
        verbose=False,
        model=["145", 0],
    )


def build_fl2v(source: dict) -> dict:
    removed = {"123", "164", "207", "220", "221", "222", "224", "231"}
    workflow = prune(source, removed)
    set_common(workflow, "226")
    workflow["182"]["inputs"]["clip"] = ["128", 0]
    workflow["145"]["inputs"]["model"] = ["152", 0]
    workflow["900"] = {
        "inputs": {
            "filename_prefix": "video/MiniMax_H3_FL2V",
            "format": "auto",
            "codec": "auto",
            "video": ["130", 0],
        },
        "class_type": "SaveVideo",
        "_meta": {"title": "Save Video"},
    }
    return workflow


def build_r2v(source: dict) -> dict:
    removed = {
        "123", "137", "139", "149", "164", "172", "174", "176",
        "177", "185", "208", "212", "213",
    }
    workflow = prune(source, removed)
    set_common(workflow, "210")
    workflow["145"]["inputs"]["model"] = ["152", 0]
    workflow["130"]["inputs"]["audio"] = ["121", 0]
    conditioning = workflow["136"]["inputs"]
    for key in list(conditioning):
        if key.startswith("ref_images.") or key.startswith("ref_audios."):
            del conditioning[key]
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fl2v-source", type=Path, required=True)
    parser.add_argument("--r2v-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output / "fl2v.json", build_fl2v(load(args.fl2v_source)))
    save(args.output / "r2v.json", build_r2v(load(args.r2v_source)))


if __name__ == "__main__":
    main()

