#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CUSTOM_ALLOWED = {
    "ImageResizeKJv2",
    "MiniMaxH3FirstBlockCache",
    "MiniMaxH3MemoryEfficientSageAttentionPatch",
    "MiniMaxH3TurboLoRA",
    "MiniMaxH3TurboSampler",
}


def validate(path: Path) -> None:
    workflow = json.loads(path.read_text(encoding="utf-8"))
    ids = set(workflow)
    classes = {node["class_type"] for node in workflow.values()}
    assert "SolAttnMiniMaxH3" not in classes
    assert "Power Lora Loader (rgthree)" not in classes
    assert "SaveVideo" in classes
    assert workflow["124"]["inputs"]["steps"] == 6
    assert workflow["128"]["inputs"]["clip_name"] == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    for node_id, node in workflow.items():
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in ids, f"{path.name}: {node_id} refers to missing {value[0]}"
    custom = {name for name in classes if name in CUSTOM_ALLOWED}
    assert custom <= CUSTOM_ALLOWED


for workflow_path in sorted((ROOT / "workflows").glob("*.json")):
    validate(workflow_path)
    print(f"OK {workflow_path.name}")

