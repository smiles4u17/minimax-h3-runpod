#!/usr/bin/env python3
"""Build gate for the masked/sparse H3 runtime; no GPU allocation or downloads."""
import importlib.metadata
import os
from pathlib import Path

root = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
checks = {
    "comfy/ldm/minimax/model.py": ["out[0] = out[0] * denoise_mask", "out[1] = out[1] * audio_denoise_mask", "sample_sigmas"],
    "comfy/model_management.py": ["RAM limited by cgroup"],
    "comfy_extras/nodes_sparse_attention.py": ['node_id="BlockSparseAttention"', '"sla"', '"sol-attn"', '"vsa"'],
    "comfy_extras/nodes_minimax_h3.py": ['node_id="MiniMaxH3SigmaShift"', 'io.Image.Input("first_frame", optional=True)'],
}
for filename, markers in checks.items():
    source = (root / filename).read_text(encoding="utf-8")
    for marker in markers:
        assert marker in source, f"Missing H3 runtime feature: {filename}: {marker}"
for package, version in {"comfy-aimdo": "0.5.3", "comfy-kitchen": "0.2.33"}.items():
    actual = importlib.metadata.version(package)
    assert actual == version, f"Unvalidated {package}: {actual}; expected {version}"
print("H3 masked, sparse, PDD and cgroup runtime build checks passed")
