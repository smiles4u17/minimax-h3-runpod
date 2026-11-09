#!/usr/bin/env python3
"""Build gate for the masked/sparse H3 runtime; no GPU allocation or downloads."""
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path

root = Path(os.environ.get("COMFY_ROOT", "/comfyui"))
checks = {
    "comfy_extras/nodes_minimax_h3.py": ['node_id="MiniMaxH3LatentUpscaleWithModel"', 'node_id="MiniMaxH3SigmaShift"', 'io.Image.Input("first_frame", optional=True)'],
    "comfy_extras/nodes_hunyuan.py": ['node_id="LatentUpscaleModelLoader"', "MiniMaxH3LatentUpscaler"],
    "comfy_extras/nodes_context_windows.py": ['"MiniMaxH3ContextWindows"', "default=124", "default=39"],
    "comfy/sd.py": ["noncausal_decoder=noncausal"],
    "comfy/ldm/minimax/model.py": ["out[0] = out[0] * denoise_mask", "out[1] = out[1] * audio_denoise_mask", "sample_sigmas"],
    "comfy/model_management.py": ["RAM limited by cgroup"],
    "comfy_extras/nodes_sparse_attention.py": ['node_id="BlockSparseAttention"', '"sla"', '"sol-attn"', '"vsa"'],
    "custom_nodes/h3_pdd/pdd_node.py": ['H3PDDAcceleration', 'strict_schedule'],
}
for filename, markers in checks.items():
    source = (root / filename).read_text(encoding="utf-8")
    for marker in markers:
        assert marker in source, f"Missing H3 runtime feature: {filename}: {marker}"
for filename in ('h3_silu_temb_grid_fl2va.safetensors', 'h3_silu_temb_grid_ref2va.safetensors'):
    assert (root / 'custom_nodes/h3_pdd' / filename).stat().st_size > 10000000, f'Missing PDD support grid: {filename}'
# The upstream loader logs import errors instead of raising them. Make them fatal
# during image validation so a green build really includes the required node.
sys.path.insert(0, str(root))
sys.argv.append('--cpu')
import comfy.options
comfy.options.enable_args_parsing()
from comfy_extras.nodes_context_windows import MiniMaxH3ContextWindowsNode
assert MiniMaxH3ContextWindowsNode.define_schema().node_id == 'MiniMaxH3ContextWindows'
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy_extras.nodes_minimax_h3 import MiniMaxH3LatentUpscaleWithModel
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
assert MiniMaxH3LatentUpscaleWithModel.define_schema().node_id == "MiniMaxH3LatentUpscaleWithModel"
node_root = root / 'custom_nodes/h3_pdd'
spec = importlib.util.spec_from_file_location('h3_pdd_gate', node_root / '__init__.py', submodule_search_locations=[str(node_root)])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module._EXTENSION is not None, 'PDD extension failed to import'
for package, version in {"comfy-aimdo": "0.6.0", "comfy-kitchen": "0.2.40", "torch": "2.10.0+cu130", "sageattention": "2.2.0"}.items():
    actual = importlib.metadata.version(package)
    assert actual == version, f"Unvalidated {package}: {actual}; expected {version}"
print("H3 masked, sparse, PDD and cgroup runtime build checks passed")
