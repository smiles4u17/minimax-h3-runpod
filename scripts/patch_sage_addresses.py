"""Guard the pinned KJ attention implementation's CUDA quantizer inputs."""
from pathlib import Path


def patch(source):
    header = 'def _sageattn_int8_fp8_nhd(qkv, dtype):'
    if source.count(header) != 1:
        raise RuntimeError('Pinned Sage dispatch changed')
    source = source.replace(header, 'from .h3_quant_safety import needs_compact_copy, compact_quant_input\n\n' + header, 1)
    before = '        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km=k.mean(dim=1, keepdim=True), tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)'
    if source.count(before) != 1:
        raise RuntimeError('Pinned Blackwell CUDA quantizer changed')
    source = source.replace(before, '        q = compact_quant_input(q)\n        k = compact_quant_input(k)\n' + before, 1)
    lines = source.splitlines(keepends=True)
    count = 0
    for index in range(len(lines)-1, -1, -1):
        if lines[index].startswith('        v_fp8, v_scale, _ = per_channel_fp8(v,'):
            lines.insert(index, '        v = compact_quant_input(v)\n')
            count += 1
    if count != 3:
        raise RuntimeError('Pinned FP8 CUDA quantizers changed')
    source = ''.join(lines)
    before = '    n = min(transformer_options.get("minimax_head_chunks", 1), self.heads) if isinstance(transformer_options, dict) else 1'
    if source.count(before) != 1:
        raise RuntimeError('Pinned MiniMax head chunk dispatch changed')
    source = source.replace(before, before + '\n    if needs_compact_copy(tuple(v.shape), tuple(v.stride())):\n        n = max(n, min(4, self.heads))\n', 1)
    return source


if __name__ == '__main__':
    path = Path('/comfyui/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py')
    path.write_text(patch(path.read_text()))
    print('Patched Sage Q/K/V address bounds and bounded unsafe head copies')
