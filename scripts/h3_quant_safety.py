"""Address bounds for SageAttention 2.2 CUDA quantization of NHD views."""
import math

UINT32_CAPACITY = 1 << 32
_reported = set()


def needs_compact_copy(shape, strides, padding=128):
    if len(shape) != 4 or len(strides) != 4 or any(n <= 0 for n in shape):
        raise RuntimeError('Sage quantization requires a nonempty four-dimensional NHD tensor')
    padded = list(shape)
    padded[1] = ((padded[1] + padding - 1) // padding) * padding
    if math.prod(padded) > UINT32_CAPACITY:
        raise RuntimeError('Attention exceeds safe CUDA address capacity; reduce reference detail or use native attention')
    max_offset = sum((n-1) * abs(s) for n, s in zip(padded, strides))
    return max_offset >= UINT32_CAPACITY or strides[-1] != 1 or any(s < 0 for s in strides)


def compact_quant_input(tensor):
    shape, strides = tuple(tensor.shape), tuple(tensor.stride())
    if not needs_compact_copy(shape, strides):
        return tensor
    compact = tensor.contiguous()
    if needs_compact_copy(tuple(compact.shape), tuple(compact.stride())):
        raise RuntimeError('Could not create address-safe attention input')
    geometry = (shape, strides)
    if geometry not in _reported:
        print(f'H3_SAGE_ADDRESS_GUARD compacted shape={shape} stride={strides}')
        _reported.add(geometry)
    return compact
