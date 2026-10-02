"""Read-only safetensors header and numerical checks for an H3 volume LoRA."""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


def inspect(name, scan=False):
    helper = app.require_h3_s3()
    key = 'models/loras/H3/' + name.removeprefix('H3/')
    def read(byte_range):
        response = helper.call('get_object', Bucket=helper.bucket, Key=key, Range=byte_range)
        try:
            return response['Body'].read()
        finally:
            response['Body'].close()
    length = struct.unpack('<Q', read('bytes=0-7'))[0]
    if not 0 < length < 16 * 1024 * 1024:
        raise ValueError('Invalid safetensors header length')
    header = json.loads(read(f'bytes=8-{length + 7}'))
    metadata = header.pop('__metadata__', {})
    result = {'name': name, 'metadata_keys': sorted(metadata),
              'metadata': {key: value for key, value in metadata.items()
                           if any(part in key.lower() for part in ('base_model', 'alpha', 'rank', 'step', 'format', 'network_module'))},
              'tensor_count': len(header), 'sample': dict(list(header.items())[:3])}
    if scan:
        import numpy as np
        data = read(f'bytes={length + 8}-')
        result['tensor_data_sha256'] = hashlib.sha256(data).hexdigest()
        tensors = []
        for tensor_name, entry in header.items():
            start, end = entry['data_offsets']
            raw = memoryview(data)[start:end]
            dtype = entry['dtype']
            if dtype == 'BF16':
                values = (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32)
            else:
                values = np.frombuffer(raw, dtype={'F16': '<f2', 'F32': '<f4', 'F64': '<f8'}[dtype]).astype(np.float32)
            finite = np.isfinite(values)
            tensors.append({'name': tensor_name, 'shape': entry['shape'], 'dtype': dtype,
                            'nonfinite': int((~finite).sum()),
                            'rms': float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
                            'max_abs': float(np.max(np.abs(values))),
                            **({'scalar': float(values[0])} if values.size == 1 else {})})
        result['nonfinite_total'] = sum(item['nonfinite'] for item in tensors)
        result['largest_tensors'] = sorted(tensors, key=lambda item: item['max_abs'], reverse=True)[:8]
        result['alpha_values'] = sorted({item['scalar'] for item in tensors if 'alpha' in item['name'] and 'scalar' in item})
        result['down_rms_median'] = float(np.median([item['rms'] for item in tensors if 'lora_down' in item['name'] or 'lora_A' in item['name']]))
        result['up_rms_median'] = float(np.median([item['rms'] for item in tensors if 'lora_up' in item['name'] or 'lora_B' in item['name']]))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    parser.add_argument('--stats', action='store_true')
    args = parser.parse_args()
    for name in args.names:
        print(json.dumps(inspect(name, args.stats), indent=2))
