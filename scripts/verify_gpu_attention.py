"""Small real-GPU correctness gate before the handler accepts any paid work."""
import json
import os
from pathlib import Path
import re
import requests


def verify(base_url, session=requests):
    response = session.get(base_url + '/h3/runtime', timeout=15)
    response.raise_for_status()
    inventory = response.json()
    capability = tuple(inventory.get('cuda_capability', []))
    if capability not in {(8, 9), (12, 0)}:
        return {'gpu': inventory.get('gpu'), 'capability': list(capability),
                'attention': 'native', 'skipped': 'No validated Sage build for this architecture'}
    response = session.post(base_url + '/h3/attention-check', json={'mode': 'smoke'}, timeout=180)
    response.raise_for_status()
    result = response.json()
    if result.get('passed') is not True:
        raise RuntimeError('Sage production attention failed its GPU correctness check: ' + json.dumps(result))
    return result


if __name__ == '__main__':
    result = verify(os.environ.get('COMFY_URL', 'http://127.0.0.1:8188').rstrip('/'))
    print('H3_ATTENTION_VALIDATION ' + json.dumps(result), flush=True)
    worker = re.sub(r'[^A-Za-z0-9_-]', '_', os.environ.get('RUNPOD_POD_ID', 'unknown'))
    root = Path(os.environ.get('RUNPOD_VOLUME_ROOT', '/runpod-volume')) / 'diagnostics/runtime'
    root.mkdir(parents=True, exist_ok=True)
    target = root / (worker + '.json')
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(result))
    temp.replace(target)
