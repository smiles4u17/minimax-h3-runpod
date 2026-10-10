"""Persistent local copies of submitted requests; never stores HTTP credentials."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def readable(value, key=''):
    if isinstance(value, dict):
        return {k: (f'[inline media: {len(v)} characters; retained in request JSON]'
                    if k == 'data' and (value.get('type') == 'base64' or 'name' in value) and isinstance(v, str)
                    else readable(v, k)) for k, v in value.items()}
    if isinstance(value, list):
        return [readable(v, key) for v in value]
    if isinstance(value, str) and (key == 'base64' or key.endswith('_base64')):
        return f'[inline media: {len(value)} characters; retained in request JSON]'
    return value


class SubmissionLog:
    def __init__(self, directory, endpoint, body, source=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.created = datetime.now(timezone.utc).isoformat()
        self.endpoint = endpoint
        self.generation = {k: body['input'][k] for k in ('task', 'filename_prefix', 'seed', 'duration', 'prompt', 'model', 'steps', 'megapixels', 'aspect_ratio', 'sampler', 'latent_upscale', 'rtx_upscale') if k in body['input']}
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex
        self.text_path = directory / (name + '.txt')
        self.receipt_path = directory / (name + '_receipt.json')
        request_path = directory / (name + '_request.json')
        # Save before sending any paid request, including embedded media and the final seed.
        request_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')
        self.header = f'Submitted at (UTC): {self.created}\nEndpoint: {endpoint}\nExact request: {request_path.name}\n'
        if source:
            self.header += '\nSource media:\n' + json.dumps(source, ensure_ascii=False, indent=2) + '\n'
        self.header += '\nGeneration request:\n' + json.dumps(readable(body), ensure_ascii=False, indent=2) + '\n'
        self.record('SUBMITTING')

    def record(self, status, job=None, error=None):
        receipt = {'created_at': self.created, 'updated_at': datetime.now(timezone.utc).isoformat(),
                   'status': status, 'job': job, 'error': error,
                   'endpoint_id': self.endpoint, 'generation': self.generation}
        self.receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        self.text_path.write_text(f'Status: {status}\nJob ID: {(job or {}).get("id", "")}\n'
                                  + (f'Error: {error}\n' if error else '') + self.header, encoding='utf-8')

    def finish(self, status, job=None, error=None):
        try:
            self.record(status, job, error)
        except OSError as exc:
            # The exact request was saved before submission; don't hide an accepted job.
            print(f'Generation receipt could not be updated: {self.text_path}: {exc}')
