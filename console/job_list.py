"""Merge sent-request history with bulk provider status without fetching media."""
import json
from datetime import datetime
from pathlib import Path

TERMINAL = {'COMPLETED', 'FAILED', 'TIMED_OUT', 'CANCELLED'}


def sent_requests(history, log_directory):
    rows = {}
    for event in history:
        endpoint, job = event.get('endpoint_id'), event.get('job_id')
        if not endpoint or not job:
            continue
        row = rows.setdefault((endpoint, job), {'endpoint_id': endpoint, 'job_id': job,
                                               'status': event.get('status', 'UNKNOWN')})
        if event.get('target') not in ('status', None):
            row.setdefault('target', event['target'])
        if event.get('status') == 'SUBMITTED':
            row['submitted_at'] = event.get('time', '')
            row['seed'] = (event.get('debug') or {}).get('seed')
            row['duration'] = (event.get('debug') or {}).get('duration')
    # Receipts stay small; never open the full request JSON or its inline media.
    for path in sorted(Path(log_directory).glob('*_receipt.json'), reverse=True)[:1000]:
        try:
            receipt = json.loads(path.read_text(encoding='utf-8'))
            job = receipt.get('job') or {}
            endpoint = receipt.get('endpoint_id')
            if not endpoint and job.get('id'):
                matches = [ep for ep, job_id in rows if job_id == job['id']]
                endpoint = matches[0] if len(matches) == 1 else None
            if not endpoint or not job.get('id'):
                continue
            row = rows.setdefault((endpoint, job['id']), {'endpoint_id': endpoint, 'job_id': job['id'],
                                                        'status': job.get('status', 'SUBMITTED')})
            row.setdefault('submitted_at', receipt.get('created_at', ''))
            for name, value in receipt.get('generation', {}).items():
                row.setdefault(name, value)
            # Older receipts lack prompt metadata. Their TXT contains scrubbed JSON,
            # so recover text without loading multi-megabyte inline media requests.
            if not row.get('prompt'):
                text_path = path.with_name(path.name.replace('_receipt.json', '.txt'))
                if text_path.exists() and text_path.stat().st_size < 2_000_000:
                    text = text_path.read_text(encoding='utf-8')
                    payload = json.loads(text.split('\nGeneration request:\n', 1)[1]).get('input', {})
                    for name in ('prompt', 'model', 'steps', 'megapixels'):
                        if name in payload:
                            row.setdefault(name, payload[name])
        except (OSError, ValueError, TypeError, AttributeError, IndexError):
            continue
    return rows


def merge_live_requests(rows, provider, endpoint):
    present = set()
    for request in provider:
        job = request.get('id')
        if not job:
            continue
        present.add(job)
        row = rows.setdefault((endpoint, job), {'endpoint_id': endpoint, 'job_id': job})
        status = request.get('status', 'UNKNOWN')
        if row.get('status') in TERMINAL and status not in TERMINAL:
            status = row['status']
        elif status == 'IN_PROGRESS' and isinstance(request.get('output'), dict) and request['output'].get('stage') == 'completed':
            status = 'FINALIZING'
        row.update(status=status, live=True,
                   worker_id=request.get('workerId'), execution_ms=request.get('executionTime'))
    for (row_endpoint, job), row in rows.items():
        if row_endpoint == endpoint and job not in present:
            row['live'] = False
            if row.get('status') not in TERMINAL:
                row['status'] = 'UNAVAILABLE'


def sorted_requests(rows):
    def submitted(row):
        try:
            return datetime.fromisoformat(row.get('submitted_at', '')).timestamp()
        except (ValueError, TypeError):
            return 0
    return sorted(rows.values(), key=submitted, reverse=True)
