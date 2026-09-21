"""Bounded RunPod worker-log reads. No infrastructure mutations."""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


def safe_id(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,150}', value):
        raise ValueError('Invalid RunPod identifier')
    return value


def redact(text, secrets=()):
    text = str(text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, '[redacted]')
    text = re.sub(r'https?://[^\s"<>]+', '[URL]', text)
    return re.sub(r'(?i)(bearer\s+|(?:token|password|secret|api_key)\s*[=:]\s*)[^\s,;]+', r'\1[redacted]', text)[:2000]


def worker_logs(endpoint, worker, key, since=''):
    return endpoint_worker_logs(endpoint, key, worker=worker, since=since)


def _pod_log_line(value, key):
    """Normalize one RunPod Pod log record without exposing credentials."""
    if not isinstance(value, dict):
        return None
    source = str(value.get('source') or 'container')
    line = value.get('line', value.get('message', value.get('msg', '')))
    timestamp = value.get('ts', value.get('timestamp', value.get('time', '')))
    if isinstance(line, (dict, list)):
        line = json.dumps(line, ensure_ascii=False)
    result = {
        'source': redact(source, [key]),
        'line': redact(line, [key]),
        'ts': redact(timestamp, [key]),
    }
    worker_id = value.get('workerId', value.get('worker_id', ''))
    if worker_id:
        result['worker_id'] = redact(worker_id, [key])
    return result


def _stream_log_records(url, key, params, tail=200, max_wait_ms=5000):
    """Read JSON-lines or SSE records from a RunPod v2 log stream."""
    if params.get('source') == 'both':
        rows, failures = [], []
        with ThreadPoolExecutor(max_workers=2) as pool:
            reads = [pool.submit(_stream_log_records, url, key, {**params, 'source': source}, tail, max_wait_ms) for source in ('container', 'system')]
            for read in reads:
                try:
                    rows.extend(read.result())
                except requests.RequestException as exc:
                    failures.append(exc)
        if failures and not rows:
            raise failures[0]
        return sorted(rows, key=lambda row: str(row.get('ts', '')))[-tail:]
    try:
        max_wait_ms = max(250, min(30000, int(max_wait_ms)))
    except (TypeError, ValueError):
        max_wait_ms = 5000
    try:
        tail = max(1, min(5000, int(tail)))
    except (TypeError, ValueError):
        tail = 200
    read_timeout = max(5, min(31, max_wait_ms / 1000 + 1))
    lines, event = [], []
    deadline = time.monotonic() + (max_wait_ms / 1000)
    with requests.get(
        url,
        headers={'Authorization': f'Bearer {key}', 'Accept': 'application/x-ndjson, application/json, text/event-stream'},
        params=params,
        stream=True,
        timeout=(5, read_timeout),
    ) as response:
        response.raise_for_status()
        try:
            for raw in response.iter_lines(chunk_size=1):
                if time.monotonic() >= deadline or len(lines) >= tail:
                    break
                text = raw.decode('utf-8', 'replace') if isinstance(raw, bytes) else str(raw)
                stripped = text.strip()
                if not stripped:
                    if event:
                        try:
                            parsed = json.loads('\n'.join(event))
                        except ValueError:
                            parsed = None
                        normalized = _pod_log_line(parsed, key)
                        if normalized:
                            lines.append(normalized)
                        event = []
                    continue
                if stripped.startswith(('id:', 'event:', 'retry:', ':')):
                    continue
                if stripped.startswith('data:'):
                    event.append(stripped[5:].strip())
                    continue
                try:
                    parsed = json.loads(stripped)
                except ValueError:
                    parsed = {'source': params.get('source', 'both'), 'line': stripped}
                normalized = _pod_log_line(parsed, key)
                if normalized:
                    lines.append(normalized)
        except requests.RequestException:
            if not lines and not event:
                raise
        if event and len(lines) < tail:
            try:
                parsed = json.loads('\n'.join(event))
            except ValueError:
                parsed = None
            normalized = _pod_log_line(parsed, key)
            if normalized:
                lines.append(normalized)
    return lines[-tail:]


def pod_logs(pod_id, key, source='both', tail=200, since='', max_wait_ms=5000):
    """Read bounded container/system logs from a RunPod Pod.

    Pod logs are newline-delimited JSON on the REST v2 API. Some deployments
    proxy the endpoint as SSE, so the parser accepts both forms. The deadline
    is intentional: a quiet Pod or a follow-like response must never hold up
    the console request indefinitely.
    """
    pod_id = safe_id(str(pod_id or ''))
    source = str(source or 'both').strip().lower()
    if source not in {'container', 'system', 'both'}:
        raise ValueError('Invalid Pod log source')
    try:
        tail = max(1, min(5000, int(tail)))
    except (TypeError, ValueError):
        tail = 200
    params = {'source': source, 'tail': tail}
    if since:
        params['since'] = str(since)[:80]
    url = f'https://api.runpod.io/v2/pods/{pod_id}/logs'
    return _stream_log_records(url, key, params, tail=tail, max_wait_ms=max_wait_ms)


def list_endpoint_workers(endpoint, key):
    """Return worker IDs from the v2 endpoint-worker listing."""
    endpoint = safe_id(str(endpoint or ''))
    url = f'https://api.runpod.io/v2/serverless/{endpoint}/workers'
    with requests.get(url, headers={'Authorization': f'Bearer {key}', 'Accept': 'application/json'}, timeout=(5, 15)) as response:
        response.raise_for_status()
        body = response.json()
    if isinstance(body, list):
        records = body
    elif isinstance(body, dict):
        records = body.get('items') or body.get('workers') or body.get('data') or []
    else:
        records = []
    result = []
    for item in records:
        value = item if isinstance(item, str) else (item.get('id') or item.get('workerId') or item.get('worker_id')) if isinstance(item, dict) else ''
        try:
            result.append(safe_id(str(value)))
        except ValueError:
            continue
    return list(dict.fromkeys(result))


def endpoint_worker_logs(endpoint, key, worker='', source='both', tail=200, since='', max_wait_ms=5000, worker_ids=None):
    """Read actual serverless endpoint worker logs.

    Omitting ``worker`` lists the endpoint's workers and reads each worker's
    stream concurrently. Supplying one pins the stream to the worker that
    owns a selected job. RunPod's v2 worker-log operation is per-worker; the
    endpoint-wide view is assembled here so the UI can show the fleet.
    """
    endpoint = safe_id(str(endpoint or ''))
    worker = safe_id(str(worker)) if worker else ''
    source = str(source or 'both').strip().lower()
    if source not in {'container', 'system', 'both'}:
        raise ValueError('Invalid worker log source')
    try:
        tail = max(1, min(5000, int(tail)))
    except (TypeError, ValueError):
        tail = 200
    params = {'source': source, 'tail': tail}
    if since:
        params['since'] = str(since)[:80]
    if worker:
        url = f'https://api.runpod.io/v2/serverless/{endpoint}/workers/{worker}/logs'
        return [{**line, 'worker_id': worker} for line in _stream_log_records(url, key, params, tail=tail, max_wait_ms=max_wait_ms)]
    ids = list_endpoint_workers(endpoint, key) if worker_ids is None else list(worker_ids)
    ids = [safe_id(str(value)) for value in ids][:32]
    if not ids:
        return []
    def read_one(worker_id):
        url = f'https://api.runpod.io/v2/serverless/{endpoint}/workers/{worker_id}/logs'
        return [{**line, 'worker_id': worker_id} for line in _stream_log_records(url, key, params, tail=tail, max_wait_ms=max_wait_ms)]
    lines = []
    failures = []
    with ThreadPoolExecutor(max_workers=min(8, len(ids))) as pool:
        futures = {pool.submit(read_one, worker_id): worker_id for worker_id in ids}
        for future in as_completed(futures):
            try:
                lines.extend(future.result())
            except requests.RequestException as exc:
                failures.append(exc)
    if failures and not lines:
        raise failures[0]
    return sorted(lines, key=lambda line: str(line.get('ts', '')))[-tail:]
