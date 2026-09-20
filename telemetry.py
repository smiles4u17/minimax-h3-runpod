"""Job-scoped ComfyUI events, bounded logs and durable volume diagnostics."""
from __future__ import annotations
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import requests
import websocket


def redact(text):
    text = re.sub(r'https?://[^\s"<>]+', '[URL]', str(text))
    return re.sub(r'(?i)(bearer\s+|(?:token|password|secret|api_key)\s*[=:]\s*)[^\s,;]+', r'\1[redacted]', text)[:1600]


def is_oom(text):
    return any(x in str(text).lower() for x in ('out of memory', 'outofmemory', 'cuda error: memory allocation', 'cublas_status_alloc_failed'))


class JobTelemetry:
    def __init__(self, job, root, comfy_url, max_seconds=14400, idle_seconds=1800):
        self.job = job
        self.root = Path(root)
        self.comfy_url = comfy_url
        self.client_id = os.urandom(16).hex()
        self.prompt_id = None
        self.started = self.last_activity = time.monotonic()
        self.max_seconds, self.idle_seconds = max_seconds, idle_seconds
        self.state = {'schema': 1, 'job_id': str(job.get('id', self.client_id)),
                      'worker_id': os.environ.get('RUNPOD_POD_ID', ''), 'stage': 'accepted',
                      'max_seconds': max_seconds, 'idle_seconds': idle_seconds, 'events': []}
        safe_id = re.sub(r'[^A-Za-z0-9_-]', '_', self.state['job_id'])
        self.path = self.root / 'diagnostics' / 'h3' / f'{safe_id}.json'
        self.log_path = Path(os.environ.get('H3_COMFY_LOG', '/tmp/h3-comfy.log'))
        self.log_offset = self.log_path.stat().st_size if self.log_path.exists() else 0
        self.lock = threading.RLock()
        self.publish_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.ws = None
        self.last_publish = 0
        self.error = None
        self.thread = None
        self.node_names = {}

    def stage(self, name, **fields):
        with self.lock:
            self.last_activity = time.monotonic()
            self.state.update(stage=name, **fields)
            if name != 'sampling':
                self.state.pop('step', None)
                self.state.pop('total_steps', None)
            self.state['events'].append({'time': datetime.now(timezone.utc).isoformat(), 'message': name})
            self.state['events'] = self.state['events'][-40:]
        self.publish(force=name != 'sampling')

    def start(self):
        # Connect before /prompt so that even the first model-loading event is captured.
        parsed = urlparse(self.comfy_url)
        url = ('wss' if parsed.scheme == 'https' else 'ws') + '://' + parsed.netloc + '/ws?clientId=' + self.client_id
        try:
            self.ws = websocket.create_connection(url, timeout=5)
            self.ws.settimeout(1)
            self.state['progress_transport'] = 'comfy_websocket'
        except Exception as exc:
            self.state['progress_transport'] = 'logs_and_history'
            self.state['progress_warning'] = redact(exc)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.publish(force=True)

    def consume(self, event):
        data = event.get('data') or {}
        if self.prompt_id and data.get('prompt_id') and data['prompt_id'] != self.prompt_id:
            return
        kind = event.get('type')
        if kind == 'executing' and data.get('node') is not None:
            self.stage('executing', node=str(data['node']), node_name=self.node_names.get(str(data['node']), ''))
        elif kind == 'progress' and data.get('max'):
            self.stage('sampling', step=data.get('value', 0), total_steps=data['max'], node=str(data.get('node') or self.state.get('node', '')))
        elif kind == 'execution_error':
            self.error = redact(data.get('exception_type', '') + ': ' + data.get('exception_message', ''))
            self.stage('error', error=self.error, oom=is_oom(self.error))
        elif kind == 'execution_cached':
            self.last_activity = time.monotonic()

    def _read_logs(self):
        try:
            size = self.log_path.stat().st_size
            if size == self.log_offset:
                return
            if size < self.log_offset:
                self.log_offset = 0
            with self.log_path.open('rb') as f:
                f.seek(max(self.log_offset, size - 16384))
                data = f.read(16384)
                self.log_offset = f.tell()
            lines = [redact(x) for x in data.decode('utf-8', 'replace').replace('\r', '\n').splitlines() if x.strip()]
            if lines:
                with self.lock:
                    self.state['logs'] = (self.state.get('logs', []) + lines)[-100:]
                    self.last_activity = time.monotonic()
                    if is_oom('\n'.join(lines)):
                        self.error = 'ComfyUI reported an out-of-memory error'
                        self.state.update(oom=True, error=self.error)
        except OSError:
            pass

    def _loop(self):
        while not self.stop_event.is_set():
            if self.ws:
                try:
                    raw = self.ws.recv()
                    if isinstance(raw, str) and raw:
                        self.consume(json.loads(raw))
                except websocket.WebSocketTimeoutException:
                    pass
                except Exception:
                    self.state['progress_transport'] = 'logs_and_history'
                    self.ws.close()
                    self.ws = None
            else:
                self.stop_event.wait(1)
            self._read_logs()
            self.publish()

    def publish(self, force=False):
        if not self.publish_lock.acquire(blocking=False):
            return
        try:
            self._publish(force)
        finally:
            self.publish_lock.release()

    def _publish(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_publish < 5:
            return
        self.last_publish = now
        with self.lock:
            self.state.update(updated_at=datetime.now(timezone.utc).isoformat(),
                              elapsed_seconds=round(now-self.started),
                              inactive_seconds=round(now-self.last_activity))
            snapshot = json.loads(json.dumps(self.state))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(snapshot), encoding='utf-8')
            temp.replace(self.path)
        except OSError as exc:
            print('H3 diagnostics write failed: ' + redact(exc), flush=True)
        compact = {k: v for k, v in snapshot.items() if k not in ('logs', 'events', 'models', 'capabilities')}
        print('H3_PROGRESS ' + json.dumps(compact), flush=True)
        # RunPod displays this in its status response and Console job view.
        try:
            import runpod
            runpod.serverless.progress_update(self.job, compact)
        except Exception:
            pass  # Volume snapshot and stdout remain available if the API is unavailable.

    def check(self):
        if self.error:
            raise RuntimeError(self.error)
        now = time.monotonic()
        if now - self.started >= self.max_seconds:
            raise TimeoutError(f'H3 maximum runtime reached ({self.max_seconds}s)')
        if now - self.last_activity >= self.idle_seconds:
            raise TimeoutError(f'ComfyUI produced no progress or log activity for {self.idle_seconds}s')

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.ws:
            self.ws.close()
        self._read_logs()
        self.publish(force=True)


def runtime_inventory(comfy_url, volume_root):
    """The custom node reports the actual loader search paths inside ComfyUI."""
    r = requests.get(comfy_url + '/h3/runtime', timeout=20)
    r.raise_for_status()
    result = r.json()
    if not result.get('volume_only') or not result.get('volume_mounted'):
        raise RuntimeError('H3 model paths are not restricted to the mounted network volume')
    return result
