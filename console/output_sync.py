"""Incremental, restart-safe catch-up of generated media on configured volumes."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


class OutputSync:
    def __init__(self, state_path, sources, output_root, extensions):
        self.state_path = Path(state_path)
        self.sources = sources
        self.output_root = output_root
        self.extensions = extensions
        self.lock = threading.Lock()
        self.status = {"running": False, "downloaded": 0, "existing": 0, "errors": []}

    def snapshot(self):
        with self.lock:
            return dict(self.status)

    def update(self, **values):
        with self.lock:
            self.status.update(values)

    def start(self):
        with self.lock:
            if self.status["running"]:
                return dict(self.status)
            self.status = {"running": True, "downloaded": 0, "existing": 0, "errors": []}
            threading.Thread(target=self.run, daemon=True, name="output-catch-up").start()
            return dict(self.status)

    def save(self, state):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temp, self.state_path)

    def run(self):
        errors = []
        downloaded = existing = 0
        started = time.time()
        configured = False
        try:
            # A corrupt checkpoint must be visible, never silently reset to today.
            state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {"stores": {}}
            root = Path(self.output_root()).resolve()
            root.mkdir(parents=True, exist_ok=True)
            local = {}
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in self.extensions:
                    local.setdefault(path.stat().st_size, []).append(path)
            hashes = {}
            sources = self.sources()
            configured = bool(sources)
            for name, helper, prefixes, endpoint in sources:
                identity = hashlib.sha256(f"{endpoint}|{helper.bucket}|{root}".encode()).hexdigest()
                store = state["stores"].setdefault(identity, {
                    "since": datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0).timestamp(),
                    "files": {},
                })
                self.save(state)
                since = store["since"]
                failed = False
                seen = set()
                self.update(message=f"Checking {name} outputs…")
                try:
                    for prefix in dict.fromkeys(prefixes):
                        token = None
                        while True:
                            args = {"Bucket": helper.bucket, "Prefix": prefix.rstrip("/") + "/", "MaxKeys": 1000}
                            if token:
                                args["ContinuationToken"] = token
                            page = helper.call("list_objects_v2", **args)
                            for obj in page.get("Contents", []):
                                key = obj["Key"]
                                if key in seen or Path(key).suffix.lower() not in self.extensions or any(part.startswith("_") for part in key.split("/")[:-1]):
                                    continue
                                seen.add(key)
                                record = store["files"].get(key)
                                modified = obj["LastModified"].timestamp()
                                if modified < since and not record:
                                    continue
                                size = int(obj["Size"])
                                etag = str(obj.get("ETag", "")).strip('"')
                                version = f"{etag}|{size}|{modified}"
                                try:
                                    if record and record["version"] == version:
                                        path = Path(record["path"])
                                        if path.is_file() and path.stat().st_size == size:
                                            existing += 1
                                            continue
                                    candidates = local.get(size, [])
                                    match = None
                                    # Single-part S3 ETags let us recognize numbered local outputs without downloading again.
                                    if len(etag) == 32 and all(c in "0123456789abcdef" for c in etag.lower()):
                                        for candidate in candidates:
                                            cache_key = (str(candidate), candidate.stat().st_mtime_ns, "md5")
                                            if cache_key not in hashes:
                                                hashes[cache_key] = digest(candidate, "md5")
                                            if hashes[cache_key] == etag:
                                                match = candidate
                                                break
                                    if match is None:
                                        folder = root / "recovered" / identity[:12]
                                        folder.mkdir(parents=True, exist_ok=True)
                                        target = folder / (hashlib.sha256(f"{key}|{version}".encode()).hexdigest()[:24] + Path(key).suffix.lower())
                                        partial = target.with_suffix(target.suffix + ".part")
                                        try:
                                            helper.download_key(key, partial)
                                            if partial.stat().st_size != size:
                                                raise ValueError("Downloaded size does not match the volume object")
                                            if len(etag) == 32 and digest(partial, "md5") != etag:
                                                raise ValueError("Downloaded checksum does not match the volume object")
                                            fingerprint = digest(partial)
                                            for candidate in candidates:
                                                cache_key = (str(candidate), candidate.stat().st_mtime_ns, "sha256")
                                                if cache_key not in hashes:
                                                    hashes[cache_key] = digest(candidate)
                                                if hashes[cache_key] == fingerprint:
                                                    match = candidate
                                                    break
                                            if match is None:
                                                os.replace(partial, target)
                                                match = target
                                                local.setdefault(size, []).append(target)
                                                downloaded += 1
                                            else:
                                                existing += 1
                                        finally:
                                            partial.unlink(missing_ok=True)
                                    else:
                                        existing += 1
                                    store["files"][key] = {"path": str(match), "version": version}
                                    self.save(state)
                                    self.update(downloaded=downloaded, existing=existing)
                                except Exception as exc:
                                    failed = True
                                    errors.append(f"{name}: download failed ({type(exc).__name__}); refresh to retry.")
                            if not page.get("IsTruncated"):
                                break
                            token = page.get("NextContinuationToken")
                            if not token:
                                raise ValueError("Incomplete storage listing")
                except Exception as exc:
                    failed = True
                    errors.append(f"{name}: storage check failed ({type(exc).__name__}); refresh to retry.")
                if not failed:
                    # Scan start, not finish: objects arriving during the scan remain eligible.
                    store["since"] = max(since, started - 2)
                self.save(state)
        except Exception as exc:
            errors.append(f"Output sync failed ({type(exc).__name__}); refresh to retry.")
        finally:
            self.update(running=False, downloaded=downloaded, existing=existing, errors=errors[:10],
                        message="Output check needs attention." if errors else
                        "Outputs are up to date." if configured else
                        "Configure output storage to enable automatic downloads.")
