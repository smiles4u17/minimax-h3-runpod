"""Install pinned acceleration weights into the console's configured H3 volume.

Writes persistent progress without credentials. Verifies source SHA256 before
completing multipart uploads, Content-MD5 on each part, and final object size.
Existing objects are verified and never overwritten.
"""
import base64
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import app

MANIFEST = ROOT / "endpoint/minimax-h3-runpod/assets/h3_lightx2v_current.json"
if not MANIFEST.exists():
    MANIFEST = ROOT.parent / "assets/h3_lightx2v_current.json"
STATE = ROOT / "outputs/h3_lightx_current_install.json"
CHUNK = 32 * 1024 * 1024


def main():
    items = json.loads(MANIFEST.read_text())
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    s3 = app.require_h3_s3()
    download_settings = app.settings().get("model_download") or {}

    def save(key, **fields):
        state.setdefault(key, {}).update(fields)
        STATE.parent.mkdir(exist_ok=True)
        temp = STATE.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(STATE)

    for item in items:
        key = item["key"]
        try:
            head = s3.client.head_object(Bucket=s3.bucket, Key=key)
        except Exception as exc:
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
            head = None
        if head:
            if head["ContentLength"] != item["size"]:
                raise RuntimeError(f"Existing object has wrong size; refusing to overwrite {key}")
            if head.get("Metadata", {}).get("sha256") != item["sha256"]:
                digest = hashlib.sha256()
                body = s3.client.get_object(Bucket=s3.bucket, Key=key)["Body"]
                try:
                    for chunk in body.iter_chunks(CHUNK):
                        digest.update(chunk)
                finally:
                    body.close()
                if digest.hexdigest() != item["sha256"]:
                    raise RuntimeError(f"Existing object checksum mismatch: {key}")
            save(key, status="VERIFIED", bytes=item["size"], sha256=item["sha256"], bucket=s3.bucket)
            print("Verified existing " + key, flush=True)
            continue

        url = f'https://huggingface.co/{item["repo"]}/resolve/{item["revision"]}/{item["file"]}'
        upload = s3.client.create_multipart_upload(Bucket=s3.bucket, Key=key,
            Metadata={"sha256": item["sha256"], "source-revision": item["revision"]})["UploadId"]
        save(key, status="TRANSFERRING", upload_id=upload, total=item["size"], bytes=0, bucket=s3.bucket)
        print("Installing " + key, flush=True)

        def send(number, chunk):
            md5 = base64.b64encode(hashlib.md5(chunk).digest()).decode()
            for attempt in range(4):
                try:
                    result = s3.client.upload_part(Bucket=s3.bucket, Key=key, UploadId=upload,
                        PartNumber=number, Body=chunk, ContentMD5=md5)
                    return {"PartNumber": number, "ETag": result["ETag"]}
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)

        digest = hashlib.sha256()
        count = 0
        parts = []
        pending = []
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                with app.scoped_model_get(url, download_settings, stream=True,
                        headers={"Accept-Encoding": "identity"}, timeout=(30, 3600)) as response:
                    response.raise_for_status()
                    for number, chunk in enumerate(response.iter_content(CHUNK), 1):
                        if not chunk:
                            continue
                        digest.update(chunk)
                        count += len(chunk)
                        if count > item["size"]:
                            raise RuntimeError("Source exceeded declared size")
                        pending.append(pool.submit(send, number, chunk))
                        if len(pending) >= 4:
                            parts.append(pending.pop(0).result())
                        save(key, bytes=count, uploaded_parts=len(parts))
                    parts.extend(f.result() for f in pending)
            if count != item["size"] or digest.hexdigest() != item["sha256"]:
                raise RuntimeError("Source size or SHA256 mismatch; object was not committed")
            s3.client.complete_multipart_upload(Bucket=s3.bucket, Key=key, UploadId=upload,
                MultipartUpload={"Parts": sorted(parts, key=lambda p:p["PartNumber"])})
            head = s3.client.head_object(Bucket=s3.bucket, Key=key)
            if head["ContentLength"] != count:
                raise RuntimeError("Final object size verification failed")
            # RunPod's gateway may discard user metadata. Verify the stored bytes
            # rather than treating missing metadata as corrupt weights.
            remote_digest = hashlib.sha256()
            body = s3.client.get_object(Bucket=s3.bucket, Key=key)["Body"]
            try:
                for chunk in body.iter_chunks(CHUNK):
                    remote_digest.update(chunk)
            finally:
                body.close()
            if remote_digest.hexdigest() != item["sha256"]:
                raise RuntimeError("Stored object SHA256 verification failed")
            save(key, status="VERIFIED", bytes=count, sha256=item["sha256"], upload_id=None)
            print("Verified " + key, flush=True)
        except Exception as exc:
            save(key, status="FAILED", error=type(exc).__name__ + ": " + str(exc))
            try:
                s3.client.abort_multipart_upload(Bucket=s3.bucket, Key=key, UploadId=upload)
            except Exception as abort_error:
                code = str(getattr(abort_error, "response", {}).get("Error", {}).get("Code", ""))
                if code != "NoSuchUpload":
                    raise
            raise


if __name__ == "__main__":
    main()
