#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path)
    parser.add_argument("--endpoint", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    args = parser.parse_args()
    if not args.endpoint or not args.api_key:
        raise SystemExit("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY.")

    payload = json.loads(args.request.read_text(encoding="utf-8"))
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    base = f"https://api.runpod.ai/v2/{args.endpoint}"
    response = requests.post(f"{base}/run", headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    job_id = response.json()["id"]
    print(f"job: {job_id}")

    while True:
        status = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=60)
        status.raise_for_status()
        body = status.json()
        if body.get("status") in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            print(json.dumps(body, indent=2))
            return
        print(body.get("status", "unknown"))
        time.sleep(5)


if __name__ == "__main__":
    main()

