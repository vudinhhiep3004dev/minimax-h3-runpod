#!/usr/bin/env python3
"""Create or update the MiniMax H3 Runpod Serverless Queue endpoint via API v2."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_dotenv() -> None:
    path = Path(".env")
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def api(method: str, url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        err = exc.read().decode(errors="replace")
        raise SystemExit(f"HTTP {exc.code} {url}: {err}") from exc


def main() -> int:
    load_dotenv()
    if not os.environ.get("RUNPOD_API_KEY"):
        print("RUNPOD_API_KEY required", file=sys.stderr)
        return 2

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        token_path = Path.home() / ".cache/huggingface/token"
        if token_path.is_file():
            hf_token = token_path.read_text().strip()

    image = os.environ.get(
        "H3_WORKER_IMAGE",
        "ghcr.io/ruizmr/minimax-h3-runpod:latest",
    )

    # Prefer economical 48GB PRO pool, fall back to A100 80GB.
    payload = {
        "name": os.environ.get("H3_ENDPOINT_NAME", "minimax-h3"),
        "image": image,
        "disk": int(os.environ.get("H3_CONTAINER_DISK_GB", "50")),
        "gpu": {
            "pools": ["ADA_48_PRO", "AMPERE_80"],
            "count": 1,
        },
        "workers": {"min": 0, "max": int(os.environ.get("H3_WORKERS_MAX", "2"))},
        "scaling": {
            "type": "QUEUE_DELAY",
            "value": 4,
            "idleTimeout": 120,
        },
        "timeout": 3600_000,
        "flashboot": "ON",
        "env": {
            "HF_TOKEN": hf_token,
            "H3_MEMORY_MODE": os.environ.get("H3_MEMORY_MODE", "auto"),
            "H3_EAGER_LOAD": "1",
            "H3_DEFAULT_PRESET": os.environ.get("H3_DEFAULT_PRESET", "draft"),
            "H3_WORKFLOW": os.environ.get("H3_WORKFLOW", "t2va"),
            "H3_NUM_INFERENCE_STEPS": os.environ.get("H3_NUM_INFERENCE_STEPS", "20"),
            "RUNPOD_GPU_TYPE": os.environ.get("RUNPOD_GPU_TYPE", "L40S"),
            "RUNPOD_GPU_RATE_PER_SEC": os.environ.get("RUNPOD_GPU_RATE_PER_SEC", "0.00053"),
            "MINIMAX_RATE_PER_SEC": os.environ.get("MINIMAX_RATE_PER_SEC", "0.13"),
            "UPSCALER": "ffmpeg",
            "OUTPUT_DIR": "/tmp/h3_outputs",
        },
    }

    # Optional model caching field — try common names if supported by API.
    if os.environ.get("H3_CACHED_MODEL", "1") == "1":
        payload["env"]["H3_MODEL_ID"] = "MiniMaxAI/MiniMax-H3"

    for k in (
        "S3_BUCKET",
        "S3_ENDPOINT_URL",
        "S3_REGION",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_PUBLIC_BASE_URL",
    ):
        if os.environ.get(k):
            payload["env"][k] = os.environ[k]

    print("Creating endpoint with image", image)
    created = api("POST", "https://api.runpod.io/v2/serverless", payload)
    print(json.dumps(created, indent=2))
    endpoint_id = created.get("id")
    if endpoint_id:
        print(f"\nexport RUNPOD_ENDPOINT_ID={endpoint_id}")
        Path(".endpoint_id").write_text(endpoint_id + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
