#!/usr/bin/env python3
"""Create a stable MiniMax-H3 Serverless endpoint (no Runpod cached-model feature).

Design goals vs the broken earlier setup:
- Never attach Runpod "cached models" (that slot corrupted the prior endpoint).
- Attach a network volume for Hugging Face weights at /runpod-volume.
- workersMax defaults to 3 for throughput; network volume shares HF weights across workers.
- H3_EAGER_LOAD=0 so the worker registers with the queue before loading weights.
- Long idle timeout so the warm model survives a second test prompt.
"""

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

    hf = os.environ.get("HF_TOKEN", "")
    if not hf:
        token_path = Path.home() / ".cache/huggingface/token"
        if token_path.is_file():
            hf = token_path.read_text().strip()

    image = os.environ.get(
        "H3_WORKER_IMAGE",
        "ttl.sh/ruizmr-minimax-h3-runpod:7d",
    )
    dc = os.environ.get("H3_DATACENTER_ID", "US-KS-2")
    volume_gb = int(os.environ.get("H3_NETWORK_VOLUME_GB", "200"))
    workers_max = int(os.environ.get("H3_WORKERS_MAX", "3"))

    # Reuse an existing volume with the same name if present.
    volumes = api("GET", "https://rest.runpod.io/v1/networkvolumes")
    if not isinstance(volumes, list):
        volumes = []
    volume = next(
        (v for v in volumes if v.get("name") == "minimax-h3-hf-cache" and v.get("dataCenterId") == dc),
        None,
    )
    if volume is None:
        print(f"Creating network volume {volume_gb}GB in {dc} ...")
        volume = api(
            "POST",
            "https://rest.runpod.io/v1/networkvolumes",
            {
                "name": "minimax-h3-hf-cache",
                "size": volume_gb,
                "dataCenterId": dc,
            },
        )
    print("VOLUME", volume.get("id"), volume.get("dataCenterId"), f"{volume.get('size')}GB")

    env = {
        "HF_TOKEN": hf,
        "H3_MEMORY_MODE": "auto",
        "H3_EAGER_LOAD": "0",
        "H3_DEFAULT_PRESET": "draft",
        "H3_NUM_INFERENCE_STEPS": "20",
        "H3_MODEL_ID": "MiniMaxAI/MiniMax-H3",
        "H3_WORKFLOW": os.environ.get("H3_WORKFLOW", "t2va"),
        "RUNPOD_GPU_TYPE": "A100",
        "RUNPOD_GPU_RATE_PER_SEC": os.environ.get("RUNPOD_GPU_RATE_PER_SEC", "0.00076"),
        "MINIMAX_RATE_PER_SEC": "0.13",
        "UPSCALER": "ffmpeg",
        "OUTPUT_DIR": "/tmp/h3_outputs",
        "HF_HOME": "/runpod-volume/huggingface-cache",
        "HUGGINGFACE_HUB_CACHE": "/runpod-volume/huggingface-cache",
    }

    tpl = api(
        "POST",
        "https://rest.runpod.io/v1/templates",
        {
            "name": "minimax-h3-stable",
            "imageName": image,
            "isServerless": True,
            "containerDiskInGb": 50,
            "volumeInGb": 0,
            "dockerEntrypoint": [],
            "dockerStartCmd": [],
            "env": env,
            "ports": [],
        },
    )
    print("TEMPLATE", tpl["id"], image)

    ep = api(
        "POST",
        "https://rest.runpod.io/v1/endpoints",
        {
            "name": os.environ.get("H3_ENDPOINT_NAME", "minimax-h3"),
            "templateId": tpl["id"],
            "gpuTypeIds": [
                "NVIDIA A100-SXM4-80GB",
                "NVIDIA A100 80GB PCIe",
                "NVIDIA L40S",
                "NVIDIA RTX 6000 Ada Generation",
            ],
            "gpuCount": 1,
            "workersMin": 0,
            "workersMax": workers_max,
            "idleTimeout": int(os.environ.get("H3_IDLE_TIMEOUT", "600")),
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
            "executionTimeoutMs": 3_600_000,
            "flashboot": True,
            "networkVolumeId": volume["id"],
            "dataCenterIds": [dc],
        },
    )
    endpoint_id = ep["id"]
    print("ENDPOINT", endpoint_id)
    print(
        json.dumps(
            {
                "id": endpoint_id,
                "templateId": ep.get("templateId"),
                "networkVolumeId": ep.get("networkVolumeId"),
                "workersMax": ep.get("workersMax"),
                "idleTimeout": ep.get("idleTimeout"),
                "dataCenterIds": ep.get("dataCenterIds"),
            },
            indent=2,
        )
    )

    env_path = Path(".env")
    lines = env_path.read_text().splitlines() if env_path.is_file() else []
    out: list[str] = []
    seen = False
    for line in lines:
        if line.startswith("RUNPOD_ENDPOINT_ID="):
            out.append(f"RUNPOD_ENDPOINT_ID={endpoint_id}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"RUNPOD_ENDPOINT_ID={endpoint_id}")
    env_path.write_text("\n".join(out) + "\n")
    print("Updated .env RUNPOD_ENDPOINT_ID")
    print(
        "\nIMPORTANT: Do not pause/delete this endpoint and do not attach a Runpod "
        "'cached model' while the first cold job downloads weights onto the volume."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
