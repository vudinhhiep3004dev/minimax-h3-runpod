#!/usr/bin/env python3
"""Local MCP gateway for the MiniMax-H3 Runpod Queue endpoint.

The GPU worker remains a plain Runpod handler.  This process is the safe
orchestration layer used by an MCP client: it turns local media into temporary
presigned R2 URLs, submits a JSON job, waits for completion, and returns the
durable output metadata instead of transferring video bytes through MCP.

Run with ``python mcp_server.py`` after installing ``mcp_requirements.txt``.
Credentials are read from environment variables (or a local ``.env`` file); no
Runpod or R2 key is embedded in a request.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without MCP extras
    raise SystemExit(
        "MCP dependencies are missing. Install them with "
        "pip install -r mcp_requirements.txt"
    ) from exc


mcp = FastMCP("minimax-h3")

_ALIASES = {
    "t2v": "t2va",
    "i2v": "fl2va",
    "i2va": "fl2va",
    "l2v": "fl2va",
    "l2va": "fl2va",
    "r2v": "ref2va",
    "r2va": "ref2va",
    "ref2v": "ref2va",
}


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _canonical_workflow(value: str) -> str:
    workflow = _ALIASES.get(str(value or "t2va").strip().lower(), str(value or "t2va").strip().lower())
    if workflow not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError("workflow must be t2va, i2va, l2va, fl2va, or ref2va")
    return workflow


def _endpoint_id(workflow: str) -> str:
    """Prefer a per-workflow endpoint, then fall back to one shared endpoint."""
    key = f"RUNPOD_ENDPOINT_ID_{workflow.upper()}"
    endpoint = os.environ.get(key) or os.environ.get("RUNPOD_ENDPOINT_ID")
    if not endpoint:
        raise ValueError(
            f"{key} or RUNPOD_ENDPOINT_ID is required to call the H3 endpoint"
        )
    return endpoint


def _runpod_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise ValueError("RUNPOD_API_KEY is required")
    return key


def _request_json(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 120,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Runpod HTTP {exc.code}: {detail[:1000]}") from exc


def _s3_client():
    import boto3
    from botocore.client import Config

    bucket = os.environ.get("S3_BUCKET")
    access = os.environ.get("S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("S3_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not bucket or not access or not secret:
        raise ValueError(
            "Local media needs S3/R2 credentials: S3_BUCKET, S3_ENDPOINT_URL, "
            "S3_ACCESS_KEY_ID, and S3_SECRET_ACCESS_KEY"
        )
    return (
        boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
            region_name=os.environ.get("S3_REGION", "auto"),
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            config=Config(signature_version="s3v4"),
        ),
        bucket,
    )


def _as_worker_media(value: str, label: str) -> str:
    """Return a URL the remote worker can read, uploading local files to R2."""
    value = str(value).strip()
    if not value:
        raise ValueError(f"{label} must not be empty")
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("data:"):
        raise ValueError(f"{label} must be a URL or local path, not an inline data URL")

    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"{label} local file does not exist: {path}")
    client, bucket = _s3_client()
    key = f"h3-inputs/{uuid.uuid4().hex}/{path.name}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(os.environ.get("S3_INPUT_PRESIGN_SECONDS", "86400")),
    )


def _prepare_references(references: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if references is None:
        return None
    prepared: list[dict[str, Any]] = []
    for index, item in enumerate(references):
        if not isinstance(item, dict):
            raise ValueError(f"references[{index}] must be an object")
        source = item.get("url") or item.get("uri") or item.get("path")
        if source is None:
            raise ValueError(f"references[{index}] needs url, uri, or path")
        copy = dict(item)
        copy["url"] = _as_worker_media(str(source), f"references[{index}]")
        copy.pop("uri", None)
        copy.pop("path", None)
        prepared.append(copy)
    return prepared


def _base_url(endpoint_id: str) -> str:
    return f"https://api.runpod.ai/v2/{endpoint_id}"


def _poll_job(
    endpoint_id: str,
    job_id: str,
    api_key: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = _request_json("GET", f"{_base_url(endpoint_id)}/status/{job_id}", api_key)
        state = result.get("status")
        if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            if state == "COMPLETED":
                output = result.get("output", result)
                return output if isinstance(output, dict) else {"output": output}
            return {
                "status": state,
                "job_id": job_id,
                "error": result.get("error") or result.get("output"),
            }
        time.sleep(max(0.25, poll_seconds))
    return {
        "status": "TIMEOUT",
        "job_id": job_id,
        "error": f"job did not finish within {timeout_seconds:g}s",
    }


@mcp.tool()
def h3_generate_video(
    prompt: str,
    workflow: str = "t2va",
    duration: float = 10.0,
    aspect_ratio: str = "9:16",
    resolution_preset: str = "draft",
    seed: int | None = None,
    num_inference_steps: int | None = None,
    upscale: bool = True,
    image: str | None = None,
    last_image: str | None = None,
    references: list[dict[str, Any]] | None = None,
    wait: bool = True,
    timeout_seconds: float = 7200.0,
    poll_seconds: float = 15.0,
) -> dict[str, Any]:
    """Generate one MiniMax-H3 video and return its R2/HTTP URL.

    ``image`` and ``last_image`` are paths or URLs for I2VA/L2VA/FL2VA.
    ``references`` is an ordered list of ``{"type": "image|video|audio",
    "url": "..."}`` for Ref2VA.  Local files are uploaded to the configured
    S3-compatible bucket before the worker is called.
    """
    canonical = _canonical_workflow(workflow)
    endpoint_id = _endpoint_id(canonical)
    api_key = _runpod_key()

    job_input: dict[str, Any] = {
        "prompt": prompt,
        "workflow": workflow,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "resolution_preset": resolution_preset,
        "upscale": upscale,
    }
    if seed is not None:
        job_input["seed"] = seed
    if num_inference_steps is not None:
        job_input["num_inference_steps"] = num_inference_steps
    if image is not None:
        job_input["image"] = _as_worker_media(image, "image")
    if last_image is not None:
        job_input["last_image"] = _as_worker_media(last_image, "last_image")
    prepared = _prepare_references(references)
    if prepared is not None:
        job_input["references"] = prepared

    submitted = _request_json(
        "POST",
        f"{_base_url(endpoint_id)}/run",
        api_key,
        {"input": job_input},
    )
    job_id = submitted.get("id")
    if not job_id:
        return submitted
    if not wait:
        return {"status": submitted.get("status", "IN_QUEUE"), "job_id": job_id, "endpoint_id": endpoint_id}
    result = _poll_job(
        endpoint_id,
        job_id,
        api_key,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    if isinstance(result, dict):
        result.setdefault("job_id", job_id)
        result.setdefault("endpoint_id", endpoint_id)
    return result


@mcp.tool()
def h3_job_status(job_id: str, workflow: str = "t2va") -> dict[str, Any]:
    """Read the status/output envelope for an existing H3 job."""
    endpoint_id = _endpoint_id(_canonical_workflow(workflow))
    return _request_json(
        "GET",
        f"{_base_url(endpoint_id)}/status/{job_id}",
        _runpod_key(),
    )


@mcp.tool()
def h3_cancel_job(job_id: str, workflow: str = "t2va") -> dict[str, Any]:
    """Cancel a queued or running H3 job."""
    endpoint_id = _endpoint_id(_canonical_workflow(workflow))
    return _request_json(
        "POST",
        f"{_base_url(endpoint_id)}/cancel/{job_id}",
        _runpod_key(),
    )


if __name__ == "__main__":
    # FastMCP defaults to stdio, which is the transport expected by Codex and
    # most desktop MCP clients.
    mcp.run()
