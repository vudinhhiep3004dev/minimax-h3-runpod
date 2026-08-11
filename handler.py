"""Runpod Serverless handler for MiniMax H3 video + audio workflows."""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any

import runpod

import h3_pipeline
from cost import compute_cost, format_cli_summary
from h3_pipeline import (
    PipelineError,
    detect_gpu_name,
    generate,
    is_model_loaded,
    load_pipeline,
    normalize_workflow,
)
from upload import UploadError, upload_video
from upscale import UpscaleError, upscale_to_tiktok

# Reduce CUDA allocator fragmentation during large component swaps.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# Worker-process cold flag: True until first successful model load completes.
_WORKER_STARTED_AT = time.perf_counter()
_FIRST_REQUEST = True
_MODEL_INIT_AT_LOAD = 0.0


def _configure_hf_cache() -> None:
    """Prefer network volume cache; fall back to local container disk."""
    for candidate in (
        "/runpod-volume/huggingface-cache",
        "/opt/h3-data/huggingface-cache",
        os.environ.get("HF_HOME") or "",
    ):
        if not candidate:
            continue
        root = Path(candidate)
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            os.environ["HF_HOME"] = str(root)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(root)
            os.environ["TRANSFORMERS_CACHE"] = str(root)
            print(f"[handler] HF cache -> {root}", flush=True)
            return
        except OSError as exc:
            print(f"[handler] HF cache unusable at {root}: {exc}", flush=True)


_configure_hf_cache()


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _first_value(raw: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = raw.get(name)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _normalize_references(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("references must be a non-empty list")
    normalized: list[Any] = []
    for index, item in enumerate(value):
        if isinstance(item, (str, os.PathLike)):
            if not str(item).strip():
                raise ValueError(f"references[{index}] must not be empty")
            normalized.append({"type": "image", "url": str(item).strip()})
            continue
        if not isinstance(item, dict):
            raise ValueError(f"references[{index}] must be a URL/path or object")
        source = item.get("url") or item.get("uri") or item.get("path")
        if source is None or not str(source).strip():
            raise ValueError(f"references[{index}] needs url, uri, or path")
        copy = dict(item)
        copy["url"] = str(source).strip()
        normalized.append(copy)
    return normalized


def _validate_and_normalize(job_input: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(job_input, dict):
        raise ValueError("input must be a JSON object")

    # Optional packed batch: run sequentially on this warm worker.
    if "jobs" in job_input:
        jobs = job_input["jobs"]
        if not isinstance(jobs, list) or not jobs:
            raise ValueError("jobs must be a non-empty list")
        return [_normalize_one(j, index=i) for i, j in enumerate(jobs)]

    return [_normalize_one(job_input)]


def _normalize_one(raw: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"job{'' if index is None else f'[{index}]'} must be an object")

    prompt = raw.get("prompt")
    if not prompt or not str(prompt).strip():
        raise ValueError("prompt is required and must be a non-empty string")

    duration = float(raw.get("duration", 10))
    if duration < 4 or duration > 15:
        raise ValueError("duration must be between 4 and 15 seconds")

    aspect_ratio = str(raw.get("aspect_ratio", "9:16"))
    preset = str(
        raw.get("resolution_preset")
        or raw.get("quality")
        or os.environ.get("H3_DEFAULT_PRESET", "native")
    )
    seed = raw.get("seed")
    if seed is not None:
        seed = int(seed)

    steps = raw.get("num_inference_steps")
    if steps is not None:
        steps = int(steps)
        if steps < 4 or steps > 50:
            raise ValueError("num_inference_steps must be between 4 and 50")

    image = _first_value(raw, "image", "image_url", "first_frame", "first_frame_url")
    last_image = _first_value(raw, "last_image", "last_image_url", "last_frame", "last_frame_url")
    references = _normalize_references(raw.get("references"))

    requested = raw.get("workflow") or raw.get("mode")
    requested_text = str(requested).strip().lower() if requested is not None else ""
    has_keyframes = image is not None or last_image is not None
    has_references = references is not None

    if not requested_text:
        if has_references:
            workflow = "ref2va"
        elif has_keyframes:
            workflow = "fl2va"
        else:
            workflow = "t2va"
    else:
        workflow = normalize_workflow(requested_text)

    if workflow == "t2va" and (has_keyframes or has_references):
        raise ValueError("t2va cannot be combined with image, last_image, or references")
    if workflow == "fl2va":
        if has_references:
            raise ValueError("fl2va accepts image/last_image, not references")
        if not has_keyframes:
            raise ValueError("fl2va requires image, last_image, or both")
        if requested_text in {"i2v", "i2va"} and image is None:
            raise ValueError("i2va requires image/first_frame")
        if requested_text in {"l2v", "l2va"} and last_image is None:
            raise ValueError("l2va requires last_image/last_frame")
    if workflow == "ref2va":
        if has_keyframes:
            raise ValueError("ref2va accepts references, not image/last_image")
        if not has_references:
            raise ValueError("ref2va requires references")

    if requested_text in {"i2v", "i2va"}:
        mode = "i2va"
    elif requested_text in {"l2v", "l2va"}:
        mode = "l2va"
    elif workflow == "fl2va" and image is not None and last_image is not None:
        mode = "fl2va"
    elif workflow == "fl2va" and image is not None:
        mode = "i2va"
    elif workflow == "fl2va":
        mode = "l2va"
    else:
        mode = workflow

    return {
        "prompt": str(prompt).strip(),
        "workflow": workflow,
        "mode": mode,
        "image": image,
        "last_image": last_image,
        "references": references,
        "duration": duration,
        "aspect_ratio": aspect_ratio,
        "resolution_preset": preset,
        "seed": seed,
        "num_inference_steps": steps,
        "upscale": _as_bool(raw.get("upscale"), True),
    }


def _run_one(spec: dict[str, Any], *, worker_cold: bool, model_init_seconds: float) -> dict[str, Any]:
    t_total0 = time.perf_counter()
    gpu_name = detect_gpu_name()

    gen = generate(
        prompt=spec["prompt"],
        workflow=spec["workflow"],
        duration=spec["duration"],
        aspect_ratio=spec["aspect_ratio"],
        resolution_preset=spec["resolution_preset"],
        seed=spec["seed"],
        num_inference_steps=spec["num_inference_steps"],
        image=spec["image"],
        last_image=spec["last_image"],
        references=spec["references"],
    )

    native_w, native_h = gen.width, gen.height
    out_path = gen.video_path
    out_w, out_h = native_w, native_h
    upscale_seconds = 0.0

    if spec["upscale"] and spec["aspect_ratio"] == "9:16":
        t_up = time.perf_counter()
        out_path = upscale_to_tiktok(gen.video_path)
        upscale_seconds = time.perf_counter() - t_up
        out_w, out_h = 1080, 1920
    elif spec["upscale"]:
        # Keep aspect; scale short side toward 1080-class when requested.
        t_up = time.perf_counter()
        if native_w >= native_h:
            out_w, out_h = 1920, 1080
        else:
            out_w, out_h = 1080, 1920
        out_path = upscale_to_tiktok(gen.video_path, width=out_w, height=out_h)
        upscale_seconds = time.perf_counter() - t_up

    t_upl = time.perf_counter()
    uploaded = upload_video(out_path)
    upload_seconds = time.perf_counter() - t_upl

    total_worker_seconds = time.perf_counter() - t_total0
    # Attribute model init only on the cold first request of this worker.
    billed = total_worker_seconds + (model_init_seconds if worker_cold else 0.0)

    cost = compute_cost(
        billed_seconds=billed,
        output_seconds=gen.duration,
        gpu_type=os.environ.get("RUNPOD_GPU_TYPE") or gpu_name,
    )

    summary = format_cli_summary(
        output_seconds=gen.duration,
        native_w=native_w,
        native_h=native_h,
        out_w=out_w,
        out_h=out_h,
        gpu_compute_seconds=billed,
        cost=cost,
    )
    print(summary)

    return {
        "video_url": uploaded["video_url"],
        "workflow": gen.workflow,
        "mode": spec["mode"],
        "reference_count": len(spec["references"] or []),
        "duration": gen.duration,
        "width": out_w,
        "height": out_h,
        "native_width": native_w,
        "native_height": native_h,
        "seed": gen.seed,
        "num_frames": gen.num_frames,
        "num_inference_steps": gen.num_inference_steps,
        "generation_seconds": round(gen.inference_seconds, 3),
        "timing": {
            "worker_cold": worker_cold,
            "model_init_seconds": round(model_init_seconds if worker_cold else 0.0, 3),
            "inference_seconds": round(gen.inference_seconds, 3),
            "upscale_seconds": round(upscale_seconds, 3),
            "upload_seconds": round(upload_seconds, 3),
            "total_worker_seconds": round(total_worker_seconds, 3),
            "billed_seconds_estimate": round(billed, 3),
            "worker_uptime_at_start": round(time.perf_counter() - _WORKER_STARTED_AT, 3),
        },
        "cost": cost,
        "storage": {k: v for k, v in uploaded.items() if k != "video_url"},
        "cli_summary": summary,
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    global _FIRST_REQUEST, _MODEL_INIT_AT_LOAD

    try:
        job_input = job.get("input") or {}
        specs = _validate_and_normalize(job_input)

        worker_cold = _FIRST_REQUEST
        first_workflow = specs[0]["workflow"]
        if not is_model_loaded(first_workflow):
            load_pipeline(first_workflow)
            _MODEL_INIT_AT_LOAD = h3_pipeline.MODEL_INIT_SECONDS
        model_init = _MODEL_INIT_AT_LOAD if worker_cold else 0.0

        results = []
        for i, spec in enumerate(specs):
            # Only the first job in a packed batch pays cold-start attribution.
            cold = worker_cold and i == 0
            results.append(
                _run_one(
                    spec,
                    worker_cold=cold,
                    model_init_seconds=model_init if cold else 0.0,
                )
            )

        _FIRST_REQUEST = False

        if len(results) == 1:
            return results[0]
        return {
            "count": len(results),
            "results": results,
            "timing": {
                "worker_cold": worker_cold,
                "model_init_seconds": round(model_init if worker_cold else 0.0, 3),
            },
        }
    except (ValueError, PipelineError, UpscaleError, UploadError) as exc:
        return {"error": str(exc), "error_type": type(exc).__name__}
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return {
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc()[-4000:],
        }


def _start_background_eager_load() -> None:
    """Optionally warm the model without blocking queue registration.

    Default is off. When enabled, load runs in a daemon thread so
    ``runpod.serverless.start`` can accept jobs immediately. Otherwise
    workers appear ready while stuck downloading H3 and jobs sit in queue.
    """
    if os.environ.get("H3_EAGER_LOAD", "0") != "1":
        return
    if os.environ.get("RUNPOD_LOCAL_TEST") == "1":
        return

    import threading

    def _bg() -> None:
        global _MODEL_INIT_AT_LOAD
        try:
            print("[handler] background eager model load starting", flush=True)
            workflow = normalize_workflow(os.environ.get("H3_WORKFLOW", "t2va"))
            load_pipeline(workflow)
            _MODEL_INIT_AT_LOAD = h3_pipeline.MODEL_INIT_SECONDS
            print(
                f"[handler] background eager model load done ({h3_pipeline.MODEL_INIT_SECONDS:.1f}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[handler] background eager load failed (will retry on first job): {exc}",
                flush=True,
            )

    threading.Thread(target=_bg, name="h3-eager-load", daemon=True).start()


if __name__ == "__main__":
    _start_background_eager_load()
    runpod.serverless.start({"handler": handler})
