"""MiniMax H3 ModularPipeline adapter.

The upstream checkpoint exposes three workflows:

* ``t2va``: text to video + audio
* ``fl2va``: first/last keyframe to video + audio (one or both images)
* ``ref2va``: ordered image/video/audio references to video + audio

The worker keeps one workflow loaded at a time.  That is intentional: the
``transformer/`` and ``transformer_ref/`` partitions are separate, and loading
both on every worker would exceed the small network volume used by the default
deployment.  A worker can switch lazily when a request asks for another
workflow, but production deployments should normally dedicate an endpoint (or
an adequately sized volume) to ``ref2va``.
"""

from __future__ import annotations

import os
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

# Timing populated at worker init
MODEL_INIT_SECONDS: float = 0.0
MODEL_LOADED: bool = False
_PIPE = None
_MANAGER = None
_LOADED_WORKFLOW: str | None = None
_PIPE_LOCK = RLock()

WORKFLOWS = ("t2va", "fl2va", "ref2va")
_WORKFLOW_ALIASES = {
    "t2v": "t2va",
    "i2v": "fl2va",
    "i2va": "fl2va",
    "l2v": "fl2va",
    "l2va": "fl2va",
    "r2v": "ref2va",
    "r2va": "ref2va",
    "ref2v": "ref2va",
}


RESOLUTION_PRESETS: dict[str, dict[str, int]] = {
    # All multiples of 32; 9:16 vertical
    "draft": {"width": 544, "height": 960},
    "720p": {"width": 720, "height": 1280},
    "native": {"width": 768, "height": 1344},
}

ASPECT_SIZES: dict[str, dict[str, dict[str, int]]] = {
    "9:16": RESOLUTION_PRESETS,
    "16:9": {
        "draft": {"width": 960, "height": 544},
        "720p": {"width": 1280, "height": 720},
        "native": {"width": 1344, "height": 768},
    },
    "4:3": {
        # Closely matches the supplied poster while preserving the full layout.
        "draft": {"width": 768, "height": 576},
        "720p": {"width": 1024, "height": 768},
        "native": {"width": 1152, "height": 864},
    },
    "1:1": {
        "draft": {"width": 704, "height": 704},
        "720p": {"width": 768, "height": 768},
        "native": {"width": 768, "height": 768},
    },
}

FPS = 24
MIN_DURATION = 5.0
MAX_DURATION = 15.0


class PipelineError(RuntimeError):
    pass


def normalize_workflow(value: str | None) -> str:
    """Return the Diffusers workflow name for a public mode alias."""
    raw = str(value or os.environ.get("H3_WORKFLOW", "t2va")).strip().lower()
    workflow = _WORKFLOW_ALIASES.get(raw, raw)
    if workflow not in WORKFLOWS:
        raise PipelineError(
            f"Unsupported workflow '{value}'. Supported: t2va, fl2va, ref2va "
            "(aliases: i2va, l2va, r2va)."
        )
    return workflow


def loaded_workflow() -> str | None:
    """Return the workflow currently resident in this worker, if any."""
    return _LOADED_WORKFLOW if MODEL_LOADED and _PIPE is not None else None


def is_model_loaded(workflow: str | None = None) -> bool:
    """Avoid importing the mutable ``MODEL_LOADED`` flag by value in callers."""
    if not MODEL_LOADED or _PIPE is None:
        return False
    return workflow is None or _LOADED_WORKFLOW == normalize_workflow(workflow)


@dataclass
class GenerateResult:
    workflow: str
    video_path: Path
    width: int
    height: int
    duration: float
    num_frames: int
    seed: int
    inference_seconds: float
    num_inference_steps: int


def resolve_model_path() -> str:
    """Prefer Runpod cached HF model path when present."""
    override = os.environ.get("H3_MODEL_PATH")
    if override and Path(override).exists():
        return override

    cached_root = Path("/runpod-volume/huggingface-cache/hub/models--MiniMaxAI--MiniMax-H3")
    if cached_root.is_dir():
        snaps = cached_root / "snapshots"
        if snaps.is_dir():
            candidates = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for c in candidates:
                if (c / "modular_model_index.json").exists() or (c / "model_index.json").exists():
                    return str(c)
        # refs/main may point at snapshot
        ref = cached_root / "refs" / "main"
        if ref.is_file():
            snap = cached_root / "snapshots" / ref.read_text().strip()
            if snap.is_dir():
                return str(snap)

    return os.environ.get("H3_MODEL_ID", "MiniMaxAI/MiniMax-H3")


def duration_to_num_frames(duration_sec: float) -> tuple[int, float]:
    """Snap duration to H3's 17n+5 frame grid at 24fps."""
    duration_sec = max(MIN_DURATION, min(MAX_DURATION, float(duration_sec)))
    target = int(round(duration_sec * FPS))
    # find smallest 17n+5 >= target (or closest)
    n = max(0, (target - 5 + 16) // 17)
    frames = 17 * n + 5
    # keep within duration window
    while frames / FPS > MAX_DURATION and n > 0:
        n -= 1
        frames = 17 * n + 5
    while frames / FPS < MIN_DURATION:
        n += 1
        frames = 17 * n + 5
    return frames, frames / FPS


def resolve_resolution(aspect_ratio: str, preset: str) -> tuple[int, int]:
    aspect = aspect_ratio.strip()
    table = ASPECT_SIZES.get(aspect)
    if table is None:
        raise PipelineError(
            f"Unsupported aspect_ratio '{aspect_ratio}'. "
            f"Supported: {', '.join(sorted(ASPECT_SIZES))}"
        )
    key = preset.strip().lower()
    if key not in table:
        raise PipelineError(
            f"Unsupported resolution_preset '{preset}'. Supported: {', '.join(sorted(table))}"
        )
    w, h = table[key]["width"], table[key]["height"]
    if w % 32 or h % 32:
        raise PipelineError(f"Resolution {w}x{h} must be multiples of 32")
    return w, h


def detect_memory_mode() -> str:
    mode = os.environ.get("H3_MEMORY_MODE", "auto").lower()
    if mode != "auto":
        return mode
    try:
        import torch

        if not torch.cuda.is_available():
            return "int8_group_offload"
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        # BF16 ComponentsManager offload needs ~120GB+ host RAM for TE+DiT and
        # still OOMs on A100 when both fight for VRAM. Prefer int8 group-offload
        # unless explicitly overridden — matches the documented 24–48GB recipe
        # that community runs successfully.
        prefer_bf16 = os.environ.get("H3_PREFER_BF16_OFFLOAD", "0") == "1"
        if prefer_bf16 and vram_gb >= 70:
            return "a100_bf16_offload"
        return "int8_group_offload"
    except Exception:  # noqa: BLE001
        return "int8_group_offload"


def _try_set_attention_backend(pipe: Any) -> str:
    backend = os.environ.get("H3_ATTENTION_BACKEND", "auto")
    transformer = next(
        (
            getattr(pipe, name, None)
            for name in ("transformer", "transformer_ref")
            if getattr(pipe, name, None) is not None
        ),
        None,
    )
    if transformer is None or not hasattr(transformer, "set_attention_backend"):
        return "default"
    if backend == "sdpa":
        return "sdpa"
    candidates: list[str]
    if backend != "auto":
        candidates = [backend]
    else:
        # Hopper flash-3 first, then sage, then sdpa
        candidates = ["_flash_3_hub", "sage_attention", "flash_attention_2", "sdpa"]
    for name in candidates:
        try:
            transformer.set_attention_backend(name)
            return name
        except Exception as exc:  # noqa: BLE001
            print(f"[h3] attention backend {name} unavailable: {exc}")
    return "default"


def _release_loaded_pipeline() -> None:
    """Release the previous workflow before switching partitions."""
    global _PIPE, _MANAGER, MODEL_LOADED, _LOADED_WORKFLOW
    old_pipe = _PIPE
    _PIPE = None
    _MANAGER = None
    MODEL_LOADED = False
    _LOADED_WORKFLOW = None
    if old_pipe is not None:
        # Keep this explicit: ComponentsManager may otherwise retain every
        # module and make a workflow switch look like a GPU OOM.
        del old_pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def load_pipeline(workflow: str | None = None) -> Any:
    """Load one H3 workflow, switching partitions lazily when requested."""
    global _PIPE, _MANAGER, MODEL_INIT_SECONDS, MODEL_LOADED, _LOADED_WORKFLOW
    workflow = normalize_workflow(workflow)

    with _PIPE_LOCK:
        if MODEL_LOADED and _PIPE is not None and _LOADED_WORKFLOW == workflow:
            return _PIPE
        if _PIPE is not None:
            print(
                f"[h3] switching workflow {_LOADED_WORKFLOW or 'unknown'} -> {workflow}; "
                "releasing the previous transformer partition",
                flush=True,
            )
            _release_loaded_pipeline()

        t0 = time.perf_counter()
        import torch

        from diffusers import ModularPipeline

        model_path = resolve_model_path()
        memory_mode = detect_memory_mode()
        denoiser_name = "transformer_ref" if workflow == "ref2va" else "transformer"
        print(
            f"[h3] loading workflow={workflow} denoiser={denoiser_name} "
            f"from {model_path} mode={memory_mode}",
            flush=True,
        )
        print("[h3] stage=resolve_components (CPU/network; VRAM stays low until denoise)", flush=True)

        if memory_mode == "a100_bf16_offload":
            from diffusers import ComponentsManager

            manager = ComponentsManager()
            # Selecting the workflow here is important: omitting it declares
            # both transformer partitions and can fill a 200GB volume.
            pipe = ModularPipeline.from_pretrained(
                model_path,
                workflow=workflow,
                components_manager=manager,
            )
            print(
                f"[h3] stage=load_components {workflow} "
                "(download+deserialize; expect empty VRAM)",
                flush=True,
            )
            pipe.load_components(dtype=torch.bfloat16)
            print("[h3] stage=enable_auto_cpu_offload", flush=True)
            manager.enable_auto_cpu_offload(
                device="cuda",
                memory_reserve_margin=os.environ.get("H3_MEMORY_RESERVE", "48GB"),
            )
            _MANAGER = manager
            _PIPE = pipe
        elif memory_mode == "int8_group_offload":
            from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
            from diffusers.hooks import apply_group_offloading
            from torchao.quantization import Int8WeightOnlyConfig
            from transformers import Qwen3VLForConditionalGeneration
            from transformers import TorchAoConfig as TransformersTorchAoConfig

            pipe = ModularPipeline.from_pretrained(model_path, workflow=workflow)
            pipe.update_components(
                **{
                    denoiser_name: MiniMaxH3Transformer3DModel.from_pretrained(
                        model_path,
                        subfolder=denoiser_name,
                        dtype=torch.bfloat16,
                        quantization_config=TorchAoConfig(
                            Int8WeightOnlyConfig(version=2),
                            modules_to_not_convert=[
                                "proj_in",
                                "audio_proj_in",
                                "context_embedder",
                                "time_embedder",
                                "time_proj",
                                "token_refiner",
                                "norm_out",
                                "proj_out",
                                "audio_proj_out",
                            ],
                        ),
                        # Required True when quantization_config is set (TorchAo).
                        low_cpu_mem_usage=True,
                    ),
                    "text_encoder": Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path,
                        subfolder="text_encoder",
                        dtype=torch.bfloat16,
                        quantization_config=TransformersTorchAoConfig(
                            Int8WeightOnlyConfig(version=2),
                            modules_to_not_convert=[
                                "model.visual",
                                "model.language_model.embed_tokens",
                                "model.language_model.norm",
                                "lm_head",
                            ],
                        ),
                    ),
                }
            )
            pipe.load_components(dtype=torch.bfloat16)
            denoiser = getattr(pipe, denoiser_name)
            denoiser.requires_grad_(False)
            pipe.text_encoder.requires_grad_(False)
            # Streamed group-offload (use_stream=True) can roughly double host RAM
            # via pinned prefetch buffers — common OOM cause on ~100GB serverless
            # hosts. Default off.
            use_stream = os.environ.get("H3_OFFLOAD_USE_STREAM", "0") == "1"
            offload = dict(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                use_stream=use_stream,
            )
            group_kwargs = {**offload}
            try:
                import inspect

                if "low_cpu_mem_usage" in inspect.signature(
                    denoiser.enable_group_offload
                ).parameters:
                    group_kwargs["low_cpu_mem_usage"] = True
            except Exception:  # noqa: BLE001
                pass
            denoiser.enable_group_offload(
                offload_type="block_level",
                num_blocks_per_group=int(os.environ.get("H3_OFFLOAD_BLOCKS", "1")),
                **group_kwargs,
            )
            apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
            pipe.vae.to("cuda")
            pipe.audio_vae.to("cuda")
            _PIPE = pipe
        else:
            raise PipelineError(
                f"Unknown H3_MEMORY_MODE={memory_mode}. "
                "Use auto | a100_bf16_offload | int8_group_offload"
            )

        attn = _try_set_attention_backend(_PIPE)
        print(f"[h3] attention backend: {attn}", flush=True)

        MODEL_INIT_SECONDS = time.perf_counter() - t0
        MODEL_LOADED = True
        _LOADED_WORKFLOW = workflow
        print(
            f"[h3] model ready workflow={workflow} in {MODEL_INIT_SECONDS:.1f}s",
            flush=True,
        )
        return _PIPE


def _media_source(value: Any, label: str) -> str:
    """Validate a path/URL before handing it to Diffusers' media decoder."""
    if value is None:
        raise PipelineError(f"{label} is required")
    if not isinstance(value, (str, os.PathLike)):
        raise PipelineError(f"{label} must be a local path or an http(s) URL")
    source = str(value).strip()
    if not source:
        raise PipelineError(f"{label} must not be empty")
    if source.startswith("data:"):
        raise PipelineError(
            f"{label} must be an http(s) URL or local path; inline data URLs are not supported "
            "because Runpod request payloads are too small for media."
        )
    if source.startswith(("http://", "https://")):
        return source
    path = Path(source).expanduser()
    if not path.is_file():
        raise PipelineError(f"{label} file does not exist: {path}")
    return str(path)


def _load_keyframe(value: Any, label: str) -> Any:
    try:
        from diffusers.utils import load_image

        return load_image(_media_source(value, label))
    except PipelineError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Could not decode {label}: {exc}") from exc


def _build_references(raw_references: list[Any]) -> list[Any]:
    """Decode URL/path reference specs into MiniMax-H3 reference dataclasses."""
    if not isinstance(raw_references, list) or not raw_references:
        raise PipelineError("references must be a non-empty list")
    if len(raw_references) > 12:
        raise PipelineError("ref2va accepts at most 12 references in total")

    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    aliases = {
        "picture": "image",
        "img": "image",
        "clip": "video",
        "sound": "audio",
    }
    classes = {
        "image": MiniMaxH3ImageReference,
        "video": MiniMaxH3VideoReference,
        "audio": MiniMaxH3AudioReference,
    }
    counts = {"image": 0, "video": 0, "audio": 0}
    decoded: list[Any] = []
    for index, item in enumerate(raw_references):
        if isinstance(item, (str, os.PathLike)):
            kind = "image"
            source = item
            options: Mapping[str, Any] = {}
        elif isinstance(item, Mapping):
            kind = str(item.get("type") or item.get("kind") or "image").strip().lower()
            kind = aliases.get(kind, kind)
            source = item.get("url") or item.get("uri") or item.get("path")
            options = item
        else:
            raise PipelineError(f"references[{index}] must be a path/URL or object")

        if kind not in classes:
            raise PipelineError(
                f"references[{index}] has unsupported type '{kind}'; use image, video, or audio"
            )
        source = _media_source(source, f"references[{index}]")
        try:
            reference = classes[kind].from_file(source)
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"Could not decode references[{index}] ({kind}): {exc}") from exc

        # The decoder preserves the container's real rate.  Advanced callers
        # can override it when the source metadata is known to be wrong.
        if kind == "video" and options.get("fps") is not None:
            reference.fps = float(options["fps"])
        if kind in {"video", "audio"} and options.get("sample_rate") is not None:
            reference.sample_rate = int(options["sample_rate"])

        counts[kind] += 1
        decoded.append(reference)

    if counts["image"] > 9:
        raise PipelineError("ref2va accepts at most 9 image references")
    if counts["video"] > 3:
        raise PipelineError("ref2va accepts at most 3 video references")
    if counts["audio"] > 3:
        raise PipelineError("ref2va accepts at most 3 audio references")
    if counts["audio"] and not (counts["image"] or counts["video"]):
        raise PipelineError("an audio reference must be paired with an image or video reference")
    return decoded


def generate(
    *,
    prompt: str,
    workflow: str = "t2va",
    duration: float = 10.0,
    aspect_ratio: str = "9:16",
    resolution_preset: str = "native",
    seed: int | None = None,
    num_inference_steps: int | None = None,
    image: str | os.PathLike | None = None,
    last_image: str | os.PathLike | None = None,
    references: list[Any] | None = None,
    output_dir: str | Path | None = None,
) -> GenerateResult:
    import torch
    from diffusers.utils.export_utils import encode_video

    if not prompt or not str(prompt).strip():
        raise PipelineError("prompt is required")

    workflow = normalize_workflow(workflow)
    if workflow == "t2va" and (image is not None or last_image is not None or references):
        raise PipelineError("t2va does not accept image, last_image, or references")
    if workflow == "fl2va" and references:
        raise PipelineError("fl2va accepts image and/or last_image, not references")
    if workflow == "fl2va" and image is None and last_image is None:
        raise PipelineError("fl2va requires image, last_image, or both")
    if workflow == "ref2va" and (image is not None or last_image is not None):
        raise PipelineError("ref2va accepts references, not image or last_image")
    if workflow == "ref2va" and not references:
        raise PipelineError("ref2va requires at least one reference")

    # Decode media before denoising, while the request still owns clear error
    # context.  References are intentionally passed as paths/URLs into the
    # official Diffusers classes; they preserve video/audio rates correctly.
    keyframe = _load_keyframe(image, "image") if image is not None else None
    last_keyframe = _load_keyframe(last_image, "last_image") if last_image is not None else None
    decoded_references = _build_references(references) if workflow == "ref2va" else None

    width, height = resolve_resolution(aspect_ratio, resolution_preset)
    num_frames, snapped_duration = duration_to_num_frames(duration)
    steps = int(num_inference_steps or os.environ.get("H3_NUM_INFERENCE_STEPS", "20"))
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")
    seed = int(seed)

    out_root = Path(output_dir or os.environ.get("OUTPUT_DIR", "/tmp/h3_outputs"))
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"h3_{seed}_{width}x{height}_{num_frames}f.mp4"

    generator = torch.Generator(device="cpu").manual_seed(seed)
    outputs = ["videos", "audio", "sampling_rate"]

    pipeline_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "num_inference_steps": steps,
        "generator": generator,
        "output": outputs,
    }
    if workflow == "fl2va":
        if keyframe is not None:
            pipeline_kwargs["image"] = keyframe
        if last_keyframe is not None:
            pipeline_kwargs["last_image"] = last_keyframe
    elif workflow == "ref2va":
        pipeline_kwargs["references"] = decoded_references

    t0 = time.perf_counter()
    with _PIPE_LOCK:
        # Keep loading and denoising in the same critical section.  Otherwise a
        # second request could switch transformer partitions after ``pipe`` was
        # captured but before it was called.
        pipe = load_pipeline(workflow)
        results = pipe(**pipeline_kwargs)
    inference_seconds = time.perf_counter() - t0

    encode_video(
        results["videos"][0],
        fps=FPS,
        output_path=str(out_path),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )

    if not out_path.is_file() or out_path.stat().st_size < 1000:
        raise PipelineError("Generation produced an empty or missing video file")

    return GenerateResult(
        workflow=workflow,
        video_path=out_path,
        width=width,
        height=height,
        duration=snapped_duration,
        num_frames=num_frames,
        seed=seed,
        inference_seconds=inference_seconds,
        num_inference_steps=steps,
    )


def detect_gpu_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("RUNPOD_GPU_TYPE", "unknown")
