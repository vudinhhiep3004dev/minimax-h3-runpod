# MiniMax H3 on Runpod Serverless

Smallest production-usable **Runpod Serverless Queue** worker for open-weight [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).

Submit an ad-hoc prompt, get a durable MP4 URL (TikTok 1080×1920 by default), with **$/generated-second** instrumentation vs MiniMax’s hosted baseline (`$0.13/sec`).

## Architecture

- Python 3.10+ / `runpod` SDK / `handler.py`
- Official **diffusers ModularPipeline** with `t2va`, `fl2va`, and `ref2va` workflows
- One workflow loaded per worker lifecycle; switching partitions is lazy and can be avoided with dedicated endpoints
- Flex workers, **scale-to-zero**, Queue endpoint
- Burst = many independent `/run` jobs sharing a warm worker (native queue)
- Outputs uploaded to S3-compatible storage (or local `file://` for tests)

```text
cold worker → init container → load H3 once → job1 → job2 → … → idle timeout → scale to zero
```

## Quick start (client)

```bash
cp .env.example .env
# fill RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID, optional S3_*, HF_TOKEN

python client.py --input test_input.json
# or
python client.py --prompt "..." --duration 10 --aspect-ratio 9:16 --quality draft
```

Expected CLI summary:

```text
H3 generation complete
Output: 10.1 sec
Resolution: 544x960 -> 1080x1920
GPU compute: 188.4 sec
Estimated Runpod cost: $0.091
Cost/output-sec: $0.0091
MiniMax equivalent: $1.31
Savings: 93.0%
```

Local handler smoke (needs a capable GPU + weights):

```bash
RUNPOD_LOCAL_TEST=1 H3_EAGER_LOAD=0 python client.py --local --input test_input.json
# or
python -u handler.py  # picks up test_input.json when RUNPOD_LOCAL_TEST patterns apply
```

## Request schema

```json
{
  "input": {
    "prompt": "...",
    "workflow": "t2va",
    "duration": 10,
    "aspect_ratio": "9:16",
    "seed": 123,
    "resolution_preset": "draft",
    "upscale": true
  }
}
```

| Field | Required | Notes |
|---|---|---|
| `prompt` | yes | H3-style shot/audio description recommended |
| `workflow` / `mode` | no | `t2va`, `i2va`, `l2va`, `fl2va`, or `ref2va`; omitted mode is inferred from media |
| `duration` | no | 4–15s; snapped to H3 `17n+5` frames @ 24fps |
| `aspect_ratio` | no | `9:16` (default), `16:9`, `1:1` |
| `seed` | no | int |
| `resolution_preset` / `quality` | no | `draft` (544×960), `720p` (704×1248, 0.88 MP), `native` (768×1344) for 9:16 |
| `upscale` | no | default `true` → 1080×1920 for 9:16 |
| `num_inference_steps` | no | default `20` |
| `image` / `first_frame` | for I2VA | Local path or HTTP(S) URL for the first keyframe |
| `last_image` / `last_frame` | for L2VA/FL2VA | Local path or HTTP(S) URL for the last keyframe |
| `references` | for Ref2VA | Ordered list of `{type: image|video|audio, url|path}`; max 9 images, 3 videos, 3 audio, 12 total |
| `jobs` | no | optional list of the above; runs sequentially on one warm worker |

### Image-to-video and reference-to-video examples

MiniMax H3 calls the first/last-keyframe family `fl2va`.  One `image` is
first-frame image-to-video (I2VA), one `last_image` is last-frame-to-video
(L2VA), and both images form first-and-last-frame video.  `ref2va` is different:
its ordered references describe subjects, motion, style, or audio; they do not
pin the generated first/last frame.

```json
{
  "input": {
    "workflow": "i2va",
    "prompt": "The product rotates slowly, soft studio light, subtle whoosh.",
    "image": "https://example.com/product.png",
    "duration": 8,
    "resolution_preset": "draft"
  }
}
```

```json
{
  "input": {
    "workflow": "ref2va",
    "prompt": "Keep the character identity and outfit from <Picture 1>; use the camera motion from <Video 1>.",
    "references": [
      {"type": "image", "url": "https://example.com/character.png"},
      {"type": "video", "url": "https://example.com/motion.mp4"}
    ],
    "duration": 8,
    "resolution_preset": "draft"
  }
}
```

Use URLs (or paths already visible inside the worker), not base64 media in the
Runpod JSON request.  The worker decodes URLs with the official H3 reference
classes so video frame rates and audio sample rates are retained.

### MCP gateway

`mcp_server.py` is a small local MCP gateway.  It keeps Runpod credentials off
the GPU worker, accepts URL or local-path media, uploads local media to the
configured S3/R2 bucket, submits the queue job, and returns the durable MP4
URL.  Install its separate dependencies so the GPU image does not gain an MCP
server:

```bash
python -m venv .venv-mcp
source .venv-mcp/bin/activate
pip install -r mcp_requirements.txt
export RUNPOD_API_KEY=rpa_...
export RUNPOD_ENDPOINT_ID=...
python mcp_server.py
```

The MCP exposes `h3_generate_video`, `h3_job_status`, and `h3_cancel_job`.
For a larger deployment, set `RUNPOD_ENDPOINT_ID_T2VA`,
`RUNPOD_ENDPOINT_ID_FL2VA`, and `RUNPOD_ENDPOINT_ID_REF2VA`; the gateway will
route each workflow to its own endpoint and avoid repeatedly swapping the two
large transformer partitions.

## Build the worker image (remote — do not upload from your laptop)

Layers (`pytorch` base + `pip`) should be pulled on a fast builder, not pushed from a home connection.

### Option A (preferred): Runpod builds from GitHub

1. Runpod console → **Settings → Connections → GitHub** → authorize this repo
2. **Serverless → New Endpoint → Import Git Repository** → `ruizmr/minimax-h3-runpod`
3. Branch `main`, Dockerfile path `Dockerfile`
4. Runpod pulls the base image + deps on their network, stores the image in their registry, and deploys

Updates: create a GitHub **Release** to rebuild (per Runpod’s GitHub integration).

### Option B: GitHub Actions builds + public pull URL

`.github/workflows/build-push.yml` builds on `ubuntu-latest` and pushes:

| Registry | Image |
|---|---|
| Public (ttl.sh, 7-day TTL) | `ttl.sh/ruizmr-minimax-h3-runpod:7d` |
| GHCR (may be private until you flip visibility) | `ghcr.io/ruizmr/minimax-h3-runpod:latest` |

```bash
gh workflow run build-push.yml
# then point the Runpod template imageName at the public tag above
```

See `docs/worker_image.md` after CI runs.

### Option C: bootstrap (runtime install)

Public `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime` + `scripts/worker_entrypoint.sh` (clone + pip on the worker). Slow cold starts; fine as a fallback.

**Do not bake weights into the image.** Prefer Runpod **cached models** (`MiniMaxAI/MiniMax-H3`) so download time is not billed.

Writers-room / brand / canon trees (`room/`, `10-Brand/`, `20-Canon/`) stay in the repo for prompt generation; `.dockerignore` keeps them out of the GPU image.

## Airing (distribution)

The `distribute/` package is the carriage layer: Assembly MCP + Codex stitch episodes, a human curator approves final cuts, Postiz posts to same-name channel accounts (TikTok / Instagram / YouTube). See [`distribute/README.md`](distribute/README.md) and [`room/00-System/Network-Ops.md`](room/00-System/Network-Ops.md).

## Deployed endpoint

Prefer `python scripts/deploy_stable_endpoint.py` which creates:

- A **network volume** for Hugging Face weights (`/runpod-volume/huggingface-cache`)
- A Queue endpoint with **no Runpod cached-model slot** (that feature corrupted an earlier endpoint)
- `workersMax` default **3** for throughput (HF weights shared via the network volume)
- `H3_EAGER_LOAD=0` so workers register with the queue before loading weights

**Do not** pause or delete the endpoint while the first job is downloading the model. That cancels the job and wastes the download.

**Do not** attach a Runpod console “cached model” to this endpoint; use the network volume + `HF_TOKEN` instead.

Current stable endpoint ID is in `.env` as `RUNPOD_ENDPOINT_ID` (recreated if the previous one was deleted).

The default 200GB volume is sized for the shared components plus the `t2va` /
`fl2va` transformer. `ref2va` uses a separate `transformer_ref` partition;
running both families on the same cache can exceed 200GB. For Ref2VA, use a
separate endpoint/volume (or increase the volume after checking the current
checkpoint sizes) and set `RUNPOD_ENDPOINT_ID_REF2VA` in the MCP gateway.
## Exact Runpod endpoint configuration

Console: [Serverless → New Endpoint → Docker](https://www.console.runpod.io/serverless)

| Field | Value |
|---|---|
| **Endpoint name** | `minimax-h3` (any) |
| **Endpoint type** | **Queue-based** |
| **Container image** | `YOUR_DOCKERHUB_USER/minimax-h3-runpod:latest` |
| **Container disk** | **50 GB** (encode temp + pip caches; models via HF cache volume) |
| **GPU** priority | **1)** L40 / L40S / 6000 Ada (48GB PRO) **2)** A100 (80GB) **3)** optional H100 |
| **GPUs per worker** | `1` |
| **Active workers** | `0` (Flex / scale-to-zero) |
| **Max workers** | `2` (raise for large bursts) |
| **Idle timeout** | **120 seconds** (critical for burst amortization; do **not** leave at 5s) |
| **Execution timeout** | **3600 seconds** |
| **FlashBoot** | **Enabled** |
| **Model (cached)** | `MiniMaxAI/MiniMax-H3` |
| **Auto-scaling type** | Queue delay, threshold ~4s |
| **Expose HTTP ports** | none (queue handler only) |

### Environment variables (endpoint)

Copy from `.env.example`. Minimum:

| Key | Example |
|---|---|
| `HF_TOKEN` | Hugging Face token (if needed for download/gated access) |
| `H3_MEMORY_MODE` | `auto` (or `int8_group_offload` / `a100_bf16_offload`) |
| `RUNPOD_GPU_TYPE` | `L40S` or `A100` (for cost math) |
| `RUNPOD_GPU_RATE_PER_SEC` | `0.00053` (L40S) or `0.00076` (A100) |
| `MINIMAX_RATE_PER_SEC` | `0.13` |
| `S3_BUCKET` / `S3_ENDPOINT_URL` / keys | durable outputs |
| `S3_PUBLIC_BASE_URL` | optional CDN base |
| `H3_EAGER_LOAD` | `1` |
| `H3_DEFAULT_PRESET` | `draft` for cheapest TikToks |
| `H3_WORKFLOW` | optional warm-start workflow; `t2va` by default |

After deploy, set locally:

```bash
export RUNPOD_ENDPOINT_ID=xxxxxxxx
export RUNPOD_API_KEY=rpa_...
```

## Burst / warm-worker usage

Prefer **independent jobs** (native queue) over packing one giant request:

```bash
python burst.py --count 10 --quality draft --duration 10
```

With **Idle timeout = 120s**, a rapid burst typically:

1. Pays model load once (cold)
2. Serves remaining jobs warm
3. Scales to zero after the queue drains

Optional packed batch (same worker, sequential):

```json
{"input": {"jobs": [{"prompt": "...", "duration": 10}, {"prompt": "...", "duration": 8}]}}
```

## Benchmarks

```bash
python benchmark.py --duration 10
# writes docs/benchmark_results.md
```

Runs:

1. draft / 720p / native resolution matrix (same prompt)
2. cold then warm pair (waits for scale-to-zero)
3. 10-job burst economics

## Cost model

Estimate = `(model_init_if_cold + inference + upscale + upload) * RUNPOD_GPU_RATE_PER_SEC`

Savings vs MiniMax = `1 - cost_per_output_second / MINIMAX_RATE_PER_SEC`

Rates are env-configurable — never hard-code production pricing in call sites.

## Memory modes

| Mode | GPU | Notes |
|---|---|---|
| `a100_bf16_offload` | ≥70GB VRAM (A100) | Official ComponentsManager BF16 offload |
| `int8_group_offload` | 48GB class | Official torchao int8 + group offload; needs ample host RAM (~75GB+ ideal → prefer L40S/6000 Ada over A40) |
| `auto` | detects VRAM | chooses between the two |

## Repo layout

```text
handler.py          Runpod entry
h3_pipeline.py      load-once ModularPipeline
cost.py             $/sec math
upload.py           S3-compatible upload
upscale.py          → 1080×1920 + audio
client.py           single job CLI
mcp_server.py       local MCP gateway
mcp_requirements.txt MCP-only dependencies
burst.py            N independent jobs
benchmark.py        economics suite
Dockerfile
requirements.txt
test_input.json
.env.example
docs/benchmark_results.md
```

## Notes / limits

- Official **H3-Regenerate-2K is not open-sourced**; 1080×1920 is native 9:16 + high-quality upscale (ffmpeg lanczos by default).
- H3 emits **native stereo audio** in one pass.
- Payload limits mean videos are **not** returned inline — always upload (or local file URL).
- Never commit `.env` or credentials.

## License

Model weights are under the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). This worker code is provided as-is for deployment experiments.
