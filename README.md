# MiniMax H3 RunPod Serverless

Production RunPod Serverless worker generated from the supplied FL2V/I2V and R2V ComfyUI workflows.

## What is included

- ComfyUI v0.33.1
- MiniMax H3 FL2VA and Ref2VA pruned INT8 ConvRot models
- MiniMax H3 video and audio VAEs
- Larry v4 step-600 EMA Turbo LoRA
- 6-step Turbo sampler preset
- The tested `Apache0ne/ComfyUI-fasterminimax` FirstBlockCache at `0.18`, warmup `1`, maximum consecutive reuse `1`
- MiniMax H3 memory-efficient SageAttention patch
- Compiled SageAttention 2.2.0 for Ampere, Ada, Hopper, datacenter Blackwell, and RTX 50-series
- A RunPod handler that returns MP4 files, not just images
- Simple `fl2v` and `r2v` request schemas plus raw ComfyUI API-workflow passthrough

Sol-Attn is removed from the production graphs. It was still present in the exported API files even though it is bypassed by the H3 Sage patch and was slower in the measured test.

## Cloud-only RunPod deployment

The final Dockerfile stage is intentionally the lightweight `cloud` stage. A
plain RunPod GitHub build therefore installs ComfyUI, the custom nodes, the
handler, Torch, and SageAttention without downloading the model weights into
the image. The explicit `--target final` commands below still create a baked
image when desired.

1. Put this repository on GitHub and connect GitHub under RunPod Settings → Connections.
2. Create a RunPod network volume with at least 100 GB in a data center offering RTX 5090 Serverless workers.
3. Temporarily attach that volume to any inexpensive RunPod Pod and clone this repository in its web terminal.
4. Populate the volume entirely in the cloud:

```bash
cd /workspace/minimax-h3-runpod
VOLUME_ROOT=/workspace MODEL_PROFILE=blackwell bash scripts/setup_runpod_volume.sh
```

5. Stop and delete the temporary Pod, preserving the network volume.
6. In RunPod Serverless, choose New Endpoint → Import Git Repository, select this repository and its root `Dockerfile`, then attach the same network volume.

The runtime reads the shared models from `/runpod-volume/models` through
`extra_model_paths.yaml`. No model files need to be downloaded to the local PC.

## Build the RTX 5090 image

The default build uses CUDA 13.0, the NVFP4 AWQ Qwen text encoder, and bakes all public model files into the image.

```bash
docker build --target final -t YOUR_DOCKERHUB/minimax-h3-runpod:5090 .
docker push YOUR_DOCKERHUB/minimax-h3-runpod:5090
```

If Hugging Face requires authentication for the model license on your account, expose the token only as a BuildKit secret:

```bash
docker build --target final \
  --secret id=hf_token,env=HF_TOKEN \
  -t YOUR_DOCKERHUB/minimax-h3-runpod:5090 .
```

The baked image is large: approximately 64 GB of model weights before container-layer compression. For the smallest image, build `runtime-only` and place the models on a RunPod network volume instead.

```bash
docker build --target runtime-only -t YOUR_DOCKERHUB/minimax-h3-runpod:runtime .
```

## Universal NVIDIA build

Use this variant when the endpoint may receive RTX 30/40/50, A-series, L-series, or H-series GPUs. It uses CUDA 12.8 and the INT8 ConvRot Qwen encoder.

```bash
docker build --target final \
  --build-arg CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  --build-arg MODEL_PROFILE=universal \
  -t YOUR_DOCKERHUB/minimax-h3-runpod:universal .
```

To put both text encoders in one image and automatically select by compute capability, use `--build-arg MODEL_PROFILE=dual`. This adds about 27 GB.

## RunPod endpoint

1. Push the image to Docker Hub or another registry.
2. Create a RunPod Serverless template using that image.
3. Set container disk to at least 100 GB for the baked image.
4. Create a queue-based Serverless endpoint from the template.
5. For the 5090 image, select RTX 5090 workers. For the universal image, select the GPU families you want.
6. Set execution timeout to at least 30 minutes while validating; reduce it after measuring your longest 15-second job.
7. Configure S3-compatible output storage, a network-volume output directory, or send presigned PUT URLs in each request.

## Output configuration

MP4 results often exceed RunPod's response-size limit. The worker chooses output delivery in this order:

1. Request field `output_upload_urls`: presigned HTTPS PUT URLs, one per output.
2. S3-compatible storage configured through environment variables.
3. Base64 only when the file is no larger than `MAX_RETURN_BASE64_MB`.
4. `OUTPUT_VOLUME_DIR` if configured.

S3 variables:

```text
S3_BUCKET=your-bucket
S3_REGION=us-west-2
S3_ENDPOINT_URL=https://s3.us-west-2.amazonaws.com
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_PREFIX=minimax-h3
S3_PUBLIC_URL=https://optional-public-base.example.com
```

The equivalent RunPod worker names `BUCKET_NAME`, `BUCKET_REGION`, `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, and `BUCKET_SECRET_ACCESS_KEY` are also accepted.

## Custom LoRAs

The public base models and Turbo LoRA are baked in. Your private/custom H3 LoRAs are not.

Put them on a RunPod network volume at:

```text
/runpod-volume/models/loras/H3/H3_Combat_V2.safetensors
```

Then reference them in a request:

```json
"loras": [
  {"name": "H3/H3_Combat_V2.safetensors", "strength": 1.0}
]
```

The handler inserts ordinary model-only LoRA loader nodes before the Turbo node. Disabled LoRAs from the UI workflow are not validated or loaded.

## FL2V request

`first_frame` is required; `last_frame` is optional. Each asset accepts either `url` or base64 `data`.

```json
{
  "input": {
    "task": "fl2v",
    "prompt": "Your full MiniMax H3 prompt",
    "first_frame": {"name": "first.png", "url": "https://..."},
    "last_frame": {"name": "last.png", "url": "https://..."},
    "duration": 5,
    "megapixels": 0.6,
    "steps": 6,
    "seed": 123456,
    "cache_threshold": 0.18,
    "attention": "auto",
    "loras": []
  }
}
```

## R2V request

The endpoint accepts one to nine reference images and one optional audio reference.

```json
{
  "input": {
    "task": "r2v",
    "prompt": "Use <Picture 1>, <Picture 2>, and <Audio 1>...",
    "references": [
      {"name": "picture_1.png", "url": "https://..."},
      {"name": "picture_2.png", "url": "https://..."}
    ],
    "audio": {"name": "audio_1.wav", "url": "https://..."},
    "use_reference_audio_as_output": false,
    "duration": 5,
    "megapixels": 0.6,
    "steps": 6,
    "cache_threshold": 0.18,
    "attention": "auto"
  }
}
```

Use `/run` rather than `/runsync` for video jobs, then poll `/status/{job_id}`. A ready client is included at `examples/client.py`.

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
python examples/client.py examples/fl2v_request.json
```

## Runtime options

| Variable | Default | Purpose |
| --- | --- | --- |
| `ATTENTION_MODE` | `auto` | `auto`, `sage`, or `native` |
| `TEXT_ENCODER_FILE` | automatic | Override the text encoder filename |
| `COMFY_ARGS` | empty | Extra ComfyUI launch flags |
| `JOB_TIMEOUT_SECONDS` | `3600` | Per-job ComfyUI wait timeout |
| `MAX_ASSET_MB` | `250` | Maximum downloaded/decoded input asset size |
| `MAX_RETURN_BASE64_MB` | `6` | Maximum inline output size; keeps base64 results below RunPod's 10 MB async payload limit |
| `OUTPUT_VOLUME_DIR` | unset | Persistent output folder fallback |

`attention: auto` selects Sage only when CUDA capability and the compiled module are available. Setting `native` rewires FBCache directly after the Turbo LoRA, so the same image can safely run on a card where Sage is undesirable.

## Raw workflow mode

You can also send any ComfyUI API-format workflow directly:

```json
{
  "input": {
    "workflow": {"...": "API workflow JSON"},
    "assets": [
      {"name": "input.png", "data": "BASE64..."}
    ]
  }
}
```

Raw assets keep their supplied names so `LoadImage`, `LoadAudio`, or `LoadVideo` nodes can refer to them.

## Local validation

```bash
python scripts/validate_project.py
python -m unittest discover -s tests -v
bash -n start.sh
docker build --target runtime-only -t minimax-h3-runpod:test .
```

The original four workflow exports are preserved in `source_workflows/`. See `docs/NODE_INVENTORY.md` for the production and GUI-only custom-node inventory.
