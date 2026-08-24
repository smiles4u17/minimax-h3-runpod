# syntax=docker/dockerfile:1.7

ARG WORKER_BASE=runpod/worker-comfyui:5.8.6-base
ARG SAGEATTENTION_WHEEL_URL=https://huggingface.co/JahJedi/sageattention-flashattn-blackwell-cu130-torch211-cp312/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
ARG SAGEATTENTION_WHEEL_SHA256=4a47a462059bf09b7d10be68f93a0a462dd6184068298fc6ff10e94751fa6dd4
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

FROM ${WORKER_BASE} AS runtime
ARG SAGEATTENTION_WHEEL_URL
ARG SAGEATTENTION_WHEEL_SHA256
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG TORCH_INDEX_URL
ARG COMFYUI_REF=v0.33.1
ARG KJNODES_REF=35e5956193769d18a13136cdedb73a36a05c73e6
ARG TURBO_REF=55fee864dd7b2976b1c4ce3c3d5f7968f181409f
ARG FBCACHE_REF=18362a23175771e68e4aa737d333bf4d4ee825fc

ENV COMFY_ROOT=/comfyui \
    COMFY_URL=http://127.0.0.1:8188 \
    WORKFLOW_DIR=/opt/minimax-h3/workflows \
    ATTENTION_MODE=auto \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=900 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN uv pip install --force-reinstall \
      "torch==${TORCH_VERSION}" \
      "torchvision==${TORCHVISION_VERSION}" \
      "torchaudio==${TORCHAUDIO_VERSION}" \
      --index-url "${TORCH_INDEX_URL}"

RUN git -C /comfyui fetch --depth 1 origin "${COMFYUI_REF}" \
    && git -C /comfyui checkout --force FETCH_HEAD \
    && uv pip install -r /comfyui/requirements.txt \
    && uv pip install "transformers>=4.50.3,<5" "huggingface-hub[hf_transfer]<1"

RUN set -eux; \
    install_node() { \
      destination="$1"; repository="$2"; revision="$3"; \
      git init "$destination"; \
      git -C "$destination" remote add origin "$repository"; \
      git -C "$destination" fetch --depth 1 origin "$revision"; \
      git -C "$destination" checkout --detach FETCH_HEAD; \
    }; \
    install_node /comfyui/custom_nodes/ComfyUI-KJNodes \
      https://github.com/kijai/ComfyUI-KJNodes.git "$KJNODES_REF"; \
    install_node /comfyui/custom_nodes/ComfyUI-MiniMax-H3-Turbo \
      https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo.git "$TURBO_REF"; \
    install_node /comfyui/custom_nodes/ComfyUI-fasterminimax \
      https://github.com/Apache0ne/ComfyUI-fasterminimax.git "$FBCACHE_REF"; \
    for requirements in /comfyui/custom_nodes/*/requirements.txt; do \
      [ ! -f "$requirements" ] || uv pip install -r "$requirements"; \
    done

RUN set -eux; \
    wheel=/tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl; \
    wget --tries=5 --timeout=60 -O "$wheel" "$SAGEATTENTION_WHEEL_URL"; \
    echo "$SAGEATTENTION_WHEEL_SHA256  $wheel" | sha256sum -c -; \
    uv pip install "$wheel"; \
    rm -f "$wheel"; \
    python -c "import importlib.metadata as m, torch, sageattention; print(torch.__version__, torch.version.cuda, m.version('sageattention'))"

COPY requirements-handler.txt /opt/minimax-h3/requirements-handler.txt
RUN uv pip install -r /opt/minimax-h3/requirements-handler.txt

COPY handler.py start.sh extra_model_paths.yaml /opt/minimax-h3/
COPY scripts /opt/minimax-h3/scripts
COPY workflows /opt/minimax-h3/workflows
RUN chmod +x /opt/minimax-h3/start.sh /opt/minimax-h3/scripts/*.py \
    && cd /comfyui \
    && timeout 300 python main.py --quick-test-for-ci --cpu

CMD ["/opt/minimax-h3/start.sh"]

FROM runtime AS runtime-only

FROM runtime AS models
ARG MODEL_PROFILE=blackwell
ENV MODEL_PROFILE=${MODEL_PROFILE}
RUN --mount=type=secret,id=hf_token,required=false \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    python /opt/minimax-h3/scripts/download_models.py --asset fl2v
RUN --mount=type=secret,id=hf_token,required=false \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    python /opt/minimax-h3/scripts/download_models.py --asset r2v
RUN --mount=type=secret,id=hf_token,required=false \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    python /opt/minimax-h3/scripts/download_models.py --asset video-vae --asset audio-vae
RUN --mount=type=secret,id=hf_token,required=false \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    python /opt/minimax-h3/scripts/download_models.py --asset turbo
RUN --mount=type=secret,id=hf_token,required=false \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    if [ "$MODEL_PROFILE" = "blackwell" ]; then \
      python /opt/minimax-h3/scripts/download_models.py --asset text-blackwell; \
    elif [ "$MODEL_PROFILE" = "universal" ]; then \
      python /opt/minimax-h3/scripts/download_models.py --asset text-universal; \
    elif [ "$MODEL_PROFILE" = "dual" ]; then \
      python /opt/minimax-h3/scripts/download_models.py --asset text-blackwell --asset text-universal; \
    else \
      echo "MODEL_PROFILE must be blackwell, universal, or dual" >&2; exit 2; \
    fi

FROM models AS final

# The RunPod GitHub builder does not expose Docker target selection. Keeping
# this lightweight stage last makes a plain cloud build produce the runtime
# image, while `docker build --target final` still creates the baked image.
FROM runtime-only AS cloud
