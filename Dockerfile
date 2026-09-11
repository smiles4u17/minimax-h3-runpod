# syntax=docker/dockerfile:1.7

ARG WORKER_BASE=runpod/comfyui:1.4.7-cuda13.0
ARG SAGEATTENTION_WHEEL_URL=https://huggingface.co/harryming/sageattention-blackwell-wheels/resolve/main/wheels/torch2.10.0-cu130-cp312-blackwell/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl
ARG SAGEATTENTION_WHEEL_SHA256=b6a1c65287a1e7d802c98bf1b7267446823a15dc160b4beed3046d118bc45570

FROM ${WORKER_BASE} AS runtime
ARG SAGEATTENTION_WHEEL_URL
ARG SAGEATTENTION_WHEEL_SHA256
ARG COMFYUI_REF=1f641fd9337f0ec4d635a28415a8d25a8d15f753
ARG KJNODES_REF=35e5956193769d18a13136cdedb73a36a05c73e6
ARG TURBO_REF=55fee864dd7b2976b1c4ce3c3d5f7968f181409f
ARG FBCACHE_REF=18362a23175771e68e4aa737d333bf4d4ee825fc
ARG VHS_REF=4ee72c065db22c9d96c2427954dc69e7b908444b
ARG H3_LOWVRAM_REF=f34cb9b8311ab1a1df31e9aa3c74bcd851c7bd36

ENV COMFY_ROOT=/comfyui \
    COMFY_URL=http://127.0.0.1:8188 \
    WORKFLOW_DIR=/opt/minimax-h3/workflows \
    ATTENTION_MODE=auto \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN cp -a /opt/comfyui-baked /comfyui \
    && rm -rf /comfyui/custom_nodes/* \
    && git -C /comfyui fetch --depth 1 origin "${COMFYUI_REF}" \
    && git -C /comfyui checkout --force FETCH_HEAD \
    && python3.12 -m pip install --no-cache-dir \
      --constraint /opt/comfyui-runtime-constraints.txt \
      -r /comfyui/requirements.txt \
      "transformers>=4.50.3,<5" \
      "huggingface-hub[hf_transfer]<1"

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
    install_node /comfyui/custom_nodes/ComfyUI-VideoHelperSuite \
      https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "$VHS_REF"; \
    install_node /comfyui/custom_nodes/ComfyUI-MiniMaxH3-LowVRAM \
      https://github.com/lericogit/ComfyUI-MiniMaxH3-LowVRAM.git "$H3_LOWVRAM_REF"; \
    for requirements in /comfyui/custom_nodes/*/requirements.txt; do \
      [ ! -f "$requirements" ] || python3.12 -m pip install --no-cache-dir \
        --constraint /opt/comfyui-runtime-constraints.txt -r "$requirements"; \
    done

RUN set -eux; \
    wheel=/tmp/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl; \
    wget --tries=5 --timeout=60 -O "$wheel" "$SAGEATTENTION_WHEEL_URL"; \
    echo "$SAGEATTENTION_WHEEL_SHA256  $wheel" | sha256sum -c -; \
    python3.12 -m pip install --no-cache-dir --no-deps "$wheel"; \
    rm -f "$wheel"; \
    python3.12 -c "import importlib.metadata as m, torch, sageattention; assert torch.__version__.startswith('2.10.0+cu130'), torch.__version__; print(torch.__version__, torch.version.cuda, m.version('sageattention'))"

COPY requirements-handler.txt /opt/minimax-h3/requirements-handler.txt
RUN python3.12 -m pip install --no-cache-dir \
      --constraint /opt/comfyui-runtime-constraints.txt \
      -r /opt/minimax-h3/requirements-handler.txt

COPY handler.py start.sh extra_model_paths.yaml /opt/minimax-h3/
COPY scripts /opt/minimax-h3/scripts
COPY workflows /opt/minimax-h3/workflows
COPY custom_nodes/samimate_h3 /comfyui/custom_nodes/samimate_h3
RUN chmod +x /opt/minimax-h3/start.sh /opt/minimax-h3/scripts/*.py \
    && cd /comfyui \
    && python3.12 /opt/minimax-h3/scripts/verify_h3_runtime.py \
    && timeout 300 python3.12 main.py --quick-test-for-ci --cpu

ENTRYPOINT ["/opt/minimax-h3/start.sh"]
CMD []

FROM runtime AS runtime-only

FROM runtime AS models
ARG MODEL_PROFILE=blackwell
ENV MODEL_PROFILE=${MODEL_PROFILE}
RUN --mount=type=secret,id=hf_token,required=false \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    python3.12 /opt/minimax-h3/scripts/download_models.py --asset fl2v
RUN --mount=type=secret,id=hf_token,required=false \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    python3.12 /opt/minimax-h3/scripts/download_models.py --asset r2v
RUN --mount=type=secret,id=hf_token,required=false \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    python3.12 /opt/minimax-h3/scripts/download_models.py --asset video-vae --asset audio-vae
RUN --mount=type=secret,id=hf_token,required=false \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    python3.12 /opt/minimax-h3/scripts/download_models.py --asset turbo
RUN --mount=type=secret,id=hf_token,required=false \
    export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    if [ "$MODEL_PROFILE" = "blackwell" ]; then \
      python3.12 /opt/minimax-h3/scripts/download_models.py --asset text-blackwell; \
    elif [ "$MODEL_PROFILE" = "universal" ]; then \
      python3.12 /opt/minimax-h3/scripts/download_models.py --asset text-universal; \
    elif [ "$MODEL_PROFILE" = "dual" ]; then \
      python3.12 /opt/minimax-h3/scripts/download_models.py --asset text-blackwell --asset text-universal; \
    else \
      echo "MODEL_PROFILE must be blackwell, universal, or dual" >&2; exit 2; \
    fi

FROM models AS final

# The RunPod GitHub builder does not expose Docker target selection. Keeping
# this lightweight stage last makes a plain cloud build produce the runtime
# image, while `docker build --target final` still creates the baked image.
FROM runtime-only AS cloud
