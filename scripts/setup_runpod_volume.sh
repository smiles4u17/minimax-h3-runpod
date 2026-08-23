#!/usr/bin/env bash
set -euo pipefail

VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
MODEL_PROFILE="${MODEL_PROFILE:-blackwell}"

if [[ ! -d "$VOLUME_ROOT" ]]; then
  echo "RunPod network volume is not mounted at $VOLUME_ROOT" >&2
  exit 2
fi

python -m pip install --quiet --upgrade "huggingface-hub[hf_transfer]<1"

export COMFY_ROOT="$VOLUME_ROOT"
export HF_HUB_ENABLE_HF_TRANSFER=1

assets=(
  --asset fl2v
  --asset r2v
  --asset video-vae
  --asset audio-vae
  --asset turbo
)

case "$MODEL_PROFILE" in
  blackwell) assets+=(--asset text-blackwell) ;;
  universal) assets+=(--asset text-universal) ;;
  dual) assets+=(--asset text-blackwell --asset text-universal) ;;
  *) echo "MODEL_PROFILE must be blackwell, universal, or dual" >&2; exit 2 ;;
esac

python "$(dirname "$0")/download_models.py" "${assets[@]}"

echo "MiniMax H3 $MODEL_PROFILE models are ready in $VOLUME_ROOT/models"
