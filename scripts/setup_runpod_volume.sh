#!/usr/bin/env bash
set -euo pipefail

VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume}"
MODEL_PROFILE="${MODEL_PROFILE:-blackwell}"
DOWNLOAD_VENV="${DOWNLOAD_VENV:-/tmp/minimax-h3-download}"

if [[ ! -d "$VOLUME_ROOT" ]]; then
  echo "RunPod network volume is not mounted at $VOLUME_ROOT" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  apt-get update
  apt-get install -y python3 python3-venv
fi

if [[ ! -x "$DOWNLOAD_VENV/bin/python" ]]; then
  if ! python3 -m venv "$DOWNLOAD_VENV"; then
    apt-get update
    apt-get install -y python3-venv
    python3 -m venv "$DOWNLOAD_VENV"
  fi
fi

DOWNLOAD_PYTHON="$DOWNLOAD_VENV/bin/python"
"$DOWNLOAD_PYTHON" -m pip install --quiet --upgrade pip "huggingface-hub[hf_transfer]<1"

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

"$DOWNLOAD_PYTHON" "$(dirname "$0")/download_models.py" "${assets[@]}"

echo "MiniMax H3 $MODEL_PROFILE models are ready in $VOLUME_ROOT/models"
