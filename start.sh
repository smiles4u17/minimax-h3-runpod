#!/usr/bin/env bash
set -Eeuo pipefail

COMFY_ROOT="${COMFY_ROOT:-/comfyui}"
COMFY_HOST="${COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${COMFY_PORT:-8188}"

GPU_MEMORY_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sed -n '1p' | tr -dc '0-9' || true)"
if [[ "$GPU_MEMORY_MB" =~ ^[0-9]+$ ]]; then
  export H3_GPU_MEMORY_MB="$GPU_MEMORY_MB"
fi

declare -a LOW_VRAM_ARGS=()
LOW_VRAM_MODE="${H3_LOW_VRAM_MODE:-auto}"
LOW_VRAM_THRESHOLD_MB="${H3_LOW_VRAM_THRESHOLD_MB:-49152}"
if [[ "$LOW_VRAM_MODE" =~ ^(1|true|yes|on|force)$ ]] || \
   { [[ "$LOW_VRAM_MODE" == "auto" ]] && [[ "$GPU_MEMORY_MB" =~ ^[0-9]+$ ]] && (( GPU_MEMORY_MB <= LOW_VRAM_THRESHOLD_MB )); }; then
  LOW_VRAM_ARGS=(
    --reserve-vram "${H3_LOW_VRAM_RESERVE_GB:-1}"
    --vram-headroom "${H3_LOW_VRAM_HEADROOM_GB:-1}"
    --disable-smart-memory
    --cache-none
  )
  echo "Enabling the MiniMax H3 low-VRAM worker profile for ${GPU_MEMORY_MB:-unknown} MiB VRAM."
fi

declare -a USER_COMFY_ARGS=()
if [[ -n "${COMFY_ARGS:-}" ]]; then
  read -r -a USER_COMFY_ARGS <<< "$COMFY_ARGS"
fi

mkdir -p "$COMFY_ROOT/input" "$COMFY_ROOT/output" /runpod-volume/models/loras

python3.12 "$COMFY_ROOT/main.py" \
  --listen "$COMFY_HOST" \
  --port "$COMFY_PORT" \
  --extra-model-paths-config /opt/minimax-h3/extra_model_paths.yaml \
  "${LOW_VRAM_ARGS[@]}" \
  "${USER_COMFY_ARGS[@]}" &
COMFY_PID=$!

shutdown() {
  kill "$COMFY_PID" 2>/dev/null || true
  wait "$COMFY_PID" 2>/dev/null || true
}
trap shutdown EXIT INT TERM

for _ in $(seq 1 180); do
  if wget -q -O /dev/null "http://$COMFY_HOST:$COMFY_PORT/system_stats"; then
    exec python3.12 -u /opt/minimax-h3/handler.py
  fi
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "ComfyUI exited before becoming ready." >&2
    wait "$COMFY_PID"
  fi
  sleep 1
done

echo "ComfyUI did not become ready within 180 seconds." >&2
exit 1

