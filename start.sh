#!/usr/bin/env bash
set -Eeuo pipefail

COMFY_ROOT="${COMFY_ROOT:-/comfyui}"
COMFY_HOST="${COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${COMFY_PORT:-8188}"

mkdir -p "$COMFY_ROOT/input" "$COMFY_ROOT/output" /runpod-volume/models/loras

python "$COMFY_ROOT/main.py" \
  --listen "$COMFY_HOST" \
  --port "$COMFY_PORT" \
  --extra-model-paths-config /opt/minimax-h3/extra_model_paths.yaml \
  ${COMFY_ARGS:-} &
COMFY_PID=$!

shutdown() {
  kill "$COMFY_PID" 2>/dev/null || true
  wait "$COMFY_PID" 2>/dev/null || true
}
trap shutdown EXIT INT TERM

for _ in $(seq 1 180); do
  if wget -q -O /dev/null "http://$COMFY_HOST:$COMFY_PORT/system_stats"; then
    exec python -u /opt/minimax-h3/handler.py
  fi
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "ComfyUI exited before becoming ready." >&2
    wait "$COMFY_PID"
  fi
  sleep 1
done

echo "ComfyUI did not become ready within 180 seconds." >&2
exit 1

