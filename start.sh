#!/usr/bin/env bash
# Launch the RVC GPU worker server (used by the cron @reboot entry and by hand).
# Secrets live in $HOME/.config/rvc-server/env (chmod 600, NOT in git):
#   RVC_GPU_SERVER_TOKEN=<shared bearer token>
set -euo pipefail
REPO="$HOME/rvc-gpu-server"
ENV_FILE="${RVC_SERVER_ENV:-$HOME/.config/rvc-server/env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi
export RVC_MODEL_ROOT="${RVC_MODEL_ROOT:-$REPO/rvc_models}"
export RVC_GPU_SERVER_PORT="${RVC_GPU_SERVER_PORT:-8001}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$HOME/rvc-server-venv/bin/python" "$REPO/server.py" >>/tmp/rvc_server.log 2>&1
