# rvc-gpu-server — RVC GPU worker server (desktop side)

FastAPI server exposing the thin client's RVC conversions to the GPU box:
`POST /convert` (audio + params -> converted wav), `GET /health` (status,
device, loaded model). Self-contained — zero imports from the TTS-bot repo.

## Layout

| Path | Role |
|---|---|
| `server.py` | FastAPI shell: `/health`, `/convert`, auth, protocol gate |
| `worker_owner.py` | owns the persistent `worker.py` subprocess (stdlib only) |
| `worker.py` | **verbatim copy** of TTS-bot's `ttsbot/rvc/worker.py` — do not edit, re-copy on change |
| `rvc_infer/` | upstream RVC (pinned commit; rsynced working tree, no git) |
| `rvc_models/` | voice `.pth`/`.index` files (rsynced from thin client) |
| `tools/rvc_models.py` | voice-models.com downloader (prints thin-config block) |
| `tests/` | `test_owner.py` (stdlib-only) + `test_server.py` (needs server venv) |

## Desktop setup (fresh box)

```bash
nvidia-smi  # driver >= 531 for cu121
python3 -m venv ~/rvc-server-venv && source ~/rvc-server-venv/bin/activate
# torch CUDA wheel swap (version-identical with the thin client):
pip install "torch==2.4.1+cu121" "torchaudio==2.4.1+cu121" \
  --index-url https://download.pytorch.org/whl/cu121
# RVC stack (same pins as TTS-bot requirements-rvc.txt):
pip install -r <ttsbot>/requirements-rvc.txt  # thin checkout or copy
pip install -r requirements-server.txt
python -c "import torch; print(torch.cuda.is_available())"  # must be True
```

## Run

```bash
source ~/rvc-server-venv/bin/activate
RVC_MODEL_ROOT=$HOME/rvc-gpu-server/rvc_models \
RVC_GPU_SERVER_PORT=8001 \
python server.py
curl localhost:8001/health  # want "device": "cuda"
```

Env: `RVC_GPU_SERVER_HOST` (default 0.0.0.0), `RVC_GPU_SERVER_PORT`
(default 8001), `RVC_GPU_SERVER_TOKEN` (Bearer auth, empty = LAN trust),
`RVC_MODEL_ROOT` (flat voice-model dir fallback), `RVC_WORKER_ROOT`
(default `<repo>/rvc_infer`), `RVC_WORKER_THREADS`, `RVC_WORKER_MAX_RSS_MB`
(default 2500), `RVC_GPU_SERVER_JOB_TIMEOUT` (default 1800),
`RVC_SERVER_CHUNK_SEC` (default 30 — inputs longer than this are converted
in overlapping windows and crossfade-stitched, bounding VRAM on long
media), `RVC_SERVER_CHUNK_OVERLAP_SEC` (default 2).

## Persistence

`start.sh` launches the server by hand (log: `/tmp/rvc_server.log`). For
boot persistence install the user unit (secrets in
`~/.config/rvc-server/env`, chmod 600, never in git):

```bash
mkdir -p ~/.config/rvc-server ~/.config/systemd/user
printf 'RVC_GPU_SERVER_TOKEN=<token>\n' > ~/.config/rvc-server/env
chmod 600 ~/.config/rvc-server/env
cp systemd/rvc-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now rvc-server
```

NOTE: user units only start at boot if lingering is enabled, which needs
root once: `sudo loginctl enable-linger haama`. Without it the service
starts at first login instead. Check with
`loginctl show-user haama -p Linger`.

The thin client's `.env` needs the same `RVC_GPU_SERVER_TOKEN`.

## Contract

HTTP `X-RVC-Protocol: 1` — bump only in lockstep with the thin client's
`RvcRunner`. 4xx = deterministic (fail fast); 5xx/timeout = transient
(thin client uses its local slow path).
