# rvc-gpu-server — RVC/TTS worker server (ifrit side)

FastAPI server for the bot: `GET /health`, `POST /convert`, `POST /diarize`,
`POST /separate`, `POST /tts`, `POST /chatterbox`, `POST /vc`, `POST /whisperx`. The Python
stack is in the image; the large assets are bind-mounted from the host.

## Layout

| Path | Role |
|---|---|
| `server.py` | FastAPI shell: endpoints, auth, protocol gate |
| `worker_owner.py` | owns the persistent `worker.py` subprocess (stdlib only) |
| `worker.py` | **verbatim copy** of TTS-bot's `ttsbot/rvc/worker.py` — do not edit, re-copy on change |
| `Dockerfile` | image build (CUDA base, Python stack, server code) |
| `compose.yaml` | CPU deployment, mounts, env file |
| `compose.gpu.yaml` | adds `gpus: all` when the NVIDIA card is installed |
| `systemd/rvc-server.service` | legacy venv unit (historical) |
| `systemd/rvc-server-resume.service` | restarts the container after suspend/resume |
| `requirements-gpu.txt` | lock generated from the live ifrit venv |
| `rvc_infer/` | upstream RVC working tree (mounted at `/rvc_infer`, not in the image) |
| `models/` | Kokoro models (mounted at `/models`, not in the image) |
| `rvc_models/` | mirror of the bot host's voice models (mounted at the same absolute path) |
| `ttsbot/media/diarize.py`, `ttsbot/media/audio.py` | **byte-identical mirrors** of the thin client's analysis code (`analyze_media` etc.) — re-copy on change, never edit (same rule as `worker.py`) |
| `ttsbot-diarize/` (mounted) | thin client's `ttsbot_diarize` package (mounted at `/ttsbot-diarize`) |
| `tests/` | server tests |

## Docker (current)

```bash
cd ~/rvc-gpu-server
# GPU host (the ifrit deployment):
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
# CPU-only host:
docker compose up -d --build
curl localhost:8001/health   # {"status":"ok","device":"cpu"|"cuda",...}
```

`compose.yaml` mounts:

| Host | Container |
|---|---|
| `~/TTS-bot/rvc_models` | `/home/haama/TTS-bot/rvc_models` (ro; the client sends these absolute paths) |
| `~/rvc-gpu-server/rvc_infer` | `/rvc_infer` (ro) |
| `~/rvc-gpu-server/models` | `/models` (ro) |
| `~/ttsbot-diarize` | `/ttsbot-diarize` (ro) |
| `~/.cache/huggingface` | `/home/haama/.cache/huggingface` (rw) |
| `~/.cache/torch` | `/home/haama/.cache/torch` (rw) |

Server secrets and `HF_TOKEN`/`PYANNOTE_MODEL_ID` come from
`~/.config/rvc-server/env` (chmod 600, never in git). The thin client's
`.env` needs the same `RVC_GPU_SERVER_TOKEN`.

### Vocal separation (`!rsing`)

`POST /separate` accepts multipart `audio` and `model` (default
`vocals-bs-roformer-368`) with the same bearer token and protocol header as
`/convert`. It returns JSON `vocals_b64` and `instrumental_b64` WAVs, as expected
by the bot. PyMSS uses CUDA/mixed precision on ifrit and releases its model after
each request. Separation jobs are serialized to avoid loading multiple models.

Provision the default checkpoint on the **ifrit host**, outside the read-only
container mount (about 640 MB; not included in the Docker image):

```bash
curl -fL --retry 2 -o ~/rvc-gpu-server/rvc_infer/assets/pymss_weights/model_bs_roformer_ep_368_sdr_12.9628.ckpt \
  https://huggingface.co/baicai1145/pymss/resolve/main/vocal/vocal_extraction/model_bs_roformer_ep_368_sdr_12.9628.ckpt
```

The matching YAML ships with `rvc_infer`. Missing weights/CUDA failures return
503; invalid models/audio return 400. Rebuild **on ifrit** using both compose
files above; rebuilding the bot on boksi does not deploy a server endpoint.

To test while the old service is still on 8001, publish a spare port:

```bash
RVC_SERVER_PORT=8002 docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
curl localhost:8002/health
```

Then switch over:

```bash
systemctl --user disable --now rvc-server   # old venv unit
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

CPU works as-is (the CUDA base image still runs; `device: cpu`). With the
NVIDIA card + driver + `nvidia-container-toolkit` installed:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

The GPU override adds `gpus: all` and
`ONNX_PROVIDER=CUDAExecutionProvider` (Kokoro on CUDA; otherwise
kokoro-onnx tries TensorRT first and falls back to CPU). Chunking defaults
to 20s for the 6GB RTX 2060; raise `RVC_SERVER_CHUNK_SEC` on 8GB+ cards.

Logs: `docker compose logs -f rvc-server`. Rebuild after changing server
code: `docker compose -f compose.yaml -f compose.gpu.yaml up -d --build`
(drop the overlay only on a CPU-only host; on ifrit it silently recreates
the container without the GPU). `ttsbot-diarize`, `rvc_infer`, models,
and caches are mounts, so updating them does not require a rebuild.

After a suspend/resume the worker's CUDA context is dead on the open
`nvidia-driver-595` modules (kernel 7.x): `/convert` fails with "CUDA-capable
device(s) is/are busy or unavailable" until the process restarts. Install
`systemd/rvc-server-resume.service` on the host so it does that automatically:

```bash
sudo cp systemd/rvc-server-resume.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable rvc-server-resume.service
```

The same condition is reported to the thin client as a transient 503, so it
falls back to its local worker instead of failing the command.

## Legacy venv (development only)

```bash
python3 -m venv ~/rvc-server-venv && source ~/rvc-server-venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu126 \
  torch==2.8.0+cu126 torchaudio==2.8.0+cu126 torchvision==0.23.0+cu126
pip install --no-deps -r requirements-gpu.txt
# requirements-gpu.txt pins ORT 1.30 (CUDA 13). Install the CUDA 12.x build
# or Kokoro silently falls back to CPUExecutionProvider.
pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0
python server.py
```

The old `start.sh`/systemd user unit starts this venv; Docker is the current
deployment path.

## Contract

HTTP `X-RVC-Protocol: 1` — bump only in lockstep with the thin client's
`RvcRunner`. 4xx = deterministic (fail fast); 5xx/timeout = transient (thin
client uses its local slow path).
