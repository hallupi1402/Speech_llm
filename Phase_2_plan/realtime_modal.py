"""Persistent Modal WebSocket endpoint for direct microphone testing.

The model services stay inside one 4-GPU container and the upstream
``omni_realtime_server.py`` is exposed on port 8765.  Run with ``modal serve``
for an ephemeral URL while testing, then connect the local microphone client
to the printed ``wss://`` URL.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal


APP_NAME = "duplexomni-realtime-4gpu"
VOLUME_NAME = "duplexomni-smoke-vol-v2"
WORKSPACE = Path("/root/workspace")
REPO_DIR = WORKSPACE / "DuplexOmni"
VLLM_DIR = REPO_DIR / "inference_framework" / "vllm_qwen3_omni"
SERVING_DIR = REPO_DIR / "inference_framework" / "realtime_serving"
CORE_DIR = SERVING_DIR / "serving_core"
MODEL_ROOT = WORKSPACE / "models" / "DuplexOmni"
VLLM_VERSION = "0.0.0+duplexomni.33bfba1"
LOG_DIR = SERVING_DIR / "logs_realtime_4gpu"

RUNTIME_PACKAGES = (
    "torch==2.9.1", "torchaudio==2.9.1", "torchvision==0.24.1",
    "fastapi", "uvicorn", "websockets", "openai", "librosa", "soundfile", "scipy",
    "requests", "aiohttp", "transformers>=4.56.0,<5", "safetensors", "sentencepiece",
    "protobuf>=5.29.6", "regex", "packaging>=24.2", "jinja2>=3.1.6", "numpy", "msgspec",
    "cachetools", "psutil", "blake3", "py-cpuinfo", "prometheus_client",
    "prometheus-fastapi-instrumentator", "tiktoken", "diskcache", "lark",
    "partial-json-parser", "pyzmq", "gguf>=0.17.0", "einops", "compressed-tensors==0.13.0",
    "depyf==0.20.0", "cloudpickle", "watchfiles", "python-json-logger", "ninja", "pybase64",
    "cbor2", "ijson", "setproctitle", "grpcio", "grpcio-reflection", "openai-harmony>=0.0.3",
    "anthropic>=0.71.0", "model-hosting-container-standards>=0.1.13,<1.0.0", "mcp",
    "ray[cgraph]>=2.48.0", "lm-format-enforcer==0.11.3", "llguidance>=1.3.0,<1.4.0",
    "outlines_core==0.2.11", "xgrammar==0.1.29", "mistral_common[image]>=1.9.0",
    "opencv-python-headless>=4.13.0", "flashinfer-python==0.6.3",
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(*RUNTIME_PACKAGES)
    .run_commands("apt-get update && apt-get install -y --no-install-recommends build-essential")
)


def _ensure_vllm_metadata() -> None:
    dist_info = VLLM_DIR / f"vllm-{VLLM_VERSION}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: vllm\n"
        f"Version: {VLLM_VERSION}\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text("vllm\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")


def _health(port: int) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status == 200, body[:200]
    except (OSError, urllib.error.URLError) as exc:
        return False, str(exc)


def _start(name: str, script: Path, log: Path, env: dict[str, str], args: list[str]) -> subprocess.Popen[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    print(f"[phase:realtime] start_service={name} log={log}", flush=True)
    return subprocess.Popen(
        [sys.executable, str(script), *args],
        cwd=CORE_DIR,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


@app.function(
    image=image,
    gpu="A100-80GB:4",
    cpu=16,
    memory=64 * 1024,
    timeout=24 * 60 * 60,
    startup_timeout=20 * 60,
    volumes={"/root/workspace": volume},
)
@modal.concurrent(max_inputs=4)
@modal.web_server(8765, startup_timeout=20 * 60)
def serve_realtime() -> None:
    print("[phase:realtime] start", flush=True)
    print("[phase:realtime] layout=thinker:0,1 tp=2; talker:2,3 tp=2; mtp:3", flush=True)
    if not MODEL_ROOT.is_dir():
        raise RuntimeError(f"Missing checkpoint: {MODEL_ROOT}")

    _ensure_vllm_metadata()
    volume.commit()
    base_env = os.environ.copy()
    base_env.update(
        {
            "VLLM_ROOT": str(VLLM_DIR),
            "VLLM_TARGET_DEVICE": "cuda",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "CC": "gcc", "CXX": "g++",
            "THINKER_MODEL": str(MODEL_ROOT),
            "TALKER_MODEL": str(MODEL_ROOT),
            "S1_MODEL_NAME": str(MODEL_ROOT),
            "S2_MODEL_NAME": "disabled",
            "S2_THINK_BASE_URL": "http://127.0.0.1:9/v1",
            "S2_API_KEY": "EMPTY",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ":".join((str(VLLM_DIR), str(SERVING_DIR), str(CORE_DIR), base_env.get("PYTHONPATH", ""))),
        }
    )
    thinker_env = {
        **base_env, "CUDA_VISIBLE_DEVICES": "0,1", "THINKER_PORT": "19999",
        "THINKER_TP": "2", "THINKER_GPU_MEM_UTIL": "0.8",
    }
    talker_env = {
        **base_env, "CUDA_VISIBLE_DEVICES": "2,3", "TALKER_PORT": "20000",
        "TALKER_TP": "2", "TALKER_GPU_MEM_UTIL": "0.6",
        "TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES": "1", "TALKER_MTP_SUBPROC_TP": "1",
        "TALKER_MTP_GPU_MEM_UTIL": "0.12", "TALKER_MTP_PROFILE": "0",
    }
    orch_env = {
        **base_env, "ORCH_PORT": "21000",
        "THINKER_INTERNAL_URL": "http://127.0.0.1:19999/internal/chat_turn",
        "TALKER_INTERNAL_URL_BASE": "http://127.0.0.1:20000/internal/talker/turn",
        "TALKER_DELETE_URL_BASE": "http://127.0.0.1:20000/v1/talker/session",
    }
    bridge_env = {
        **base_env,
        "S1_OMNI_BASE_URL": "http://127.0.0.1:21000/v1",
        "S1_AUDIO_WS_BASE": "ws://127.0.0.1:21000/v1/audio/stream",
        "S1_CLEAR_SESSION_BASE": "http://127.0.0.1:21000/v1/session",
        "S1_HEALTH_URL": "http://127.0.0.1:21000/health",
    }
    for path in LOG_DIR.glob("*.log"):
        path.unlink()

    processes: list[subprocess.Popen[str]] = []
    processes.append(_start("thinker", CORE_DIR / "server_thinker.py", LOG_DIR / "thinker.log", thinker_env, []))
    processes.append(_start("talker", CORE_DIR / "server_talker.py", LOG_DIR / "talker.log", talker_env, ["--tp", "2"]))
    processes.append(_start("orchestrator", CORE_DIR / "server_orchestrator.py", LOG_DIR / "orchestrator.log", orch_env, []))

    started = time.monotonic()
    stable_since: float | None = None
    while time.monotonic() - started < 3.5 * 60 * 60:
        statuses = {name: _health(port) for name, port in (("thinker", 19999), ("talker", 20000), ("orchestrator", 21000))}
        print("[phase:realtime] health " + " ".join(f"{name}={'UP' if ok else 'DOWN'}({detail})" for name, (ok, detail) in statuses.items()), flush=True)
        if all(ok for ok, _ in statuses.values()):
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= 30:
                break
        else:
            stable_since = None
        for name, process in zip(("thinker", "talker", "orchestrator"), processes):
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited with code {process.returncode}")
        time.sleep(10)
    else:
        raise TimeoutError("Timed out waiting for Thinker/Talker/Orchestrator")

    print(f"[phase:realtime] services_ready_elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
    bridge = _start(
        "realtime-bridge",
        SERVING_DIR / "omni_realtime_server.py",
        LOG_DIR / "bridge.log",
        bridge_env,
        ["--host", "0.0.0.0", "--port", "8765"],
    )
    processes.append(bridge)
    time.sleep(2)
    if bridge.poll() is not None:
        raise RuntimeError(f"realtime bridge exited with code {bridge.returncode}")
    print("[phase:realtime] READY websocket_port=8765", flush=True)



if __name__ == "__main__":
    with app.run():
        serve_realtime.remote()
