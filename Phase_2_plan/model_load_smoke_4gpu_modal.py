"""Cost-effective 4-GPU model-load experiment for DuplexOmni.

This keeps the official 8-GPU launcher untouched. It tests the documented
4x-A100 target with Thinker TP=2 and Talker TP=2, then stops after health
checks; no end-to-end request is sent here.
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


APP_NAME = "duplexomni-model-load-smoke-4gpu"
VOLUME_NAME = "duplexomni-smoke-vol-v2"
WORKSPACE = Path("/root/workspace")
REPO_DIR = WORKSPACE / "DuplexOmni"
VLLM_DIR = REPO_DIR / "inference_framework" / "vllm_qwen3_omni"
SERVING_DIR = REPO_DIR / "inference_framework" / "realtime_serving"
CORE_DIR = SERVING_DIR / "serving_core"
MODEL_ROOT = WORKSPACE / "models" / "DuplexOmni"
VLLM_VERSION = "0.0.0+duplexomni.33bfba1"
LOG_DIR = SERVING_DIR / "logs_4gpu"

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
            return response.status == 200, f"status={response.status} body={body[:200]}"
    except (OSError, urllib.error.URLError) as exc:
        return False, f"error={exc}"


def _drain_logs(offsets: dict[Path, int]) -> None:
    for name in ("thinker", "talker", "orchestrator"):
        path = LOG_DIR / f"{name}.log"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(offsets.get(path, 0))
            for line in handle.readlines():
                print(f"[service:{name}] {line.rstrip()}", flush=True)
            offsets[path] = handle.tell()


def _start(name: str, script: Path, log: Path, env: dict[str, str], args: list[str]) -> subprocess.Popen[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    print(f"[phase:model-load-4gpu] start_service={name} log={log}", flush=True)
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
    timeout=4 * 60 * 60,
    startup_timeout=20 * 60,
    volumes={"/root/workspace": volume},
)
def run_model_load_smoke_4gpu() -> None:
    print("[phase:model-load-4gpu] start", flush=True)
    print("[phase:model-load-4gpu] layout=thinker:0,1 tp=2; talker:2,3 tp=2; mtp:3", flush=True)
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
            "CC": "gcc",
            "CXX": "g++",
            "THINKER_MODEL": str(MODEL_ROOT),
            "TALKER_MODEL": str(MODEL_ROOT),
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_USE_V1": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONPATH": ":".join((str(VLLM_DIR), str(SERVING_DIR), str(CORE_DIR), base_env.get("PYTHONPATH", ""))),
        }
    )
    thinker_env = {**base_env, "CUDA_VISIBLE_DEVICES": "0,1", "THINKER_PORT": "19999", "THINKER_TP": "2", "THINKER_GPU_MEM_UTIL": "0.8"}
    talker_env = {
        **base_env,
        "CUDA_VISIBLE_DEVICES": "2,3",
        "TALKER_PORT": "20000",
        "TALKER_TP": "2",
        "TALKER_GPU_MEM_UTIL": "0.6",
        "TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES": "1",
        "TALKER_MTP_SUBPROC_TP": "1",
        "TALKER_MTP_GPU_MEM_UTIL": "0.12",
        "TALKER_MTP_PROFILE": "0",
    }
    orch_env = {**base_env, "ORCH_PORT": "21000", "THINKER_INTERNAL_URL": "http://127.0.0.1:19999/internal/chat_turn", "TALKER_INTERNAL_URL_BASE": "http://127.0.0.1:20000/internal/talker/turn", "TALKER_DELETE_URL_BASE": "http://127.0.0.1:20000/v1/talker/session"}

    for path in LOG_DIR.glob("*.log"):
        path.unlink()
    processes: list[subprocess.Popen[str]] = []
    offsets: dict[Path, int] = {}
    started = time.monotonic()
    try:
        thinker = _start("thinker", CORE_DIR / "server_thinker.py", LOG_DIR / "thinker.log", thinker_env, [])
        processes.append(thinker)
        talker = _start("talker", CORE_DIR / "server_talker.py", LOG_DIR / "talker.log", talker_env, ["--tp", "2"])
        processes.append(talker)
        orch = _start("orchestrator", CORE_DIR / "server_orchestrator.py", LOG_DIR / "orchestrator.log", orch_env, [])
        processes.append(orch)

        stable_since: float | None = None
        last_report = 0.0
        while time.monotonic() - started < 3.5 * 60 * 60:
            _drain_logs(offsets)
            statuses = {name: _health(port) for name, port in (("thinker", 19999), ("talker", 20000), ("orchestrator", 21000))}
            now = time.monotonic()
            if now - last_report >= 30:
                print("[phase:model-load-4gpu] health " + " ".join(f"{n}={'UP' if ok else 'DOWN'}({d})" for n, (ok, d) in statuses.items()), flush=True)
                last_report = now
            if all(ok for ok, _ in statuses.values()):
                stable_since = stable_since or now
                if now - stable_since >= 30:
                    _drain_logs(offsets)
                    print(f"[phase:model-load-4gpu] ready_elapsed_seconds={now - started:.1f}", flush=True)
                    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"], check=False, capture_output=True, text=True)
                    print("[phase:model-load-4gpu] nvidia_smi=" + smi.stdout.strip(), flush=True)
                    print("[phase:model-load-4gpu] SUCCESS", flush=True)
                    return
            else:
                stable_since = None
            for name, process in zip(("thinker", "talker", "orchestrator"), processes):
                if process.poll() is not None:
                    _drain_logs(offsets)
                    raise RuntimeError(f"{name} exited with code {process.returncode}")
            time.sleep(10)
    finally:
        _drain_logs(offsets)
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    with app.run():
        run_model_load_smoke_4gpu.remote()
