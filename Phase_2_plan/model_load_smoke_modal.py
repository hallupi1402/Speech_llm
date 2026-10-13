"""Load the official DuplexOmni Thinker/Talker stack on one Modal node.

This is a model-load smoke test only: it starts the three services, waits for
their health endpoints, records startup logs/VRAM, and does not send an E2E
request yet.
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


APP_NAME = "duplexomni-model-load-smoke"
VOLUME_NAME = "duplexomni-smoke-vol-v2"
WORKSPACE = Path("/root/workspace")
REPO_DIR = WORKSPACE / "DuplexOmni"
VLLM_DIR = REPO_DIR / "inference_framework" / "vllm_qwen3_omni"
SERVING_DIR = REPO_DIR / "inference_framework" / "realtime_serving"
MODEL_ROOT = WORKSPACE / "models" / "DuplexOmni"
LOG_DIR = SERVING_DIR / "logs"
START_SCRIPT = SERVING_DIR / "start_thinker_talker_stack.sh"
STOP_SCRIPT = SERVING_DIR / "stop_thinker_talker_stack.sh"

RUNTIME_PACKAGES = (
    "torch==2.9.1",
    "torchaudio==2.9.1",
    "torchvision==0.24.1",
    "fastapi",
    "uvicorn",
    "websockets",
    "openai",
    "librosa",
    "soundfile",
    "scipy",
    "requests",
    "aiohttp",
    "transformers>=4.56.0,<5",
    "safetensors",
    "sentencepiece",
    "protobuf>=5.29.6",
    "regex",
    "packaging>=24.2",
    "jinja2>=3.1.6",
    "numpy",
    "msgspec",
    "cachetools",
    "psutil",
    "blake3",
    "py-cpuinfo",
    "prometheus_client",
    "prometheus-fastapi-instrumentator",
    "tiktoken",
    "diskcache",
    "lark",
    "partial-json-parser",
    "pyzmq",
    "gguf>=0.17.0",
    "einops",
    "compressed-tensors==0.13.0",
    "depyf==0.20.0",
    "cloudpickle",
    "watchfiles",
    "python-json-logger",
    "ninja",
    "pybase64",
    "cbor2",
    "ijson",
    "setproctitle",
    "grpcio",
    "grpcio-reflection",
    "openai-harmony>=0.0.3",
    "anthropic>=0.71.0",
    "model-hosting-container-standards>=0.1.13,<1.0.0",
    "mcp",
    "ray[cgraph]>=2.48.0",
    "lm-format-enforcer==0.11.3",
    "llguidance>=1.3.0,<1.4.0",
    "outlines_core==0.2.11",
    "xgrammar==0.1.29",
    "mistral_common[image]>=1.9.0",
    "opencv-python-headless>=4.13.0",
    "flashinfer-python==0.6.3",
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(*RUNTIME_PACKAGES)
    .run_commands("apt-get update && apt-get install -y --no-install-recommends build-essential")
)


def _read_new_lines(path: Path, offsets: dict[Path, int]) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offsets.get(path, 0))
        lines = handle.readlines()
        offsets[path] = handle.tell()
    return lines


def _print_logs(offsets: dict[Path, int]) -> None:
    for name in ("thinker", "talker", "orchestrator"):
        path = LOG_DIR / f"{name}.log"
        for line in _read_new_lines(path, offsets):
            print(f"[service:{name}] {line.rstrip()}", flush=True)


def _health(port: int) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status == 200, f"status={response.status} body={body[:300]}"
    except (OSError, urllib.error.URLError) as exc:
        return False, f"error={exc}"


def _nvidia_smi() -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    print("[phase:model-load] nvidia_smi_start", flush=True)
    print(result.stdout.rstrip() or result.stderr.rstrip(), flush=True)
    print("[phase:model-load] nvidia_smi_end", flush=True)


@app.function(
    image=image,
    gpu="A100-80GB:8",
    cpu=32,
    memory=128 * 1024,
    timeout=4 * 60 * 60,
    volumes={"/root/workspace": volume},
)
def run_model_load_smoke() -> None:
    print("[phase:model-load] start", flush=True)
    print(f"[phase:model-load] source_commit=33bfba1", flush=True)
    print(f"[phase:model-load] model={MODEL_ROOT}", flush=True)
    print("[phase:model-load] gpu_layout=thinker:0-3,talker:4-7,mtp:7", flush=True)
    if not MODEL_ROOT.is_dir():
        raise RuntimeError(f"Missing checkpoint: {MODEL_ROOT}")
    if not START_SCRIPT.is_file() or not STOP_SCRIPT.is_file():
        raise RuntimeError("Missing official stack start/stop scripts")

    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "ROOT_DIR": str(SERVING_DIR),
            "VLLM_ROOT": str(VLLM_DIR),
            "THINKER_MODEL": str(MODEL_ROOT),
            "TALKER_MODEL": str(MODEL_ROOT),
            "S1_MODEL_NAME": "qwen3-omni",
            "S2_MODEL_NAME": "disabled",
            "S2_THINK_BASE_URL": "http://127.0.0.1:8000/v1",
            "S2_API_KEY": "EMPTY",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONPATH": ":".join(
                value for value in (str(VLLM_DIR), str(SERVING_DIR), env.get("PYTHONPATH", "")) if value
            ),
        }
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    offsets: dict[Path, int] = {}
    started = time.monotonic()
    launcher = subprocess.run(
        ["bash", str(START_SCRIPT)],
        cwd=SERVING_DIR,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    print(f"[phase:model-load] launcher_exit={launcher.returncode}", flush=True)
    print(launcher.stdout.rstrip(), flush=True)
    if launcher.returncode != 0:
        print(launcher.stderr.rstrip(), flush=True)
        raise RuntimeError("Official stack launcher failed")

    try:
        last_health_print = 0.0
        stable_ready_since: float | None = None
        while time.monotonic() - started < 3.5 * 60 * 60:
            _print_logs(offsets)
            statuses = {name: _health(port) for name, port in (("thinker", 19999), ("talker", 20000), ("orchestrator", 21000))}
            ready = all(value[0] for value in statuses.values())
            now = time.monotonic()
            if now - last_health_print >= 30:
                print(
                    "[phase:model-load] health "
                    + " ".join(f"{name}={'UP' if ok else 'DOWN'}({detail})" for name, (ok, detail) in statuses.items()),
                    flush=True,
                )
                last_health_print = now
            if ready:
                if stable_ready_since is None:
                    stable_ready_since = now
                    print("[phase:model-load] all_health_200; stabilizing", flush=True)
                elif now - stable_ready_since >= 30:
                    print(f"[phase:model-load] ready_elapsed_seconds={now - started:.1f}", flush=True)
                    _print_logs(offsets)
                    _nvidia_smi()
                    print("[phase:model-load] SUCCESS", flush=True)
                    return
            else:
                stable_ready_since = None

            for name in ("thinker", "talker", "orchestrator"):
                pid_path = SERVING_DIR / "run" / f"{name}.pid"
                if pid_path.exists():
                    pid = int(pid_path.read_text(encoding="utf-8").strip())
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError as exc:
                        _print_logs(offsets)
                        raise RuntimeError(f"{name} process exited before readiness") from exc
            time.sleep(10)
    finally:
        _print_logs(offsets)
        stop = subprocess.run(
            ["bash", str(STOP_SCRIPT)],
            cwd=SERVING_DIR,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        print(f"[phase:model-load] stop_exit={stop.returncode}", flush=True)
        print(stop.stdout.rstrip(), flush=True)
        if stop.stderr:
            print(stop.stderr.rstrip(), flush=True)


if __name__ == "__main__":
    with app.run():
        run_model_load_smoke.remote()
