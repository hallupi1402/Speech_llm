"""Shared Modal runtime configuration for DuplexOmni inference jobs.

This module is the single interface for paths, the persistent Volume, the
runtime Image, and the official 8-GPU service environment. Individual jobs
should own only their experiment lifecycle (download, smoke, batch, or
realtime), not duplicate infrastructure configuration.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import modal


APP_PREFIX = "duplexomni"
VOLUME_NAME = "duplexomni-smoke-vol-v2"
WORKSPACE = Path("/root/workspace")
REPO_DIR = WORKSPACE / "DuplexOmni"
VLLM_DIR = REPO_DIR / "inference_framework" / "vllm_qwen3_omni"
SERVING_DIR = REPO_DIR / "inference_framework" / "realtime_serving"
CORE_DIR = SERVING_DIR / "serving_core"
MODEL_ROOT = WORKSPACE / "models" / "DuplexOmni"
MODEL_ID = "MuyeHuang/DuplexOmni"
MODEL_MANIFEST_PATH = MODEL_ROOT.parent / "duplexomni_model_manifest.json"
VLLM_VERSION = "0.0.0+duplexomni.33bfba1"

THINKER_PORT = 19999
TALKER_PORT = 20000
ORCHESTRATOR_PORT = 21000
REALTIME_PORT = 8765

# Official source layout: Thinker TP=4 on GPUs 0-3 and Talker TP=4 on GPUs
# 4-7, with the Talker MTP subprocess sharing the final Talker GPU.
GPU_SPEC = "A100-80GB:8"
INFERENCE_CPU_COUNT = 32
BUILD_CPU_COUNT = 8
BUILD_MAX_JOBS = 4
MEMORY_MIB = 128 * 1024
JOB_TIMEOUT_SECONDS = 4 * 60 * 60
STARTUP_TIMEOUT_SECONDS = 20 * 60

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


def get_volume(*, create_if_missing: bool = False) -> modal.Volume:
    """Return the named persistent Volume for the active Workspace."""

    return modal.Volume.from_name(VOLUME_NAME, create_if_missing=create_if_missing)


def runtime_image() -> modal.Image:
    """Build the shared inference image definition (no GPU allocation yet)."""

    return (
        modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu22.04", add_python="3.11")
        .entrypoint([])
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install(*RUNTIME_PACKAGES)
        .run_commands("apt-get update && apt-get install -y --no-install-recommends build-essential")
        .add_local_python_source("modal_runtime")
    )


def build_image() -> modal.Image:
    """Build the CUDA-devel image used only by the vLLM compiler job."""

    return (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
        .entrypoint([])
        .env(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "CUDA_PATH": "/usr/local/cuda",
                "VLLM_TARGET_DEVICE": "cuda",
                "TORCH_CUDA_ARCH_LIST": "8.0",
                "MAX_JOBS": str(BUILD_MAX_JOBS),
                "VERBOSE": "1",
                "NINJA_STATUS": "[%f/%t %o/sec] ",
            }
        )
        .apt_install("git", "git-lfs", "build-essential", "cmake", "ninja-build", "ffmpeg", "libsndfile1")
        .pip_install(
            "torch==2.9.1", "torchaudio==2.9.1", "torchvision==0.24.1",
            "cmake>=3.26.1", "ninja", "packaging>=24.2", "setuptools>=77.0.3,<81.0.0",
            "setuptools-scm>=8", "wheel", "jinja2>=3.1.6", "regex", "build",
            "protobuf>=5.29.6", "grpcio-tools==1.78.0",
        )
        .add_local_python_source("modal_runtime")
    )


def ensure_vllm_metadata() -> None:
    """Expose the persisted editable vLLM tree as an installed distribution."""

    dist_info = VLLM_DIR / f"vllm-{VLLM_VERSION}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: vllm\n"
        f"Version: {VLLM_VERSION}\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text("vllm\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")


def inference_env() -> dict[str, str]:
    """Return the canonical environment shared by all inference services."""

    env = os.environ.copy()
    env.update(
        {
            "VLLM_ROOT": str(VLLM_DIR),
            "VLLM_TARGET_DEVICE": "cuda",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "CC": "gcc",
            "CXX": "g++",
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
            "PYTHONPATH": ":".join(
                value for value in (str(VLLM_DIR), str(SERVING_DIR), str(CORE_DIR), env.get("PYTHONPATH", "")) if value
            ),
        }
    )
    return env


def official_launcher_env() -> dict[str, str]:
    """Return env expected by the upstream 8-GPU launcher scripts."""

    env = inference_env()
    env.update(
        {
            "PYTHON_BIN": sys.executable,
            "ROOT_DIR": str(SERVING_DIR),
        }
    )
    return env


def health(port: int) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            body = response.read().decode("utf-8", errors="replace")
            return response.status == 200, f"status={response.status} body={body[:300]}"
    except (OSError, urllib.error.URLError) as exc:
        return False, f"error={exc}"


def print_nvidia_smi(prefix: str) -> None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    print(f"[{prefix}] nvidia_smi_start", flush=True)
    print(result.stdout.rstrip() or result.stderr.rstrip(), flush=True)
    print(f"[{prefix}] nvidia_smi_end", flush=True)
