"""Validate the inference runtime image against the persisted source artifact.

This job intentionally does not load the 70 GB checkpoint. It catches missing
Python/runtime dependencies before allocating the 8-GPU model-load job.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal


APP_NAME = "duplexomni-runtime-import-smoke"
VOLUME_NAME = "duplexomni-smoke-vol-v2"
WORKSPACE = Path("/root/workspace")
REPO_DIR = WORKSPACE / "DuplexOmni"
VLLM_DIR = REPO_DIR / "inference_framework" / "vllm_qwen3_omni"
SERVING_DIR = REPO_DIR / "inference_framework" / "realtime_serving"
SERVING_CORE_DIR = SERVING_DIR / "serving_core"
VLLM_VERSION = "0.0.0+duplexomni.33bfba1"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
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
)


def _ensure_vllm_metadata() -> None:
    """Expose the persisted source artifact as an installed distribution."""

    dist_info = VLLM_DIR / f"vllm-{VLLM_VERSION}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: vllm\n"
        f"Version: {VLLM_VERSION}\n",
        encoding="utf-8",
    )
    (dist_info / "top_level.txt").write_text("vllm\n", encoding="utf-8")
    (dist_info / "RECORD").write_text("", encoding="utf-8")


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16 * 1024,
    timeout=45 * 60,
    volumes={"/root/workspace": volume},
)
def run_import_smoke() -> None:
    print("[phase:runtime-import] start", flush=True)
    for path in (REPO_DIR, VLLM_DIR, SERVING_DIR, SERVING_CORE_DIR):
        if not path.exists():
            raise RuntimeError(f"Missing persisted source path: {path}")

    sys.path.insert(0, str(VLLM_DIR))
    sys.path.insert(0, str(SERVING_DIR))
    sys.path.insert(0, str(SERVING_CORE_DIR))
    os.environ["VLLM_TARGET_DEVICE"] = "cuda"
    os.environ["VLLM_ROOT"] = str(VLLM_DIR)
    _ensure_vllm_metadata()
    volume.commit()

    import torch

    print(f"[phase:runtime-import] python={sys.version.replace(chr(10), ' ')}", flush=True)
    print(f"[phase:runtime-import] torch={torch.__version__}", flush=True)
    print(f"[phase:runtime-import] torch_cuda={torch.version.cuda}", flush=True)
    print(f"[phase:runtime-import] cuda_available={torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Runtime image cannot access the Modal GPU")

    import vllm

    print(f"[phase:runtime-import] vllm_version={vllm.__version__}", flush=True)
    print(f"[phase:runtime-import] vllm_file={vllm.__file__}", flush=True)

    modules = (
        "serving_core.server_thinker",
        "serving_core.server_talker",
        "serving_core.server_orchestrator",
    )
    for module_name in modules:
        print(f"[phase:runtime-import] importing={module_name}", flush=True)
        __import__(module_name)
        print(f"[phase:runtime-import] imported={module_name}", flush=True)

    print("[phase:runtime-import] SUCCESS", flush=True)


if __name__ == "__main__":
    with app.run():
        run_import_smoke.remote()
