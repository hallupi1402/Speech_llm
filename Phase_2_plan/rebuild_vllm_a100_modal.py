"""Rebuild the modified DuplexOmni vLLM fork for A100 (sm80) on Modal."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal


APP_NAME = "duplexomni-vllm-rebuild-a100"
VOLUME_NAME = "duplexomni-smoke-vol-v2"
WORKSPACE = Path("/root/workspace")
REPO_DIR = WORKSPACE / "DuplexOmni"
VLLM_DIR = REPO_DIR / "inference_framework" / "vllm_qwen3_omni"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "VLLM_TARGET_DEVICE": "cuda",
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "MAX_JOBS": "4",
            "VERBOSE": "1",
            "NINJA_STATUS": "[%f/%t %o/sec] ",
        }
    )
    .apt_install("git", "git-lfs", "build-essential", "cmake", "ninja-build", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.9.1",
        "torchaudio==2.9.1",
        "torchvision==0.24.1",
        "cmake>=3.26.1",
        "ninja",
        "packaging>=24.2",
        "setuptools>=77.0.3,<81.0.0",
        "setuptools-scm>=8",
        "wheel",
        "jinja2>=3.1.6",
        "regex",
        "build",
        "protobuf>=5.29.6",
        "grpcio-tools==1.78.0",
    )
)


def run_checked(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = " ".join(command)
    print(f"[phase:a100-rebuild] command_start={printable}", flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=cwd, env=env, check=True)
    print(f"[phase:a100-rebuild] command_done elapsed={time.monotonic() - started:.1f}s", flush=True)


def clean_build_outputs() -> None:
    """Remove only generated vLLM build outputs, retaining source and model files."""

    targets = [VLLM_DIR / "build", VLLM_DIR / "vllm" / "vllm_flash_attn"]
    for target in targets:
        if target.exists():
            print(f"[phase:a100-rebuild] remove_generated={target}", flush=True)
            shutil.rmtree(target)
    for path in VLLM_DIR.glob("vllm/**/*.so"):
        print(f"[phase:a100-rebuild] remove_generated={path}", flush=True)
        path.unlink()
    (VLLM_DIR / "vllm" / "vllm_flash_attn").mkdir(parents=True, exist_ok=True)


def prepare_env() -> dict[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access the Modal A100")
    device = torch.cuda.get_device_properties(0)
    print(
        f"[phase:a100-rebuild] gpu={device.name} capability={device.major}.{device.minor} "
        f"torch={torch.__version__} torch_cuda={torch.version.cuda}",
        flush=True,
    )
    if (device.major, device.minor) != (8, 0):
        raise RuntimeError(f"Expected A100 sm80, got sm{device.major}{device.minor}")
    env = os.environ.copy()
    revision = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    cuda_home = Path("/usr/local/cuda")
    env.update(
        {
            "CUDA_HOME": str(cuda_home),
            "CUDA_PATH": str(cuda_home),
            "VLLM_TARGET_DEVICE": "cuda",
            "TORCH_CUDA_ARCH_LIST": "8.0",
            "SETUPTOOLS_SCM_PRETEND_VERSION": f"0.0.0+duplexomni.{revision}",
            "MAX_JOBS": "4",
            "CMAKE_ARGS": f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_home} -DCMAKE_CUDA_COMPILER={cuda_home / 'bin' / 'nvcc'}",
        }
    )
    env["PATH"] = f"{cuda_home / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{cuda_home / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["CMAKE_PREFIX_PATH"] = ":".join(
        value for value in (str(torch.utils.cmake_prefix_path), env.get("CMAKE_PREFIX_PATH", "")) if value
    )
    print(f"[phase:a100-rebuild] revision={revision} arch={env['TORCH_CUDA_ARCH_LIST']}", flush=True)
    return env


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8,
    memory=32 * 1024,
    timeout=4 * 60 * 60,
    startup_timeout=20 * 60,
    volumes={"/root/workspace": volume},
)
def rebuild_vllm_a100() -> None:
    print("[phase:a100-rebuild] start", flush=True)
    if not VLLM_DIR.is_dir():
        raise RuntimeError(f"Missing vLLM source: {VLLM_DIR}")
    env = prepare_env()
    clean_build_outputs()
    requirements = VLLM_DIR / "requirements" / "build.txt"
    run_checked([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements)], cwd=VLLM_DIR, env=env)
    run_checked(
        [sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation", "--no-deps", "-e", "."],
        cwd=VLLM_DIR,
        env=env,
    )
    volume.commit()
    print("[phase:a100-rebuild] committed_architecture_specific_artifacts", flush=True)


if __name__ == "__main__":
    with app.run():
        rebuild_vllm_a100.remote()
