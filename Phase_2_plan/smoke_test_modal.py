"""Build and validate the DuplexOmni modified-vLLM fork on Modal.

This is intentionally a build smoke test. It does not download model
checkpoints or start the realtime serving stack.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

from modal_runtime import REPO_DIR, VLLM_DIR, VOLUME_NAME, get_volume


APP_NAME = "duplexomni-vllm-smoke-test"
app = modal.App(APP_NAME)
volume = get_volume(create_if_missing=True)


# Keep the compiler/toolkit and the Torch CUDA build on the same CUDA major
# line. The current DuplexOmni fork pins Torch 2.9.1.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "CUDA_PATH": "/usr/local/cuda",
            "VLLM_TARGET_DEVICE": "cuda",
            "MAX_JOBS": "8",
            "VERBOSE": "1",
            "NINJA_STATUS": "[%f/%t %o/sec] ",
        }
    )
    .apt_install(
        "git",
        "git-lfs",
        "build-essential",
        "cmake",
        "ninja-build",
        "ffmpeg",
        "libsndfile1",
    )
    .pip_install(
        # Versions required by the current modified-vLLM fork.
        "torch==2.9.1",
        "torchaudio==2.9.1",
        "torchvision==0.24.1",
        # Build requirements mirrored from requirements/build.txt.
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
    .add_local_python_source("modal_runtime")
)


def run_checked(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a command and preserve its output in the Modal task log."""

    printable = " ".join(command)
    print(f"\n[command:start] {printable}", flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=cwd, env=env, check=True)
    elapsed = time.monotonic() - started
    print(f"[command:done +{elapsed:.1f}s] {printable}", flush=True)


def resolve_cuda_home() -> Path:
    """Find the CUDA toolkit containing nvcc, not just the CUDA runtime."""

    candidates: list[Path] = []
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured:
        candidates.append(Path(configured))

    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(Path(nvcc).resolve().parent.parent)

    candidates.extend(
        [Path("/usr/local/cuda"), *sorted(Path("/usr/local").glob("cuda-*"))]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        nvcc_path = candidate / "bin" / "nvcc"
        if nvcc_path.is_file():
            return candidate

    visible = sorted(str(path) for path in Path("/usr/local").glob("cuda*"))
    raise RuntimeError(
        "CUDA toolkit was not found. Expected nvcc under /usr/local/cuda*/bin. "
        f"Visible CUDA paths: {visible or '<none>'}"
    )


def print_environment(cuda_home: Path) -> None:
    """Fail early with useful diagnostics before invoking CMake."""

    import torch

    print("\n[1/4] Environment")
    print(" - Python:", sys.version.replace("\n", " "))
    print(" - Torch:", torch.__version__)
    print(" - Torch CUDA:", torch.version.cuda)
    print(" - CUDA_HOME:", cuda_home)
    print(" - nvcc:", cuda_home / "bin" / "nvcc")

    if not torch.cuda.is_available():
        raise RuntimeError("Torch cannot access the Modal GPU.")

    device = torch.cuda.get_device_properties(0)
    print(" - GPU:", device.name)
    print(" - GPU compute capability:", f"{device.major}.{device.minor}")

    nvcc = cuda_home / "bin" / "nvcc"
    run_checked([str(nvcc), "--version"])

    required_paths = [
        cuda_home / "include" / "cuda_runtime_api.h",
        cuda_home / "lib64" / "libcudart.so",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"CUDA toolkit is incomplete; missing: {missing}")


def prepare_build_environment(cuda_home: Path) -> dict[str, str]:
    """Tell both PyTorch and CMake exactly which CUDA toolkit to use."""

    build_env = os.environ.copy()
    cuda_home = cuda_home.resolve()
    nvcc = cuda_home / "bin" / "nvcc"

    build_env.update(
        {
            "CUDA_HOME": str(cuda_home),
            "CUDA_PATH": str(cuda_home),
            "VLLM_TARGET_DEVICE": "cuda",
            "VERBOSE": "1",
            "NINJA_STATUS": "[%f/%t %o/sec] ",
            # Modal's smoke test uses a T4; compile only its architecture.
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", "7.5"),
            "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
            "CMAKE_ARGS": (
                f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_home} "
                f"-DCMAKE_CUDA_COMPILER={nvcc}"
            ),
        }
    )

    build_env["PATH"] = f"{cuda_home / 'bin'}:{build_env.get('PATH', '')}"
    build_env["LD_LIBRARY_PATH"] = (
        f"{cuda_home / 'lib64'}:{build_env.get('LD_LIBRARY_PATH', '')}"
    )

    # Help CMake find Torch's CMake package without replacing existing paths.
    import torch

    torch_cmake = str(torch.utils.cmake_prefix_path)
    existing_cmake = build_env.get("CMAKE_PREFIX_PATH", "")
    build_env["CMAKE_PREFIX_PATH"] = ":".join(
        value for value in (torch_cmake, existing_cmake) if value
    )
    return build_env


def clone_repo() -> None:
    """Use a clean shallow clone and never mutate upstream source files."""

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if not REPO_DIR.exists():
        print("\n[2/4] Cloning DuplexOmni")
        run_checked(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/MuyeHuang/DuplexOmni.git",
                str(REPO_DIR),
            ]
        )
    else:
        print("\n[2/4] Reusing existing repository in the smoke-test volume")

    run_checked(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"])


def build_vllm(build_env: dict[str, str]) -> None:
    """Install the fork's declared build requirements and build editable."""

    build_requirements = VLLM_DIR / "requirements" / "build.txt"
    if not build_requirements.is_file():
        raise FileNotFoundError(f"Missing build requirements: {build_requirements}")

    # setuptools-scm does not search the parent Git worktree by default, while
    # this package intentionally lives below the DuplexOmni repository root.
    # Use a reproducible local version for the smoke build instead of editing
    # the upstream pyproject.toml.
    revision = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
    build_env = build_env.copy()
    build_env["SETUPTOOLS_SCM_PRETEND_VERSION"] = f"0.0.0+duplexomni.{revision}"
    print(" - setuptools-scm fallback version:", build_env["SETUPTOOLS_SCM_PRETEND_VERSION"])

    # The fork declares FA2/FA3 as CMake extensions, but the source checkout
    # does not contain this package directory. During an editable build,
    # setuptools copies the compiled .so files into the source tree and fails
    # if the destination directory has not been created yet.
    flash_attn_package = VLLM_DIR / "vllm" / "vllm_flash_attn"
    flash_attn_package.mkdir(parents=True, exist_ok=True)
    print(" - ensured editable extension directory:", flash_attn_package)

    print("\n[3/4] Building modified vLLM")
    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            str(build_requirements),
        ],
        cwd=VLLM_DIR,
        env=build_env,
    )
    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-v",
            "--no-build-isolation",
            "--no-deps",
            "-e",
            ".",
        ],
        cwd=VLLM_DIR,
        env=build_env,
    )


def validate_import() -> None:
    """Validate the installed package and prove it resolves from this fork."""

    print("\n[4/4] Validating modified vLLM import", flush=True)
    print(" - Python:", sys.executable, flush=True)
    print(" - cwd:", Path.cwd(), flush=True)
    print(" - source package:", VLLM_DIR / "vllm", flush=True)
    print(" - source package exists:", (VLLM_DIR / "vllm").is_dir(), flush=True)
    print(" - sys.path before fallback:", sys.path, flush=True)
    # Modal invokes the function from /root. Make the fork source root explicit
    # if the editable finder is not visible in this runtime context.
    if str(VLLM_DIR) not in sys.path:
        sys.path.insert(0, str(VLLM_DIR))
        print(" - added fork source root to sys.path", flush=True)

    import vllm

    location = Path(vllm.__file__).resolve()
    print(" - vLLM version:", vllm.__version__)
    print(" - vLLM location:", location)
    source_package = VLLM_DIR / "vllm"
    try:
        from_fork = os.path.samefile(source_package, location.parent)
    except FileNotFoundError:
        from_fork = False
    if not from_fork:
        raise RuntimeError(
            f"Imported vLLM from {location}, not from the DuplexOmni fork at {source_package}"
        )


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    volumes={"/root/workspace": volume},
    timeout=5400,
    memory=32 * 1024,
)
def run_smoke_test() -> None:
    print("=" * 60)
    print("DuplexOmni modified-vLLM smoke test on Modal")
    print("=" * 60)

    cuda_home = resolve_cuda_home()
    print_environment(cuda_home)
    build_env = prepare_build_environment(cuda_home)
    clone_repo()
    build_vllm(build_env)
    # Persist compiled artifacts before validation so a diagnostic failure does
    # not discard a 45-minute CUDA build.
    volume.commit()
    print(" - committed compiled artifacts before validation", flush=True)
    validate_import()

    volume.commit()
    print(f"\nSUCCESS: build artifacts committed to Modal Volume {VOLUME_NAME!r}")


if __name__ == "__main__":
    with app.run():
        run_smoke_test.remote()
