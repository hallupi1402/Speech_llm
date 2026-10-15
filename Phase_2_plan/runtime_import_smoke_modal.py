"""Validate the inference runtime image against the persisted source artifact.

This job intentionally does not load the 70 GB checkpoint. It catches missing
Python/runtime dependencies before allocating the 8-GPU model-load job.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

from modal_runtime import (
    ensure_vllm_metadata,
    REPO_DIR,
    SERVING_DIR,
    VLLM_DIR,
    get_volume,
    runtime_image,
)


APP_NAME = "duplexomni-runtime-import-smoke"
SERVING_CORE_DIR = SERVING_DIR / "serving_core"

app = modal.App(APP_NAME)
volume = get_volume()
image = runtime_image()


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
    ensure_vllm_metadata()
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
