"""Download and verify the public DuplexOmni checkpoint on Modal.

This job is intentionally separate from the vLLM build smoke test and from
model inference. It stores the checkpoint in a persistent Modal Volume so
later GPU jobs do not download it again.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

from modal_runtime import MODEL_ROOT, VOLUME_NAME, get_volume


APP_NAME = "duplexomni-model-download"
MODEL_ID = "MuyeHuang/DuplexOmni"
MANIFEST_PATH = MODEL_ROOT.parent / "duplexomni_model_manifest.json"

app = modal.App(APP_NAME)
volume = get_volume(create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.30.0", "tqdm")
)


def _iter_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts
    )


def _write_manifest(
    *,
    model_revision: str,
    started_at: float,
    completed_at: float,
    files: list[Path],
) -> None:
    payload = {
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "local_dir": str(MODEL_ROOT),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "started_unix": started_at,
        "completed_unix": completed_at,
        "files": [
            {
                "path": str(path.relative_to(MODEL_ROOT)),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@app.function(
    image=image,
    cpu=4,
    memory=16 * 1024,
    timeout=12 * 60 * 60,
    # Modal requires at least 524288 MiB for an explicitly requested disk.
    # Keep the model in the persistent Volume; this disk is only scratch/cache.
    ephemeral_disk=524288,
    volumes={"/root/workspace": volume},
)
def download_model() -> None:
    from huggingface_hub import HfApi, snapshot_download

    started = time.time()
    print("[phase:model-download] start", flush=True)
    print(f"[phase:model-download] model_id={MODEL_ID}", flush=True)
    print(f"[phase:model-download] target={MODEL_ROOT}", flush=True)

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    info = HfApi(token=token).model_info(MODEL_ID)
    revision = info.sha
    print(f"[phase:model-download] hub_revision={revision}", flush=True)

    downloaded_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=revision,
        local_dir=str(MODEL_ROOT),
        token=token,
        max_workers=8,
    )
    print(f"[phase:model-download] snapshot_path={downloaded_path}", flush=True)

    files = _iter_files(MODEL_ROOT)
    required_names = {"config.json"}
    present_names = {path.name for path in files}
    missing = sorted(required_names - present_names)
    if missing:
        raise RuntimeError(f"Checkpoint verification failed; missing files: {missing}")

    completed = time.time()
    _write_manifest(
        model_revision=revision,
        started_at=started,
        completed_at=completed,
        files=files,
    )
    manifest_size = MANIFEST_PATH.stat().st_size
    total_bytes = sum(path.stat().st_size for path in files)
    print(f"[phase:model-download] files={len(files)}", flush=True)
    print(f"[phase:model-download] total_bytes={total_bytes}", flush=True)
    print(f"[phase:model-download] manifest_bytes={manifest_size}", flush=True)
    print(f"[phase:model-download] elapsed_seconds={completed - started:.1f}", flush=True)

    volume.commit()
    print(
        f"[phase:model-download] SUCCESS committed volume={VOLUME_NAME} "
        f"revision={revision}",
        flush=True,
    )


if __name__ == "__main__":
    with app.run():
        download_model.remote()
