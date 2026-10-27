"""Inspect the downloaded DuplexOmni checkpoint in the persistent Modal Volume."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from modal_runtime import MODEL_MANIFEST_PATH, MODEL_ROOT, get_volume


APP_NAME = "duplexomni-model-inspect"

app = modal.App(APP_NAME)
volume = get_volume()
image = modal.Image.debian_slim(python_version="3.11")


def _print_json_summary(path: Path) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[phase:model-inspect] json_error path={path} error={exc}", flush=True)
        return
    if not isinstance(data, dict):
        print(f"[phase:model-inspect] json path={path} type={type(data).__name__}", flush=True)
        return
    keys = sorted(data)
    print(f"[phase:model-inspect] json path={path.relative_to(MODEL_ROOT)} keys={keys}", flush=True)
    for key in ("model_type", "architectures", "torch_dtype", "num_hidden_layers", "hidden_size"):
        if key in data:
            print(f"[phase:model-inspect] config.{key}={data[key]}", flush=True)


@app.function(
    image=image,
    cpu=2,
    memory=4 * 1024,
    timeout=15 * 60,
    volumes={"/root/workspace": volume},
)
def inspect_model() -> None:
    print("[phase:model-inspect] start", flush=True)
    print(f"[phase:model-inspect] model_root={MODEL_ROOT}", flush=True)
    if not MODEL_ROOT.is_dir():
        raise RuntimeError(f"Missing model directory: {MODEL_ROOT}")
    if not MODEL_MANIFEST_PATH.is_file():
        raise RuntimeError(f"Missing manifest: {MODEL_MANIFEST_PATH}")

    manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"[phase:model-inspect] manifest_revision={manifest['model_revision']}", flush=True)
    print(f"[phase:model-inspect] manifest_files={manifest['file_count']}", flush=True)
    print(f"[phase:model-inspect] manifest_bytes={manifest['total_bytes']}", flush=True)

    all_files = sorted(path for path in MODEL_ROOT.rglob("*") if path.is_file())
    print(f"[phase:model-inspect] actual_files={len(all_files)}", flush=True)
    for path in all_files:
        relative = path.relative_to(MODEL_ROOT)
        print(f"[phase:model-inspect] file={relative} bytes={path.stat().st_size}", flush=True)
        if path.name.endswith(".json"):
            _print_json_summary(path)

    top_level = sorted(path.name for path in MODEL_ROOT.iterdir())
    print(f"[phase:model-inspect] top_level={top_level}", flush=True)
    print("[phase:model-inspect] SUCCESS", flush=True)


if __name__ == "__main__":
    with app.run():
        inspect_model.remote()
