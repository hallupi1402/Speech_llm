"""Pin the DuplexOmni source tree into the active Modal Volume.

This is a CPU-only bootstrap step.  It intentionally does not build vLLM,
download weights, or allocate a GPU; the next job consumes this pinned tree.
"""

from __future__ import annotations

import subprocess

import modal

from modal_runtime import REPO_DIR, VOLUME_NAME, get_volume


APP_NAME = "duplexomni-source-bootstrap"
REPO_URL = "https://github.com/MuyeHuang/DuplexOmni.git"

app = modal.App(APP_NAME)
volume = get_volume(create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs")
    .add_local_python_source("modal_runtime")
)


def _run(command: list[str]) -> None:
    print(f"[phase:source] command={' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


@app.function(
    image=image,
    cpu=2,
    memory=8 * 1024,
    timeout=60 * 60,
    volumes={"/root/workspace": volume},
)
def prepare_source() -> None:
    print(f"[phase:source] target={REPO_DIR}", flush=True)
    if not REPO_DIR.exists():
        REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", REPO_URL, str(REPO_DIR)])
    else:
        print("[phase:source] repository already exists; preserving it", flush=True)

    revision = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "-C", str(REPO_DIR), "branch", "--show-current"], text=True
    ).strip() or "detached"
    print(f"[phase:source] source_revision={revision} branch={branch}", flush=True)
    volume.commit()
    print(f"[phase:source] SUCCESS committed volume={VOLUME_NAME}", flush=True)


if __name__ == "__main__":
    with app.run():
        prepare_source.remote()
