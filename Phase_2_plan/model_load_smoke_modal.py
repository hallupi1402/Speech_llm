"""Official 8-GPU DuplexOmni model-load smoke test on one Modal node.

This job validates only startup: Thinker, Talker/MTP, and Orchestrator must
reach their health endpoints.  It never opens a microphone or websocket client
and always stops the services before the function exits.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import modal

from modal_runtime import (
    GPU_SPEC,
    JOB_TIMEOUT_SECONDS,
    MEMORY_MIB,
    MODEL_ROOT,
    SERVING_DIR,
    STARTUP_TIMEOUT_SECONDS,
    THINKER_PORT,
    TALKER_PORT,
    ORCHESTRATOR_PORT,
    VLLM_DIR,
    get_volume,
    health,
    official_launcher_env,
    print_nvidia_smi,
    runtime_image,
)


APP_NAME = "duplexomni-model-load-smoke-8gpu"
LOG_DIR = SERVING_DIR / "logs_model_load_8gpu"
START_SCRIPT = SERVING_DIR / "start_thinker_talker_stack.sh"
STOP_SCRIPT = SERVING_DIR / "stop_thinker_talker_stack.sh"

app = modal.App(APP_NAME)
volume = get_volume()
image = runtime_image()


def _read_new_lines(path: Path, offsets: dict[Path, int]) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        handle.seek(offsets.get(path, 0))
        for line in handle.readlines():
            print(f"[service:{path.stem}] {line.rstrip()}", flush=True)
        offsets[path] = handle.tell()


def _drain_logs(offsets: dict[Path, int]) -> None:
    for name in ("thinker", "talker", "orchestrator"):
        _read_new_lines(LOG_DIR / f"{name}.log", offsets)


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=32,
    memory=MEMORY_MIB,
    timeout=JOB_TIMEOUT_SECONDS,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={"/root/workspace": volume},
)
def run_model_load_smoke() -> None:
    print("[phase:model-load] start", flush=True)
    print(f"[phase:model-load] gpu_spec={GPU_SPEC}", flush=True)
    print("[phase:model-load] layout=thinker:0-3 tp=4; talker:4-7 tp=4; mtp:7", flush=True)
    print(f"[phase:model-load] model={MODEL_ROOT}", flush=True)
    print(f"[phase:model-load] vllm_root={VLLM_DIR}", flush=True)

    if not MODEL_ROOT.is_dir():
        raise RuntimeError(f"Missing checkpoint: {MODEL_ROOT}")
    if not START_SCRIPT.is_file() or not STOP_SCRIPT.is_file():
        raise RuntimeError("Missing official Thinker/Talker launcher scripts")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    offsets: dict[Path, int] = {}
    env = official_launcher_env()
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
    if launcher.stdout:
        print(launcher.stdout.rstrip(), flush=True)
    if launcher.returncode != 0:
        if launcher.stderr:
            print(launcher.stderr.rstrip(), flush=True)
        raise RuntimeError("Official stack launcher failed")

    try:
        stable_ready_since: float | None = None
        last_health_report = 0.0
        while time.monotonic() - started < 3.5 * 60 * 60:
            _drain_logs(offsets)
            statuses = {
                name: health(port)
                for name, port in (
                    ("thinker", THINKER_PORT),
                    ("talker", TALKER_PORT),
                    ("orchestrator", ORCHESTRATOR_PORT),
                )
            }
            now = time.monotonic()
            if now - last_health_report >= 30:
                print(
                    "[phase:model-load] health "
                    + " ".join(
                        f"{name}={'UP' if ok else 'DOWN'}({detail})"
                        for name, (ok, detail) in statuses.items()
                    ),
                    flush=True,
                )
                last_health_report = now

            if all(ok for ok, _ in statuses.values()):
                stable_ready_since = stable_ready_since or now
                if now - stable_ready_since >= 30:
                    _drain_logs(offsets)
                    print(f"[phase:model-load] ready_elapsed_seconds={now - started:.1f}", flush=True)
                    print_nvidia_smi("phase:model-load")
                    print("[phase:model-load] SUCCESS", flush=True)
                    return
            else:
                stable_ready_since = None
            time.sleep(10)
        raise TimeoutError("Timed out waiting for the official 8-GPU stack")
    finally:
        _drain_logs(offsets)
        stop = subprocess.run(
            ["bash", str(STOP_SCRIPT)],
            cwd=SERVING_DIR,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        print(f"[phase:model-load] stop_exit={stop.returncode}", flush=True)
        if stop.stdout:
            print(stop.stdout.rstrip(), flush=True)
        if stop.stderr:
            print(stop.stderr.rstrip(), flush=True)


if __name__ == "__main__":
    with app.run():
        run_model_load_smoke.remote()
