"""Persistent 8-GPU Modal WebSocket endpoint for DuplexOmni realtime tests.

This file is only the realtime adapter.  Paths, image, Volume, environment,
and GPU sizing live in :mod:`modal_runtime`; the local microphone client is a
separate process and is never started by this server.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import modal

from modal_runtime import (
    GPU_SPEC,
    MEMORY_MIB,
    MODEL_ROOT,
    REALTIME_PORT,
    SERVING_DIR,
    STARTUP_TIMEOUT_SECONDS,
    THINKER_PORT,
    TALKER_PORT,
    ORCHESTRATOR_PORT,
    CORE_DIR,
    get_volume,
    health,
    inference_env,
    runtime_image,
)


APP_NAME = "duplexomni-realtime-8gpu"
LOG_DIR = SERVING_DIR / "logs_realtime_8gpu"

app = modal.App(APP_NAME)
volume = get_volume()
image = runtime_image()


def _start(
    name: str,
    script: Path,
    log: Path,
    env: dict[str, str],
    args: list[str],
) -> subprocess.Popen[str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    print(f"[phase:realtime] start_service={name} log={log}", flush=True)
    return subprocess.Popen(
        [env.get("PYTHON_BIN", "python"), str(script), *args],
        cwd=CORE_DIR,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _terminate(processes: list[subprocess.Popen[str]]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    for process in reversed(processes):
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


@app.function(
    image=image,
    gpu=GPU_SPEC,
    cpu=32,
    memory=MEMORY_MIB,
    timeout=24 * 60 * 60,
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
    volumes={"/root/workspace": volume},
)
@modal.concurrent(max_inputs=4)
@modal.web_server(REALTIME_PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS)
def serve_realtime() -> None:
    print("[phase:realtime] start", flush=True)
    print(f"[phase:realtime] gpu_spec={GPU_SPEC}", flush=True)
    print("[phase:realtime] layout=thinker:0-3 tp=4; talker:4-7 tp=4; mtp:7", flush=True)
    if not MODEL_ROOT.is_dir():
        raise RuntimeError(f"Missing checkpoint: {MODEL_ROOT}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in LOG_DIR.glob("*.log"):
        path.unlink()

    base_env = inference_env()
    base_env["PYTHON_BIN"] = base_env.get("PYTHON_BIN", "python")
    thinker_env = {
        **base_env,
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "THINKER_PORT": str(THINKER_PORT),
        "THINKER_TP": "4",
        "THINKER_GPU_MEM_UTIL": "0.8",
    }
    talker_env = {
        **base_env,
        "CUDA_VISIBLE_DEVICES": "4,5,6,7",
        "TALKER_PORT": str(TALKER_PORT),
        "TALKER_TP": "4",
        # The Talker subprocess sees local device 3 as physical GPU 7.
        "TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES": "3",
        "TALKER_MTP_SUBPROC_TP": "1",
        "TALKER_MTP_GPU_MEM_UTIL": "0.12",
        "TALKER_MTP_PROFILE": "0",
    }
    orch_env = {
        **base_env,
        "ORCH_PORT": str(ORCHESTRATOR_PORT),
        "THINKER_INTERNAL_URL": f"http://127.0.0.1:{THINKER_PORT}/internal/chat_turn",
        "TALKER_INTERNAL_URL_BASE": f"http://127.0.0.1:{TALKER_PORT}/internal/talker/turn",
        "TALKER_DELETE_URL_BASE": f"http://127.0.0.1:{TALKER_PORT}/v1/talker/session",
    }
    bridge_env = {
        **base_env,
        "S1_OMNI_BASE_URL": f"http://127.0.0.1:{ORCHESTRATOR_PORT}/v1",
        "S1_AUDIO_WS_BASE": f"ws://127.0.0.1:{ORCHESTRATOR_PORT}/v1/audio/stream",
        "S1_CLEAR_SESSION_BASE": f"http://127.0.0.1:{ORCHESTRATOR_PORT}/v1/session",
        "S1_HEALTH_URL": f"http://127.0.0.1:{ORCHESTRATOR_PORT}/health",
    }

    processes: list[subprocess.Popen[str]] = []
    try:
        processes.extend(
            [
                _start("thinker", CORE_DIR / "server_thinker.py", LOG_DIR / "thinker.log", thinker_env, []),
                _start("talker", CORE_DIR / "server_talker.py", LOG_DIR / "talker.log", talker_env, ["--tp", "4"]),
                _start("orchestrator", CORE_DIR / "server_orchestrator.py", LOG_DIR / "orchestrator.log", orch_env, []),
            ]
        )

        started = time.monotonic()
        stable_since: float | None = None
        while time.monotonic() - started < 3.5 * 60 * 60:
            statuses = {
                name: health(port)
                for name, port in (
                    ("thinker", THINKER_PORT),
                    ("talker", TALKER_PORT),
                    ("orchestrator", ORCHESTRATOR_PORT),
                )
            }
            print(
                "[phase:realtime] health "
                + " ".join(f"{name}={'UP' if ok else 'DOWN'}({detail})" for name, (ok, detail) in statuses.items()),
                flush=True,
            )
            if all(ok for ok, _ in statuses.values()):
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= 30:
                    break
            else:
                stable_since = None
            for name, process in zip(("thinker", "talker", "orchestrator"), processes):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited with code {process.returncode}")
            time.sleep(10)
        else:
            raise TimeoutError("Timed out waiting for Thinker/Talker/Orchestrator")

        bridge = _start(
            "realtime-bridge",
            SERVING_DIR / "omni_realtime_server.py",
            LOG_DIR / "bridge.log",
            bridge_env,
            ["--host", "0.0.0.0", "--port", str(REALTIME_PORT)],
        )
        processes.append(bridge)
        time.sleep(2)
        if bridge.poll() is not None:
            raise RuntimeError(f"realtime bridge exited with code {bridge.returncode}")
        print(f"[phase:realtime] READY websocket_port={REALTIME_PORT}", flush=True)
        # Keep the web server process alive; the client is run locally by the user.
        while True:
            time.sleep(60)
    finally:
        _terminate(processes)


if __name__ == "__main__":
    with app.run():
        serve_realtime.remote()
