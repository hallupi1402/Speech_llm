"""Replay a prepared WAV through the existing PersonaPlex WebSocket protocol."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit
import uuid

import numpy as np

from phase3.audio import read_pcm16_mono, write_pcm16_mono
from phase3.scenarios.manifest import Condition, ScenarioPair


@dataclass(frozen=True)
class RolloutSettings:
    ws_base_url: str
    output_root: Path
    model_repo: str
    upstream_commit: str
    chunk_ms: int = 20
    realtime_replay: bool = True
    quiet_timeout_s: float = 2.0
    max_extra_wait_s: float = 15.0
    handshake_timeout_s: float = 30.0
    sample_rate_hz: int = 24000


@dataclass
class RolloutResult:
    rollout_id: str
    pair_id: str
    condition_id: str
    expected_action: str
    status: str
    transcript: str
    input_duration_s: float
    output_duration_s: float
    audio_path: str
    audio_timeline_path: str
    transcript_path: str
    started_at_utc: str
    settings: dict[str, Any]
    error: str | None = None


def _session_url(base_url: str, *, text_prompt: str, voice_prompt: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError("WebSocket URL must start with ws:// or wss://")
    path = parsed.path.rstrip("/")
    if not path.endswith("/api/chat"):
        path = f"{path}/api/chat" if path else "/api/chat"
    query = urlencode({"text_prompt": text_prompt, "voice_prompt": voice_prompt})
    return urlunsplit((parsed.scheme, parsed.netloc, path, query, ""))


def _load_sphn():
    try:
        import sphn
    except ImportError as exc:
        raise RuntimeError(
            "The rollout client needs sphn. Install Phase 3 dependencies with `uv sync --project Phase_3`."
        ) from exc
    return sphn


async def run_scenario_condition(
    pair: ScenarioPair,
    condition: Condition,
    settings: RolloutSettings,
) -> RolloutResult:
    """Run one condition. The client only encodes/decodes audio; the model stays remote."""
    try:
        from aiohttp import ClientSession, ClientTimeout, WSMsgType
    except ImportError as exc:
        raise RuntimeError("Install Phase 3 dependencies with `uv sync --project Phase_3`.") from exc

    if pair.status != "ready":
        raise ValueError(f"Scenario pair {pair.pair_id!r} is draft; rollout is disabled")
    if settings.chunk_ms <= 0 or settings.sample_rate_hz != 24000:
        raise ValueError("Pilot replay requires positive chunks and a 24000 Hz sample rate")

    sphn = _load_sphn()
    user_audio = read_pcm16_mono(pair.audio_path, sample_rate_hz=settings.sample_rate_hz)
    input_duration_s = user_audio.size / settings.sample_rate_hz
    text_prompt = f"{pair.role_prompt.strip()}\nTurn-taking instruction: {condition.instruction.strip()}"
    url = _session_url(settings.ws_base_url, text_prompt=text_prompt, voice_prompt=pair.voice_prompt)

    rollout_id = f"{pair.pair_id}_{condition.id}_{uuid.uuid4().hex[:8]}"
    rollout_dir = settings.output_root / rollout_id
    rollout_dir.mkdir(parents=True, exist_ok=False)
    output_audio_path = rollout_dir / "assistant.wav"
    audio_timeline_path = rollout_dir / "audio_timeline.json"
    transcript_path = rollout_dir / "transcript.txt"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output_parts: list[np.ndarray] = []
    text_parts: list[str] = []
    received_any_output = False
    timed_out = False
    sent_audio_samples = 0
    output_sample_count = 0
    audio_timeline: list[dict[str, float]] = []

    timeout = ClientTimeout(total=None, sock_connect=30)
    origin = f"http://{urlsplit(settings.ws_base_url).netloc}"
    async with ClientSession(timeout=timeout, auto_decompress=False) as session:
        async with session.ws_connect(
            url,
            origin=origin,
            heartbeat=30,
            max_msg_size=0,
        ) as ws:
            handshake = await ws.receive(timeout=settings.handshake_timeout_s)
            if handshake.type != WSMsgType.BINARY or not handshake.data or handshake.data[0] != 0:
                raise RuntimeError(f"Expected PersonaPlex handshake, got {handshake.type}")

            capture_done = asyncio.Event()
            last_output_at = time.monotonic()
            capture_started_at = time.monotonic()
            opus_reader = sphn.OpusStreamReader(settings.sample_rate_hz)

            async def receive_output() -> None:
                nonlocal received_any_output, timed_out, last_output_at, output_sample_count
                while True:
                    elapsed = time.monotonic() - capture_started_at
                    if capture_done.is_set() and time.monotonic() - last_output_at >= settings.quiet_timeout_s:
                        return
                    if elapsed >= input_duration_s + settings.max_extra_wait_s:
                        timed_out = True
                        return
                    try:
                        message = await ws.receive(timeout=0.25)
                    except asyncio.TimeoutError:
                        continue
                    if message.type in {WSMsgType.CLOSED, WSMsgType.CLOSE, WSMsgType.ERROR}:
                        return
                    if message.type != WSMsgType.BINARY or not message.data:
                        continue
                    last_output_at = time.monotonic()
                    received_any_output = True
                    kind, payload = message.data[0], message.data[1:]
                    if kind == 1:
                        opus_reader.append_bytes(payload)
                        pcm = opus_reader.read_pcm()
                        if pcm.size:
                            pcm = np.asarray(pcm, dtype=np.float32).reshape(-1).copy()
                            output_parts.append(pcm)
                            output_sample_count += pcm.size
                            elapsed_s = time.monotonic() - capture_started_at
                            user_time_s = (
                                sent_audio_samples / settings.sample_rate_hz
                                if not settings.realtime_replay
                                else min(elapsed_s, input_duration_s)
                            )
                            audio_timeline.append(
                                {
                                    "output_end_s": output_sample_count / settings.sample_rate_hz,
                                    "user_audio_time_s": user_time_s,
                                }
                            )
                    elif kind == 2:
                        text_parts.append(payload.decode("utf-8", errors="replace"))

            receiver = asyncio.create_task(receive_output())
            frame_samples = max(1, round(settings.sample_rate_hz * settings.chunk_ms / 1000))
            opus_writer = sphn.OpusStreamWriter(settings.sample_rate_hz)
            try:
                for start in range(0, user_audio.size, frame_samples):
                    chunk = user_audio[start : start + frame_samples]
                    opus_writer.append_pcm(chunk)
                    packet = opus_writer.read_bytes()
                    sent_audio_samples += chunk.size
                    if packet:
                        await ws.send_bytes(b"\x01" + packet)
                    if settings.realtime_replay:
                        await asyncio.sleep(chunk.size / settings.sample_rate_hz)
                remaining = opus_writer.read_bytes()
                if remaining:
                    await ws.send_bytes(b"\x01" + remaining)
                capture_done.set()
                await receiver
            finally:
                capture_done.set()
                if not receiver.done():
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)

    output_audio = np.concatenate(output_parts) if output_parts else np.zeros(0, dtype=np.float32)
    transcript = "".join(text_parts).strip()
    if output_audio.size:
        write_pcm16_mono(output_audio_path, output_audio, sample_rate_hz=settings.sample_rate_hz)
        audio_timeline_path.write_text(
            json.dumps(audio_timeline, indent=2) + "\n", encoding="utf-8"
        )
    transcript_path.write_text(transcript + "\n", encoding="utf-8")
    result = RolloutResult(
        rollout_id=rollout_id,
        pair_id=pair.pair_id,
        condition_id=condition.id,
        expected_action=condition.expected_action,
        status="timed_out" if timed_out else ("completed" if received_any_output else "no_output"),
        transcript=transcript,
        input_duration_s=input_duration_s,
        output_duration_s=output_audio.size / settings.sample_rate_hz,
        audio_path=str(output_audio_path) if output_audio.size else "",
        audio_timeline_path=str(audio_timeline_path) if output_audio.size else "",
        transcript_path=str(transcript_path),
        started_at_utc=started_at,
        settings={
            "ws_base_url": settings.ws_base_url,
            "model_repo": settings.model_repo,
            "upstream_commit": settings.upstream_commit,
            "voice_prompt": pair.voice_prompt,
            "instruction": condition.instruction,
            "sample_rate_hz": settings.sample_rate_hz,
            "chunk_ms": settings.chunk_ms,
            "realtime_replay": settings.realtime_replay,
            "seed": None,
            "seed_note": "The current upstream WebSocket interface does not expose a reliable per-session seed.",
            "audio_timeline_note": "Output chunks are aligned approximately to replay time at arrival; --fast mode uses sent input samples instead of wall time.",
        },
    )
    (rollout_dir / "rollout.json").write_text(
        json.dumps(result.__dict__, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result

