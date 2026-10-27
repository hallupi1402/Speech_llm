"""Send a 24 kHz mono PCM16 WAV to the DuplexOmni realtime bridge.

The bridge accepts raw PCM16 binary WebSocket messages and returns raw PCM16
audio messages plus JSON events such as ASR/TTS. Start the Modal realtime
bridge first, then run:

    python test_duplexomni_realtime.py \
        --url wss://<modal-endpoint>/ \
        --input question.wav \
        --output answer.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import websockets
from websockets.exceptions import ConnectionClosed


SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_MS = 480
CHUNK_BYTES = SAMPLE_RATE * CHUNK_MS // 1000 * SAMPLE_WIDTH


def read_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        if (
            wav_file.getframerate() != SAMPLE_RATE
            or wav_file.getnchannels() != CHANNELS
            or wav_file.getsampwidth() != SAMPLE_WIDTH
        ):
            raise ValueError(
                f"{path} must be mono/24kHz/PCM16; got "
                f"channels={wav_file.getnchannels()} "
                f"rate={wav_file.getframerate()} "
                f"width={wav_file.getsampwidth()}"
            )
        return wav_file.readframes(wav_file.getnframes())


async def receive_audio(
    websocket: websockets.ClientConnection,
    audio: bytearray,
) -> None:
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                audio.extend(message)
                continue
            try:
                event = json.loads(message)
            except json.JSONDecodeError:
                print(f"[server:text] {message}", flush=True)
            else:
                print(f"[server:event] {json.dumps(event, ensure_ascii=False)}", flush=True)
    except ConnectionClosed:
        pass


async def run(args: argparse.Namespace) -> None:
    input_pcm = read_wav(args.input)
    output_pcm = bytearray()
    print(
        f"[client] input={args.input} duration={len(input_pcm) / (SAMPLE_RATE * SAMPLE_WIDTH):.2f}s",
        flush=True,
    )

    async with websockets.connect(
        args.url,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ) as websocket:
        receiver = asyncio.create_task(receive_audio(websocket, output_pcm))
        started = time.monotonic()

        for offset in range(0, len(input_pcm), CHUNK_BYTES):
            chunk = input_pcm[offset : offset + CHUNK_BYTES]
            if len(chunk) < CHUNK_BYTES:
                chunk += b"\x00" * (CHUNK_BYTES - len(chunk))
            await websocket.send(chunk)
            await asyncio.sleep(CHUNK_MS / 1000)

        print(f"[client] input_sent_elapsed={time.monotonic() - started:.2f}s", flush=True)
        await asyncio.sleep(args.tail_seconds)
        receiver.cancel()
        try:
            await receiver
        except asyncio.CancelledError:
            pass

    with wave.open(str(args.output), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(bytes(output_pcm))

    print(
        f"[client] saved={args.output} "
        f"duration={len(output_pcm) / (SAMPLE_RATE * SAMPLE_WIDTH):.2f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="wss:// URL printed by modal serve")
    parser.add_argument("--input", type=Path, required=True, help="mono 24kHz PCM16 WAV")
    parser.add_argument("--output", type=Path, default=Path("duplexomni_answer.wav"))
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=15.0,
        help="time to wait for trailing Talker audio after input ends",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
