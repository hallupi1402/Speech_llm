"""Talk to a running DuplexOmni realtime bridge with the local microphone.

The Modal bridge accepts mono 24 kHz PCM16 binary WebSocket messages and
returns PCM16 audio messages plus JSON ASR/TTS events.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import sounddevice as sd
import websockets
from websockets.exceptions import ConnectionClosed


SAMPLE_RATE = 24_000
CHANNELS = 1
DTYPE = "int16"
CHUNK_MS = 480
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000


async def send_microphone(websocket: websockets.ClientConnection) -> None:
    print("[mic] đang nghe... nhấn Ctrl+C để dừng", flush=True)
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SAMPLES,
        channels=CHANNELS,
        dtype=DTYPE,
    ) as microphone:
        while True:
            data, overflowed = await asyncio.to_thread(microphone.read, CHUNK_SAMPLES)
            if overflowed:
                print("[mic] warning: input overflow", flush=True)
            await websocket.send(bytes(data))


async def play_response(websocket: websockets.ClientConnection) -> None:
    with sd.RawOutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
    ) as speaker:
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await asyncio.to_thread(speaker.write, message)
                    continue
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    print(f"[server] {message}", flush=True)
                else:
                    event_type = event.get("type", "event")
                    text = event.get("text") or event.get("status") or event
                    print(f"[{event_type}] {text}", flush=True)
        except ConnectionClosed:
            print("[server] WebSocket đã đóng", flush=True)


async def run(url: str) -> None:
    async with websockets.connect(
        url,
        max_size=10 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=60,
    ) as websocket:
        sender = asyncio.create_task(send_microphone(websocket))
        player = asyncio.create_task(play_response(websocket))
        try:
            await asyncio.gather(sender, player)
        finally:
            sender.cancel()
            player.cancel()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="wss:// URL printed by modal serve")
    args = parser.parse_args()
    try:
        asyncio.run(run(args.url))
    except KeyboardInterrupt:
        print("\n[client] stopped", flush=True)


if __name__ == "__main__":
    main()
