"""Small WAV helpers; model inference remains on the remote PersonaPlex server."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


SAMPLE_RATE_HZ = 24000


def read_pcm16_mono(path: str | Path, *, sample_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    """Read the pilot's required mono PCM16 WAV and return float32 samples in [-1, 1]."""
    wav_path = Path(path)
    try:
        with wave.open(str(wav_path), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise ValueError(f"Expected uncompressed PCM WAV: {wav_path}")
            if wav.getnchannels() != 1 or wav.getframerate() != sample_rate_hz or wav.getsampwidth() != 2:
                raise ValueError(f"Expected mono {sample_rate_hz} Hz PCM16 WAV: {wav_path}")
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()
    except wave.Error as exc:
        raise ValueError(f"Cannot read WAV {wav_path}: {exc}") from exc
    if pcm.size == 0:
        raise ValueError(f"WAV has no audio samples: {wav_path}")
    return pcm.astype(np.float32) / 32768.0


def write_pcm16_mono(path: str | Path, samples: np.ndarray, *, sample_rate_hz: int = SAMPLE_RATE_HZ) -> None:
    """Write float samples as a mono PCM16 WAV."""
    wav_path = Path(path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm.tobytes())

