"""Transparent starter metrics for listen-versus-backchannel behavior.

These energy-based metrics are screening signals, not a validated speech
recognizer or a final research reward. Calibrate them against a human-reviewed
sample before considering any training use.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from phase3.audio import read_pcm16_mono
from phase3.scenarios.manifest import Event, ScenarioPair


@dataclass(frozen=True)
class ScoreSettings:
    frame_ms: int = 20
    absolute_rms_threshold: float = 0.012
    noise_multiplier: float = 3.0
    min_speech_s: float = 0.08
    merge_gap_s: float = 0.12
    backchannel_max_s: float = 1.0
    timing_tolerance_s: float = 0.4


def speech_intervals(samples: np.ndarray, sample_rate_hz: int, settings: ScoreSettings) -> list[tuple[float, float]]:
    """Find coarse speech-like spans from short-frame RMS energy."""
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    frame_size = max(1, round(sample_rate_hz * settings.frame_ms / 1000))
    if audio.size == 0:
        return []
    frame_count = (audio.size + frame_size - 1) // frame_size
    rms = np.empty(frame_count, dtype=np.float32)
    for index in range(frame_count):
        frame = audio[index * frame_size : (index + 1) * frame_size]
        rms[index] = np.sqrt(np.mean(frame * frame)) if frame.size else 0.0
    quiet = rms[rms <= np.percentile(rms, 40)]
    noise_floor = float(np.percentile(quiet, 90)) if quiet.size else 0.0
    threshold = max(settings.absolute_rms_threshold, settings.noise_multiplier * noise_floor)
    active = rms >= threshold

    raw: list[tuple[float, float]] = []
    start_frame: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start_frame is None:
            start_frame = index
        elif not is_active and start_frame is not None:
            raw.append((start_frame * frame_size / sample_rate_hz, index * frame_size / sample_rate_hz))
            start_frame = None
    if start_frame is not None:
        raw.append((start_frame * frame_size / sample_rate_hz, audio.size / sample_rate_hz))

    merged: list[list[float]] = []
    for start_s, end_s in raw:
        if merged and start_s - merged[-1][1] <= settings.merge_gap_s:
            merged[-1][1] = end_s
        else:
            merged.append([start_s, end_s])
    return [
        (start_s, end_s)
        for start_s, end_s in merged
        if end_s - start_s >= settings.min_speech_s
    ]


def _event_intervals(events: tuple[Event, ...], kind: str) -> list[tuple[float, float]]:
    return [
        (float(event.start_s), float(event.end_s))
        for event in events
        if event.kind == kind and event.start_s is not None and event.end_s is not None
    ]


def _overlap_s(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _speech_inside_events(
    speech: list[tuple[float, float]], events: list[tuple[float, float]]
) -> tuple[float, int]:
    overlap = 0.0
    episodes = 0
    for span in speech:
        amount = sum(_overlap_s(span, event) for event in events)
        if amount > 0:
            overlap += amount
            episodes += 1
    return overlap, episodes


def _align_to_user_timeline(
    intervals: list[tuple[float, float]], timeline_path: str | Path | None
) -> list[tuple[float, float]]:
    """Map output-audio positions to approximate input replay positions."""
    if timeline_path is None:
        return intervals
    timeline = json.loads(Path(timeline_path).read_text(encoding="utf-8"))
    if not timeline:
        return intervals
    anchors = [
        (float(item["output_end_s"]), float(item["user_audio_time_s"]))
        for item in timeline
    ]

    def map_time(output_s: float) -> float:
        if output_s <= anchors[0][0]:
            return max(0.0, output_s + anchors[0][1] - anchors[0][0])
        for (out_left, user_left), (out_right, user_right) in zip(anchors, anchors[1:]):
            if output_s <= out_right:
                fraction = (output_s - out_left) / max(1e-9, out_right - out_left)
                return user_left + fraction * (user_right - user_left)
        out_last, user_last = anchors[-1]
        return max(0.0, output_s + user_last - out_last)

    return [(map_time(start), map_time(end)) for start, end in intervals]


def score_audio(
    pair: ScenarioPair,
    action: str,
    audio_path: str | Path,
    settings: ScoreSettings = ScoreSettings(),
    *,
    audio_timeline_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return component scores for one output. No single score hides its parts."""
    if action not in {"listen", "backchannel"}:
        raise ValueError(f"Unsupported P03 action: {action}")
    audio = read_pcm16_mono(audio_path)
    output_speech = speech_intervals(audio, 24000, settings)
    speech = _align_to_user_timeline(output_speech, audio_timeline_path)
    held_floor = _event_intervals(pair.events, "user_holds_floor")
    opportunities = _event_intervals(pair.events, "backchannel_opportunity")
    intrusion_s, intrusion_episodes = _speech_inside_events(speech, held_floor)
    held_floor_s = sum(end - start for start, end in held_floor)
    listen_intrusion_fraction = min(1.0, intrusion_s / held_floor_s) if held_floor_s else None
    listen_compliance = 1.0 - listen_intrusion_fraction if listen_intrusion_fraction is not None else None

    short_speech = [span for span in speech if span[1] - span[0] <= settings.backchannel_max_s]
    long_speech = [span for span in speech if span[1] - span[0] > settings.backchannel_max_s]
    matched_speech: set[int] = set()
    matched_opportunities: set[int] = set()
    expanded = [
        (start - settings.timing_tolerance_s, end + settings.timing_tolerance_s)
        for start, end in opportunities
    ]
    for event_index, event_span in enumerate(expanded):
        for speech_index, speech_span in enumerate(short_speech):
            if speech_index not in matched_speech and _overlap_s(speech_span, event_span) > 0:
                matched_speech.add(speech_index)
                matched_opportunities.add(event_index)
                break
    bc_recall = len(matched_opportunities) / len(opportunities) if opportunities else None
    bc_precision = len(matched_speech) / len(short_speech) if short_speech else (1.0 if not opportunities else 0.0)
    false_short_count = len(short_speech) - len(matched_speech)
    long_takeover_count = sum(
        1 for span in long_speech if any(_overlap_s(span, event) > 0 for event in opportunities)
    )

    if action == "listen":
        adherence = listen_compliance
    else:
        adherence = max(0.0, (bc_recall or 0.0) - (false_short_count / max(1, len(opportunities))) - long_takeover_count)

    return {
        "pair_id": pair.pair_id,
        "expected_action": action,
        "speech_interval_count": len(speech),
        "speech_intervals_s": [[round(start, 3), round(end, 3)] for start, end in speech],
        "audio_alignment": "approximate_stream_arrival_timeline" if audio_timeline_path else "output_audio_timeline_unaligned",
        "user_holds_floor_seconds": round(held_floor_s, 3),
        "listen_intrusion_seconds": round(intrusion_s, 3),
        "listen_intrusion_episodes": intrusion_episodes,
        "listen_intrusion_fraction": None if listen_intrusion_fraction is None else round(listen_intrusion_fraction, 4),
        "listen_compliance": None if listen_compliance is None else round(listen_compliance, 4),
        "backchannel_opportunities": len(opportunities),
        "backchannel_hits": len(matched_opportunities),
        "backchannel_recall": None if bc_recall is None else round(bc_recall, 4),
        "backchannel_precision": round(bc_precision, 4),
        "unmatched_short_responses": false_short_count,
        "long_takeovers_during_opportunities": long_takeover_count,
        "provisional_instruction_adherence": None if adherence is None else round(adherence, 4),
        "scorer": "energy_vad_v1",
        "warning": "Heuristic speech detector; calibrate against human review before training use.",
    }

