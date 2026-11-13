"""Load and validate paired, instruction-conditioned P03 scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal
import wave


Action = Literal["listen", "backchannel"]
ScenarioStatus = Literal["draft", "ready"]
REQUIRED_ACTIONS = {"listen", "backchannel"}
REQUIRED_EVENT_KINDS = {"user_holds_floor", "backchannel_opportunity"}


@dataclass(frozen=True)
class Condition:
    id: str
    instruction: str
    expected_action: Action


@dataclass(frozen=True)
class Event:
    id: str
    kind: str
    start_s: float | None
    end_s: float | None


@dataclass(frozen=True)
class Audio:
    path: str
    sample_rate_hz: int
    channels: int
    source: str
    license: str


@dataclass(frozen=True)
class ScenarioPair:
    pair_id: str
    status: ScenarioStatus
    split: str
    note: str
    audio: Audio
    duration_s: float | None
    role_prompt: str
    voice_prompt: str
    events: tuple[Event, ...]
    conditions: tuple[Condition, Condition]
    manifest_path: Path

    @property
    def audio_path(self) -> Path:
        return (self.manifest_path.parent / self.audio.path).resolve()

    def condition(self, action: Action) -> Condition:
        for condition in self.conditions:
            if condition.expected_action == action:
                return condition
        raise KeyError(f"Pair {self.pair_id!r} has no {action!r} condition")


def _require_string(value: Any, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} must be a non-empty string")
    return value.strip()


def _parse_event(value: Any, context: str) -> Event:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: each event must be an object")
    event_id = _require_string(value.get("id"), "id", f"{context}.event")
    kind = _require_string(value.get("kind"), "kind", f"{context}.event")
    start = value.get("start_s")
    end = value.get("end_s")
    if (start is None) != (end is None):
        raise ValueError(f"{context}.{event_id}: start_s and end_s must both be set or both be null")
    if start is not None:
        if not isinstance(start, (float, int)) or not isinstance(end, (float, int)):
            raise ValueError(f"{context}.{event_id}: event times must be numbers")
        if start < 0 or end <= start:
            raise ValueError(f"{context}.{event_id}: require 0 <= start_s < end_s")
    return Event(event_id, kind, start, end)


def _parse_pair(value: Any, manifest_path: Path, line_number: int) -> ScenarioPair:
    context = f"{manifest_path}:{line_number}"
    if not isinstance(value, dict):
        raise ValueError(f"{context}: each JSONL line must be an object")
    if value.get("schema_version") != 1:
        raise ValueError(f"{context}: schema_version must be 1")

    status = value.get("status")
    if status not in {"draft", "ready"}:
        raise ValueError(f"{context}: status must be 'draft' or 'ready'")
    audio_data = value.get("audio")
    if not isinstance(audio_data, dict):
        raise ValueError(f"{context}: audio must be an object")
    audio = Audio(
        path=_require_string(audio_data.get("path"), "audio.path", context),
        sample_rate_hz=audio_data.get("sample_rate_hz"),
        channels=audio_data.get("channels"),
        source=_require_string(audio_data.get("source"), "audio.source", context),
        license=_require_string(audio_data.get("license"), "audio.license", context),
    )
    if audio.sample_rate_hz != 24000 or audio.channels != 1:
        raise ValueError(f"{context}: pilot audio must be mono at 24000 Hz")

    duration = value.get("duration_s")
    if duration is not None and (not isinstance(duration, (float, int)) or duration <= 0):
        raise ValueError(f"{context}: duration_s must be a positive number or null")

    events_value = value.get("events")
    if not isinstance(events_value, list):
        raise ValueError(f"{context}: events must be a list")
    events = tuple(_parse_event(event, context) for event in events_value)
    if len({event.id for event in events}) != len(events):
        raise ValueError(f"{context}: event ids must be unique within a pair")

    conditions_value = value.get("conditions")
    if not isinstance(conditions_value, list) or len(conditions_value) != 2:
        raise ValueError(f"{context}: conditions must contain exactly two entries")
    conditions: list[Condition] = []
    for index, item in enumerate(conditions_value):
        condition_context = f"{context}.conditions[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{condition_context}: each condition must be an object")
        action = item.get("expected_action")
        if action not in REQUIRED_ACTIONS:
            raise ValueError(f"{condition_context}: expected_action must be listen or backchannel")
        conditions.append(
            Condition(
                id=_require_string(item.get("id"), "id", condition_context),
                instruction=_require_string(item.get("instruction"), "instruction", condition_context),
                expected_action=action,
            )
        )
    if {condition.expected_action for condition in conditions} != REQUIRED_ACTIONS:
        raise ValueError(f"{context}: each pair must contain one listen and one backchannel condition")
    if conditions[0].instruction == conditions[1].instruction:
        raise ValueError(f"{context}: paired instructions must differ")
    if len({condition.id for condition in conditions}) != 2:
        raise ValueError(f"{context}: condition ids must be unique")

    pair = ScenarioPair(
        pair_id=_require_string(value.get("pair_id"), "pair_id", context),
        status=status,
        split=_require_string(value.get("split"), "split", context),
        note=_require_string(value.get("note"), "note", context),
        audio=audio,
        duration_s=duration,
        role_prompt=_require_string(value.get("role_prompt"), "role_prompt", context),
        voice_prompt=_require_string(value.get("voice_prompt"), "voice_prompt", context),
        events=events,
        conditions=(conditions[0], conditions[1]),
        manifest_path=manifest_path.resolve(),
    )
    _validate_ready_pair(pair, context)
    return pair


def _validate_ready_pair(pair: ScenarioPair, context: str) -> None:
    if pair.status != "ready":
        return
    if pair.duration_s is None:
        raise ValueError(f"{context}: ready pair must include duration_s")
    if pair.audio.source.lower() in {"pending", "todo", "unknown"}:
        raise ValueError(f"{context}: ready pair must record the audio source")
    if pair.audio.license.lower() in {"pending", "todo", "unknown"}:
        raise ValueError(f"{context}: ready pair must record the audio license/terms")
    kinds = {event.kind for event in pair.events}
    if not REQUIRED_EVENT_KINDS.issubset(kinds):
        raise ValueError(f"{context}: ready pair needs user_holds_floor and backchannel_opportunity events")
    if not pair.events:
        raise ValueError(f"{context}: ready pair must have annotated events")
    for event in pair.events:
        if event.start_s is None or event.end_s is None:
            raise ValueError(f"{context}.{event.id}: ready pair event times cannot be null")
        if event.end_s > pair.duration_s:
            raise ValueError(f"{context}.{event.id}: event extends beyond duration_s")
    if not pair.audio_path.is_file():
        raise ValueError(f"{context}: audio file does not exist: {pair.audio_path}")
    _validate_wav(pair.audio_path, pair.audio.sample_rate_hz, pair.audio.channels, context)


def _validate_wav(path: Path, sample_rate_hz: int, channels: int, context: str) -> None:
    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise ValueError(f"{context}: {path} must be uncompressed PCM WAV")
            if wav.getframerate() != sample_rate_hz or wav.getnchannels() != channels or wav.getsampwidth() != 2:
                raise ValueError(f"{context}: {path} must be mono 24000 Hz PCM16 WAV")
    except wave.Error as exc:
        raise ValueError(f"{context}: cannot read WAV {path}: {exc}") from exc


def load_manifest(path: str | Path) -> list[ScenarioPair]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    pairs: list[ScenarioPair] = []
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_number}: invalid JSON: {exc.msg}") from exc
            pairs.append(_parse_pair(value, manifest_path, line_number))
    return pairs


def validate_manifest(path: str | Path, *, require_ready: bool = False) -> list[str]:
    try:
        pairs = load_manifest(path)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    errors: list[str] = []
    if not pairs:
        return [f"{path}: manifest has no scenarios"]
    pair_ids = [pair.pair_id for pair in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        errors.append("pair_id values must be unique")
    if require_ready:
        for pair in pairs:
            if pair.status != "ready":
                errors.append(f"{pair.pair_id}: status is draft; prepare audio and annotations first")
    return errors

