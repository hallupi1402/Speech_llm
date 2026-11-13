"""Command line entry points for the P03 pilot."""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any
import uuid

from phase3.rollout.websocket import RolloutSettings, run_scenario_condition
from phase3.scenarios.manifest import ScenarioPair, load_manifest, validate_manifest
from phase3.scoring.metrics import ScoreSettings, score_audio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "p03_pilot.json"


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = _project_path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {config_path}")
    return config


def _scenario_pairs(manifest_path: str | Path, *, require_ready: bool = False) -> list[ScenarioPair]:
    errors = validate_manifest(manifest_path, require_ready=require_ready)
    if errors:
        raise ValueError("Manifest validation failed:\n- " + "\n- ".join(errors))
    return load_manifest(manifest_path)


def command_validate(args: argparse.Namespace) -> int:
    manifest_path = _project_path(args.manifest)
    errors = validate_manifest(manifest_path, require_ready=args.require_ready)
    if errors:
        print("Manifest is not valid:")
        for error in errors:
            print(f"- {error}")
        return 1
    pairs = load_manifest(manifest_path)
    draft_count = sum(pair.status == "draft" for pair in pairs)
    print(f"Valid JSONL: {len(pairs)} scenario pairs ({draft_count} draft).")
    return 0


async def _rollout(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    manifest_path = _project_path(args.manifest or config["manifest"])
    pairs = _scenario_pairs(manifest_path, require_ready=True)
    if args.pair_id:
        pairs = [pair for pair in pairs if pair.pair_id == args.pair_id]
        if not pairs:
            raise ValueError(f"No scenario pair with id {args.pair_id!r}")
    repeats = args.repeats or int(config.get("repeats", 2))
    if repeats < 1:
        raise ValueError("--repeats must be at least 1")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    run_root = _project_path(config["run_root"]) / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    rollout_settings = RolloutSettings(
        ws_base_url=config["ws_base_url"],
        output_root=run_root / "rollouts",
        model_repo=config["model_repo"],
        upstream_commit=config["upstream_commit"],
        chunk_ms=int(config.get("chunk_ms", 20)),
        realtime_replay=not args.fast,
        quiet_timeout_s=float(config.get("quiet_timeout_s", 2.0)),
        max_extra_wait_s=float(config.get("max_extra_wait_s", 15.0)),
        handshake_timeout_s=float(config.get("handshake_timeout_s", 30.0)),
    )
    rollout_settings.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for pair in pairs:
        for condition in pair.conditions:
            for repeat in range(1, repeats + 1):
                print(f"[{pair.pair_id} / {condition.id} / repeat {repeat}/{repeats}]", flush=True)
                try:
                    result = await run_scenario_condition(pair, condition, rollout_settings)
                    record = result.__dict__
                    record["repeat"] = repeat
                    (run_root / "rollouts" / result.rollout_id / "rollout.json").write_text(
                        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
                except Exception as exc:  # Keep a failed case visible in the run manifest.
                    record = {
                        "pair_id": pair.pair_id,
                        "condition_id": condition.id,
                        "expected_action": condition.expected_action,
                        "repeat": repeat,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                records.append(record)
                (run_root / "run.json").write_text(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "manifest": str(manifest_path),
                            "model_repo": config["model_repo"],
                            "upstream_commit": config["upstream_commit"],
                            "repeats": repeats,
                            "rollouts": records,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ) + "\n",
                    encoding="utf-8",
                )
    failed = sum(item.get("status") not in {"completed", "no_output", "timed_out"} for item in records)
    print(f"Saved run to {run_root} ({failed} errors).")
    return 1 if failed else 0


def command_rollout(args: argparse.Namespace) -> int:
    return asyncio.run(_rollout(args))


def command_score(args: argparse.Namespace) -> int:
    pairs = {pair.pair_id: pair for pair in _scenario_pairs(_project_path(args.manifest), require_ready=True)}
    run_dir = Path(args.run_dir).expanduser().resolve()
    score_settings = ScoreSettings()
    results: list[dict[str, Any]] = []
    for rollout_file in sorted(run_dir.glob("rollouts/*/rollout.json")):
        item = json.loads(rollout_file.read_text(encoding="utf-8"))
        if item.get("status") not in {"completed", "no_output", "timed_out"}:
            continue
        audio_path = item.get("audio_path")
        if not audio_path or not Path(audio_path).is_file():
            scored: dict[str, Any] = {
                "pair_id": item["pair_id"],
                "condition_id": item["condition_id"],
                "rollout_id": item["rollout_id"],
                "expected_action": item["expected_action"],
                "repeat": item.get("repeat"),
                "status": item["status"],
                "provisional_instruction_adherence": None,
                "warning": "No generated audio to score.",
            }
        else:
            pair = pairs[item["pair_id"]]
            scored = score_audio(
                pair,
                item["expected_action"],
                audio_path,
                score_settings,
                audio_timeline_path=item.get("audio_timeline_path") or None,
            )
            scored.update(
                {
                    "condition_id": item["condition_id"],
                    "rollout_id": item["rollout_id"],
                    "repeat": item.get("repeat"),
                    "status": item["status"],
                }
            )
        results.append(scored)
    out_path = run_dir / "scores.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    paired = _paired_instruction_summary(results)
    paired_path = run_dir / "paired_instruction_sensitivity.json"
    paired_path.write_text(json.dumps(paired, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Scored {len(results)} rollouts; results: {out_path}")
    print(f"Paired instruction summary: {paired_path}")
    return 0


def _paired_instruction_summary(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare matched repeats for opposite expected actions on the same audio."""
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for score in scores:
        repeat = score.get("repeat")
        if repeat is None:
            continue
        groups.setdefault((score["pair_id"], int(repeat)), {})[score["expected_action"]] = score

    matched: list[dict[str, Any]] = []
    for (pair_id, repeat), actions in sorted(groups.items()):
        listen = actions.get("listen")
        backchannel = actions.get("backchannel")
        if listen is None or backchannel is None:
            continue
        listen_metric = listen.get("listen_intrusion_fraction")
        bc_hits = int(backchannel.get("backchannel_hits", 0))
        bc_opportunities = int(backchannel.get("backchannel_opportunities", 0))
        long_takeovers = int(backchannel.get("long_takeovers_during_opportunities", 0))
        matched.append(
            {
                "pair_id": pair_id,
                "repeat": repeat,
                "listen_no_floor_intrusion": listen_metric is not None and listen_metric == 0,
                "backchannel_hit": bc_opportunities > 0 and bc_hits > 0,
                "backchannel_no_long_takeover": long_takeovers == 0,
                "provisional_instruction_sensitive": (
                    listen_metric is not None
                    and listen_metric == 0
                    and bc_opportunities > 0
                    and bc_hits > 0
                    and long_takeovers == 0
                ),
            }
        )
    return {
        "matched_pair_repeats": len(matched),
        "provisional_instruction_sensitive_rate": (
            sum(row["provisional_instruction_sensitive"] for row in matched) / len(matched)
            if matched else None
        ),
        "rows": matched,
        "warning": (
            "This coarse audio-timing contrast does not establish semantic instruction following. "
            "Review paired audio manually; calibrate the VAD and annotations first."
        ),
    }


def command_review_export(args: argparse.Namespace) -> int:
    scores_path = Path(args.scores).expanduser().resolve()
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rollout_id", "pair_id", "condition_id", "repeat", "expected_action",
        "human_action_correct", "human_notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for score in scores:
            writer.writerow({key: score.get(key, "") for key in columns})
    print(f"Review sheet created: {output_path}")
    print("Fill human_action_correct with 1, 0, or unsure; review the audio while blinded to the heuristic score.")
    return 0


def command_gate(args: argparse.Namespace) -> int:
    scores = json.loads(Path(args.scores).expanduser().resolve().read_text(encoding="utf-8"))
    with Path(args.review).expanduser().resolve().open("r", encoding="utf-8", newline="") as stream:
        reviews = list(csv.DictReader(stream))
    indexed_scores = {item.get("rollout_id"): item for item in scores if item.get("rollout_id")}
    reviewed: list[tuple[float, float]] = []
    for row in reviews:
        score = indexed_scores.get(row.get("rollout_id"))
        label = (row.get("human_action_correct") or "").strip().lower()
        if score is None or label not in {"0", "1"}:
            continue
        auto = score.get("provisional_instruction_adherence")
        if isinstance(auto, (int, float)):
            reviewed.append((float(auto >= 0.5), float(label)))
    agreement = sum(auto == human for auto, human in reviewed) / len(reviewed) if reviewed else None

    grouped: dict[tuple[str, str], list[float]] = {}
    for item in scores:
        value = item.get("provisional_instruction_adherence")
        if isinstance(value, (int, float)):
            grouped.setdefault((item["pair_id"], item["expected_action"]), []).append(float(value))
    informative_groups = [values for values in grouped.values() if len(values) > 1]
    variable_groups = [values for values in informative_groups if statistics.pvariance(values) > 1e-6]
    report = {
        "reviewed_rollouts": len(reviewed),
        "human_auto_binary_agreement": agreement,
        "groups_with_multiple_rollouts": len(informative_groups),
        "groups_with_reward_variance": len(variable_groups),
        "minimum_review_count": args.min_reviews,
        "minimum_agreement": args.min_agreement,
        "status": "go_to_manual_research_review" if (
            len(reviewed) >= args.min_reviews
            and agreement is not None
            and agreement >= args.min_agreement
            and variable_groups
        ) else "no_go_for_training",
        "limitations": [
            "This gate only checks coarse energy-based scores against a small manual audit.",
            "A go result is not approval to train; validate task labels, semantic behavior, and reward ranking first.",
            "The current upstream WebSocket does not expose sampled token log-probabilities, so no GRPO trainer is included.",
        ],
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "go_to_manual_research_review" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phase3-p03", description="P03 baseline rollout and screening tools")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a JSONL scenario manifest")
    validate.add_argument("--manifest", default="data/p03/scenarios.jsonl")
    validate.add_argument("--require-ready", action="store_true")
    validate.set_defaults(handler=command_validate)

    rollout = commands.add_parser("rollout", help="replay ready scenarios through the Modal WebSocket proxy")
    rollout.add_argument("--config", default=str(DEFAULT_CONFIG))
    rollout.add_argument("--manifest")
    rollout.add_argument("--pair-id")
    rollout.add_argument("--repeats", type=int)
    rollout.add_argument("--fast", action="store_true", help="send audio without real-time pacing")
    rollout.set_defaults(handler=command_rollout)

    score = commands.add_parser("score", help="apply provisional energy-based metrics to a rollout folder")
    score.add_argument("--manifest", default="data/p03/scenarios.jsonl")
    score.add_argument("--run-dir", required=True)
    score.set_defaults(handler=command_score)

    review = commands.add_parser("review-export", help="create a blinded human review CSV")
    review.add_argument("--scores", required=True)
    review.add_argument("--output", required=True)
    review.set_defaults(handler=command_review_export)

    gate = commands.add_parser("gate", help="summarize verifier agreement and rollout reward variation")
    gate.add_argument("--scores", required=True)
    gate.add_argument("--review", required=True)
    gate.add_argument("--output", required=True)
    gate.add_argument("--min-reviews", type=int, default=10)
    gate.add_argument("--min-agreement", type=float, default=0.8)
    gate.set_defaults(handler=command_gate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        code = args.handler(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)


if __name__ == "__main__":
    main()
