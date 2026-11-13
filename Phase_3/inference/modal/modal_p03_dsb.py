"""Run the selected DSB-IFEval P03 cases with PersonaPlex on Modal.

    .venv/bin/modal run Phase_3/inference/modal/modal_p03_dsb.py --smoke
    .venv/bin/modal run Phase_3/inference/modal/modal_p03_dsb.py --repeats 4

Results are stored in the p03-dsb-results Modal Volume. The benchmark adapter
keeps model and user audio on the same frame clock (24 kHz stereo recordings).
"""

import json
import os
from pathlib import Path
from uuid import uuid4

import modal

from modal_personaplex import HF_SECRET_NAME, MODEL_CACHE_PATH, MODEL_REPO, image, model_cache


LOCAL_SOURCE = Path(__file__).resolve().parent != Path("/root")
HERE = Path(__file__).resolve().parents[2] if LOCAL_SOURCE else Path("/opt")
SUBSET = HERE / "data" / "p03" / "dsb_pilot_subset" if LOCAL_SOURCE else Path("/opt/p03_subset")
DSB_PACKAGE = HERE / "upstream" / "dsb-ifeval" / "dsb_ifeval" if LOCAL_SOURCE else Path("/opt/dsb_ifeval")
MODAL_SOURCE = HERE / "inference" / "modal" / "modal_personaplex.py" if LOCAL_SOURCE else Path("/root/modal_personaplex.py")
RESULTS_PATH = "/results"
results = modal.Volume.from_name("p03-dsb-results", create_if_missing=True)
app = modal.App("personaplex-p03-dsb")
pilot_image = (
    image.pip_install(
        "torchaudio==2.4.1", extra_options="--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install("soundfile>=0.12,<0.14", "silero-vad>=5.1,<6")
    .add_local_file(str(MODAL_SOURCE), remote_path="/root/modal_personaplex.py")
    .add_local_dir(str(DSB_PACKAGE), remote_path="/opt/dsb_ifeval")
    .add_local_dir(str(SUBSET), remote_path="/opt/p03_subset")
)


@app.function(
    image=pilot_image,
    gpu="A100-80GB",
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={MODEL_CACHE_PATH: model_cache, RESULTS_PATH: results},
    timeout=4 * 60 * 60,
    max_containers=1,
)
def run_pilot(run_id: str, repeats: int, smoke: bool) -> dict:
    import sys
    import tarfile

    os.environ["HF_HOME"] = MODEL_CACHE_PATH
    sys.path.insert(0, "/opt")
    import numpy as np
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from dsb_ifeval.adapters.reference.personaplex import PersonaPlex, build_input
    from dsb_ifeval.scoring.deterministic import score_case, utterances, vad_segments

    snapshot = snapshot_download(MODEL_REPO, local_files_only=True)
    if not (Path(snapshot) / "voices").is_dir():
        with tarfile.open(Path(snapshot) / "voices.tgz", "r:gz") as archive:
            archive.extractall(snapshot, filter="data")
    pairs = [json.loads(line) for line in open("/opt/p03_subset/manifest.jsonl")]
    if smoke:
        pairs = [next(pair for pair in pairs if pair["action"] == action)
                 for action in ("LISTEN", "BACKCHANNEL", "NO_BACKCHANNEL")]
    out = Path(RESULTS_PATH) / run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "run_config.json").write_text(json.dumps({
        "model": MODEL_REPO, "subset": "DSB-IFEval P03 pilot", "voice": "NATF0",
        "repeats": repeats, "smoke": smoke, "adapter": "DSB reference PersonaPlex offline",
        "gap_s": 3.0, "trail_s": 5.0,
    }, indent=2))
    results.commit()

    engine = PersonaPlex(weights_dir=snapshot)
    rows = []
    for repeat in range(1, repeats + 1):
        rec_dir = out / f"repeat-{repeat}"
        rec_dir.mkdir()
        for pair in pairs:
            for condition in pair["cases"]:
                case = {**pair, **condition}
                case["conversation_id"] = pair["pair_id"]
                case["expected_action"] = pair["action"]
                case["audio"] = {**pair["audio"], "dir": "/opt/p03_subset/" + pair["audio"]["dir"]}
                user, turn2_onset = build_input(case)
                model, transcript = engine.run(user, case["system_instruction"], "NATF0")
                # The offline adapter generates one model frame per input frame.
                if len(model) != len(user):
                    raise RuntimeError(f"Frame count differs for {case['test_case_id']}: {len(model)} != {len(user)}")
                stereo = np.column_stack((user, model)).astype(np.float32)
                wav_path = rec_dir / f"{case['test_case_id']}.stereo.wav"
                sf.write(wav_path, stereo, 24000, subtype="FLOAT")
                window = {key: round(value + turn2_onset, 3)
                          for key, value in case["trigger_window"].items()}
                record = {
                    "test_case_id": case["test_case_id"], "conversation_id": case["conversation_id"],
                    "level": case["level"], "role_id": case["role_id"], "probe_id": case["probe_id"],
                    "expected_action": case["expected_action"], "verifier": case["verifier"],
                    "model": MODEL_REPO, "voice": "NATF0", "stereo_wav": str(wav_path),
                    "turn2_onset_s": round(turn2_onset, 3), "trigger_window_rec": window,
                    "transcript": transcript, "model_speech_onset_s": None, "repeat": repeat,
                }
                (rec_dir / f"{case['test_case_id']}.rec.json").write_text(json.dumps(record, indent=2))
                results.commit()
                user_spans = utterances(vad_segments(user, 24000))
                model_spans = utterances(vad_segments(model, 24000))
                score = score_case(user_spans, model_spans, case,
                                   (window["start_s"], window["end_s"]), transcript=transcript)
                row = {"test_case_id": case["test_case_id"], "pair_id": pair["pair_id"],
                       "action": pair["action"], "level": case["level"], "repeat": repeat,
                       "verdict": score["verdict"], "measure": score["measure"]}
                rows.append(row)
                with (out / "scores.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                results.commit()
                print(f"{len(rows)}/{len(pairs) * 2 * repeats} {case['test_case_id']} "
                      f"repeat={repeat} IAS={score['verdict']}", flush=True)
    return {"run_id": run_id, "recordings": len(rows), "passed": sum(r["verdict"] for r in rows)}


@app.local_entrypoint()
def main(repeats: int = 1, smoke: bool = False):
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    run_id = "p03-" + uuid4().hex[:12]
    summary = run_pilot.remote(run_id, repeats, smoke)
    print(json.dumps(summary, indent=2))
    print(f"Download: .venv/bin/modal volume get p03-dsb-results /{run_id} Phase_3/runs/{run_id}")
