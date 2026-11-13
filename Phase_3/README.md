# PersonaPlex Phase 3

Phase 3 is the research workspace for evaluating behavior changes in
PersonaPlex. Model inference stays on Modal; the local Phase 3 code prepares
scenario audio, sends it through the authenticated WebSocket proxy, captures
outputs, and computes screening metrics.

## P03 pilot

The selected DSB-IFEval pilot is in
[`data/p03/dsb_pilot_subset/`](data/p03/dsb_pilot_subset/). It has 10 situations,
each with identical audio under an explicit L1 instruction and a persona-only
L2 instruction. The Modal runner and result commands are in
[`inference/modal/README.md`](inference/modal/README.md#dsb-ifeval-p03-baseline).
This is the active baseline experiment.

### Earlier custom scenario prototype

P03 asks whether PersonaPlex follows different turn-taking instructions when
the **same user audio** is paired with either “listen quietly” or “give a short
backchannel.” The instruction is included in the session's initial text prompt.
The current upstream server does not support changing it mid-session.

Five draft templates are in `data/p03/scenarios.jsonl`. They need real audio,
source/license details, and event timestamps before rollout is allowed. See
[`data/README.md`](data/README.md). The client and provisional scorer are under
`src/phase3/`. This small custom pilot is for checking that the pipeline and
paired comparison work; it does not reproduce the full Instruct-FD benchmark.

### 1. Install and validate scenarios

Run from the repository root:

```bash
uv sync --project Phase_3
uv run --project Phase_3 phase3-p03 validate
```

This should report five draft pairs. Prepare audio and annotations as described
in `data/README.md`, then validate again:

```bash
uv run --project Phase_3 phase3-p03 validate --require-ready
```

This must succeed before any connection to the model.

### 2. Connect to the Modal server

Deploy or start the private Modal server using
[`inference/modal/README.md`](inference/modal/README.md). In another terminal,
set the proxy URL to the active deployment and start the local proxy:

```bash
PERSONAPLEX_MODAL_URL=https://<current-modal-web-url> \
  uv run --with aiohttp python Phase_3/inference/modal/modal_personaplex_proxy.py
```

Enter the Modal Proxy Token when prompted and leave the proxy running. It binds
to `127.0.0.1:8765`; the rollout client connects there and does not need the
token itself.

### 3. Capture baseline outputs

In a third terminal, after the manifest is ready and the proxy is connected:

```bash
uv run --project Phase_3 phase3-p03 rollout --pair-id p03_001 --repeats 2
```

The run is saved under `Phase_3/runs/p03/baseline/<run-id>/`: audio, transcript,
rollout metadata, and a run summary. Realtime replay is the default because
PersonaPlex is streaming. `--fast` is for transport debugging only.

The one-pair command above is a connection and format check. For the first
behavior comparison, run all ready pairs (omit `--pair-id`) so both instructions
are tested on each recording and there are enough outputs for a small review.

### 4. Score and manually inspect

```bash
uv run --project Phase_3 phase3-p03 score \
  --manifest data/p03/scenarios.jsonl \
  --run-dir Phase_3/runs/p03/baseline/<run-id>
uv run --project Phase_3 phase3-p03 review-export \
  --scores Phase_3/runs/p03/baseline/<run-id>/scores.json \
  --output Phase_3/runs/p03/baseline/<run-id>/human_review.csv
```

Listen to each output and fill `human_action_correct` with `1`, `0`, or `unsure`.
Review audio before looking at the heuristic score. Then summarize agreement and
whether repeated outputs vary:

```bash
uv run --project Phase_3 phase3-p03 gate \
  --scores Phase_3/runs/p03/baseline/<run-id>/scores.json \
  --review Phase_3/runs/p03/baseline/<run-id>/human_review.csv \
  --output Phase_3/runs/p03/baseline/<run-id>/gate.json
```

`paired_instruction_sensitivity.json` compares matched repeats: the listen
condition should avoid speaking over the user's floor, while the backchannel
condition should produce a short response near its marked opportunity. The
current energy-based scorer only detects rough speech timing. It cannot judge
whether a backchannel is meaningful, so its score is not a training reward. A
positive gate is only a prompt for research review, not approval to train. No
LoRA-GRPO trainer is included: the current upstream WebSocket does not expose
sampled token log-probabilities required to update the model correctly.

The custom scenario prototype described above has not been run or trained.

## Reading the backbone

See [`docs/architecture-and-code-reading.md`](docs/architecture-and-code-reading.md),
[`docs/personaplex_source_reading_guide.md`](docs/personaplex_source_reading_guide.md),
and [`docs/experiment-design.md`](docs/experiment-design.md).


shaty cua voice, nhan context moi trong khi dang tra loi, actine the nao the think tra loi cho talk

optimization

training 
