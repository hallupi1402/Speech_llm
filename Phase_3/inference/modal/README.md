# Modal inference adapters

`modal_personaplex.py` and `modal_personaplex_proxy.py` are copied from
`Phase_2_plan_inference/personaplex/` as the current serving path. The first
downloads the gated checkpoint to a persistent Modal Volume and serves
`moshi.server` on an A100-80GB. The image now checks out the pinned upstream
commit in `configs/p03_pilot.json`. The second forwards local browser/WebSocket
traffic to the authenticated Modal endpoint.

This is the model serving adapter. The P03 experiment runner is implemented in
`src/phase3/rollout/` and uses the same upstream WebSocket protocol through the
local proxy. It captures audio, text, and run metadata.

From the repository root, prepare the persistent model cache once and deploy
the private server with:

```bash
uv run --with modal modal run Phase_3/inference/modal/modal_personaplex.py
uv run --with modal modal deploy Phase_3/inference/modal/modal_personaplex.py
```

The first command downloads the gated weights to the Modal Volume (no GPU is
requested for the download function). The deploy command prints the active web
URL; use that URL as `PERSONAPLEX_MODAL_URL` when starting the local proxy.

## Temporary PersonaPlex preview

The current development endpoint is
`https://hyennguyen140205--personaplex-serve-personaplex-dev.us-east.modal.run`.
It requires Modal proxy credentials. To open the upstream UI through a
loopback-only local proxy, run from the repository root:

```bash
PERSONAPLEX_MODAL_URL=https://hyennguyen140205--personaplex-serve-personaplex-dev.us-east.modal.run \
  .venv/bin/python Phase_3/inference/modal/modal_personaplex_proxy.py
```

Enter your Modal Proxy Token ID and Secret when prompted, then open
`http://127.0.0.1:8765`. The proxy binds to loopback and forwards both the UI
and `/api/chat` WebSocket traffic to the protected dev endpoint. Do not paste
the token values into chat or save them in a file.

For controlled P03 experiments:

- Keep Hugging Face and Modal credentials in Modal Secrets or an interactive
  local prompt. Do not copy Phase 2 `.env` files into this directory.
- Set `PERSONAPLEX_MODAL_URL` to the active deployment when starting the local
  proxy. Its fallback URL is only a convenience and may no longer be current.
- Rollout metadata records the model/source revisions, voice and role prompts,
  instruction, replay settings, and that a reproducible per-session seed is not
  exposed by the current server.

## DSB-IFEval P03 baseline

`modal_p03_dsb.py` runs the selected 10 audio pairs (20 L1/L2 cases) with the
official DSB-IFEval PersonaPlex offline adapter on one Modal A100. It uses the
same initial text conditioning as the WebSocket server, a fixed `NATF0` voice,
and the benchmark's 3 s gap and 5 s tail. The adapter generates on a shared
audio frame clock and saves float32 stereo WAVs plus `.rec.json` files in the
`p03-dsb-results` Modal Volume. It scores each case with the official IAS
primitives and that case's verifier parameters. No proxy token is needed.

```bash
.venv/bin/modal run Phase_3/inference/modal/modal_p03_dsb.py --smoke
.venv/bin/modal run Phase_3/inference/modal/modal_p03_dsb.py --repeats 4
.venv/bin/modal volume get p03-dsb-results <run-id>/scores.jsonl Phase_3/runs/<run-id>/scores.jsonl
```

The smoke command runs one L1/L2 pair for each of LISTEN, BACKCHANNEL and
NO_BACKCHANNEL. The full command runs 80 rollouts. Replace `<run-id>` with the
identifier printed at completion. The selected dataset is in
`data/p03/dsb_pilot_subset/`; the reference scorer source is cloned from
`https://github.com/dsb-ifeval/dsb-ifeval` into `upstream/dsb-ifeval`.

The DSB-IFEval audio license is CC-BY-NC-4.0. Report these results as a
20-case pilot, not as the full benchmark score. Human listening checks are
still needed before using IAS as a training reward.
