# Phase 3 — PersonaPlex research and experiments

Phase 3 is the research workspace for understanding the PersonaPlex backbone,
reproducing its inference behavior, and then evaluating post-training ideas for
P1–P3. Model execution and training are intended to run on Modal GPUs. This
checkout contains the Modal inference adapters, a local NVIDIA source checkout
for reading, and the planned code layout. Model weights remain remote.

## Current state

- `inference/modal/` contains the inference adapter and local authenticated
  WebSocket proxy copied from Phase 2.
- `upstream/personaplex/` contains the local NVIDIA source checkout for reading
  only. The parent repo ignores this nested checkout; its current commit is
  recorded in `upstream/README.md`.
- `src/phase3/` reserves modules for scenarios, rollouts, rewards, training,
  and evaluation. These are design placeholders until the experiment
  interfaces are settled.
- `data/` and `runs/` hold manifests and run metadata. Keep gated weights,
  credentials, and large/private audio out of Git.

## Reading order

1. [`docs/architecture-and-code-reading.md`](docs/architecture-and-code-reading.md)
   — model components and upstream code path.
2. [`docs/personaplex_source_reading_guide.md`](docs/personaplex_source_reading_guide.md)
   — entry points, call chains, and exact upstream line ranges.
3. [`docs/personaplex_architecture_slides.html`](docs/personaplex_architecture_slides.html)
   — standalone 16:9 slide deck with the architecture and code map.
4. [`inference/modal/README.md`](inference/modal/README.md) — current Modal
   adapter behavior and limitations.
5. [`docs/experiment-design.md`](docs/experiment-design.md) — planned test
   structure for P1–P3 and the RL path.
5. Inspect `moshi/server.py`, `moshi/offline.py`,
   `moshi/models/loaders.py`, `moshi/models/lm.py`, then the streaming and
   transformer modules. Follow the inference call chain before making patches.

## Execution rule

Use this repository to define experiments and submit jobs. Run model inference,
rollouts, and training on Modal; do not make local GPU execution a requirement.
The copied adapter still clones NVIDIA's default branch during image build.
Before relying on it for experiments, pin that clone to the same commit recorded
for `upstream/personaplex/` or change the image to package the pinned checkout.
