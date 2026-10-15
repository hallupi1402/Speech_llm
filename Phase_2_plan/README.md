# Phase 2 deployment adapters

The code in this directory is orchestration code. It is not a second model
implementation: the model, serving modules, and modified vLLM are loaded from
the pinned DuplexOmni source tree in Modal Volume.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `modal_runtime.py` | Shared infrastructure interface: paths, Volume, images, environment, health, and resource constants. |
| `prepare_source_modal.py` | Clone/reuse the upstream source and record its Git revision. |
| `download_model_modal.py` / `inspect_model_modal.py` | Persist and verify the Hugging Face checkpoint. |
| `rebuild_vllm_a100_modal.py` | Compile the modified vLLM fork for A100 `sm80`; commit only after a successful build. |
| `runtime_import_smoke_modal.py` | Check runtime imports without loading the large checkpoint. |
| `model_load_smoke_modal.py` | Start the official 8-GPU Thinker/Talker stack and poll health endpoints. |
| `realtime_modal.py` | Keep the 8-GPU services and WebSocket bridge alive for a local client. |
| `baseline_bf16_modal.py` | Historical 4-GPU BF16 WAV regression/evidence runner. |
| `omni_realtime_mac_client.py` | Local microphone and speaker client; it does not allocate GPU. |

Each Modal script owns one lifecycle. Shared deployment details belong in
`modal_runtime.py`; experiment-specific process management belongs in the
individual adapter.

## Canonical resource layout

- Modal GPU: `A100-80GB:8` on one node.
- Thinker: physical GPUs `0–3`, tensor parallel size 4.
- Talker: physical GPUs `4–7`, tensor parallel size 4.
- Talker MTP: local Talker device 3, corresponding to physical GPU 7.
- Inference container: 32 CPUs and 128 GiB RAM.
- Build container: 8 CPUs, CUDA devel image, and four Ninja jobs.

## Submission notes

The source/build/checkpoint artifacts are intentionally not committed to Git;
they are large and persist in Modal Volume. The repository records the exact
paths, commands, resource configuration, historical evidence, and known
limitations so another operator can reproduce the run.
