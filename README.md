# DuplexOmni inference on Modal

This repository contains the Phase 2 deployment and validation work for
DuplexOmni inference. The upstream model and modified vLLM source are kept in
a persistent Modal Volume; the local repository contains the deployment
adapters, smoke tests, evidence, and reports.

## What is implemented

- Shared Modal infrastructure in [`Phase_2_plan_inference/modal_runtime.py`](Phase_2_plan_inference/modal_runtime.py): Volume, paths, CUDA images, runtime environment, GPU layout, and health helpers.
- Source/bootstrap and artifact jobs: [`prepare_source_modal.py`](Phase_2_plan_inference/prepare_source_modal.py), [`download_model_modal.py`](Phase_2_plan_inference/download_model_modal.py), and [`inspect_model_modal.py`](Phase_2_plan_inference/inspect_model_modal.py).
- Modified-vLLM build and import validation: [`rebuild_vllm_a100_modal.py`](Phase_2_plan_inference/rebuild_vllm_a100_modal.py), [`smoke_test_modal.py`](Phase_2_plan_inference/smoke_test_modal.py), and [`runtime_import_smoke_modal.py`](Phase_2_plan_inference/runtime_import_smoke_modal.py).
- Canonical 8-GPU service adapters: [`model_load_smoke_modal.py`](Phase_2_plan_inference/model_load_smoke_modal.py) and [`realtime_modal.py`](Phase_2_plan_inference/realtime_modal.py).
- Historical 4-GPU BF16 WAV evidence: [`baseline_bf16_modal.py`](Phase_2_plan_inference/baseline_bf16_modal.py) and [`Phase_2_plan_inference/evidence/`](Phase_2_plan_inference/evidence/).
- Local microphone/WebSocket clients are separate from the Modal model server; see [`omni_realtime_mac_client.py`](Phase_2_plan_inference/omni_realtime_mac_client.py).

## Execution order

Run commands from the repository root with the intended Modal profile active:

```bash
./.venv/bin/modal profile activate modal-new
./.venv/bin/modal run Phase_2_plan_inference/prepare_source_modal.py
./.venv/bin/modal run Phase_2_plan_inference/download_model_modal.py
./.venv/bin/modal run Phase_2_plan_inference/inspect_model_modal.py
./.venv/bin/modal run Phase_2_plan_inference/rebuild_vllm_a100_modal.py
./.venv/bin/modal run Phase_2_plan_inference/runtime_import_smoke_modal.py
./.venv/bin/modal run Phase_2_plan_inference/model_load_smoke_modal.py
```

The vLLM build is isolated from inference. It uses CUDA 12.8 devel, Python
3.11, PyTorch 2.9.1, and `TORCH_CUDA_ARCH_LIST=8.0` for A100 `sm80`. A
successful build writes a marker into the Volume, so a later run can skip the
same source revision instead of compiling again.

For the persistent realtime endpoint, use `modal serve` with
`Phase_2_plan_inference/realtime_modal.py`, then connect from the local client. Do not
run the microphone client inside the GPU container.

## Current status

The historical 4-GPU experiment passed model loading and BF16 WAV inference,
including ASR text and generated audio. The new canonical configuration is
8×A100-80GB: Thinker TP=4 on GPUs 0–3 and Talker TP=4 on GPUs 4–7, with MTP
on GPU 7. The 8-GPU adapters are prepared, but the new workspace still needs
its successful build/download/model-load run before claiming an 8-GPU result.
Realtime microphone output was not accepted as a pass because transcript and
playback were not reliably observed.

See [`docs/submission_summary.md`](docs/submission_summary.md) for the
reviewer-facing summary and [`docs/duplexomni_inference_plan.md`](docs/duplexomni_inference_plan.md)
for the detailed execution plan and evidence.
