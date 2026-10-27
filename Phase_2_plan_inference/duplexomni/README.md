# DuplexOmni inference on Modal

This directory contains the deployment adapters and validation work for
DuplexOmni inference. The upstream model and modified vLLM source are kept in a
persistent Modal Volume; this repository contains the orchestration code,
clients, evidence, and run notes.

## Modules

- [`modal_runtime.py`](modal_runtime.py): shared Volume, paths, images, runtime
  environment, GPU layout, and health helpers.
- [`prepare_source_modal.py`](prepare_source_modal.py),
  [`download_model_modal.py`](download_model_modal.py), and
  [`inspect_model_modal.py`](inspect_model_modal.py): source and checkpoint
  setup.
- [`rebuild_vllm_a100_modal.py`](rebuild_vllm_a100_modal.py),
  [`smoke_test_modal.py`](smoke_test_modal.py), and
  [`runtime_import_smoke_modal.py`](runtime_import_smoke_modal.py): build and
  import validation.
- [`model_load_smoke_modal.py`](model_load_smoke_modal.py) and
  [`realtime_modal.py`](realtime_modal.py): model-load and realtime services.
- [`baseline_bf16_modal.py`](baseline_bf16_modal.py): historical 4-GPU BF16
  inference run.
- [`omni_realtime_mac_client.py`](omni_realtime_mac_client.py): local
  microphone and speaker client.

## Run order

Run commands from the repository root with the intended Modal profile active:

```bash
./.venv/bin/modal profile activate modal-new
./.venv/bin/modal run Phase_2_plan_inference/duplexomni/prepare_source_modal.py
./.venv/bin/modal run Phase_2_plan_inference/duplexomni/download_model_modal.py
./.venv/bin/modal run Phase_2_plan_inference/duplexomni/inspect_model_modal.py
./.venv/bin/modal run Phase_2_plan_inference/duplexomni/rebuild_vllm_a100_modal.py
./.venv/bin/modal run Phase_2_plan_inference/duplexomni/runtime_import_smoke_modal.py
./.venv/bin/modal run Phase_2_plan_inference/duplexomni/model_load_smoke_modal.py
```

The vLLM build is isolated from inference. It uses CUDA 12.8 devel, Python
3.11, PyTorch 2.9.1, and `TORCH_CUDA_ARCH_LIST=8.0` for A100 `sm80`. A
successful build writes a marker into the Volume so a later run can skip the
same source revision.

For the persistent realtime endpoint, run `modal serve` with
`Phase_2_plan_inference/duplexomni/realtime_modal.py`, then connect from the
local client. Run the microphone client locally, outside the GPU container.

## Recorded status

The historical 4-GPU experiment passed model loading and BF16 WAV inference,
including ASR text and generated audio. The newer configuration uses
8×A100-80GB: Thinker TP=4 on GPUs 0–3 and Talker TP=4 on GPUs 4–7, with MTP on
GPU 7. The 8-GPU adapters are present, but the repository's earlier notes do
not record a completed build/download/model-load run for that configuration.
Realtime microphone output was not accepted as a pass because transcript and
playback were not reliably observed.

See [`docs/inference_duplexomni/submission_summary.md`](../../docs/inference_duplexomni/submission_summary.md)
and [`docs/inference_duplexomni/duplexomni_inference_plan.md`](../../docs/inference_duplexomni/duplexomni_inference_plan.md)
for the recorded evidence and plan.
