# DuplexOmni trên Modal: các bước setup và debug smoke test

## 1. Mục tiêu

Mục tiêu của giai đoạn này là build và import modified vLLM của DuplexOmni trên Modal GPU trước khi load checkpoint hoặc chạy inference thật.

Smoke test hiện tại **chưa** kiểm tra:

- download/load model weights;
- Thinker/Talker runtime;
- codec audio;
- realtime latency;
- chất lượng output.

File smoke test:

```text
Phase_2_plan /smoke_test_modal.py
```

Lưu ý: thư mục `Phase_2_plan ` có một dấu cách ở cuối tên. Khi chạy command phải giữ nguyên dấu cách và nên quote path.

## 2. Lỗi ban đầu

Smoke test cũ chạy lệnh:

```bash
pip install --no-build-isolation -e .
```

Build vLLM fail ở CMake với log chính:

```text
CUDA_TOOLKIT_ROOT_DIR not found or specified
Could NOT find CUDA
Your installed Caffe2 version uses CUDA but I cannot find the CUDA libraries.
```

Lỗi xảy ra trong `CMakeLists.txt` của vLLM fork, không phải ở bước pip metadata.

## 3. Cách phân tích root cause

### 3.1. Phân biệt PyTorch CUDA runtime và CUDA toolkit

PyTorch wheel có thể chứa CUDA runtime để chạy tensor, nhưng quá trình build C++/CUDA extension của vLLM cần toolkit đầy đủ:

- `nvcc`;
- CUDA headers, ví dụ `cuda_runtime_api.h`;
- CUDA libraries, ví dụ `libcudart.so`;
- CMake có thể tìm thấy CUDA toolkit.

Vì vậy image runtime hoặc image chỉ có PyTorch CUDA không đủ cho editable build.

### 3.2. Kiểm tra version của fork

Đối chiếu source của DuplexOmni cho thấy modified vLLM hiện yêu cầu stack CUDA/PyTorch mới hơn cấu hình smoke test cũ. Các dependency build được lấy từ source hiện tại, không tự đoán version:

- Python 3.10–3.13;
- PyTorch 2.9.1;
- CUDA 12.x;
- CMake và Ninja cho build extension.

Tham khảo:

- [DuplexOmni INSTALL.md](https://github.com/MuyeHuang/DuplexOmni/blob/main/INSTALL.md)
- [requirements/build.txt](https://github.com/MuyeHuang/DuplexOmni/blob/main/inference_framework/vllm_qwen3_omni/requirements/build.txt)
- [vLLM fork pyproject.toml](https://github.com/MuyeHuang/DuplexOmni/blob/main/inference_framework/vllm_qwen3_omni/pyproject.toml)

### 3.3. Phát hiện lỗi version metadata sau khi sửa CUDA

Sau khi chuyển sang CUDA devel image, CMake đã nhận CUDA đúng. Lỗi kế tiếp là:

```text
LookupError: setuptools-scm was unable to detect version for .../vllm_qwen3_omni.
However, a repository was found in a parent directory: .../DuplexOmni
```

Nguyên nhân là vLLM nằm trong subdirectory của repository DuplexOmni. `setuptools-scm` tìm Git metadata tại thư mục package và không tự dùng parent repository.

## 4. Các thay đổi trong smoke test

### 4.1. Dùng CUDA devel image

Image được đổi sang:

```python
modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.11",
)
```

Image này cung cấp compiler/toolkit cần cho CMake build.

### 4.2. Pin build stack

Các dependency quan trọng:

```text
torch==2.9.1
torchaudio==2.9.1
torchvision==0.24.1
cmake>=3.26.1
ninja
setuptools>=77.0.3,<81.0.0
setuptools-scm>=8
```

Build requirements được cài từ file source của fork trước khi editable install.

### 4.3. Cấu hình CUDA explicit

Smoke test tự resolve CUDA toolkit và truyền các biến sau vào build:

```text
CUDA_HOME
CUDA_PATH
CUDACXX
VLLM_TARGET_DEVICE=cuda
CMAKE_ARGS=-DCUDA_TOOLKIT_ROOT_DIR=... -DCMAKE_CUDA_COMPILER=...
CMAKE_PREFIX_PATH=<Torch CMake prefix>
PATH=<CUDA_HOME>/bin:$PATH
LD_LIBRARY_PATH=<CUDA_HOME>/lib64:$LD_LIBRARY_PATH
```

Script cũng kiểm tra trước:

```text
<CUDA_HOME>/bin/nvcc
<CUDA_HOME>/include/cuda_runtime_api.h
<CUDA_HOME>/lib64/libcudart.so
```

Nếu thiếu một trong các thành phần này, script fail sớm với message rõ ràng thay vì chờ CMake fail ở sâu trong build.

### 4.4. Kiểm tra GPU trước khi build

Script import PyTorch và kiểm tra:

- `torch.cuda.is_available()`;
- tên GPU;
- compute capability;
- Torch version;
- CUDA version của Torch;
- output `nvcc --version`.

Cấu hình hiện tại dùng Modal T4 và mặc định:

```text
TORCH_CUDA_ARCH_LIST=7.5
MAX_JOBS=4
```

### 4.5. Xử lý setuptools-scm trong nested repository

Trước khi build, script lấy commit của parent repository DuplexOmni rồi set:

```text
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+duplexomni.<commit>
```

Giá trị này chỉ là fallback cho package metadata của smoke test; không thay đổi source Git.

### 4.6. Không cài dependency runtime ngoài phạm vi smoke test

Editable install dùng:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

Lý do:

- build dependencies đã được cài explicit;
- tránh pip tự resolve lại toàn bộ runtime stack trong lúc debug build;
- dễ phân biệt lỗi build với lỗi dependency runtime.

## 5. Execution flow hiện tại

```text
modal run smoke_test_modal.py
        |
        v
Modal CUDA devel image + T4
        |
        v
resolve_cuda_home()
        |
        v
print_environment()
  - Python / Torch / CUDA
  - GPU / compute capability
  - nvcc / headers / libraries
        |
        v
prepare_build_environment()
        |
        v
clone_repo()
  - clone DuplexOmni vào Modal Volume nếu chưa có
  - reuse repo nếu đã tồn tại
        |
        v
build_vllm()
  - set setuptools-scm fallback version
  - install requirements/build.txt
  - editable build modified vLLM
        |
        v
validate_import()
  - import vllm
  - kiểm tra location trỏ vào fork
        |
        v
commit Modal Volume
```

## 6. Các lần test đã thực hiện

### 6.1. Kiểm tra cục bộ

```bash
./.venv/bin/python -m py_compile './Phase_2_plan /smoke_test_modal.py'
```

Kết quả: pass.

Import module Modal cục bộ cũng pass.

### 6.2. Modal run lần đầu sau khi sửa CUDA

Các bước đã pass:

```text
Python 3.11.5
Torch 2.9.1+cu128
Torch CUDA 12.8
GPU Tesla T4
Compute capability 7.5
CUDA_HOME=/usr/local/cuda-12.8
nvcc=12.8.93
```

Build dừng ở lỗi `setuptools-scm` do nested repository. Lỗi này đã được xử lý bằng `SETUPTOOLS_SCM_PRETEND_VERSION`.

### 6.3. Modal run lần hai sau khi sửa setuptools-scm

Kết quả đã xác nhận:

- CUDA diagnostics: pass;
- GPU diagnostics: pass;
- clone/reuse repo: pass;
- setuptools-scm metadata: pass;
- bắt đầu editable build vLLM: pass.

Job bị Modal timeout sau 1800 giây trong lúc compile:

```text
Building editable for vllm (pyproject.toml): still running...
Task ... hit its timeout of 1800s
```

Không có traceback CMake mới. Kết luận hiện tại là build đã qua lỗi CUDA ban đầu nhưng chưa đủ thời gian hoàn tất compile trên T4 với `MAX_JOBS=4`.

## 7. Command chạy lại

Từ root repo:

```bash
./.venv/bin/modal run './Phase_2_plan /smoke_test_modal.py'
```

Theo dõi app/log trên Modal bằng app ID của lần chạy tương ứng. Không dùng unquoted path vì tên thư mục có trailing space.

## 8. Checklist debug nếu chạy lại vẫn fail

### Bước 1: kiểm tra log lỗi cuối cùng

Không kết luận từ warning đầu tiên. Tìm traceback cuối cùng và xác định bước fail:

- image build;
- CUDA diagnostics;
- clone repo;
- pip build requirements;
- setuptools metadata;
- CMake compile;
- import runtime.

### Bước 2: xác nhận CUDA toolkit trong container

Các lệnh cần kiểm tra:

```bash
which nvcc
nvcc --version
echo "$CUDA_HOME"
test -f "$CUDA_HOME/include/cuda_runtime_api.h"
test -f "$CUDA_HOME/lib64/libcudart.so"
```

Nếu `nvcc` không tồn tại, image đang sai loại; cần dùng CUDA `devel`, không dùng `runtime`.

### Bước 3: xác nhận Torch/CUDA compatibility

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

Không đổi version ngẫu nhiên. Trước tiên đối chiếu với `requirements/build.txt` và `pyproject.toml` của fork.

### Bước 4: xác nhận CMake nhận đúng compiler

Kiểm tra các biến:

```bash
echo "$CUDACXX"
echo "$CMAKE_ARGS"
echo "$CMAKE_PREFIX_PATH"
```

Nếu CMake vẫn báo không tìm thấy CUDA, kiểm tra lại path thực tế của `nvcc`, headers và `libcudart.so` trong image.

### Bước 5: nếu lỗi lại là setuptools-scm

Kiểm tra:

```bash
git -C /root/workspace/DuplexOmni rev-parse --short HEAD
echo "$SETUPTOOLS_SCM_PRETEND_VERSION"
```

Đảm bảo biến version được set trong cùng environment với lệnh pip build.

### Bước 6: nếu chỉ bị timeout khi compile

Ưu tiên theo thứ tự:

1. Tăng `MAX_JOBS` nếu container có đủ CPU/RAM.
2. Tăng Modal function timeout.
3. Tách build thành Modal image build layer để cache artifact.
4. Dùng CPU/GPU instance phù hợp hơn cho build đầu tiên.
5. Chỉ sau khi build pass mới tối ưu runtime hoặc quantization.

Không nên coi timeout compile là lỗi inference hay lỗi CUDA nếu log chưa có compile error.

## 9. Kết quả build smoke ban đầu

### Kết quả cuối cùng

Kết quả build smoke ban đầu đã hoàn tất:

- Compile 352/352 CUDA/C++ tasks.
- Editable install của modified vLLM thành công.
- `import vllm` thành công.
- Kiểm tra source bằng `os.path.samefile` thành công dù Modal expose cùng volume qua hai path alias khác nhau.
- Artifact đã được commit vào Modal Volume để tái sử dụng.

Môi trường lần chạy thành công:

```text
GPU: NVIDIA T4
CPU: 8
RAM: 32 GiB
Timeout: 5400 giây
Python: 3.11
Torch: 2.9.1+cu128
CUDA: 12.8
Fork commit: 33bfba1
Volume: duplexomni-smoke-vol-v2
```

Thời gian editable build khoảng 2861 giây (~47 phút 41 giây). Đây là build môi trường, chưa phải thời gian load hoặc inference model.

### Bước tiếp theo

1. Đối chiếu serving entrypoints và runtime requirements.
2. Tải checkpoint `MuyeHuang/DuplexOmni` vào persistent volume; không nhúng checkpoint vào image.
3. Chạy model-load smoke test cho Thinker/Talker và health check.
4. Chạy baseline BF16 end-to-end bằng input WAV cố định.
5. Đo VRAM, thời gian load, latency và RTF.
6. Chỉ sau khi baseline ổn định mới chạy realtime, quantization hoặc giảm số GPU.

## Kết quả triển khai tiếp theo (2026-10-12)

### Phase 2–4 đã hoàn tất

- Checkpoint `MuyeHuang/DuplexOmni` đã tải vào Modal Volume `duplexomni-smoke-vol-v2`, revision `b8a5ff6395ae51460d0402424fbd3359614a901a`, 29 file model, khoảng 70.5 GB.
- Runtime import đã pass cho `serving_core.server_thinker`, `server_talker` và `server_orchestrator`. Lỗi metadata package được xử lý bằng dist-info shim để vLLM nhận `NvmlCudaPlatform` và `device_type=cuda`.
- Lần model-load 4-GPU đầu tiên load weights thành công nhưng fail ở FlashAttention vì binary build trên T4 (`sm75`) chạy trên A100 (`sm80`). Việc chỉ set `VLLM_ATTENTION_BACKEND=TRITON_ATTN` không đủ vì multimodal encoder vẫn dùng FlashAttention.
- Đã rebuild modified vLLM trên 1x A100-80GB với `TORCH_CUDA_ARCH_LIST=8.0`; CMake ghi nhận `-gencode arch=compute_80,code=sm_80`, hoàn tất 369/369 targets và commit artifact vào Volume. Build elapsed khoảng 6546 giây (~109 phút).
- Model-load smoke sau rebuild đã pass trên 4x A100-80GB: Thinker TP=2 trên GPU 0–1, Talker TP=2 trên GPU 2–3, MTP subprocess trên GPU 3, Orchestrator trên port 21000. Tất cả health 200, `ready_elapsed_seconds=530.6`, không OOM.

Apps: [A100 rebuild](https://modal.com/apps/yennguyen140205/main/ap-uLP70X0agPQCQeEUFepDeV), [model-load success](https://modal.com/apps/yennguyen140205/main/ap-8vfb9CT3XFdvwJNillWCXP).

### Warning còn ghi nhận

- A100 experiment dùng TP=2 + TP=2 để tiết kiệm GPU; chưa phải baseline chính thức 8-GPU TP=4 + TP=4 của upstream.
- Thinker và MTP có warning dùng default MoE config do thiếu file profile `E=128,N=384`; service vẫn ready nhưng hiệu năng có thể chưa tối ưu.
- Có warning tokenizer regex và kernel cache directory; không chặn model load, cần xử lý trước benchmark chất lượng/latency.

### Bước tiếp theo trước khi chạy baseline

1. Tạo `baseline_bf16_modal.py`, giữ checkpoint/artifact hiện tại và launcher 4-GPU đã pass.
2. Gửi một WAV cố định qua `simulate_v8.py` hoặc client chính thức.
3. Ghi text output, audio output, request latency, first-audio latency và peak VRAM.
4. Chỉ sau E2E pass mới chạy baseline 8-GPU chính thức, realtime/interruption và quantization.


## Kết quả baseline BF16 E2E (2026-10-12)

- Job thành công: [Modal app ap-0h25nzZzuKJZoVzg0yBoB1](https://modal.com/apps/yennguyen140205/main/ap-0h25nzZzuKJZoVzg0yBoB1).
- Cấu hình: 4x A100-80GB; Thinker TP=2 trên GPU 0–1; Talker TP=2 trên GPU 2–3; MTP subprocess dùng GPU 3; S2 disabled.
- Input upstream `google_Leda.wav`: mono PCM16, 24 kHz, 220,082 frames (~9.17 giây). Client `simulate_v8.py` đã chạy cold và warm.
- Cold: request đầu tiên mất 48.78 giây; audio WebSocket đầu tiên đến sau khoảng 50.715 giây; output hợp lệ 2,818,134 bytes, mono PCM16/24 kHz, 1,409,045 frames (~58.71 giây).
- Warm: request mỗi chunk khoảng 0.24–0.36 giây; audio đầu tiên đến sau khoảng 0.747 giây; output hợp lệ 493,048 bytes, mono PCM16/24 kHz, 246,502 frames (~10.27 giây).
- ASR text đã nhận được qua các chunk, gồm `The weather is`, `nice and I speak calmly` và phần tiếng Trung; các chunk đều HTTP 200. Audio talker đến qua WebSocket và session được cleanup.
- VRAM cuối job: GPU0 67,433 MiB; GPU1 78,255 MiB; GPU2 52,393 MiB; GPU3 50,897 MiB. Tổng job 625.7 giây.
- Lỗi trong run đầu: alias `S1_MODEL_NAME=qwen3-omni` khiến Thinker trả 400; đã sửa thành checkpoint path tuyệt đối và thêm fail-fast khi client có `S1 Error`. Retry sau đó PASS.

### Kết luận hiện tại

Build/import, checkpoint, runtime import, model-load và E2E functional text+audio 4-GPU đã pass. Chưa thể kết luận realtime/RTF cuối vì đây mới là cold/warm functional baseline, MoE profile `E=128,N=384` còn warning default config, và chưa chạy benchmark lặp lại hoặc baseline chính thức 8-GPU. Bước kế tiếp là benchmark có kiểm soát, sau đó realtime interruption và quantization theo plan.
