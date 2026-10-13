# DuplexOmni Inference Research Plan

## Mục tiêu

Research và thử nghiệm cách chạy inference DuplexOmni, từ việc dựng môi trường GPU, chạy baseline BF16, test realtime, đến quantization và benchmark để tìm cấu hình GPU hợp lý nhất.

---

## Trạng thái hiện tại (2026-10-12)

### Đã hoàn thành

- [x] Đọc execution flow trong `inference_framework/realtime_serving`: Thinker → Talker → Orchestrator → client/benchmark.
- [x] Xác định modified vLLM tại `inference_framework/vllm_qwen3_omni`; đây là fork của DuplexOmni, không phải vLLM upstream.
- [x] Tạo smoke-test environment trên Modal bằng CUDA 12.8 devel, Python 3.11 và PyTorch 2.9.1+cu128.
- [x] Chạy trên 1x NVIDIA T4, 8 CPU, 32 GiB RAM, timeout 5400 giây (90 phút).
- [x] Clone/reuse source trong Modal Volume `duplexomni-smoke-vol-v2`, fork commit `33bfba1`.
- [x] Sửa các lỗi CUDA toolkit, setuptools-scm trong nested repository và thư mục package CUDA bị thiếu khi editable build.
- [x] Compile thành công 352/352 CUDA/C++ tasks và cài editable modified vLLM.
- [x] Validate `import vllm` và xác nhận package được import từ source fork bằng kiểm tra inode/path alias của Modal Volume.
- [x] Commit artifact sau build để các lần chạy sau có thể tái sử dụng volume.
- [x] Tải checkpoint `MuyeHuang/DuplexOmni` revision `b8a5ff6395ae51460d0402424fbd3359614a901a`: 29 file model, khoảng 70.5 GB, vào Volume.
- [x] Kiểm tra manifest thực tế: 15 safetensors shards, `Qwen3OmniMoeForConditionalGeneration`, BF16, gồm Thinker/Talker/Code2Wav.
- [x] Runtime import smoke pass cho Thinker, Talker và Orchestrator; sửa metadata shim để vLLM chọn `NvmlCudaPlatform`.
- [x] Rebuild modified vLLM trên A100 với `TORCH_CUDA_ARCH_LIST=8.0`; 369/369 CUDA/C++ targets và FlashAttention `sm80` đã commit vào Volume.
- [x] Model-load smoke 4-GPU pass: Thinker/Talker/MTP/Orchestrator đều health 200, elapsed 530.6 giây.
- [x] Baseline BF16 E2E 4-GPU pass: cold/warm `simulate_v8.py` đều trả HTTP 200, nhận ASR text và audio WebSocket; job elapsed 625.7 giây.

Evidence:

- [`smoke_test_modal.py`](./smoke_test_modal.py)
- [`duplexomni_modal_smoke_test_debug.md`](../docs/duplexomni_modal_smoke_test_debug.md)
- [`download_model_modal.py`](./download_model_modal.py)
- [`inspect_model_modal.py`](./inspect_model_modal.py)
- [`runtime_import_smoke_modal.py`](./runtime_import_smoke_modal.py)
- [`model_load_smoke_4gpu_modal.py`](./model_load_smoke_4gpu_modal.py)
- [`rebuild_vllm_a100_modal.py`](./rebuild_vllm_a100_modal.py)
- [`baseline_bf16_modal.py`](./baseline_bf16_modal.py)
- Modal app: [ap-hb1KxXToeKEmONPcsJ16jw](https://modal.com/apps/yennguyen140205/main/ap-hb1KxXToeKEmONPcsJ16jw)

### Chưa hoàn thành

- [x] Đã chạy inference DuplexOmni end-to-end với WAV cố định `google_Leda.wav`.
- [x] Đã chạy inference DuplexOmni thật ở baseline BF16 4-GPU, gồm cold và warm run.
- [ ] Chưa chạy benchmark chính thức 8-GPU, realtime interruption hoặc quantization.

Build/import smoke, model-load smoke và E2E text/audio 4-GPU đã pass. Chưa có kết luận realtime interruption, benchmark chính thức 8-GPU hoặc quantization.

---

## Kế hoạch bước tiếp theo sau smoke test

Mục tiêu gần nhất là chạy baseline BF16 tối thiểu trước khi quantization hoặc tối ưu realtime.

### Bước 1 – Kiểm tra source serving

- Đối chiếu `start_thinker_talker_stack.sh`, `server_thinker.py`, `server_talker.py`, `server_orchestrator.py` và `simulate_v8.py`.
- Xác định chính xác model path, runtime requirements, biến môi trường, tensor parallel và health checks.
- Điều kiện đạt: có command khởi động rõ ràng cho cả ba service.

### Bước 2 – Chuẩn bị checkpoint trên persistent volume

- Tải `MuyeHuang/DuplexOmni` vào Modal Volume; không đưa checkpoint 70.5 GB vào image build.
- Kiểm tra đủ shard/config/processor và ghi lại trạng thái cache để không tải lại khi restart.
- Điều kiện đạt: container mới có thể mở lại source, modified vLLM và checkpoint từ volume.

### Bước 3 – Model-load smoke test

- Chạy Thinker/Talker ở cấu hình nhỏ nhất mà source hỗ trợ để phát hiện lỗi import, config hoặc OOM.
- In log theo mốc: checkpoint discovery → config load → weight load → CUDA graph → server ready.
- Điều kiện đạt: health check HTTP 200, không OOM, có thời gian load và peak VRAM.

### Bước 4 – Baseline BF16 end-to-end (PASS trên 4 GPU)

- Chạy Thinker + Talker + Orchestrator theo cấu hình chính thức; tắt S2 để cô lập pipeline nội bộ.
- Gửi một input WAV cố định bằng `simulate_v8.py`.
- Điều kiện đạt: sinh được text/audio output và lưu log, output, VRAM metrics.

### Bước 5 – Benchmark, realtime và quantization

- Đo model load time, first-audio latency, end-to-end latency, RTF, peak VRAM và chi phí/giờ.
- Chỉ sau khi baseline WAV ổn định mới kiểm tra WebSocket/interruption.
- Chỉ sau baseline BF16 mới thử Thinker FP8 + Talker BF16, rồi mới giảm số GPU.

### Kết quả baseline BF16 4-GPU (2026-10-12)

- Script: [`baseline_bf16_modal.py`](./baseline_bf16_modal.py)
- Modal app: [ap-0h25nzZzuKJZoVzg0yBoB1](https://modal.com/apps/yennguyen140205/main/ap-0h25nzZzuKJZoVzg0yBoB1)
- Input: `google_Leda.wav`, mono PCM16, 24 kHz, 220,082 frames (~9.17 giây).
- Cold run: request đầu tiên 48.78 giây; audio đầu tiên nhận ở khoảng 50.715 giây; output WAV 2,818,134 bytes, 1,409,045 frames (~58.71 giây).
- Warm run: các request theo chunk khoảng 0.24–0.36 giây; audio đầu tiên nhận ở khoảng 0.747 giây; output WAV 493,048 bytes, 246,502 frames (~10.27 giây).
- VRAM cuối job: GPU0 67,433 MiB; GPU1 78,255 MiB; GPU2 52,393 MiB; GPU3 50,897 MiB.
- Job tổng cộng 625.7 giây; cả cold/warm không có `S1 Error`, session được cleanup và output WAV hợp lệ.
- Đây là E2E functional pass trên layout 4 GPU TP=2; chưa dùng làm kết luận latency/realtime cuối vì MoE profile `E=128,N=384` còn thiếu và chưa chạy đủ benchmark lặp lại.

### Thứ tự thực thi gần nhất

```text
Kiểm tra serving entrypoints
        ↓
Chuẩn bị checkpoint trong volume
        ↓
Model-load smoke test
        ↓
Baseline BF16 end-to-end
        ↓
Benchmark 8 GPU
        ↓
Realtime/interruption
        ↓
Quantization + giảm GPU
```

Không dùng lại timeout 1800 giây cho build. Artifact build đã hoàn tất; các bước tiếp theo cần tách timeout theo loại công việc: download checkpoint, model load và inference.

---

## Kế hoạch triển khai chi tiết (research-backed)

### Quyết định kiến trúc

- Inference sẽ chạy trong một Modal Function/container có GPU; máy local chỉ gọi job và theo dõi log.
- Source, modified vLLM build artifact và checkpoint phải nằm trên Modal Volume, không nhúng weights vào image.
- Tách thành các job có thể chạy lại độc lập: download artifact, model-load smoke, full-stack smoke và benchmark.
- Modal hỗ trợ nhiều GPU trong cùng một container bằng cấu hình dạng GPU:n; baseline chính thức cần kiểm tra trên cùng một node, không mặc định chia Thinker/Talker sang hai container.

### Phase 0 – Reproducibility gate

Mục tiêu: đảm bảo mọi lần chạy dùng cùng source và artifact.

1. Ghi lại DuplexOmni commit đang có trong Volume: 33bfba1.
2. Không clone floating main cho các lần chạy chính; script phải checkout một commit cụ thể và in git rev-parse.
3. Xác nhận modified vLLM đã tồn tại trong Volume bằng import vllm, vllm.__file__ và commit source.
4. Ghi manifest gồm source commit, image/runtime versions, GPU layout, model revision và thời điểm chạy.

Pass: chạy lại container mới vẫn thấy đúng source, vLLM và manifest mà không rebuild CUDA.

### Phase 1 – Kiểm tra model artifact trước khi tải lớn

Mục tiêu: không giả định sai cấu trúc checkpoint.

1. Kiểm tra model card và danh sách file của MuyeHuang/DuplexOmni.
2. Xác nhận repository chứa cả Thinker và Talker, đồng thời xác định chúng nằm trong một thư mục chung hay các thư mục con.
3. Đo dung lượng thực tế sau khi tải. Con số khoảng 70.5 GB trong tài liệu cũ không được coi là nguồn sự thật duy nhất; model card hiện tại ghi 35B/BF16.
4. Map chính xác các biến THINKER_MODEL, TALKER_MODEL, S1_MODEL_NAME vào cấu trúc file thực tế.

Pass: có bảng path thực tế, file config/processor và shard weights cần thiết; không start server bằng path đoán.

### Phase 2 – Download checkpoint vào Modal Volume

Mục tiêu: tải một lần, resume được và không mất khi container kết thúc.

1. Tạo một function riêng, mount Volume tại /root/workspace.
2. Dùng huggingface_hub.snapshot_download với local_dir dưới /root/workspace/models/; dùng revision cố định khi đã xác định.
3. In log theo mốc: bắt đầu, model revision, số file, bytes đã tải, tốc độ, kết thúc.
4. Lưu marker/manifest sau khi download và gọi volume.commit() chỉ khi kiểm tra file thành công.
5. Dùng timeout và ephemeral disk riêng cho download; không dùng lại timeout build 90 phút một cách mù quáng.
6. Nếu cần token, truyền bằng Modal Secret, không ghi token vào log hoặc file.

Pass: job retry không tải lại toàn bộ; Volume chứa checkpoint hoàn chỉnh và manifest khớp.

### Phase 3 – Runtime image cho inference

Mục tiêu: dùng build artifact đã có và bổ sung dependency runtime.

1. Giữ CUDA/Torch tương thích với fork: Python 3.11, CUDA 12.8, Torch 2.9.1+cu128.
2. Không chạy lại editable build trong job model-load.
3. Cài/kiểm tra dependency realtime: FastAPI, Uvicorn, WebSockets, Requests, Transformers và các dependency audio cần bởi source.
4. Set VLLM_ROOT trỏ đến /root/workspace/DuplexOmni/inference_framework/vllm_qwen3_omni.
5. Redirect stdout/stderr của Thinker, Talker và Orchestrator vào log có prefix rõ ràng; lưu log vào Volume.

Pass: import được server modules và modified vLLM trong container inference trước khi chiếm GPU load model.

### Phase 4 – Model-load smoke trên multi-GPU Modal

**Trạng thái: PASS (experiment 4-GPU, 2026-10-12).** Artifact runtime được rebuild trên A100 `sm80`; cấu hình đã pass Thinker/Talker/MTP/Orchestrator health 200. Baseline chính thức 8-GPU vẫn cần chạy riêng trước benchmark.

Mục tiêu: chứng minh model load được, tách lỗi load khỏi lỗi request.

1. Dùng multi-GPU cùng node. Cấu hình đầu tiên bám launcher upstream: Thinker trên GPU 0–3, Talker trên GPU 4–7, MTP subprocess trên GPU 7, TP=4.
2. Không dùng T4 đơn lẻ cho bước này; T4 đã hoàn thành vai trò build smoke test.
3. Khởi động Thinker trước và ghi các mốc: config, tokenizer/processor, safetensors, CUDA graph, ready.
4. Khởi động Talker và xác nhận cả MoE, MTP subprocess và Code2Wav đã ready.
5. Khởi động Orchestrator sau khi hai backend ready.
6. Poll ba health endpoints: 19999/health, 20000/health, 21000/health; health HTTP 200 chưa đủ nếu log model chưa ready, nên cần cả hai điều kiện.
7. S2 để disabled/local stub trong smoke đầu tiên nếu source cho phép; không để dependency external che khuất lỗi S1 Thinker/Talker.

Pass: ba service ready, không OOM, không worker crash, ghi được peak VRAM từng GPU và thời gian load từng service.

### Phase 5 – Baseline BF16 end-to-end

**Trạng thái: PASS functional trên 4-GPU (2026-10-12).** Cold và warm `simulate_v8.py` đều nhận text/audio; số liệu chi tiết ở mục kết quả baseline phía trên.

Mục tiêu: chạy request thật trước realtime và quantization.

1. Dùng một WAV cố định, mono 24 kHz PCM16; lưu input cùng metadata.
2. Gửi qua simulate_v8.py hoặc client chính thức tới Orchestrator.
3. Chạy một cold run sau startup và tối thiểu một warm run.
4. Kiểm tra text response, audio output không rỗng, sample rate/format và session cleanup.
5. Lưu request id, logs, output WAV, latency timestamps, GPU metrics và exit status.

Pass: request end-to-end thành công với output text + audio; có thể tái hiện bằng cùng input.

### Phase 6 – Realtime và interruption

Chỉ làm sau Phase 5. Chạy WebSocket bridge/client, kiểm tra streaming audio, session lifecycle và user interruption. Tách lỗi microphone/network khỏi lỗi model bằng cách giữ lại một test WAV offline làm regression test.

### Phase 7 – Benchmark và quantization

1. Baseline BF16: model-load time, first-token/first-audio latency, end-to-end latency, RTF, peak VRAM, GPU utilization và cost/hour.
2. Warm-up trước khi đo chính thức vì model card cảnh báo first packet có thể chậm nếu chưa warm-up.
3. Quantization theo thứ tự: Thinker FP8 + Talker BF16, kiểm tra chất lượng, sau đó mới thử thay đổi TP/GPU.
4. Không kết luận cấu hình ít GPU chạy được chỉ vì model load; phải đạt cả output đúng và RTF mục tiêu.
5. Giữ cùng checkpoint revision, input, sampling config và số lần lặp khi so sánh.

### Các file/job cần tạo hoặc cập nhật

- Phase_2_plan/smoke_test_modal.py: chỉ giữ vai trò build/import smoke và pin commit.
- Phase_2_plan/download_model_modal.py: download/resume/verify checkpoint vào Volume.
- Phase_2_plan/model_load_smoke_modal.py: import, start services, health polling và log model load.
- Phase_2_plan/baseline_bf16_modal.py: full-stack request với WAV cố định và metrics.
- docs/duplexomni_next_steps_research.md: findings và nguồn tham khảo.
- Phase_2_plan/duplexomni_inference_plan.md: checklist trạng thái, acceptance criteria và kết quả benchmark.

Không sửa start_thinker_talker_stack.sh để giảm GPU trước khi baseline chính thức chạy; nếu cần cấu hình TP nhỏ hơn, tạo một experiment config riêng để giữ nguyên baseline.

### Ma trận quyết định sau mỗi phase

| Phase | Điều kiện pass | Nếu fail |
|---|---|---|
| Artifact | Path/config/shards hợp lệ | Dừng, không load GPU |
| Download | Manifest + commit volume thành công | Resume download |
| Import | Server/vLLM import pass | Sửa runtime image |
| Model load | 3 service ready, không OOM | Phân tích log/VRAM |
| E2E | Text + audio output | Debug request/session |
| Realtime | Stream + interruption pass | Cô lập WebSocket |
| Quantization | Chất lượng/RTF không giảm ngoài ngưỡng | Giữ BF16 baseline |

### Nguồn chính đã đối chiếu

- https://github.com/MuyeHuang/DuplexOmni/blob/main/INSTALL.md
- https://github.com/MuyeHuang/DuplexOmni/blob/main/inference_framework/README.md
- https://raw.githubusercontent.com/MuyeHuang/DuplexOmni/main/inference_framework/realtime_serving/README.md
- https://raw.githubusercontent.com/MuyeHuang/DuplexOmni/main/inference_framework/realtime_serving/start_thinker_talker_stack.sh
- https://huggingface.co/MuyeHuang/DuplexOmni
- https://huggingface.co/docs/huggingface_hub/en/guides/download
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/timeouts

---

## Task 1 – Hiểu cách inference chính thức của DuplexOmni

### Việc cần làm
- Đọc thư mục `inference_framework/realtime_serving`.
- Xác định các thành phần:
  - Thinker
  - Talker
  - Orchestrator
  - Client / WebSocket server
- Xác định:
  - file nào start Thinker
  - file nào start Talker
  - file nào start Orchestrator
  - file nào dùng để gửi request test
- Xem repo yêu cầu:
  - Python version
  - CUDA version
  - PyTorch
  - modified vLLM
  - GPU / VRAM

### Output
Một sơ đồ ngắn:

```text
Client
  ↓
Orchestrator
  ↓
Thinker + Talker
  ↓
Text / Speech output
```

---

## Task 2 – Chuẩn bị GPU để test

### Việc cần làm
- Tìm GPU cloud phù hợp:
  - Modal
  - Vast.ai
- Ưu tiên dùng free credit / test credit trước để:
  - tạo instance
  - SSH vào server
  - kiểm tra CUDA
  - clone repo
  - thử cài environment
- Sau khi xác nhận setup ổn mới thuê GPU mạnh hơn.

### Các thông số cần ghi lại

```text
GPU model:
VRAM/GPU:
Số GPU:
CPU:
RAM:
Disk:
Interconnect:
Giá/giờ:
```

---

## Task 3 – Setup server GPU

### Mô hình làm việc

```text
Laptop
   ↓ SSH / VS Code Remote
GPU Server
   ├── source code
   ├── model weights
   ├── Python environment
   ├── CUDA
   ├── modified vLLM
   └── inference process
```

Laptop chỉ dùng để điều khiển, sửa code và gửi request.

### Việc cần làm
1. Tạo GPU instance.
2. SSH vào server.
3. Clone repo DuplexOmni.
4. Tạo Python environment.
5. Cài dependency.
6. Cài / build modified vLLM của DuplexOmni.
7. Tải model checkpoint lên server.

---

## Task 4 – Chạy baseline BF16

### Mục tiêu
Chạy model nguyên bản trước, chưa quantize và chưa optimize.

### Cấu hình

```text
Precision: BF16
Runtime: official modified vLLM
Code: official DuplexOmni inference stack
```

### Luồng chạy dự kiến

```text
cd inference_framework/realtime_serving

export VLLM_ROOT=../vllm_qwen3_omni
export THINKER_MODEL=/path/to/thinker
export TALKER_MODEL=/path/to/talker

./start_thinker_talker_stack.sh
```

Sau đó chạy script test/simulation của repo.

### Mục tiêu hoàn thành

```text
Model load thành công
→ không OOM
→ inference chạy
→ sinh được text/audio output
```

---

## Task 5 – Test realtime inference

Sau khi baseline chạy được, test full realtime flow.

```text
Laptop microphone/audio
        ↓
WebSocket / API
        ↓
GPU Server
        ↓
DuplexOmni
        ↓
Streaming text + speech
        ↓
Laptop
```

### Test
- Audio input.
- Streaming speech output.
- User interruption.
- Continuous conversation.
- Thinking layer chạy song song nếu cấu hình có S2.

---

## Task 6 – Benchmark baseline

Ghi lại kết quả của BF16 trước khi tối ưu.

```text
GPU:
Số GPU:
Peak VRAM:
GPU utilization:
Model load time:
First-token latency:
First-audio latency:
End-to-end latency:
RTF:
Chi phí/giờ:
```

### RTF

```text
RTF = thời gian generate audio / thời lượng audio sinh ra
```

Mục tiêu realtime:

```text
RTF < 1
```

---

## Task 7 – Research quantization

Chỉ làm sau khi BF16 baseline chạy được.

### Thứ tự thử

```text
BF16 baseline
     ↓
FP8
     ↓
INT8 / AWQ / GPTQ nếu custom runtime support
     ↓
INT4 nếu khả thi
```

### Ưu tiên đầu tiên

Không quantize toàn bộ model ngay.

Thử:

```text
Thinker → FP8
Talker  → BF16
```

Sau đó mới nghiên cứu quantize thêm các phần khác.

### Những câu cần xác minh
- Custom `vllm_qwen3_omni` có hỗ trợ FP8 không?
- Có hỗ trợ AWQ/GPTQ không?
- Thinker và Talker có thể dùng precision khác nhau không?
- MTP / speech codec có chạy được khi quantize không?
- Quantization có ảnh hưởng chất lượng speech không?

---

## Task 8 – Benchmark sau quantization

So sánh trực tiếp với baseline.

| Metric | BF16 | FP8 / Quantized |
|---|---:|---:|
| Peak VRAM | ? | ? |
| GPU count | ? | ? |
| First audio latency | ? | ? |
| RTF | ? | ? |
| Speech quality | ? | ? |
| Stability | ? | ? |
| Cost/hour | ? | ? |

Mục tiêu là xác định quantization có thực sự giúp giảm GPU/cost mà vẫn giữ realtime hay không.

---

## Task 9 – Tìm cấu hình GPU tối thiểu

Sau khi model chạy ổn:

```text
Cấu hình mạnh chạy được
        ↓
Giảm số GPU
        ↓
Thử GPU khác
        ↓
Kết hợp quantization
        ↓
Benchmark lại
```

### Cần trả lời
- Cấu hình tối thiểu để model load được là gì?
- Cấu hình tối thiểu để giữ `RTF < 1` là gì?
- Quantization giảm được bao nhiêu VRAM?
- Có thể chạy ít GPU hơn không?
- GPU nào có cost/performance tốt nhất?

---

## Task 10 – Report mentor

Report cuối cần trả lời 4 câu chính:

```text
1. DuplexOmni inference bằng cách nào?
2. Cần GPU gì và tối thiểu bao nhiêu GPU?
3. Quantization nào chạy được?
4. Cấu hình nào có cost/performance tốt nhất?
```

Có thể kèm bảng:

| Config | GPU | Precision | VRAM | RTF | Latency | Cost |
|---|---|---|---:|---:|---:|---:|
| Baseline | ? | BF16 | ? | ? | ? | ? |
| Exp 1 | ? | FP8 | ? | ? | ? | ? |
| Exp 2 | ? | Other | ? | ? | ? | ? |

---

# Thứ tự thực hiện

```text
1. Đọc inference framework
        ↓
2. Xác định requirement
        ↓
3. Test GPU cloud bằng free/test credit
        ↓
4. Setup server
        ↓
5. Chạy DuplexOmni BF16 baseline
        ↓
6. Test realtime
        ↓
7. Benchmark baseline
        ↓
8. Research + thử quantization
        ↓
9. Benchmark lại
        ↓
10. Tìm cấu hình GPU tối thiểu
        ↓
11. Tổng hợp cost/performance
        ↓
12. Report mentor
```

---

# Task cần làm NGAY (ĐÃ XÁC MINH CHI TIẾT TỪ CODEBASE)

Đã hoàn thành phân tích mã nguồn chính thức tại `DuplexOmni/inference_framework`:

```text
DuplexOmni/
└── inference_framework/
    ├── realtime_serving/
    │   ├── start_thinker_talker_stack.sh   # Master script khởi động cả 3 microservices
    │   ├── stop_thinker_talker_stack.sh    # Dừng an toàn các services
    │   ├── serving_core/
    │   │   ├── server_thinker.py          # Start Thinker (FastAPI + AsyncLLM v1, port 19999)
    │   │   ├── server_talker.py           # Start Talker (FastAPI + LLM + MTP Subprocess, port 20000)
    │   │   └── server_orchestrator.py     # Start Orchestrator (FastAPI + WS routing, port 21000)
    │   ├── simulate_v8.py                 # Client mô phỏng tự động dùng file WAV (Quantitative benchmark)
    │   ├── omni_realtime_server.py        # Realtime WebSocket Bridge (port 8765)
    │   └── omni_realtime_mac_client.py    # Client thu âm trực tiếp từ microphone
    └── vllm_qwen3_omni/                   # Modified vLLM v1 runtime fork (hỗ trợ Qwen3-Omni MoE & MTP)
```

### Các thông số cốt lõi đã xác định:
1. **Thinker**: Chạy file `serving_core/server_thinker.py` tại port `19999`. Tensor Parallel mặc định là `TP=4` trên GPU `0,1,2,3`. Quản lý bộ nhớ qua `THINKER_GPU_MEM_UTIL=0.8`.
2. **Talker**: Chạy file `serving_core/server_talker.py` tại port `20000`. Tensor Parallel chính là `TP=4` (`--tp 4`) trên GPU `4,5,6,7`. Đi kèm tiến trình con MTP trên GPU `7` (`TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES=7`, `TALKER_MTP_GPU_MEM_UTIL=0.12`).
3. **Orchestrator**: Chạy file `serving_core/server_orchestrator.py` tại port `21000`. Điều phối chia frame 480ms và routing.
4. **Client Benchmark**:
   - `simulate_v8.py`: Dùng file WAV mẫu gửi qua HTTP/WS tới Orchestrator để đo TTFT, Latency và RTF tự động.
   - `omni_realtime_server.py` + `omni_realtime_mac_client.py`: Dùng WebSocket port `8765` để demo micro nói và ngắt lời.
5. **GPU Phân bổ mặc định**: **8 GPU** (4 cho Thinker, 4 cho Talker). Tối thiểu có thể dồn về 4 GPU (TP=2) hoặc 2 GPU (TP=1) khi nén bằng FP8.

> **Tài liệu hướng dẫn triển khai, cài đặt và benchmark chi tiết đầy đủ**:
> Xem tại [`duplexomni_inference_guide.md`](./duplexomni_inference_guide.md)

