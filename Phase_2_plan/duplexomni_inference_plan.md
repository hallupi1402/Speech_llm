# DuplexOmni Inference Research Plan

## Mục tiêu

Research và thử nghiệm cách chạy inference DuplexOmni, từ việc dựng môi trường GPU, chạy baseline BF16, test realtime, đến quantization và benchmark để tìm cấu hình GPU hợp lý nhất.

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

