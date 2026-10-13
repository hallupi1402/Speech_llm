# DuplexOmni: Sổ Tay Kỹ Thuật & Hướng Dẫn Thực Thi Inference, Benchmark & Quantization (Phase 2)

> **Tài liệu kỹ thuật nội bộ dành cho Seminar & Nghiên cứu Hệ thống Multimodal Full-Duplex**  
> **Dự án**: DuplexOmni Inference & Hardware Optimization  
> **Model gốc**: `MuyeHuang/DuplexOmni` (Qwen3-Omni MoE Architecture)  
> **Cập nhật ngày**: 2026-09-29  

---

## MỤC LỤC
1. [Tổng Quan Kiến Trúc & Dữ Kiện Kỹ Thuật Cốt Lõi](#1-tổng-quan-kiến-trúc--dữ-kiện-kỹ-thuật-cốt-lõi)
2. [Chiến Lược Hạ Tầng & Phân Bổ Ngân Sách GPU](#2-chiến-lược-hạ-tầng--phân-bổ-ngân-sách-gpu)
3. [Phân Tích Chi Tiết 3 Tiến Trình Phục Vụ (Serving Stack)](#3-phân-tích-chi-tiết-3-tiến-trình-phục-vụ-serving-stack)
4. [Hướng Dẫn Cài Đặt Môi Trường Từng Bước (Runbook)](#4-hướng-dẫn-cài-đặt-môi-trường-từng-bước-runbook)
5. [Quy Trình Chạy Baseline BF16 (Official Stack)](#5-quy-trình-chạy-baseline-bf16-official-stack)
6. [Phương Pháp Đo Đạc Benchmark (Quantitative & Qualitative)](#6-phương-pháp-đo-đạc-benchmark-quantitative--qualitative)
7. [Nghiên Cứu & Triển Khai Quantization (FP8 Strategy)](#7-nghiên-cứu--triển-khai-quantization-fp8-strategy)
8. [Chiến Lược Tối Thiểu Hóa GPU (Finding the Breaking Point)](#8-chiến-lược-tối-thiểu-hóa-gpu-finding-the-breaking-point)
9. [Kịch Bản Báo Cáo Mentor & Bảng Tổng Hợp Kết Quả](#9-kịch-bản-báo-cáo-mentor--bảng-tổng-hợp-kết-quả)
10. [Troubleshooting & Xử Lý Sự Cố Thường Gặp](#10-troubleshooting--xử-lý-sự-cố-thường-gặp)

---

## 1. TỔNG QUAN KIẾN TRÚC & DỮ KIỆN KỸ THUẬT CỐT LÕI

DuplexOmni là hệ thống AI đa phương thức tương tác toàn song công thời gian thực (real-time full-duplex multimodal interaction), cho phép nghe (listen), nhìn (see), suy nghĩ (think) và nói (speak) đồng thời.

```
                    ┌──────────────────────────────────────────────┐
                    │               Client Interface               │
                    │   (WebSocket Audio Stream 24kHz / Text API)  │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │          S1 Orchestrator (Port 21000)         │
                    │   Điều phối luồng âm thanh, cắt chunk 480ms,  │
                    │   quản lý session, interrupter & S2 bridge   │
                    └──────────────┬───────────────────────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
┌──────────────────────────────┐    ┌───────────────────────────────────┐
│     Thinker (Port 19999)     │    │        Talker (Port 20000)        │
│ • Architecture: MoE          │    │ • Architecture: MoE               │
│ • 48 Layers, 128 Experts     │    │ • 20 Layers, 128 Experts          │
│ • Nhận streaming audio/text  │───►│ • Code Predictor (MTP 5 layers)   │
│ • Trích xuất hidden states   │    │ • Code2Wav Vocoder (Mimi codec)   │
│ • Tensor Parallel: TP=4      │    │ • Tensor Parallel: TP=4 + MTP=1   │
└──────────────────────────────┘    └───────────────────────────────────┘
                │
                ▼ (Optional Pluggable Background Path)
┌──────────────────────────────┐
│  S2 Thinking LLM (External)  │
│ • DeepSeek-R1 / Qwen-Max     │
│ • Suy luận logic chuyên sâu  │
└──────────────────────────────┘
```

### Thông Số Mô Hình (Verified Facts từ Hugging Face & Repo):
* **Model ID**: `MuyeHuang/DuplexOmni`
* **Dung lượng Checkpoint BF16**: **~70.5 GB** (gồm 15 file shards safetensors).
* **Kiến trúc Thinker**:
  - Model type: `qwen3_omni_moe_thinker`
  - 48 hidden layers, hidden_size = 2048, 128 experts (top-k = 6 hoặc 8).
* **Kiến trúc Talker**:
  - Model type: `qwen3_omni_moe_talker`
  - 20 hidden layers, hidden_size = 1024, 128 experts, intermediate_size = 2048.
  - Sub-network: `code_predictor_config` (MTP - Multi-Token Prediction 5 layers).
  - Sub-network: `code2wav_config` (Mimi vocoder biến codec token thành sóng âm 24kHz).
* **Mã nguồn Inference**:
  - Tọa lạc tại: `inference_framework/realtime_serving`
  - Runtime custom: `inference_framework/vllm_qwen3_omni` (vLLM v1 fork patched riêng cho Qwen3-Omni MoE).

---

## 2. CHIẾN LƯỢC HẠ TẦNG & PHÂN BỔ NGÂN SÁCH GPU

Do mô hình có dung lượng nguyên bản ~70.5 GB BF16 và chia làm 2 tiến trình vLLM riêng biệt, việc cấp phát GPU được tối ưu theo 2 pha rõ rệt nhằm tránh lãng phí ngân sách:

### Pha 1: Smoke Test & Đóng gói Môi trường (Low-cost / Free Tier)
* **Mục tiêu**: Verify toàn bộ quá trình clone repo, cài đặt dependencies, biên dịch modified vLLM C++/CUDA extension, kiểm tra import và CUDA graphs.
* **Hạ tầng khuyến nghị**: 
  - 1x NVIDIA GPU phổ thông (RTX 4090 24GB, RTX 3090, A10G, hoặc T4 trên Modal / Vast.ai / RunPod).
  - Tận dụng $30 free credit của Modal hoặc credit thử nghiệm của Vast.ai.
  - Lưu ý: Ở pha này **không tải checkpoint 70GB**, chỉ chạy unit test môi trường (`python -c "import vllm"`).

### Pha 2: Multi-GPU Baseline & Benchmark (Full Scale)
* **Mục tiêu**: Tải trọn bộ checkpoint 70.5GB, chạy full serving stack Thinker + Talker + Orchestrator, đo đạc RTF và thử nghiệm quantization.
* **Hạ tầng chuẩn Paper (Upper-bound Baseline)**:
  - **8x NVIDIA GPU**: 8x L40S (48GB VRAM/GPU) hoặc 8x A100 80GB / H100 SXM trên Vast.ai / RunPod / Lambda Labs.
  - CPU: >= 32 vCPU, RAM hệ thống: >= 128 GB, Ổ cứng SSD/NVMe: >= 250 GB (đủ lưu repo + 70GB weights + logs).
* **Hạ tầng Tối ưu Chi phí (Cost-effective Target)**:
  - **4x NVIDIA A100 80GB** (Dồn Thinker TP=2 trên GPU 0,1; Talker TP=2 trên GPU 2,3).
  - Hoặc **4x NVIDIA L40S 48GB** khi áp dụng FP8 cho Thinker.

---

## 3. PHÂN TÍCH CHI TIẾT 3 TIẾN TRÌNH PHỤC VỤ (SERVING STACK)

Hệ thống serving của DuplexOmni không phải là một monolithic process mà được chia thành 3 service microservices giao tiếp qua REST/WebSocket nội bộ (localhost):

### 3.1. Thinker Server (`server_thinker.py`)
* **Cổng mặc định**: `http://127.0.0.1:19999`
* **Health Check**: `GET /health`
* **Nhiệm vụ**:
  - Nhận luồng âm thanh đầu vào từ Orchestrator (PCM 16-bit 24kHz chuyển đổi sang audio features).
  - Xử lý biểu diễn ngữ nghĩa đa phương thức, sinh text token và trích xuất các vector trạng thái ẩn (`top_hidden_state`, `text_embedding`).
  - Gửi các hidden states này sang cho Talker Server.
* **Cấu hình GPU**:
  - Mặc định: `CUDA_VISIBLE_DEVICES=0,1,2,3` (TP=4).
  - Biến môi trường: `THINKER_PORT=19999`, `THINKER_GPU_MEM_UTIL=0.8`, `THINKER_DTYPE=auto`.

### 3.2. Talker Server (`server_talker.py`)
* **Cổng mặc định**: `http://127.0.0.1:20000`
* **Health Check**: `GET /health`
* **Nhiệm vụ**:
  - Nhận hidden states và text prompt từ Thinker.
  - Sử dụng Talker MoE kết hợp Code Predictor (MTP - Multi-Token Prediction) để dự đoán chuỗi codec audio (Mimi acoustic tokens).
  - Sử dụng Code2Wav vocoder để giải mã acoustic tokens thành dạng sóng âm thanh (raw audio waveforms).
* **Cấu hình GPU**:
  - Mặc định: `CUDA_VISIBLE_DEVICES=4,5,6,7` (TP=4).
  - Tiến trình con MTP: `TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES=7` (chạy trên GPU vật lý số 7 với `TALKER_MTP_GPU_MEM_UTIL=0.12`).
  - Biến môi trường: `TALKER_TP=4`, `TALKER_GPU_MEM_UTIL=0.6`.

### 3.3. Orchestrator Server (`server_orchestrator.py`)
* **Cổng mặc định**: `http://127.0.0.1:21000`
* **API Endpoints**:
  - Text Completions: `http://127.0.0.1:21000/v1/chat/completions`
  - Real-time Audio Stream: `ws://127.0.0.1:21000/v1/audio/stream/{session_id}`
* **Nhiệm vụ**:
  - Tiếp nhận kết nối từ Client qua WebSocket.
  - Chia cắt âm thanh đầu vào thành các frame 480ms liên tục.
  - Phát hiện ngắt lời (user interruption) để gửi tín hiệu xóa bộ đệm (cancel turn) sang Talker ngay lập tức.
  - Ghép nối các audio chunks trả về và stream ngược lời nói của Agent về phía Client.

---

## 4. HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG TỪNG BƯỚC (RUNBOOK)

Thực hiện toàn bộ các lệnh sau trên GPU Server (thông qua SSH):

### Bước 1: Chuẩn bị Thư mục và Clone Mã Nguồn
```bash
# Cập nhật package hệ thống
sudo apt-get update && sudo apt-get install -y git git-lfs build-essential cmake ninja-build ffmpeg libsndfile1

# Clone repo chính thức DuplexOmni
cd /workspace
git clone https://github.com/MuyeHuang/DuplexOmni.git
cd DuplexOmni
```

### Bước 2: Tạo Python Virtual Environment cho vLLM Runtime
```bash
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -U pip setuptools wheel
```

### Bước 3: Cài đặt và Biên dịch `vllm_qwen3_omni`
```bash
cd inference_framework/vllm_qwen3_omni

# Cài đặt các gói build dependencies
pip install -r requirements/build.txt

# Cài đặt Torch và CUDA runtime tương thích
pip install -r requirements/cuda.txt

# Biên dịch vLLM custom từ mã nguồn (Editable mode)
# Quá trình này mất khoảng 5 - 15 phút tùy CPU
pip install -e .

# Kiểm tra xác nhận vLLM đã cài đặt thành công
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

### Bước 4: Cài đặt Thư viện cho Realtime Serving
```bash
cd ../realtime_serving
pip install fastapi uvicorn websockets openai librosa soundfile scipy
```

### Bước 5: Tải Checkpoint Mô Hình (Chỉ thực hiện trên Multi-GPU Server)
```bash
# Tạo thư mục chứa models
mkdir -p /workspace/models
cd /workspace/models

# Cài huggingface_hub cli và tải weights (~70.5 GB)
pip install huggingface_hub
huggingface-cli download MuyeHuang/DuplexOmni --local-dir /workspace/models/DuplexOmni --local-dir-use-symlinks False
```

---

## 5. QUY TRÌNH CHẠY BASELINE BF16 (OFFICIAL STACK)

### Bước 1: Cấu hình Biến Môi Trường Chạy Dịch Vụ
Tạo file script cấu hình hoặc export trực tiếp:

```bash
cd /workspace/DuplexOmni/inference_framework/realtime_serving
source ../../.venv-vllm/bin/activate

export VLLM_ROOT="$(pwd)/../vllm_qwen3_omni"
export PYTHON_BIN="$(which python)"

# Đường dẫn model weights đã tải
export THINKER_MODEL="/workspace/models/DuplexOmni"
export TALKER_MODEL="/workspace/models/DuplexOmni"
export S1_MODEL_NAME="qwen3-omni"

# Cô lập S1 (Tắt S2 thinking layer để benchmark thuần inference nội bộ)
export S2_THINK_BASE_URL="http://127.0.0.1:8000/v1"
export S2_MODEL_NAME="disabled"
export S2_API_KEY="EMPTY"
```

### Bước 2: Khởi động Cụm Thinker - Talker - Orchestrator
```bash
chmod +x start_thinker_talker_stack.sh stop_thinker_talker_stack.sh
./start_thinker_talker_stack.sh
```

### Bước 3: Kiểm Tra Log Khởi Động & Health Check
Mở terminal riêng hoặc theo dõi log:

```bash
# Kiểm tra log Thinker
tail -f logs/thinker.log

# Kiểm tra log Talker (chờ CUDA graphs capture hoàn tất)
tail -f logs/talker.log

# Kiểm tra log Orchestrator
tail -f logs/orchestrator.log
```

Kiểm tra trạng thái HTTP Health Check:
```bash
curl http://127.0.0.1:19999/health  # Thinker status
curl http://127.0.0.1:20000/health  # Talker status
curl http://127.0.0.1:21000/health  # Orchestrator status
```
*Tất cả phải trả về HTTP 200 OK trước khi tiến hành benchmark.*

---

## 6. PHƯƠNG PHÁP ĐO ĐẠC BENCHMARK (QUANTITATIVE & QUALITATIVE)

Theo quyết định đã thống nhất, quá trình kiểm thử được chia làm 2 giai đoạn:

### Giai Đoạn 6.1: Benchmark Định Lượng Tự Động (Quantitative Benchmark)
Sử dụng script mô phỏng `simulate_v8.py`. Phương pháp này cấp một file âm thanh mẫu (24kHz mono PCM 16-bit) để kích hoạt toàn bộ pipeline mà không bị ảnh hưởng bởi độ trễ Internet hay tạp âm môi trường.

```bash
cd /workspace/DuplexOmni/inference_framework/realtime_serving

# Chuẩn bị file audio test mẫu (hoặc dùng assets/reference_audio/google_Leda.wav)
export TEST_INPUT_AUDIO="../../assets/reference_audio/google_Leda.wav"
export OUTPUT_FILENAME="outputs/baseline_bf16_output.wav"
mkdir -p outputs

# Chạy mô phỏng tương tác
python simulate_v8.py
```

#### Giám sát VRAM & Tải GPU trong lúc Benchmark:
Chạy song song lệnh giám sát trong một tmux pane riêng:
```bash
nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1 > outputs/gpu_metrics_baseline.csv
```

#### Công Thức Tính Các Chỉ Số Hiệu Năng Cốt Lõi:
1. **First-Audio Latency (Độ trễ gói âm thanh đầu tiên)**:
   $$\text{Latency}_{\text{first}} = T_{\text{client\_receive\_audio\_chunk\_0}} - T_{\text{client\_finish\_input\_speech}}$$
   *Mục tiêu chuẩn duplex*: $< 500\text{ ms}$.
2. **Real-Time Factor (RTF)**:
   $$\text{RTF} = \frac{\text{Tổng thời gian sinh audio (Wall-clock Generation Time)}}{\text{Tổng thời lượng file audio sinh ra (Audio Duration)}}$$
   *Điều kiện bắt buộc cho Realtime Serving*: $\mathbf{RTF < 1.0}$ (Lý tưởng: $\text{RTF} \le 0.3 - 0.5$).
3. **Peak VRAM Per GPU**: Lấy giá trị `memory.used` lớn nhất trên từng GPU 0–7 từ file log `gpu_metrics_baseline.csv`.

---

### Giai Đoạn 6.2: Kiểm Thử Định Tính & Demo Trực Tiếp (Qualitative Demo)
Sau khi có bảng số liệu benchmark, triển khai WebSocket Server trên GPU Server và kết nối Micro từ máy cá nhân qua Reverse SSH Tunnel:

1. **Trên GPU Server**:
   ```bash
   python omni_realtime_server.py --host 0.0.0.0 --port 8765
   ```
2. **Trên Laptop Cá Nhân (Mở SSH Tunnel)**:
   ```bash
   ssh -L 8765:127.0.0.1:8765 user@gpu-server-ip
   ```
3. **Chạy Client Thu Âm Trực Tiếp**:
   ```bash
   python omni_realtime_mac_client.py --server ws://127.0.0.1:8765
   ```
4. **Kiểm tra các kịch bản Full-Duplex**:
   * Nói một câu hỏi dài và kiểm tra độ trễ phản hồi.
   * Cố tình nói chen ngang (User Interruption) khi Agent đang phát biểu $\rightarrow$ Kiểm tra xem Agent có lập tức ngừng phát âm thanh trong vòng < 200ms hay không.

---

## 7. NGHIÊN CỨU & TRIỂN KHAI QUANTIZATION (FP8 STRATEGY)

### 7.1. Phân Tích Khả Năng Tương Thích Của Runtime
Kiểm tra mã nguồn `vllm_qwen3_omni`:
* `qwen3_omni_moe_thinker.py` và `qwen3_omni_moe_talker.py` kế thừa từ `SharedFusedMoE`, hỗ trợ cấu hình `quant_config`.
* **Rủi ro kỹ thuật**: Kiến trúc 128 experts kết hợp audio code predictor không tương thích với AWQ/GPTQ out-of-the-box nếu không có custom CUDA kernel cho audio head.
* **Chiến lược tối ưu khả thi nhất**: **FP8 (W8A8)** trên các GPU hỗ trợ phần cứng (Ada Lovelace L40S, RTX 4090, hoặc Hopper H100).

### 7.2. Lộ Trình Triển Khai Quantization Không Xâm Lấn (Non-intrusive)
Không quantize đồng loạt toàn bộ hệ thống ngay. Thực hiện theo chiến lược từng phần:

#### Kịch Bản 1 (Ưu tiên số 1): Thinker $\rightarrow$ FP8, Talker $\rightarrow$ BF16
* **Cơ sở lý luận**:
  - Thinker chiếm 48 layers MoE (khoảng ~50GB trọng số). Nén Thinker về FP8 giúp giảm 50% VRAM của Thinker (từ ~50GB xuống ~25GB).
  - Talker chỉ có 20 layers MoE nhưng mang sub-network Code Predictor và Mimi Vocoder. Giữ nguyên Talker ở BF16 để đảm bảo độ chính xác của âm thanh và tránh méo giọng / rè tiếng.
* **Cách kích hoạt**:
  Điều chỉnh `THINKER_DTYPE` và bổ sung argument quantization trong `server_thinker.py` (hoặc truyền qua biến môi trường):
  ```bash
  export THINKER_DTYPE="fp8"
  # Hoặc cấu hình quantization="fp8" trong AsyncEngineArgs
  ```

#### Kịch Bản 2: Đánh giá Chất lượng Âm thanh Sau Quantization
So sánh file `outputs/baseline_bf16_output.wav` và `outputs/fp8_output.wav`:
* Kiểm tra hiện tượng mất âm (audio clipping), rè tiếng hoặc biến dạng cao độ.
* Đo đạc lại RTF và First-Audio Latency để đánh giá tốc độ sinh FP8 GEMM kernel trên Tensor Cores.

---

## 8. CHIẾN LƯỢC TỐI THIỂU HÓA GPU (FINDING THE BREAKING POINT)

Mục tiêu của Task 9 là tìm cấu hình phần cứng tối thiểu để hệ thống vừa đủ chạy mượt (`RTF < 1`):

```
[8x GPU L40S / A100] (Chuẩn Paper, RTF ~ 0.3)
         │
         ▼ (Hạ số lượng GPU)
[4x GPU A100 80GB] (Thinker TP=2 trên GPU 0,1; Talker TP=2 trên GPU 2,3)
         │
         ▼ (Áp dụng Thinker FP8)
[4x GPU L40S 48GB] (Thinker FP8 chiếm ~25GB VRAM, Talker BF16 chiếm ~20GB VRAM)
         │
         ▼ (Stress-test ép giới hạn)
[2x GPU A100 80GB] (Thinker GPU 0, Talker GPU 1 - Không TP) --> Đo RTF xem có vượt quá 1.0 không!
```

### Các bước điều chỉnh file `start_thinker_talker_stack.sh` để test cấu hình 4 GPU:
```bash
# Thinker dồn về 2 GPU
CUDA_VISIBLE_DEVICES=0,1 \
THINKER_PORT=19999 \
"${PYTHON_BIN}" "${THINKER_SCRIPT}" --tensor-parallel-size 2

# Talker dồn về 2 GPU
CUDA_VISIBLE_DEVICES=2,3 \
TALKER_PORT=20000 \
TALKER_TP=2 \
TALKER_MTP_SUBPROC_CUDA_VISIBLE_DEVICES=3 \
"${PYTHON_BIN}" "${TALKER_SCRIPT}" --tp 2
```

---

## 9. KỊCH BẢN BÁO CÁO MENTOR & BẢNG TỔNG HỢP KẾT QUẢ

Báo cáo kết thúc Phase 2 sẽ trả lời trực diện 4 câu hỏi chiến lược của Mentor thông qua bảng số liệu thực nghiệm:

### Bảng Ma Trận Benchmark Mẫu (Cần điền đầy đủ số liệu sau thực nghiệm):

| Cấu Hình | GPU & Số Lượng | Precision (Thinker/Talker) | Peak VRAM (Thinker) | Peak VRAM (Talker) | First Audio Latency | RTF (Real-Time Factor) | Chi Phí ($/giờ) | Trạng Thái Realtime |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Ref** | 8x A100 80GB | BF16 / BF16 | ~42 GB | ~38 GB | ~380 ms | 0.28 | ~$12.00 | Đạt chuẩn |
| **Exp 1 (L40S)** | 8x L40S 48GB | BF16 / BF16 | ~42 GB | ~38 GB | ~430 ms | 0.35 | ~$7.50 | Đạt chuẩn |
| **Exp 2 (FP8)** | 8x L40S 48GB | **FP8** / BF16 | ~22 GB | ~38 GB | ~390 ms | 0.30 | ~$7.50 | Đạt chuẩn |
| **Exp 3 (4-GPU)**| 4x A100 80GB | BF16 / BF16 | ~65 GB | ~58 GB | ~510 ms | 0.58 | ~$6.00 | Đạt chuẩn |
| **Exp 4 (Min)** | 4x L40S 48GB | **FP8** / BF16 | ~32 GB | ~36 GB | ~550 ms | 0.65 | ~$3.75 | **Tối ưu chi phí nhất** |
| **Exp 5 (Break)**| 2x A100 80GB | BF16 / BF16 | OOM / N/A | OOM / N/A | >1200 ms | 1.15 | ~$3.00 | **Điểm gãy (RTF > 1)** |

### Cấu Trúc Trình Bày Bài Báo Cáo Mentor:
1. **Kiến trúc & Cơ chế Full-Duplex**: Giải thích cách Thinker và Talker phân tách nhiệm vụ và cơ chế ngắt lời thời gian thực qua Orchestrator.
2. **Chi phí & Rào cản Kỹ thuật**: Trình bày dung lượng 70.5GB và lý do tại sao repo gốc đòi hỏi cụm 8 GPU H20/A100.
3. **Hiệu quả của Quantization**: Minh chứng số liệu FP8 giúp giảm tải VRAM cho Thinker mà vẫn bảo toàn 100% chất lượng âm thanh của Talker.
4. **Cấu hình Đề Xuất Tối Ưu**: Đưa ra kết luận cấu hình GPU rẻ nhất có thể đưa vào thực tế (ví dụ: cụm 4x L40S thay vì 8x A100, tiết kiệm ~70% chi phí vận hành).
5. **Live Demo**: Trình diễn ngắt lời trực tiếp qua micro.

---

## 10. TROUBLESHOOTING & XỬ LÝ SỰ CỐ THƯỜNG GẶP

1. **Lỗi "Cannot re-initialize CUDA in forked subprocess"**:
   * *Nguyên nhân*: Python `multiprocessing` mặc định dùng `fork` kế thừa CUDA context đã khởi tạo ở tiến trình cha.
   * *Khắc phục*: Đã được fix trong code repo qua `os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"`. Đảm bảo không xóa cờ này.
2. **Lỗi OOM khi load Talker (Out Of Memory trên GPU 7)**:
   * *Nguyên nhân*: GPU 7 vừa gánh rank 3 của Talker MoE vừa gánh MTP sub-process.
   * *Khắc phục*: Giảm `TALKER_GPU_MEM_UTIL` từ `0.6` xuống `0.55` và `TALKER_MTP_GPU_MEM_UTIL` từ `0.12` xuống `0.08`.
3. **Lỗi vLLM CUDA Extension Compile Thất Bại**:
   * *Nguyên nhân*: Phiên bản CUDA compiler (`nvcc`) không khớp với PyTorch CUDA runtime.
   * *Khắc phục*: Kiểm tra `nvcc --version` và `python -c "import torch; print(torch.version.cuda)"`. Luôn sử dụng CUDA Toolkit 12.1 hoặc 12.4 đồng nhất.
4. **Port In Use (19999 / 20000 / 21000)**:
   * *Khắc phục*: Chạy `./stop_thinker_talker_stack.sh`. Nếu vẫn bị chiếm dụng, chạy:
     ```bash
     fuser -k 19999/tcp 20000/tcp 21000/tcp
     ```
