# DuplexOmni: Research cho các bước inference tiếp theo

Ngày research: 2026-10-11

## Kết luận ngắn

Smoke test hiện tại mới xác nhận modified vLLM build/import thành công trên Modal. Bước kế tiếp phải là tải và xác minh model artifact, sau đó chạy model-load smoke trên multi-GPU rồi mới chạy baseline BF16 end-to-end.

Inference source yêu cầu các path model riêng cho Thinker và Talker, dù model card nói checkpoint phát hành chứa cả hai thành phần. Vì vậy không được tự giả định rằng `THINKER_MODEL` và `TALKER_MODEL` cùng trỏ vào một path; cần inspect cấu trúc file sau khi download.

## Findings từ source chính thức

### 1. Runtime và serving stack

DuplexOmni tách inference thành modified vLLM fork và `realtime_serving`. README chính thức liệt kê ba service HTTP:

- Thinker: `127.0.0.1:19999/health`
- Talker: `127.0.0.1:20000/health`
- Orchestrator: `127.0.0.1:21000/health`

Text entrypoint là `/v1/chat/completions`; audio streaming entrypoint là `/v1/audio/stream/{session_id}`. Script `simulate_v8.py` là đường chạy mô phỏng có thể dùng để kiểm tra offline trước microphone/WebSocket.

Sources: [realtime serving README](https://raw.githubusercontent.com/MuyeHuang/DuplexOmni/main/inference_framework/realtime_serving/README.md), [inference framework README](https://github.com/MuyeHuang/DuplexOmni/blob/main/inference_framework/README.md).

### 2. GPU layout chính thức

`start_thinker_talker_stack.sh` mặc định đặt Thinker trên GPU 0–3, Talker trên GPU 4–7 với TP=4, và MTP subprocess của Talker trên GPU 7. Đây là cấu hình baseline cần thử trước khi giảm GPU hoặc thay đổi TP.

Source: [official stack launcher](https://raw.githubusercontent.com/MuyeHuang/DuplexOmni/main/inference_framework/realtime_serving/start_thinker_talker_stack.sh).

Model card khuyến nghị ít nhất 8 NVIDIA H20 cho deployment low-latency. Đây là recommendation của upstream, không phải bằng chứng rằng mọi GPU đều bắt buộc phải là H20; cần đo trên GPU Modal thực tế.

Source: [MuyeHuang/DuplexOmni model card](https://huggingface.co/MuyeHuang/DuplexOmni).

### 3. Model artifact

Model card hiện ghi checkpoint là 35B parameters, BF16 và chứa cả Thinker/Talker. Tài liệu nội bộ cũ ghi khoảng 70.5 GB; dung lượng thực tế và cấu trúc shard phải được đo sau download, không dùng một con số cũ làm giả định cấp disk.

Source: [MuyeHuang/DuplexOmni model card](https://huggingface.co/MuyeHuang/DuplexOmni).

`INSTALL.md` cũng ghi model weights/checkpoints không nằm trong source repository. Do đó source clone và model download phải là hai bước độc lập.

Source: [DuplexOmni INSTALL.md](https://github.com/MuyeHuang/DuplexOmni/blob/main/INSTALL.md).

### 4. Download và persistence

Hugging Face khuyến nghị `snapshot_download` cho toàn bộ repository; có thể chỉ định `revision`, `local_dir`, `allow_patterns` hoặc `ignore_patterns`. Khi dùng `local_dir`, thư mục sẽ giữ cấu trúc file của repository và có metadata `.cache/huggingface` để hỗ trợ lần chạy sau.

Source: [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download).

Weights nên được tải vào Modal Volume, không vào image. Modal docs hỗ trợ mount Volume vào Function và gọi `volume.commit()` sau khi ghi dữ liệu. Volume CLI cũng có các thao tác `ls`, `put`, `get` và `rm` để kiểm tra artifact.

Source: [Modal Volumes](https://modal.com/docs/guide/volumes).

### 5. GPU và timeout trên Modal

Modal hỗ trợ request nhiều GPU trong một container bằng dạng như `A100-80GB:4` hoặc `H100:8`; các GPU được gắn trên cùng máy vật lý. Đây phù hợp với layout Thinker/Talker của launcher hơn việc chạy hai Function độc lập.

Modal Function timeout mặc định là 5 phút và có thể tăng tối đa 24 giờ. `startup_timeout` nên tách riêng cho giai đoạn import/load model nếu container startup dài.

Sources: [Modal GPU configuration](https://modal.com/docs/guide/gpu), [Modal Functions](https://modal.com/docs/guide/functions), [Modal timeouts](https://modal.com/docs/guide/timeouts).

## Plan triển khai được đề xuất

### Phase 0: reproducibility

1. Pin DuplexOmni commit thay vì clone floating `main`.
2. In source commit, model revision, image versions, GPU layout và timestamp vào manifest.
3. Reuse modified vLLM artifact đã build trong `duplexomni-smoke-vol-v2`; không compile lại trong job model-load.

Pass condition: container mới đọc được source/vLLM/manifest từ Volume.

### Phase 1: artifact inspection

1. Download hoặc inspect metadata của `MuyeHuang/DuplexOmni`.
2. Xác định config, processor, safetensors shards và path mapping cho Thinker/Talker.
3. Đo bytes thực tế và dung lượng còn lại trong Volume.

Pass condition: có path thật cho `THINKER_MODEL`, `TALKER_MODEL`, `S1_MODEL_NAME`; không dùng placeholder.

### Phase 2: persistent download job

1. Tạo job Modal riêng mount `/root/workspace`.
2. Dùng `snapshot_download(..., local_dir=..., revision=...)`.
3. In progress theo file/bytes và lưu manifest.
4. Verify file config và shard trước `volume.commit()`.
5. Dùng timeout dài riêng cho download; nếu cần token thì truyền qua Modal Secret.

Pass condition: retry có thể resume, Volume có checkpoint hoàn chỉnh và manifest.

### Phase 3: inference runtime image

1. Giữ Python 3.11, CUDA 12.8, Torch 2.9.1+cu128 tương thích build.
2. Bổ sung dependency realtime theo dependency map: FastAPI, Uvicorn, WebSockets, Requests, Transformers và audio dependencies cần thiết.
3. Set `VLLM_ROOT` tới modified fork trong Volume.
4. Ghi log Thinker/Talker/Orchestrator vào stdout và log files trong Volume.

Pass condition: import server modules và vLLM pass trước khi load model.

### Phase 4: model-load smoke

1. Dùng multi-GPU cùng node; bắt đầu với layout 8 GPU/TP=4 theo launcher upstream.
2. Khởi động Thinker, ghi mốc config → weight load → CUDA graphs → ready.
3. Khởi động Talker, xác nhận MoE, MTP subprocess và Code2Wav ready.
4. Khởi động Orchestrator sau hai backend.
5. Poll ba health endpoints và đồng thời kiểm tra log ready; HTTP 200 riêng lẻ chưa đủ.
6. Ghi peak VRAM, load time và lỗi worker/OOM.

Pass condition: ba service ready, không OOM/crash, path model đúng.

### Phase 5: BF16 end-to-end

1. Chuẩn hóa một WAV mono 24 kHz PCM16 cố định.
2. Chạy `simulate_v8.py` qua Orchestrator.
3. Chạy cold run và warm run.
4. Kiểm tra text response, audio output không rỗng, sample rate và cleanup session.

Pass condition: cùng input tái hiện được text + audio output.

### Phase 6: realtime và benchmark

1. Sau khi offline E2E pass mới chạy WebSocket bridge/client.
2. Kiểm tra streaming, interruption và session lifecycle.
3. Warm-up trước benchmark vì model card cảnh báo first packet có thể chậm nếu chưa warm-up.
4. Ghi model-load time, first-audio latency, end-to-end latency, RTF, peak VRAM, GPU utilization và cost/hour.

### Phase 7: quantization

Chỉ bắt đầu sau baseline BF16. Thứ tự thử:

1. Thinker FP8 + Talker BF16.
2. So sánh chất lượng audio và latency với BF16.
3. Sau đó mới thử giảm TP/GPU.

Không coi model load thành công là đủ; cấu hình chỉ được chấp nhận khi output đúng, ổn định và đạt mục tiêu RTF.

## Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Clone `main` thay đổi giữa các lần chạy | Pin commit và ghi manifest |
| Model path không giống dự kiến | Inspect file tree/config trước khi start |
| Download timeout | Tách download job, tăng timeout, dùng Volume resume |
| Model load OOM | Ghi peak VRAM, điều chỉnh GPU layout/TP sau baseline |
| Health 200 nhưng backend chưa ready | Kết hợp health polling với log readiness |
| Realtime lỗi do WebSocket/microphone | Giữ test WAV offline làm regression path |
| Quantization làm giảm speech quality | Chỉ benchmark sau BF16, giữ cùng input/config |

## Trạng thái sau khi triển khai

Research note này mô tả kế hoạch trước khi chạy model. Đến 2026-10-12, checkpoint, model-load smoke 4-GPU và baseline BF16 E2E đã hoàn thành; xem `Phase_2_plan/duplexomni_inference_plan.md` và `docs/duplexomni_modal_smoke_test_debug.md` để biết số liệu thực tế.

## Phạm vi của bước kế tiếp ngay lập tức

Checkpoint, model-load smoke 4-GPU và baseline BF16 E2E đã hoàn thành. Việc tiếp theo là benchmark có kiểm soát, sau đó realtime interruption và quantization. Không dùng lại `smoke_test_modal.py` cho download hoặc inference; file đó chỉ dành cho build/import modified vLLM.
