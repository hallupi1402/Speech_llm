## Part 1: 
- Nghiên cứu kiến trúc DuplexOmni: Phân tích pipeline Thinker, Talker/MTP và Orchestrator cùng vai trò của modified vLLM.
- Thiết lập hạ tầng trên nền tảng Modal: Tạo Workspace, Volume, CUDA image 
- Chuẩn bị source và model: Clone source DuplexOmni, lưu trên Volume, tải và kiểm tra pretrained checkpoint BF16.
- Build và validate vLLM: Biên dịch các module C++/CUDA cho A100 sm80, cài modified vLLM và kiểm tra import các service inference.
- Thực nghiệm inference GPU: Đã chạy được model-load và BF16 inference bằng file WAV trên 4×A100 ; cấu hình 8×A100 đã chuẩn bị cho bước tiếp theo.
  
- Deployment code: https://github.com/hallupi1402/Speech_llm

