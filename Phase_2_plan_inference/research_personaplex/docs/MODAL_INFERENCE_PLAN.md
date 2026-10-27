# Kế hoạch chạy inference PersonaPlex trên GPU Modal

## Mục tiêu và giả định

Triển khai NVIDIA PersonaPlex thành dịch vụ hội thoại giọng nói full-duplex trên Modal. Mốc đầu tiên là bản pilot riêng tư, dùng một GPU và cung cấp giao diện trình duyệt cùng WebSocket API của dự án gốc. Kế hoạch giả định đầu vào và đầu ra là tiếng Anh, chạy một bản sao model và có ít phiên hội thoại đồng thời. Cần đo thực tế để xác định mức đồng thời và độ trễ phù hợp cho production.

## Những điểm rút ra từ paper và repository

- PersonaPlex dựa trên Moshi, kết hợp prompt văn bản mô tả vai trò với prompt âm thanh xác định giọng nói. Trong hội thoại, model vừa nhận âm thanh người dùng vừa tạo văn bản và âm thanh phản hồi. Vì vậy, một phiên streaming liên tục phù hợp hơn pipeline chỉ nhận yêu cầu rồi trả kết quả. [Paper](https://arxiv.org/pdf/2602.06053)
- Checkpoint được phát hành là `nvidia/personaplex-7b-v1`. Hugging Face mô tả model có 7 tỷ tham số, định dạng BF16, âm thanh vào/ra 24 kHz; phần cứng dùng để kiểm tra inference là A100 80 GB. Lệnh khởi chạy tương tác tiêu chuẩn trong repository là `python -m moshi.server`; có thể chọn checkpoint bằng `--hf-repo`. [Model card](https://huggingface.co/nvidia/personaplex-7b-v1) · [README của repository](https://github.com/NVIDIA/personaplex)
- Mã nguồn dùng giấy phép MIT, nhưng trọng số model tuân theo NVIDIA Open Model License. Cần chấp thuận điều khoản truy cập model được giới hạn trên Hugging Face trước khi tải. [Thông tin license trong repo](https://github.com/NVIDIA/personaplex#license) · [Truy cập model trên Hugging Face](https://huggingface.co/nvidia/personaplex-7b-v1)
- Web Functions của Modal hỗ trợ WebSocket. Hiện tại, mỗi thông điệp WebSocket bị giới hạn ở 2 MiB và mỗi kết nối tương ứng với một function call. [Modal Web Functions](https://modal.com/docs/guide/webhooks)

## Phương án triển khai đầu tiên

1. **Ưu tiên dùng `moshi.server` của dự án gốc.** Đóng gói project Python trong thư mục `moshi/` cùng thư viện hệ thống Opus vào Modal image. Khởi chạy server trên `0.0.0.0`, dùng port Modal expose và bỏ `--ssl`; Modal xử lý HTTPS/WSS ở proxy. Chỉ định rõ checkpoint bằng `--hf-repo nvidia/personaplex-7b-v1`.
2. **Expose server qua Modal Web Function.** Bọc tiến trình server chạy dài bằng `@modal.web_server(...)`, cấp GPU, đặt thời gian khởi động đủ cho việc nạp model và đặt giới hạn `max_inputs` ban đầu thấp. Web Function chuyển tiếp WebSocket đến server. Bắt đầu với một hội thoại đang hoạt động trên mỗi replica cho đến khi đo được cơ chế khóa phiên và mức dùng bộ nhớ GPU của server gốc.
3. **Dùng A100 80 GB làm cấu hình chuẩn ban đầu.** Cấu hình này khớp với phần cứng inference trong model card và có dư địa để đo bộ nhớ cho activation và trạng thái phiên. Sau đó so sánh với L40S (48 GB) để giảm chi phí. Riêng trọng số BF16 của model 8B chiếm khoảng 16 GB theo cách tính thập phân, chưa tính bộ nhớ runtime; đây chỉ là mức tối thiểu, không đảm bảo GPU 48 GB đáp ứng mục tiêu thông lượng hoặc số phiên. Modal hiện có A100 80 GB và L40S; giá riêng GPU tương ứng khoảng $2.50/giờ và $1.95/giờ, chưa tính CPU, RAM, lưu trữ và network. [Hướng dẫn GPU Modal](https://modal.com/docs/guide/gpu) · [Bảng giá Modal](https://modal.com/pricing)
4. **Lưu cache trọng số bền vững.** Tạo Modal Volume cho cache Hugging Face và mount tại đường dẫn dùng bởi `HF_HOME`/`HUGGINGFACE_HUB_CACHE`. Cấp `HF_TOKEN` qua Modal Secret. Sau khi chấp thuận quyền truy cập, chạy tác vụ tải và warmup một lần, rồi cho function phục vụ đọc checkpoint từ cache. Không đưa token hoặc trọng số gated vào container image. [Modal Volumes](https://modal.com/docs/guide/volumes) · [Modal Secrets](https://modal.com/docs/guide/secrets)
5. **Bảo vệ endpoint và giới hạn chi phí.** Giữ endpoint ở chế độ riêng tư trong pilot; yêu cầu xác thực qua Modal proxy hoặc bổ sung xác thực ở tầng ứng dụng. Đặt giới hạn số replica tối đa và chính sách scale-down. Chỉ giữ sẵn container luôn warm nếu số đo cold start cho thấy việc đó đáng với chi phí. Không để endpoint công khai mà không xác thực vì người khác có thể sử dụng GPU và truy cập phiên âm thanh.

## Các giai đoạn triển khai

### Giai đoạn 0 — chuẩn bị

- Chấp thuận NVIDIA Open Model License trên Hugging Face và xác nhận token của tài khoản tải được các file model.
- Thiết lập Modal workspace/phương thức thanh toán và tạo Modal Secret chứa `HF_TOKEN`.
- Chốt cách tương tác: UI demo gốc, WebSocket client riêng hay chỉ API. Kế hoạch ban đầu giả định dùng UI gốc cùng WebSocket.

### Giai đoạn 1 — triển khai thử trên một GPU

- Cố định commit/tag upstream cùng phiên bản Python/CUDA/PyTorch; không deploy trực tiếp từ nhánh `main` luôn thay đổi.
- Tạo Modal image có thư viện hệ thống Opus cần thiết và cài Moshi từ source repository đã cố định phiên bản.
- Gắn HF cache Volume và HF token Secret; tải checkpoint một lần, chạy `moshi.server` không bật TLS cục bộ, rồi expose port qua Web Function hỗ trợ WebSocket của Modal.
- Kiểm tra health, nạp model, truy cập trình duyệt, kết nối WSS, âm thanh vào/ra 24 kHz, persona prompt, voice prompt và ngắt/kết nối lại phiên.

### Giai đoạn 2 — kiểm tra độ trễ, bộ nhớ và phiên

- Đo riêng cold start và độ trễ khi phiên đã warm. Ghi lại thời gian đến âm thanh đầu tiên, thời gian tạo mỗi frame, phản hồi khi bị ngắt/barge-in, mức dùng GPU cao nhất và lỗi phát sinh.
- Bắt đầu với một phiên rồi tăng dần số phiên đồng thời. Đặt giới hạn replica/phiên dựa trên bộ nhớ và tốc độ xử lý thời gian thực đo được; không mặc định model 7B có thể chia sẻ an toàn một GPU cho nhiều luồng.
- So sánh A100 80 GB và L40S. Giữ A100 nếu GPU ít bộ nhớ hơn bị OOM hoặc không theo kịp tốc độ âm thanh thời gian thực.
- Kiểm tra trạng thái từng phiên có được cô lập không và server có tuần tự hóa inference trên GPU không. Nếu chưa, giữ một phiên đồng thời cho mỗi tiến trình model hoặc bổ sung queue/semaphore trước khi tăng concurrency.

### Giai đoạn 3 — hoàn thiện cho production

- Bổ sung xác thực, chính sách origin/CORS cho browser client, giới hạn request/phiên, timeout, log có cấu trúc và dọn dẹp rõ ràng khi client ngắt kết nối.
- Cố định và quét dependency; dùng Modal image có thể tái tạo và một revision model xác định. Chỉ lưu thông tin xác thực Hugging Face trong Modal Secrets.
- Bổ sung tín hiệu startup/readiness và health check khi deploy. Theo dõi cold start, số container, mức sử dụng/bộ nhớ GPU, độ dài phiên, WebSocket bị ngắt và chi phí mỗi phút hội thoại.
- Dựa trên lưu lượng đo được để chọn giữ container warm hay scale-to-zero. Chỉ scale replica theo chiều ngang sau khi xác nhận mỗi container có tiến trình model và nhóm phiên độc lập.

## Rủi ro và quyết định cần lưu ý

- **Full-duplex cần WebSocket.** Endpoint HTTP đơn giản nhận toàn bộ WAV rồi trả WAV mới sẽ không giữ được khả năng model nghe trực tiếp, xử lý ngắt lời và đồng thời phát lời nói.
- **Cold start có thể làm tăng độ trễ đáng kể.** Dùng Volume bền vững cho checkpoint và warmup; đo riêng hiệu năng cold/warm. Timeout request của Modal Web Function là 150 giây, vì vậy cần xác nhận hành vi phiên dài và lúc ngắt kết nối theo vòng đời kết nối của nền tảng. [Hướng dẫn timeout của Modal](https://modal.com/docs/guide/webhook-timeouts)
- **Lựa chọn GPU cần được kiểm chứng.** A100 80 GB khớp phần cứng đã công bố; L40S là phương án thử nghiệm chi phí/hiệu năng, chưa được chứng minh tương đương cho workload mục tiêu.
- **Cần xem xét cả hai license.** Mã nguồn repository dùng MIT, còn trọng số model tuân theo NVIDIA Open Model License và điều kiện tải có kiểm soát. Hãy rà soát điều khoản cho mục đích sử dụng dự kiến.
- **Model card nêu hỗ trợ tiếng Anh.** Không nên suy ra khả năng đa ngôn ngữ chỉ từ giao diện speech tổng quát.

## Tiêu chí nghiệm thu pilot

- Bản deploy trên Modal khởi động và nạp checkpoint gated từ cache, không tải lại trọng số sau mỗi lần restart.
- Trình duyệt hoặc WebSocket client duy trì hội thoại âm thanh full-duplex và nhận giọng nói tạo ra ở 24 kHz.
- Ít nhất một phiên trực tiếp theo kịp thời gian thực, không phát sinh lỗi hết bộ nhớ GPU.
- Đã bật xác thực; log có đủ dữ liệu để đo thời gian khởi động, độ trễ, tỷ lệ lỗi phiên và bộ nhớ GPU.
- Đã ghi lại chi phí mỗi giờ GPU hoạt động và chi phí mỗi phút hội thoại cho GPU cùng cấu hình scale-down đã chọn.

## Nguồn tham khảo

- [PersonaPlex paper, arXiv:2602.06053](https://arxiv.org/pdf/2602.06053)
- [NVIDIA/personaplex repository](https://github.com/NVIDIA/personaplex)
- [PersonaPlex Hugging Face model card](https://huggingface.co/nvidia/personaplex-7b-v1)
- [Modal GPU acceleration](https://modal.com/docs/guide/gpu)
- [Modal Web Functions and WebSockets](https://modal.com/docs/guide/webhooks)
- [Modal Secrets](https://modal.com/docs/guide/secrets)
- [Modal Volumes](https://modal.com/docs/guide/volumes)
- [Modal pricing](https://modal.com/pricing)
