# Script thuyết trình — PersonaPlex research progress

Thời lượng gợi ý: 10–12 phút. Số liệu về limitation là kết quả paper báo cáo, chưa phải kết quả reproduce của dự án. Các phương pháp P01–P03 là đề xuất, trừ khi slide ghi rõ đã chạy.

## Slide 1 — Từ thử inference đến câu hỏi nghiên cứu

Em sẽ trình bày theo tiến trình công việc: inference hai model, đọc kiến trúc PersonaPlex, khảo sát literature để tìm research gap, rồi xây phương án cho ba vấn đề. Hiện em tập trung kiểm chứng P03. Em sẽ phân biệt kết quả paper, phần em đã chạy và phương án còn đang đề xuất.

## Slide 2 — DuplexOmni: thử nghiệm và giới hạn compute

Em bắt đầu với DuplexOmni để hiểu pipeline full-duplex và cách triển khai inference. Em đã thử model load và BF16 inference từ WAV trên 4 GPU A100, nhận transcript và audio đầu ra. Cấu hình realtime mục tiêu lớn hơn, cần 8×A100-80GB và modified vLLM. Realtime microphone transcript/playback chưa được xác nhận ổn định. Do chi phí compute, em không dùng DuplexOmni cho các thí nghiệm tiếp theo; phần này chủ yếu giúp em hiểu hạ tầng và code inference.

## Slide 3 — Các bước thử inference DuplexOmni

Quy trình em đi theo là: pin source và checkpoint trong Modal Volume; kiểm tra artifact; dựng CUDA/PyTorch runtime phù hợp A100; build modified vLLM cho sm80; sau đó chạy import và model-load smoke checks. Khi runtime hoạt động, em chạy BF16 WAV baseline và kiểm tra transcript cùng file WAV. Em có adapter realtime, nhưng chưa coi microphone test là kết quả đạt do đầu ra chưa quan sát ổn định. Vì vậy kết luận của phần này là đã thử inference offline, còn cấu hình 8 GPU/realtime chưa được xác nhận hoàn tất.

## Slide 4 — Các bước chạy PersonaPlex trên Modal

Với PersonaPlex, checkpoint gated được lưu trong Modal Volume và source được pin ở một revision cụ thể. Modal adapter khởi chạy moshi.server trên GPU A100. Máy local chạy authenticated proxy tới endpoint; browser/client kết nối proxy qua WebSocket, gửi role prompt, voice prompt và audio. Server nhận Opus, giải mã thành PCM, Mimi mã hóa từng frame, LMGen cập nhật trạng thái streaming và sinh text/audio; audio được giải mã rồi gửi về client. Sau khi thử inference PersonaPlex 7B, em đọc cả đường WAV offline lẫn WebSocket để chuẩn bị rollout P03.

## Slide 5 — Kiến trúc PersonaPlex

PersonaPlex nhận role prompt dạng text và voice prompt dạng audio ở đầu phiên. User audio được Mimi mã hóa thành frame tokens, sau đó LMGen ghép với trạng thái trước đó và delay của các stream. Mô hình sinh text và audio codes theo từng bước; Mimi giải mã audio để gửi lại client. Điều này cho thấy turn-taking và nội dung cùng xuất hiện trong một luồng streaming, nên đánh giá cần xem cả action, thời điểm và nội dung.

## Slide 6 — Các block và vị trí trong source

Em lần theo server.py, offline.py, loaders.py, compression.py, lm.py và streaming state. Server xử lý session và audio; Mimi làm codec; LMGen giữ cache/delay; Temporal Transformer xử lý ngữ cảnh liên frame; Depformer sinh audio codebooks. Em cũng xác định hai giới hạn cho hướng RL: instruction hiện được nạp khi mở phiên, còn inference path chạy no-grad. forward_train có gradient nhưng chưa phải online GRPO trainer, nên cần làm rõ cách tính log-prob và giữ đúng stream/delay mask.

## Slide 7 — Đọc gì để tìm research gap?

Em đọc survey 25 công trình về full-duplex speech, turn-taking, proactive intervention, repair và evaluation. Sau đó em tập trung vào các paper đánh giá trực tiếp PersonaPlex: Continue, Adapt, or Yield cho thích nghi khi bị chen lời; Take the Floor When Asked, Not When Needed cho can thiệp theo nội dung; và Instruct-FD cho tuân thủ instruction. Những paper này gợi ra ba failure mode P01–P03. Em dùng số liệu như bằng chứng có giới hạn của từng benchmark, không diễn giải chúng thành kết quả mà em đã tái lập.

## Slide 8 — P01: In-turn semantic adaptation

P01 hỏi: khi PersonaPlex đang nói và user chen thêm constraint hoặc sửa thông tin, model có cập nhật phần nói tiếp theo không? Paper báo cáo ADAPT 34,8% cho PersonaPlex và 68,2% cho người trên 66 collaboration pairs. Nhưng nhãn ADAPT có thể chỉ là acknowledgement, chưa chắc model dùng đúng constraint mới. Gap là correct content uptake trong overlap. Phương pháp có thể thử là LoRA-GRPO với reward cho uptake đúng và guardrail cho stale content, hoặc LoRA-SFT nếu có đủ continuation speech chuẩn. Controller chỉ có ý nghĩa nếu nó thật sự điều khiển được lời model phát ra.

## Slide 9 — P02: Content-driven proactive intervention

P02 hỏi liệu model có chủ động lên tiếng khi nội dung tạo lý do cần can thiệp, nhưng vẫn im lặng với tình huống trung tính. Paper báo cáo onset thấp hơn ở false-fact trigger so với câu hỏi trực tiếp; chỉ nhìn onset thì chưa biết model có nhận ra sự thật hay không. Vì vậy em tách nhận diện trigger, quyết định nói, và tính đúng của lời sửa. Hướng đề xuất là cấp evidence hoặc ground truth theo scenario, rồi tối ưu reward chung cho correctness và timing, có neutral matched controls để phát hiện false alarm.

## Slide 10 — P03: Instruction-conditioned turn control

P03 hỏi: nếu giữ nguyên audio mà đổi instruction, hành vi có đổi theo policy không? Instruct-FD báo cáo IAS tổng thể 47%; Continue 90%, trong khi Interrupt 10% và Backchannel 5,8%. Thay đổi theo instruction nhỏ và không có ý nghĩa sau hiệu chỉnh thống kê trong setup paper. Gap là đo policy sensitivity trên matched input, không chỉ tăng interrupt rate. Đây là hướng hiện em chọn để dựng pilot.

## Slide 11 — Phương pháp đọc để thiết kế giải pháp

Sau khi xác định problem evidence, em đọc các paper về phương pháp: Multi-Faceted Interactivity Alignment có GRPO trên PersonaPlex; ASPIRin và HiPLEX tách quyết định control/timing khỏi nội dung; DuplexPO giới hạn update quanh cửa sổ sự kiện; SteerablePlex liên quan instruction runtime và LoRA/GDPO. Những công trình này cho recipe và cảnh báo về reward gaming, credit assignment và compute. Chúng là prior art để thiết kế pilot, không chứng minh recipe đó chạy được trên GPU của dự án.

## Slide 12 — Phương án RL dự kiến cho P03

Phương án em đề xuất là LoRA-GRPO có điều kiện theo instruction. Với cùng scenario, audio, instruction và trigger, sinh một nhóm rollout. Chấm riêng action adherence, timing, chất lượng nội dung và false action; ví dụ backchannel ngắn đúng cơ hội được thưởng, takeover dài hoặc phát ngôn trong neutral control bị phạt. GRPO so sánh reward trong cùng nhóm rollout để tăng xác suất các hành vi tốt; LoRA giới hạn phần cập nhật và reference/KL giúp hạn chế regression. Ứng viên ban đầu là Temporal Transformer/text policy, nhưng phải xác định chính xác stream, log-prob path và masks. Đây là proposal; chưa có trainer hay kết quả RL trong dự án.

## Slide 13 — Trạng thái kiểm chứng P03

Pilot hiện tại dùng DSB-IFEval subset: 10 tình huống và 20 ca ở hai mức conditioning L1/L2, gồm LISTEN, BACKCHANNEL và NO_BACKCHANNEL. Modal adapter chạy PersonaPlex offline, tạo recording hai kênh và ghi score theo benchmark. Đây là benchmark pilot; pipeline Phase 3 tự xây là nhánh khác, hiện có năm scenario draft và scorer RMS/VAD sơ bộ nhưng scenario chưa sẵn sàng. Em chưa kết luận P03 pass hay fail trước khi xem rollout và đối chiếu với người chấm.

## Slide 14 — Baseline trước, training sau

Các bước kế tiếp là hoàn tất baseline, lưu model revision và outputs, review mù một phần audio, rồi đo mức đồng thuận giữa scorer và người nghe. Nếu failure có thể lặp lại và reward phân biệt được rollout tốt/xấu, em mới profile một optimizer step trên GPU mục tiêu. Inference chạy được không có nghĩa LoRA-GRPO sẽ fit bộ nhớ hoặc có log-probability đúng. Go/no-go dựa trên pilot, chất lượng verifier và phép đo compute thực tế.

## Câu kết

Tóm lại, em đã thử inference hai model, chọn PersonaPlex làm backbone nghiên cứu tiếp theo, dùng literature để thu hẹp các limitation thành P01–P03, rồi đề xuất P03 làm pilot với hướng LoRA-GRPO. Trạng thái hiện tại là kiểm chứng baseline bằng DSB-IFEval; chưa phải kết quả training hay cải thiện model.
