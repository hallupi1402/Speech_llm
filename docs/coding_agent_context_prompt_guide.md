# Prompt Guide cho Coding Agent khi bắt đầu task / project / source code mới

## Mục tiêu

Khi nhận một source code mới, không nên yêu cầu agent sửa code ngay.

Luồng tốt nhất:

```text
Đọc repo
→ Hiểu kiến trúc
→ Xác định file liên quan
→ Hiểu execution flow
→ Lập plan
→ Thực hiện từng bước
→ Test
→ Tổng kết
```

Nguyên tắc quan trọng:

> **Understand before modify.**

Agent phải hiểu source trước khi sửa.

---

# 1. Prompt tổng quát khi mới nhận một repo

```text
Bạn đang làm việc với project này:

Repo:
<đường dẫn repo>

Mục tiêu tổng thể:
<mô tả project dùng để làm gì>

Task hiện tại:
<mô tả chính xác yêu cầu cần làm>

Context tôi đã biết:
- ...
- ...
- ...

Ràng buộc:
- Không tự ý sửa code trước khi hiểu luồng.
- Không đoán kiến trúc nếu chưa đọc source.
- Khi nói về code phải ghi rõ file/path liên quan.
- Ưu tiên tận dụng code hiện có, không viết lại nếu không cần.
- Nếu có điều chưa chắc chắn, ghi rõ đó là giả định.

Trước tiên hãy:

1. Khảo sát cấu trúc repo.
2. Xác định entry point.
3. Xác định luồng code liên quan trực tiếp tới task.
4. Xác định config, dependency, model và script liên quan.
5. Chỉ ra những file tôi cần đọc trước.
6. Tóm tắt kiến trúc liên quan tới task bằng sơ đồ đơn giản.
7. Đề xuất plan triển khai từng bước.

CHƯA sửa code.

Output mong muốn:

Repo overview
↓
Relevant files
↓
Execution flow
↓
Current behavior
↓
What needs to change/test
↓
Implementation plan
```

### Ý nghĩa

Prompt này ép agent làm nhiệm vụ giống một kỹ sư mới join project: hiểu repo trước, tìm entry point, chỉ đọc phần liên quan, tránh sửa nhầm và tạo context dùng cho các bước sau.

---

# 2. Prompt khi muốn agent đọc source theo task cụ thể

```text
Task tôi cần làm:

<task>

Hãy chỉ tập trung vào code liên quan trực tiếp đến task này.

Yêu cầu:

1. Tìm entry point của task.
2. Trace execution flow từ đầu đến cuối.
3. Liệt kê file/class/function tham gia vào flow.
4. Với mỗi file, nói ngắn gọn vai trò của nó.
5. Chỉ ra config và dependency ảnh hưởng đến task.
6. Chỉ ra dữ liệu đi vào và output đi ra.
7. Chỉ ra điểm có khả năng cần sửa/test.

Không sửa code.

Mọi kết luận phải kèm path file cụ thể.
```

Dùng khi repo lớn và không muốn agent đọc lan man toàn bộ project.

---

# 3. Prompt yêu cầu agent vẽ execution flow

```text
Dựa trên source code hiện tại, hãy trace execution flow của:

<feature/task>

Bắt đầu từ:
<entry point hoặc command>

Đi đến output cuối cùng.

Trình bày dạng:

command / request
↓
file.py
↓
function/class
↓
module tiếp theo
↓
output

Ở mỗi bước ghi:
- path file
- function/class
- input
- output
- vai trò ngắn gọn

Không giải thích lan man.
Không sửa code.
```

---

# 4. Prompt hỏi file nào cần đọc trước

```text
Tôi mới join project này và cần hiểu task:

<task>

Hãy chọn tối đa 5-10 file quan trọng nhất tôi cần đọc trước.

Với mỗi file ghi:

1. Path.
2. Tại sao cần đọc.
3. Phần/function/class nào cần tập trung.
4. File đó đứng ở đâu trong execution flow.

Sắp xếp theo thứ tự nên đọc.

Không liệt kê file không liên quan trực tiếp.
```

---

# 5. Prompt lập implementation plan trước khi code

```text
Dựa trên source vừa phân tích, hãy lập plan để thực hiện:

Task:
<task>

Yêu cầu:

1. Chia thành các bước nhỏ.
2. Mỗi bước phải ghi:
   - mục tiêu;
   - file liên quan;
   - cần đọc/sửa gì;
   - cách kiểm tra thành công.
3. Ưu tiên thay đổi nhỏ nhất.
4. Không refactor ngoài phạm vi task.
5. Chỉ rõ dependency giữa các bước.
6. Chỉ rõ rủi ro hoặc phần chưa chắc chắn.

CHƯA code.

Output:

Step 1
- Goal:
- Files:
- Change/Test:
- Expected result:

Step 2
...
```

---

# 6. Prompt bắt đầu implement từng bước

```text
Bây giờ thực hiện Step 1 trong plan.

Yêu cầu:

1. Trước khi sửa, nói ngắn gọn:
   - đang sửa gì;
   - file nào;
   - vì sao.

2. Chỉ sửa phần cần thiết cho Step 1.

3. Không làm trước Step 2.

4. Sau khi sửa:
   - tóm tắt thay đổi;
   - đưa command test;
   - nói expected output.

5. Nếu test chưa chắc chắn pass thì dừng lại và chỉ rõ nguyên nhân.
```

Nguyên tắc:

```text
Step 1
→ test
→ Step 2
→ test
→ Step 3
```

---

# 7. Prompt khi có bug

```text
Có bug sau:

<error/log>

Context:
- Command chạy:
<command>

- Expected:
<kết quả mong muốn>

- Actual:
<kết quả hiện tại>

Hãy debug theo source hiện tại.

Trước tiên:

1. Trace error từ stack trace.
2. Xác định file/function gây lỗi.
3. Tìm root cause, không chỉ symptom.
4. Kiểm tra config/dependency có liên quan không.
5. Đề xuất fix nhỏ nhất.

CHƯA sửa code cho tới khi xác định root cause.

Sau đó trình bày:

Root cause
↓
Affected code
↓
Proposed fix
↓
How to test
```

---

# 8. Prompt review code agent vừa sửa

```text
Review các thay đổi vừa thực hiện.

Kiểm tra:

1. Có đúng yêu cầu task không?
2. Có sửa ngoài phạm vi không?
3. Có duplicate logic hiện có không?
4. Có phá API/interface cũ không?
5. Có edge case nào chưa xử lý không?
6. Có dependency/config nào bị bỏ sót không?
7. Test hiện tại có đủ chứng minh thay đổi hoạt động không?

Nếu có vấn đề, chỉ rõ:
file → dòng/đoạn → vấn đề → cách sửa.

Không refactor chỉ để làm code đẹp hơn.
```

---

# 9. Prompt tạo context handoff

```text
Hãy tạo một context handoff ngắn gọn cho agent tiếp theo.

Bao gồm:

Project:
Task:

Architecture liên quan:
- ...

Execution flow:
...

Files quan trọng:
- path: vai trò

Những gì đã làm:
- ...

Những gì đã test:
- command:
- result:

Những gì chưa làm:
- ...

Các quyết định quan trọng:
- ...

Known issues / assumptions:
- ...

Next step:
- ...

Không thêm thông tin không có trong source hoặc quá trình làm việc.
```

---

# 10. Prompt yêu cầu agent giữ working context

```text
Trong suốt task này, sau mỗi bước hãy giữ một working context gồm:

- Goal hiện tại
- Files đã đọc
- Files đã sửa
- Execution flow đã xác nhận
- Assumptions
- Commands đã chạy
- Test results
- Next step

Không cần in toàn bộ context sau mỗi câu trả lời.
Chỉ dùng nó để giữ tính nhất quán trong quá trình làm việc.
```

---

# 11. Prompt riêng cho DuplexOmni inference

```text
Repo:
DuplexOmni

Mục tiêu:
Research và reproduce inference DuplexOmni, sau đó nghiên cứu:
- GPU requirement
- quantization
- inference optimization
- realtime serving

Hiện tại chưa sửa model.

Trước tiên hãy khảo sát source và trả lời:

1. Inference entry point nằm ở đâu?
2. Thinker được start từ file nào?
3. Talker được start từ file nào?
4. Orchestrator nằm ở đâu?
5. Client gửi request như thế nào?
6. Model weights được load ở đâu?
7. Modified vLLM nằm ở đâu?
8. GPU / tensor parallel config nằm ở đâu?
9. Repo hiện có support quantization nào?
10. Command tối thiểu để chạy một inference là gì?

Trace flow:

client
→ orchestrator
→ thinker
→ talker
→ codec
→ output

Yêu cầu:
- Không sửa code.
- Không đoán.
- Mọi kết luận phải kèm path file.
- Nếu không tìm thấy trong source, nói rõ chưa tìm thấy.
- Sau cùng lập plan để chạy model trên GPU server.
```

---

# 12. Prompt nghiên cứu GPU cho DuplexOmni

```text
Dựa trên source và config inference của DuplexOmni:

Hãy xác định yêu cầu phần cứng để chạy model.

Tìm:

1. dtype mặc định: BF16/FP16/FP8...
2. kích thước checkpoint.
3. tensor parallel mặc định.
4. số GPU mà script inference kỳ vọng.
5. VRAM cần cho weights.
6. VRAM bổ sung cho KV cache/runtime.
7. GPU architecture hoặc CUDA requirement.
8. có dependency vào NVLink/NVSwitch hay không.
9. cấu hình tối thiểu có khả năng load model.
10. cấu hình cần để đạt realtime.

Phân biệt rõ:
- thông tin xác nhận từ source;
- estimate;
- phần cần benchmark thực tế.

Không tự suy đoán thành fact.
```

---

# 13. Prompt nghiên cứu quantization cho DuplexOmni

```text
Tôi muốn giảm VRAM của DuplexOmni bằng quantization.

Trước tiên KHÔNG sửa code.

Hãy kiểm tra source và modified vLLM để xác định:

1. Có support FP8 không?
2. Có support INT8 không?
3. Có support AWQ không?
4. Có support GPTQ không?
5. Có support INT4 không?
6. Quantization được xử lý ở file/module nào?
7. CLI/config nào chọn quantization?
8. Thinker có thể quantize riêng không?
9. Talker có thể quantize riêng không?
10. MTP/codec có bắt buộc giữ BF16 không?

Sau đó đề xuất thứ tự experiment an toàn nhất:

BF16 baseline
→ FP8
→ ...

Với mỗi experiment ghi:
- phần được quantize;
- thay đổi config/code;
- GPU requirement;
- metric cần đo;
- cách rollback nếu lỗi.

Không giả định upstream vLLM support nghĩa là fork này cũng support.
```

---

# 14. Prompt sau khi thuê GPU server

```text
Tôi đang ở một GPU server mới.

Repo:
<path>

GPU:
<nvidia-smi output>

Task:
Chạy DuplexOmni inference baseline.

Trước tiên hãy kiểm tra environment:

1. GPU và VRAM.
2. NVIDIA driver.
3. CUDA.
4. Python.
5. PyTorch.
6. repo dependencies.
7. disk còn trống.
8. checkpoint đã có chưa.

Sau đó dựa đúng theo README/source hiện tại:
- tạo environment;
- cài dependency;
- cấu hình model path;
- start inference services;
- chạy minimal test.

Mỗi lần chỉ làm một bước.
Sau mỗi command, đợi output trước khi sang bước tiếp theo.
Không tự ý thay version nếu chưa chứng minh cần thiết.
```

---

# 15. Prompt khi agent cần chạy command trên server

```text
Hướng dẫn tôi chạy từng command một.

Mỗi lần chỉ đưa:
1. command;
2. command dùng để làm gì;
3. expected output ngắn gọn.

Chờ tôi gửi output rồi mới đưa command tiếp theo.

Không đưa một lúc 20 command.
```

---

# 16. Prompt benchmark inference

```text
Model đã chạy được.

Bây giờ hãy thiết kế benchmark tối thiểu để đo:

- GPU model
- GPU count
- peak VRAM
- model loading time
- first token latency
- first audio latency
- RTF
- end-to-end latency
- GPU utilization
- cost/hour

Không optimize trước.

Mục tiêu là tạo BF16 baseline để sau này so với quantization.

Hãy chỉ rõ:
- metric lấy ở đâu;
- command/log nào cần xem;
- cách ghi kết quả vào bảng.
```

---

# 17. Prompt so sánh BF16 và quantized model

```text
Tôi đã có baseline BF16.

Bây giờ cần so sánh với:
<FP8/AWQ/GPTQ/...>

Hãy giữ nguyên workload và input.

So sánh:

| Metric | BF16 | Quantized |
|---|---|---|
| GPU count | | |
| Peak VRAM | | |
| Load time | | |
| First audio latency | | |
| RTF | | |
| Throughput | | |
| Output quality | | |
| Stability | | |
| Cost/hour | | |

Không kết luận quantized tốt hơn chỉ vì VRAM thấp hơn.
Ưu tiên realtime latency và stability.
```

---

# 18. Prompt giải thích code cho người mới

```text
Giải thích đoạn code này cho tôi như người mới join project.

Yêu cầu:
- ngắn gọn;
- tập trung logic;
- không giảng lý thuyết lan man;
- nói input → xử lý → output;
- ghi rõ biến/function quan trọng;
- liên hệ nó với execution flow toàn hệ thống.

Nếu code gọi sang file khác, ghi path file đó.

Cuối cùng tóm tắt bằng 1-3 câu:
"Đoạn code này dùng để..."
```

---

# 19. Prompt đọc từng file

```text
Chúng ta sẽ đọc file:

<path>

Không giải thích cả file một lúc.

Hãy:

1. Tóm tắt vai trò file trong 2-3 câu.
2. Chia file thành các block logic.
3. Bắt đầu block đầu tiên.
4. Giải thích ngắn gọn:
   input → logic → output.
5. Dừng lại sau block đó.

Chờ tôi nói tiếp rồi mới sang block tiếp theo.
```

---

# 20. Prompt hiểu một function

```text
Giải thích function:

<function name>

trong:
<file path>

Chỉ cần:

1. Function này dùng để làm gì?
2. Input là gì?
3. Logic chính là gì?
4. Output là gì?
5. Ai gọi function này?
6. Nó gọi tiếp function/module nào?

Trình bày ngắn gọn.
```

---

# 21. Prompt chống hallucination

```text
Grounding rules:

- Không được đoán tên file, function, config hoặc command.
- Chỉ kết luận khi đã thấy trong source.
- Nếu chưa xác minh, ghi "chưa xác minh".
- Phân biệt rõ:
  [SOURCE] thông tin đọc trực tiếp từ code/docs
  [INFERENCE] suy luận từ code
  [TODO] cần test để xác nhận
```

---

# 22. Prompt chuẩn nên dùng thường xuyên

```text
Bạn đang làm việc trong repo:

<repo path>

Task:
<task>

Mục tiêu:
<goal>

Trước khi code:

1. Khảo sát source liên quan.
2. Tìm entry point.
3. Trace execution flow.
4. Liệt kê file/class/function liên quan.
5. Xác định current behavior.
6. Xác định phần cần thay đổi/test.
7. Lập plan từng bước.

Rules:
- Understand before modify.
- Không đoán.
- Mọi kết luận về code phải có path.
- Không sửa ngoài scope.
- Ưu tiên reuse code hiện có.
- Chỉ thay đổi nhỏ nhất cần thiết.
- Sau mỗi bước phải có cách test.
- Nếu chưa xác minh thì ghi rõ.

CHƯA sửa code cho tới khi plan được xác nhận.
```

---

# Workflow khuyến nghị

```text
PROMPT 1
Khảo sát repo
        ↓
PROMPT 2
Trace execution flow
        ↓
PROMPT 3
Chọn file cần đọc
        ↓
PROMPT 4
Lập implementation plan
        ↓
PROMPT 5
Implement Step 1
        ↓
Test
        ↓
Implement Step 2
        ↓
Test
        ↓
Review
        ↓
Context handoff / report
```

## Nguyên tắc cuối cùng

Context tốt không phải là đưa thật nhiều text. Context tốt là agent biết rõ:

```text
Project là gì?
↓
Task là gì?
↓
Entry point ở đâu?
↓
Execution flow thế nào?
↓
File nào liên quan?
↓
Current behavior là gì?
↓
Muốn thay đổi điều gì?
↓
Test thành công bằng cách nào?
```

Nếu 8 câu này rõ, coding agent thường làm việc ổn định hơn rất nhiều.
