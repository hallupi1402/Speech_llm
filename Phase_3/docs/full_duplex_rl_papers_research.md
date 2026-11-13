# RL cho full-duplex speech: paper, cách đọc và hướng áp dụng P01–P03

Ngày khảo sát: 28-10-2026. Đây là bản tổng hợp từ paper gốc, trang dự án/model card của tác giả và source PersonaPlex trong workspace. Các kết quả dưới đây là kết quả paper công bố; chúng chưa phải kết quả đã reproduce trong dự án này.

## Kết luận để định hướng

RL cho full-duplex không chỉ là “cho model nói nhiều lần rồi chấm điểm”. Vòng lặp nghiên cứu gồm:

1. **Định nghĩa một sự kiện hội thoại có thể kiểm tra** (user ngập ngừng, nhường lượt, chen ngang, instruction yêu cầu nghe/nói).
2. **Tạo input/rollout đúng thời điểm**: cùng audio/context, lấy nhiều continuation từ policy hiện tại.
3. **Tách chấm hành vi và chấm nội dung**: timing/action reward thường dùng VAD, timestamps và rules; semantic reward thường dùng ASR rồi LLM judge.
4. **Kiểm tra reward trước khi train**: xem reward có đúng với người đánh giá và có xếp hạng được các rollout khác nhau không.
5. **Cập nhật policy có giới hạn**: group-relative policy optimization (GRPO/PPO-style), KL với reference, và kiểm tra regression ngoài task.

Đối với đề tài này, paper gần PersonaPlex nhất là **Multi-Faceted Interactivity Alignment** vì họ đã post-train chính PersonaPlex bằng GRPO. Ý tưởng phương pháp hữu ích nhất để tránh timing reward làm hỏng lời nói là **HiPLEX/ASPIRin/DuplexPO**; hướng liên quan nhất P03 là **SteerablePlex**, nhưng paper này phải thêm giao diện chèn instruction trong lúc stream đang chạy. Trong PersonaPlex upstream hiện có ở workspace, đường giao tiếp đó chưa tồn tại.

## Paper nên đọc theo thứ tự

| Thứ tự | Công trình | Đóng góp cần hiểu | Quan hệ với đề tài |
|---|---|---|---|
| 1 | [ORISE, COLM 2025](https://openreview.net/pdf?id=QbLbXz8Idp) | Tiền lệ RL trực tuyến cho full-duplex; reward lấy từ annotation tự động trên speech sinh ra. | Nền tảng lịch sử cho turn-taking, barge-in, backchannel; model/kiến trúc không phải PersonaPlex. |
| 2 | [ASPIRin, 2026](https://arxiv.org/abs/2604.10065) | Chiếu action token thành hai trạng thái “đang nói / im lặng” rồi tối ưu timing bằng GRPO. | Giải thích vì sao raw-token timing reward có thể làm semantic generation collapse. |
| 3 | [Multi-Faceted Interactivity Alignment, 2026](https://arxiv.org/abs/2606.11167) | GRPO trên Moshi **và PersonaPlex**, reward riêng cho bốn loại tương tác + reward chất lượng phản hồi. | Baseline quan trọng nhất để đối chiếu; chưa giải quyết trực tiếp content uptake, factual intervention hoặc instruction-conditioned actions. |
| 4 | [DuplexPO, 2026](https://arxiv.org/abs/2607.07148) | Chỉ cập nhật trên cửa sổ thời gian quan trọng, reward phân rã theo sự kiện. | Gợi ý cách giảm credit assignment rộng và bảo toàn năng lực ngôn ngữ; không phải cùng kiến trúc/backbone. |
| 5 | [HiPLEX, 2026](https://arxiv.org/abs/2610.07727) | Tách phân phối text thành quyết định control (PAD/EPAD/CONTENT) và quyết định nội dung; route reward đến các frame liên quan. | Hướng gần nhất để xử lý “khi nào nói” khác với “nói gì”; đã chạy trên PersonaPlex nhưng báo cáo dùng full-parameter updates. |
| 6 | [SteerablePlex, 2026](https://arxiv.org/abs/2610.12201) | LoRA-GDPO, cho phép chèn instruction vào model-text stream lúc đang hội thoại; có actor–director. | Liên quan P03 và LoRA; kiểm tra input interface trước khi copy training recipe. |
| 7 | [Instruct-FD](https://arxiv.org/abs/2607.20460) | Benchmark đo khả năng thay đổi turn-taking theo instruction; đây là benchmark/evidence, không phải RL recipe. | Cơ sở thiết kế P03 scenarios và evaluation; các số benchmark cần tự reproduce. |

## 1. ORISE — RL online đầu tiên cần nắm

[ORISE: Reinforcement Learning Enhanced Full-Duplex Spoken Dialogue Language Models for Conversational Interactions](https://openreview.net/pdf?id=QbLbXz8Idp) giới thiệu Online Reinforcement with Interactive Speech Evaluation. Tác giả dùng reward tùy biến từ annotation tự động của speech do model sinh ra, nhắm đến turn-taking, user barge-in và backchannel; paper báo cáo khả năng chịu noise chưa thấy trong dữ liệu gán nhãn. Đây là một tiền lệ quan trọng về cách chuyển hành vi speech thành reward, nhưng không phải phép thử PersonaPlex: model và pipeline của ORISE khác, và không nên lấy kết quả của nó làm bằng chứng rằng PersonaPlex sẽ học được cùng hành vi.

**Cách đọc:** xem phần ORISE reward/training rồi bảng ablation so sánh DPO, IPO, KTO, ORISE. Câu hỏi cần rút ra là reward được tính từ tín hiệu quan sát nào, phản ánh được hành vi nào, và phụ thuộc vào kiến trúc nào.

## 2. ASPIRin — Vì sao tối ưu timing bằng token reward có thể nguy hiểm

[ASPIRin](https://arxiv.org/abs/2604.10065) nói rằng text tokens trong model full-duplex cùng lúc mang nội dung và điều khiển thời điểm nói. Nếu reward chỉ quan tâm overlap/latency mà GRPO cập nhật trực tiếp từng text token, policy có thể tối ưu bằng lặp từ, lời nói suy thoái hoặc im lặng quá mức.

ASPIRin gom text vocabulary thành hai nhóm **padding = silence** và **non-padding = active speech**, tạo một policy nhị phân trên trạng thái này rồi tối ưu timing bằng GRPO. Reward chính phạt nói đè user và phản hồi chậm. Paper báo cáo giảm hơn 50% repeated n-grams so với standard GRPO trong thí nghiệm của họ. Training dùng tập hội thoại nội bộ 43 giờ và benchmark của họ tập trung turn-taking/backchannel/interrupt; không phải P01/P02/P03.

**Bài học:** P03 có thể cần reward “chọn đúng hành vi” thay vì phạt trực tiếp toàn bộ nội dung token. Nhưng đừng áp dụng nguyên action projection nếu action cần đo là ADAPT hoặc sửa sai: “đang nói/im lặng” không đủ biểu diễn semantic uptake hay chủ động correction.

## 3. Multi-Faceted — baseline gần PersonaPlex nhất

[Paper](https://arxiv.org/abs/2606.11167) áp dụng post-training RL trên Moshi và PersonaPlex. Tác giả trích các đoạn ngắn từ hội thoại người thật, tạo bốn nhóm: pause handling, turn-taking, backchanneling, user interruption. Với một input, model sinh group gồm nhiều rollout; mỗi rollout được decode thành audio để chấm bằng reward theo trục đó. Họ tối ưu xác suất của **text stream**, không trực tiếp tối ưu log-probability audio tokens, và thêm LLM judge reward để bảo vệ tính liên quan/tự nhiên của câu trả lời.

Reward minh họa:

- **Pause:** phạt nếu model tạo đoạn nói dài hơn 1 giây trong khoảng user đang ngập ngừng.
- **Turn-taking:** thưởng response latency thấp sau khi user kết thúc lượt.
- **Backchannel:** F1 giữa các vocalization ngắn và vị trí backchannel gốc.
- **User interruption:** đo thời gian từ khi user chen ngang đến lúc model nói tiếp; kèm judge cho chất lượng nội dung.

Training data trong paper gồm Fisher và Seamless Interaction; trích tối đa 2.000 segment cho mỗi trục. Setup được báo cáo là 100 epochs, group size 16, phân phối trên 32 NVIDIA H100; vì vậy **không phải bằng chứng rằng nguyên recipe chạy được trên một A100 80GB**. Paper báo cáo cải thiện các metric tương tác, nhưng cũng nêu nguy cơ safety regression do phong cách dữ liệu hợp tác Fisher. Các số liệu dùng đánh giá tự động/LLM judge, chưa thay cho human evaluation.

**Artifacts:** Kyutai phát hành [PersonaPlex RL Seamless checkpoint/model card](https://huggingface.co/kyutai/personaplex-rl-seamless), tương thích với inference code PersonaPlex, cùng mô hình và một số audio samples. Trong các nguồn chính thức đã kiểm tra, xác minh được checkpoint/model card; **chưa xác minh được training code đầy đủ của paper**. Không nên mặc định checkpoint đồng nghĩa có pipeline để tái tạo huấn luyện.

**Liên hệ với ba vấn đề:**

- P01: interruption là gần về tình huống overlap, nhưng paper đo yield/response latency; không đo model có cập nhật đúng thông tin vừa được thêm hay không.
- P02: user interruption là user ngắt model, không phải model tự ngắt user để sửa thông tin sai. Hai hướng floor control ngược nhau.
- P03: Full-Duplex-Bench v2 có instruction-following score, nhưng paper không huấn luyện theo explicit turn-taking instruction như P03.

## 4. DuplexPO — tối ưu ở cửa sổ sự kiện

[DuplexPO](https://arxiv.org/abs/2607.07148) không lấy cả hội thoại làm một đơn vị update. Nó trích **dynamics-critical windows** quanh lượt nói/backchannel/barge-in, teacher-force lịch sử trước cửa sổ và chỉ tối ưu continuation trong cửa sổ. Reward FCDR tách turn-initiation, backchannel, yielding sau barge-in và regularization; group-relative objective kiểu GRPO so sánh nhiều continuation cho cùng một window.

Paper báo cáo cải thiện hành vi full-duplex và giữ các đánh giá reasoning/instruction-following của thiết lập của họ. Tuy vậy, đây là pipeline với stages pretraining/SFT/RL riêng và mô hình/dữ liệu khác, không phải recipe đã được chứng minh chạy trực tiếp trên PersonaPlex 7B. Hãy xem đây là ý tưởng về **đơn vị rollout + vùng credit assignment**, không phải code thay thế PersonaPlex inference.

**Liên hệ đề tài:** P01 nên bắt đầu rollout gần timestamp overlap, P02 gần thời điểm trigger false fact, P03 quanh thời điểm instruction/action opportunity. Việc giới hạn cửa sổ có thể giúp reward gắn với quyết định liên quan, nhưng chỉ đúng nếu state/cache trước cửa sổ được khôi phục đúng như inference streaming.

## 5. HiPLEX — điều khiển “nói hay giữ im lặng” khác với nội dung câu nói

[HiPLEX](https://arxiv.org/abs/2610.07727) bắt đầu từ quan sát rằng text stream Moshi/PersonaPlex vừa biểu đạt nội dung, vừa điều khiển phát ngôn qua các token PAD/EPAD. Paper phân rã cùng một phân phối text thành:

- **Control policy:** chọn `pad`, `epad` hay `content` ở mỗi frame — quyết định có phát token nội dung mới không.
- **Content policy:** chỉ khi chọn `content` mới chọn từ/token cụ thể.

Timing reward được route tới những quyết định control/frame có quan hệ nhân quả với sự kiện (event-causal masks); semantic judge reward chỉ route tới quyết định content. Đây là factorization của logits hiện có, không thêm parameter mới, nhưng hai factor vẫn chia sẻ weights nên không độc lập tuyệt đối. Paper báo cáo trên PersonaPlex/HiPLEX các kết quả tốt hơn baseline GRPO ở một số trục restraint và interruption timing, nhưng một metric khác có thể suy giảm; tác giả cũng phân tích collapse sang gần-im lặng khi reward balance/learning rate không phù hợp.

Setup trong appendix: 73.561 windows Seamless, bốn trục; group size 16; 100 epochs; PersonaPlex dùng 2e-6; 16×H100 khoảng 13 giờ cho 100 epochs. Paper ghi rõ **training update tất cả model parameters**. Vì vậy đây là bằng chứng về phương pháp, nhưng không phải LoRA/A100 feasibility proof. Paper cũng phát hiện kết quả benchmark có thể nhạy với streaming setup và xử lý greeting; điều này nhấn mạnh cần khóa baseline inference protocol trước khi so sánh.

**Liên hệ P01/P02/P03:**

- P01: có thể dùng control factor để quyết định tiếp tục/yield, còn semantic reward xét uptake của thông tin chen vào. Tuy nhiên uptake reward cần verifier riêng, HiPLEX không cung cấp sẵn.
- P02: timing/control factor hữu ích cho thời điểm can thiệp; correctness reward phải xử lý separately.
- P03: action-specific instruction có thể biến thành reward cho control actions; cần bổ sung instruction-conditioning ở input trước.

## 6. SteerablePlex — gần P03 nhất và có LoRA

[SteerablePlex](https://arxiv.org/abs/2610.12201) nghiên cứu full-duplex model có thể nhận instruction **trong lúc cuộc hội thoại đang chạy**. Actor là PersonaPlex; director là một text LLM bất đồng bộ theo dõi ASR + model text, cập nhật goal state định kỳ và gửi instruction khi cần. Instruction được tokenizer encode rồi chèn vào model-text stream giữa special start/end tokens; trong các bước chèn, hai audio streams nhận silence và conversation clock không tiến.

Họ dùng Fisher để tạo 9.417 ví dụ instruction (130,8 giờ rollout) và thêm 2.000 hội thoại synthetic overlap cho instruction theo kịch bản. Training dùng **LoRA adapters**, GDPO với group size 8; paper báo cáo LoRA rank temporal/depth là 128/32 và hai biến thể reward/KL. Reward goal-completion đứng một mình dễ bị hack (nói tuột goal hoặc giữ sàn quá lâu), nên tác giả kết hợp completion, naturalness và turn-taking. Với training chỉ instruction đơn, khả năng hoàn tất nhiều goals theo đúng thứ tự còn yếu; dữ liệu instruction tuần tự/in-domain giúp hơn nhưng không giải quyết hết.

**Giới hạn áp dụng:** bài này hướng tới simulator hoàn thành kịch bản nhiều bước, không phải benchmark P03 từng action như `listen/backchannel/interrupt`. Cách chèn instruction runtime là một thay đổi model input path, không phải chỉ thêm query parameter.

## 7. Instruct-FD — benchmark, không phải phương pháp RL

[Instruct-FD](https://arxiv.org/abs/2607.20460) xây benchmark để hỏi: cùng một cơ hội hội thoại, instruction khác nhau có khiến model chọn hành vi khác không? Paper mô tả human-validated synthetic pipeline và multi-turn evaluation với LLM judge; trên sáu systems, model tốt nhất đạt 64,4% adherence, các proactive behaviors như backchannel/interrupt khó hơn. Những số này là benchmark results trong setup paper, chưa phải kết quả local của dự án.

Sử dụng cho P03:

- Tạo cặp counterfactual: **cùng user audio/context, đổi instruction**; expected action phải đổi theo instruction.
- Đo cả **action adherence** lẫn **instruction sensitivity**. Nếu model luôn Continue thì có thể đạt một số cases nhưng không chứng minh nó đọc instruction.
- Đo timing, false interruption, và semantic quality riêng; không gộp tất cả thành một scalar sớm.

Trước khi dựa vào dataset/code bên ngoài, cần xác minh quyền truy cập, format, license và khả năng reproduce đúng prompt/protocol. Việc paper mô tả một benchmark không tự động có nghĩa toàn bộ data/audio public.

## Khoảng trống ở source PersonaPlex đang có

Trong upstream checkout hiện tại:

- [`server.py:135–171`](/home/yennguyen/Seminar/Phase_3/upstream/personaplex/moshi/moshi/server.py:135) đọc `text_prompt`/`voice_prompt` khi mở WebSocket; text prompt được encode rồi gán vào `lm_gen.text_prompt_tokens`.
- [`server.py:258–282`](/home/yennguyen/Seminar/Phase_3/upstream/personaplex/moshi/moshi/server.py:258) reset streaming state, nạp system prompts, gửi handshake và mới bắt đầu các loop audio.
- [`lm.py:1096–1127`](/home/yennguyen/Seminar/Phase_3/upstream/personaplex/moshi/moshi/models/lm.py:1096) đưa text prompt vào LM trong bước system-prompt đầu phiên.
- [`lm.py:727–850`](/home/yennguyen/Seminar/Phase_3/upstream/personaplex/moshi/moshi/models/lm.py:727) giữ state/cache và xử lý từng bước inference; không có API public ở đây để chèn một instruction giữa phiên như SteerablePlex.

**Suy luận từ source:** `server.py` hiện có session-start prompt, còn runtime instruction injection cần sửa cách dựng input/stream cache và xử lý clock/token delays. Vì vậy P03 có hai bài toán riêng: (a) model có tuân thủ instruction không, và (b) upstream input path có thể nhận instruction giữa stream không. Chỉ thêm `instruction` vào URL/WebSocket query lúc connect sẽ là instruction tĩnh đầu phiên, không tương đương instruction phát sinh giữa hội thoại.

## Hiểu GRPO bằng một ví dụ P03

Giả sử scenario có cùng một đoạn user audio đang kể chuyện. Tạo hai điều kiện:

- Instruction A: “Chỉ lắng nghe, không chen ngang.”
- Instruction B: “Đưa một backchannel ngắn ở chỗ ngắt tự nhiên.”

Cho mỗi điều kiện, sinh `G` rollout từ cùng context. Verifier đo action, timestamp, độ dài overlap và nội dung. Chấm reward theo từng thành phần; chuẩn hóa reward giữa G rollout cùng context để thấy ứng viên nào tốt hơn các ứng viên khác. GRPO tăng xác suất các quyết định/token của rollout tốt, giảm xác suất rollout xấu, đồng thời dùng KL/reference để tránh policy đi quá xa.

Ba điều kiện để có tín hiệu học:

1. Rollout phải khác hành vi thật; nếu cả G đều Continue, reward không phân biệt được.
2. Verifier phải đánh giá đúng action/timing, không chỉ thấy từ khóa “yeah” rồi kết luận có backchannel đúng.
3. Loss phải gán credit tới quyết định liên quan. Timing reward trên mọi frame/token có thể đẩy model vào lặp từ hoặc im lặng; xem ASPIRin/HiPLEX/DuplexPO để hiểu các phương án sửa.

Training reward không phải evaluation metric. Sau pilot, so sánh Base và RL trên held-out scenarios và benchmark riêng, cùng inference protocol; báo cáo cả task metric lẫn speech/content regression.

## Đề xuất hướng implementation cho dự án

1. **Chốt scenario/eval trước:** bắt đầu P03 như handoff đề xuất; xây paired cases + labels/timestamps, xác minh instruction hiện truyền được đến đâu.
2. **Đo baseline và reward verifier trước RL:** chạy nhiều stochastic rollouts, đánh giá action/timing, kiểm tra verifier thủ công trên sample nhỏ và đo pairwise discrimination trong group.
3. **Chọn tối ưu theo quyết định:** thử control/content reward routing (HiPLEX) hoặc event-window policy update (DuplexPO); coi đây là giả thuyết cần ablation, không mặc định tốt hơn.
4. **Thử LoRA-GRPO/GDPO pilot:** SteerablePlex là precedent trực tiếp về LoRA full-duplex + multi-reward, nhưng phải tự cắm adapter và lấy log-prob đúng từ PersonaPlex inference/training path.
5. **Không hứa hẹn 1×A100 trước smoke test:** paper Multi-Faceted dùng 32×H100; HiPLEX dùng 16×H100; SteerablePlex ghi LoRA nhưng paper setup không chứng minh cấu hình Modal 1×A100. Ước lượng phải đo peak VRAM, rollout speed, backward memory và reward latency trên một optimizer step.

P01 và P02 nên được giữ làm task riêng trong evaluation. Các paper turn-taking RL hiện có chưa chứng minh được **correct content uptake** sau overlap (P01), cũng chưa chứng minh **factual intervention** khi người dùng nói sai (P02). Không nên dùng reward latency/backchannel thay cho các mục tiêu semantic này.

## Nguồn chính đã đối chiếu

- [ORISE, OpenReview PDF (COLM 2025)](https://openreview.net/pdf?id=QbLbXz8Idp)
- [ASPIRin, arXiv](https://arxiv.org/abs/2604.10065)
- [Multi-Faceted Interactivity Alignment, arXiv](https://arxiv.org/abs/2606.11167)
- [Kyutai PersonaPlex RL Seamless model card/checkpoint](https://huggingface.co/kyutai/personaplex-rl-seamless)
- [DuplexPO, arXiv](https://arxiv.org/abs/2607.07148)
- [HiPLEX, arXiv](https://arxiv.org/abs/2610.07727)
- [SteerablePlex, arXiv](https://arxiv.org/abs/2610.12201)
- [Instruct-FD, arXiv](https://arxiv.org/abs/2607.20460)
- [PersonaPlex upstream repository](https://github.com/NVIDIA/personaplex)
