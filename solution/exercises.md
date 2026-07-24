# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> Ở temperature 0.0, model trả lời trực tiếp về 36 phố phường, còn các mức 0.7, 1.2 và 1.8 lần lượt chọn những thông tin đa dạng hơn về đường tàu, cầu Long Biên và con đường gốm sứ. Cả bốn phản hồi đều có cấu trúc hợp lý; ngay cả mức 1.8 vẫn chưa biểu hiện kém mạch lạc rõ ràng, mặc dù nội dung thay đổi và có tính ngẫu nhiên cao hơn. Các câu bị dừng giữa chừng là do giới hạn `max_tokens=120`, không phải do temperature.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> Tôi sẽ đặt temperature khoảng 0.2 cho trợ lý soạn thảo hợp đồng pháp lý vì tác vụ này cần câu chữ ổn định, chính xác và hạn chế nội dung sáng tạo ngoài yêu cầu. Với trợ lý viết slogan quảng cáo, tôi sẽ chọn temperature khoảng 1.2 để tạo ra nhiều cách diễn đạt mới lạ và đa dạng hơn. Hợp đồng ưu tiên tính nhất quán và độ tin cậy, trong khi slogan ưu tiên khả năng sáng tạo.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> Workload này tạo ra 20.000 × 2 = 40.000 lượt gọi, tương đương 40.000 × 500 = 20 triệu output token mỗi ngày. Theo bảng giá trong `template.py`, GPT-4o tốn khoảng 20.000 × 0,010 = 200 USD/ngày, còn GPT-4o-mini tốn khoảng 20.000 × 0,0006 = 12 USD/ngày; phép tính này chưa bao gồm input token. Model lớn xứng đáng với chi phí cho tác vụ quan trọng như phân tích hoặc rà soát hợp đồng phức tạp, còn model nhỏ phù hợp với chatbot FAQ hoặc yêu cầu đơn giản có lưu lượng lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> Với persona nhà thơ, phản hồi sử dụng các hình ảnh ví von như hạt giống, khu vườn tri thức và ánh mặt trời nên có giọng văn giàu hình ảnh, dễ tiếp cận và gần như không dùng thuật ngữ kỹ thuật. Với persona kỹ sư phần mềm, phản hồi dài và có cấu trúc rõ ràng hơn, đưa ra định nghĩa chính xác, phân loại các phương pháp học máy và bắt đầu minh họa bằng code Python. Hai phản hồi trả lời cùng một câu hỏi nhưng khác rõ rệt về giọng văn, độ dài và mức độ kỹ thuật. Qua đó, system prompt có thể điều khiển persona, cách dùng từ, mức chi tiết, cấu trúc trình bày và loại ví dụ trong phản hồi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> Đoạn văn được chọn có 166 từ; `count_tokens` đếm được 188 token, trong khi công thức số từ chia cho 0,75 ước lượng 221,33 token. Hai kết quả chênh lệch khoảng 15,06%, trong đó phương pháp ước lượng thô cao hơn thực tế khoảng 33,33 token nên sẽ làm dự toán ngân sách bị thừa đối với đoạn văn này. Nguyên nhân là tokenizer chia văn bản thành các token hoặc subword phụ thuộc vào nội dung, dấu câu, ký tự tiếng Việt và bộ mã hóa của model, nên không tồn tại một tỷ lệ cố định giữa số từ và số token.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> Trợ lý giọng nói hưởng lợi nhiều nhất từ streaming vì nó có thể bắt đầu đọc ngay khi những phần đầu của phản hồi được tạo ra, giúp giảm đáng kể thời gian chờ mà người dùng cảm nhận được. Chatbot văn bản cũng hưởng lợi vì người dùng có thể đọc dần câu trả lời thay vì chờ toàn bộ nội dung hoàn thành. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm không cần streaming vì không có người dùng chờ phản hồi theo thời gian thực; với ứng dụng này, độ chính xác và việc xử lý hoàn chỉnh toàn bộ tài liệu quan trọng hơn tốc độ hiển thị từng phần.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> Khi API quá tải, exponential backoff làm thời gian chờ tăng dần sau mỗi lần thất bại, nhờ đó giảm số request dồn lên server và cho hệ thống thêm thời gian phục hồi; delay cố định vẫn khiến client gửi lại request liên tục với cùng tần suất. Tuy nhiên, nếu hàng nghìn client cùng thất bại tại một thời điểm và dùng cùng công thức backoff, chúng vẫn có thể thức dậy và retry đồng thời, tạo ra hiện tượng “thundering herd”. Jitter thêm một khoảng ngẫu nhiên vào thời gian chờ để làm lệch thời điểm retry giữa các client, phân tán tải đều hơn và tránh các đợt request đồng loạt lặp lại.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> System prompt tôi sử dụng là: “Bạn là trợ giảng AI thân thiện, trả lời ngắn gọn bằng tiếng Việt.” Nếu xóa cụm “trợ giảng AI thân thiện”, model sẽ mất vai trò hỗ trợ học tập và có thể trả lời với giọng văn chung chung hoặc ít gần gũi hơn. Nếu xóa cụm “trả lời ngắn gọn bằng tiếng Việt”, phản hồi có thể dài dòng hơn hoặc không duy trì tiếng Việt ổn định. Hai phần này lần lượt điều khiển vai trò, thái độ, độ dài và ngôn ngữ của trợ lý.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> Giả sử ở lượt đầu người dùng cung cấp các yêu cầu quan trọng cho một dự án, sau đó trao đổi thêm năm lượt về từng chi tiết và cuối cùng hỏi lại yêu cầu ban đầu. Vì history chỉ giữ bốn lượt gần nhất, thông tin ở lượt đầu đã bị loại bỏ nên trợ lý có thể trả lời thiếu hoặc sai ngữ cảnh. Một cách khắc phục là tóm tắt các lượt cũ thành một phần bộ nhớ ngắn trước khi cắt history, đồng thời giữ lại có chọn lọc những dữ kiện quan trọng như tên người dùng, mục tiêu và các quyết định đã thống nhất.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
