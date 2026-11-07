# Zalo AI Challenge 2026 — DataMate

Team chốt **bảng DataMate** ngày 26/10/2026. Tài liệu này chỉ giữ phần áp dụng cho DataMate.
Nguồn: PDF thể lệ BTC + các mục Description / Evaluation / Data / Rule / Submission Guidance của bảng DataMate trên portal.

## 0. Chốt nhanh

- Đội tối đa 5 người; mỗi người một tài khoản Zalo, tự đăng ký trước khi vào đội.
- Vòng public 26/10 → 16/11: nộp JSONL **5 lượt/ngày cho cả đội**, reset cuối ngày, chỉ chấm bài cuối.
- Giải pháp Docker: **2 lượt**, chỉ chấm lượt cuối, hạn **16/11/2026 23:59**.
- Private test phát **09:00 20/11**, nộp kết quả trước **23:59 23/11**; điểm chính thức là lần BTC chạy lại Docker.
- Chấm **PA = (1/N) · Σ sᵢ**, `s = +1` đúng / `0` abstain / `−1` sai; chỉ nên trả lời khi P(đúng) > 0,5.
- Ngân sách suy luận: 2×RTX 5090 (32GB/card, 64GB tổng), 512GB RAM, CUDA ≥ 12,8, offline, **6h tính cả thời gian nạp mô hình**, tổng tham số ≤ 15B.

## 1. Lịch thi (GMT+7)

| Mốc | Việc |
|---|---|
| 26/10/2026 | Khai mạc; công bố dữ liệu train, public test, công cụ đánh giá |
| 16/11/2026 23:59 | Hạn nộp giải pháp cuối (Docker: mã huấn luyện, mã suy luận, mô hình, tài liệu, hướng dẫn chạy) |
| 20/11/2026 09:00 | Công bố private test |
| 23/11/2026 23:59 | Hạn nộp kết quả cuối trên private test |
| 02/12/2026 | Công bố top 2 mỗi đề |
| 12/12/2026 | Công bố kết quả chung cuộc |

BTC có quyền đổi mốc thời gian; mọi thay đổi công bố trên portal, đội tự theo dõi.

## 2. Đội thi

- Quốc tịch Việt Nam, có tài khoản Zalo hợp lệ, từ 18 tuổi.
- Nhân viên, cộng tác viên, người có hợp đồng với VNG không được thi.
- Mỗi người một tài khoản Zalo duy nhất, tham gia một đội duy nhất, tự đăng ký trước khi được thêm vào đội.
- Đội tối đa 5 người, gồm đội trưởng; đội trưởng thêm hoặc xóa thành viên.
- Được gộp đội; sau khi gộp, tổng lượt nộp không vượt (lượt/ngày) × (số ngày tính đến lúc gộp).
- Cấm nhiều tài khoản, tài khoản giả, hoặc lập nhiều đội để tranh giải.

## 3. Nộp bài

| Hạng mục | Giới hạn |
|---|---|
| Kết quả vòng public (JSONL) | 5 lượt/ngày/đội, reset cuối ngày, chỉ chấm bài cuối |
| Giải pháp cuối (Docker) | 2 lượt, chỉ chấm lượt cuối, hạn 16/11/2026 23:59 |
| Kết quả vòng private (JSONL) | Hạn 23/11/2026 23:59. Thể lệ chung ghi 5 lượt/ngày; trang Submission Guidance không nêu số lượt cho vòng này |

- Docker lớn nên phải gửi checksum và link Google Drive công khai để BTC tải.
- Private: đội chạy giải pháp đã đóng băng trên private test rồi nộp JSONL; BTC chạy lại Docker và đó là điểm chính thức.
- Bài nộp muộn, sai định dạng, thiếu file, bị sửa trái phép hoặc gian lận không được tính.
- Câu thiếu trong `submission.jsonl` tính là abstain.

### Gói Docker phải có

- Mã huấn luyện + mã suy luận, trọng số mô hình, README mô tả giải pháp và hướng dẫn chạy.
- `predict.sh` đọc test set từ `/data` (mount read-only) và ghi `/result/submission.jsonl`.
- Log từng câu vào `/result/logs/`: tối thiểu `id`, đáp án cuối, thời gian chạy; kèm dấu vết dẫn tới đáp án (SQL đã sinh, kết quả thực thi…). BTC dùng log để phân xử khi lệch PA giữa bản nộp và bản chạy lại.

## 4. Dữ liệu

Ba phần: train, public test, private test. Bản tải về ngày 26/10/2026:

| | train | public_test |
|---|---|---|
| business | 30 (`biz_101`..`biz_130`) | 14 (`dev_001`..`dev_014`) |
| câu hỏi | 3.000 | 1.000 |
| đáp án | `answers.jsonl` | không có |
| dung lượng | 194 MB | 109 MB |

- `<split>/data/<business_id>/`: `*.csv` | `*.xlsx` | `*.sqlite` + `README.md`; ngoài ra `rulebook.md`, `calendar.csv`, `questions.jsonl` (train thêm `answers.jsonl`).
- `questions.jsonl`: `id`, `business_id`, `question`, `as_of_date`.
- `answers.jsonl` (chỉ train): `id`, `type`, `answer`, `alias`; `alias` là dạng tên được chấp nhận + mã hàng, bài nộp không cần.
- Đuôi file: train 107 xlsx · 20 csv · 1 sqlite; public 69 xlsx · 10 csv · 1 sqlite.
- `rulebook.md` **khác nhau giữa hai split** → đọc rulebook của split đang trả lời, không dùng bản train.
- README từng business chất lượng không đều (có bộ đầy đủ, có gần như trống) → dựng catalog từ dữ liệu thật.
- Private test có ngành nghề không xuất hiện trong train + public test.
- Dữ liệu chỉ dùng phi thương mại/học thuật, phải trích dẫn "challenge.zalo.ai"; không chia sẻ ra ngoài đội trong thời gian thi.

## 5. Bài toán

### 5.1 Vào / ra

```
Input : {"id": "q_06de33fa21cc", "business_id": "biz_118", "question": "sp bán dc nhiều nhất tuần rồi là gì", "as_of_date": "2026-03-04"}
Output: {"id": "q_06de33fa21cc", "type": "string", "answer": "Xi măng Hà Tiên PCB40 bao 50kg"}
```

Các dạng dòng nộp:

```
{"id": "q_01235", "type": "number",  "answer": 48250000}
{"id": "q_01238", "type": "boolean", "answer": true}
{"id": "q_01239", "type": "list",    "answer": ["Cà phê sữa", "Bạc xỉu", "Trà đào"]}
{"id": "q_01240", "type": "table",   "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": [["Chị Lan", 3200000]]}}
{"id": "q_01241", "type": "unanswerable", "reason": "Dữ liệu không có giá vốn nên không tính được lợi nhuận"}
{"id": "q_01242", "type": "abstain"}
```

### 5.2 Chấm điểm

`PA = (1/N) · Σ sᵢ` với `sᵢ = +1` đúng, `0` abstain, `−1` sai; PA ∈ [−1, 1].
Trả lời có lợi về kỳ vọng khi P(đúng) > 0,5. `unanswerable` là đáp án nhận +1 điểm khi đúng; trả một giá trị cụ thể cho câu có tham chiếu `unanswerable` bị −1 điểm.

### 5.3 Chín kiểu đáp án

| Kiểu | Nội dung | Cách khớp |
|---|---|---|
| number | tiền VND, đếm, số lượng | chính xác; tham chiếu là số nguyên hoặc làm tròn 2 chữ số thập phân |
| percent | phần trăm, đơn vị % | làm tròn 2 chữ số thập phân; dấu phải đúng |
| string | tên sản phẩm, khách, chi nhánh, thứ trong tuần | NFC → lowercase → ký tự không phải chữ/số/khoảng trắng thành space → gộp space, trim. Không bỏ dấu. Nhận mọi cách viết trong dữ liệu + mã (`sp0042`) |
| date | ngày ISO | `yyyy-mm-dd`; câu theo tháng thì `yyyy-mm` |
| boolean | true / false | chính xác |
| list | danh sách có thứ tự | so từng vị trí; thứ tự theo câu hỏi, không có thì theo rulebook; phần tử theo luật string/number |
| table | bảng nhỏ: tên cột + dòng | cột so theo tên đã chuẩn hóa, không khớp thì theo vị trí; dòng so như tập hợp |
| unanswerable | dữ liệu không chứa thứ cần trả lời | đúng khi và chỉ khi tham chiếu là `unanswerable` |
| abstain | tự chọn không trả lời | luôn 0, bất kể tham chiếu |

Phân bố type trong 3.000 câu train: number 1.884 · percent 578 · list 193 · date 145 · string 88 · unanswerable 70 · boolean 23 · table 19. Không có `abstain` trong gold.

### 5.4 Câu xếp hạng

- Tiêu chí theo câu hỏi, không có thì theo rulebook; xếp giảm dần trên các thứ tự hợp lệ.
- Bằng giá trị thì xếp theo mã tăng dần.
- Hòa ở ranh giới `k` (hoặc hạng 1 với câu "… nhất") → câu bị loại khỏi test set.
- Ít hơn `k` ứng viên thì đáp án là tất cả.

### 5.5 Ràng buộc kỹ thuật

- Mô hình công bố trọng số trước 01/10/2026, giấy phép cho phép dùng.
- Tổng tham số mọi mô hình lúc suy luận ≤ 15 tỷ, tính cả MoE; được lượng tử hóa.
- Khai báo toàn bộ mô hình trong README của giải pháp.
- Suy luận offline trong Docker: không API thương mại, không tải dữ liệu, không có mạng.
- Được sinh và chạy mã trong container; `/data` chỉ đọc; không can thiệp thủ công; cấm hard-code đáp án hoặc logic theo mã câu hỏi.
- Được dùng dữ liệu train, dữ liệu tự tổng hợp, dataset công khai; cấm dùng public/private test để train.
- Quá 6h hoặc dừng sớm: chỉ phần đã ghi vào `/result/submission.jsonl` được chấm, phần còn lại là abstain.
- Lệch `|ΔPA| > 0,03` giữa bản nộp và bản chạy lại bị kiểm tra, có thể bị loại.

### 5.6 Hòa điểm

Tổng thời gian suy luận thấp hơn xếp trên, sau đó đến bài nộp sớm hơn.

## 6. Giải thưởng (DataMate)

| Giải | Tiền mặt | Quy đổi |
|---|---|---|
| Nhất | 90.412.000 VND | ~3.500 USD |
| Nhì | 64.580.000 VND | ~2.500 USD |

- Tỷ giá tham chiếu Ngân hàng TMCP Hàng Hải Việt Nam ngày 23/10/2026; BTC trừ thuế theo quy định Việt Nam.
- Đội thắng nộp mã nguồn cuối, tài liệu mô tả, hướng dẫn chạy và ký hồ sơ nhận thưởng trong 2 tuần; không phản hồi trong 1 tuần thì mất quyền nhận giải.
- Giải trao trong 30 ngày kể từ khi đủ hồ sơ, không chuyển nhượng.

## 7. Sở hữu trí tuệ và khiếu nại

- Bài dự thi thuộc quyền sở hữu của tác giả; tác giả cấp BTC quyền không độc quyền, không thời hạn, không thu phí để dùng/trích dẫn/công bố cho nghiên cứu, quảng bá, giáo dục, truyền thông.
- Không chia sẻ riêng mã nguồn phát triển từ dữ liệu cuộc thi ra ngoài đội trong thời gian thi; khuyến khích công khai sau khi kết thúc.
- Mã nguồn mở phải có giấy phép OSI hợp lệ, không hạn chế sử dụng thương mại; tuân thủ Luật 91/2025/QH15 và Nghị định 356/2025/NĐ-CP về dữ liệu cá nhân.
- Khiếu nại trong 7 ngày làm việc kể từ ngày công bố chính thức, kèm bằng chứng; quyết định của BTC là cuối cùng; tranh chấp theo pháp luật Việt Nam.
- Yêu cầu về dữ liệu cá nhân gửi về aichallenge@zalo.me.

## 8. Điểm chưa xác minh

- Docker guideline chính thức (base image, ràng buộc kích thước) chưa công bố.
- Số lượt nộp vòng private: thể lệ ghi 5/ngày, trang Submission Guidance không nêu.
- Định dạng log trong `/result/logs/`: portal chỉ ghi "any format", chưa có mẫu.
- Portal có chấp nhận field ngoài `id`/`type`/`answer` hay không.
- Mốc 22/10/2026 12:00 xuất hiện trong mã portal nhưng không có trong thể lệ.

## 9. Checklist trước khi nộp

- [x] Cả 5 thành viên đã có tài khoản Zalo trên portal và đã vào đội.
- [ ] Giải pháp chạy được với `docker run --network none`.
- [ ] Container đọc `/data` và ghi `/result/submission.jsonl` đúng schema, kèm `/result/logs/`.
- [ ] Scorer local khớp PA (1/0/−1) và bảng 9 kiểu; chạy được trên train trước mỗi thay đổi lớn.
- [ ] Seed cố định, giải mã tất định, đã đo thời gian trên input cỡ private test.
- [ ] Đã khai báo toàn bộ mô hình + giấy phép trong README của gói nộp.
- [ ] Có checksum và link Google Drive công khai, nộp trước 16/11/2026 23:59.

## Nguồn

- Thể lệ tiếng Việt (PDF): <https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-vi.pdf>
- Thể lệ tiếng Anh (PDF): <https://dl-challenge.zalo.ai/zalo-ai-challenge-regulations-en.pdf>
- Portal: <https://challenge.zalo.ai>
