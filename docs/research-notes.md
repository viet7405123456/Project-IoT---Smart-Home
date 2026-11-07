# Ghi chú phục vụ viết bài báo khoa học

Tài liệu này tổng hợp lại các quyết định thiết kế, phương pháp, và hạn chế đã
biết của project tính đến hiện tại, để dùng làm nguyên liệu khi viết bài báo/
báo cáo khoa học. Các mục đánh dấu **TODO** là phần chưa có số liệu thật —
cần điền sau khi tự chạy thực nghiệm, không nên bịa số liệu.

## 1. Bối cảnh & mục tiêu

- **Vấn đề**: điều khiển thiết bị nhà thông minh (đèn, quạt, khóa cửa) bằng
  cử chỉ/ký hiệu tay, không cần chạm hoặc dùng giọng nói.
- **Mục tiêu**: xây dựng hệ thống end-to-end: camera → nhận diện cử chỉ/ký
  hiệu tay bằng AI → gửi lệnh điều khiển thiết bị IoT thực qua ESP32.
- **Đối tượng ứng dụng tiềm năng**: điều khiển không chạm nói chung; cần nêu
  rõ trong bài là hệ thống hiện dùng ASL (American Sign Language) chuẩn, nếu
  hướng tới người khiếm thính/khiếm ngôn dùng ngôn ngữ ký hiệu Việt Nam thì
  cần nêu rõ đây là hướng mở rộng, chưa triển khai.

## 2. Kiến trúc hệ thống

```
[Camera: webcam hoặc ESP32-CAM] --> [PC: Python + OpenCV + MediaPipe + SigLIP2]
                                          |
                                          | nhận diện cử chỉ/chữ cái -> lệnh HTTP
                                          v
                         [ESP32 điều khiển relay] --> Đèn / Quạt / Khóa cửa
```

- Camera (ESP32-CAM) chỉ truyền video (MJPEG qua WiFi), **không** chạy AI —
  lý do: ESP32-CAM chỉ có ~520KB RAM, không đủ để chạy mô hình nhận diện tay
  thời gian thực.
- Toàn bộ suy luận AI chạy trên PC (Python).
- Thiết bị điều khiển relay là một ESP32 riêng, nhận lệnh qua HTTP REST
  (`/light`, `/fan`, `/door`, `/status`).
- Tách camera và control thành 2 vi điều khiển riêng để mỗi board chỉ làm
  một việc, đơn giản hóa firmware và không cần ESP32-CAM (yếu hơn) phải xử lý
  thêm tải điều khiển relay.

## 3. Phương pháp nhận diện

Hệ thống hỗ trợ 2 phương pháp (chọn qua `RECOGNITION_MODE`), nên có thể trình
bày như một so sánh trong bài:

### 3.1 Rule-based finger counting (không cần huấn luyện)

- Dùng MediaPipe Hands để lấy 21 điểm landmark trên bàn tay.
- Đếm ngón xòe bằng quy tắc hình học: so sánh tọa độ y của đầu ngón (tip) với
  khớp giữa ngón (PIP) cho 4 ngón; so sánh tọa độ x của ngón cái (thumb) với
  khớp IP, có xét chiều tay (trái/phải) từ `handedness`.
- Số ngón (0-5) ánh xạ trực tiếp sang lệnh.
- **Ưu điểm**: không cần dataset/huấn luyện, chạy rất nhanh (không cần GPU),
  không phụ thuộc `torch`.
- **Nhược điểm**: không phải ngôn ngữ ký hiệu thật, từ vựng giới hạn 6 lệnh.

### 3.2 Phân loại chữ cái ASL bằng SigLIP2 (pretrained)

- Model: [`prithivMLmods/Alphabet-Sign-Language-Detection`](https://huggingface.co/prithivMLmods/Alphabet-Sign-Language-Detection),
  fine-tuned từ `google/siglip2-base-patch16-224`.
- Kiến trúc: Vision Transformer (SigLIP2), ~92.9M tham số, Safetensors F32.
- Input: ảnh tay đã crop (không dùng trực tiếp cả khung hình webcam).
- Pipeline crop: lấy bounding box từ landmark MediaPipe, mở rộng thêm 30%
  margin mỗi chiều (`ASL_CROP_MARGIN_RATIO`), rồi mới đưa vào model.
  - **Lý do crop trước**: ảnh huấn luyện của model có bàn tay chiếm phần lớn
    khung hình, nền đơn giản; đưa thẳng khung hình webcam thô (tay nhỏ, nền
    phức tạp) vào sẽ bị lệch miền dữ liệu (domain shift) so với lúc huấn
    luyện, giảm độ chính xác thực tế.
- Output: softmax trên 26 lớp (A-Z); chỉ chấp nhận dự đoán có độ tin cậy
  ≥ `ASL_CONFIDENCE_THRESHOLD` (mặc định 0.6) để giảm trigger nhầm.
- Chữ cái → lệnh: mapping tường minh 6 chữ (L/D/F/S/O/C), 20 chữ còn lại chỉ
  hiển thị, không gắn lệnh.
- **Hạn chế đã biết**: 2 chữ J và Z trong ASL cần chuyển động (vẽ hình trong
  không khí), không phải tư thế tĩnh — pipeline hiện tại chỉ phân loại ảnh
  tĩnh nên không nhận diện đúng 2 chữ này. Đây là điểm cần nêu rõ trong phần
  "Limitations".
- **Không tự huấn luyện/thu thập dataset riêng** cho model này — dùng thẳng
  trọng số pretrained. Model card công bố accuracy 99.96% trên tập eval
  121,769 ảnh **của họ** — số liệu này không đại diện cho hiệu năng thực tế
  trên webcam của project (cần đo lại, xem mục 6).

### 3.3 Cơ chế chống rung tín hiệu (debounce)

- Yêu cầu N lần phân loại liên tiếp cho cùng 1 kết quả trước khi gửi lệnh
  (`GESTURE_STABLE_FRAMES` = 8 cho finger-count, `ASL_STABLE_CLASSIFICATIONS`
  = 3 cho ASL) — tránh kích hoạt nhầm khi tay đang di chuyển qua khung hình.
- Cooldown `COMMAND_COOLDOWN_SECONDS` (1.5s) giữa 2 lần gửi lệnh giống nhau —
  tránh relay đóng/cắt liên tục (chatter) nếu kết quả phân loại dao động ở
  biên ngưỡng.
- Với mode ASL (model nặng, chạy chậm trên CPU): chỉ chạy suy luận mỗi
  `ASL_FRAME_SKIP` (= 5) khung hình, các khung hình giữa vẫn hiển thị video
  mượt (MediaPipe + vẽ landmark chạy mỗi frame, chỉ SigLIP là bị skip) —
  đáng nói trong phần "Implementation trade-offs".

## 4. Phần cứng & giao tiếp

- **ESP32-CAM** (AI-Thinker): stream MJPEG qua WiFi tại `http://<ip>:81/stream`.
- **ESP32 điều khiển**: HTTP server (`WebServer.h`), endpoint `/light`,
  `/fan`, `/door` (tham số `state`), `/status` (trả JSON trạng thái).
- Relay: giả định loại active-LOW (phổ biến), có cấu hình đổi sang active-HIGH.
- Lưu ý an toàn điện khi relay đóng/cắt điện lưới 220V cho đèn/quạt — nên nêu
  trong phần "Implementation" như một ràng buộc thực tế của hệ thống.

## 5. Phần mềm / dependency đáng chú ý

- Python + OpenCV (`opencv-python==4.10.0.84`) + MediaPipe (`==0.10.21`,
  pin cố định vì bản `mediapipe>=1.0` đã bỏ API `solutions.hands` dùng
  trong project) + `requests` (gọi HTTP tới ESP32) + Flask (giao diện web
  test qua trình duyệt, stream MJPEG đã xử lý).
- Mode ASL cần thêm `torch` + `transformers` (nặng, ảnh hưởng tốc độ suy
  luận trên CPU — có thể nêu như một trade-off giữa độ chính xác/từ vựng
  rộng hơn (26 chữ) và chi phí tính toán, so với finger-counting (nhẹ,
  6 lệnh)).

## 6. Kế hoạch thực nghiệm — CẦN BẠN TỰ ĐO VÀ ĐIỀN

> Chưa có số liệu thật nào ở mục này — đây là khung để điền sau khi test.

### 6.1 Độ chính xác nhận diện theo từng chữ/cử chỉ

| Ký hiệu | Số lần test | Số lần đúng | Độ chính xác (%) | Ghi chú |
|---|---|---|---|---|
| L | | | | |
| D | | | | |
| F | | | | |
| S | | | | |
| O | | | | |
| C | | | | |
| (các chữ khác, nếu test) | | | | |

Gợi ý điều kiện nên ghi lại khi test (để đưa vào "Experimental setup"): ánh
sáng, khoảng cách tay-camera, nền phía sau, tay trái/phải, số người tham gia
test.

### 6.2 Độ trễ / hiệu năng

- Thời gian từ lúc giữ ổn định ký hiệu đến lúc lệnh được gửi (giây).
- FPS video thực tế ở mode `finger_count` vs `asl_alphabet`.
- Ảnh hưởng của `ASL_FRAME_SKIP` tới độ trễ và độ mượt (thử vài giá trị khác
  nhau, vd 1/3/5/10).

### 6.3 So sánh 2 phương pháp

| Tiêu chí | finger_count | asl_alphabet (SigLIP2) |
|---|---|---|
| Độ chính xác | | |
| Tốc độ / FPS | | |
| Cần huấn luyện? | Không | Không (dùng pretrained) |
| Từ vựng | 6 cử chỉ | 26 chữ (trừ J, Z) |
| Dependency | Nhẹ | Nặng (torch+transformers) |

## 7. Hạn chế đã biết (Limitations)

- Domain shift: model ASL huấn luyện trên ảnh tay cận cảnh/nền đơn giản,
  khác điều kiện webcam thực tế — dù đã crop vùng tay, vẫn có thể lệch miền
  dữ liệu, cần đo thực nghiệm (mục 6.1) để biết mức độ ảnh hưởng.
- 2 chữ J, Z (ASL) không nhận diện được vì cần chuyển động, hệ thống hiện
  chỉ phân loại ảnh tĩnh từng khung hình độc lập.
- Chưa tự thu thập/huấn luyện trên dữ liệu riêng — độ chính xác công bố của
  model (99.96%) là trên tập eval của tác giả model, không phải trên dữ
  liệu/điều kiện của project này.
- Tính đến thời điểm viết tài liệu này, phần điều khiển thiết bị mới được
  test bằng controller giả lập (`MockDeviceController`), chưa có relay/khóa
  cửa thật — cập nhật mục này khi đã test với phần cứng thật.
- Chỉ nhận diện 1 tay tại một thời điểm (`max_num_hands=1`).

## 8. Hướng phát triển (Future work)

- Huấn luyện classifier nhẹ (vd scikit-learn MLP/RandomForest) trên tọa độ
  landmark MediaPipe thay vì ảnh pixel thô — bỏ được `torch`/`transformers`,
  nhẹ và nhanh hơn, có thể generalize tốt hơn sang webcam thực tế (landmark
  ít phụ thuộc nền/ánh sáng hơn ảnh pixel). Có thể dùng dataset ảnh ASL công
  khai (vd Kaggle) để trích landmark làm dữ liệu huấn luyện, không cần tự
  chụp ảnh từ đầu.
- Fine-tune model hiện tại trên một lượng nhỏ ảnh webcam tự chụp, nếu đo
  thực nghiệm cho thấy domain shift ảnh hưởng đáng kể.
- Phản hồi trạng thái thiết bị 2 chiều (ESP32 → PC/app) dùng endpoint
  `/status` đã có sẵn.
- Mở rộng từ vựng/số thiết bị điều khiển được.
- Xử lý ký hiệu động (J, Z, hoặc ký hiệu nhiều bước) bằng model theo chuỗi
  thời gian (vd LSTM/temporal model trên landmark theo thời gian) thay vì
  phân loại ảnh tĩnh.

## 9. Thông tin cần quyết định cho bài báo (chưa có câu trả lời)

- [ ] Mục đích: hội nghị khoa học / tạp chí / báo cáo đồ án-khóa luận / báo
      cáo nội bộ?
- [ ] Yêu cầu định dạng cụ thể (IEEE, mẫu của trường, v.v.)?
- [ ] Ngôn ngữ viết bài: tiếng Việt hay tiếng Anh?

## 10. Tài liệu tham khảo cần đưa vào (nguồn đã dùng trong project)

- Model ASL: prithivMLmods, "Alphabet-Sign-Language-Detection",
  <https://huggingface.co/prithivMLmods/Alphabet-Sign-Language-Detection>
  (fine-tuned từ `google/siglip2-base-patch16-224`, license Apache-2.0).
- MediaPipe Hands: Google, thư viện nhận diện landmark tay realtime,
  <https://developers.google.com/mediapipe>.
- (Cần bổ sung) bài báo gốc SigLIP/SigLIP2 — tìm citation chính thức khi
  viết phần Related Work.
- (Cần bổ sung) các công trình liên quan "sign language recognition" +
  "smart home" / "IoT gesture control" để viết phần Related Work — chưa
  tổng hợp trong tài liệu này.
