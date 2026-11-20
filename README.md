# ASL Landmark Smart Home

Điều khiển đèn, quạt, khóa cửa bằng ký hiệu tay (chữ cái ASL A-Z), nhận diện
qua webcam bằng model do chính project này tự huấn luyện (không dùng model
pretrained của bên thứ ba cho bước phân loại).

## Kiến trúc

```
Webcam --> MediaPipe HandLandmarker (landmark tay) --> Logistic Regression tự train (A-Z) --> Lệnh HTTP --> ESP32 relay
```

- **MediaPipe HandLandmarker**: phát hiện tay, trả về 21 điểm landmark —
  dùng model có sẵn (pretrained) của Google, không tự train lại phần này.
- **Logistic Regression tự train**: nhận 21 điểm landmark làm input (63 chiều:
  x,y,z mỗi điểm), phân loại ra 1 trong 26 chữ cái. Train trên dataset ASL-HG
  (10 người làm ký hiệu). Đây là phần model do project tự huấn luyện.

## Vì sao chọn cách này

- Model là Logistic Regression (một ma trận 63 × 26, file 14 KB), nhận input
  là landmark tay (đã được MediaPipe trích sẵn) thay vì ảnh pixel thô — rất
  nhẹ, không cần GPU, chạy được mỗi khung hình.
- Vì sao không dùng MLP: khi đánh giá leave-one-signer-out (test trên người
  chưa từng có trong dữ liệu train), Logistic Regression đúng 93.8% so với
  91.6% của MLP (63 → 128 → 64 → 26), và khi đoán sai thì ít "tự tin" hơn, nên
  ngưỡng tin cậy 0.6 chặn được khoảng một nửa số lỗi (MLP: khoảng 13%).
  SVM đạt tương đương (94.0%) nhưng file 5.6 MB và cần cài scikit-learn.
- Model được lưu dưới dạng **trọng số numpy thuần (`.npz`)**, không phải
  object scikit-learn pickle — tránh được lỗi tương thích version numpy
  giữa môi trường train (Kaggle) và môi trường chạy thật. `landmark_classifier.py`
  tự tính xác suất (nhân ma trận + softmax) bằng numpy — đã kiểm chứng cho
  kết quả giống hệt `predict_proba` của scikit-learn. File này cũng đọc được
  trọng số MLP (tự nhận dạng loại model từ file .npz).

## Cài đặt & chạy

Yêu cầu: Python 3.9+ (đã test với 3.12), webcam.

```bash
git clone <URL_repo_cua_ban>
cd <ten-repo>
python -m venv .venv
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Chạy (chọn 1):

```bash
python main.py        # cửa sổ OpenCV
```

hoặc

```bash
python web_app.py     # giao diện web, http://localhost:5001
```

Mặc định `config.py` đã bật `USE_WEBCAM = True` và `USE_MOCK_CONTROLLER =
True` — chạy thử ngay được, không cần phần cứng ESP32. Đưa tay làm ký hiệu
chữ cái ASL trước webcam, giữ ổn định ~0.3-0.5 giây để lệnh được gửi.

## Bảng lệnh (chỉnh trong `config.py` → `LETTER_TO_COMMAND`)

| Chữ | Lệnh |
|---|---|
| L | Bật đèn |
| D | Tắt đèn |
| F | Bật quạt |
| S | Tắt quạt |
| O | Mở khóa cửa |
| C | Khóa cửa |

20 chữ còn lại vẫn được nhận diện và hiển thị, chỉ không gắn lệnh nào.

## Khi có phần cứng thật (ESP32 điều khiển relay)

Đổi trong `config.py`:
- `USE_MOCK_CONTROLLER = False`
- `ESP32_CONTROL_BASE_URL = "http://<ip-esp32-cua-ban>"`

`device_controller.py` gửi lệnh qua HTTP GET tới ESP32 theo dạng:
- `/light?state=on|off`
- `/fan?state=on|off`
- `/door?state=lock|unlock`

ESP32 cần chạy firmware tự xử lý các endpoint trên (bật/tắt relay tương
ứng) và trả JSON trạng thái tại `/status`.

## Cấu trúc project

```
config.py              # cấu hình camera/thiết bị/ngưỡng tin cậy/mapping lệnh
device_controller.py   # gửi lệnh HTTP tới ESP32, hoặc giả lập (mock)
landmark_classifier.py # load trọng số .npz, tự tính xác suất bằng numpy
recognizer.py           # vòng lặp xử lý từng khung hình, debounce, gửi lệnh
main.py                 # chạy bằng cửa sổ OpenCV
web_app.py               # chạy bằng giao diện web (Flask, cổng 5001)
models/
  asl_landmark_logreg_weights.npz  # trọng số Logistic Regression đã train
  hand_landmarker.task           # model MediaPipe HandLandmarker
requirements.txt
```
