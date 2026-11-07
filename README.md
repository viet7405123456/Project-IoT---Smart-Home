# Nhà thông minh điều khiển bằng ký hiệu tay

Điều khiển đèn, quạt, khóa cửa bằng cử chỉ/ký hiệu tay, nhận diện qua camera,
xử lý AI trên máy tính (Python + MediaPipe), gửi lệnh tới ESP32 điều khiển relay.

## Kiến trúc

```
[Camera: webcam hoặc ESP32-CAM] --> [PC: Python + OpenCV + MediaPipe + (SigLIP)]
                                          |
                                          | nhận diện cử chỉ/chữ cái -> lệnh HTTP
                                          v
                         [ESP32 điều khiển relay]  --hoặc--  [Mock giả lập, chưa có hardware]
                                          |
                                          v
                                 Đèn / Quạt / Khóa cửa
```

Camera chỉ truyền video, không chạy AI. Toàn bộ nhận diện chạy trên PC. Có 2
chế độ nhận diện, chọn bằng `RECOGNITION_MODE` trong `ai/sign_recognition/config.py`:

- **`finger_count`** (mặc định): đếm số ngón xòe bằng landmark MediaPipe,
  quy tắc hình học, không cần dataset/huấn luyện, chạy rất nhanh.
- **`asl_alphabet`**: nhận diện chữ cái ASL (A-Z) bằng model đã huấn luyện
  sẵn [prithivMLmods/Alphabet-Sign-Language-Detection](https://huggingface.co/prithivMLmods/Alphabet-Sign-Language-Detection)
  (SigLIP2), nặng hơn, cần `torch` + `transformers`.

## Yêu cầu

- Python 3.12 (đã test với bản này; các bản 3.9-3.11 nhiều khả năng cũng chạy được nhưng chưa test)
- Git
- Webcam (để test không cần phần cứng) hoặc ESP32-CAM + ESP32 (để chạy thật)

## Chạy thử ngay (không cần ESP32/phần cứng)

Mặc định project đã cấu hình để test bằng webcam laptop + thiết bị giả lập
(`USE_WEBCAM = True`, `USE_MOCK_CONTROLLER = True` trong `config.py`), nên
bạn chạy thử được ngay dù chưa có ESP32 hay relay nào.

### 1. Clone project

```bash
git clone <URL_repo_cua_ban>
cd IoT/ai/sign_recognition
```

### 2. Tạo virtual environment và cài thư viện

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu muốn dùng chế độ `asl_alphabet` (nhận diện chữ cái ASL), cài thêm:

```bash
pip install -r requirements-asl.txt
```

### 3. Model ASL tự động tải về (chỉ mode `asl_alphabet`)

Thư viện (`torch` + `transformers`) và model là 2 thứ khác nhau:

- **Thư viện**: cài qua `pip install -r requirements-asl.txt` ở bước trên,
  khoảng 1-2GB, tải 1 lần.
- **Model** (`prithivMLmods/Alphabet-Sign-Language-Detection`, ~370MB): tự
  động tải từ HuggingFace Hub **trong lần chạy đầu tiên** (`main.py` hoặc
  `web_app.py` khi `RECOGNITION_MODE = "asl_alphabet"`), không cần làm gì
  thêm, chỉ cần có mạng. Model được cache tại `~/.cache/huggingface/hub`
  (Windows: `C:\Users\<user>\.cache\huggingface\hub`) nên các lần chạy sau
  không tải lại. Lần đầu chạy ASL mode sẽ in dòng
  `Loading ASL classifier '...' (first run downloads it)...` trong terminal.
- Không có GPU vẫn chạy được (CPU), chỉ chậm hơn — project đã xử lý bằng
  cách chỉ chạy model mỗi `ASL_FRAME_SKIP` khung hình để video không bị giật.

### 4. Chạy

Có 2 cách chạy, chọn 1:

**a) Cửa sổ OpenCV** (đơn giản, chạy trực tiếp trên máy):

```powershell
python main.py
```

Nhấn `q` để thoát.

**b) Giao diện web** (xem qua trình duyệt, tiện khi muốn demo/chia sẻ màn hình):

```powershell
python web_app.py
```

Mở trình duyệt vào `http://localhost:5000`. `Ctrl+C` trong terminal để dừng.

Cả 2 cách đều hiển thị: khung xương tay (landmark), chữ/cử chỉ nhận diện
được, độ tin cậy (nếu dùng mode ASL), lệnh tương ứng, và trạng thái thiết bị
giả lập (đèn/quạt/cửa). Đưa tay vào khung hình, giữ ký hiệu ổn định khoảng
nửa giây để lệnh được gửi (tránh kích hoạt nhầm khi tay đang di chuyển qua).

## Cấu hình (`ai/sign_recognition/config.py`)

| Biến | Ý nghĩa |
|---|---|
| `USE_WEBCAM` | `True` = dùng webcam laptop, `False` = dùng stream ESP32-CAM (`ESP32_CAM_STREAM_URL`) |
| `USE_MOCK_CONTROLLER` | `True` = giả lập thiết bị (console/overlay), `False` = gửi HTTP thật tới ESP32 (`ESP32_CONTROL_BASE_URL`) |
| `RECOGNITION_MODE` | `"finger_count"` hoặc `"asl_alphabet"` |
| `GESTURE_TO_COMMAND` | mapping số ngón → lệnh (mode `finger_count`) |
| `LETTER_TO_COMMAND` | mapping chữ cái → lệnh (mode `asl_alphabet`) |

### Bảng lệnh — mode `finger_count`

| Cử chỉ | Lệnh |
|---|---|
| 1 ngón (trỏ) | Bật đèn |
| 2 ngón (chữ V) | Tắt đèn |
| 3 ngón | Bật quạt |
| 4 ngón | Tắt quạt |
| Bàn tay mở (5 ngón) | Mở khóa cửa |
| Nắm tay (0 ngón) | Khóa cửa |

### Bảng lệnh — mode `asl_alphabet`

| Chữ | Lệnh | Gợi nhớ |
|---|---|---|
| L | Bật đèn | Light |
| D | Tắt đèn | Dark |
| F | Bật quạt | Fan |
| S | Tắt quạt | Stop |
| O | Mở khóa cửa | Open |
| C | Khóa cửa | Close |

20 chữ còn lại vẫn được nhận diện và hiển thị, nhưng không gắn lệnh nào.
**Lưu ý**: 2 chữ J và Z trong ASL cần chuyển động (vẽ hình trong không khí),
không phải tư thế tĩnh, nên model phân loại ảnh tĩnh này sẽ không nhận diện
đúng 2 chữ đó.

## Khi có phần cứng thật (ESP32-CAM + ESP32 điều khiển relay)

### Flash ESP32-CAM (`firmware/esp32_cam/esp32_cam.ino`)

- Board trong Arduino IDE: "AI Thinker ESP32-CAM".
- Sửa `WIFI_SSID`, `WIFI_PASSWORD` trong file.
- Flash xong, mở Serial Monitor (115200 baud), lấy IP in ra, ví dụ
  `http://192.168.1.50:81/stream`.
- Trong `config.py`: đặt `USE_WEBCAM = False`, dán URL vào `ESP32_CAM_STREAM_URL`.

### Flash ESP32 điều khiển (`firmware/esp32_control/esp32_control.ino`)

- Sửa `WIFI_SSID`, `WIFI_PASSWORD`.
- Nối relay: `LIGHT_PIN` (16), `FAN_PIN` (17), `LOCK_PIN` (18) — đổi chân nếu
  cần. Hầu hết module relay active-LOW; nếu relay của bạn active-HIGH thì đổi
  `RELAY_ON`/`RELAY_OFF`.
- **An toàn điện**: nếu relay đóng/cắt điện lưới (220V) cho đèn/quạt, dùng
  module relay đúng định mức điện áp, cách ly phần cao áp khỏi ESP32, và nên
  có người có kinh nghiệm điện kiểm tra lại. Khi test ban đầu, nên dùng LED/
  quạt DC/khóa solenoid điện áp thấp trước.
- Lấy IP in ra Serial Monitor, điền vào `ESP32_CONTROL_BASE_URL` trong
  `config.py`, đặt `USE_MOCK_CONTROLLER = False`.

## Ghi chú kỹ thuật / troubleshooting

- `mediapipe` bản mới (>=1.0) đã bỏ API `solutions.hands` cũ — project pin
  cố định `mediapipe==0.10.21` (bản cuối còn API này).
- `mediapipe==0.10.21` yêu cầu `numpy<2`, trong khi `opencv-python` mới nhất
  yêu cầu `numpy>=2` → project pin `opencv-python==4.10.0.84` (bản 4.x,
  tương thích `numpy<2`) để tránh xung đột.
- Không chạy đồng thời `main.py` và `web_app.py` — cả 2 đều mở camera, sẽ
  tranh quyền truy cập thiết bị.

## Mở rộng

- Nhiều lệnh/thiết bị hơn: thêm pin + endpoint trong `esp32_control.ino`,
  thêm vào `COMMAND_TO_REQUEST`/`COMMAND_TO_STATE` trong `device_controller.py`,
  và vào `GESTURE_TO_COMMAND` hoặc `LETTER_TO_COMMAND`.
- Phản hồi trạng thái thiết bị thật về PC: dùng endpoint `/status` đã có sẵn
  trên ESP32 điều khiển.
- Nhận diện ký hiệu tay thật tốt hơn (ngoài đếm ngón/ASL ảnh tĩnh): trích
  landmark MediaPipe từ dataset, huấn luyện classifier nhẹ (vd scikit-learn)
  trên tọa độ landmark thay vì ảnh pixel — nhẹ hơn, có thể generalize tốt
  hơn sang webcam thực tế so với model phân loại ảnh.
