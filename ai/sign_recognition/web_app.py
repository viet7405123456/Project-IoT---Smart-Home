"""Simple local web UI to test gesture/ASL recognition in a browser instead
of an OpenCV window. Opens the same camera source as main.py (per config.py)
server-side and streams the processed video over HTTP.

Run:
    python web_app.py
Then open http://localhost:5000 in a browser. Ctrl+C in the terminal to stop.
"""

import cv2
from flask import Flask, Response, render_template_string

import config
from recognizer import SignRecognizer

app = Flask(__name__)

if config.USE_WEBCAM:
    VIDEO_SOURCE = config.WEBCAM_INDEX
    print(f"Using laptop webcam (index {VIDEO_SOURCE}).")
else:
    VIDEO_SOURCE = config.ESP32_CAM_STREAM_URL
    print(f"Using ESP32-CAM stream: {VIDEO_SOURCE}")

cap = cv2.VideoCapture(VIDEO_SOURCE)
recognizer = SignRecognizer()

INDEX_HTML = """
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Sign Language Smart Home - Test</title>
<style>
  body {
    background: #111418;
    color: #e6e6e6;
    font-family: system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px;
  }
  h1 { font-size: 1.3rem; margin-bottom: 4px; }
  p.mode { color: #9aa; margin-top: 0; }
  img {
    max-width: 100%;
    width: 720px;
    border-radius: 8px;
    border: 1px solid #333;
  }
</style>
</head>
<body>
  <h1>Sign Language Smart Home — Test nhận diện</h1>
  <p class="mode">Mode hiện tại: <b>{{ mode }}</b> (đổi trong config.py)</p>
  <img src="/video_feed" alt="video stream">
</body>
</html>
"""


def generate_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        processed = recognizer.process(frame)
        ok2, buffer = cv2.imencode(".jpg", processed)
        if not ok2:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template_string(INDEX_HTML, mode=config.RECOGNITION_MODE)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    if not cap.isOpened():
        print(f"Could not open video source: {VIDEO_SOURCE}")
    else:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
