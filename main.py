"""Reads the camera (webcam or ESP32-CAM, per config.py), recognizes ASL
letters with the self-trained landmark classifier, and sends the
corresponding command to the relay controller.

Run:
    python main.py
Press 'q' in the video window to quit.
"""

import time

import cv2

import config
from recognizer import SignRecognizer


def main():
    if config.USE_WEBCAM:
        video_source = config.WEBCAM_INDEX
        print(f"Using laptop webcam (index {video_source}).")
    else:
        video_source = config.ESP32_CAM_STREAM_URL
        print(f"Using ESP32-CAM stream: {video_source}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Could not open video source: {video_source}")
        return

    recognizer = SignRecognizer()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Stream read failed, retrying...")
                time.sleep(0.5)
                continue

            frame = recognizer.process(frame)
            cv2.imshow("ASL Landmark Smart Home Control", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        recognizer.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
