"""Shared per-frame recognition logic: hand detection, landmark classifier,
debouncing, and sending the resulting command. Used by both main.py (OpenCV
window) and web_app.py (browser view).

Uses mediapipe.tasks.vision.HandLandmarker (not the legacy mp.solutions.hands
used by ai/sign_recognition/) because that's what extracted the landmarks
the model was trained on (see ../training/asl_landmark/train_on_kaggle.py).
Using the legacy API here instead measurably hurt accuracy in testing (the
two APIs produce slightly different landmark coordinates for the same
image) — this project is independent of ai/sign_recognition/ specifically
so it's free to use whichever MediaPipe API matches its own training data.
"""

import time

import cv2
import mediapipe as mp

import config
import landmark_classifier
from device_controller import DeviceController, MockDeviceController

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


def build_controller():
    if config.USE_MOCK_CONTROLLER:
        print("Using MOCK device controller (no ESP32 relay board yet).")
        return MockDeviceController()
    return DeviceController(config.ESP32_CONTROL_BASE_URL)


def draw_landmarks(frame, hand_landmarks):
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
    for conn in HAND_CONNECTIONS:
        cv2.line(frame, points[conn.start], points[conn.end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 3, (0, 0, 255), -1)


class SignRecognizer:
    def __init__(self):
        self.controller = build_controller()
        print(f"Using ASL landmark recognizer ('{config.LANDMARK_MODEL_PATH}').")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=config.HAND_LANDMARKER_MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        self.stable_command = None
        self.stable_count = 0
        self.last_sent_time = {}
        self.last_overlay_text = "No hand detected"

    def _recognize_letter(self, hand_landmarks, handedness_label):
        letter, confidence = landmark_classifier.classify(
            hand_landmarks, handedness_label, config.LANDMARK_MODEL_PATH
        )
        if confidence < config.LANDMARK_CONFIDENCE_THRESHOLD:
            return None, f"Letter: {letter}? ({confidence:.2f}, low confidence)"

        command = config.LETTER_TO_COMMAND.get(letter)
        return command, f"Letter: {letter} ({confidence:.2f})  Command: {command}"

    def process(self, frame):
        """Takes a BGR frame, returns a BGR frame with landmarks/overlay drawn,
        sending a device command as a side effect when a letter is confirmed."""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect(mp_image)

        hand_present = bool(result.hand_landmarks and result.handedness)
        current_command = None

        if hand_present:
            hand_landmarks = result.hand_landmarks[0]
            handedness_label = result.handedness[0][0].category_name
            draw_landmarks(frame, hand_landmarks)
            current_command, self.last_overlay_text = self._recognize_letter(
                hand_landmarks, handedness_label
            )
        else:
            self.last_overlay_text = "No hand detected"

        if current_command == self.stable_command and current_command is not None:
            self.stable_count += 1
        else:
            self.stable_command = current_command
            self.stable_count = 1

        if current_command is not None and self.stable_count == config.LANDMARK_STABLE_FRAMES:
            now = time.time()
            last_time = self.last_sent_time.get(current_command, 0)
            if now - last_time >= config.COMMAND_COOLDOWN_SECONDS:
                if self.controller.send_command(current_command):
                    print(f"Sent command: {current_command}")
                    self.last_sent_time[current_command] = now

        cv2.putText(
            frame, self.last_overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        state = self.controller.get_state()
        state_text = f"Light: {state['light']}  Fan: {state['fan']}  Door: {state['door']}"
        cv2.putText(
            frame, state_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2
        )
        return frame

    def close(self):
        self.landmarker.close()
