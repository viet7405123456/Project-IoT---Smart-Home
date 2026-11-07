"""Shared per-frame recognition logic: hand detection, gesture/letter
classification, debouncing, and sending the resulting command. Used by both
main.py (OpenCV window) and web_app.py (browser view).
"""

import time

import cv2
import mediapipe as mp

import config
from device_controller import DeviceController, MockDeviceController
from gesture_classifier import classify as classify_finger_count

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def build_controller():
    if config.USE_MOCK_CONTROLLER:
        print("Using MOCK device controller (no ESP32 relay board yet).")
        return MockDeviceController()
    return DeviceController(config.ESP32_CONTROL_BASE_URL)


class SignRecognizer:
    def __init__(self):
        self.controller = build_controller()

        self.use_asl = config.RECOGNITION_MODE == "asl_alphabet"
        self.asl_classifier = None
        if self.use_asl:
            import asl_classifier as asl_classifier_module

            self.asl_classifier = asl_classifier_module
            self.stable_target = config.ASL_STABLE_CLASSIFICATIONS
            print(f"Using ASL alphabet recognizer ('{config.ASL_MODEL_NAME}').")
        else:
            self.stable_target = config.GESTURE_STABLE_FRAMES
            print("Using finger-count recognizer.")

        self.hands = mp_hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.stable_command = None
        self.stable_count = 0
        self.last_sent_time = {}
        self.last_overlay_text = "No hand detected"
        self.frame_index = 0

    def _recognize_finger_count(self, hand_landmarks, handedness_label):
        command, finger_count = classify_finger_count(
            hand_landmarks.landmark, handedness_label, config.GESTURE_TO_COMMAND
        )
        return command, f"Fingers: {finger_count}  Command: {command}"

    def _recognize_asl_letter(self, frame, hand_landmarks):
        cropped = self.asl_classifier.crop_hand_region(
            frame, hand_landmarks.landmark, config.ASL_CROP_MARGIN_RATIO
        )
        if cropped is None or cropped.size == 0:
            return None, "Letter: -  Command: None"

        letter, confidence = self.asl_classifier.classify(cropped, config.ASL_MODEL_NAME)
        if confidence < config.ASL_CONFIDENCE_THRESHOLD:
            return None, f"Letter: {letter}? ({confidence:.2f}, low confidence)"

        command = config.LETTER_TO_COMMAND.get(letter)
        return command, f"Letter: {letter} ({confidence:.2f})  Command: {command}"

    def process(self, frame):
        """Takes a BGR frame, returns a BGR frame with landmarks/overlay drawn,
        sending a device command as a side effect when a gesture is confirmed."""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_present = bool(results.multi_hand_landmarks and results.multi_handedness)
        hand_landmarks = None
        handedness_label = None
        if hand_present:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness_label = results.multi_handedness[0].classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        new_result = None
        if not hand_present:
            new_result = (None, "No hand detected")
        elif self.use_asl:
            if self.frame_index % config.ASL_FRAME_SKIP == 0:
                new_result = self._recognize_asl_letter(frame, hand_landmarks)
        else:
            new_result = self._recognize_finger_count(hand_landmarks, handedness_label)

        if new_result is not None:
            current_command, self.last_overlay_text = new_result
            if current_command == self.stable_command and current_command is not None:
                self.stable_count += 1
            else:
                self.stable_command = current_command
                self.stable_count = 1

            if current_command is not None and self.stable_count == self.stable_target:
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

        self.frame_index += 1
        return frame

    def close(self):
        self.hands.close()
